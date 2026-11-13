"""Shared native eight-table merge engine (Python standard library only).

The caller supplies the official schema and owns payload transactions. This
module preserves observations and conflicts; it does not discover files, load
Maa modules, audit note intervals, or maintain historical source catalogues.
Maintained in MaaStellaSora/tools/data and mirrored in the research repository.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3


FORMAT = "maa-state-space-database"
SCHEMA_VERSION = 1
TABLES = ("presets", "runs", "records", "objects", "record_objects", "readings",
          "record_readings", "quality_issues")
_LINK_TABLES = {"record_objects", "record_readings"}
_QUALITY_RANK = {"valid": 0, "partial": 1, "suspect": 2, "invalid": 3}
_RESTRICTIVE_FLAGS = {"training_eligible", "probability_eligible", "sequence_inference_eligible",
                      "teacher_label_available", "analysis_eligible", "usable_for_training"}
_UNION_LISTS = {"data_sources", "incoming_run_ids", "source_origins"}
_CHUNK = 400


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _loads(value):
    return json.loads(value) if isinstance(value, str) else value


def _empty(value):
    return value is None or value == "" or value == {}


def _union(left, right):
    result = list(left)
    seen = {_json(value) for value in result}
    for value in right:
        key = _json(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _merge_json(left, right, path=""):
    """Fill missing values; retain the earlier trusted value on disagreement."""
    if left == right:
        return left, []
    key = path.rsplit(".", 1)[-1]
    if key in _RESTRICTIVE_FLAGS and (left is False or right is False):
        return False, [path] if left is not None and right is not None else []
    if _empty(left):
        return right, []
    if _empty(right):
        return left, []
    if isinstance(left, dict) and isinstance(right, dict):
        merged, conflicts = dict(left), []
        for name, value in right.items():
            child_path = f"{path}.{name}" if path else name
            if name not in left:
                merged[name] = value
            else:
                merged[name], differences = _merge_json(left[name], value, child_path)
                conflicts.extend(differences)
        return merged, conflicts
    if isinstance(left, list) and isinstance(right, list):
        if key in _UNION_LISTS:
            return _union(left, right), []
        # Candidates, owned snapshots, vectors and raw OCR are atomic ordered
        # observations. Even an explicit empty list is a recorded value.
    return left, [path]


def initialize_database(connection, schema_path):
    """Apply the current official schema without rebuilding observation tables.

    Only the small presets table loses its two retired training-scope columns.
    A savepoint keeps that change and the official views atomic, including when
    the caller already owns a transaction. No data or filesystem scan is used.
    """
    schema = Path(schema_path)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("SAVEPOINT real_game_schema")
    removed = []
    try:
        connection.execute("DROP VIEW IF EXISTS run_training_scope")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(presets)")}
        for name in ("training_scope_match", "training_scope_basis"):
            if name in columns:
                connection.execute(f'ALTER TABLE presets DROP COLUMN "{name}"')
                removed.append(name)
        statement = ""
        for line in schema.read_text(encoding="utf-8").splitlines(keepends=True):
            statement += line
            if sqlite3.complete_statement(statement):
                connection.execute(statement)
                statement = ""
        connection.execute("RELEASE real_game_schema")
    except BaseException:
        connection.execute("ROLLBACK TO real_game_schema")
        connection.execute("RELEASE real_game_schema")
        raise
    return dict(removed_preset_columns=removed)


class ResearchMerger:
    """One connection, bounded primary-key lookups and batched writes."""

    def __init__(self, connection):
        self.connection = connection
        self.connection.row_factory = sqlite3.Row
        self.columns = {}
        self.primary = {}
        for table in TABLES:
            info = list(connection.execute(f'PRAGMA table_info("{table}")'))
            self.columns[table] = tuple(row["name"] for row in info)
            self.primary[table] = tuple(row["name"] for row in sorted(info, key=lambda r: r["pk"]) if row["pk"])
        self.source = {}
        self.variants = []
        self.variant_object_ids = []
        self.record_runs = {}
        self.run_records = defaultdict(set)
        self.object_records = defaultdict(set)
        self.reading_records = defaultdict(set)
        self.preset_runs = defaultdict(set)

    def _key(self, table, row):
        return tuple(row[name] for name in self.primary[table])

    def _existing(self, table, rows, columns="*"):
        keys = list(dict.fromkeys(self._key(table, row) for row in rows))
        primary = self.primary[table]
        result = {}
        for start in range(0, len(keys), _CHUNK):
            part = keys[start:start + _CHUNK]
            if len(primary) == 1:
                where = f'"{primary[0]}" IN ({",".join("?" for _ in part)})'
            else:
                names = ",".join(f'"{name}"' for name in primary)
                values = ",".join("(" + ",".join("?" for _ in primary) + ")" for _ in part)
                where = f"({names}) IN ({values})"
            for row in self.connection.execute(f'SELECT {columns} FROM "{table}" WHERE {where}',
                                               [value for key in part for value in key]):
                result[self._key(table, row)] = dict(row)
        return result

    def _recompute_target_completeness(self, preset):
        vector = _loads(preset.get("target_vector_json"))
        complete = (bool(preset.get("target_vector_complete")) and isinstance(vector, list)
                    and len(vector) == 36 and all(level is not None for level in vector))
        preset["target_vector_complete"] = int(complete)

    def _normalise_run_links(self, tables):
        """Use exact existing record identities to attach evidence fragments."""
        referenced_records = list(tables["records"])
        referenced_records.extend({"record_id": row["record_id"]} for table in
            ("record_objects", "record_readings", "quality_issues") for row in tables[table] if row.get("record_id"))
        existing = self._existing("records", referenced_records, 'record_id,run_id')
        owners = {key[0]: row["run_id"] for key, row in existing.items()}
        self.record_runs.update(owners)
        for record_id, run_id in owners.items():
            self.run_records[run_id].add(record_id)
        found = defaultdict(set)
        for record in tables["records"]:
            if record["record_id"] in owners:
                found[record["run_id"]].add(owners[record["record_id"]])
        aliases = {run_id: next(iter(run_owners)) for run_id, run_owners in found.items() if len(run_owners) == 1}
        for record in tables["records"]:
            original = record["run_id"]
            record["run_id"] = owners.get(record["record_id"], aliases.get(original, original))
            self.record_runs[record["record_id"]] = record["run_id"]
            self.run_records[record["run_id"]].add(record["record_id"])
        linked_reading_owners = defaultdict(set)
        for link in tables["record_readings"]:
            run_id = self.record_runs.get(link["record_id"])
            if run_id:
                linked_reading_owners[link["reading_id"]].add(run_id)
        for reading in tables["readings"]:
            linked = linked_reading_owners[reading["reading_id"]]
            reading["run_id"] = next(iter(linked)) if len(linked) == 1 else aliases.get(reading["run_id"], reading["run_id"])
        for issue in tables["quality_issues"]:
            issue["run_id"] = self.record_runs.get(issue.get("record_id"), aliases.get(issue["run_id"], issue["run_id"]))
        for run in tables["runs"]:
            old = run["run_id"]
            run["run_id"] = aliases.get(old, old)
            config = _loads(run["config_json"])
            if old != run["run_id"]:
                config["incoming_run_ids"] = _union(config.get("incoming_run_ids", []), [old])
            if self.source:
                config["data_sources"] = _union(config.get("data_sources", []), [self.source])
            run["config_json"] = _json(config)
            if run.get("preset_key"):
                self.preset_runs[run["preset_key"]].add(run["run_id"])

    def _merged_row(self, table, old, incoming):
        merged, conflicts = dict(old), []
        primary = self.primary[table]
        for name in self.columns[table]:
            if name in primary or name not in incoming or old.get(name) == incoming[name]:
                continue
            before, after = old.get(name), incoming[name]
            if name.endswith("_json"):
                value, differences = _merge_json(_loads(before), _loads(after), name[:-5])
                if differences or value != _loads(before):
                    merged[name] = _json(value)
                conflicts.extend(differences)
            elif table == "runs" and name in {"full_run_start_observed", "completion_observed"}:
                merged[name] = int(bool(before or after))
            elif table == "runs" and name == "lifecycle_status":
                order = {"open": 0, "interrupted": 1, "completed": 2}
                merged[name] = max((before, after), key=order.get)
            elif table == "runs" and name == "ended_at":
                merged[name] = max(value for value in (before, after) if value) if before or after else None
            elif table == "runs" and name == "started_at":
                merged[name] = min(value for value in (before, after) if value) if before or after else None
            elif name == "quality_status":
                merged[name] = max((before, after), key=_QUALITY_RANK.get)
            elif table == "presets" and name == "target_vector_complete":
                continue  # Derived again from the merged original vector below.
            elif _empty(before):
                merged[name] = after
            elif not _empty(after):
                conflicts.append(name)
        if table == "runs" and incoming.get("completion_observed") and not old.get("completion_observed"):
            # A later observed boundary completes an earlier snapshot.
            for name in ("termination_reason", "termination_basis"):
                if incoming.get(name) is not None:
                    merged[name] = incoming[name]
                    if name in conflicts:
                        conflicts.remove(name)
        if table == "presets":
            merged["target_vector_complete"] = int(bool(old.get("target_vector_complete") or incoming.get("target_vector_complete")))
            self._recompute_target_completeness(merged)
        return merged, conflicts

    def _preserve_variant(self, table, row, conflicts):
        if conflicts:
            self.variants.append((table, row, sorted(set(conflicts))))

    def _merge_table(self, table, rows):
        if not rows:
            return {"inserted": 0, "updated": 0}
        columns = self.columns[table]
        names = ",".join(f'"{name}"' for name in columns)
        marks = ",".join("?" for _ in columns)
        if table in _LINK_TABLES:
            before = self.connection.total_changes
            self.connection.executemany(f'INSERT OR IGNORE INTO "{table}" ({names}) VALUES ({marks})',
                                        ([row.get(name) for name in columns] for row in rows))
            return {"inserted": self.connection.total_changes - before, "updated": 0}
        # Coalesce same-ID rows within a payload before asking SQLite for them.
        unique = {}
        for row in rows:
            key = self._key(table, row)
            if key in unique:
                unique[key], conflicts = self._merged_row(table, unique[key], row)
                self._preserve_variant(table, row, conflicts)
            else:
                unique[key] = row
        existing = self._existing(table, list(unique.values()))
        inserts, updates = [], []
        for key, row in unique.items():
            if key not in existing:
                inserts.append(row)
                continue
            merged, conflicts = self._merged_row(table, existing[key], row)
            self._preserve_variant(table, row, conflicts)
            if merged != existing[key]:
                updates.append(merged)
        self.connection.executemany(f'INSERT INTO "{table}" ({names}) VALUES ({marks})',
                                    ([row.get(name) for name in columns] for row in inserts))
        nonprimary = tuple(name for name in columns if name not in self.primary[table])
        if updates:
            assignments = ",".join(f'"{name}"=?' for name in nonprimary)
            where = " AND ".join(f'"{name}"=?' for name in self.primary[table])
            self.connection.executemany(f'UPDATE "{table}" SET {assignments} WHERE {where}',
                ([row.get(name) for name in (*nonprimary, *self.primary[table])] for row in updates))
        return {"inserted": len(inserts), "updated": len(updates)}

    def _write_variants(self):
        objects, links, issues = [], [], []
        for table, row, conflicts in self.variants:
            content = dict(table=table, primary_key=dict(zip(self.primary[table], self._key(table, row))),
                           conflicting_fields=conflicts, source=self.source, source_row=row)
            variant_id = "source_variant:" + _hash(content)
            records = set()
            if table == "records":
                records.add(row["record_id"])
            elif table == "runs" and self.run_records[row["run_id"]]:
                records.add(min(self.run_records[row["run_id"]]))
            elif table == "objects":
                records.update(self.object_records[row["object_id"]])
            elif table == "readings":
                records.update(self.reading_records[row["reading_id"]])
            elif table == "quality_issues" and row.get("record_id"):
                records.add(row["record_id"])
            if records:
                self.variant_object_ids.append(variant_id)
                objects.append(dict(object_id=variant_id, kind="source_variant",
                    basis="conflicting_original_source_preserved", content_json=_json(content)))
                links.extend(dict(record_id=record_id, field_path=f"source_variants.{table}", object_id=variant_id)
                             for record_id in sorted(records))
            else:
                run_ids = self.preset_runs.get(row.get("preset_key"), set()) if table == "presets" else {row.get("run_id")}
                for run_id in sorted(run_id for run_id in run_ids if run_id):
                    issues.append(dict(issue_id=variant_id + ":" + run_id, run_id=run_id, record_id=None,
                        field_path="source_variants." + table, code="source_value_conflict", severity="suspect",
                        message="同一标识的来源值存在差异；保留可信主值及完整来源变体。", evidence_json=_json(content)))
        for table, rows in (("objects", objects), ("record_objects", links), ("quality_issues", issues)):
            columns = self.columns[table]
            names = ",".join(f'"{name}"' for name in columns)
            marks = ",".join("?" for _ in columns)
            self.connection.executemany(f'INSERT OR IGNORE INTO "{table}" ({names}) VALUES ({marks})',
                                        ([row.get(name) for name in columns] for row in rows))

    def merge_payload(self, source_label, payload):
        if payload.get("format") != FORMAT or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported research payload: {source_label}")
        tables = {table: [dict(row) for row in payload["tables"].get(table, [])] for table in TABLES}
        self.source = payload.get("_source", {})
        self.variants = []
        self.variant_object_ids = []
        self.record_runs = {}
        self.run_records = defaultdict(set)
        self.object_records = defaultdict(set)
        self.reading_records = defaultdict(set)
        self.preset_runs = defaultdict(set)
        for preset in tables["presets"]:
            self._recompute_target_completeness(preset)
        self._normalise_run_links(tables)
        for link in tables["record_objects"]:
            self.object_records[link["object_id"]].add(link["record_id"])
        for link in tables["record_readings"]:
            self.reading_records[link["reading_id"]].add(link["record_id"])
        result = {}
        for table in TABLES:
            result[table] = self._merge_table(table, tables[table])
        self._write_variants()
        return dict(source=source_label, runs=len(tables["runs"]), records=len(tables["records"]),
                    inserted=sum(value["inserted"] for value in result.values()),
                    updated=sum(value["updated"] for value in result.values()), variants=len(self.variants))


def _source_observation_facts(payload):
    """Extract the observed quantities represented in the retained variants.

    Provenance IDs, OCR scores, completeness flags, inferred expectations and
    policy metadata are intentionally absent. Explicit list identities align
    facts for comparison only; the stored lists are never rewritten.
    """
    facts = {}

    def add(key, path, value):
        if value is not None and type(value) in (int, float, str):
            facts[key] = (path, value)

    for name in ("floor", "coins", "current_coins", "price", "quantity"):
        if name in payload:
            add(name, name, payload[name])
    for name in ("counts", "before", "after", "delta"):
        for melody, value in (payload.get(name) or {}).items():
            add((name, melody), f"{name}.{melody}", value)
    for index, value in enumerate(payload.get("potential_counts") or []):
        add(("potential_counts", index), f"potential_counts[{index}]", value)
    for name, identity_fields in (("actual_concertos", ("group_id", "level")),
                                  ("main_discs", ("id",))):
        for index, item in enumerate(payload.get(name) or []):
            slot = item.get("slot", index)
            for field in identity_fields:
                add((name, slot, field), f"{name}[{index}].{field}", item.get(field))
    observation = payload.get("observation") or {}
    for name, readings in (("observation.readings", observation.get("readings")),
                           ("readings", payload.get("readings"))):
        for index, item in enumerate(readings or []):
            if not isinstance(item, dict):
                continue
            box = tuple(item.get("box") or [])
            identity = (item.get("source"), box) if box else (item.get("source"), item.get("melody"))
            for field in ("melody", "current"):
                add((name, identity, field), f"{name}[{index}].{field}", item.get(field))
    return facts


def annotate_source_value_conflicts(connection, object_ids=None):
    """Mark actual source disagreements without changing retained observations.

    This is a final transaction step, owned by the caller. Only facts present
    on both sides are compared; extra evidence, unknown values and changes to
    references or source classification do not become observation conflicts.
    """
    if object_ids is None:
        rows = connection.execute("SELECT object_id,content_json FROM objects WHERE kind='source_variant'")
        variants = [(row[0], _loads(row[1])) for row in rows]
    else:
        variants = []
        ids = list(dict.fromkeys(object_ids))
        for start in range(0, len(ids), _CHUNK):
            part = ids[start:start + _CHUNK]
            marks = ",".join("?" for _ in part)
            variants.extend((row[0], _loads(row[1])) for row in connection.execute(
                f"SELECT object_id,content_json FROM objects WHERE object_id IN ({marks}) AND kind='source_variant'", part))
    summary = dict(source_variants_examined=len(variants), record_variants_examined=0,
                   substantive_conflicts=0, new_issues=0, already_marked=0,
                   records_marked_suspect=0, fields={})
    primary_rows, existing_issues = {}, {}
    inserts, affected = [], set()
    for object_id, variant in variants:
        if variant.get("table") != "records":
            continue
        summary["record_variants_examined"] += 1
        record_id = variant["primary_key"]["record_id"]
        if record_id not in primary_rows:
            cursor = connection.execute("SELECT run_id,quality_status,payload_json FROM records WHERE record_id=?", (record_id,))
            row = cursor.fetchone()
            primary_rows[record_id] = tuple(row)
            existing_issues[record_id] = [tuple(issue) for issue in connection.execute(
                "SELECT field_path,code,severity FROM quality_issues WHERE record_id=?", (record_id,))]
        run_id, quality, payload_json = primary_rows[record_id]
        primary = _source_observation_facts(_loads(payload_json))
        incoming = _source_observation_facts(_loads(variant["source_row"]["payload_json"]))
        conflict_paths = variant.get("conflicting_fields", [])
        for key in primary.keys() & incoming.keys():
            field_path, before = primary[key]
            _, after = incoming[key]
            if before == after:
                continue
            # A variant may retain the entire source row for a single metadata
            # disagreement. Annotate only the fields that actually conflicted.
            full_path = "payload." + field_path
            if not any(full_path == path or full_path.startswith(path + ".")
                       or full_path.startswith(path + "[") for path in conflict_paths):
                continue
            summary["substantive_conflicts"] += 1
            summary["fields"][field_path] = summary["fields"].get(field_path, 0) + 1
            equivalent = any(severity in ("suspect", "invalid") and "conflict" in code
                and (field_path == path or field_path.startswith(path + ".") or field_path.startswith(path + "["))
                for path, code, severity in existing_issues[record_id])
            if equivalent:
                summary["already_marked"] += 1
                continue
            evidence = dict(source_variant_object_id=object_id, primary_value=before,
                source_value=after, source=variant.get("source", {}), confirmation_status="unconfirmed")
            issue_id = "source_value_conflict:" + _hash([record_id, field_path, object_id, before, after])
            inserts.append((issue_id, run_id, record_id, field_path, "source_value_conflict", "suspect",
                            "同一原始观测的来源读数不一致，保留主值及来源变体，此字段未确认。", _json(evidence)))
            existing_issues[record_id].append((field_path, "source_value_conflict", "suspect"))
            if quality in ("valid", "partial"):
                affected.add(record_id)
    before = connection.total_changes
    connection.executemany("INSERT OR IGNORE INTO quality_issues "
        "(issue_id,run_id,record_id,field_path,code,severity,message,evidence_json) VALUES (?,?,?,?,?,?,?,?)", inserts)
    summary["new_issues"] = connection.total_changes - before
    before = connection.total_changes
    connection.executemany("UPDATE records SET quality_status='suspect' "
                           "WHERE record_id=? AND quality_status IN ('valid','partial')", ((record_id,) for record_id in affected))
    summary["records_marked_suspect"] = connection.total_changes - before
    return summary
