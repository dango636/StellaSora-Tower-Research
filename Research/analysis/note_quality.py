"""Read-only, aggregate audit of native Maa note intervals (standard library).

Inspired by beichen24a1's PR #1: unknown is not zero, reobservations are not
new actions, and a conflicting/invalid copy excludes its entire action group.
This implementation uses Maa's native tables/views, not that PR's importer.
https://github.com/dango636/StellaSora-Tower-Research/pull/1
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from zipfile import BadZipFile, ZipFile, is_zipfile


FORMAT = "maa-state-space-database"
TABLES = ("presets", "runs", "records", "objects", "record_objects", "readings",
          "record_readings", "quality_issues")
COMMON = tuple("melody_of_" + name for name in
               ("pummel", "luck", "burst", "stamina", "focus", "skill", "ultimate"))
PAIRS = tuple(frozenset("melody_of_" + name for name in pair) for pair in
              (("lux", "terra"), ("aqua", "ventus"), ("ignis", "umbra")))
MELODIES = COMMON + tuple("melody_of_" + name for name in
                         ("aqua", "ignis", "ventus", "terra", "lux", "umbra"))
SCHEMA = Path(__file__).resolve().parents[2] / "Background/RealGameData/research_schema.sql"


@contextmanager
def native_source(path, schema=SCHEMA):
    """SQLite is SELECT-only; a native ZIP is materialized solely in memory."""
    path = Path(path).resolve()
    if is_zipfile(path):
        with ZipFile(path) as archive:
            payload = json.loads(archive.read("research.json"))
            manifest = json.loads(archive.read("manifest.json"))
        if any(x.get("format") != FORMAT or x.get("schema_version") != 1
               for x in (payload, manifest)):
            raise ValueError("unsupported_native_format")
        runs = payload["tables"]["runs"]
        if len(runs) != 1 or runs[0]["run_id"] != manifest.get("run_id"):
            raise ValueError("manifest_run_mismatch")
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(Path(schema).read_text(encoding="utf-8"))
            for table in TABLES:
                rows = payload["tables"].get(table, [])
                columns = [c[1] for c in connection.execute(f'PRAGMA table_info("{table}")')]
                if rows:
                    names = ",".join('"' + column + '"' for column in columns)
                    marks = ",".join("?" for _ in columns)
                    connection.executemany(f'INSERT INTO "{table}" ({names}) VALUES ({marks})',
                                           ([row[column] for column in columns] for row in rows))
            connection.commit()
            connection.execute("PRAGMA query_only=ON")
            yield connection
        finally:
            connection.close()
    else:
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")  # One consistent read, including a live WAL if present.
            yield connection
        finally:
            connection.close()


def known_count(value):
    return type(value) is int and value >= 0


def examine(row):
    """Return local interval checks; native eligibility remains necessary."""
    payload = json.loads(row["payload_json"])
    scope = json.loads(row["scope_melodies"] or "[]")
    flags = json.loads(row["statistical_flags"] or "{}")
    effect = payload.get("note_effect") or {}
    source = row["source_type"]
    kind = ("combined_interval" if source == "combined_interval" else
            "battle" if source == "battle" else
            "exchange" if payload.get("spends_notes") is True or effect.get("operation") == "spend"
            else "other")
    reasons = set()
    if row["quality_status"] != "valid":
        reasons.add("native_record_quality_not_valid")
    if source == "combined_interval":
        reasons.add("combined_sources_not_separable")
    elif source not in ("battle", "event", "purchase"):
        reasons.add("unsupported_interval_source")
    for field, reason in (("independently_observed", "source_not_independent"),
                          ("analysis_eligible", "native_analysis_not_eligible"),
                          ("probability_eligible", "native_probability_not_eligible")):
        if row[field] != 1:
            reasons.add(reason)
    if payload.get("complete") != True:
        reasons.add("native_interval_incomplete")
    if payload.get("floor_complete") != True or payload.get("floor") is None:
        reasons.add("floor_not_confirmed")
    if not scope or not set(scope).issubset(MELODIES):
        reasons.add("native_scope_missing_or_unknown")

    required = scope
    active = payload.get("active_melodies") or []
    if kind == "battle":
        if frozenset(active) not in PAIRS:
            reasons.add("tower_scope_not_confirmed")
            required = []
        else:
            required = list(COMMON) + sorted(active)
    if not set(required).issubset(scope):
        reasons.add("required_notes_outside_native_scope")
    if not required or any(flags.get(melody) != 1 for melody in required):
        reasons.add("native_interval_statistics_not_eligible")

    before, after, delta = (payload.get(key) or {} for key in ("before", "after", "delta"))
    if any(not known_count(counts.get(melody))
           for counts in (before, after) for melody in required):
        reasons.add("required_inventory_unknown_or_invalid")
    elif any(type(delta.get(melody)) is not int
             or delta[melody] != after[melody] - before[melody] for melody in required):
        reasons.add("signed_delta_disagrees_with_inventory")
    elif required and (type(payload.get("total")) is not int
                       or payload["total"] != sum(delta[melody] for melody in required)):
        reasons.add("scope_total_disagrees_with_delta")

    # A partial exchange may describe its observed scope. It does not identify
    # the full random-spend outcome when off-tower supply remains unknown.
    strict_reasons = set()
    if kind == "exchange":
        if any(not known_count(counts.get(melody)) for counts in (before, after) for melody in MELODIES):
            strict_reasons.add("exchange_hidden_supply_unknown")
        if set(scope) != set(MELODIES):
            strict_reasons.add("exchange_scope_not_all_13")
        cost = effect.get("count")
        if (effect.get("operation") != "spend" or effect.get("selection") != "random"
                or type(cost) is not int or cost <= 0):
            strict_reasons.add("exchange_cost_or_rule_not_declared")
        elif not strict_reasons:
            changes = [after[melody] - before[melody] for melody in MELODIES]
            if any(change > 0 for change in changes) or sum(changes) != -cost:
                strict_reasons.add("exchange_full_deduction_not_verified")

    # Timestamps, record IDs, reasons for capture and OCR copies are not action
    # identity. Compare inventory and attribution evidence, including flags.
    compared = required if kind == "battle" else MELODIES
    signature = json.dumps(dict(kind=kind, source=source, floor=payload.get("floor"),
        scope=sorted(required if kind == "battle" else scope), active=sorted(active),
        before={key: before.get(key) for key in compared},
        after={key: after.get(key) for key in compared},
        delta={key: delta.get(key) for key in compared}, total=payload.get("total"),
        note_effect=effect, independently_observed=row["independently_observed"],
        analysis_eligible=row["analysis_eligible"], probability_eligible=row["probability_eligible"]),
        sort_keys=True)
    return kind, reasons, strict_reasons, signature


def audit(connection, run_id=None):
    """Aggregate ended runs; a later bad interval does not reject an earlier one."""
    connection.row_factory = sqlite3.Row
    selected = " AND u.run_id=?" if run_id is not None else ""
    arguments = (run_id,) if run_id is not None else ()
    lifecycle = Counter({row[0]: row[1] for row in connection.execute(
        "SELECT u.lifecycle_status,COUNT(*) FROM runs u WHERE 1=1" + selected +
        " GROUP BY u.lifecycle_status", arguments)})
    query = """
        WITH eligibility AS (
            SELECT record_id, MAX(operation_id) AS operation_id, MAX(source_type) AS source_type,
                MAX(scope_melodies) AS scope_melodies,
                MAX(independently_observed) AS independently_observed,
                MAX(analysis_eligible) AS analysis_eligible,
                MAX(probability_eligible) AS probability_eligible,
                json_group_object(melody,usable_for_interval_statistics) AS statistical_flags
            FROM note_deltas GROUP BY record_id
        )
        SELECT r.run_id,r.record_id,r.payload_json,
            r.effective_quality_status AS quality_status,
            COALESCE(e.operation_id,json_extract(r.payload_json,'$.operation_id')) AS action_operation_id,e.*
        FROM record_quality r JOIN runs u USING(run_id)
        LEFT JOIN eligibility e ON e.record_id=r.record_id
        WHERE r.event_type='note_delta' AND u.lifecycle_status IN ('completed','interrupted')
    """ + selected
    groups = {}
    observations = 0
    for row in connection.execute(query, arguments):
        observations += 1
        kind, reasons, strict_reasons, signature = examine(row)
        # json_each(delta) emits no native view rows for an empty/missing delta.
        # Its original operation ID still links that bad copy to the good copy.
        operation_id = row["action_operation_id"]
        if operation_id is None or operation_id == "":
            reasons.add("operation_id_missing")
            key = (row["run_id"], "missing", row["record_id"])
        else:
            key = (row["run_id"], "operation", operation_id)
        group = groups.setdefault(key, dict(kinds=set(), reasons=set(), strict_reasons=set(),
                                            signatures=set(), observations=0))
        group["kinds"].add(kind)
        group["reasons"].update(reasons)
        group["strict_reasons"].update(strict_reasons)
        group["signatures"].add(signature)
        group["observations"] += 1

    accepted, reasons, strict_reasons = Counter(), Counter(), Counter()
    conflicts = excluded = strict_exchanges = exchanges = hidden_unknown = 0
    for group in groups.values():
        if len(group["signatures"]) > 1:
            conflicts += 1
            group["reasons"].add("conflicting_observations_for_action")
        if "exchange" in group["kinds"]:
            exchanges += 1
            strict_reasons.update(group["strict_reasons"])
            hidden_unknown += "exchange_hidden_supply_unknown" in group["strict_reasons"]
        if group["reasons"]:
            excluded += 1
            reasons.update(group["reasons"])
        else:
            kind = next(iter(group["kinds"]))
            accepted[kind] += 1
            strict_exchanges += kind == "exchange" and not group["strict_reasons"]
    return dict(format=FORMAT, schema_version=1, scope="local_note_interval_statistics",
        training_approved=False, ended_runs_audited=lifecycle["completed"] + lifecycle["interrupted"],
        open_runs_skipped=lifecycle["open"], observations=observations, action_groups=len(groups),
        reobservations_not_independent=observations-len(groups), conflicting_action_groups=conflicts,
        local_statistical_actions=dict(sorted(accepted.items())), excluded_action_groups=excluded,
        exclusion_reason_counts=dict(sorted(reasons.items())), exchange_action_groups=exchanges,
        exchange_strict_13_inventory_checks=strict_exchanges,
        exchange_hidden_supply_unknown_groups=hidden_unknown,
        exchange_strict_subset_reason_counts=dict(sorted(strict_reasons.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Native SQLite or ZIP(research.json, manifest.json)")
    parser.add_argument("--run-id", help="Audit only this native run; no IDs are emitted")
    args = parser.parse_args()
    try:
        with native_source(args.source) as connection:
            report = audit(connection, args.run_id)
    except (OSError, sqlite3.Error, BadZipFile, KeyError, ValueError) as error:
        # Database/JSON error text can contain private source values or paths.
        parser.exit(2, "audit_failed: " + type(error).__name__ + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
