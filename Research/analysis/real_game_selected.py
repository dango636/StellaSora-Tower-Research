"""Read explicitly selected native JSON/GZIP/ZIP/SQLite files only.

No discovery, extraction, local Maa imports, source repair, or note audit is
performed. SQLite sources use a consistent read-only transaction; transport
payloads retain their native identities and observation fields.
Maintained in MaaStellaSora/tools/data and mirrored in the research repository.
"""
from __future__ import annotations

from contextlib import closing
import gzip
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from typing import Iterator
from zipfile import ZipFile

if __package__:
    from .real_game_merge import FORMAT, SCHEMA_VERSION, TABLES
else:
    from real_game_merge import FORMAT, SCHEMA_VERSION, TABLES

_DATABASE_SUFFIXES = {".sqlite3", ".sqlite", ".db"}


def _sqlite_payloads(path: Path, run_ids=None) -> Iterator[dict]:
    queries = {
        "presets": "SELECT p.* FROM presets p JOIN runs r USING(preset_key) WHERE r.run_id=? ORDER BY p.preset_key",
        "runs": "SELECT * FROM runs WHERE run_id=?",
        "records": "SELECT * FROM records WHERE run_id=? ORDER BY sequence,record_id",
        "objects": "SELECT DISTINCT o.* FROM objects o JOIN record_objects l USING(object_id) JOIN records r USING(record_id) WHERE r.run_id=? ORDER BY o.object_id",
        "record_objects": "SELECT l.* FROM record_objects l JOIN records r USING(record_id) WHERE r.run_id=? ORDER BY l.record_id,l.field_path,l.object_id",
        "readings": "SELECT * FROM readings WHERE run_id=? OR reading_id IN (SELECT l.reading_id FROM record_readings l JOIN records r USING(record_id) WHERE r.run_id=?) ORDER BY reading_id",
        "record_readings": "SELECT l.* FROM record_readings l JOIN records r USING(record_id) WHERE r.run_id=? ORDER BY l.record_id,l.reading_id,l.field_path",
        "quality_issues": "SELECT * FROM quality_issues WHERE run_id=? ORDER BY issue_id",
    }
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        if run_ids is None:
            run_ids = [row[0] for row in connection.execute("SELECT run_id FROM runs ORDER BY started_at,run_id")]
        for run_id in run_ids:
            tables = {table: [dict(row) for row in connection.execute(
                queries[table], (run_id, run_id) if table == "readings" else (run_id,))]
                for table in TABLES}
            yield {"format": FORMAT, "schema_version": SCHEMA_VERSION, "tables": tables}


def _split_payload(payload: dict) -> Iterator[dict]:
    """Keep shared table rows only when linked to the emitted run."""
    all_tables = payload["tables"]
    for run in all_tables.get("runs", []):
        run_id = run["run_id"]
        records = [row for row in all_tables.get("records", []) if row["run_id"] == run_id]
        record_ids = {row["record_id"] for row in records}
        object_links = [row for row in all_tables.get("record_objects", []) if row["record_id"] in record_ids]
        object_ids = {row["object_id"] for row in object_links}
        reading_links = [row for row in all_tables.get("record_readings", []) if row["record_id"] in record_ids]
        reading_ids = {row["reading_id"] for row in reading_links}
        tables = {
            "presets": [row for row in all_tables.get("presets", []) if row["preset_key"] == run.get("preset_key")],
            "runs": [run], "records": records,
            "objects": [row for row in all_tables.get("objects", []) if row["object_id"] in object_ids],
            "record_objects": object_links,
            "readings": [row for row in all_tables.get("readings", [])
                         if row.get("run_id") == run_id or row["reading_id"] in reading_ids],
            "record_readings": reading_links,
            "quality_issues": [row for row in all_tables.get("quality_issues", []) if row["run_id"] == run_id],
        }
        yield {**payload, "tables": tables}


def _archive_payloads(archive: ZipFile, label: str) -> Iterator[tuple[str, dict, dict]]:
    members = archive.namelist()
    manifests = {str(Path(name).parent).replace("\\", "/"): json.loads(archive.read(name))
                 for name in members if Path(name).name.lower() == "manifest.json"}
    for name in members:
        if name.lower().endswith(".zip"):
            with ZipFile(BytesIO(archive.read(name))) as nested:
                yield from _archive_payloads(nested, label + "!" + name)
        elif Path(name).name.lower() in {"research.json", "research.json.gz"}:
            encoded = archive.read(name)
            payload = json.loads(gzip.decompress(encoded) if name.lower().endswith(".gz") else encoded)
            if payload.get("format") == FORMAT and payload.get("schema_version") == SCHEMA_VERSION:
                manifest = manifests.get(str(Path(name).parent).replace("\\", "/"), {})
                yield label + "!" + name, payload, manifest


def _with_source(payload: dict, metadata: dict) -> dict:
    run = payload["tables"]["runs"][0]
    observation_kind = run.get("observation_kind", "")
    return {**payload, "_source": {
        **payload.get("_source", {}), **metadata,
        "observation_kind": observation_kind,
        "started_at": run.get("started_at"), "ended_at": run.get("ended_at"),
        "full_run_start_observed": run.get("full_run_start_observed"),
        "completion_observed": run.get("completion_observed"),
        "diagnostic_segment": "diagnostic" in str(metadata.get("capture_kind", "")),
    }}


def iter_selected_native_payloads(paths, work_dir=None):
    """Yield (source_label, payload) in argument/member order, without staging.

    ``work_dir`` remains accepted for the local importer's reader interface;
    it is unused. JSON and GZIP files may have arbitrary filenames. Upload ZIPs
    contain research.json or research.json.gz and may contain nested ZIPs.
    """
    for value in paths:
        path = Path(value).resolve()
        found = False
        if path.suffix.lower() == ".zip":
            with ZipFile(path) as archive:
                for label, payload, manifest in _archive_payloads(archive, str(path)):
                    for part in _split_payload(payload):
                        found = True
                        yield label, _with_source(part, dict(source_kind="native_archive",
                            path=str(path), archive_member=label, manifest=manifest))
        else:
            if path.suffix.lower() in _DATABASE_SUFFIXES:
                kind, payloads = "native_sqlite", _sqlite_payloads(path)
            else:
                encoded = path.read_bytes()
                document = json.loads(gzip.decompress(encoded) if path.suffix.lower() == ".gz" else encoded)
                kind, payloads = "native_json", _split_payload(document)
            for payload in payloads:
                found = True
                run_id = payload["tables"]["runs"][0]["run_id"]
                yield str(path) + "#" + run_id, _with_source(payload,
                    dict(source_kind=kind, path=str(path)))
        if not found:
            raise ValueError(f"No native research payload found in {path}")
