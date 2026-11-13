"""Merge explicitly selected native run files into an eight-table SQLite store.

Only Python's standard library is required. The shared engine preserves native
identities, complementary evidence and conflicting source values. This entry
does not discover local files, load Maa, or audit statistical/training quality.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

if __package__:
    from .real_game_merge import (
        ResearchMerger, annotate_source_value_conflicts, initialize_database,
    )
    from .real_game_selected import iter_selected_native_payloads
else:
    from real_game_merge import (
        ResearchMerger, annotate_source_value_conflicts, initialize_database,
    )
    from real_game_selected import iter_selected_native_payloads


SCHEMA = Path(__file__).resolve().parents[2] / "Background/RealGameData/research_schema.sql"


def import_payloads(connection, payloads, *, schema=SCHEMA):
    """Merge (label, payload) pairs, committing each run's rows and conflicts."""
    initialize_database(connection, schema)
    merger = ResearchMerger(connection)
    for label, payload in payloads:
        with connection:
            result = merger.merge_payload(label, payload)
            conflicts = annotate_source_value_conflicts(connection, merger.variant_object_ids)
        yield dict(result, source_conflicts=conflicts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+", help="Native JSON/GZIP, upload ZIP, or SQLite file")
    parser.add_argument("--database", type=Path, required=True, help="SQLite destination to create or merge into")
    parser.add_argument("--schema", type=Path, default=SCHEMA, help="Official schema SQL; defaults to the repository copy")
    args = parser.parse_args(argv)
    args.database.parent.mkdir(parents=True, exist_ok=True)
    totals = dict(event="complete", batches=0, inserted=0, updated=0, variants=0)
    with closing(sqlite3.connect(args.database)) as connection:
        payloads = iter_selected_native_payloads(args.sources)
        for result in import_payloads(connection, payloads, schema=args.schema):
            totals["batches"] += 1
            for key in ("inserted", "updated", "variants"):
                totals[key] += result[key]
            print(json.dumps(dict(event="batch", batch=totals["batches"], **result),
                             ensure_ascii=False, sort_keys=True), flush=True)
    print(json.dumps(totals, ensure_ascii=False, sort_keys=True), flush=True)
    return totals


if __name__ == "__main__":
    main()
