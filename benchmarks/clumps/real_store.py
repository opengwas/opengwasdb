#!/usr/bin/env python3
"""ADR 0106 (#418): `opengwasdb clump` end to end on a real 0.3.0 Store Release, on a copy.

    real_store.py SOURCE WORK --ld-store ANCESTRY=PATH ... [--write-sidecars] --output OUT.json

SOURCE is read only. It is copied to WORK (a reflink where the filesystem can), validated,
clumped, and validated again with the same LD Stores, so `clumps.reclump` runs. The record
holds the step's own summary, its wall time, and the findings validation adds. WORK must
not lie under a published Store's directory.

`--write-sidecars` gives a copy made before #468 or #470 the position and `source_alid`
sidecars a 0.3.0 release carries, from its own Variant Table, through the writers a build
calls; validation then checks them (`variant_table.position_sidecar`,
`variant_table.source_alid_sidecar`). The 0.3.0 copies epic #450 left were made before both.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from opengwasdb.clumps import clump_store
from opengwasdb.store_reader.position_sidecar import PositionSidecar, write_position_sidecar
from opengwasdb.store_reader.source_alid_sidecar import (
    SourceAlidSidecar,
    write_source_alid_sidecar,
)
from opengwasdb.validation import validate_store
from opengwasdb.variants.legacy_table import iter_variant_records
from opengwasdb.variants.normalise import chromosome_sort_key

PUBLISHED = Path("/data/opengwasdb/stores")


def write_sidecars(store: Path) -> list[str]:
    """Write whichever of the two sidecars `store` lacks, from its Variant Table."""
    records = list(iter_variant_records(store / "variants.tsv.gz"))
    written = []
    if PositionSidecar.open(store) is None:
        write_position_sidecar(store, [r.chromosome for r in records],
                               [r.position for r in records], chromosome_key=chromosome_sort_key)
        written.append("position")
    if SourceAlidSidecar.open(store) is None:
        write_source_alid_sidecar(store, [r.source_alid for r in records],
                                  [r.alid for r in records])
        written.append("source_alid")
    return written


def findings(store: Path, ld_stores: dict[str, str] | None) -> set[tuple[str, str, str, str]]:
    result = validate_store(store, ld_stores=ld_stores)
    return {(f.rule, str(f.severity), f.target, f.message) for f in result.findings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--ld-store", action="append", required=True)
    parser.add_argument("--write-sidecars", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if PUBLISHED in args.work.resolve().parents or args.work.exists():
        raise SystemExit(f"{args.work} exists or lies under {PUBLISHED}; give a new scratch path")
    mapping = dict(value.split("=", 1) for value in args.ld_store)

    subprocess.run(["cp", "-a", "--reflink=auto", str(args.source), str(args.work)], check=True)
    sidecars = write_sidecars(args.work) if args.write_sidecars else []
    started = time.perf_counter()
    before = findings(args.work, None)
    validate_before_s = time.perf_counter() - started

    started = time.perf_counter()
    run = clump_store(args.work, mapping)
    clump_s = time.perf_counter() - started

    started = time.perf_counter()
    after = findings(args.work, mapping)
    validate_after_s = time.perf_counter() - started

    record: dict[str, Any] = {
        "source": str(args.source),
        "sidecars_written": sidecars,
        "clump": {**run.as_dict(), "store": args.source.name,
                  "ld_stores": {a: {k: e[k] for k in ("name", "kind", "format_version",
                                                       "provenance_sha256", "t_far", "r_scale")}
                                for a, e in run.ld_stores.items()}},
        "clump_wall_seconds": round(clump_s, 1),
        "validate": {
            "errors_before": sum(1 for f in before if f[1] == "error"),
            "errors_after": sum(1 for f in after if f[1] == "error"),
            "new_findings": sorted(f"{f[0]}: {f[3]}" for f in after - before),
            "clumps_findings": sorted(f"{f[0]} {f[1]}: {f[3]}" for f in after
                                      if f[0].startswith("clumps.")),
            "seconds_before": round(validate_before_s, 1),
            "seconds_after": round(validate_after_s, 1),
        },
    }
    args.output.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps(record["clump"], indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
