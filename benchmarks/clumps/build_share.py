#!/usr/bin/env python3
"""ADR 0106 (#418): the Clumps step's time as a share of its Store's build.

    build_share.py STORE=CLUMP_RECORD.json ... --output OUT.json

The build's time is the sum of every step the Store's build records hold
(`/data/opengwasdb/stores/<STORE>/records/*.json`, each step's `elapsed_seconds`): its build,
completion, Top-Hit Index, subsets, validation and the rest, as each Store was built. The
step's time is from this directory's records: `clump_wall_seconds` of `real_store.py` (the
step itself), or `seconds.total` of `parity.py` (the compute path, which reads the tiers and
clumps them; writing the arrays is a small part of the step).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RECORDS = Path("/data/opengwasdb/stores")


def clump_seconds(record: dict[str, Any]) -> tuple[float, str]:
    if "clump_wall_seconds" in record:
        return float(record["clump_wall_seconds"]), "opengwasdb clump (the step)"
    return float(record["seconds"]["total"]), "compute_clumps (the compute path)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pairs", nargs="+", help="STORE=CLUMP_RECORD.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for pair in args.pairs:
        store, path = pair.split("=", 1)
        steps = {}
        for step in sorted((RECORDS / store / "records").glob("*.json")):
            elapsed = json.loads(step.read_text()).get("elapsed_seconds")
            if elapsed is not None:
                steps[step.stem] = round(float(elapsed), 1)
        seconds, measured = clump_seconds(json.loads(Path(path).read_text()))
        build = sum(steps.values())
        rows.append({
            "store": store, "build_seconds": round(build, 1), "build_steps": steps,
            "clump_seconds": seconds, "clump_measured_as": measured,
            "share_of_build": round(seconds / build, 4) if build else None,
        })
    args.output.write_text(json.dumps(rows, indent=1) + "\n")
    print(json.dumps([{k: r[k] for k in ("store", "build_seconds", "clump_seconds",
                                         "share_of_build")} for r in rows], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
