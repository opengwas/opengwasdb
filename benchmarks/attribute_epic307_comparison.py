#!/usr/bin/env python3
"""Attribute the completed #322 comparison using existing artifacts and eight metadata rows.

No statistic-plane scan, rebuild or comparison is performed. The SE envelope
includes #355's historical float16 narrowing before residual encoding; it does
not relabel the comparison's stricter codec-only counts or prove source equality.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from gzip import open as gzip_open
from math import log1p
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks._artifact import provenance, write_artifact
from benchmarks.compare_store_releases import _variant_rows
from opengwasdb.store.open import open_store


def source_conflicts(directory: Path) -> dict[str, int]:
    """Count rejected cells only where each Analysis logged exactly two conflicting sources."""
    cells: Counter[str] = Counter()
    for path in sorted(directory.glob("*.tsv.gz")):
        sources: Counter[str] = Counter()
        with gzip_open(path, "rt") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["reason"] == "duplicate_conflicting":
                    sources[row["source_key"]] += 1
        if any(count != 2 for count in sources.values()):
            raise ValueError(f"{path}: a conflict is not a pair; cannot infer rejected cell counts")
        cells.update(sources.keys())
    if not cells:
        raise ValueError("no conflicting source pairs; attribution would prove nothing")
    return dict(cells)


def attribute(scratch: Path) -> dict[str, Any]:
    base = scratch / "offclock/compare/observed-vs-published"
    report = json.loads(base.with_suffix(".json").read_text())
    admission = scratch / "OGS-00009/store.opengwasdb.admission"
    cells = source_conflicts(admission / "dropped")
    codes = np.load(base.with_suffix(".codes.npz"))
    presence = codes["matched_left_rows"][(codes["matched_codes"] & 16) != 0]
    eaf = codes["matched_left_rows"][(codes["matched_codes"] & 8) != 0]
    rows = _variant_rows(Path(report["left"]), presence)
    examples = {
        row["left_row"]: row["cells_in_row"]
        for row in report["matched_by_alid"]["planes"]["z"]["examples"]
    }
    with base.with_suffix(".unmatched.tsv").open() as stream:
        unmatched = list(csv.DictReader(stream, delimiter="\t"))
    primary = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}
    off_primary = all(
        row["kind"] == "right_only" and row["chromosome"] not in primary and row["source_alid"]
        for row in unmatched
    )
    matched = report["matched_by_alid"]["planes"]
    se = matched["se"]
    bound = se["se_log_bound"] + log1p(2**-11)
    conflicts_match = {
        record["alid"]: examples.get(index) for index, record in rows.items()
    } == cells and sum(cells.values()) == matched["z"]["only_right"]
    left = open_store(report["left"]).arrays(mode="r")["eaf_baseline"]
    right = open_store(report["right"]).arrays(mode="r")["eaf_baseline"]
    metadata = []
    for index in eaf:
        slot = np.searchsorted(codes["matched_left_rows"], index)
        mapped = int(codes["matched_right_rows"][slot])
        metadata.append(
            {
                "left_row": int(index),
                "right_row": mapped,
                "baseline_left": float(left[int(index)]),
                "baseline_right": float(right[mapped]),
            }
        )
    return {
        "comparison": str(base.with_suffix(".json")),
        "lift_off_primary_376": {
            "rows": len(unmatched),
            "all_off_primary_lifts": bool(off_primary),
        },
        "duplicate_conflicting_357": {
            "cells_by_alid": cells,
            "matches_presence_differences": conflicts_match,
        },
        "se_355": {
            "historical_float16_log_half_step": log1p(2**-11),
            "two_codec_log_bound": se["se_log_bound"],
            "combined_log_envelope": bound,
            "observed_max_abs_log_ratio": se["max_abs_log_ratio"],
            "max_within_historical_envelope": se["max_abs_log_ratio"] <= bound,
            "note": "#355 removed float16 narrowing before the residual fit, not just "
            "from exceptions. Counts beyond the codec-only bound are unchanged; "
            "this larger historical envelope is not a source-equality proof.",
        },
        "eaf_after_conflict_drops": {
            "differing_cells": matched["eaf"]["value_differs"],
            "rows": metadata,
            "only_at_conflicting_rows": set(eaf).issubset(set(presence)),
            "all_baselines_changed": all(
                row["baseline_left"] != row["baseline_right"] for row in metadata
            ),
        },
        "reference_first_381": {"respelled_rows": report["respelled"]["rows"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scratch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_artifact(
        args.output,
        {
            "generator": "benchmarks/attribute_epic307_comparison.py",
            **provenance(),
            **attribute(args.scratch),
        },
    )


if __name__ == "__main__":
    main()
