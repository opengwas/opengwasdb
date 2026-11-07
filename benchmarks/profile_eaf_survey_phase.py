#!/usr/bin/env python3
"""Attribute the wall time of a Dense build's EAF survey and orientation verification (#310).

The phase runs between Pass 2 and the encoding plan. Reaching it needs Pass 1
and Pass 2, so this script builds a stride-sampled subset of a manifest's
Analyses and stops the build the moment the phase ends: nothing after the
survey is written. It records:

* the build's own phase lines (survey, parent accumulation, EAF Reference
  load, orientation verification, the whole phase), as logged;
* the split of one survey worker's per-Analysis cost, measured serially on the
  spills the build produced, step by step, checked equal to the production
  sampler's output.

  pixi run -e dev python benchmarks/profile_eaf_survey_phase.py \
      /data/opengwasdb/stores/OGS-00009/work/analyses.tsv --stride 10 \
      --work /data/opengwasdb/work/orchestrate/opengwasdb-307/scratch/310/ogs00009 \
      --n-workers 32 --source-reader-capability opengwasdb.gwas-vcf \
      --source-assembly hg19 --output docs/benchmark-output/eaf_survey_phase_ogs00009.json

The phase's cost is linear in the Analyses, so a subset's seconds scale by the
ratio of Analysis counts at the same ``--n-workers``; the artifact records both
counts so the scaling is stated, not assumed.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from benchmarks import _artifact
from opengwasdb.build import eaf_orientation as eo
from opengwasdb.layouts.dense import build_vcf

_PHASE_LINE = re.compile(r"^(EAF [^:]+): (start|done in (.+?))( across .*)?$")
_SAMPLED_COLUMNS = 8


class _StopAfterSurvey(Exception):
    """Raised in place of the encoding step, so nothing after the phase runs."""


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("EAF "):
            self.lines.append({"t": round(record.created, 3), "message": message})


def _subset_manifest(source: Path, destination: Path, stride: int) -> int:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        rows = [row for i, row in enumerate(reader) if i % stride == 0]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for i, row in enumerate(rows):
            if "analysis_index" in row:
                row["analysis_index"] = str(i)
            writer.writerow(row)
    return len(rows)


def _split_one_column(spill_dir: Path, col: int, hashes: np.ndarray) -> dict[str, float]:
    """The survey worker's steps for one spill, timed apart and checked against the sampler."""
    clock = time.perf_counter
    started = clock()
    with np.load(spill_dir / f"{col}.npz") as data:
        rows = data["rows"].astype(np.int64)
        eaf = np.asarray(data["eaf"], dtype=np.float64)
    t_load = clock() - started

    started = clock()
    with_eaf = rows[np.isfinite(eaf)]
    n_eaf = int(np.count_nonzero(np.isfinite(eaf)))
    t_finite = clock() - started

    started = clock()
    selected = eo.select_rows(hashes, with_eaf, k=eo.DEFAULT_SAMPLE_SITES)
    t_select = clock() - started

    started = clock()
    lookup = dict(zip(rows.tolist(), eaf.tolist(), strict=True))
    values = np.fromiter(
        (lookup[row] for row in selected.tolist()), dtype=np.float64, count=len(selected)
    )
    t_lookup = clock() - started

    production = eo.sample_column_rows(rows, eaf, hashes, k=eo.DEFAULT_SAMPLE_SITES)
    if not (np.array_equal(production[0], selected) and np.array_equal(production[1], values)):
        raise AssertionError(f"column {col}: the split disagrees with sample_column_rows")
    return {
        "n_rows": int(rows.size),
        "n_eaf_cells": n_eaf,
        "load_npz_and_cast": t_load,
        "finite_filter": t_finite,
        "select_rows_by_hash": t_select,
        "row_to_eaf_dict_and_gather": t_lookup,
    }


def _profile(args: argparse.Namespace) -> dict[str, Any]:
    split: list[dict[str, float]] = []

    def stop(staged: Any, spill_dir: Path, eaf_survey: Any, axis: Any, *rest: Any) -> None:
        hashes = eo.site_hashes(axis.alids)
        cols = sorted(int(p.stem) for p in Path(spill_dir).glob("*.npz") if p.stem.isdigit())
        for col in cols[:: max(1, len(cols) // _SAMPLED_COLUMNS)][:_SAMPLED_COLUMNS]:
            split.append(_split_one_column(Path(spill_dir), col, hashes))
        raise _StopAfterSurvey

    out = args.work / "store.opengwasdb"
    with mock.patch.object(build_vcf, "_write_encoded_bands", stop):
        try:
            build_vcf.build_dense_from_vcf_manifest(
                args.work / "manifest.tsv",
                out,
                store_id="profile",
                release_id="profile",
                overwrite=True,
                n_workers=args.n_workers,
                source_reader_capability=args.source_reader_capability,
                source_assembly=args.source_assembly,
                eaf_reference=args.eaf_reference,
                eaf_reference_ancestry=args.eaf_reference_ancestry,
                admission_record=args.work / "admission.json",
            )
        except _StopAfterSurvey:
            pass
    return {"column_split": split}


def _phases(lines: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    phases: dict[str, dict[str, Any]] = {}
    for line in lines:
        match = _PHASE_LINE.match(line["message"])
        if match:
            entry = phases.setdefault(match.group(1), {})
            entry["start" if match.group(2) == "start" else "done_in"] = (
                match.group(3) or line["t"]
            )
    return phases


def _tree_is_clean() -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    return not status.stdout.strip()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--n-workers", type=int, required=True)
    parser.add_argument("--source-reader-capability")
    parser.add_argument("--source-assembly")
    parser.add_argument("--eaf-reference", type=Path)
    parser.add_argument("--eaf-reference-ancestry")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.work.exists():
        raise SystemExit(f"{args.work}: already exists; refusing to overwrite it")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    args.work.mkdir(parents=True)
    n_total = sum(1 for _ in args.manifest.open(encoding="utf-8")) - 1
    n_subset = _subset_manifest(args.manifest, args.work / "manifest.tsv", args.stride)
    capture = _Capture()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(capture)

    load_before = os.getloadavg()
    started = time.monotonic()
    result = _profile(args)
    wall = time.monotonic() - started
    _artifact.write_artifact(
        args.output,
        {
            **_artifact.provenance(),
            "worktree_clean": _tree_is_clean(),
            "nproc": os.cpu_count(),
            "argv": sys.argv if argv is None else argv,
            "n_analyses_in_manifest": n_total,
            "n_analyses_profiled": n_subset,
            "scale_to_manifest": n_total / n_subset,
            "n_workers": args.n_workers,
            "wall_seconds_to_survey_end": wall,
            "loadavg_before": load_before,
            "loadavg_after": os.getloadavg(),
            "phases": _phases(capture.lines),
            "log": capture.lines,
            **result,
        },
    )


if __name__ == "__main__":
    main()
