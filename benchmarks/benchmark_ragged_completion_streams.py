#!/usr/bin/env python3
"""Ragged Reference Completion's phases on a subset of a real Store (issue #424).

`complete-ragged` held the whole completed Store in memory in Phase 3 (merge,
assembly and write): OGS-00006's 10.0B fills would have needed about 1.7 TB.
This completes a subset of a Ragged source and records each phase's wall time
and peak RSS, so Phase 3 can be projected to the full Store.

The subset is two existing mechanisms, nothing new in the package: the
ancestry-match impute mask (`impute_analysis_ids`, ADR 0028) completes the
first `--n-analyses` Analyses and carries the rest observed-only, and the LD
panel may be a directory of a few chromosomes linked from the real one.

Phases are timed by wrapping the completion module's own phase functions, by
whichever name the code under test has, so the same script times the code
before #424 (whose merge is `_merge_checkpoint_rows`) and after it (whose is
`shard_checkpoint_fills`). Phase 3 is everything after the blocks: the merge
or shard pass, the assembly, the CSR write with its variant index, and the
top hits with `analyses.tsv`. RSS is the process's current RSS sampled during
each phase (`benchmarks/_rss.py`); run the script under `/usr/bin/time -v` for
the process's lifetime peak too. Phase 2's workers are other processes.

A run records the commit of the tree whose package it imported, and refuses a
tree with uncommitted changes under `opengwasdb/` or one that is no checkout:
the code before #424 is measured from a detached worktree of its commit
(`PYTHONPATH` set to it, `python -P`), never from an unversioned copy.

  /usr/bin/time -v pixi run -e dev python benchmarks/benchmark_ragged_completion_streams.py \\
      run --source /data/opengwasdb/stores/OGS-00006/store.opengwasdb \\
      --ld-dir /data/opengwasdb/work/odb424/panel-chr15-22 --n-analyses 100 \\
      --dest /data/opengwasdb/work/odb424/runs/head-100.opengwasdb --n-workers 16 \\
      --output /data/opengwasdb/work/odb424/runs/head-100.json

  pixi run -e dev python benchmarks/benchmark_ragged_completion_streams.py \\
      shard-append --output /data/opengwasdb/work/odb424/runs/shard-append.json
  pixi run -e dev python benchmarks/benchmark_ragged_completion_streams.py \\
      batch-peak --output /data/opengwasdb/work/odb424/runs/batch-peak.json

  pixi run -e dev python benchmarks/benchmark_ragged_completion_streams.py \\
      project RUN_SMALL.json RUN_LARGE.json --full-fills 10001552498 \\
      --full-checkpoints 1348 --shard-append .../shard-append.json \\
      --batch-peak .../batch-peak.json --output .../projection.json

Since issue #430 each Analysis is completed only inside its regions, so a
Store whose Analyses have no Trait position (OGS-00006) is run with its
bundle's region table, `--regions .../sidecars/sparse_regions.tsv`. A run given
one also accounts for every completed Analysis, read back from the completed
Store and checked against the table by a test written here, not the package's:
its significant regions, the cells completion added and how many of them lie
outside those regions (which must be none). `regions-projection` counts, for
every Analysis of the full Store, the panel variants inside its regions that it
does not observe: the cells a full completion adds. Run on the subset's panel
it is checked against a run's measured cells, Analysis by Analysis, and on the
full panel it gives the `--full-fills` that `project` reads.

  pixi run -e dev python benchmarks/benchmark_ragged_completion_streams.py \\
      regions-projection --source SOURCE --regions .../sparse_regions.tsv \\
      --ld-dir /data/opengwasdb/reference/ld --run RUN.json --output .../full.json

Two costs a subset cannot show are measured apart. `shard-append`: the shard
pass files each checkpoint's records one Analysis batch at a time, and a
subset's checkpoints span one batch where the full Store's span up to its
whole batch count; it times that step on a checkpoint of the full Store's
shape at one batch and at the full count, and `project` adds the difference
for every checkpoint. `batch-peak`: a subset's fills all fit one batch, so its
assembly never holds a full one; it measures what reading one batch shard
allocates per byte of shard, and what stays held while the next is read.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
import sys
import tempfile
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from benchmarks._artifact import provenance, write_artifact
from benchmarks._rss import RssSampler
from opengwasdb.completion.regions import COMPLETED_KINDS
from opengwasdb.layouts.ragged import complete
from opengwasdb.model.analyses import read_analysis_records

#: The completion module's phase functions, under either generation's names.
PHASES = (
    "_plan_blocks",
    "_run_pending_blocks",
    "_merge_checkpoint_rows",
    "shard_checkpoint_fills",
    "_assemble_completed_csr",
    "_write_completed_zarr",
    "build_variant_index",
    "_write_top_hits_and_analyses",
)
MERGES = ("_merge_checkpoint_rows", "shard_checkpoint_fills")
#: Phase 3 after the merge; the CSR write's time includes its variant index.
PHASE3_AFTER_MERGE = (
    "_assemble_completed_csr",
    "_write_completed_zarr",
    "_write_top_hits_and_analyses",
)


def _measured_commit() -> str:
    """The commit of the tree whose `opengwasdb` this process imported."""
    import opengwasdb

    root = Path(opengwasdb.__file__).resolve().parent.parent

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )

    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        raise SystemExit(f"{root} is no git checkout; measure a worktree of the commit instead")
    dirty = git("status", "--porcelain", "--", "opengwasdb").stdout.strip()
    if dirty:
        raise SystemExit(f"{root} has uncommitted changes under opengwasdb/:\n{dirty}")
    return head.stdout.strip()


def _checkpoint_fills(blocks_dir: Path) -> tuple[int, int]:
    """How many fills the block checkpoints hold, and their bytes on disk."""
    fills, size = 0, 0
    for path in sorted(Path(blocks_dir).glob("*.npz")):
        size += path.stat().st_size
        with np.load(path, allow_pickle=False) as checkpoint:
            fills += len(checkpoint["f_ai"])
    return fills, size


def _phase_modules() -> list[Any]:
    """Where the code under test looks its phase functions up: the completion
    module, and the shared shard module it calls through, where there is one."""
    try:
        from opengwasdb.completion import fill_shards
    except ImportError:  # the code before #424 has no shared shard module
        return [complete]
    return [complete, fill_shards]


def _instrument(records: dict[str, Any]) -> None:
    """Wrap each phase function the modules have, recording its time and peak RSS."""
    for module in _phase_modules():
        for name in PHASES:
            _wrap(module, name, records)


def _wrap(module: Any, name: str, records: dict[str, Any]) -> None:
    """Replace `module.name`, when it exists, with a timed and sampled call."""
    function = getattr(module, name, None)
    if function is None:
        return

    def timed(*args: Any, _function: Any = function, _name: str = name, **kwargs: Any) -> Any:
        if _name in MERGES:
            records["checkpoint_fills"], records["checkpoint_bytes"] = _checkpoint_fills(args[0])
        started = perf_counter()
        with RssSampler(interval=0.2) as sampler:
            result = _function(*args, **kwargs)
        records.setdefault("phases", {})[_name] = {
            "seconds": round(perf_counter() - started, 2),
            "peak_rss_gb": round(sampler.peak_mb / 1024, 3),
        }
        return result

    setattr(module, name, timed)


def _phase3(phases: dict[str, dict[str, float]]) -> dict[str, float]:
    """Phase 3's time and peak: the merge (or shard pass) and every phase after it."""
    parts = [next(phases[name] for name in MERGES if name in phases)]
    parts += [phases[name] for name in PHASE3_AFTER_MERGE]
    return {
        "seconds": round(sum(part["seconds"] for part in parts), 2),
        "peak_rss_gb": max(part["peak_rss_gb"] for part in parts),
    }


def run(args: argparse.Namespace) -> None:
    measured = _measured_commit()
    analyses = sorted(
        read_analysis_records(args.source / "analyses.tsv"), key=lambda a: int(a.analysis_index)
    )
    chosen = {a.analysis_id for a in analyses[: args.n_analyses]}
    records: dict[str, Any] = {}
    _instrument(records)
    started = perf_counter()
    regions = {} if args.regions is None else {"regions": args.regions}
    result = complete.complete_ragged_store(
        args.source,
        args.dest,
        args.ld_dir,
        ancestry=args.ancestry,
        n_workers=args.n_workers,
        overwrite=args.overwrite,
        impute_analysis_ids=chosen,
        **regions,
    )
    payload = {
        **provenance(),
        # The code measured, not the harness's checkout (they differ for 1ab2b09).
        "commit": measured,
        "argv": sys.argv,
        "n_analyses_completed": len(chosen),
        "n_analyses": len(analyses),
        "result": {
            name: int(getattr(result, name))
            for name in ("n_variants", "n_associations", "n_imputed", "n_missing")
        },
        "total_seconds": round(perf_counter() - started, 2),
        "ru_maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
        **records,
        "phase3": _phase3(records["phases"]),
    }
    if args.regions is not None:
        payload["region_accounting"] = _region_accounting(
            args.source, args.dest, args.regions, args.ld_dir, args.ancestry,
            [i for i, a in enumerate(analyses) if a.analysis_id in chosen],
        )
    write_artifact(args.output, payload)
    print(json.dumps({
        "phase3": payload["phase3"], "result": payload["result"],
        "regions": payload.get("region_accounting", {}).get("summary"),
    }))



def _table_regions(table: Path) -> dict[str, list[tuple[str, int, int, str]]]:
    """Each Analysis's `cis` and `significant_trans` rows, as `(chromosome, start, end, kind)`."""
    import csv

    out: dict[str, list[tuple[str, int, int, str]]] = {}
    with open(table, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["region_kind"] in COMPLETED_KINDS:
                out.setdefault(row["analysis_id"], []).append(
                    (row["chromosome"], int(row["start"]), int(row["end"]), row["region_kind"])
                )
    return out


def _in_any(
    chromosome: np.ndarray, position: np.ndarray, held: list[tuple[str, int, int, str]]
) -> np.ndarray:
    found = np.zeros(len(position), dtype=bool)
    for c, start, end, _kind in held:
        found |= (chromosome == c) & (position >= start) & (position <= end)
    return found


def _identities(store: Path, variant_index: np.ndarray) -> dict[str, np.ndarray]:
    from contextlib import closing

    from opengwasdb.variants.axis import VariantAxis

    with closing(VariantAxis(store)) as axis:
        found = axis.identity_by_indices(np.asarray(variant_index, dtype=np.int64))
    if found is None:
        raise SystemExit(f"{store} has no ALID index to read identities from")
    return found


def _region_accounting(
    source: Path, dest: Path, table: Path, ld_dir: Path, ancestry: str, completed: list[int]
) -> dict[str, Any]:
    """Each completed Analysis's significant regions and the cells completion added it,
    read back from the two Stores: an added cell is a completed row whose ALID the
    source does not hold for that Analysis."""
    from opengwasdb.associations import Cells, CsrReader
    from opengwasdb.completion.ld_panel import list_chromosomes

    regions = _table_regions(table)
    on_panel = set(list_chromosomes(ld_dir, ancestry))
    analyses = sorted(
        read_analysis_records(dest / "analyses.tsv"), key=lambda a: int(a.analysis_index)
    )
    src_reader, dst_reader = CsrReader.open(source), CsrReader.open(dest, requires_mask=False)
    rows = []
    for i in completed:
        before = src_reader.inspect(Cells.analysis(i))
        after = dst_reader.inspect(Cells.analysis(i))
        observed = set(_identities(source, before.variant_index)["alid"].tolist())
        ident = _identities(dest, after.variant_index)
        added = np.array([a not in observed for a in ident["alid"].tolist()], dtype=bool)
        held = regions.get(analyses[i].analysis_id, [])
        here = [r for r in held if r[0] in on_panel]
        inside = _in_any(ident["chromosome"], ident["position"], here)
        imputed = (after.flags == 1) if after.flags is not None else np.zeros(len(added), bool)
        rows.append({
            "analysis_id": analyses[i].analysis_id,
            "trait_position": bool(analyses[i].trait_chr and analyses[i].trait_bp),
            "significant_regions": sum(r[3] == "significant_trans" for r in held),
            "regions_on_panel": len(here),
            "region_bp_on_panel": int(sum(r[2] - r[1] + 1 for r in here)),
            "observed": int(len(before.variant_index)),
            "imputed": int(imputed.sum()),
            "missing_added": int((added & np.isnan(after.z)).sum()),
            "added": int(added.sum()),
            "added_outside_regions": int((added & ~inside).sum()),
            "imputed_outside_regions": int((imputed & ~inside).sum()),
            "observed_rows_kept": int((~added).sum()),
        })
    no_region = [r for r in rows if r["regions_on_panel"] == 0]
    return {
        "panel_chromosomes": sorted(on_panel, key=lambda c: (len(c), c)),
        "analyses": rows,
        "summary": {
            "analyses": len(rows),
            "imputed": sum(r["imputed"] for r in rows),
            "missing_added": sum(r["missing_added"] for r in rows),
            "added_outside_regions": sum(r["added_outside_regions"] for r in rows),
            "imputed_outside_regions": sum(r["imputed_outside_regions"] for r in rows),
            "observed_rows_lost": sum(r["observed"] - r["observed_rows_kept"] for r in rows),
            "no_region_on_panel": len(no_region),
            "no_significant_region": sum(r["significant_regions"] == 0 for r in rows),
            "added_to_no_region_analyses": sum(r["added"] for r in no_region),
            "imputed_per_region_mb": round(
                sum(r["imputed"] for r in rows)
                / max(1, sum(r["region_bp_on_panel"] for r in rows)) * 1e6, 1
            ),
        },
    }


@dataclass(frozen=True)
class _PanelChromosome:
    """One chromosome of a panel: its distinct canonical ALIDs, their positions, and its
    blocks' extents."""

    alids: np.ndarray
    positions: np.ndarray
    blocks: list[tuple[str, int, int]]


def _panel(ld_dir: Path, ancestry: str) -> dict[str, _PanelChromosome]:
    from opengwasdb.completion.ld_panel import (
        canonical_panel_alid,
        list_all_blocks,
        list_chromosomes,
    )

    out = {}
    for chromosome in list_chromosomes(ld_dir, ancestry):
        blocks = list_all_blocks(ld_dir, ancestry, chromosome)
        found: dict[str, int] = {}
        for block in blocks:
            for snp in block.snp_ids:
                alid = canonical_panel_alid(snp)
                if alid is not None:
                    found.setdefault(alid, int(alid.split(":")[1]))
        out[chromosome] = _PanelChromosome(
            np.array(list(found), dtype=object),
            np.fromiter(found.values(), dtype=np.int64, count=len(found)),
            [(b.block_id, b.start_bp, b.end_bp) for b in blocks],
        )
    return out


def _expected_added(
    panel: dict[str, _PanelChromosome], held: list[tuple[str, int, int, str]],
    identities: dict[str, np.ndarray],
) -> tuple[int, set[str]]:
    """One Analysis's panel variants inside its regions that it does not observe, and the
    blocks its regions overlap."""
    added, touched = 0, set()
    for chromosome in {r[0] for r in held} & panel.keys():
        here = [r for r in held if r[0] == chromosome]
        p = panel[chromosome]
        inside = _in_any(np.full(len(p.positions), chromosome, dtype=object), p.positions, here)
        observed = set(identities["alid"][identities["chromosome"] == chromosome].tolist())
        added += len(set(p.alids[inside].tolist()) - observed)
        touched |= {
            block for block, start, end in p.blocks
            if any(not (end < r[1] or start > r[2]) for r in here)
        }
    return added, touched


def regions_projection(args: argparse.Namespace) -> None:
    """The cells a completion adds each Analysis: the panel variants inside its regions it
    does not observe. On a run's panel, checked against what that run added, Analysis by
    Analysis; on the full panel, the full Store's cells, split into imputed and missing by
    the run's measured share."""
    from opengwasdb.associations import Cells, CsrReader

    regions = _table_regions(args.regions)
    analyses = sorted(
        read_analysis_records(args.source / "analyses.tsv"), key=lambda a: int(a.analysis_index)
    )
    reader = CsrReader.open(args.source)
    started = perf_counter()
    panel = _panel(args.ld_dir, args.ancestry)
    expected: dict[str, int] = {}
    blocks_touched: set[str] = set()
    analysis_blocks = 0
    for i, a in enumerate(analyses):
        held = regions.get(a.analysis_id, [])
        if not held:
            expected[a.analysis_id] = 0
            continue
        identities = _identities(args.source, reader.inspect(Cells.analysis(i)).variant_index)
        expected[a.analysis_id], touched = _expected_added(panel, held, identities)
        blocks_touched |= touched
        analysis_blocks += len(touched)
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    summary = run["region_accounting"]["summary"]
    # A run is checked against only on its own panel: elsewhere it completed nothing.
    same_panel = run["region_accounting"]["panel_chromosomes"] == sorted(
        panel, key=lambda c: (len(c), c)
    )
    measured = {
        r["analysis_id"]: r["added"] for r in run["region_accounting"]["analyses"]
    } if same_panel else {}
    total = sum(expected.values())
    imputed_share = summary["imputed"] / max(1, summary["imputed"] + summary["missing_added"])
    out = {
        **provenance(),
        "commit": _measured_commit(),
        "argv": sys.argv,
        "seconds": round(perf_counter() - started, 1),
        "analyses": len(analyses),
        "with_region": sum(1 for a in analyses if regions.get(a.analysis_id)),
        "blocks_touched": len(blocks_touched),
        "analysis_blocks": analysis_blocks,
        "added_cells": total,
        "added_cells_per_analysis": {
            "median": float(np.median(list(expected.values()))),
            "max": max(expected.values()),
        },
        "imputed_share_from_run": round(imputed_share, 4),
        "imputed_cells": round(total * imputed_share),
        "missing_cells": round(total * (1 - imputed_share)),
        "checked_against_run": {
            "analyses": len(measured),
            "disagreeing": {k: [expected[k], v] for k, v in measured.items() if expected[k] != v},
        } if same_panel else None,
    }
    write_artifact(args.output, out)
    print(json.dumps({k: v for k, v in out.items() if k != "argv"}, indent=1))


def _line(small: float, large: float, x_small: float, x_large: float, x: float) -> float:
    """The straight line through two measurements, read at `x`."""
    return large + (large - small) / (x_large - x_small) * (x - x_large)


def shard_append(args: argparse.Namespace) -> None:
    """Time `fill_shards._append_to_shards` on one checkpoint of the full Store's
    shape, keyed into one Analysis batch and into `--batches`."""
    from opengwasdb.completion import fill_shards

    rng = np.random.default_rng(424)
    ai = np.repeat(np.arange(args.analyses, dtype=np.int32), args.fills_per_block)
    rows = rng.integers(0, args.n_variants, len(ai)).astype(np.int32)
    z = rng.normal(size=len(ai)).astype(np.float32)
    fills = (rows, ai, z, np.abs(z) + np.float32(0.1))
    seconds: dict[str, float] = {}
    for label, batches in (("one_batch", 1), ("full_batches", args.batches)):
        batch_of = np.arange(args.analyses, dtype=np.int64) // math.ceil(args.analyses / batches)
        times = []
        for _ in range(args.repeats):
            with tempfile.TemporaryDirectory(dir=args.scratch) as scratch:
                started = perf_counter()
                fill_shards._append_to_shards(Path(scratch), fills, batch_of[ai])
                times.append(perf_counter() - started)
        seconds[label] = round(float(np.median(times)), 3)
    out = {
        **provenance(),
        "commit": _measured_commit(),
        "records": len(ai),
        "batches": int(args.batches),
        "seconds": seconds,
        "extra_seconds_per_checkpoint": round(seconds["full_batches"] - seconds["one_batch"], 3),
    }
    write_artifact(args.output, out)
    print(json.dumps(out))


def batch_peak(args: argparse.Namespace) -> None:
    """What reading one batch shard (`_batch_fills`) allocates, per byte of shard,
    and what it leaves held, by `tracemalloc` (numpy reports its buffers to it)."""
    from opengwasdb.completion.fill_shards import FILL_RECORD_DTYPE, fill_shard_path

    rng = np.random.default_rng(424)
    records = np.empty(args.records, dtype=FILL_RECORD_DTYPE)
    records["ai"] = rng.integers(0, args.analyses, args.records)
    records["row"] = rng.integers(0, args.n_variants, args.records)
    records["z"], records["se"] = 1.0, 0.1
    with tempfile.TemporaryDirectory(dir=args.scratch) as scratch:
        records.tofile(fill_shard_path(Path(scratch), 0))
        shard_bytes = records.nbytes
        del records
        tracemalloc.start()
        held = complete._batch_fills(Path(scratch), 0)
        after, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del held
    out = {
        **provenance(),
        "commit": _measured_commit(),
        "shard_bytes": shard_bytes,
        "peak_per_shard_byte": round(peak / shard_bytes, 3),
        "held_per_shard_byte": round(after / shard_bytes, 3),
    }
    write_artifact(args.output, out)
    print(json.dumps(out))


#: What the writer holds per completed cell until it flushes: variant index, z,
#: SE and EAF, and the imputed flag (`RaggedCSRWriter.add_analysis`).
HELD_BYTES_PER_CELL = sum(np.dtype(t).itemsize for t in ("i4", "f4", "f4", "f4", "u1"))
#: A fill record's bytes in a batch shard (`fill_shards.FILL_RECORD_DTYPE`).
RECORD_BYTES = sum(np.dtype(t).itemsize for t in ("i4", "i4", "f4", "f4"))


def _gb(n_bytes: float) -> float:
    return n_bytes / 1024**3


def _memory_model(
    run: dict[str, Any], cells: float, batch_records: float, peak: dict[str, Any]
) -> dict[str, float]:
    """Phase 3's peak RSS, from its parts. Before assembly the process holds what
    Phases 1 and 2 left (the shard pass's peak). Assembly adds the writer's held
    cells and, while one batch shard is read (`batch_peak`), its peak allocation
    beside the previous batch, still held. The write frees the batch and adds one
    region's working set, which `run` measures: its write peak less what it held."""
    before = run["phases"]["shard_checkpoint_fills"]["peak_rss_gb"]
    held = _gb(HELD_BYTES_PER_CELL * cells)
    shard = batch_records * RECORD_BYTES
    batch = _gb((peak["peak_per_shard_byte"] + peak["held_per_shard_byte"]) * shard)
    region = run["phases"]["_write_completed_zarr"]["peak_rss_gb"] - before - _gb(
        HELD_BYTES_PER_CELL * run["result"]["n_associations"]
    )
    return {
        "before_assembly_gb": round(before, 2),
        "held_cells_gb": round(held, 1),
        "batch_gb": round(batch, 2),
        "write_region_gb": round(region, 2),
        "assembly_peak_gb": round(before + held + batch, 1),
        "write_peak_gb": round(before + held + region, 1),
    }


def project(args: argparse.Namespace) -> None:
    """Phase 3 at the full Store, from two runs on one panel subset.

    Time is read off the line through both runs' Phase 3 time against their
    checkpoint fills, plus the shard pass's extra cost per checkpoint at the
    full Store's batch count (`shard_append`), which must be at least the
    count the full Store's cells make at the batch budget.

    Memory is not read off a line: a subset's fills all fit one Analysis batch,
    so its assembly peak grows by a batch's records per fill, which at full
    size stop at one batch, or at every fill where they fit in one. It is
    modelled from its parts (`_memory_model`) at the full Store's cells -- its
    observed rows, which do not grow with the fills, plus the fills with the
    missing rows the larger run makes per fill -- and the model is set beside
    the larger run's measured peaks. The larger
    run must hold more cells than one sequence shard, so its write region is
    full size.
    """
    small, large = (json.loads(Path(p).read_text(encoding="utf-8")) for p in args.runs)
    fills = [run["checkpoint_fills"] for run in (small, large)]
    completed = large["result"]["n_imputed"] + large["result"]["n_missing"]
    observed = large["result"]["n_associations"] - completed
    full_cells = observed + args.full_fills * completed / fills[1]
    budget = complete._FILL_BATCH_CELLS
    full_batches = math.ceil(full_cells / budget)
    append = json.loads(Path(args.shard_append).read_text(encoding="utf-8"))
    if append["batches"] < full_batches:
        raise SystemExit(
            f"shard-append measured {append['batches']} batches; the full Store makes "
            f"{full_batches}"
        )
    peak = json.loads(Path(args.batch_peak).read_text(encoding="utf-8"))
    shard_extra = append["extra_seconds_per_checkpoint"] * args.full_checkpoints
    fitted = _line(small["phase3"]["seconds"], large["phase3"]["seconds"], *fills, args.full_fills)
    out: dict[str, Any] = {
        **provenance(),
        "commit": large["commit"],
        "runs": args.runs,
        "full_fills": args.full_fills,
        "full_cells": round(full_cells),
        "full_batches": full_batches,
        "phase3_seconds_fitted": round(fitted),
        "shard_batches_extra_seconds": round(shard_extra),
        "phase3_seconds": round(fitted + shard_extra),
        "phase3_hours": round((fitted + shard_extra) / 3600, 2),
        # A batch holds at most the budget, and at most every fill (issue #430's Stores
        # fill far less than one batch).
        "memory_full": _memory_model(large, full_cells, min(args.full_fills, budget), peak),
        "memory_model_at_larger_run": _memory_model(
            large, large["result"]["n_associations"], min(fills[1], budget), peak
        ),
        "memory_measured_at_larger_run": {
            "assembly_peak_gb": large["phases"]["_assemble_completed_csr"]["peak_rss_gb"],
            "write_peak_gb": large["phases"]["_write_completed_zarr"]["peak_rss_gb"],
        },
    }
    write_artifact(args.output, out)
    print(json.dumps(out, indent=1))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run", help="complete a subset and time its phases")
    one.add_argument("--source", type=Path, required=True)
    one.add_argument("--ld-dir", type=Path, required=True)
    one.add_argument("--ancestry", default="EUR")
    one.add_argument("--n-analyses", type=int, required=True)
    one.add_argument("--dest", type=Path, required=True)
    one.add_argument("--n-workers", type=int, default=16)
    one.add_argument("--overwrite", action="store_true")
    one.add_argument("--regions", type=Path, default=None, help="the bundle's region table")
    one.add_argument("--output", type=Path, required=True)
    regions = commands.add_parser(
        "regions-projection", help="count the cells a completion adds each Analysis"
    )
    regions.add_argument("--source", type=Path, required=True)
    regions.add_argument("--regions", type=Path, required=True)
    regions.add_argument("--ld-dir", type=Path, required=True)
    regions.add_argument("--ancestry", default="EUR")
    regions.add_argument("--run", type=Path, required=True, help="a run's JSON to check against")
    regions.add_argument("--output", type=Path, required=True)
    append = commands.add_parser("shard-append", help="time the shard pass's per-batch filing")
    append.add_argument("--analyses", type=int, default=1100, help="Analyses one block fills")
    append.add_argument("--fills-per-block", type=int, default=6600)
    append.add_argument("--batches", type=int, default=47, help="batches at full size")
    append.add_argument("--n-variants", type=int, default=12_400_000)
    append.add_argument("--repeats", type=int, default=5)
    append.add_argument("--scratch", type=Path, default=None)
    append.add_argument("--output", type=Path, required=True)
    peak = commands.add_parser("batch-peak", help="measure what reading a batch shard holds")
    peak.add_argument("--records", type=int, default=1 << 25)
    peak.add_argument("--analyses", type=int, default=30, help="Analyses in one batch")
    peak.add_argument("--n-variants", type=int, default=12_400_000)
    peak.add_argument("--scratch", type=Path, default=None)
    peak.add_argument("--output", type=Path, required=True)
    both = commands.add_parser("project", help="project Phase 3 to the full Store")
    both.add_argument("runs", nargs=2, help="the smaller run's JSON, then the larger's")
    both.add_argument("--full-fills", type=int, required=True)
    both.add_argument("--full-checkpoints", type=int, required=True)
    both.add_argument("--shard-append", type=Path, required=True)
    both.add_argument("--batch-peak", type=Path, required=True)
    both.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    commands = {
        "run": run, "shard-append": shard_append, "batch-peak": batch_peak,
        "regions-projection": regions_projection,
    }
    commands.get(args.command, project)(args)


if __name__ == "__main__":
    main()
