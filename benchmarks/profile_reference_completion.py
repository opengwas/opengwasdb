#!/usr/bin/env python3
"""Attribute a Reference Completion's wall time to what it does (#309).

A full completion of OGS-00009 took 14 h 05 m, and its log said only how many LD
blocks had finished. This harness measures the two places the time can hide, on
real input, without running a completion:

``block``
    One real LD block completed for every Analysis of a Store, with its time
    split into the LD-panel read, the source-plane read, the elastic-net fits,
    the polynomial rescale, the per-cell fill loop, the rest of the per-block
    bookkeeping and the checkpoint write. The stages are *exclusive*: a stage
    entered inside another pauses it, so the shares add up to the block's wall
    time.

``bands``
    The post-block band write over a prefix of the completed matrix's row
    bands, with real fills: it completes the LD blocks that cover the prefix,
    shards their fills, and runs the statistic pass, the EAF pass and the
    residual-SE rewrite on those bands only. The prefix is whole bands of the
    real Store's union axis, so each band reads the same source planes and
    writes the same shape as in a full run; the artifact states how many bands
    of the full matrix it ran.

Every artifact records the revision it ran, its argv and the host's load before
and after, so a number cannot outlive the conditions that produced it.

  pixi run -e dev python benchmarks/profile_reference_completion.py block \
      SOURCE LD_PANEL --work SCRATCH --output block.json
  pixi run -e dev python benchmarks/profile_reference_completion.py bands \
      SOURCE LD_PANEL --work SCRATCH --bands 2 --output bands.json
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from benchmarks._artifact import provenance, write_artifact
from opengwasdb.encoding.timing import PhaseTimer

DEFAULT_MIN_COR = 0.7
DEFAULT_THRESH = 0.9


class StageClock:
    """Exclusive seconds per stage, over stages that nest.

    ``PhaseTimer`` partitions work and asks its callers not to nest. A block's
    fits run inside the per-Analysis kernel which runs inside the block runner,
    and the answer wanted is how long each takes *itself*. Entering a stage
    stops the clock on the stage it interrupts and entering it again resumes
    that one afterwards, so `total()` is the wall time and shares are real
    fractions of it.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.timer = PhaseTimer()
        self._now = clock
        self._stack: list[str] = []
        self._mark = clock()
        self.calls: dict[str, int] = {}

    def _charge(self, now: float) -> None:
        if self._stack:
            name = self._stack[-1]
            self.timer.seconds[name] = self.timer.seconds.get(name, 0.0) + now - self._mark
        self._mark = now

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self._charge(self._now())
        self._stack.append(name)
        self.calls[name] = self.calls.get(name, 0) + 1
        try:
            yield
        finally:
            self._charge(self._now())
            self._stack.pop()

    def wrap(self, name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def timed(*args: Any, **kwargs: Any) -> Any:
            with self.stage(name):
                return function(*args, **kwargs)

        return timed

    def summary(self) -> dict[str, Any]:
        return {
            "total_seconds": self.timer.total(),
            "stages": [
                {
                    "stage": name,
                    "seconds": seconds,
                    "share": share,
                    "calls": self.calls.get(name, 0),
                }
                for name, seconds, share in self.timer.report()
            ],
        }


def host_load() -> dict[str, Any]:
    one, five, fifteen = os.getloadavg()
    return {"load_1m": one, "load_5m": five, "load_15m": fifteen, "cpus": os.cpu_count()}


def artifact_header(argv: list[str]) -> dict[str, Any]:
    """What an artifact says about the run that made it, taken before it starts."""
    return {
        "tool": "benchmarks/profile_reference_completion.py",
        "argv": argv,
        "host_load_start": host_load(),
        **provenance(),
    }


def _source_axis_size(source: Path) -> tuple[int, int]:
    from opengwasdb.store.open import open_store

    shape = open_store(source).arrays(mode="r")["z"].shape
    return int(shape[0]), int(shape[1])


def choose_block(tsv_paths: list[Path], which: str) -> Path:
    """The block to profile: ``median`` or ``largest`` by variant count, or a path.

    The block size sets the fit cost (eigenvectors grow with the block), so the
    artifact names which one it took.
    """
    if which not in ("median", "largest"):
        return Path(which)
    sizes = sorted(
        (sum(1 for _ in path.open()) - 1, str(path), path) for path in tsv_paths
    )
    return sizes[len(sizes) // 2][2] if which == "median" else sizes[-1][2]


def profile_block(
    source: Path, ld_dir: Path, ancestry: str, work: Path, which: str, n_workers_hint: int = 1
) -> dict[str, Any]:
    """Complete one LD block for every Analysis of ``source``, split by stage."""
    from opengwasdb.completion import block as block_module
    from opengwasdb.completion import impute as impute_module
    from opengwasdb.completion.parallel import init_block_worker
    from opengwasdb.layouts.dense import complete as dense

    # A pool worker runs one BLAS thread; so does the block measured here.
    init_block_worker()
    panel = dense._enumerate_panel(ld_dir, ancestry)
    tsv_path = choose_block(panel.tsv_paths, which)
    n_snps = sum(1 for _ in tsv_path.open()) - 1
    work.mkdir(parents=True, exist_ok=True)
    task = dense._BlockTask(
        tsv_path=tsv_path,
        source_path=source,
        min_cor=DEFAULT_MIN_COR,
        thresh=DEFAULT_THRESH,
        checkpoint_path=work / "profile-block.npz",
        ceded=panel.ceded.get(tsv_path, frozenset()),
    )
    clock = StageClock()
    saved: list[tuple[Any, str, Any]] = []

    def patch(module: Any, name: str, stage: str) -> None:
        saved.append((module, name, getattr(module, name)))
        setattr(module, name, clock.wrap(stage, getattr(module, name)))

    # `run_block` looks these up as globals, so a patched module attribute is
    # what it calls. Each is a leaf except the per-Analysis kernel, whose own
    # remainder after its fits is the per-cell fill loop.
    patch(block_module, "load_block", "ld_panel_read")
    patch(block_module, "load_ld_eigenvectors", "ld_panel_read")
    patch(block_module, "complete_block_for_analysis", "fill_loop_and_cell_prep")
    patch(block_module, "scalar_n_se", "se_scale")
    patch(impute_module, "elastic_net_impute", "elastic_net_fit")
    patch(impute_module, "poly_rescale", "poly_rescale")
    patch(dense, "write_block_checkpoint", "checkpoint_write")
    make_reader = dense._make_reader

    def timed_make_reader(task_: Any) -> Any:
        factory = make_reader(task_)

        def timed_factory(*args: Any, **kwargs: Any) -> Any:
            with clock.stage("source_read"):
                return factory(*args, **kwargs)

        return timed_factory

    saved.append((dense, "_make_reader", make_reader))
    dense._make_reader = timed_make_reader
    try:
        with clock.stage("block_bookkeeping"):
            result = dense._run_block(task)
    finally:
        for module, name, original in reversed(saved):
            setattr(module, name, original)
    assert result is not None, f"{tsv_path} could not be loaded"
    checkpoint_bytes = task.checkpoint_path.stat().st_size
    n_fills = _count_fills(task.checkpoint_path)
    summary = clock.summary()
    return {
        "kind": "block",
        "source": str(source),
        "ld_panel": str(ld_dir),
        "ancestry": ancestry,
        "block": str(tsv_path),
        "block_variants": n_snps,
        "chosen_by": which,
        "n_analyses": _source_axis_size(source)[1],
        "fills": n_fills,
        "checkpoint_bytes": checkpoint_bytes,
        **summary,
    }


def _count_fills(checkpoint: Path) -> int:
    import numpy as np

    with np.load(checkpoint) as loaded:
        return int(loaded["f_alid"].shape[0]) if "f_alid" in loaded else -1


class _DiscardSlice(Exception):
    """Ends the staging block of a profile run, so nothing is published."""


#: What `bands` times, as (module attribute, stage). The merge's stages are leaves
#: of `_merge_checkpoint_fills`; the band write's are the three passes of
#: `_write_completed_bands` and, inside the statistic pass, the per-band steps.
_MERGE_STAGES = (
    ("_sorted_union_map", "merge_union_map"),
    ("panel_reference_eaf", "merge_reference_eaf"),
    ("_create_completed_zarr", "merge_create_planes"),
    ("_shard_checkpoint_fills_by_band", "merge_shard_fills_and_quality"),
)
_BAND_STAGES = (
    ("_seed_paired_bands", "band_seed_from_source"),
    ("_count_band_off_panel_missing", "band_counts"),
    ("_count_band_imputation_failed", "band_counts"),
    ("_apply_fill_shard_records", "band_apply_fills"),
    ("_write_judged_band", "band_judge_and_write"),
    ("_write_completed_statistic_bands", "band_statistic_pass_other"),
    ("_write_completed_eaf_bands", "band_eaf_pass"),
    ("_rewrite_completed_residual_se", "band_residual_se_rewrite"),
)


@contextmanager
def _patched(module: Any, stages: tuple[tuple[str, str], ...], clock: StageClock) -> Iterator[None]:
    saved = [(name, getattr(module, name)) for name, _ in stages]
    for (name, stage), (_, original) in zip(stages, saved, strict=True):
        setattr(module, name, clock.wrap(stage, original))
    try:
        yield
    finally:
        for name, original in saved:
            setattr(module, name, original)


def _blocks_covering(axis: Any, n_rows: int) -> list[Path]:
    """The LD blocks, in panel order, that start at or before the last slice row.

    The union axis is sorted by chromosome and position, so a prefix of its rows
    is the start of one chromosome and these blocks are the ones whose fills land
    in it. Fills a block places beyond the prefix are dropped by the merge.
    """
    last = axis.merged_variants[n_rows - 1]
    chosen: list[Path] = []
    for path in axis.tsv_paths:
        if path.parent.name != str(last.chromosome):
            if chosen:
                break
            continue
        if int(path.stem.split("-")[0]) <= last.position:
            chosen.append(path)
    return chosen


def profile_bands(
    source: Path, ld_dir: Path, ancestry: str, work: Path, n_bands: int, n_workers: int
) -> dict[str, Any]:
    """Run the merge and the band write over the first ``n_bands`` of the axis.

    The axis is the real one (its construction is timed as the first phase). The
    slice is a prefix of it: the same rows, the same source-row remap and the same
    real fills a full run would have for them, written through the same functions.
    """
    import dataclasses

    from opengwasdb.completion.checkpoint import checkpoint_blocks_dir
    from opengwasdb.completion.parallel import run_block_tasks
    from opengwasdb.layouts.dense import complete as dense
    from opengwasdb.store.open import OpenGWASDBStore, open_store

    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    clock = StageClock()
    manifest = open_store(source).manifest
    out: dict[str, Any] = {}
    try:
        with OpenGWASDBStore.staging(work / "slice.opengwasdb") as staged:
            with clock.stage("axis_construction_full"):
                axis = dense._build_union_axis(source, staged, ld_dir, ancestry, None)
            # The completed planes take the source's shard height; the merge
            # below returns the real one and the two are checked against each other.
            source_z = open_store(source).arrays(mode="r")["z"]
            shards = getattr(source_z, "shards", None) or source_z.chunks
            band_rows_hint = dense.completion_band_rows(int(shards[0]))
            blocks = _blocks_covering(axis, min(n_bands * band_rows_hint, axis.n_variants))
            slice_axis_rows = min(n_bands * band_rows_hint, axis.n_variants)
            checkpoint_dir = work / "checkpoints"
            blocks_dir = checkpoint_blocks_dir(checkpoint_dir)
            tasks = [
                dense._BlockTask(
                    tsv_path=path,
                    source_path=source,
                    min_cor=DEFAULT_MIN_COR,
                    thresh=DEFAULT_THRESH,
                    checkpoint_path=blocks_dir / f"{path.parent.name}_{path.stem}.npz",
                    ceded=axis.ceded.get(path, frozenset()),
                )
                for path in blocks
            ]
            with clock.stage("ld_block_phase_slice"):
                run_block_tasks(tasks, n_workers, dense._run_block, every=max(1, len(tasks) // 5))

            sliced = dataclasses.replace(
                axis,
                n_variants=slice_axis_rows,
                merged_variants=axis.merged_variants[:slice_axis_rows],
                on_panel=axis.on_panel[:slice_axis_rows],
                out_to_src=axis.out_to_src[:slice_axis_rows],
            )
            src_root = open_store(source).arrays(mode="r")
            with _patched(dense, _MERGE_STAGES, clock), _patched(dense, _BAND_STAGES, clock):
                with clock.stage("merge_other"):
                    encoding, _, effective_chunks, fill_dir = dense._merge_checkpoint_fills(
                        staged, sliced, blocks_dir, ld_dir, ancestry, manifest.encoding
                    )
                band_rows = dense.completion_band_rows(
                    dense._dense_shard_rows(staged, effective_chunks)
                )
                assert band_rows == band_rows_hint, (band_rows, band_rows_hint)
                with clock.stage("band_write_other"):
                    tallies = dense._write_completed_bands(
                        staged, src_root, sliced.out_to_src, sliced.on_panel, fill_dir,
                        effective_chunks, sliced.n_variants, sliced.n_analyses,
                        manifest.encoding, encoding, impute_mask=sliced.impute_mask,
                        name_row=lambda row: sliced.merged_variants[row].alid,
                    )
            n_bands_run = -(-slice_axis_rows // band_rows)
            out = {
                "kind": "bands",
                "source": str(source),
                "ld_panel": str(ld_dir),
                "ancestry": ancestry,
                "n_workers": n_workers,
                "union_variants": axis.n_variants,
                "n_analyses": axis.n_analyses,
                "band_rows": band_rows,
                "slice_rows": slice_axis_rows,
                "bands_run": n_bands_run,
                "bands_in_full_matrix": -(-axis.n_variants // band_rows),
                "blocks_run": len(blocks),
                "blocks_in_panel": len(axis.tsv_paths),
                "slice_imputed_cells": int(tallies.total_imputed),
                "slice_cells": slice_axis_rows * axis.n_analyses,
                **clock.summary(),
            }
            raise _DiscardSlice
    except _DiscardSlice:
        pass
    return out


def main(argv: list[str] | None = None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("block", "bands"):
        p = sub.add_parser(name)
        p.add_argument("source", type=Path)
        p.add_argument("ld_panel", type=Path)
        p.add_argument("--ancestry", default="EUR")
        p.add_argument("--work", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name == "block":
            p.add_argument("--block", default="median", help="median, largest, or a block .tsv")
        else:
            p.add_argument("--bands", type=int, default=2)
            p.add_argument("--n-workers", type=int, default=32)
    args = parser.parse_args(args_in)
    header = artifact_header(["profile_reference_completion.py", *args_in])
    if args.command == "block":
        body = profile_block(args.source, args.ld_panel, args.ancestry, args.work, args.block)
    else:
        body = profile_bands(
            args.source, args.ld_panel, args.ancestry, args.work, args.bands, args.n_workers
        )
    write_artifact(args.output, {**header, **body, "host_load_end": host_load()})
    shutil.rmtree(args.work / "profile-block.npz", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
