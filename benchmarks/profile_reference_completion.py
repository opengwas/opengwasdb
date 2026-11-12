#!/usr/bin/env python3
"""Attribute a Reference Completion's wall time to what it does (#309).

A full completion of OGS-00009 took 14 h 05 m, and its log said only how many LD
blocks had finished. This harness measures the two places the time can hide, on
real input, without running a completion:

``block``
    One real LD block completed for every Analysis of a Store, with its time
    split into the LD-panel read, the source-plane read, the elastic-net fits,
    the polynomial rescale, the per-cell fill loop, the fill-row bookkeeping and
    the checkpoint write. It also counts the block's distinct observed sets.

``bands``
    The checkpoint merge and the band write over a prefix of the completed
    matrix's row bands, with real fills: it completes the LD blocks that cover the
    prefix, shards their fills, and writes those bands only. The band write is
    split into reading source planes, computing, and writing (#321 writes each
    band in one pass, which reads the source once and codes every plane). The
    artifact scales each stage to the full run and sets the result beside the
    figures OGS-00009's own run recorded.

    The stages are read in this process, so the band write runs here, on one
    worker, unless ``--write-workers`` asks for more: then the bands run on the
    write pool and only the band write's wall time is known, as
    ``band_write_pool``. The phases are called directly, below the source
    format check `complete_dense_store` makes, so a 0.2.0 Store such as
    OGS-00009 is profiled as it is; nothing the profile writes is published.

Stages are *exclusive*: a stage entered inside another pauses it, so the stages
add up to the wall time. The profile checks that against a separate reading of
the same call's wall time and fails when they differ by more than 5%, so a stage
left out of the table is an error, not a smaller number. Every artifact records
the revision (and whether the tree was clean), its argv, and the host's load
sampled every 30 s through the run.

  pixi run -e dev python benchmarks/profile_reference_completion.py block \
      SOURCE LD_PANEL --work NEW_SCRATCH_DIR --output block.json
  pixi run -e dev python benchmarks/profile_reference_completion.py bands \
      SOURCE LD_PANEL --work NEW_SCRATCH_DIR --bands 2 --output bands.json \
      --record /data/opengwasdb/stores/OGS-00009/records/complete.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks._artifact import LoadSampler, artifact_header, host_load, write_artifact
from opengwasdb.encoding.timing import PhaseTimer

DEFAULT_MIN_COR = 0.7
DEFAULT_THRESH = 0.9
#: The accounted stages may differ from the call's wall time by this share.
WALL_TOLERANCE = 0.05
#: Seconds of setup and teardown a call may leave outside every stage, whatever its length.
WALL_SLACK_SECONDS = 0.05


class StageClock:
    """Exclusive seconds per stage, over stages that nest.

    `PhaseTimer` partitions work and asks its callers not to nest. A block's fits
    run inside the per-Analysis kernel which runs inside the block runner, and the
    answer wanted is how long each takes *itself*. Entering a stage stops the
    clock on the stage it interrupts and entering it again resumes that one
    afterwards, so `total()` is the wall time of the outermost stage.

    `PhaseTimer` has no public way to add seconds it did not time itself, so the
    stack writes `timer.seconds` directly; `encoding/timing.py` is #308's.
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

    def add(self, name: str, seconds: float) -> None:
        """Move `seconds` into stage `name` from the stage now running, which a
        callee timed on its own."""
        self._charge(self._now())
        if self._stack:
            self.timer.seconds[self._stack[-1]] -= seconds
        self.timer.seconds[name] = self.timer.seconds.get(name, 0.0) + seconds

    def wrap(self, name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def timed(*args: Any, **kwargs: Any) -> Any:
            with self.stage(name):
                return function(*args, **kwargs)

        return timed

    def summary(self) -> dict[str, Any]:
        return {
            "accounted_seconds": self.timer.total(),
            "stages": [
                {
                    "stage": name, "seconds": seconds, "share": share,
                    "calls": self.calls.get(name, 0),
                }
                for name, seconds, share in self.timer.report()
            ],
        }


def check_accounts_for_wall(
    accounted: float, wall: float, tolerance: float = WALL_TOLERANCE
) -> None:
    """Fail when the stages' sum is not the call's wall time, within `tolerance`.

    The stages come from one clock and their shares are fractions of their own
    sum, so they always add up to 1. Only a second reading of the call's wall time
    can say whether a stage was left out, or time was charged twice.
    """
    if wall <= 0 or abs(accounted - wall) > tolerance * wall + WALL_SLACK_SECONDS:
        raise RuntimeError(
            f"the stages account for {accounted:.3f} s of a {wall:.3f} s call "
            f"(tolerance {tolerance:.0%} + {WALL_SLACK_SECONDS} s): "
            "a stage is missing from the table or counted twice"
        )


def choose_block(tsv_paths: list[Path], which: str) -> Path:
    """The block to profile: ``median`` or ``largest`` by variant count, or a path.

    The block size sets the fit cost (eigenvectors grow with the block), so the
    artifact names which one it took.
    """
    if which not in ("median", "largest"):
        return Path(which)
    sizes = sorted((_n_variants(path), str(path), path) for path in tsv_paths)
    return sizes[len(sizes) // 2][2] if which == "median" else sizes[-1][2]


def _n_variants(tsv_path: Path) -> int:
    with tsv_path.open() as handle:
        return sum(1 for _ in handle) - 1


def require_new_directory(path: Path) -> None:
    """Refuse an existing `--work`, so the profile never deletes what it did not make."""
    if path.exists():
        raise SystemExit(f"{path}: already exists; refusing to use or delete it")
    path.mkdir(parents=True)


def _distinct_observed_sets(read: Callable[[int], Any], n_analyses: int) -> dict[str, int]:
    """How many different sets of observed variants the block's Analyses have."""
    import numpy as np

    sets: set[bytes] = set()
    observed = 0
    for analysis in range(n_analyses):
        mask = np.isfinite(read(analysis)[0])
        sets.add(np.packbits(mask).tobytes())
        observed += int(mask.sum())
    return {"distinct_observed_sets": len(sets), "mean_observed_variants": observed // n_analyses}


@contextmanager
def _patched(stages: list[tuple[Any, str, str]], clock: StageClock) -> Iterator[None]:
    """Wrap each ``(owner, attribute)`` so a call is charged to its stage."""
    saved = [(owner, name, getattr(owner, name)) for owner, name, _ in stages]
    for (owner, name, stage), (_, _, original) in zip(stages, saved, strict=True):
        setattr(owner, name, clock.wrap(stage, original))
    try:
        yield
    finally:
        for owner, name, original in reversed(saved):
            setattr(owner, name, original)


def profile_block(
    source: Path, ld_dir: Path, ancestry: str, work: Path, which: str,
    clock: Callable[[], float] = time.perf_counter,
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
    require_new_directory(work)
    task = dense._BlockTask(
        tsv_path=tsv_path, source_path=source, min_cor=DEFAULT_MIN_COR, thresh=DEFAULT_THRESH,
        checkpoint_path=work / "profile-block.npz", placement=panel.blocks[tsv_path],
        ceded=panel.ceded.get(tsv_path, frozenset()),
    )
    stages = StageClock(clock)
    captured: dict[str, Any] = {}
    make_reader = dense._make_reader

    def timed_make_reader(task_: Any) -> Any:
        factory = make_reader(task_)

        def timed_factory(*args: Any, **kwargs: Any) -> Any:
            with stages.stage("source_read"):
                analyses, read = factory(*args, **kwargs)
            captured["read"], captured["n"] = read, len(analyses)
            return analyses, read

        return timed_factory

    # `run_block` looks these up as globals, so a patched module attribute is what
    # it calls. The per-Analysis kernel's own remainder after its fits is the cell
    # prep and the per-cell fill loop.
    table = [
        (block_module, "load_block", "ld_panel_read"),
        (block_module, "load_ld_eigenvectors", "ld_panel_read"),
        (block_module, "complete_block_for_analysis", "kernel_cell_prep_and_fill_loop"),
        (block_module, "scalar_n_se", "se_scale"),
        (block_module, "_owned_rows", "owned_rows"),
        (impute_module, "elastic_net_impute", "elastic_net_fit"),
        (impute_module, "poly_rescale", "poly_rescale"),
        (dense, "write_block_checkpoint", "checkpoint_write"),
    ]
    saved_reader, dense._make_reader = make_reader, timed_make_reader
    started = clock()
    try:
        with _patched(table, stages), stages.stage("block_other"):
            result = dense._run_block(task)
    finally:
        dense._make_reader = saved_reader
    wall = clock() - started
    assert result is not None, f"{tsv_path} could not be loaded"
    accounted = stages.summary()
    check_accounts_for_wall(accounted["accounted_seconds"], wall)
    fills = _checkpoint_fills(task.checkpoint_path)
    checkpoint_bytes = task.checkpoint_path.stat().st_size
    task.checkpoint_path.unlink()
    work.rmdir()
    fits = next(s for s in accounted["stages"] if s["stage"] == "elastic_net_fit")
    return {
        "kind": "block",
        "source": str(source), "ld_panel": str(ld_dir), "ancestry": ancestry,
        "block": str(tsv_path), "chosen_by": which,
        "block_variants": _n_variants(tsv_path),
        "n_analyses": captured["n"],
        "fills": fills, "checkpoint_bytes": checkpoint_bytes,
        "wall_seconds": wall,
        "seconds_per_fit": fits["seconds"] / max(fits["calls"], 1),
        **_distinct_observed_sets(captured["read"], captured["n"]),
        **accounted,
    }


def _checkpoint_fills(checkpoint: Path) -> int:
    import numpy as np

    with np.load(checkpoint) as loaded:
        return int(loaded["f_alid"].shape[0])


def _timed_generator(clock: StageClock, stage: str, original: Callable[..., Any]) -> Any:
    """Charge only the time spent inside the generator's own `next`, to ``stage``."""

    def wrapped(*args: Any, **kwargs: Any) -> Iterator[Any]:
        iterator = original(*args, **kwargs)
        while True:
            with clock.stage(stage):
                try:
                    item = next(iterator)
                except StopIteration:
                    return
            yield item

    return wrapped


class _DiscardSlice(Exception):
    """Ends the staging block of a profile run, so nothing is published."""


def _band_write_table(dense: Any) -> list[tuple[Any, str, str]]:
    """The band write's stages, as reading source planes, computing, and writing.

    One pass per band: its source read yields z, SE and the EAF the SE was
    decoded against, and its encode codes z, EAF, and SE against the decoded
    EAF (`_decoded_eaf` is the SE encode's own decode; the source's is the
    read's). The side arrays written once, after every band, are their own stage.
    """
    from opengwasdb.encoding import StoreCodec, association_codec

    return [
        (dense, "_seed_band", "band_compute_seed"),
        (dense, "_seed_paired_bands", "band_read_source"),
        (dense, "_count_band_off_panel_missing", "band_compute_counts"),
        (dense, "_count_band_imputation_failed", "band_compute_counts"),
        (dense, "_apply_fill_shard_records", "band_compute_apply_fills"),
        (association_codec, "storable", "band_compute_judge"),
        (StoreCodec, "encode_z", "band_compute_encode_z"),
        (StoreCodec, "encode_eaf", "band_compute_encode_eaf"),
        (association_codec.AssociationCodec, "_decoded_eaf", "band_compute_encode_se"),
        (StoreCodec, "encode_se", "band_compute_encode_se"),
        (dense, "_encode_band", "band_compute_encode_other"),
        (dense, "_write_band", "band_write"),
        (dense, "_carried_eaf_baseline", "band_side_arrays"),
        (dense, "_source_se_coefficients", "band_side_arrays"),
        (dense, "_write_side_arrays", "band_side_arrays"),
    ]


_MERGE_TABLE = (
    ("sorted_union_map", "merge_union_map"),
    ("panel_reference_eaf", "merge_reference_eaf"),
    ("_create_completed_zarr", "merge_create_planes"),
    ("shard_checkpoint_fills", "merge_shard_fills_and_quality"),
)


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
    source: Path, ld_dir: Path, ancestry: str, work: Path, n_bands: int, n_workers: int,
    clock: Callable[[], float] = time.perf_counter, write_workers: int = 1,
) -> dict[str, Any]:
    """Run the merge and the band write over the first ``n_bands`` of the axis.

    The axis is the real one (its construction is timed as the first phase). The
    slice is a prefix of it: the same rows, the same source-row remap and the same
    real fills a full run would have for them, written through the same functions.
    """
    from opengwasdb.layouts.dense import complete as dense
    from opengwasdb.layouts.dense.constants import dense_layout_records
    from opengwasdb.store.open import OpenGWASDBStore, open_store

    require_new_directory(work)
    stages = StageClock(clock)
    manifest = open_store(source).manifest
    out: dict[str, Any] = {}
    started = clock()
    try:
        with OpenGWASDBStore.staging(work / "slice.opengwasdb") as staged:
            with stages.stage("axis_construction_full"):
                axis = dense._build_union_axis(source, staged, ld_dir, ancestry, None)
            # The completed planes' shard height, as the full run would write it.
            layout = dense_layout_records((axis.n_variants, axis.n_analyses))
            band_rows = dense.completion_band_rows(int(layout["shard_shape"][0]))
            slice_rows = min(n_bands * band_rows, axis.n_variants)
            blocks = _blocks_covering(axis, slice_rows)
            # The real scheduler, over the blocks the slice needs.
            with stages.stage("ld_block_phase_slice"):
                blocks_dir = dense._complete_pending_blocks(
                    dataclasses.replace(axis, tsv_paths=blocks), source, work / "checkpoints",
                    min_cor=DEFAULT_MIN_COR, thresh=DEFAULT_THRESH, n_workers=n_workers,
                )
            sliced = dataclasses.replace(
                axis,
                n_variants=slice_rows,
                merged_variants=axis.merged_variants[:slice_rows],
                on_panel=axis.on_panel[:slice_rows],
                out_to_src=axis.out_to_src[:slice_rows],
            )
            src_root = open_store(source).arrays(mode="r")
            tallies = _merge_and_write(dense, stages, staged, sliced, src_root, blocks_dir,
                                       ld_dir, ancestry, manifest.encoding, band_rows,
                                       write_workers)
            out = {
                "kind": "bands", "source": str(source), "ld_panel": str(ld_dir),
                "ancestry": ancestry, "n_workers": n_workers, "write_workers": write_workers,
                "union_variants": axis.n_variants, "n_analyses": axis.n_analyses,
                "band_rows": band_rows, "slice_rows": slice_rows,
                "bands_run": -(-slice_rows // band_rows),
                "bands_in_full_matrix": -(-axis.n_variants // band_rows),
                "blocks_run": len(blocks), "blocks_in_panel": len(axis.tsv_paths),
                "slice_imputed_cells": int(tallies.total_imputed),
                "slice_cells": slice_rows * axis.n_analyses,
            }
            raise _DiscardSlice
    except _DiscardSlice:
        pass
    finally:
        _remove_made_tree(work)
    wall = clock() - started
    accounted = stages.summary()
    check_accounts_for_wall(accounted["accounted_seconds"], wall)
    return {**out, "wall_seconds": wall, **accounted}


def _remove_made_tree(work: Path) -> None:
    """Delete the checkpoints and the discarded slice this profile made under `work`."""
    import shutil

    shutil.rmtree(work)


def _merge_and_write(
    dense: Any, stages: StageClock, staged: Any, sliced: Any, src_root: Any, blocks_dir: Path,
    ld_dir: Path, ancestry: str, source_encoding: Any, band_rows: int, write_workers: int = 1,
) -> Any:
    """The merge, then the band write, split into read, compute and write.

    The merge runs on one worker, as it did when it was serial, so its stages
    are this process's. On more than one write worker the bands run in the
    pool's processes, where this clock cannot follow them: the band write is
    then one stage, ``band_write_pool``.
    """
    merge = [(dense, name, stage) for name, stage in _MERGE_TABLE]
    band = _band_write_table(dense) if write_workers <= 1 else []
    original_fills = dense.iter_fill_records
    dense.iter_fill_records = _timed_generator(stages, "band_read_fills", original_fills)
    try:
        with _patched(merge, stages), _patched(band, stages):
            with stages.stage("merge_other"):
                encoding, _, effective_chunks, fill_dir = dense._merge_checkpoint_fills(
                    staged, sliced, blocks_dir, ld_dir, ancestry, source_encoding
                )
            assert dense.completion_band_rows(
                dense._dense_shard_rows(staged, effective_chunks)
            ) == band_rows
            outer = "band_write_other" if write_workers <= 1 else "band_write_pool"
            with stages.stage(outer):
                return dense._write_completed_bands(
                    staged, src_root, sliced.out_to_src, sliced.on_panel, fill_dir,
                    effective_chunks, sliced.n_variants, sliced.n_analyses, source_encoding,
                    encoding, impute_mask=sliced.impute_mask,
                    name_row=lambda row: sliced.merged_variants[row].alid,
                    n_workers=write_workers,
                )
    finally:
        dense.iter_fill_records = original_fills


# What a slice's stage seconds scale by in a full run: per band, per row, or not at all.
_SCALE_BY_BANDS = ("band_",)
_SCALE_BY_ROWS = ("merge_shard_fills_and_quality", "merge_reference_eaf", "merge_union_map")


def scale_to_full_run(summary: dict[str, Any]) -> dict[str, Any]:
    """Each stage's slice seconds scaled to the full matrix, with the rule used.

    Band-write stages scale with the bands run (a pass over each band costs the
    same); the merge's fill sharding and reference-EAF read scale with rows; axis
    construction already ran on the full axis. LD-block time depends on block size
    and is left unscaled here; the recorded run's figure is the evidence for it.
    A band write on the pool (``band_write_pool``) scales the same way, which
    holds while the slice has at least as many bands as the pool has workers.
    """
    bands = summary["bands_in_full_matrix"] / summary["bands_run"]
    rows = summary["union_variants"] / summary["slice_rows"]
    scaled: dict[str, Any] = {}
    for entry in summary["stages"]:
        name, seconds = entry["stage"], entry["seconds"]
        if name.startswith(_SCALE_BY_BANDS):
            scaled[name] = {"factor": bands, "rule": "bands", "seconds": seconds * bands}
        elif name in _SCALE_BY_ROWS:
            scaled[name] = {"factor": rows, "rule": "rows", "seconds": seconds * rows}
    band_write = sum(
        v["seconds"] for k, v in scaled.items() if k.startswith(_SCALE_BY_BANDS)
    )
    merge = sum(v["seconds"] for k, v in scaled.items() if k.startswith("merge_"))
    return {"stages": scaled, "band_write_seconds": band_write, "merge_seconds": merge}


_STAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ")


def recorded_full_run(record: Path) -> dict[str, Any]:
    """The full run's own figures, read from a step record's stderr.

    Times in the log are the node's local time; only differences are used. The
    block phase ends no earlier than its last per-block warning, so its length
    is a lower bound and the post-block work (merge, band write, SE rewrite) an
    upper bound.
    """
    data = json.loads(record.read_text())
    stamps = [
        (datetime.fromisoformat(m.group(1)), line)
        for line in data["stderr"].splitlines()
        if (m := _STAMP.match(line))
    ]
    first = stamps[0][0]
    last_block = max(t for t, line in stamps if "ld_panel: Block " in line)
    se_start = next(t for t, line in stamps if "SE rewrite count: start" in line)
    se_end = next(t for t, line in stamps if "SE rewrite: done in" in line)
    return {
        "record": str(record),
        "elapsed_seconds": data["elapsed_seconds"],
        "blocks_at_least_seconds": (last_block - first).total_seconds(),
        "post_block_at_most_seconds": (se_end - last_block).total_seconds(),
        "se_rewrite_seconds": (se_end - se_start).total_seconds(),
    }


def main(argv: list[str] | None = None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("block", "bands"):
        p = sub.add_parser(name)
        p.add_argument("source", type=Path)
        p.add_argument("ld_panel", type=Path)
        p.add_argument("--ancestry", default="EUR")
        p.add_argument("--work", type=Path, required=True, help="a directory that does not exist")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--allow-dirty", action="store_true")
        p.add_argument("--load-period", type=float, default=30.0)
        if name == "block":
            p.add_argument("--block", default="median", help="median, largest, or a block .tsv")
        else:
            p.add_argument("--bands", type=int, default=2)
            p.add_argument("--n-workers", type=int, default=32)
            p.add_argument("--write-workers", type=int, default=1,
                           help="workers for the band write; above 1 it is timed as one stage")
            p.add_argument("--record", type=Path, help="the full run's step record to compare")
    args = parser.parse_args(args_in)
    header = artifact_header("benchmarks/profile_reference_completion.py",
                             ["profile_reference_completion.py", *args_in])
    if not header["worktree_clean"] and not args.allow_dirty:
        raise SystemExit(
            "the tree has uncommitted changes: the artifact would name code it did not run"
        )
    with LoadSampler(args.load_period) as load:
        if args.command == "block":
            body = profile_block(args.source, args.ld_panel, args.ancestry, args.work, args.block)
        else:
            body = profile_bands(
                args.source, args.ld_panel, args.ancestry, args.work, args.bands, args.n_workers,
                write_workers=args.write_workers,
            )
            body["scaled_to_full_run"] = scale_to_full_run(body)
            if args.record is not None:
                body["recorded_full_run"] = recorded_full_run(args.record)
    write_artifact(
        args.output,
        {**header, **body, "host_load_end": host_load(), "host_load_samples": load.samples},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
