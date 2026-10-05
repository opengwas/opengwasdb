"""Every fork-pool build path at `n_workers=2`, after the parent used the read levers.

#244's fork-safety check (ADR 0056). The parent first encodes and decodes
multi-block chunks through the array seam, so Blosc's global-context thread
pool and whatever the codec pipeline keeps are live. Then it runs each forking
path, printing one JSON line as each finishes, so a hang names the path it hung
in. Run it under `timeout`: a hang is the failure being looked for.

  fixture_builds        the conformance suite's builds (benchmarks/zarr3_fixture_trees.py):
                        Dense VCF, Hybrid and Hybrid Reference Completion at n_workers=2
  dense_completion_n2   Dense Reference Completion (run_block_tasks pool)
  ragged_completion_n2  Ragged Reference Completion (run_block_tasks pool)
  gather_n2             the top-hit gather, called directly, on the Hybrid's Dense Component
  gather_real_n2        the top-hit gather on real row bands of `--store` (OGS-00009): three
                        `[1000, 1000]` chunks per worker read, the multi-chunk read a
                        one-chunk fixture never makes. Skipped without `--store`.

  --mode asis  the checkout's own configuration
  --mode pool  as committed, but FusedCodecPipeline with its default thread pool
               (codec_pipeline.max_workers unset): the negative control, which hangs
               in gather_real_n2 on zarr 3.4.0

The fixture paths cannot detect the hang on their own: every fixture array is one
chunk, and a single-chunk read never uses the pool.

#244 ran this with #243's `build_stores.py` building Dense VCF and Hybrid one
path at a time; `fixture_builds` now runs them through the conformance suite's
builder in one step.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import _zarr3_levers as levers
from benchmarks import zarr3_fixture_trees as trees
from benchmarks._zarr3_levers import say

#: Every top-hit gather's worker count, in call order.
GATHERS: list[int] = []


def warm_parent(out: Path) -> None:
    """Threaded Blosc encode and decode of 16 multi-block chunks through the seam."""
    import numcodecs.blosc
    import zarr
    import zarr.core.codec_pipeline as codec_pipeline

    from opengwasdb.store import arrays
    from opengwasdb.store.arrays import ArrayRole

    warm = out / "warm.zarr"
    root = arrays.open_group_for_write(warm, "w")
    data = np.random.default_rng(244).integers(-3000, 3000, size=(4000, 4000), dtype=np.int16)
    arrays.create_array(root, "z", ArrayRole.DENSE_STATISTIC_PLANE, data=data, hint=(1000, 1000))
    z = arrays.open_group(warm)["z"]
    if not np.array_equal(np.asarray(z[:]), data):
        raise SystemExit("the parent read back different values")
    say(
        step="parent_warm",
        use_threads=numcodecs.blosc.use_threads,
        threaded_here=numcodecs.blosc._get_use_threads(),
        chunk_bytes=1000 * 1000 * 2,
        pipeline=type(z._async_array.codec_pipeline).__name__,
        max_workers=zarr.config.get("codec_pipeline.max_workers", None),
        fused_pool_live=codec_pipeline._pool is not None,
    )


def count_gathers() -> None:
    """Record the worker count of every top-hit gather a path makes."""
    import opengwasdb.layouts.dense.top_hits as dense_top_hits

    original = dense_top_hits._gather_in_row_chunks

    def counted(root: Any, rows: Any, cols: Any, read: Any, n_workers: int) -> Any:
        GATHERS.append(int(n_workers))
        return original(root, rows, cols, read, n_workers)

    dense_top_hits._gather_in_row_chunks = counted


def dense_completion_n2(out: Path) -> None:
    import test_dense_completion as tdc

    from opengwasdb.build.observed import build_dense_observed_from_sources
    from opengwasdb.layouts.dense.complete import complete_dense_store

    d = out / "dense-completion-n2"
    d.mkdir()
    src = d / "associations.tsv"
    src.write_text("\n".join([tdc.SOURCE_HEADER, *tdc.SOURCE_ROWS]) + "\n", encoding="utf-8")
    observed = d / "obs.opengwasdb"
    build_dense_observed_from_sources(
        [src], observed, store_id="test", release_id="obs-v1", reference_assembly="GRCh38"
    )
    panel = tdc._make_ld_panel(d)
    kwargs = {"ancestry": "EUR", "min_cor": 0.0, "release_id": "comp-v1", "n_workers": 2}
    complete_dense_store(observed, d / "comp.opengwasdb", panel, **kwargs)


def ragged_completion_n2(out: Path) -> None:
    import test_ragged_completion as trc

    from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd
    from opengwasdb.layouts.ragged.complete import complete_ragged_store

    d = out / "ragged-completion-n2"
    d.mkdir()
    observed = d / "obs.opengwasdb"
    prefix = trc._make_besd_fixture(d)
    build_ragged_from_besd(prefix, observed, store_id="test", release_id="obs-v1", tissue="Blood")
    panel = trc._make_ld_panel(d, "1", 900_000, 1_300_000)
    kwargs = {"ancestry": "EUR", "cis_window_bp": 500_000, "min_cor": 0.0, "release_id": "comp-v1"}
    complete_ragged_store(observed, d / "comp.opengwasdb", panel, n_workers=2, **kwargs)


def gather_n2(out: Path) -> None:
    import opengwasdb.layouts.dense.top_hits as dense_top_hits
    from opengwasdb.model.manifest import StoreManifest
    from opengwasdb.store import arrays

    store = out / "builds" / "hybrid-n2" / "store.opengwasdb" / "dense"
    encoding = StoreManifest.load(store).encoding
    n = int(arrays.open_group(store / "data.zarr")["z"].shape[0])
    rows = np.arange(n, dtype=np.int64)
    cols = np.zeros(n, dtype=np.int64)
    parallel = dense_top_hits._collect_top_hit_eaf(store, rows, cols, encoding, 2)
    serial = dense_top_hits._collect_top_hit_eaf(store, rows, cols, encoding, 1)
    if (
        parallel is None
        or len(parallel) != n
        or not np.array_equal(parallel, serial, equal_nan=True)
    ):
        raise SystemExit("the parallel gather disagrees with the serial one")


def gather_real_n2(store: Path) -> None:
    """Each worker decodes a full-width row band: three `[1000, 1000]` chunks per plane."""
    import opengwasdb.layouts.dense.top_hits as dense_top_hits
    from opengwasdb.model.manifest import StoreManifest

    encoding = StoreManifest.load(store).encoding
    rng = np.random.default_rng(244)
    starts = (6_500_000, 6_501_000, 6_502_000, 6_503_000)
    rows = np.sort(np.concatenate([rng.integers(r0, r0 + 1000, 50) for r0 in starts]))
    cols = rng.integers(0, 2024, len(rows))
    for collect in (dense_top_hits._collect_top_hit_eaf, dense_top_hits._collect_top_hit_se):
        parallel = collect(store, rows, cols, encoding, 2)
        serial = collect(store, rows, cols, encoding, 1)
        if parallel is None or serial is None or np.isfinite(parallel).sum() <= 100:
            raise SystemExit("the gather must return real values to mean anything")
        if not np.array_equal(parallel, serial, equal_nan=True):
            raise SystemExit("the parallel gather disagrees with the serial one")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, default=Path("."), help="the checkout under test")
    ap.add_argument("--out", type=Path, required=True, help="a new scratch directory")
    ap.add_argument("--mode", choices=("asis", "pool"), default="asis")
    ap.add_argument("--store", type=Path, default=None, help="OGS-00009, for gather_real_n2")
    args = ap.parse_args()
    trees.use_checkout(args.repo)
    import numcodecs.blosc
    import zarr

    if args.mode == "pool":
        pipeline = levers.PIPELINES["fused"]
        zarr.config.set({**pipeline, "codec_pipeline.max_workers": None})
    args.out.mkdir(parents=True, exist_ok=False)
    warm_parent(args.out)
    count_gathers()
    steps = {
        "fixture_builds": lambda: trees.build_all(args.out / "builds"),
        "dense_completion_n2": lambda: dense_completion_n2(args.out),
        "ragged_completion_n2": lambda: ragged_completion_n2(args.out),
        "gather_n2": lambda: gather_n2(args.out),
    }
    if args.store is not None:
        steps["gather_real_n2"] = lambda: gather_real_n2(args.store)
    for name, run in steps.items():
        before = len(GATHERS)
        t0 = time.perf_counter()
        run()
        say(
            step=name,
            ok=True,
            s=round(time.perf_counter() - t0, 2),
            gathers=GATHERS[before:],
            use_threads=numcodecs.blosc.use_threads,
        )
    say(step="all", ok=True)


if __name__ == "__main__":
    main()
