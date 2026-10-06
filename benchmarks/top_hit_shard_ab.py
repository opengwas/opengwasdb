"""Interleaved A/B of the top-hit index: sharded against effectively unsharded (#246).

The top-hit index is six flat columns per significance tier, read together by a
Top-Hit Query. Sharding them adds a codec hop per read (ADR 0056); a query makes
six reads. #246 measures how much that costs, on OGS-00009, by comparing two
0.2.0 releases that differ in exactly one thing:

* `v3-c64` — the conversion with the seam's default, 64 inner chunks per
  top-hit shard;
* `v3-c64-topshard1` — the same conversion with `--top-hit-shard-chunks 1`, one
  inner chunk per shard, "effectively unsharded".

It is not an absolute harness run. A conversion was running in the same window,
so the honest measurement is a paired one: the two sides alternate round by
round (`A, B` then `B, A`), every sample is kept, and the artifact records the
top-hit layout and footprint of each store, the once-resolved selection, the
round order, the effective reader configuration, and a per-array result digest
compared between the sides every round -- a sharding change may not change an
answer.

The child is this same file (`--child`); it prints one `SHARD_AB_RESULT <json>`
line. Run the parent with the heavy-job lock; it queries a 33 GB store.

Usage:

    pixi run -e dev python benchmarks/top_hit_shard_ab.py \\
        --config sharded=/data/opengwasdb/work/epic240/245/OGS-00009-v3-c64 \\
        --config unsharded=/data/opengwasdb/work/epic240/246/OGS-00009-v3-c64-topshard1 \\
        --rounds 5 --reps 25 \\
        --output docs/benchmark-output/opengwasdb_246_shapes/opengwasdb_top_hit_shard_ab.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from benchmarks import _query_ab as ab
from benchmarks._artifact import provenance, tree_fingerprint, write_artifact
from benchmarks.benchmark_store_comparison import footprint

#: The shape under test. A Top-Hit Query is the only reader of the top-hit
#: index, so nothing else needs timing.
SHAPES = ("tophits",)

#: The top-hit tier whose layout distinguishes the two sides; every tier is laid
#: out the same way, so naming one is enough to tell them apart.
TIER = "top_hits/p_5e_08/z"

#: A Top-Hit Query makes six reads (ADR 0056), so the sharding cost per read is
#: the median difference over six.
READS_PER_TOP_HIT_QUERY = 6


def _store_layout(store: Path) -> dict[str, Any]:
    """The top-hit array's inner chunk and shard, read back from the store."""
    from opengwasdb.store.arrays import open_group

    array = open_group(store / "data.zarr", "r")[TIER]
    shards = getattr(array, "shards", None)
    return {
        "array": TIER,
        "chunk_shape": [int(size) for size in array.chunks],
        "shard_shape": None if shards is None else [int(size) for size in shards],
        "dtype": str(array.dtype),
    }


def _effective_reader(store: Path) -> dict[str, Any]:
    """The reader configuration in force, read back from the process (#244, #253)."""
    import numcodecs
    import zarr

    from opengwasdb.store.arrays import open_group

    plane = open_group(store / "data.zarr", "r")["z"]
    return {
        "use_threads": bool(numcodecs.blosc.use_threads),
        "pipeline": type(plane._async_array.codec_pipeline).__name__,
        "max_workers": zarr.config.get("codec_pipeline.max_workers", None),
    }


def _child_main(args: argparse.Namespace) -> int:
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    out: dict[str, Any] = {
        "store": str(args.store),
        "shapes": ab.time_shapes(args.store, selection, args.shapes, args.reps),
    }
    print("SHARD_AB_RESULT " + json.dumps(out), flush=True)
    return 0


def _parse_config(text: str) -> tuple[str, Path]:
    label, separator, path = text.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(f"--config wants LABEL=PATH, got {text!r}")
    return label, Path(path)


def _resolve_selection(store: Path, path: Path) -> dict[str, Any]:
    """The #242 harness's once-resolved selection, saved for the children."""
    from benchmarks.benchmark_store_comparison import StoreSpec, _resolve_selection

    selection = _resolve_selection(StoreSpec(label="ab", path=store))
    path.write_text(json.dumps(selection, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return selection


def _child_command(args: argparse.Namespace, store: Path, selection: Path) -> list[str]:
    return ab.child_argv(store, selection, args.shapes, args.reps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--selection", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--store", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--config",
        action="append",
        type=_parse_config,
        default=[],
        metavar="LABEL=PATH",
        help="one side of the A/B; exactly two, the first the reference",
    )
    parser.add_argument(
        "--shapes", nargs="+", default=list(SHAPES), help="which #242 shapes to time"
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=None,
        help="reuse an already-resolved selection instead of deriving it",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.child:
        if args.store is None or args.selection is None:
            raise SystemExit("--child needs --store and --selection")
        return _child_main(args)
    if args.output is None:
        raise SystemExit("--output is required")
    if len(args.config) != 2:
        raise SystemExit(f"exactly two --config sides are needed, got {len(args.config)}")
    labels = [label for label, _ in args.config]
    if len(set(labels)) != 2:
        raise SystemExit(f"the two --config labels must differ, got {labels}")
    stores = {label: path for label, path in args.config}
    missing = [str(path) for path in stores.values() if not (path / "manifest.json").is_file()]
    if missing:
        raise SystemExit(f"not a Store Release: {missing[0]}")

    workdir = Path(tempfile.mkdtemp(prefix="top_hit_shard_ab_"))
    selection_path = workdir / "selection.json"
    if args.selection_json is not None:
        selection = json.loads(args.selection_json.read_text(encoding="utf-8"))
        selection_path.write_text(
            json.dumps(selection, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    else:
        selection = _resolve_selection(stores[labels[0]], selection_path)

    layouts = {label: _store_layout(path) for label, path in stores.items()}
    if len({json.dumps(layout, sort_keys=True) for layout in layouts.values()}) != 2:
        raise SystemExit(
            f"the two sides have the same top-hit layout {layouts}; there is nothing to compare"
        )
    readers = {label: _effective_reader(path) for label, path in stores.items()}
    # The file count is half the decision: one inner chunk per shard makes the
    # top-hit index a file per 16,384 hits. The walker is the #242 harness's, so
    # these totals are the same measurement the shape comparison publishes.
    footprints = {label: footprint(path) for label, path in stores.items()}

    script = Path(__file__).resolve()

    def run_side(label: str, _index: int) -> dict[str, Any]:
        return ab.run_child(
            script, _child_command(args, stores[label], selection_path), "SHARD_AB_RESULT"
        )

    samples, digests, rounds = ab.interleave(labels, run_side, args.rounds)
    side_medians = ab.medians(samples)
    reference, other = labels
    ratio = {
        name: round(side_medians[reference][name] / side_medians[other][name], 4)
        for name in samples[reference]
    }
    per_read = {
        name: round(
            (side_medians[reference][name] - side_medians[other][name]) / READS_PER_TOP_HIT_QUERY,
            4,
        )
        for name in samples[reference]
    }
    differing = ab.differing_shapes(labels, digests)
    artifact = {
        "harness": "top_hit_shard_ab",
        "reference_side": reference,
        "configs": {
            label: {
                "path": str(stores[label]),
                "top_hit_layout": layouts[label],
                "effective_reader": readers[label],
                "footprint": footprints[label],
            }
            for label in labels
        },
        "sides": [
            {
                "label": label,
                "revision": provenance()["commit"],
                "opengwasdb_fingerprint": tree_fingerprint(script.parents[1]),
            }
            for label in labels
        ],
        **ab.measurement_block(selection, args.shapes, args.reps, args.rounds, samples, rounds),
        "medians_ms": side_medians,
        "ratio_reference_over_other": ratio,
        "per_read_ms": per_read,
        "identity": ab.identity_block(differing),
        "environment": ab.environment_block(),
        **provenance(),
    }
    write_artifact(args.output, artifact)
    return ab.identity_verdict(differing)


if __name__ == "__main__":
    raise SystemExit(main())
