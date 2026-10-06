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
top-hit layout of each store, the once-resolved selection, the round order, the
effective reader configuration, and a per-array result digest compared between
the sides every round -- a sharding change may not change an answer.

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
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks._artifact import provenance, tree_fingerprint, write_artifact

# The digest contract is #253's; one definition keeps two A/B artifacts
# comparable about what "identical results" means.
from benchmarks.eaf_read_once_ab import _digest

#: The shape under test. A Top-Hit Query is the only reader of the top-hit
#: index, so nothing else needs timing.
SHAPES = ("tophits",)


def _store_layout(store: Path, tier: str = "top_hits/p_5e_08/z") -> dict[str, Any]:
    """The top-hit array's inner chunk and shard, read back from the store."""
    from opengwasdb.store.arrays import open_group

    root = open_group(store / "data.zarr", "r")
    array = root[tier]
    shards = getattr(array, "shards", None)
    return {
        "array": tier,
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
    from benchmarks import _query_shapes
    from opengwasdb.query import query_store

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    region = (
        selection["region"]["chrom"],
        int(selection["region"]["start"]),
        int(selection["region"]["end"]),
    )
    out: dict[str, Any] = {"store": str(args.store), "shapes": {}}
    with query_store(args.store) as query:
        patterns = _query_shapes.common_query_patterns(
            query,
            exposure=selection["exposure_analysis_id"],
            phewas_alid=selection["phewas_alid"],
            region=region,
            random_alids=selection["random_alids"],
            random_analyses=selection["random_analyses"],
        )
        for name in args.shapes:
            fn = patterns[name]
            warm = fn()
            digest = _digest(warm)
            n_rows = len(warm["z"])
            samples: list[float] = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                result = fn()
                samples.append(round((time.perf_counter() - t0) * 1000, 4))
                if len(result["z"]) != n_rows:
                    raise SystemExit(
                        f"{name}: returned {len(result['z'])} rows after {n_rows}; "
                        "a shape whose size changes cannot be timed"
                    )
            out["shapes"][name] = {"samples_ms": samples, "digest": digest, "n_rows": n_rows}
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
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--store",
        str(store),
        "--selection",
        str(selection),
        "--shapes",
        *args.shapes,
        "--reps",
        str(args.reps),
    ]


def _run_child(args: argparse.Namespace, store: Path, selection: Path) -> dict[str, Any]:
    proc = subprocess.run(_child_command(args, store, selection), capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"A/B child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    lines = [line for line in proc.stdout.splitlines() if line.startswith("SHARD_AB_RESULT ")]
    if len(lines) != 1:
        raise SystemExit(f"A/B child printed {len(lines)} result lines:\n{proc.stdout[-4000:]}")
    return json.loads(lines[0][len("SHARD_AB_RESULT ") :])


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 3)


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
    readers = {label: _effective_reader(path) for label, path in stores.items()}
    if len({json.dumps(layout, sort_keys=True) for layout in layouts.values()}) != 2:
        raise SystemExit(
            f"the two sides have the same top-hit layout {layouts}; there is nothing to compare"
        )

    samples: dict[str, dict[str, list[float]]] = {label: defaultdict(list) for label in labels}
    digests: dict[str, dict[str, dict[str, str]]] = {label: {} for label in labels}
    rounds: list[dict[str, Any]] = []
    for index in range(args.rounds):
        order = list(labels) if index % 2 == 0 else list(reversed(labels))
        record: dict[str, Any] = {"round": index, "order": order, "results": {}}
        for label in order:
            child = _run_child(args, stores[label], selection_path)
            record["results"][label] = child
            for name, rec in child["shapes"].items():
                samples[label][name].extend(rec["samples_ms"])
                digests[label].setdefault(name, {})[str(index)] = rec["digest"]
            print(
                f"round {index} {label}: "
                f"{json.dumps({n: rec['n_rows'] for n, rec in child['shapes'].items()})}",
                flush=True,
            )
        rounds.append(record)

    differing = sorted(
        name for name in samples[labels[0]] if digests[labels[0]][name] != digests[labels[1]][name]
    )
    medians = {
        label: {name: _median(values) for name, values in samples[label].items()}
        for label in labels
    }
    ratio = {
        name: round(medians[labels[0]][name] / medians[labels[1]][name], 4)
        for name in samples[labels[0]]
    }
    # A Top-Hit Query makes six reads (ADR 0056), so the sharding cost per read
    # is the median difference over six.
    per_read = {
        name: round((medians[labels[0]][name] - medians[labels[1]][name]) / 6, 4)
        for name in samples[labels[0]]
    }
    artifact = {
        "harness": "top_hit_shard_ab",
        "configs": {
            label: {
                "path": str(stores[label]),
                "top_hit_layout": layouts[label],
                "effective_reader": readers[label],
            }
            for label in labels
        },
        "sides": [
            {
                "label": label,
                "revision": provenance()["commit"],
                "opengwasdb_fingerprint": tree_fingerprint(Path(__file__).resolve().parents[1]),
            }
            for label in labels
        ],
        "selection": selection,
        "shapes": list(args.shapes),
        "reps": args.reps,
        "rounds_requested": args.rounds,
        "round_order": [record["order"] for record in rounds],
        "rounds": rounds,
        "samples_ms": {
            label: {name: values for name, values in samples[label].items()} for label in labels
        },
        "medians_ms": medians,
        "ratio_reference_over_other": ratio,
        "per_read_ms": per_read,
        "identity": {
            "identical": not differing,
            "differing_shapes": differing,
            "note": "sha256 per returned array, compared between the two sides every round",
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "hostname": __import__("socket").gethostname(),
        },
        **provenance(),
    }
    write_artifact(args.output, artifact)
    if differing:
        print(f"IDENTITY FAILED for {differing}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
