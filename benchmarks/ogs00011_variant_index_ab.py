#!/usr/bin/env python3
"""Nine-shape scan-vs-index A/B on one OGS-00011 store (#252 step 5).

Every #242/#252 shape is run twice on the same store copy and the same code:
once with `ragged/by_variant/` renamed aside (the step-3 scan) and once with it
in place. Renaming the group is what makes the scan side genuine -- no code path
is monkeypatched -- and the store copy is ours, so the original stays untouched.
Each run is a fresh open with an RSS sampler and the same load gate the extras
runner uses, and the two answers must be identical (count and sha256) or the
shape fails the run.

Shape construction and the one-shape probe are `ogs00011_ab`'s, so the nine
shapes measured here are the committed ones, not a second copy:

    pixi run -e dev python benchmarks/ogs00011_variant_index_ab.py \
        --store /data/opengwasdb/work/epic252/OGS-00011-0.2.0 \
        --output docs/benchmark-output/opengwasdb_ogs00011_252_variant_index_ab.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from json import loads
from pathlib import Path

from benchmarks._artifact import provenance, write_artifact
from benchmarks.ogs00011_ab import SHAPES

#: The repository root, so a probe subprocess can import `benchmarks`.
_REPO = Path(__file__).resolve().parent.parent

#: The shapes that read the variant index (ADR 0060).  The rest are controls:
#: their scan and index answers must be equal *and* their cost unchanged.
_INDEX_SHAPES = frozenset({"phewas_off_axis", "regional"})

#: The index group, relative to the release directory.
_INDEX_REL = Path("data.zarr") / "ragged" / "by_variant"
_HIDDEN_REL = Path("data.zarr") / "ragged" / "by_variant.scan-ab-hidden"


def _probe(store: Path, shape: str, limit: float, max_load: float, *, warm: bool) -> dict:
    """Run one shape in a **fresh interpreter** (clean peak RSS) with a load gate.

    A fresh process is what makes the two sides' RSS comparable: run in one
    process, the second side's sampler carries the first side's resident memory.
    `ogs00011_ab --one-shape` is the committed probe, here with the load gate it
    records in `gate_waited_s`.
    """
    argv = [
        sys.executable,
        str(_REPO / "benchmarks" / "ogs00011_ab.py"),
        "--one-shape",
        "--store",
        str(store),
        "--shape",
        shape,
        "--limit",
        str(limit),
        "--max-start-load",
        str(max_load),
        "--canonical-identity",
    ]
    if warm:
        argv.append("--warm-index")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(argv, cwd=str(_REPO), env=env, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"{shape} probe failed:\n{out.stdout}\n{out.stderr}")
    return loads(out.stdout.strip().splitlines()[-1])


@contextmanager
def _index_hidden(store: Path) -> Iterator[None]:
    """Rename the index aside for the scan run, restoring it whatever happens."""
    index = store / _INDEX_REL
    hidden = store / _HIDDEN_REL
    if hidden.exists():
        raise SystemExit(f"{hidden} already exists; a previous A/B run did not clean up")
    os.replace(index, hidden)
    try:
        yield
    finally:
        os.replace(hidden, index)


def measure(store: Path, shapes: list[str], *, limit: float, max_load: float) -> dict:
    if not (store / _INDEX_REL).exists():
        raise SystemExit(f"{store}: no {_INDEX_REL}; run `ogdb build-variant-index` first")
    out: dict[str, dict] = {}
    for shape in shapes:
        with _index_hidden(store):
            scan = _probe(store, shape, limit, max_load, warm=False)
        # Warm only the shapes that decode the index: warming the rest would
        # read the exception tables for a shape that never needs them and
        # inflate its peak RSS against the scan side's.
        warm = shape in _INDEX_SHAPES
        indexed = _probe(store, shape, limit, max_load, warm=warm)
        for side in (scan, indexed):
            if side.get("timed_out"):
                raise SystemExit(
                    f"{shape}: hit the {limit}s limit; a timed-out run is not evidence"
                )
        if scan.get("sha256") != indexed.get("sha256"):
            raise SystemExit(
                f"{shape}: scan {scan.get('result_count')} rows/{scan.get('sha256')} "
                f"!= index {indexed.get('result_count')} rows/{indexed.get('sha256')}"
            )
        out[shape] = {"scanned": scan, "indexed": indexed}
        print(
            f"{shape}: scan {scan['elapsed_ms']:.1f} ms ({scan['peak_mb'] / 1024:.2f} GiB) "
            f"-> index {indexed['elapsed_ms']:.1f} ms ({indexed['peak_mb'] / 1024:.2f} GiB) "
            f"rows {indexed.get('result_count')}",
            flush=True,
        )
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--limit", type=float, default=600.0, help="per-run seconds limit")
    parser.add_argument(
        "--max-start-load", type=float, default=3.0, help="gate each side below this load"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    shapes = [name.strip() for name in args.shapes.split(",") if name.strip()]
    artifact = {
        "harness": "benchmarks/ogs00011_variant_index_ab.py",
        "store": str(args.store),
        **provenance(),
        "shapes": measure(
            args.store, shapes, limit=args.limit, max_load=args.max_start_load
        ),
    }
    write_artifact(args.output, artifact)
    return 0


if __name__ == "__main__":
    sys.exit(main())
