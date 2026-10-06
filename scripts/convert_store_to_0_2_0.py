#!/usr/bin/env python3
"""Convert a Dense Observed-Only Store Release to format 0.2.0 (Zarr v3, sharded).

The logic lives in `opengwasdb.store.convert` so tests and later tickets can
reuse it; this is the thin CLI beside `scripts/restamp_store_to_0_1_0.py`.  It
derives a **new** release: a fresh `release_id` and `created_at`, a
`zarr_v3_conversion` provenance block, a regenerated `overview.html`, and a
`data.zarr` whose every array is Zarr v3 with the sharding codec.  The source is
never written; an existing destination is refused; the result is staged,
verified bit-exact against the source, validated with no errors, and published
by rename.

See ADR 0057 for what 0.2.0 is and why conversion (rather than a rebuild) is the
migration route.

Usage:

    convert_store_to_0_2_0.py STORE --into DEST
        [--dense-analysis-chunk N] [--dense-shard ROWSxCOLS]
        [--top-hit-shard-chunks N] [--workers N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from opengwasdb.store.arrays import DENSE_SHARD_SHAPE, TOP_HIT_SHARD_CHUNKS
from opengwasdb.store.convert import ConversionError, convert_dense_release


def _parse_shard(text: str) -> tuple[int, int]:
    """``"100000x1024"`` -> ``(100000, 1024)``; anything else is a hard error."""
    parts = text.lower().split("x")
    if len(parts) != 2 or not all(part.isdigit() and int(part) > 0 for part in parts):
        raise argparse.ArgumentTypeError(
            f"--dense-shard must be ROWSxCOLS with positive integers, e.g. "
            f"{DENSE_SHARD_SHAPE[0]}x{DENSE_SHARD_SHAPE[1]}; got {text!r}"
        )
    return (int(parts[0]), int(parts[1]))


def _parser() -> argparse.ArgumentParser:
    """The CLI's one parser, kept out of `main` so neither outgrows the gate."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "store",
        type=Path,
        help="the Dense Observed-Only 0.1.0 release to convert; never modified",
    )
    parser.add_argument(
        "--into",
        type=Path,
        required=True,
        help="where the new 0.2.0 release is published (must not already exist); the "
        "source release is immutable and is never written",
    )
    parser.add_argument(
        "--dense-analysis-chunk",
        type=int,
        default=64,
        help="the Dense planes' Analysis-axis inner chunk (default 64), the width a "
        "query reads; the variant-axis inner chunk stays 1000",
    )
    parser.add_argument(
        "--dense-shard",
        type=_parse_shard,
        default=DENSE_SHARD_SHAPE,
        metavar="ROWSxCOLS",
        help=f"the Dense planes' shard, as a whole multiple of the inner chunk "
        f"(default {DENSE_SHARD_SHAPE[0]}x{DENSE_SHARD_SHAPE[1]})",
    )
    parser.add_argument(
        "--top-hit-shard-chunks",
        type=int,
        default=TOP_HIT_SHARD_CHUNKS,
        metavar="N",
        help=f"how many top-hit inner chunks one top-hit shard holds (default "
        f"{TOP_HIT_SHARD_CHUNKS}, the seam's policy). 1 makes every top-hit shard one "
        f"inner chunk -- 'effectively unsharded' for #246's measurement -- while the "
        f"array stays a Zarr v3 sharded array, so the 0.2.0 rule that every array is "
        f"sharded still holds",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes writing destination shards (default 1). Every worker owns whole "
        "shards, so more workers only bound throughput, never correctness",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        destination = convert_dense_release(
            args.store,
            args.into,
            dense_analysis_chunk=args.dense_analysis_chunk,
            dense_shard=args.dense_shard,
            top_hit_shard_chunks=args.top_hit_shard_chunks,
            workers=args.workers,
        )
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Converted {args.store} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
