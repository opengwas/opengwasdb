#!/usr/bin/env python3
"""Add the variant-centric `by_variant/` index to an existing 0.2.0 release (ADR 0060).

The logic lives in `opengwasdb.layouts.ragged.by_variant.add_variant_index`; this
is the thin CLI beside `convert_store_to_0_2_0.py`, mirroring
`ogdb build-variant-index`. It writes the release in place: the index group is
built under `by_variant.building`, moved into place with one rename, and the
manifest and any consolidated-metadata record are refreshed only after it is
complete, so an interrupted run leaves either no index or a complete one.

A `0.1.0` release is converted first, as completion does:

    convert_store_to_0_2_0.py STORE --into DEST
    add_variant_index.py DEST

Usage:

    add_variant_index.py STORE [--force] [--spill-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from opengwasdb.layouts.ragged.by_variant import VariantIndexError, add_variant_index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "store",
        type=Path,
        help="the 0.2.0 Ragged or Hybrid release to augment in place",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild an index the release already carries",
    )
    parser.add_argument(
        "--spill-dir",
        type=Path,
        default=None,
        help="directory for the build's transient spill files (default: beside the store)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = add_variant_index(args.store, force=args.force, spill_dir=args.spill_dir)
    except VariantIndexError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"indexed {result.n_rows:,} associations over {result.n_axis:,} variants "
        f"in {result.elapsed_seconds:.1f}s "
        f"(peak RSS {result.peak_rss_bytes / 2**30:.2f} GiB, "
        f"{result.disk_bytes / 2**30:.2f} GiB on disk)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
