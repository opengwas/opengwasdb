#!/usr/bin/env python3
"""Publish a local mmap companion from each Store's validated canonical index (ADR 0098).

Run on an isolated or quiescent local Store; never modify a live reader's generation.
A failed partial publication can be retried with the same command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from opengwasdb.ld.store.assembly import derive_alid_mmap as derive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stores", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.stores:
        print(f"{path}: {derive(path)} canonical entries published", flush=True)


if __name__ == "__main__":
    main()
