"""What the `derive_*` scripts share: the load average and the loop over the stores to derive."""

from __future__ import annotations

import argparse
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import zarr


def uptime() -> str:
    return subprocess.run(["uptime"], capture_output=True, text=True, check=True).stdout.strip()


def derive_each(
    description: str, derive: Callable[[Path], int], noun: str, argv: list[str] | None
) -> int:
    """Parse `stores` and `--threads`, run `derive` over each store, and print what it counted
    (`noun`) with the seconds it took and the load average, before and after."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("stores", nargs="+", type=Path)
    parser.add_argument("--threads", type=int, default=8, help="zarr's threads (8)")
    args = parser.parse_args(argv)
    zarr.config.set({"threading.max_workers": args.threads})
    for store in args.stores:
        print(f"{store}: start  {uptime()}", flush=True)
        started = time.perf_counter()
        count = derive(store)
        seconds = time.perf_counter() - started
        print(f"{store}: {count:,} {noun} in {seconds:.1f} s  {uptime()}", flush=True)
    return 0
