#!/usr/bin/env python3
"""Build HGDP+1kGP EUR chr22 as the published store was built, in each Near Band layout (ogld#8).

Spec §6 names three layouts of `near/<c>/r` as (inner chunk, shard) in codes: A (4,096,
4,194,304), B (32,768, 16,777,216) and C (262,144, 67,108,864), C provisionally. They differ
only in that array; every other array keeps its one shape. ogld#8 times the reads that remain
on each, so each layout is a real build, by `ogld build`, of the same inputs the published EUR
store read for chr22, at the encoding that store records (ADR 0004) and k = 10, with
`ogld.store.layout.NEAR_R` set for the run, as `tests/test_assembly.py` sets it. Nothing in
`ogld/` changes: the layout is the owner's decision, and this only measures.

    pixi run -e dev python scripts/bench/build_layouts.py OUT [--layout A B C] [--threads 8]

Each build runs in a child process, so its wall time and peak memory are its own, and writes
`OUT/chr22-<layout>.store`, its provenance as `.json`, and `.time.json` with the wall time, the
peak RSS and the node's load before and after. `--check` then holds every layout's decoded
`near/22/r` and `index/22/*` equal to each other's, so a timing compares layouts, not data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

#: Spec §6's Near Band layouts, as (inner chunk, shard) in codes.
LAYOUTS: Final = {
    "A": (4_096, 4_194_304),
    "B": (32_768, 16_777_216),
    "C": (262_144, 67_108_864),
}

#: The published EUR store, whose encoding each build takes, and the inputs it recorded for chr22
#: (its `source.sha256`).
_PUBLISHED: Final = Path("/data/opengwasdb/work/ogld/stores/hgdp1kgp-v2-EUR-k10.store")
_INPUTS: Final = Path("/data/opengwasdb/work/ogld")
_BUILD_ARGS: Final = (
    "--pgen",
    str(_INPUTS / "phased_v2_pgen/EUR/chr22.pgen"),
    "--keep",
    str(_INPUTS / "keep/EUR.keep"),
    "--pcs",
    str(_INPUTS / "pcs/EUR.eigenvec"),
    "--pc-source",
    "ogld scripts/make_pcs.py on HGDP+1kGP v2 EUR (ADR 0002)",
    "--panel-id",
    "hgdp1kgp-v2-EUR",
    "--ancestry",
    "EUR",
    "--genotype-source",
    "HGDP+1kGP phased v2 (Koenig et al. 2024), EUR chr22, converted by build-inputs §3",
    "--k",
    "10",
    "--info-min",
    "none",
)


def load() -> tuple[float, float, float]:
    """The node's 1, 5 and 15 minute load averages, as `uptime` prints them."""
    return os.getloadavg()


def store_bytes(path: Path) -> int:
    """The bytes of every file under `path`: what the store takes on disk, before blocks."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def published_scale() -> int:
    """The scale the published EUR store records: a build carries its store's encoding, and
    the encoding has one owner (`ogld.format.REncoding`; ADR 0004)."""
    import ogld
    from ogld.format import REncoding

    return REncoding.of(ogld.open(_PUBLISHED).provenance["encoding"]).scale


def build_one(layout: str, out: Path, threads: int) -> None:
    """Build chr22 in `layout`, in this process: `NEAR_R` is set before `ogld build` runs."""
    from ogld import cli
    from ogld.store import layout as store_layout

    store_layout.NEAR_R = LAYOUTS[layout]
    store = out / f"chr22-{layout}.store"
    if store.exists():
        shutil.rmtree(store)
    scale = published_scale()
    with (out / f"chr22-{layout}.json").open("w") as provenance:
        sys.stdout = provenance
        args = [*_BUILD_ARGS, "--r-scale", str(scale), "--threads", str(threads)]
        status = cli.main(["build", str(store), *args])
        sys.stdout = sys.__stdout__
    if status:
        raise SystemExit(f"ogld build of layout {layout} exited {status}")


def timed_child(layout: str, out: Path, threads: int) -> dict[str, object]:
    """Build `layout` in a child process and say how long it took and what it held."""
    before, start = load(), time.perf_counter()
    child = subprocess.Popen(
        [sys.executable, __file__, str(out), "--one", layout, "--threads", str(threads)]
    )
    # wait4, not wait: the child's own rusage, so each layout's peak is its own build's.
    _, status, usage = os.wait4(child.pid, 0)
    wall = time.perf_counter() - start
    if status:
        raise SystemExit(f"the build of layout {layout} failed: wait status {status}")
    near = out / f"chr22-{layout}.store" / "near" / "22" / "r"
    return {
        "layout": layout,
        "near_r": LAYOUTS[layout],
        "threads": threads,
        "wall_seconds": wall,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "peak_rss_bytes": usage.ru_maxrss * 1024,
        "store_bytes": store_bytes(out / f"chr22-{layout}.store"),
        "near_r_bytes": store_bytes(near),
        "near_r_files": sum(1 for f in near.rglob("*") if f.is_file()),
        "load_before": before,
        "load_after": load(),
    }


def same_data(out: Path, layouts: list[str]) -> None:
    """Every layout's chr22 arrays decode to the same values, read in bounded windows."""
    import numpy as np
    import zarr

    paths = ["near/22/r", "near/22/indptr", "index/22/indptr", "index/22/partner", "index/22/r"]
    first, *rest = (zarr.open_group(out / f"chr22-{name}.store", mode="r") for name in layouts)
    for path in paths:
        a = first[path]
        step = 1 << 26
        for s in range(0, a.shape[0], step):
            want = np.asarray(a[s : s + step])
            for other, name in zip(rest, layouts[1:], strict=True):
                if not np.array_equal(want, np.asarray(other[path][s : s + step])):
                    raise SystemExit(f"{path} differs between {layouts[0]} and {name} at {s}")
    print(f"same data in {', '.join(layouts)}: {', '.join(paths)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--layout", nargs="+", choices=sorted(LAYOUTS), default=sorted(LAYOUTS))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--one", choices=sorted(LAYOUTS), help=argparse.SUPPRESS)
    parser.add_argument("--check", action="store_true", help="only compare the built layouts")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.one:
        build_one(args.one, args.out, args.threads)
        return
    if not args.check:
        for name in args.layout:
            record = timed_child(name, args.out, args.threads)
            (args.out / f"chr22-{name}.time.json").write_text(json.dumps(record, indent=1))
            print(json.dumps(record))
    same_data(args.out, args.layout)


if __name__ == "__main__":
    main()
