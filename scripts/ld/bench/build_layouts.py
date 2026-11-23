#!/usr/bin/env python3
"""Build HGDP+1kGP EUR chr22 as the published store was built, in each Near Band layout (ogld#8).

ld-store §6 named three layouts of `near/<c>/r` as (inner chunk, shard) in codes: A (4,096,
4,194,304), B (32,768, 16,777,216) and C (262,144, 67,108,864). They differ only in that array;
every other array keeps its one shape. ogld#8 timed the reads on each, and the owner chose C
(12 November 2026; ADR 0092). So each layout is a real build, by `opengwasdb ld build`, of the
published EUR store's chr22, with `opengwasdb.ld.store.layout.NEAR_R` set for the run, as
`tests/ld/test_assembly.py`
sets it. The command is the recipe's (`build_hgdp1kgp.build_command`, which owns k, the scale,
the panel id and the source strings), on its chr22 fileset, keep file and PC File.

    pixi run -e dev python scripts/ld/bench/build_layouts.py [OUT] [--layout A B C] [--threads 8]

OUT is `$OPENGWASDB_LD_LAYOUTS` unless given (`scripts/ld/test_panel/roots.py`). Each build runs in
a child process, so its wall time and peak memory are its own, and writes
`OUT/chr22-<layout>.store`, its provenance as `.json`, and `.time.json` with the wall time, the peak
RSS and the node's load before and after. `--check` then holds every array of every layout, the
variant columns included, equal to the first layout's, so a timing compares layouts, not data.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

#: `scripts/`, where the recipe and the roots live.
SCRIPTS: Final = Path(__file__).resolve().parent.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
recipe = importlib.import_module("build_hgdp1kgp")
roots = importlib.import_module("test_panel.roots")

#: ld-store §6's Near Band layouts, as (inner chunk, shard) in codes.
LAYOUTS: Final = {
    "A": (4_096, 4_194_304),
    "B": (32_768, 16_777_216),
    "C": (262_144, 67_108_864),
}
#: The panel and the chromosome built, in the recipe's terms.
PANEL: Final = "EUR"
CHROM: Final = 22


def build_command(store: Path, threads: int) -> list[str]:
    """The recipe's `opengwasdb ld build` for the panel's one chromosome, without the program
    name."""
    pgen = Path(f"{recipe.prefix(PANEL, CHROM)}.pgen")
    pcs = roots.root("OPENGWASDB_LD_PCS") / f"{PANEL}.eigenvec"
    command: list[str] = recipe.build_command(
        PANEL, [pgen], store, recipe.keep_file(PANEL), pcs, threads
    )
    if command[:3] != ["opengwasdb", "ld", "build"]:
        raise SystemExit(f"the recipe's build command is not `opengwasdb ld build`: {command[:3]}")
    return command[2:]


def load() -> tuple[float, float, float]:
    """The node's 1, 5 and 15 minute load averages, as `uptime` prints them."""
    return os.getloadavg()


def store_bytes(path: Path) -> int:
    """The bytes of every file under `path`: what the store takes on disk, before blocks."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def build_one(layout: str, out: Path, threads: int) -> None:
    """Build chr22 in `layout`, in this process: `NEAR_R` is set before `opengwasdb ld build`
    runs. The child process exits after, so no other build sees the layout."""
    import contextlib

    from opengwasdb.ld import cli
    from opengwasdb.ld.store import layout as store_layout

    store_layout.NEAR_R = LAYOUTS[layout]
    store = out / f"chr22-{layout}.store"
    if store.exists():
        shutil.rmtree(store)
    provenance = out / f"chr22-{layout}.json"
    with provenance.open("w") as written, contextlib.redirect_stdout(written):
        status = cli.main(build_command(store, threads))
    if status:
        raise SystemExit(f"opengwasdb ld build of layout {layout} exited {status}")


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


def arrays(group: Any, prefix: str = "") -> list[str]:
    """Every array under `group`, by path, sorted."""
    import zarr

    found = []
    for name, member in group.members():
        path = f"{prefix}{name}"
        found += [path] if isinstance(member, zarr.Array) else arrays(member, f"{path}/")
    return sorted(found)


def same_data(out: Path, layouts: list[str]) -> None:
    """Every array of every layout decodes to the first layout's values, read in bounded
    windows: the Near Band, the Far Pairs and the Partner Rows, and the variant columns too. No
    layout this script builds holds a Proxy Index (opengwas/ogld#72)."""
    import numpy as np
    import zarr

    first, *rest = (zarr.open_group(out / f"chr22-{name}.store", mode="r") for name in layouts)
    paths = arrays(first)
    for other, name in zip(rest, layouts[1:], strict=True):
        if arrays(other) != paths:
            raise SystemExit(f"{layouts[0]} and {name} hold different arrays")
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
    parser.add_argument("out", type=Path, nargs="?", default=None)
    parser.add_argument("--layout", nargs="+", choices=sorted(LAYOUTS), default=sorted(LAYOUTS))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--one", choices=sorted(LAYOUTS), help=argparse.SUPPRESS)
    parser.add_argument("--check", action="store_true", help="only compare the built layouts")
    args = parser.parse_args()
    args.out = args.out or roots.root("OPENGWASDB_LD_LAYOUTS")
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
