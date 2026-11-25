#!/usr/bin/env python3
"""Convert the five HGDP+1kGP panels' phased BCFs to pgen filesets (ogld#7; build-inputs §3, §10).

    pixi run -e dev python scripts/ld/build_hgdp1kgp.py convert  [--panel AFR ...] [--jobs 8]

`convert` turns each panel's 22 phased v2 BCFs into one pgen fileset per chromosome. The command
is `test_panel.tools.conversion_args`, build-inputs §3's, with the panel's unrelated keep file
(§10, ADR 0088 decision 11); this script adds only plink2's threads and memory. A fileset is
current only when its stamp (`<prefix>.stamp.json`, `tools.stamp_matches`) records today's
command, source checksum, keep file hash and plink2 version. The stamp is written after plink2
exits 0 and the fileset holds exactly the keep file's samples, so a conversion that failed, or
was cut short, has none and is converted again. A conversion that finishes with other samples
than the keep file's stops the run.

The LD Stores and Genotype Stores are built from these filesets in opengwas/opengwasdb-stores
(`pixi run ld-stores build`, `tools/ld-stores/`; opengwasdb-stores#541), which runs
`opengwasdb ld build` and `opengwasdb ld validate` and writes each Store's record. This script's
`build` step moved there.

Where things live, each overridable and each in `scripts/ld/test_panel/roots.py`:
`$OPENGWASDB_LD_PHASED_V2`, `$OPENGWASDB_LD_KEEP`, `$OPENGWASDB_LD_PLINK2`, the recipe's
`$OPENGWASDB_LD_PGEN_V2` (the filesets) and `$OPENGWASDB_LD_SCRATCH` (the logs; each timing line
carries `uptime`).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from test_panel.roots import root, tool
from test_panel.tools import (
    conversion_args,
    conversion_inputs,
    run_plink2,
    stamp_matches,
    write_stamp,
)

#: AFR first: it has the most variants, so its chr6 is the memory to watch.
PANELS = ("AFR", "EUR", "EAS", "SAS", "AMR")
CHROMS = tuple(range(1, 23))
CONVERT_THREADS = 4
CONVERT_MEMORY_MB = 16000
MAX_CONVERSIONS = 8
BCF = "hgdp1kgp_chr{chrom}.filtered.SNV_INDEL.phased.shapeit5.bcf"


def prefix(panel: str, chrom: int) -> Path:
    return root("OPENGWASDB_LD_PGEN_V2") / panel / f"chr{chrom}"


def store_path(panel: str) -> Path:
    """Where a published HGDP+1kGP LD Store is, the one place its name is spelt.
    opengwas/opengwasdb-stores builds and records them (`tools/ld-stores/`); `scripts/ld/bench/`
    and the acceptance runs read them."""
    return root("OPENGWASDB_LD_STORES") / f"hgdp1kgp-v2-{panel}-k10.store"


def keep_file(panel: str) -> Path:
    return root("OPENGWASDB_LD_KEEP") / f"{panel}.keep"


def keep_iids(keep: Path) -> list[str]:
    """A keep file's IIDs: one per line under the `#IID` header."""
    lines = keep.read_text().splitlines()
    if not lines or lines[0] != "#IID":
        raise SystemExit(f"{keep}: not a keep file (first line must be '#IID')")
    return lines[1:]


def psam_iids(psam: Path) -> list[str]:
    rows = [line for line in psam.read_text().splitlines() if line]
    if not rows or not rows[0].startswith("#"):
        raise SystemExit(f"{psam}: no header")
    columns = rows[0].lstrip("#").split("\t")
    at = columns.index("IID") if "IID" in columns else 0
    return [row.split("\t")[at] for row in rows[1:]]


def uptime() -> str:
    return subprocess.run(["uptime"], capture_output=True, text=True, check=True).stdout.strip()


def log_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(text + "\n")


def fileset_files(out: Path) -> list[Path]:
    return [Path(f"{out}.{ext}") for ext in ("pgen", "pvar", "psam")]


def check_fileset(name: str, out: Path, keep: Path) -> None:
    """A finished conversion has all three files and exactly the keep file's samples."""
    for path in fileset_files(out):
        if not path.exists():
            raise SystemExit(f"{name}: plink2 exited 0 but {path} is missing")
    got, want = psam_iids(Path(f"{out}.psam")), keep_iids(keep)
    if len(got) != len(want):
        raise SystemExit(f"{name}: converted {len(got)} samples; the keep file has {len(want)}")
    if sorted(got) != sorted(want):
        raise SystemExit(f"{name}: the converted samples are not the keep file's")


def convert_one(plink2: Path, panel: str, chrom: int, threads: int, memory_mb: int) -> str:
    out, keep = prefix(panel, chrom), keep_file(panel)
    bcf = root("OPENGWASDB_LD_PHASED_V2") / BCF.format(chrom=chrom)
    if not bcf.exists():
        raise SystemExit(f"{bcf}: missing")
    args = conversion_args(bcf, keep, out)
    inputs = conversion_inputs(plink2, args, bcf, keep)
    name = f"{panel} chr{chrom}"
    if stamp_matches(out, inputs) and all(p.exists() for p in fileset_files(out)):
        return f"{name}: skipped, stamp matches"
    Path(f"{out}.stamp.json").unlink(missing_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    timing = root("OPENGWASDB_LD_SCRATCH") / "convert.log"
    log_line(timing, f"{name} start {uptime()}")
    try:
        run_plink2(plink2, args,
                   root("OPENGWASDB_LD_SCRATCH") / "convert" / f"{panel}.chr{chrom}.log",
                   threads, memory_mb)  # fmt: skip
    finally:
        log_line(timing, f"{name} end {uptime()}")
    check_fileset(name, out, keep)
    write_stamp(out, inputs)
    return f"{name}: converted, {len(keep_iids(keep))} samples"


def run_all[T](work: Sequence[T], fn: Callable[[T], str], workers: int) -> None:
    """`fn` over `work`, `workers` at a time, each result printed as it arrives. On the first
    failure the items not yet started are cancelled, those running finish, and it is raised."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, item) for item in work]
        for done in as_completed(futures):
            try:
                print(done.result(), flush=True)
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise


def convert(panels: list[str], jobs: int, threads: int, memory_mb: int) -> None:
    if not 1 <= jobs <= MAX_CONVERSIONS:
        raise SystemExit(f"--jobs must be 1 to {MAX_CONVERSIONS} (the node is shared)")
    plink2 = tool("plink2")
    work = [(p, c) for p in panels for c in CHROMS]
    run_all(work, lambda w: convert_one(plink2, *w, threads, memory_mb), jobs)


def parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("convert")
    one.add_argument("--panel", action="append", choices=PANELS, help="default: all five")
    one.add_argument("--jobs", type=int, default=MAX_CONVERSIONS)
    one.add_argument("--threads", type=int, default=CONVERT_THREADS)
    one.add_argument("--memory", type=int, default=CONVERT_MEMORY_MB)
    args = parser.parse_args(argv)
    args.panel = list(dict.fromkeys(args.panel or PANELS))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse(argv)
    convert(args.panel, args.jobs, args.threads, args.memory)


if __name__ == "__main__":
    main(sys.argv[1:])
