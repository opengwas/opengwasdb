#!/usr/bin/env python3
"""Convert, build and validate the five HGDP+1kGP LD Stores (ogld#7; spec build-inputs §3, §10).

    pixi run -e dev python scripts/ld/build_hgdp1kgp.py convert  [--panel AFR ...] [--jobs 8]
    pixi run -e dev python scripts/ld/build_hgdp1kgp.py build    [--panel AFR ...] [--builds 2]

`convert` turns each panel's 22 phased v2 BCFs into one pgen fileset per chromosome. The command
is `test_panel.tools.conversion_args`, build-inputs §3's, with the panel's unrelated keep file
(§10, ADR 0088 decision 11); this script adds only plink2's threads and memory. A fileset is
current only when its stamp (`<prefix>.stamp.json`, `tools.stamp_matches`) records today's
command, source checksum, keep file hash and plink2 version. The stamp is written after plink2
exits 0 and the fileset holds exactly the keep file's samples, so a conversion that failed, or
was cut short, has none and is converted again. A conversion that finishes with other samples
than the keep file's stops the run.

`build` runs `opengwasdb ld build` with `--k` K, `--r-scale` PANEL_SCALE (ADR 0090) and `--t-partner`
PANEL_T_PARTNER (ADR 0097), the panel's PC File,
keep file and panel id, then `opengwasdb ld validate`, each at `--threads` (16) threads, `--builds` (2)
panels at once. It refuses a store that already exists. When a panel fails, panels not yet
started are cancelled and the one still running finishes; the run then stops with that failure.

Where things live, each overridable and each in `scripts/ld/test_panel/roots.py`: `$OGLD_PHASED_V2`,
`$OGLD_PCS`, `$OGLD_KEEP`, `$OGLD_PLINK2`, and the recipe's `$OGLD_PGEN_V2` (the filesets),
`$OGLD_STORES` and `$OGLD_SCRATCH` (the logs; each timing line carries `uptime`).
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
#: The stores' encoding (ADR 0090): `BuildInputs.r_scale` has no default, so the recipe names it.
PANEL_SCALE = 511
#: The Partner Rows' threshold (ADR 0097, D3): `BuildInputs.t_partner` has no default either, and
#: 0.01 is the owner's for the HGDP+1kGP panels, below EUR's t_far 0.0319 (partner-rows §3, §11).
PANEL_T_PARTNER = 0.01
K = 10
CONVERT_THREADS = 4
CONVERT_MEMORY_MB = 16000
BUILD_THREADS = 16
MAX_CONVERSIONS = 8
MAX_BUILDS = 2
BCF = "hgdp1kgp_chr{chrom}.filtered.SNV_INDEL.phased.shapeit5.bcf"
#: The program that reports a command's peak memory (`-v`) to a file (`-o`).
TIME = "/usr/bin/time"


def prefix(panel: str, chrom: int) -> Path:
    return root("OGLD_PGEN_V2") / panel / f"chr{chrom}"


def keep_file(panel: str) -> Path:
    return root("OGLD_KEEP") / f"{panel}.keep"


def panel_id(panel: str) -> str:
    return f"hgdp1kgp-v2-{panel}"


def store_path(panel: str) -> Path:
    return root("OGLD_STORES") / f"{panel_id(panel)}-k{K}.store"


def build_command(
    panel: str, pgens: list[Path], store: Path, keep: Path, pcs: Path, threads: int
) -> list[str]:
    """`opengwasdb ld build` for one panel: k = K, r scale PANEL_SCALE, the unrelated keep, its PC File."""
    command = ["ogld", "build", str(store)]
    for pgen in pgens:
        command += ["--pgen", str(pgen)]
    return [
        *command,
        "--keep",
        str(keep),
        "--pcs",
        str(pcs),
        "--pc-source",
        f"ogld scripts/make_pcs.py on HGDP+1kGP v2 {panel} (ADR 0088)",
        "--panel-id",
        panel_id(panel),
        "--ancestry",
        panel,
        "--genotype-source",
        f"HGDP+1kGP phased v2 (Koenig et al. 2024), {panel}, converted by build-inputs §3",
        "--k",
        str(K),
        "--r-scale",
        str(PANEL_SCALE),
        "--t-partner",
        str(PANEL_T_PARTNER),
        "--info-min",
        "none",
        "--threads",
        str(threads),
    ]


def validate_command(store: Path, threads: int) -> list[str]:
    return ["ogld", "validate", "--threads", str(threads), str(store)]


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
    bcf = root("OGLD_PHASED_V2") / BCF.format(chrom=chrom)
    if not bcf.exists():
        raise SystemExit(f"{bcf}: missing")
    args = conversion_args(bcf, keep, out)
    inputs = conversion_inputs(plink2, args, bcf, keep)
    name = f"{panel} chr{chrom}"
    if stamp_matches(out, inputs) and all(p.exists() for p in fileset_files(out)):
        return f"{name}: skipped, stamp matches"
    Path(f"{out}.stamp.json").unlink(missing_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    timing = root("OGLD_SCRATCH") / "convert.log"
    log_line(timing, f"{name} start {uptime()}")
    try:
        run_plink2(plink2, args, root("OGLD_SCRATCH") / "convert" / f"{panel}.chr{chrom}.log",
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


def timed(command: list[str], log: Path, stage: str) -> None:
    """Run `command` under `TIME -v`, its stdout and the time report beside `log`, with `uptime`
    before and after; a non-zero exit stops the run."""
    stem = log.with_suffix("")
    log_line(log, f"{stage} start {uptime()}")
    with Path(f"{stem}.{stage}.out").open("w") as out:
        done = subprocess.run(
            [TIME, "-v", "-o", f"{stem}.{stage}.time", *command],
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log_line(log, f"{stage} end rc={done.returncode} {uptime()}")
    if done.returncode != 0:
        raise SystemExit(f"{stage} failed ({done.returncode}); see {stem}.{stage}.out")


def build_one(panel: str, threads: int) -> str:
    keep, store = keep_file(panel), store_path(panel)
    pcs = root("OGLD_PCS") / f"{panel}.eigenvec"
    pgens = [Path(f"{prefix(panel, c)}.pgen") for c in CHROMS]
    missing = [str(p) for p in (keep, pcs, *pgens) if not p.exists()]
    if missing:
        raise SystemExit(f"{panel}: missing inputs: {', '.join(missing)}")
    if store.exists():
        raise SystemExit(f"{store} exists; move it away or build under another $OGLD_STORES")
    store.parent.mkdir(parents=True, exist_ok=True)
    log = root("OGLD_SCRATCH") / f"{panel}.log"
    timed(build_command(panel, pgens, store, keep, pcs, threads), log, "build")
    timed(validate_command(store, threads), log, "validate")
    return f"{panel}: built and validated, {store}"


def build(panels: list[str], builds: int, threads: int) -> None:
    if not 1 <= builds <= MAX_BUILDS:
        raise SystemExit(f"--builds must be 1 to {MAX_BUILDS} (the node is shared)")
    if not 1 <= threads <= BUILD_THREADS:
        raise SystemExit(f"--threads must be 1 to {BUILD_THREADS} (the node is shared)")
    run_all(panels, lambda p: build_one(p, threads), builds)


def parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("convert", "build"):
        one = sub.add_parser(name)
        one.add_argument("--panel", action="append", choices=PANELS, help="default: all five")
    sub.choices["convert"].add_argument("--jobs", type=int, default=MAX_CONVERSIONS)
    sub.choices["convert"].add_argument("--threads", type=int, default=CONVERT_THREADS)
    sub.choices["convert"].add_argument("--memory", type=int, default=CONVERT_MEMORY_MB)
    sub.choices["build"].add_argument("--builds", type=int, default=MAX_BUILDS)
    sub.choices["build"].add_argument("--threads", type=int, default=BUILD_THREADS)
    args = parser.parse_args(argv)
    args.panel = list(dict.fromkeys(args.panel or PANELS))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse(argv)
    if args.command == "convert":
        convert(args.panel, args.jobs, args.threads, args.memory)
    else:
        build(args.panel, args.builds, args.threads)


if __name__ == "__main__":
    main(sys.argv[1:])
