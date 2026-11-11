#!/usr/bin/env python3
"""Convert, build and validate the five HGDP+1kGP LD Stores (ogld#7; spec build-inputs §3, §10).

    pixi run -e dev python scripts/build_hgdp1kgp.py convert  [--panel AFR ...] [--jobs 8]
    pixi run -e dev python scripts/build_hgdp1kgp.py build    [--panel AFR ...] [--builds 2]

`convert` turns each panel's 22 phased v2 BCFs into one pgen fileset per chromosome with
build-inputs §3's flags and the panel's unrelated keep file (§10, ADR 0002 decision 11). It is
idempotent: a fileset is skipped when its plink2 log finished, records the same options, and the
`.psam` holds exactly the keep file's samples; anything else is converted again. A conversion
whose sample count differs from the keep file's stops the run.

`build` runs `ogld build --k 10 --r-scale 511` (ADR 0004) with the panel's PC File, keep file and
panel id, then `ogld validate`, each at `--threads` (16) threads, `--builds` (2) panels at once.
It refuses a store that already exists, and stops on a failed build or validation rather than
going on to the next chromosome's or panel's: a build that cannot do the right thing says so.

Where things live, each overridable: `$OGLD_PHASED_V2`, `$OGLD_PCS`, `$OGLD_KEEP`, `$OGLD_PLINK2`
(scripts/test_panel/roots.py); `$OGLD_PGEN_V2` (default
`/data/opengwasdb/work/ogld/phased_v2_pgen`), `$OGLD_STORES`
(`/data/opengwasdb/work/ogld/stores`), `$OGLD_SCRATCH` (`/data/opengwasdb/work/ogld/scratch/7`,
the logs; each timing line carries `uptime`).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from test_panel.roots import root, tool

#: AFR first: it has the most variants, so its chr6 is the memory to watch.
PANELS = ("AFR", "EUR", "EAS", "SAS", "AMR")
CHROMS = tuple(range(1, 23))
PANEL_SCALE = 511
K = 10
CONVERT_THREADS = 4
CONVERT_MEMORY_MB = 16000
BUILD_THREADS = 16
MAX_CONVERSIONS = 8
MAX_BUILDS = 2
BCF = "hgdp1kgp_chr{chrom}.filtered.SNV_INDEL.phased.shapeit5.bcf"
#: The options a log must record for a fileset to count as this conversion's; `--threads`,
#: `--memory` and `--out` do not change what is written.
IGNORED_OPTIONS = {"--threads", "--memory", "--out"}


def directory(variable: str, default: str) -> Path:
    return Path(os.environ.get(variable, default))


def pgen_root() -> Path:
    return directory("OGLD_PGEN_V2", "/data/opengwasdb/work/ogld/phased_v2_pgen")


def stores_root() -> Path:
    return directory("OGLD_STORES", "/data/opengwasdb/work/ogld/stores")


def scratch_root() -> Path:
    return directory("OGLD_SCRATCH", "/data/opengwasdb/work/ogld/scratch/7")


def prefix(panel: str, chrom: int) -> Path:
    return pgen_root() / panel / f"chr{chrom}"


def keep_file(panel: str) -> Path:
    return root("OGLD_KEEP") / f"{panel}.keep"


def store_path(panel: str) -> Path:
    return stores_root() / f"hgdp1kgp-v2-{panel}-k{K}.store"


def conversion_command(
    plink2: Path,
    bcf: Path,
    keep: Path,
    out: Path,
    threads: int = CONVERT_THREADS,
    memory_mb: int = CONVERT_MEMORY_MB,
) -> list[str]:
    """build-inputs §3's BCF conversion, restricted to the keep file's samples."""
    return [
        str(plink2),
        "--bcf",
        str(bcf),
        "--keep",
        str(keep),
        "--max-alleles",
        "2",
        "--set-all-var-ids",
        "@:#:$r:$a",
        "--new-id-max-allele-len",
        "1000",
        "--make-pgen",
        "pvar-cols=filter",
        "--out",
        str(out),
        "--threads",
        str(threads),
        "--memory",
        str(memory_mb),
    ]


def build_command(
    panel: str, pgens: list[Path], store: Path, keep: Path, pcs: Path, threads: int
) -> list[str]:
    """`ogld build` for one panel: k = 10, r scale 511, the unrelated keep, its PC File."""
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
        f"ogld scripts/make_pcs.py on HGDP+1kGP v2 {panel} (ADR 0002)",
        "--panel-id",
        f"hgdp1kgp-v2-{panel}",
        "--ancestry",
        panel,
        "--genotype-source",
        f"HGDP+1kGP phased v2 (Koenig et al. 2024), {panel}, converted by build-inputs §3",
        "--k",
        str(K),
        "--r-scale",
        str(PANEL_SCALE),
        "--info-min",
        "none",
        "--threads",
        str(threads),
    ]


def validate_command(store: Path, threads: int) -> list[str]:
    return ["ogld", "validate", "--threads", str(threads), str(store)]


def options_in(tokens: list[str]) -> dict[str, str]:
    """A command line's `--flag value` pairs, without the options that do not change the output."""
    found: dict[str, str] = {}
    flag: str | None = None
    for token in tokens:
        if token.startswith("--"):
            flag = token
            found[flag] = ""
        elif flag is not None:
            found[flag] = (found[flag] + " " + token).strip()
    return {f: v for f, v in found.items() if f not in IGNORED_OPTIONS}


def logged_options(log_text: str) -> dict[str, str] | None:
    """The options plink2's log records, or None when the run did not finish ("End time")."""
    if "\nEnd time:" not in log_text:
        return None
    block = log_text.split("Options in effect:\n", 1)[-1].split("\n\n", 1)[0]
    lines = [line.strip() for line in block.splitlines() if line.strip().startswith("--")]
    return options_in(_split(lines))


def _split(lines: list[str]) -> list[str]:
    tokens: list[str] = []
    for line in lines:
        flag, _, value = line.partition(" ")
        tokens.append(flag)
        if value:
            tokens.append(value)
    return tokens


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


def conversion_current(command: list[str], out: Path, keep: Path) -> bool:
    """True when `out` is a finished conversion made with `command`'s options of exactly the keep
    file's samples: its log finished and records the same options, and its files exist."""
    log = Path(f"{out}.log")
    if not (
        log.exists() and all(Path(f"{out}.{ext}").exists() for ext in ("pgen", "pvar", "psam"))
    ):
        return False
    logged = logged_options(log.read_text())
    if logged is None or logged != options_in(command):
        return False
    return sorted(psam_iids(Path(f"{out}.psam"))) == sorted(keep_iids(keep))


def uptime() -> str:
    return subprocess.run(["uptime"], capture_output=True, text=True, check=True).stdout.strip()


def log_line(path: Path, text: str) -> None:
    with path.open("a") as handle:
        handle.write(text + "\n")


def convert_one(plink2: Path, panel: str, chrom: int, threads: int, memory_mb: int) -> str:
    out = prefix(panel, chrom)
    keep = keep_file(panel)
    bcf = root("OGLD_PHASED_V2") / BCF.format(chrom=chrom)
    if not bcf.exists():
        raise SystemExit(f"{bcf}: missing")
    command = conversion_command(plink2, bcf, keep, out, threads, memory_mb)
    name = f"{panel} chr{chrom}"
    if conversion_current(command, out, keep):
        return f"{name}: skipped, current"
    out.parent.mkdir(parents=True, exist_ok=True)
    timing = scratch_root() / "convert.log"
    timing.parent.mkdir(parents=True, exist_ok=True)
    log_line(timing, f"{name} start {uptime()}")
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    log_line(timing, f"{name} end rc={done.returncode} {uptime()}")
    if done.returncode != 0:
        raise SystemExit(
            f"{name}: plink2 failed ({done.returncode}):\n{done.stderr or done.stdout}"
        )
    got, want = len(psam_iids(Path(f"{out}.psam"))), len(keep_iids(keep))
    if got != want:
        raise SystemExit(f"{name}: converted {got} samples; the keep file has {want}")
    if not conversion_current(command, out, keep):
        raise SystemExit(f"{name}: the finished conversion does not match its command")
    return f"{name}: converted, {got} samples"


def convert(panels: list[str], jobs: int, threads: int, memory_mb: int) -> None:
    if not 1 <= jobs <= MAX_CONVERSIONS:
        raise SystemExit(f"--jobs must be 1 to {MAX_CONVERSIONS} (the node is shared)")
    plink2 = tool("plink2")
    work = [(p, c) for p in panels for c in CHROMS]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for message in pool.map(lambda w: convert_one(plink2, *w, threads, memory_mb), work):
            print(message, flush=True)


def timed(command: list[str], log: Path, stage: str) -> None:
    """Run `command` under /usr/bin/time -v, its stdout and the time report in `log`'s siblings,
    with `uptime` before and after; a non-zero exit stops the run."""
    stem = log.with_suffix("")
    log_line(log, f"{stage} start {uptime()}")
    with Path(f"{stem}.{stage}.out").open("w") as out:
        done = subprocess.run(
            ["/usr/bin/time", "-v", "-o", f"{stem}.{stage}.time", *command],
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
    scratch_root().mkdir(parents=True, exist_ok=True)
    log = scratch_root() / f"{panel}.log"
    timed(build_command(panel, pgens, store, keep, pcs, threads), log, "build")
    timed(validate_command(store, threads), log, "validate")
    return f"{panel}: built and validated, {store}"


def build(panels: list[str], builds: int, threads: int) -> None:
    if not 1 <= builds <= MAX_BUILDS:
        raise SystemExit(f"--builds must be 1 to {MAX_BUILDS} (the node is shared)")
    if not 1 <= threads <= BUILD_THREADS:
        raise SystemExit(f"--threads must be 1 to {BUILD_THREADS} (the node is shared)")
    with ThreadPoolExecutor(max_workers=builds) as pool:
        for message in pool.map(lambda p: build_one(p, threads), panels):
            print(message, flush=True)


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
