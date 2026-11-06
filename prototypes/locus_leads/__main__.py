"""Command line for the Locus Lead prototype (#402).

    python -m prototypes.locus_leads build-map --archive A --output M.tsv.gz [--ancestry AFR]
    python -m prototypes.locus_leads compare --store S --analysis A --genetic-map M \\
        --panel P --d-cm 0.5 --w-bp 1000000 --r2 0.01 --r2 0.001 --work-dir D --output O.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .ancestries import populations
from .eur_map import build_eur_map, build_population_map
from .genetic_map import load_map
from .run import run
from .select import SelectionRule, build_selection
from .sweep import EUR_SPIKES, sweep


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "build-map":
        # EUR through `build_eur_map`, so the command #403 documents writes the
        # map it wrote; any other ancestry through the generalised average.
        identity = (build_eur_map(args.archive, args.output) if args.ancestry == "EUR"
                    else build_population_map(args.archive, args.output,
                                              populations(args.ancestry)))
        print(json.dumps(identity, indent=2))
        return
    if args.command == "select":
        rule = SelectionRule(args.seed, args.target, args.min_z100, args.min_p0, args.max_rows,
                             args.ancestry)
        record = build_selection(args.stores, rule)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
        return
    if args.command == "sweep":
        _sweep(args)
        return
    started = time.monotonic()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result = run(args.store, args.analysis, load_map(args.genetic_map), args.panel,
                 d_cm=args.d_cm, w_bp=args.w_bp, r2_values=args.r2 or [0.01, 0.001],
                 work_dir=args.work_dir, threads=args.threads, threshold=args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Timings stay out of the JSON, so a re-run reproduces it byte for byte.
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(f"wrote {args.output} in {time.monotonic() - started:.1f} s", file=sys.stderr)


def _sweep(args: argparse.Namespace) -> None:
    started = time.monotonic()
    selection = json.loads(args.selection.read_text())
    pairs_path = args.output.with_name(args.output.stem + "_pairs.tsv.gz")
    result = sweep(selection, args.stores_dir, load_map(args.genetic_map), args.panel,
                   args.work_dir, hm3_path=args.hm3, pairs_path=pairs_path, threads=args.threads,
                   spikes=EUR_SPIKES if args.spikes == "eur" else (), only=args.only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(f"wrote {args.output} in {time.monotonic() - started:.1f} s", file=sys.stderr)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m prototypes.locus_leads")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-map", help="average the pyrho population maps into one file")
    build.add_argument("--archive", type=Path, required=True, help="hg38_maps.tar.gz")
    build.add_argument("--output", type=Path, required=True, help="<name>.tsv.gz to write")
    build.add_argument("--ancestry", default="EUR", help="which populations to average (EUR, "
                        "AFR, EAS or AMR)")
    pick = commands.add_parser("select", help="draw the sweep's Analyses for one ancestry (#403, #405)")
    pick.add_argument("--stores", type=Path, required=True, help="directory of OGS-* Stores")
    pick.add_argument("--seed", type=int, required=True)
    pick.add_argument("--ancestry", default="EUR", help="the Assigned Ancestry to draw from")
    pick.add_argument("--target", type=int, default=40)
    pick.add_argument("--min-z100", type=int, default=5, help="Analyses with |z| >= 100")
    pick.add_argument("--min-p0", type=int, default=5, help="Analyses with |z| >= 38.5")
    pick.add_argument("--max-rows", type=int, default=1_000_000)
    pick.add_argument("--output", type=Path, required=True)
    run_sweep = commands.add_parser("sweep", help="sweep D and W over the selection (#403)")
    run_sweep.add_argument("--selection", type=Path, required=True, help="the selection JSON")
    run_sweep.add_argument("--stores-dir", type=Path, required=True)
    run_sweep.add_argument("--genetic-map", type=Path, required=True, help="<name>.tsv.gz")
    run_sweep.add_argument("--panel", type=Path, required=True, help="directory of chr<N>.pgen")
    run_sweep.add_argument("--work-dir", type=Path, required=True, help="plink2's files and "
                           "each Analysis's cached record")
    run_sweep.add_argument("--hm3", type=Path, required=True,
                           help="HapMap3 canonical ALIDs, one per line")
    run_sweep.add_argument("--threads", type=int, default=4)
    run_sweep.add_argument("--spikes", choices=("eur", "none"), default="eur",
                           help="the spike set to record and measure (default: the EUR map's)")
    run_sweep.add_argument("--only", type=int, help="the first N chosen Analyses only")
    run_sweep.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="compare Locus Leads with plink2 clumping")
    compare.add_argument("--store", type=Path, required=True)
    compare.add_argument("--analysis", required=True, help="analysis_id")
    compare.add_argument("--genetic-map", type=Path, required=True, help="<name>.tsv.gz")
    compare.add_argument("--panel", type=Path, required=True, help="directory of chr<N>.pgen")
    compare.add_argument("--d-cm", type=float, required=True)
    compare.add_argument("--w-bp", type=int, required=True)
    compare.add_argument("--r2", type=float, action="append",
                         help="clump r^2 threshold; repeat for more (default 0.01 and 0.001)")
    compare.add_argument("--threshold", type=float, default=5e-8, help="Top-Hit tier")
    compare.add_argument("--work-dir", type=Path, required=True, help="for plink2's files")
    compare.add_argument("--threads", type=int, default=4, help="plink2 threads")
    compare.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    main()
