"""Command line for the Locus Lead prototype (#402).

    python -m prototypes.locus_leads build-map --archive A --output M.tsv.gz
    python -m prototypes.locus_leads compare --store S --analysis A --genetic-map M \\
        --panel P --d-cm 0.5 --w-bp 1000000 --r2 0.01 --r2 0.001 --work-dir D --output O.json
    python -m prototypes.locus_leads cost --genetic-map M --panel P --work-dir D \\
        --output docs/benchmark-output/locus_leads/cost.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .cost import measure
from .eur_map import build_eur_map
from .genetic_map import load_map
from .run import run


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "build-map":
        print(json.dumps(build_eur_map(args.archive, args.output), indent=2))
        return
    if args.command == "cost":
        _write(args, measure(store_root=args.store_root, genetic_map_path=args.genetic_map,
                             panel_dir=args.panel, work_dir=args.work_dir, repeats=args.repeats,
                             threads=args.threads, d_cm=args.d_cm, w_bp=args.w_bp,
                             r2=(args.r2 or [0.01])[0]))
        return
    started = time.monotonic()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result = run(args.store, args.analysis, load_map(args.genetic_map), args.panel,
                 d_cm=args.d_cm, w_bp=args.w_bp, r2_values=args.r2 or [0.01, 0.001],
                 work_dir=args.work_dir, threads=args.threads, threshold=args.threshold)
    # The comparison's JSON carries no timings, so a re-run reproduces it byte for byte.
    _write(args, result)
    print(f"wrote {args.output} in {time.monotonic() - started:.1f} s", file=sys.stderr)


def _write(args: argparse.Namespace, result: dict) -> None:
    """Write `result` to `args.output`, making its directory."""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m prototypes.locus_leads")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-map", help="average the pyrho EUR maps into one file")
    build.add_argument("--archive", type=Path, required=True, help="hg38_maps.tar.gz")
    build.add_argument("--output", type=Path, required=True, help="<name>.tsv.gz to write")
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
    cost = commands.add_parser("cost", help="time pruning against plink2 clumping (#406)")
    cost.add_argument("--store-root", type=Path, default=Path("/data/opengwasdb/stores"))
    cost.add_argument("--genetic-map", type=Path, required=True, help="<name>.tsv.gz")
    cost.add_argument("--panel", type=Path, required=True, help="directory of chr<N>.pgen")
    cost.add_argument("--d-cm", type=float, default=0.5)
    cost.add_argument("--w-bp", type=int, default=1_000_000)
    cost.add_argument("--r2", type=float, action="append", default=None,
                      help="clumping r^2 (default 0.01)")
    cost.add_argument("--repeats", type=int, default=3, help="timings per stage")
    cost.add_argument("--threads", type=int, default=4, help="plink2 threads")
    cost.add_argument("--work-dir", type=Path, required=True, help="for the cM arrays and plink2")
    cost.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    main()
