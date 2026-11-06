"""Command line for the Locus Lead prototype (#402).

    python -m prototypes.locus_leads build-map --archive A --output M.tsv.gz [--ancestry AFR]
    python -m prototypes.locus_leads compare --store S --analysis A --genetic-map M \\
        --panel P --d-cm 0.5 --w-bp 1000000 --r2 0.01 --r2 0.001 --work-dir D --output O.json
    python -m prototypes.locus_leads cost --genetic-map M --panel P --work-dir D \\
        --output docs/benchmark-output/locus_leads/cost.json
    python -m prototypes.locus_leads chance-ld --ancestry AFR --sweep S.json \\
        --genetic-map M --panel P --work-dir D --output ancestries_chance_ld.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .ancestries import populations
from .cost import measure
from .eur_map import build_eur_map, build_population_map
from .genetic_map import file_digest, load_map
from .panel_chance import SEED as CHANCE_SEED, merge_chance, panel_chance
from .profile_sweep import profile_sweep
from .run import run
from .select import SelectionRule, build_selection
from .sweep import EUR_SPIKES, sweep


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "build-map":
        _build_map(args)
        return
    if args.command == "cost":
        _write(args, measure(store_root=args.store_root, genetic_map_path=args.genetic_map,
                             panel_dir=args.panel, work_dir=args.work_dir, repeats=args.repeats,
                             threads=args.threads, d_cm=args.d_cm, w_bp=args.w_bp,
                             r2=(args.r2 or [0.01])[0]))
        return
    if args.command == "chance-ld":
        _chance_ld(args)
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
    if args.command == "profile-sweep":
        _profile_sweep(args)
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


def _build_map(args: argparse.Namespace) -> None:
    """Write the map `--ancestry` names, EUR through the function #403 documented."""
    identity = (build_eur_map(args.archive, args.output) if args.ancestry == "EUR"
                else build_population_map(args.archive, args.output,
                                          populations(args.ancestry)))
    print(json.dumps(identity, indent=2))


def _chance_ld(args: argparse.Namespace) -> None:
    """Add one ancestry's panel-level chance LD to the file, or start it (#405)."""
    started = time.monotonic()
    entry = panel_chance(json.loads(args.sweep.read_text()), args.stores_dir,
                         load_map(args.genetic_map), args.panel, args.work_dir,
                         threads=args.threads, seed=args.seed)
    entry |= {"sweep": {"path": str(args.sweep),
                        "sha256": file_digest(args.sweep, "sha256")}}
    existing = json.loads(args.output.read_text()) if args.output.exists() else None
    record = merge_chance(existing, args.ancestry, entry, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n")
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


def _profile_sweep(args: argparse.Namespace) -> None:
    started = time.monotonic()
    result = profile_sweep(json.loads(args.selection.read_text()),
                           json.loads(args.sweep.read_text()), args.sweep, args.stores_dir,
                           load_map(args.genetic_map), args.panel, args.work_dir,
                           hm3_path=args.hm3, blocks_path=args.blocks, threads=args.threads,
                           only=args.only)
    _write(args, result)
    print(f"wrote {args.output} in {time.monotonic() - started:.1f} s", file=sys.stderr)


def _add_profile_sweep(commands: Any) -> None:
    command = commands.add_parser("profile-sweep", help="the prominence and LD-block rules on "
                                  "#403's Analyses, shared set and clumps (#409)")
    command.add_argument("--selection", type=Path, required=True, help="#403's selection JSON")
    command.add_argument("--sweep", type=Path, required=True, help="#403's sweep JSON")
    command.add_argument("--stores-dir", type=Path, required=True)
    command.add_argument("--genetic-map", type=Path, required=True, help="#403's map")
    command.add_argument("--panel", type=Path, required=True, help="directory of chr<N>.pgen")
    command.add_argument("--hm3", type=Path, required=True, help="#403's HapMap3 ALIDs")
    command.add_argument("--blocks", type=Path, required=True, help="ld_regions_hg38.tsv")
    command.add_argument("--work-dir", type=Path, required=True, help="plink2's files and "
                         "each Analysis's cached record")
    command.add_argument("--threads", type=int, default=4)
    command.add_argument("--only", type=int, help="the first N chosen Analyses only")
    command.add_argument("--output", type=Path, required=True)


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
    _add_profile_sweep(commands)
    chance = commands.add_parser("chance-ld", help="a panel's chance LD at its own sample size "
                                                  "(#405); one ancestry per run, merged into "
                                                  "--output")
    chance.add_argument("--ancestry", required=True, help="the name the entry is filed under")
    chance.add_argument("--sweep", type=Path, required=True,
                        help="the ancestry's sweep JSON, for its swept Analyses")
    chance.add_argument("--stores-dir", type=Path, required=True)
    chance.add_argument("--genetic-map", type=Path, required=True,
                        help="the map the sweep used")
    chance.add_argument("--panel", type=Path, required=True, help="directory of chr<N>.pgen")
    chance.add_argument("--work-dir", type=Path, required=True, help="for plink2's exports")
    chance.add_argument("--threads", type=int, default=4, help="plink2 threads")
    chance.add_argument("--seed", type=int, default=CHANCE_SEED)
    chance.add_argument("--output", type=Path, required=True)
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
