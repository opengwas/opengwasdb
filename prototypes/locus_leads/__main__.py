"""Command line for the Locus Lead prototype (#402, and the maps of #404).

    python -m prototypes.locus_leads build-map --archive A --output M.tsv.gz [--ancestry AFR]
    python -m prototypes.locus_leads compare --store S --analysis A --genetic-map M \\
        --panel P --d-cm 0.5 --w-bp 1000000 --r2 0.01 --r2 0.001 --work-dir D --output O.json
    python -m prototypes.locus_leads cost --genetic-map M --panel P --work-dir D \\
        --output docs/benchmark-output/locus_leads/cost.json
    python -m prototypes.locus_leads chance-ld --ancestry AFR --sweep S.json \\
        --genetic-map M --panel P --work-dir D --output ancestries_chance_ld.json
    python -m prototypes.locus_leads profile-sweep --selection S --sweep W --stores-dir D \\
        --genetic-map M --panel P --hm3 H --blocks B --work-dir D --output O.json
    python -m prototypes.locus_leads build-hapmap2-map --original O --output M.tsv.gz
    python -m prototypes.locus_leads build-median-map --archive A --output M.tsv.gz
    python -m prototypes.locus_leads map-shared-sets --selection S --stores-dir D \\
        --panel P --genetic-map NAME=M --genetic-map NAME=M --output O.json
    python -m prototypes.locus_leads compare-maps --selection S --shared-sets H \\
        --genetic-map NAME=SWEEP.json --genetic-map NAME=SWEEP.json --output O.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .ancestries import populations
from .cost import measure
from .eur_map import build_eur_map, build_population_map
from .genetic_map import file_digest, load_map
from .maps import (
    MapSharedSet,
    build_hapmap2_map,
    build_median_map,
    comparison_record,
    map_measurements,
    measure_shared_sets,
    selection_digest,
)
from .panel_chance import SEED as CHANCE_SEED, merge_chance, panel_chance
from .profile_sweep import profile_sweep
from .run import run
from .select import SelectionRule, build_selection
from .sweep import EUR_SPIKES, sweep


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    _HANDLERS[args.command](args)


def _build_map(args: argparse.Namespace) -> None:
    """Write the map `--ancestry` names, EUR through the function #403 documented."""
    identity = (build_eur_map(args.archive, args.output) if args.ancestry == "EUR"
                else build_population_map(args.archive, args.output,
                                          populations(args.ancestry)))
    print(json.dumps(identity, indent=2))


def _build_hapmap2(args: argparse.Namespace) -> None:
    _print_identity(build_hapmap2_map(args.original, args.output))


def _build_median(args: argparse.Namespace) -> None:
    _print_identity(build_median_map(args.archive, args.output))


def _print_identity(identity: dict[str, Any]) -> None:
    print(json.dumps(identity, indent=2))


def _cost(args: argparse.Namespace) -> None:
    _write(args, measure(store_root=args.store_root, genetic_map_path=args.genetic_map,
                         panel_dir=args.panel, work_dir=args.work_dir, repeats=args.repeats,
                         threads=args.threads, d_cm=args.d_cm, w_bp=args.w_bp,
                         r2=(args.r2 or [0.01])[0]))


def _select(args: argparse.Namespace) -> None:
    rule = SelectionRule(args.seed, args.target, args.min_z100, args.min_p0, args.max_rows,
                         args.ancestry)
    _write(args, build_selection(args.stores, rule))


def _compare(args: argparse.Namespace) -> None:
    started = time.monotonic()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result = run(args.store, args.analysis, load_map(args.genetic_map), args.panel,
                 d_cm=args.d_cm, w_bp=args.w_bp, r2_values=args.r2 or [0.01, 0.001],
                 work_dir=args.work_dir, threads=args.threads, threshold=args.threshold)
    # The comparison's JSON carries no timings, so a re-run reproduces it byte for byte.
    _write(args, result)
    print(f"wrote {args.output} in {time.monotonic() - started:.1f} s", file=sys.stderr)


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
                   only=args.only, spikes=EUR_SPIKES if args.spikes == "mean" else ())
    _write(args, result)
    print(f"wrote {args.output} in {time.monotonic() - started:.1f} s", file=sys.stderr)


def _profile(args: argparse.Namespace) -> None:
    started = time.monotonic()
    result = profile_sweep(json.loads(args.selection.read_text()),
                           json.loads(args.sweep.read_text()), args.sweep, args.stores_dir,
                           load_map(args.genetic_map), args.panel, args.work_dir,
                           hm3_path=args.hm3, blocks_path=args.blocks, threads=args.threads,
                           only=args.only)
    _write(args, result)
    print(f"wrote {args.output} in {time.monotonic() - started:.1f} s", file=sys.stderr)


def _named_maps(entries: list[str]) -> dict[str, Path]:
    """`NAME=PATH` arguments as a mapping, refusing a name given twice."""
    named: dict[str, Path] = {}
    for entry in entries:
        name, _, path = entry.partition("=")
        if not name or not path or name in named:
            raise SystemExit(f"--genetic-map takes NAME=PATH once per map, not {entry!r}")
        named[name] = Path(path)
    return named


def _map_shared_sets(args: argparse.Namespace) -> None:
    """Check that the maps leave the same shared set, then write what each left."""
    named = _named_maps(args.genetic_map)
    selection = json.loads(args.selection.read_text())
    found = measure_shared_sets(selection, args.stores_dir, args.panel,
                                {name: load_map(path) for name, path in named.items()})
    reference = args.reference or next(iter(named))
    if reference not in found:
        raise SystemExit(f"--reference {reference} is not one of {sorted(found)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "selection": {"sha256": selection_digest(selection), "chosen": len(selection["chosen"])},
        "reference_map": reference,
        "maps": {name: record.record() for name, record in found.items()}},
        indent=1, allow_nan=False) + "\n")


def _compare_maps(args: argparse.Namespace) -> None:
    """The maps' counts side by side, from their sweeps' JSON, with the shared sets checked."""
    selection = json.loads(args.selection.read_text())
    found = json.loads(args.shared_sets.read_text())
    names = _named_maps(args.genetic_map)
    sweeps = {name: (path, json.loads(path.read_text())) for name, path in names.items()}
    shared = {name: MapSharedSet(name, record["analyses"], record["map_sha256"],
                                 record["clamped"])
              for name, record in found["maps"].items()}
    loaded = {name: load_map(Path(sweep["genetic_map"]["path"]))
              for name, (_, sweep) in sweeps.items()}
    record = comparison_record(selection, sweeps, shared, reference=found["reference_map"],
                               measurements=map_measurements(loaded))
    _write(args, record)


def _write(args: argparse.Namespace, result: dict) -> None:
    """Write `result` to `args.output`, making its directory."""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")


_HANDLERS: dict[str, Callable[[argparse.Namespace], None]] = {
    "build-map": _build_map, "build-hapmap2-map": _build_hapmap2,
    "build-median-map": _build_median, "select": _select, "sweep": _sweep,
    "profile-sweep": _profile, "map-shared-sets": _map_shared_sets,
    "compare-maps": _compare_maps, "compare": _compare, "cost": _cost,
    "chance-ld": _chance_ld,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m prototypes.locus_leads")
    commands = parser.add_subparsers(dest="command", required=True)
    _map_parsers(commands)
    _sweep_parsers(commands)
    _compare_parser(commands)
    return parser


def _map_parsers(commands: Any) -> None:
    build = commands.add_parser("build-map",
                                help="average the pyrho population maps into one file")
    build.add_argument("--archive", type=Path, required=True, help="hg38_maps.tar.gz")
    build.add_argument("--output", type=Path, required=True, help="<name>.tsv.gz to write")
    build.add_argument("--ancestry", default="EUR", help="which populations to average (EUR, "
                        "AFR, EAS or AMR)")
    hapmap = commands.add_parser("build-hapmap2-map",
                                 help="normalise HapMap II lifted to GRCh38 (#404)")
    hapmap.add_argument("--original", type=Path, required=True,
                        help="Eagle's genetic_map_hg38_withX.txt.gz")
    hapmap.add_argument("--output", type=Path, required=True, help="<name>.tsv.gz to write")
    median = commands.add_parser("build-median-map",
                                 help="the median of the pyrho EUR population maps (#404)")
    median.add_argument("--archive", type=Path, required=True, help="hg38_maps.tar.gz")
    median.add_argument("--output", type=Path, required=True, help="<name>.tsv.gz to write")
    identity = commands.add_parser("map-shared-sets",
                                   help="check every map leaves the same shared set (#404)")
    identity.add_argument("--selection", type=Path, required=True)
    identity.add_argument("--stores-dir", type=Path, required=True)
    identity.add_argument("--panel", type=Path, required=True)
    identity.add_argument("--genetic-map", action="append", required=True,
                          metavar="NAME=PATH", help="each map, named")
    identity.add_argument("--reference", help="the map the others must match (the first)")
    identity.add_argument("--output", type=Path, required=True)


def _sweep_parsers(commands: Any) -> None:
    pick = commands.add_parser("select",
                               help="draw the sweep's Analyses for one ancestry (#403, #405)")
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
    run_sweep.add_argument("--spikes", choices=("mean", "none"), default="mean",
                           help="measure the mean map's two single-population spikes (#402); "
                                "another map's run passes `none`")
    run_sweep.add_argument("--only", type=int, help="the first N chosen Analyses only")
    run_sweep.add_argument("--output", type=Path, required=True)
    chance = commands.add_parser("chance-ld",
                                 help="a panel's chance LD at its own sample size (#405); one "
                                      "ancestry per run, merged into --output")
    chance.add_argument("--ancestry", required=True, help="the name the entry is filed under")
    chance.add_argument("--sweep", type=Path, required=True,
                        help="the ancestry's sweep JSON, for its swept Analyses")
    chance.add_argument("--stores-dir", type=Path, required=True)
    chance.add_argument("--genetic-map", type=Path, required=True, help="the map the sweep used")
    chance.add_argument("--panel", type=Path, required=True, help="directory of chr<N>.pgen")
    chance.add_argument("--work-dir", type=Path, required=True, help="for plink2's exports")
    chance.add_argument("--threads", type=int, default=4, help="plink2 threads")
    chance.add_argument("--seed", type=int, default=CHANCE_SEED)
    chance.add_argument("--output", type=Path, required=True)
    _add_profile_sweep(commands)


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


def _compare_parser(commands: Any) -> None:
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
    compare_maps = commands.add_parser("compare-maps",
                                       help="each map's count criteria, side by side (#404)")
    compare_maps.add_argument("--selection", type=Path, required=True)
    compare_maps.add_argument("--shared-sets", type=Path, required=True)
    compare_maps.add_argument("--genetic-map", action="append", required=True, metavar="NAME=JSON",
                              help="each map's sweep JSON, under the name its map had")
    compare_maps.add_argument("--output", type=Path, required=True)
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


if __name__ == "__main__":
    main()
