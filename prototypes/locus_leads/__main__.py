"""Command line for the Locus Lead prototype (#402, and the maps of #404).

    python -m prototypes.locus_leads build-map --archive A --output M.tsv.gz
    python -m prototypes.locus_leads compare --store S --analysis A --genetic-map M \\
        --panel P --d-cm 0.5 --w-bp 1000000 --r2 0.01 --r2 0.001 --work-dir D --output O.json
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

from .eur_map import build_eur_map
from .genetic_map import load_map
from .maps import (
    MapSharedSet,
    build_hapmap2_map,
    build_median_map,
    comparison_record,
    map_measurements,
    measure_shared_sets,
    selection_digest,
)
from .run import run
from .select import SelectionRule, build_selection
from .sweep import EUR_SPIKES, sweep


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    _HANDLERS[args.command](args)


def _build_mean(args: argparse.Namespace) -> None:
    print(json.dumps(build_eur_map(args.archive, args.output), indent=2))


def _build_hapmap2(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build_hapmap2_map(args.original, args.output), indent=2))


def _build_median(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build_median_map(args.archive, args.output), indent=2))


def _select(args: argparse.Namespace) -> None:
    rule = SelectionRule(args.seed, args.target, args.min_z100, args.min_p0, args.max_rows)
    record = build_selection(args.stores, rule)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")


def _compare(args: argparse.Namespace) -> None:
    started = time.monotonic()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result = run(args.store, args.analysis, load_map(args.genetic_map), args.panel,
                 d_cm=args.d_cm, w_bp=args.w_bp, r2_values=args.r2 or [0.01, 0.001],
                 work_dir=args.work_dir, threads=args.threads, threshold=args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Timings stay out of the JSON, so a re-run reproduces it byte for byte.
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
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
    names = {name: Path(path) for name, path in _named_maps(args.genetic_map).items()}
    sweeps = {name: (path, json.loads(path.read_text())) for name, path in names.items()}
    shared = {}
    for name, record in found["maps"].items():
        shared[name] = MapSharedSet(name, record["analyses"])
    loaded = {name: load_map(Path(sweep["genetic_map"]["path"]))
              for name, (_, sweep) in sweeps.items()}
    record = comparison_record(selection, sweeps, shared, reference=found["reference_map"],
                               measurements=map_measurements(loaded))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")


def _sweep(args: argparse.Namespace) -> None:
    started = time.monotonic()
    selection = json.loads(args.selection.read_text())
    pairs_path = args.output.with_name(args.output.stem + "_pairs.tsv.gz")
    result = sweep(selection, args.stores_dir, load_map(args.genetic_map), args.panel,
                   args.work_dir, hm3_path=args.hm3, pairs_path=pairs_path, threads=args.threads,
                   only=args.only, spikes=EUR_SPIKES if args.spikes == "mean" else ())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(f"wrote {args.output} in {time.monotonic() - started:.1f} s", file=sys.stderr)


_HANDLERS: dict[str, Callable[[argparse.Namespace], None]] = {
    "build-map": _build_mean, "build-hapmap2-map": _build_hapmap2,
    "build-median-map": _build_median, "select": _select, "sweep": _sweep,
    "map-shared-sets": _map_shared_sets, "compare-maps": _compare_maps, "compare": _compare,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m prototypes.locus_leads")
    commands = parser.add_subparsers(dest="command", required=True)
    _map_parsers(commands)
    _sweep_parsers(commands)
    _compare_parser(commands)
    return parser


def _map_parsers(commands: Any) -> None:
    build = commands.add_parser("build-map", help="average the pyrho EUR maps into one file")
    build.add_argument("--archive", type=Path, required=True, help="hg38_maps.tar.gz")
    build.add_argument("--output", type=Path, required=True, help="<name>.tsv.gz to write")
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
    pick = commands.add_parser("select", help="draw the sweep's EUR Analyses (#403)")
    pick.add_argument("--stores", type=Path, required=True, help="directory of OGS-* Stores")
    pick.add_argument("--seed", type=int, required=True)
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
    run_sweep.add_argument("--only", type=int, help="the first N chosen Analyses only")
    run_sweep.add_argument("--spikes", choices=("mean", "none"), default="mean",
                           help="measure the mean map's two single-population spikes (#402); "
                                "another map's run passes `none`")
    run_sweep.add_argument("--output", type=Path, required=True)
    compare_maps = commands.add_parser("compare-maps",
                                       help="each map's count criteria, side by side (#404)")
    compare_maps.add_argument("--selection", type=Path, required=True)
    compare_maps.add_argument("--shared-sets", type=Path, required=True)
    compare_maps.add_argument("--genetic-map", action="append", required=True, metavar="NAME=JSON",
                              help="each map's sweep JSON, under the name its map had")
    compare_maps.add_argument("--output", type=Path, required=True)


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


if __name__ == "__main__":
    main()
