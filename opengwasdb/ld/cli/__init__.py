"""The `ogld` command: `ogld build`, `ogld validate`, `ogld info`, `ogld derive-partners`
(layer 3).

A thin shell over the public functions, so a job and a person run exactly what a caller
of the package runs: `build` (ogld#5), `validate` and `info` (ogld#4), and `derive-partners`
(ogld#69). Exit status: 0 done, 1 the store breaks a rule, 2 a `Refusal` or an `InputError`. Any
other exception is a fault inside ogld and propagates with its traceback. An option left out
takes the default `BuildInputs` and `build` declare, so no default is written twice.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ogld.build import BuildInputs, build, build_genotypes
from ogld.format import InputError, Refusal
from ogld.identity import IdentityError
from ogld.query import open as open_store
from ogld.store import Finding
from ogld.store.assembly import derive_partners
from ogld.store.validation import validate_at


def _optional_float(text: str) -> float | None:
    return None if text.lower() == "none" else float(text)


def _threads(text: str) -> int:
    """A count of threads; argparse names the option in front of the reason it refuses one."""
    if not text.strip().isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"needs a whole number of at least 1, not {text!r}")
    return int(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ogld", description="Build, validate and inspect LD Stores."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _build_command(commands)
    _validate_command(commands)
    _derive_command(commands)
    commands.add_parser("info", help="print the provenance").add_argument("location")
    return parser


def _build_command(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """`ogld build`: a Source Panel's genotypes to a store (docs/spec/build-inputs.md)."""
    b = commands.add_parser(
        "build", help="build an LD Store from a Source Panel (docs/spec/build-inputs.md)"
    )
    b.add_argument("out", type=Path)
    b.add_argument("--genotypes", action="store_true", help="write a Genotype Store sidecar")
    b.add_argument(
        "--pgen",
        type=Path,
        action="append",
        required=True,
        help="repeat for one file per chromosome",
    )
    b.add_argument("--keep", type=Path, required=True)
    b.add_argument("--pcs", type=Path)
    b.add_argument("--pc-source")
    b.add_argument("--panel-id", required=True)
    b.add_argument("--ancestry", required=True)
    b.add_argument("--genotype-source", required=True)
    b.add_argument(
        "--r-scale",
        type=int,
        required=True,
        help="the store's r scale, r̂ = code / r_scale, as ADR 0004 sets it for the panel; "
        "no default",
    )
    b.add_argument(
        "--t-partner",
        type=float,
        required=True,
        help="the r² a pair within near_bp must reach to be in the Partner Rows; no default, and "
        "never above t_far (partner-rows §3)",
    )
    unset = argparse.SUPPRESS
    b.add_argument("--k", type=int, default=unset)
    b.add_argument("--k-min", type=int, default=unset)
    b.add_argument("--mac-min", type=int, default=unset)
    b.add_argument("--info-min", type=_optional_float, default=unset, help="a number, or none")
    b.add_argument("--threads", type=int, default=unset)


def _validate_command(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """`ogld validate`: every rule of spec §12 against a store."""
    v = commands.add_parser("validate", help="run every validation rule")
    v.add_argument("location")
    v.add_argument("--threads", type=_threads, default=1, help="blocks read and checked at once")


def _derive_command(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """`ogld derive-partners`: the Partner Rows for a store built before them (§6.2)."""
    d = commands.add_parser(
        "derive-partners",
        help="add the Partner Rows to an existing LD Store (docs/spec/partner-rows.md §6.2)",
    )
    d.add_argument("store", type=Path)
    d.add_argument(
        "--t-partner",
        type=float,
        required=True,
        help="the r² a pair within near_bp must reach to be in the Partner Rows; no default, and "
        "never above the store's t_far (partner-rows §3)",
    )
    d.add_argument(
        "--threads", type=_threads, default=1, help="compressor threads within one chromosome"
    )
    d.add_argument(
        "--processes", type=_threads, default=1, help="chromosomes derived at once, largest first"
    )
    d.add_argument(
        "--chromosomes",
        nargs="+",
        default=None,
        metavar="CHROM",
        help="the chromosomes to derive; every one the store lacks by default",
    )
    d.add_argument(
        "--drop-index",
        action="store_true",
        help=(
            "afterwards remove a Proxy Index the store still holds — a leftover of a store derived "
            "before format 0.2.0 had none — if its Partner Rows pass their rules"
        ),
    )


def _given(args: argparse.Namespace, *names: str) -> dict[str, Any]:
    """The options among `names` the command line set; the rest keep their declared defaults."""
    return {name: getattr(args, name) for name in names if hasattr(args, name)}


def _build(args: argparse.Namespace) -> int:
    inputs = BuildInputs(
        pgen=tuple(args.pgen),
        keep=args.keep,
        pcs=args.pcs,
        pc_source=args.pc_source,
        panel_id=args.panel_id,
        ancestry=args.ancestry,
        genotype_source=args.genotype_source,
        r_scale=args.r_scale,
        t_partner=args.t_partner,
        **_given(args, "k", "k_min", "mac_min", "info_min"),
    )
    builder = build_genotypes if args.genotypes else build
    provenance = builder(inputs, args.out, **_given(args, "threads"))
    print(json.dumps(provenance, indent=1))
    return 0


def report(findings: dict[str, list[Finding]]) -> int:
    """Print each rule's verdict and its findings; 1 when any rule failed, else 0."""
    for rule, found in findings.items():
        print(f"{'FAIL' if found else 'ok  '} {rule}")
        for finding in found:
            print(f"     {finding.where}: {finding.message}")
    return 1 if any(findings.values()) else 0


def _validate(args: argparse.Namespace) -> int:
    return report(validate_at(args.location, threads=args.threads))


def _info(args: argparse.Namespace) -> int:
    print(json.dumps(open_store(args.location).provenance, indent=1))
    return 0


def _derive(args: argparse.Namespace) -> int:
    made = derive_partners(
        args.store,
        t_partner=args.t_partner,
        threads=args.threads,
        processes=args.processes,
        chromosomes=args.chromosomes,
        drop_index=args.drop_index,
    )
    if made is None:
        print(
            f"ogld derive-partners: {args.store}: the chromosomes asked for are derived, and the "
            "provenance is not written until every chromosome of the Variant Set has its Partner "
            "Rows (partner-rows §6.2)",
            file=sys.stderr,
        )
        return 0
    print(json.dumps(made, indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one `ogld` command and return its exit status."""
    args = _parser().parse_args(argv)
    run = {
        "build": _build,
        "validate": _validate,
        "info": _info,
        "derive-partners": _derive,
    }[args.command]
    try:
        return run(args)
    except (Refusal, InputError, IdentityError) as problem:
        print(f"ogld {args.command}: {problem}", file=sys.stderr)
        return 2


__all__ = ["main", "report"]
