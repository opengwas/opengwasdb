"""The `ogld` command: `ogld build`, `ogld validate`, `ogld info` (layer 3).

A thin shell over the public functions, so a job and a person run exactly what a caller
of the package runs: `build` (ogld#5), `validate` and `info` (ogld#4). Exit status: 0
done, 1 the store breaks a rule, 2 a refusal or an input it cannot use.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ogld.build import BuildInputs, build
from ogld.format import Refusal
from ogld.query import open as open_store
from ogld.store import validate


def _optional_float(text: str) -> float | None:
    return None if text.lower() == "none" else float(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ogld", description="Build, validate and inspect LD Stores."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser(
        "build", help="build an LD Store from a Source Panel (docs/spec/build-inputs.md)"
    )
    b.add_argument("out", type=Path)
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
    b.add_argument("--k", type=int, default=10)
    b.add_argument("--mac-min", type=int, default=20)
    b.add_argument("--info-min", type=_optional_float, default=0.8, help="a number, or none")
    b.add_argument("--call-rate-min", type=_optional_float, default=0.95, help="a number, or none")
    b.add_argument("--threads", type=int, default=1)
    for name, text in (("validate", "run every validation rule"), ("info", "print the provenance")):
        commands.add_parser(name, help=text).add_argument("location")
    return parser


def _build(args: argparse.Namespace) -> int:
    inputs = BuildInputs(
        pgen=tuple(args.pgen),
        keep=args.keep,
        pcs=args.pcs,
        pc_source=args.pc_source,
        panel_id=args.panel_id,
        ancestry=args.ancestry,
        genotype_source=args.genotype_source,
        k=args.k,
        mac_min=args.mac_min,
        info_min=args.info_min,
        call_rate_min=args.call_rate_min,
    )
    print(json.dumps(build(inputs, args.out, threads=args.threads), indent=1))
    return 0


def _validate(args: argparse.Namespace) -> int:
    findings = validate(args.location)
    for rule, found in findings.items():
        print(f"{'ok  ' if not found else 'FAIL'} {rule}")
        for finding in found:
            print(f"     {finding.where}: {finding.message}")
    return 1 if any(findings.values()) else 0


def _info(args: argparse.Namespace) -> int:
    print(json.dumps(open_store(args.location).provenance, indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one `ogld` command and return its exit status."""
    args = _parser().parse_args(argv)
    run = {"build": _build, "validate": _validate, "info": _info}[args.command]
    try:
        return run(args)
    except (Refusal, ValueError) as problem:
        print(f"ogld {args.command}: {problem}", file=sys.stderr)
        return 2


__all__ = ["main"]
