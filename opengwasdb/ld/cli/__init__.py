"""The `ogld` command: `ogld build`, `ogld validate`, `ogld info` (layer 3).

A thin shell over the public functions, so a job and a person run exactly what a caller
of the package runs: `build` (ogld#5), `validate` and `info` (ogld#4). Exit status: 0
done, 1 the store breaks a rule, 2 a `Refusal` or an `InputError`. Any other exception is a
fault inside ogld and propagates with its traceback. An option left out takes the default
`BuildInputs` and `build` declare, so no default is written twice.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ogld.build import BuildInputs, build
from ogld.format import InputError, Refusal
from ogld.identity import IdentityError
from ogld.query import open as open_store
from ogld.store import Finding
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
    unset = argparse.SUPPRESS
    b.add_argument("--k", type=int, default=unset)
    b.add_argument("--k-min", type=int, default=unset)
    b.add_argument("--mac-min", type=int, default=unset)
    b.add_argument("--info-min", type=_optional_float, default=unset, help="a number, or none")
    b.add_argument("--threads", type=int, default=unset)
    v = commands.add_parser("validate", help="run every validation rule")
    v.add_argument("location")
    v.add_argument("--threads", type=_threads, default=1, help="blocks read and checked at once")
    commands.add_parser("info", help="print the provenance").add_argument("location")
    return parser


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
        **_given(args, "k", "k_min", "mac_min", "info_min"),
    )
    provenance = build(inputs, args.out, **_given(args, "threads"))
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run one `ogld` command and return its exit status."""
    args = _parser().parse_args(argv)
    run = {"build": _build, "validate": _validate, "info": _info}[args.command]
    try:
        return run(args)
    except (Refusal, InputError, IdentityError) as problem:
        print(f"ogld {args.command}: {problem}", file=sys.stderr)
        return 2


__all__ = ["main", "report"]
