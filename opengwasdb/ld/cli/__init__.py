"""The `opengwasdb ld` commands: `build`, `import`, `validate`, `info`, `derive-partners` (layer 3).

A thin shell over the public functions, so a job and a person run exactly what a caller of the
package runs: `build` (ogld#5), `validate` and `info` (ogld#4), and `derive-partners` (ogld#69).
Exit status: 0 done, 1 the store breaks a rule, 2 a refusal: a `Refusal`, an `InputError`, a
`VariantNormalisationError`, a `GenotypeExtraMissing` (a build that needs pgenlib without the
`genotype` extra, which `opengwasdb[ld-build]` installs) or an environment variable under an old
name, printed as
`opengwasdb ld <command>: <problem>`. A command line the parser refuses also exits 2, but prints
Click's own `Usage: … Error: …`. Any other exception is a fault inside `opengwasdb.ld` and
propagates with its traceback.

An option left out takes the default `BuildInputs`, `ImportInputs` and the functions declare, so
no default is written twice: an option the command line did not set is None here and is not
passed on (`_given`).
"""

from __future__ import annotations

import functools
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from ogquery.ld.format import InputError, Refusal
from ogquery.opening import GenotypeExtraMissing

from opengwasdb.ld import env
from opengwasdb.ld.build import BuildInputs, build, build_genotypes
from opengwasdb.ld.build.importing.policy import DISPLACED_CAP, FAR_TARGET_BP
from opengwasdb.ld.format import FAR_BP, NEAR_BP
from opengwasdb.ld.query import open as open_store
from opengwasdb.ld.query.validation import validate_at
from opengwasdb.ld.store import Finding
from opengwasdb.ld.store.assembly import derive_partners
from opengwasdb.store_reader.identity import VariantNormalisationError

ld_app = typer.Typer(
    name="ld",
    help="Build, validate and inspect LD Stores.",
    no_args_is_help=True,
    add_completion=False,
)

_THREADS = {"min": 1}
_REFUSED = (Refusal, InputError, VariantNormalisationError, GenotypeExtraMissing)


def _exits(command: str) -> Callable[[Callable[..., int]], Callable[..., None]]:
    """Run one command: refuse an old environment variable first, map a refusal to exit 2 and
    its status to the process's exit code."""

    def decorate(run: Callable[..., int]) -> Callable[..., None]:
        @functools.wraps(run)
        def command_with_status(*args: Any, **options: Any) -> None:
            try:
                env.refuse_legacy()
                status = run(*args, **options)
            except _REFUSED as problem:
                print(f"opengwasdb ld {command}: {problem}", file=sys.stderr)
                status = 2
            if status:
                raise typer.Exit(status)

        return command_with_status

    return decorate


def _given(**options: Any) -> dict[str, Any]:
    """The options the command line set; the rest keep the defaults the callee declares."""
    return {name: value for name, value in options.items() if value is not None}


def _info_min(text: str | None) -> dict[str, Any]:
    """`--info-min`: a number, or `none` for no INFO filter; left out, `BuildInputs`' default."""
    if text is None:
        return {}
    if text.lower() == "none":
        return {"info_min": None}
    try:
        return {"info_min": float(text)}
    except ValueError:
        raise typer.BadParameter(
            f"needs a number or none, not {text!r}", param_hint="--info-min"
        ) from None


@ld_app.command("build")
@_exits("build")
def build_command(
    out: Annotated[Path, typer.Argument(help="the Store to write; it must not exist")],
    *,
    genotypes: Annotated[
        bool, typer.Option("--genotypes", help="write a Genotype Store sidecar")
    ] = False,
    pgen: Annotated[list[Path], typer.Option(help="repeat for one file per chromosome")],
    keep: Annotated[Path, typer.Option()],
    pcs: Annotated[Path | None, typer.Option()] = None,
    pc_source: Annotated[str | None, typer.Option()] = None,
    panel_id: Annotated[str, typer.Option()],
    ancestry: Annotated[str, typer.Option()],
    genotype_source: Annotated[str, typer.Option()],
    r_scale: Annotated[
        int,
        typer.Option(
            help="the store's r scale, r̂ = code / r_scale, as ADR 0090 sets it for the panel; "
            "no default",
        ),
    ],
    t_partner: Annotated[
        float,
        typer.Option(
            help="the r² a pair within near_bp must reach to be in the Partner Rows; "
            "no default, and never above t_far (partner-rows §3)",
        ),
    ],
    k: Annotated[int | None, typer.Option()] = None,
    k_min: Annotated[int | None, typer.Option()] = None,
    mac_min: Annotated[int | None, typer.Option()] = None,
    info_min: Annotated[str | None, typer.Option(help="a number, or none")] = None,
    threads: Annotated[int | None, typer.Option()] = None,
) -> int:
    """Build an LD Store from a Source Panel (docs/spec/build-inputs.md)."""
    inputs = BuildInputs(
        pgen=tuple(pgen),
        keep=keep,
        pcs=pcs,
        pc_source=pc_source,
        panel_id=panel_id,
        ancestry=ancestry,
        genotype_source=genotype_source,
        r_scale=r_scale,
        t_partner=t_partner,
        **_given(k=k, k_min=k_min, mac_min=mac_min),
        **_info_min(info_min),
    )
    builder = build_genotypes if genotypes else build
    print(json.dumps(builder(inputs, out, **_given(threads=threads)), indent=1))
    return 0


# The defaults quoted below are `ImportInputs`' own: its `far_target_bp` and `displaced_cap` default
# to these same policy constants, so the help text names the one source.
_FarTarget = Annotated[
    int | None,
    typer.Option(
        min=NEAR_BP + 1,
        max=FAR_BP,
        help="complete radius every chromosome must reach, after dropping displaced liftover "
        f"(default {FAR_TARGET_BP}; above the Near Band, at most far_bp)",
    ),
]
_DisplacedCap = Annotated[
    int | None,
    typer.Option(
        min=0,
        help="most variants one displaced liftover segment may lose; a larger one refuses "
        f"(default {DISPLACED_CAP})",
    ),
]


_Ledger = Annotated[Path | None, typer.Option(help="supplied part listing/SHA-256 ledger")]
_Parts = Annotated[Path | None, typer.Option(help="read-only directory of existing parts")]
_DeleteParts = Annotated[
    bool | None, typer.Option("--delete-parts", help="delete owned downloads after close")
]
_DownloadWorkers = Annotated[
    int | None, typer.Option(help="parts of a chromosome downloaded at once", **_THREADS)
]
_PlanOnly = Annotated[
    bool,
    typer.Option(
        "--plan-only",
        help="write WORK/plan.json and the lifted tables, then exit: no listing, no "
        "download, OUT not created; the same command without it resumes from the plan",
    ),
]


def _import_result(inputs: Any, out: Path, plan_only: bool) -> Mapping[str, object]:
    """Run the import, or only its plan, and return what the command prints."""
    # Optional dependencies are loaded only on this path, never when opening an LD Store.
    from opengwasdb.ld.build.importing.run import _plan_path, import_panukb, plan_panukb

    if not plan_only:
        return import_panukb(inputs, out)
    plan = plan_panukb(inputs, out)
    return {
        "plan": str(_plan_path(inputs)),
        "far_bp": plan["far_bp"],
        "chromosomes": list(plan["chromosomes"]),
    }


@ld_app.command("import")
@_exits("import")
def import_command(
    out: Annotated[Path, typer.Argument(help="the Store to write; it must not exist")],
    *,
    metadata: Annotated[Path, typer.Option()],
    variants: Annotated[Path, typer.Option()],
    chain: Annotated[Path, typer.Option()],
    fasta: Annotated[Path, typer.Option()],
    work: Annotated[Path, typer.Option()],
    ledger: _Ledger = None,
    parts: _Parts = None,
    delete_parts: _DeleteParts = None,
    threads: Annotated[int | None, typer.Option(**_THREADS)] = None,
    download_workers: _DownloadWorkers = None,
    far_target_bp: _FarTarget = None,
    displaced_cap: _DisplacedCap = None,
    plan_only: _PlanOnly = False,
) -> int:
    """Import Pan-UKB published LD (ADR 0101)."""
    from opengwasdb.ld.build.importing.run import ImportInputs

    inputs = ImportInputs(
        metadata=metadata,
        variants=variants,
        chain=chain,
        fasta=fasta,
        work=work,
        **_given(
            ledger=ledger,
            parts=parts,
            delete_parts=delete_parts,
            threads=threads,
            download_workers=download_workers,
            far_target_bp=far_target_bp,
            displaced_cap=displaced_cap,
        ),
    )
    try:
        made = _import_result(inputs, out, plan_only)
    except (ValueError, OSError) as problem:
        raise InputError(str(problem)) from problem
    print(json.dumps(made, indent=1))
    return 0


def report(findings: dict[str, list[Finding]]) -> int:
    """Print each rule's verdict and its findings; 1 when any rule failed, else 0."""
    for rule, found in findings.items():
        print(f"{'FAIL' if found else 'ok  '} {rule}")
        for finding in found:
            print(f"     {finding.where}: {finding.message}")
    return 1 if any(findings.values()) else 0


@ld_app.command("validate")
@_exits("validate")
def validate_command(
    location: Annotated[str, typer.Argument()],
    *,
    threads: Annotated[
        int | None, typer.Option(help="blocks read and checked at once", **_THREADS)
    ] = None,
) -> int:
    """Run every validation rule (docs/spec/ld-store.md §12)."""
    return report(validate_at(location, **_given(threads=threads)))


@ld_app.command("info")
@_exits("info")
def info_command(location: Annotated[str, typer.Argument()]) -> int:
    """Print the provenance."""
    print(json.dumps(open_store(location).provenance, indent=1))
    return 0


@ld_app.command("derive-partners")
@_exits("derive-partners")
def derive_command(
    store: Annotated[Path, typer.Argument()],
    *,
    t_partner: Annotated[
        float,
        typer.Option(
            help="the r² a pair within near_bp must reach to be in the Partner Rows; "
            "no default, and never above the store's t_far (partner-rows §3)",
        ),
    ],
    threads: Annotated[
        int | None, typer.Option(help="compressor threads within one chromosome", **_THREADS)
    ] = None,
    processes: Annotated[
        int | None, typer.Option(help="chromosomes derived at once, largest first", **_THREADS)
    ] = None,
    chromosomes: Annotated[
        list[str] | None,
        typer.Option(
            metavar="CHROM",
            help="a chromosome to derive; repeat for each; every one the store lacks by default",
        ),
    ] = None,
    drop_index: Annotated[
        bool,
        typer.Option(
            "--drop-index",
            help="afterwards remove a Proxy Index the store still holds — a leftover of a store "
            "derived before format 0.2.0 had none — if its Partner Rows pass their rules",
        ),
    ] = False,
) -> int:
    """Add the Partner Rows to an existing LD Store (docs/spec/partner-rows.md §6.2)."""
    made = derive_partners(
        store,
        t_partner=t_partner,
        chromosomes=chromosomes or None,
        drop_index=drop_index,
        **_given(threads=threads, processes=processes),
    )
    if made is None:
        print(
            f"opengwasdb ld derive-partners: {store}: the chromosomes asked for are derived, and "
            "the provenance is not written until every chromosome of the Variant Set has its "
            "Partner Rows (partner-rows §6.2)",
            file=sys.stderr,
        )
        return 0
    print(json.dumps(made, indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one `opengwasdb ld` command line and return its exit status.

    The entry point a test or a script calls; the same commands are `opengwasdb ld …`. A command
    line the parser refuses exits 2 with click's usage message, as a `SystemExit`.
    """
    try:
        status = ld_app(
            args=None if argv is None else list(argv),
            prog_name="opengwasdb ld",
            standalone_mode=False,
        )
    except Exception as problem:
        # The parser's own refusals (a ClickException, whichever click typer carries): its usage
        # message, then exit 2. Matched by what they offer, so this needs no import of click,
        # which newer typer vendors and older typer takes from the environment.
        if not (hasattr(problem, "show") and hasattr(problem, "exit_code")):
            raise
        problem.show()
        raise SystemExit(problem.exit_code) from None
    return int(status or 0)


__all__ = ["ld_app", "main", "report"]
