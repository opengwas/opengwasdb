"""Building an LD Store from a Source Panel's genotypes (layer 2; ogld#5).

The builder runs where the genotypes are and writes only the LD Store (ADR 0001,
decision 5). Its inputs are specified in `docs/spec/build-inputs.md` (ADR 0002): a PLINK
2 pgen with complete genotypes, a keep file and a PC File. ogld runs no PCA: the PCs are
computed outside the library by a recipe script, and the builder only chooses which of
them to adjust for, by the PC Check. It imports `ogld.store`, `ogld.format` and
`ogld.identity`, never `ogld.query`.

pgenlib, the pgen reader, is imported inside the build's read path only, never at module
import: `import ogld` and every reader must work on an install without it
(tests/test_surface.py holds this).
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from pathlib import Path

import zarr

from ogld.build import diagnostic, passes, pc_check, record
from ogld.build.variant_set import Settings, variant_set
from ogld.format import (
    FAR_BP,
    NEAR_BP,
    Adjustment,
    InputError,
    Provenance,
    REncoding,
)
from ogld.store import StoreWriter


def bounded(threads: int) -> AbstractContextManager[object]:
    """The build's thread bound, until the context closes (review of ogld#5): BLAS runs one
    thread, since every parallel part of a build runs on its own `threads` workers, and zarr's
    pool, which encodes the store's chunks, runs at most `threads`. OpenBLAS otherwise starts a
    thread per core in each matrix product, and zarr's default pool adds 28 to the write.

    The caller's zarr setting is restored on exit. zarr makes its pool once, on first use, so
    the cap holds when the build is its process's first zarr work, as `ogld build` is.
    threadpoolctl comes with the build extra (ADR 0002)."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as missing:
        raise InputError(
            "a build bounds its threads with threadpoolctl, which ogld installs with the build "
            "extra: pip install 'ogld[build]'"
        ) from missing
    stack = ExitStack()
    stack.enter_context(threadpool_limits(limits=1, user_api="blas"))
    stack.enter_context(zarr.config.set({"threading.max_workers": threads}))
    return stack


@dataclass(frozen=True)
class BuildInputs:
    """Everything a build reads and every setting it applies (`docs/spec/build-inputs.md`).

    Each field is recorded in the store's provenance (spec §7).

    - `pgen`: one or more PLINK 2 `.pgen` files, one per chromosome or one for the
      genome, each with its `.pvar` (or `.pvar.zst`) and `.psam` beside it; GRCh38. The
      genotypes must be complete: the build refuses any missing call, naming the variant
      and the panel, rather than impute it (owner, 7 November 2026; ADR 0002).
    - `keep`: the samples to use, one `#IID` per line; n is their count.
    - `pcs`: the PC File (plink2's `.eigenvec` layout). The PC Check scans every PC in
      it and drops the local ones; the first up to k PCs it keeps are regressed out.
      None only with k = 0.
    - `pc_source`: the PC File's declared source, recorded beside its SHA-256; None only
      with k = 0.
    - `panel_id`, `ancestry`: the Source Panel, as provenance names it.
    - `genotype_source`: what the genotypes are, such as the release and its files.
    - `r_scale`: the scale of the store's encoding, r̂ = code / r_scale, any whole number
      `ogld.format.REncoding` admits (spec §3). ADR 0004 sets each Source Panel's (owner,
      10 November 2026). There is no default, so no build falls back on a scale it was not
      given; provenance records it in `encoding`.
    - `t_partner`: the **Partner Threshold**, the r² a pair within `near_bp` must reach to be in
      the store's Partner Rows (CONTEXT.md; partner-rows §3). A build input with no default
      (ADR 0011, D3): 0.01 for the HGDP+1kGP panels and 0.001 for UK Biobank EUR, where it
      equals t_far. It is never above t_far, which the store refuses, naming both, once the
      panel's n and the covariates' k give t_far. Provenance records it as `t_partner`.
    - `k`: the target number of PCs to regress out, taken in order from those the PC
      Check keeps (owner, 7 November 2026); provenance records it as `k_target`.
    - `k_min`: the fewest PCs a build may use. When fewer than `k` survive the PC Check,
      the build uses those that survive as long as at least `k_min` do, and refuses
      otherwise, naming the dropped PCs. The number used is t_far's k, and provenance
      records it beside `k_target` and `k_min`. `k_min` applies only when PCs are used:
      with k = 0 no PC File is read, the build is not refused, and provenance records
      k = 0 and `k_min` as null.
    - `mac_min`: the minor allele count floor, counted on the kept samples.
    - `info_min`: the INFO floor; a pgen that carries no INFO with it set is refused,
      so a sequenced panel passes None explicitly rather than skip a filter quietly.
    """

    pgen: tuple[Path, ...]
    keep: Path
    pcs: Path | None
    pc_source: str | None
    panel_id: str
    ancestry: str
    genotype_source: str
    r_scale: int
    t_partner: float
    k: int = pc_check.K_TARGET
    k_min: int = pc_check.K_MIN
    mac_min: int = 20
    info_min: float | None = 0.8

    def __post_init__(self) -> None:
        if not self.pgen:
            raise InputError("BuildInputs needs at least one pgen")
        self.encoding()
        _check_partner(self.t_partner)
        _check_pc_counts(self.k, self.k_min)
        if self.k > 0 and (self.pcs is None or self.pc_source is None):
            raise InputError(
                f"k = {self.k} needs a PC File and its declared source; only k = 0 runs without"
            )

    def encoding(self) -> REncoding:
        """The encoding the build writes, from `r_scale`; an InputError naming a scale the
        format does not admit, or none at all."""
        return REncoding(self.r_scale)


def _check_partner(t_partner: float) -> None:
    """The Partner Threshold is an r², and a build is given one: there is no default
    (ADR 0011, D3). It is above t_far or not, which only the panel's n and k can say, so that
    check is the store's (`REncoding.partner_code_floor`)."""
    if not 0.0 <= t_partner <= 1.0:
        raise InputError(
            f"t_partner is the r² a pair within near_bp must reach, so it is an r² from 0 to 1, "
            f"not {t_partner!r} (partner-rows §3)"
        )


def _check_pc_counts(k: int, k_min: int) -> None:
    """The target and the minimum count PCs, and the minimum cannot exceed the target."""
    if k < 0:
        raise InputError(f"k counts PCs, so it cannot be negative: {k}")
    if k_min < 0:
        raise InputError(f"k_min counts PCs, so it cannot be negative: {k_min}")
    if k > 0 and k_min > k:
        raise InputError(f"k_min = {k_min} cannot exceed the target k = {k}")


def build(inputs: BuildInputs, out: str | os.PathLike[str], *, threads: int = 1) -> Provenance:
    """Build the LD Store of `inputs` at `out` and return its provenance (spec §7; ogld#5).

    Refuses an input with a missing call. Filters the Variant Set, then runs the PC Check
    (`docs/spec/build-inputs.md` §8): a PC is local, and dropped, when one 4 Mb window
    explains ≥ 10% of it, that share beats the window-side null, and rest share / window
    share ≤ 0.05. The covariates are the first up to k PCs kept; with fewer than k, those
    that survive; with fewer than k_min, a refusal naming the dropped PCs. At k = 0 no PC
    Check runs. It regresses the covariates out of the standardised dosages, writes the
    Near Band exactly and the Far Pairs and Partner Rows at their floors, and finishes through
    `StoreWriter.finish`, which validates the store. Provenance records the scan, the
    dropped PCs and the null the scan was judged against.

    `threads` is how many worker threads the build runs, and it bounds it: every parallel part,
    the PC Check's windows, the LD tiles and validation, runs on that many workers; BLAS runs
    one thread inside the build (threadpoolctl, from the build extra), and zarr's pool at most
    `threads`, whatever the caller's environment says. Provenance records it as
    `build.threads`.
    """
    cost = record.Cost.start()
    target = Path(out)
    if target.exists():
        raise InputError(f"{target} exists: a store is never written over another")
    if threads < 1:
        raise InputError(f"threads counts worker threads, so it must be at least 1: {threads}")
    with bounded(threads):
        return _build(inputs, target, threads, cost)


def _build(inputs: BuildInputs, target: Path, threads: int, cost: record.Cost) -> Provenance:
    """`build`'s passes, within its thread bound."""
    plan = passes.plan(inputs.pgen, inputs.keep, inputs.pcs if inputs.k else None, inputs.panel_id)
    settings = Settings(inputs.panel_id, inputs.mac_min, inputs.info_min, bool(inputs.k), threads)
    found = variant_set(plan.filesets, settings, plan.samples.n)
    chosen = passes.choose(plan, found, inputs.k, inputs.k_min)
    chromosomes = passes.drop_flat(plan, found, chosen.covariates)
    diagnostic.require_change_pairs([(c.name, c.position) for c in chromosomes], plan.panel)
    k = chosen.covariates.k
    rates = passes.chance(plan, chromosomes, chosen.covariates)
    writer = StoreWriter.create(
        target,
        variants=passes.variants_table(chromosomes),
        encoding=inputs.encoding(),
        near_bp=NEAR_BP,
        far_bp=FAR_BP,
        n=plan.samples.n,
        k=k,
        t_partner=inputs.t_partner,
        compress_workers=threads,
    )
    change = passes.write_ld(writer, plan, chromosomes, chosen.covariates, threads)
    made = record.Made(found.dropped, diagnostic.pc_diagnostic(rates, plan.samples.n, k, change))
    return writer.finish(
        record.sourced(
            _head(inputs, plan), _adjustment(inputs, plan, chosen), made, cost.record(threads)
        ),
        threads=threads,
    )


def _head(inputs: BuildInputs, plan: passes.Plan) -> record.Head:
    files = [path for f in plan.filesets for path in (f.pgen, f.pvar, f.psam)]
    return record.Head(
        record.source(inputs.genotype_source, [*files, inputs.keep]),
        {
            "id": inputs.panel_id,
            "ancestry": inputs.ancestry,
            "n": plan.samples.n,
            "samples_sha256": plan.samples.sha256(),
        },
        record.variant_filters(inputs.mac_min, inputs.info_min),
    )


def _adjustment(inputs: BuildInputs, plan: passes.Plan, chosen: passes.Choice) -> Adjustment:
    if plan.pcs is None or chosen.checked is None:
        return record.adjustment([], 0, inputs.k_min, None, None)
    assert inputs.pc_source is not None, "BuildInputs refuses k > 0 without a declared source"
    return record.adjustment(
        chosen.covariates.names,
        inputs.k,
        inputs.k_min,
        (plan.pcs.sha256, inputs.pc_source),
        chosen.checked.record,
    )


def build_genotypes(
    inputs: BuildInputs, out: str | os.PathLike[str], *, threads: int = 1
) -> Provenance:
    """Build a local sidecar over existing pgens, without materialising pairwise LD (ADR 0015).

    Variant Set selection, PC Check, sample ordering and FLAT drops are the LD builder's.
    The PC Diagnostic is null: computing the entire Near Band only for its diagnostic would
    defeat the sidecar's purpose; the selected covariates and PC Check remain recorded.
    """
    from ogld.genotypes import Genotypes, prepare
    from ogld.genotypes.sidecar import SidecarWriter, fileset_record, input_pin

    target = Path(out)
    if target.exists() or threads < 1:
        raise InputError(f"{target}: needs a new output directory and threads >= 1")
    cost = record.Cost.start()
    with bounded(threads):
        plan = passes.plan(
            inputs.pgen, inputs.keep, inputs.pcs if inputs.k else None, inputs.panel_id
        )
        settings = Settings(
            inputs.panel_id, inputs.mac_min, inputs.info_min, bool(inputs.k), threads
        )
        found = variant_set(plan.filesets, settings, plan.samples.n)
        chosen = passes.choose(plan, found, inputs.k, inputs.k_min)
        chromosomes = passes.drop_flat(plan, found, chosen.covariates)
        writer = SidecarWriter(
            target,
            passes.variants_table(chromosomes),
            chosen.covariates,
            inputs.encoding(),
            near_bp=NEAR_BP,
            far_bp=FAR_BP,
            t_partner=inputs.t_partner,
        )
        for chromosome in chromosomes:
            with Genotypes(plan.filesets[chromosome.fileset]) as genotypes:
                for first in range(0, len(chromosome), 1024):
                    rows = chromosome.rows[first : first + 1024]
                    block = prepare(
                        genotypes.rows(rows),
                        chosen.covariates,
                        chromosome.a1_is_alt[first : first + len(rows)],
                    )
                    writer.write(rows, chromosome.fileset, block)
        sourced = record.sourced(
            _head(inputs, plan),
            _adjustment(inputs, plan, chosen),
            record.Made(found.dropped, None),
            cost.record(threads),
        )
        filesets = [fileset_record(f, f.pvar, target) for f in plan.filesets]
        pins = [input_pin(inputs.keep, target)]
        if inputs.k and inputs.pcs is not None:
            pins.append(input_pin(inputs.pcs, target))
        return writer.finish(sourced, filesets, pins)


__all__ = ["BuildInputs", "build", "build_genotypes"]
