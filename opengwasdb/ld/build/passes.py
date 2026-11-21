"""The build's passes over the genotypes, between `ogld.build.build` and the modules it uses.

1. `plan`: the samples, the filesets and the PC File, each refused if unusable.
2. `ogld.build.variant_set`: every record, once; the Variant Set and the PC Check's windows.
3. `choose`: the PC Check, and the covariates it keeps (k > 0).
4. `drop_flat`: the variants with no variance left after PC Adjustment, dropped and counted
   (build-inputs §6, filter 4). It needs the covariates, which need every chromosome's windows,
   and the Variant Set must be final before the store's variant table is written, so at k > 0
   it is a pass of its own.
5. `write_ld`: each chromosome's r, tile by tile, through `StoreWriter`, which makes the store of
   it (ogld#17).
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from ogld.build import pc_check, tiles
from ogld.build.diagnostic import (
    CHANGE_BP,
    ChanceRates,
    ChangeHistogram,
    chance_rate,
    draw_pairs,
    draw_variants,
    outside_regions,
)
from ogld.build.inputs import (
    CHUNK_RECORDS,
    Fileset,
    PcFile,
    Samples,
    fileset,
    psam_iids,
    read_keep,
    read_pc_file,
)
from ogld.build.variant_set import Chromosome, Pass
from ogld.format import InputError
from ogld.format.geometry import (
    chromosome_index,
    chromosome_ranges,
    local_rows,
    windows,
)
from ogld.genotypes import Covariates, Genotypes, covariates, prepare, r, residuals
from ogld.identity import Orientation
from ogld.store import StoreWriter, Variants

STRINGS: Final = np.dtypes.StringDType()


@dataclass(frozen=True)
class Plan:
    """What a build reads: the panel's name, its samples, its filesets and its PC File."""

    panel: str
    samples: Samples
    filesets: list[Fileset]
    pcs: PcFile | None


def plan(pgens: tuple[Path, ...], keep: Path, pcs: Path | None, panel: str) -> Plan:
    """The build's inputs, read and checked; `pcs` is None when no PC File is read (k = 0)."""
    kept = read_keep(keep, panel)
    first = fileset(pgens[0], kept, None, panel)
    iids = psam_iids(first.psam, panel)
    order = [iids[i] for i in first.subset]
    sets = [first, *(fileset(p, kept, order, panel) for p in pgens[1:])]
    names = [s.pgen.name for s in sets]
    if len(set(names)) != len(names):
        raise InputError(f"panel {panel}: two pgens share a file name: {', '.join(names)}")
    samples = Samples(tuple(order))
    return Plan(panel, samples, sets, None if pcs is None else read_pc_file(pcs, samples, panel))


@dataclass(frozen=True)
class Choice:
    """The covariates PC Adjustment regresses out, and the PC Check that chose them."""

    covariates: Covariates
    checked: pc_check.Checked | None


def choose(the_plan: Plan, found: Pass, k_target: int, k_min: int) -> Choice:
    """The PC Check over every chromosome's windows, and the first up to `k_target` kept PCs."""
    n = the_plan.samples.n
    if the_plan.pcs is None:
        return Choice(covariates([], None, n), None)
    windows = pc_check.merge([found.windows[c.name] for c in found.chromosomes])
    try:
        checked = pc_check.check(
            windows, the_plan.pcs.values, names=the_plan.pcs.names, k_target=k_target, k_min=k_min
        )
    except InputError as refused:
        raise InputError(f"panel {the_plan.panel}: {refused}") from refused
    columns = [the_plan.pcs.names.index(name) for name in checked.covariates]
    return Choice(covariates(checked.covariates, the_plan.pcs.values[:, columns], n), checked)


def _blocks(
    the_plan: Plan, chromosome: Chromosome, rows: int = CHUNK_RECORDS
) -> Iterator[tuple[int, npt.NDArray[np.float64]]]:
    """A chromosome's genotypes in Variant Set order, `rows` at a time: (first row, dosages)."""
    with Genotypes(the_plan.filesets[chromosome.fileset]) as genotypes:
        for first, stop in windows(0, len(chromosome), rows):
            yield first, genotypes.rows(chromosome.rows[first:stop])


def drop_flat(the_plan: Plan, found: Pass, chosen: Covariates) -> list[Chromosome]:
    """The Variant Set without the variants PC Adjustment leaves no variance in, counted as
    `zero_variance` (build-inputs §6, filter 4)."""
    if not chosen.k:
        return found.chromosomes
    kept = []
    for chromosome in found.chromosomes:
        flat = np.zeros(len(chromosome), dtype=np.bool_)
        for first, dosages in _blocks(the_plan, chromosome):
            flat[first : first + len(dosages)] = residuals(dosages, chosen)[1]
        found.dropped["zero_variance"] += int(flat.sum())
        if not flat.all():
            kept.append(chromosome.without(flat))
    if not kept:
        raise InputError(f"panel {the_plan.panel}: no variant has variance after PC Adjustment")
    return kept


def variants_table(chromosomes: list[Chromosome]) -> Variants:
    """The store's variant table (spec §2): the chromosomes in order, strings of any length."""
    oriented = [o for c in chromosomes for o in c.oriented]
    return Variants(
        chrom=_strings(oriented, "chrom"),
        position=np.concatenate([c.position for c in chromosomes]),
        alid=_strings(oriented, "alid"),
        a1=_strings(oriented, "a1"),
        a2=_strings(oriented, "a2"),
        a1_frequency=np.concatenate([c.a1_frequency for c in chromosomes]),
    )


def _strings(oriented: list[Orientation], name: str) -> npt.NDArray[np.str_]:
    """One identity column as numpy's variable-width strings: a fixed-width array would give
    every one of millions of ALIDs the width of the longest, an indel of up to 1,000 bases."""
    column: npt.NDArray[np.str_] = np.array([getattr(o, name) for o in oriented], dtype=STRINGS)
    return column


def _labels(chromosomes: list[Chromosome]) -> npt.NDArray[np.str_]:
    """The chromosome of each row of the Variant Set, in its order."""
    return np.concatenate([np.full(len(c), c.name) for c in chromosomes])


def chance(the_plan: Plan, chromosomes: list[Chromosome], chosen: Covariates) -> ChanceRates:
    """#405's chance-LD rates over the Variant Set (`ogld.build.diagnostic`)."""
    labels = _labels(chromosomes)
    if len(set(labels.tolist())) < 2:
        return ChanceRates(0, None, None)
    used = draw_variants(len(labels))
    first, second = draw_pairs(labels[used])
    calls, signs = _chance_calls(the_plan, chromosomes, used)
    rates = []
    for cov in (covariates([], None, the_plan.samples.n), chosen):
        block = prepare(calls, cov, signs)
        rates.append(chance_rate(r(block, block)[first, second]))
    return ChanceRates(len(first), rates[0], rates[1])


def _chance_calls(
    the_plan: Plan, chromosomes: list[Chromosome], rows: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    ranges = list(chromosome_ranges(_labels(chromosomes)).values())
    stops = [stop for _, stop in ranges]
    calls: list[npt.NDArray[np.float64]] = []
    signs: list[npt.NDArray[np.bool_]] = []
    for i, c in enumerate(chromosomes):
        own = chromosome_index(stops, rows) == i
        mine = local_rows(rows[own], ranges[i][0])
        picked = c.without(~np.isin(np.arange(len(c)), mine))
        calls.extend(dosages for _, dosages in _blocks(the_plan, picked))
        signs.append(picked.a1_is_alt)
    return np.concatenate(calls), np.concatenate(signs)


def write_ld(
    writer: StoreWriter,
    the_plan: Plan,
    chromosomes: list[Chromosome],
    chosen: Covariates,
    threads: int,
) -> ChangeHistogram:
    """Each chromosome's r, tile by tile, through `writer`, which makes the store of it (spec §3,
    §4, §5); return the change in r PC Adjustment made, for the PC Diagnostic (§7)."""
    change = ChangeHistogram()
    ranges = chromosome_ranges(_labels(chromosomes))
    for chromosome in chromosomes:
        inputs = tiles.Inputs(
            tiles.Band.of(chromosome.position, writer.far_bp, CHANGE_BP),
            outside_regions(chromosome.name, chromosome.position),
            ranges[chromosome.name][0],
            writer.cut,
        )
        units = _genotype_blocks(the_plan, chromosome, chosen)
        try:
            for cut, changed in tiles.tiles(inputs, units, threads):
                writer.write(cut)
                change.add(changed)
        finally:
            units.close()
    return change


def _genotype_blocks(
    the_plan: Plan, chromosome: Chromosome, chosen: Covariates
) -> Generator[tiles.GenotypeBlock, None, None]:
    """Prepared genotypes, a block at a time, in Variant Set order: after PC
    Adjustment by `chosen`, and at k > 0 also unadjusted, for the PC Diagnostic."""
    plain = covariates([], None, the_plan.samples.n) if chosen.k else None
    for first, dosages in _blocks(the_plan, chromosome, tiles.UNIT_ROWS):
        a1_is_alt = chromosome.a1_is_alt[first : first + len(dosages)]
        adjusted = prepare(dosages, chosen, a1_is_alt)
        unadjusted = None if plain is None else prepare(dosages, plain, a1_is_alt)
        yield first, adjusted, unadjusted
