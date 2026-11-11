"""The build's passes over the genotypes, between `ogld.build.build` and the modules it uses.

1. `plan`: the samples, the filesets and the PC File, each refused if unusable.
2. `ogld.build.variant_set`: every record, once; the Variant Set and the PC Check's windows.
3. `choose`: the PC Check, and the covariates it keeps (k > 0).
4. `drop_flat`: the variants with no variance left after PC Adjustment, dropped and counted
   (build-inputs §6, filter 4). It needs the covariates, which need every chromosome's windows,
   and the Variant Set must be final before the store's variant table is written, so at k > 0
   it is a pass of its own.
5. `write_ld`: each chromosome's Near Band and Proxy Index, through `StoreWriter`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from ogld.build import pc_check, tiles
from ogld.build.adjust import Covariates, covariates, residuals, unit_vectors
from ogld.build.diagnostic import (
    CHANGE_BP,
    ChanceRates,
    ChangeHistogram,
    chance_rates,
    outside_regions,
)
from ogld.build.genotypes import Genotypes
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
from ogld.format import Codes, Counts, InputError, REncoding
from ogld.format.geometry import (
    chromosome_index,
    chromosome_ranges,
    global_rows,
    local_rows,
    windows,
)
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
    the_plan: Plan, chromosome: Chromosome
) -> Iterator[tuple[int, npt.NDArray[np.float64]]]:
    """A chromosome's genotypes in Variant Set order, a block at a time: (first row, dosages)."""
    with Genotypes(the_plan.filesets[chromosome.fileset]) as genotypes:
        for first, stop in windows(0, len(chromosome), CHUNK_RECORDS):
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


def _units(the_plan: Plan, chromosome: Chromosome, chosen: Covariates) -> npt.NDArray[np.float64]:
    """The chromosome's unit vectors after PC Adjustment by `chosen`, in Variant Set order."""
    out = np.empty((len(chromosome), the_plan.samples.n))
    for first, dosages in _blocks(the_plan, chromosome):
        stop = first + len(dosages)
        out[first:stop] = unit_vectors(dosages, chosen, chromosome.a1_is_alt[first:stop])
    return out


def _global(local: int, start: int) -> int:
    """A chromosome's local row as the Variant Set's global row (spec §2)."""
    return int(global_rows(local, start))


def _labels(chromosomes: list[Chromosome]) -> npt.NDArray[np.str_]:
    """The chromosome of each row of the Variant Set, in its order."""
    return np.concatenate([np.full(len(c), c.name) for c in chromosomes])


def chance(the_plan: Plan, chromosomes: list[Chromosome], chosen: Covariates) -> ChanceRates:
    """#405's chance-LD rates over the Variant Set (`ogld.build.diagnostic`)."""
    labels = _labels(chromosomes)
    ranges = list(chromosome_ranges(labels).values())
    stops = [stop for _, stop in ranges]
    unadjusted = covariates([], None, the_plan.samples.n)

    def units_of(rows: npt.NDArray[np.int64]) -> tuple[npt.NDArray[np.float64], ...]:
        both: tuple[list[npt.NDArray[np.float64]], ...] = ([], [])
        for i, c in enumerate(chromosomes):
            own = chromosome_index(stops, rows) == i
            mine = local_rows(rows[own], ranges[i][0])
            picked = c.without(~np.isin(np.arange(len(c)), mine))
            for out, cov in zip(both, (unadjusted, chosen), strict=True):
                out.append(_units(the_plan, picked, cov))
        return tuple(np.concatenate(out) for out in both)

    return chance_rates(labels, units_of)


@dataclass
class Written:
    """What `write_ld` wrote, counted for provenance, and the change in r it measured."""

    counts: Counts
    change: ChangeHistogram


def write_ld(
    writer: StoreWriter,
    the_plan: Plan,
    chromosomes: list[Chromosome],
    chosen: Covariates,
    floor: int,
    distances: tuple[int, int],
    threads: int,
    encoding: REncoding,
) -> Written:
    """Each chromosome's Near Band and Proxy Index, through `writer`, as `encoding`'s codes
    (spec §3, §4, §5)."""
    written = Written(
        {"near_pairs": 0, "far_candidates": 0, "far_pairs": 0, "index_entries": 0},
        ChangeHistogram(),
    )
    ranges = chromosome_ranges(_labels(chromosomes))
    for chromosome in chromosomes:
        start = ranges[chromosome.name][0]
        inputs = _inputs(the_plan, chromosome, chosen, floor, distances, encoding)
        band, rows = inputs.band, len(inputs.units)
        last, found = _write_band(writer, start, inputs, threads, written)
        # The unit vectors go before the chromosome's last rows complete its band, which the
        # writer then holds whole while it writes it (ogld#4): the two largest arrays a
        # chromosome makes are never held at once.
        del inputs
        writer.write_near(_global(last.first, start), last.lengths, last.codes)
        entries = tiles.Entries.of(*found)
        del found
        indptr, partner, codes = entries.both_directions(rows, start)
        writer.write_index(chromosome.name, indptr, partner, codes)
        _count(written, band, entries, len(partner))
    return written


def _inputs(
    the_plan: Plan,
    chromosome: Chromosome,
    chosen: Covariates,
    floor: int,
    distances: tuple[int, int],
    encoding: REncoding,
) -> tiles.Inputs:
    """What every tile of `chromosome` reads: its unit vectors, adjusted and (at k > 0) not."""
    adjusted = _units(the_plan, chromosome, chosen)
    unadjusted = None
    if chosen.k:
        unadjusted = _units(the_plan, chromosome, covariates([], None, the_plan.samples.n))
    band = tiles.Band.of(chromosome.position, *distances, CHANGE_BP)
    outside = outside_regions(chromosome.name, chromosome.position)
    return tiles.Inputs(adjusted, unadjusted, outside, band, floor, encoding)


_Found = tuple[list[npt.NDArray[np.int32]], list[npt.NDArray[np.int32]], list[Codes]]


def _write_band(
    writer: StoreWriter, start: int, inputs: tiles.Inputs, threads: int, written: Written
) -> tuple[tiles.Tile, _Found]:
    """Write every tile's Near Band rows but the last tile's, which is returned, and collect
    every tile's Proxy Index entries."""
    found: _Found = ([], [], [])
    held: tiles.Tile | None = None
    for tile in tiles.tiles(inputs, threads):
        if held is not None:
            writer.write_near(_global(held.first, start), held.lengths, held.codes)
        held = tile
        found[0].append(tile.lo)
        found[1].append(tile.hi)
        found[2].append(tile.entry_codes)
        written.change.add(tile.change)
        written.counts["near_pairs"] += len(tile.codes)
    assert held is not None, "a chromosome of the Variant Set has at least one variant"
    return held, found


def _count(written: Written, band: tiles.Band, entries: tiles.Entries, index_entries: int) -> None:
    """Add one chromosome to the counts provenance records (spec §7)."""
    written.counts["far_candidates"] += int((band.far_stop - band.near_stop).sum())
    far = entries.hi.astype(np.int64) >= band.near_stop[entries.lo]
    written.counts["far_pairs"] += int(far.sum())
    written.counts["index_entries"] += index_entries
