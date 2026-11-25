"""The Variant Set: every record judged, filtered and identified (build-inputs §2, §6, §7).

One pass over every record of every fileset, in pgen order, block by block:

1. **Missing calls first.** A block's genotypes are read and its missing calls counted before
   anything else is judged, and they are counted over the whole input, so the refusal names
   their number, the first variant by ID and the panel (build-inputs §7). Once one is found the
   rest of the input is only counted.
2. **Record rules.** Autosomes only (a record on X, Y, MT or any other label is refused, naming
   it); REF and ALT of A, C, G, T only; a chromosome in one fileset, contiguously.
3. **Filters**, in order, each dropped record counted under the first it fails: FILTER `PASS` or
   `.`; MAC ≥ `mac_min` on the kept samples; INFO > `info_min` when it is set; nonzero variance.
4. **Per chromosome:** the engine's identity in the sorted spelling an LD Store keeps until #462
   (`opengwasdb.variants.sorted_spelling.sorted_identity`), one record per ALID by that
   spelling's reversed-indel rule, and the engine's variant order (ld-store §2).

A record's REF and ALT are passed unchanged to the identity function, which neither trims nor
left-aligns (ADR 0089).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt
from ogquery.ld.format import InputError

from opengwasdb.ld.build import pc_check
from opengwasdb.ld.build.inputs import Fileset, Records, records
from opengwasdb.ld.genotypes import Genotypes
from opengwasdb.store_reader.identity import (
    AUTOSOMES,
    VariantNormalisationError,
    normalise_chromosome,
)
from opengwasdb.variants.axis import alid_sort_key
from opengwasdb.variants.normalise import chromosome_rank
from opengwasdb.variants.sorted_spelling import (
    SortedIdentity,
    resolve_sorted_reversed_indels,
    sorted_identity,
)

PASSING_FILTERS: Final = ("PASS", ".")
#: The reasons a record that passed the record rules is dropped, in the order they are judged.
REASONS: Final = ("filter", "mac", "info", "zero_variance", "duplicate_reversed_indel")
_ALLELE = re.compile(r"[ACGTacgt]+")
_ALLELES = re.compile(r"[ACGTacgt]+(?:\t[ACGTacgt]+)*")
_INFO_KEYS: Final = ("R2", "INFO")


@dataclass(frozen=True)
class Settings:
    """What the pass applies: the panel it names, and the filters' settings."""

    panel: str
    mac_min: int
    info_min: float | None
    #: Scan each chromosome for the PC Check's windows (k > 0), on this many threads.
    scan: bool
    threads: int = 1


@dataclass(frozen=True)
class Chromosome:
    """One chromosome's Variant Set, in the engine's order: where each variant is in its pgen,
    its identity, and A1's frequency."""

    name: str
    fileset: int
    rows: npt.NDArray[np.int64]
    ids: list[str]
    oriented: list[SortedIdentity]
    a1_frequency: npt.NDArray[np.float64]
    position: npt.NDArray[np.int64] = field(init=False)
    a1_is_alt: npt.NDArray[np.bool_] = field(init=False)

    def __post_init__(self) -> None:
        position = np.array([o.position for o in self.oriented], dtype=np.int64)
        object.__setattr__(self, "position", position)
        is_alt = np.array([o.a1_is_alt for o in self.oriented], dtype=np.bool_)
        object.__setattr__(self, "a1_is_alt", is_alt)

    def __len__(self) -> int:
        return len(self.rows)

    def without(self, drop: npt.NDArray[np.bool_]) -> Chromosome:
        """The chromosome without the variants `drop` marks."""
        keep = np.flatnonzero(~drop)
        return Chromosome(
            self.name,
            self.fileset,
            self.rows[keep],
            [self.ids[i] for i in keep],
            [self.oriented[i] for i in keep],
            self.a1_frequency[keep],
        )


@dataclass
class _Pending:
    """One chromosome's passing records while its fileset is read."""

    name: str
    fileset: int
    rows: list[npt.NDArray[np.int64]] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    alleles: list[tuple[str, int, str, str]] = field(default_factory=list)
    alt_sum: list[npt.NDArray[np.float64]] = field(default_factory=list)
    #: Which kept records are common (`pc_check.common`), all the PC Check keeps of their
    #: genotypes: they are re-read per window (ogld#25), when the build scans.
    common: list[npt.NDArray[np.bool_]] = field(default_factory=list)


@dataclass
class Pass:
    """The pass's running state: what it found, and what it dropped."""

    settings: Settings
    samples: int
    dropped: dict[str, int] = field(default_factory=lambda: dict.fromkeys(REASONS, 0))
    missing_calls: int = 0
    first_missing: str = ""
    chromosomes: list[Chromosome] = field(default_factory=list)
    windows: dict[str, pc_check.Windows] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)
    pending: _Pending | None = None
    #: The fileset being read, open: the PC Check re-reads its windows' genotypes from it.
    reader: Genotypes | None = None

    @property
    def panel(self) -> str:
        return self.settings.panel


def variant_set(filesets: list[Fileset], settings: Settings, samples: int) -> Pass:
    """Judge every record of `filesets` and return the Variant Set, chromosome by chromosome
    in the engine's order, with the counts of what was dropped. Refuses a missing call, after
    counting them all."""
    state = Pass(settings, samples)
    for index, fileset in enumerate(filesets):
        with Genotypes(fileset) as genotypes:
            state.reader = genotypes
            read = _read_fileset(state, index, fileset, genotypes)
            if read != genotypes.variant_count:
                raise InputError(
                    f"panel {state.panel}: {fileset.pvar.name} has {read} records but "
                    f"{fileset.pgen.name} has {genotypes.variant_count} variants"
                )
            if state.missing_calls:
                state.pending = None  # a missing call is refused before anything it would hide
            else:
                _close_pending(state)
            state.reader = None
    _refuse_missing_calls(state)
    if not state.chromosomes:
        raise InputError(f"panel {state.panel}: no variant passes the filters (build-inputs §6)")
    state.chromosomes.sort(key=lambda c: chromosome_rank(c.name))
    return state


def _read_fileset(state: Pass, index: int, fileset: Fileset, genotypes: Genotypes) -> int:
    """Judge one fileset's records, block by block; return how many it has."""
    read = 0
    info = state.settings.info_min is not None
    for block in records(fileset.pvar, state.panel, info):
        g = genotypes.range(block.first, block.first + len(block))
        read += len(block)
        if _count_missing(state, block, g) or state.missing_calls:
            continue
        for start, stop, pending in _runs(state, block):
            _judge(state, index, block, g[start:stop], start, stop, pending)
    return read


def _count_missing(state: Pass, block: Records, g: npt.NDArray[np.float64]) -> int:
    """Count the block's missing calls into `state`; remember the first by ID."""
    # One pass finds the records with a missing call (−9); only those are counted cell by cell.
    holed = np.flatnonzero(g.min(axis=1) < 0)
    found = int((g[holed] < 0).sum())
    if found and not state.missing_calls:
        state.first_missing = block.ids[int(holed[0])]
    state.missing_calls += found
    return found


def _refuse_missing_calls(state: Pass) -> None:
    if state.missing_calls:
        raise InputError(
            f"panel {state.panel} has {state.missing_calls} missing genotype(s) among its kept "
            f"samples, the first in variant {state.first_missing}: a build refuses incomplete "
            "genotypes rather than impute them (build-inputs §7)"
        )


def _chromosome(label: str, record: str, panel: str) -> str:
    """The engine's label of a record's chromosome, which must be an autosome (build-inputs §2)."""
    try:
        name = normalise_chromosome(label)
    except VariantNormalisationError as refused:
        raise InputError(f"panel {panel}: record {record}: {refused}") from refused
    if name not in AUTOSOMES:
        raise InputError(
            f"panel {panel}: record {record} is on chromosome {label}; only autosomes are built "
            "(build-inputs §2; a male's X and Y genotypes have no diploid dosage rule)"
        )
    return name


def _runs(state: Pass, block: Records) -> Iterator[tuple[int, int, _Pending]]:
    """The block's runs of one chromosome: (start, stop, the chromosome's pending records),
    each an autosome that no earlier fileset or run has held."""
    labels = block.chrom
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            name = _chromosome(labels[start], block.ids[start], state.panel)
            yield start, i, _enter(state, name)
            start = i


def _enter(state: Pass, name: str) -> _Pending:
    """Make `name` the chromosome being read, closing the one before; a chromosome met again
    after another is refused (build-inputs §2: one fileset, contiguous)."""
    if state.pending is not None and state.pending.name == name:
        return state.pending
    if name in state.seen:
        raise InputError(
            f"panel {state.panel}: chromosome {name} appears in two places; a chromosome's "
            "records must be one contiguous run of one fileset (build-inputs §2)"
        )
    _close_pending(state)
    state.seen.add(name)
    state.pending = _Pending(name, -1)
    return state.pending


def _judge(
    state: Pass,
    index: int,
    block: Records,
    run: npt.NDArray[np.float64],
    start: int,
    stop: int,
    pending: _Pending,
) -> None:
    """Apply the record rules and filters to the block's records start … stop − 1, whose
    genotypes are `run`, and keep those that pass in `pending`."""
    pending.fileset = index
    _check_alleles(block, start, stop, state.panel)
    picked = np.flatnonzero(_filters(state, block, run, start, stop))
    pending.rows.append(block.first + start + picked.astype(np.int64))
    for i in picked:
        at = start + int(i)
        pending.ids.append(block.ids[at])
        pending.alleles.append((pending.name, int(block.pos[at]), block.ref[at], block.alt[at]))
    pending.alt_sum.append(run[picked].sum(axis=1))
    if state.settings.scan:
        pending.common.append(pc_check.common(run[picked]))


def _check_alleles(block: Records, start: int, stop: int, panel: str) -> None:
    """REF and ALT are A, C, G and T in either case (build-inputs §2); anything else, such as
    `*`, `N`, a symbolic allele or `.`, is refused naming the record."""
    if _ALLELES.fullmatch("\t".join(block.ref[start:stop] + block.alt[start:stop])):
        return  # every allele of the run is A, C, G and T: one match, not one per record
    for at in range(start, stop):
        if not (_ALLELE.fullmatch(block.ref[at]) and _ALLELE.fullmatch(block.alt[at])):
            raise InputError(
                f"panel {panel}: record {block.ids[at]} has REF {block.ref[at]!r} and ALT "
                f"{block.alt[at]!r}; alleles must be A, C, G and T (build-inputs §2)"
            )


def _filters(
    state: Pass, block: Records, g: npt.NDArray[np.float64], start: int, stop: int
) -> npt.NDArray[np.bool_]:
    """Which records pass filters 1–4, counting each dropped one under the first it fails."""
    s = state.settings
    alt_sum = g.sum(axis=1)
    tests: list[tuple[str, Callable[[], npt.NDArray[np.bool_]]]] = [
        ("filter", lambda: _filter_column(block, start, stop, s.panel)),
        ("mac", lambda: np.minimum(alt_sum, 2 * state.samples - alt_sum) >= s.mac_min),
        ("info", lambda: _info(block, start, stop, s)),
        ("zero_variance", lambda: _varies(g, keep)),
    ]
    keep = np.ones(stop - start, dtype=np.bool_)
    for reason, test in tests:
        passed = test()
        state.dropped[reason] += int((keep & ~passed).sum())
        keep &= passed
    return keep


def _varies(g: npt.NDArray[np.float64], keep: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Filter 4 on the records still kept (most fail MAC first): nonzero genotype variance."""
    varies = np.ones(len(g), dtype=np.bool_)
    kept = g[keep]
    varies[keep] = kept.max(axis=1) > kept.min(axis=1)
    return varies


def _filter_column(block: Records, start: int, stop: int, panel: str) -> npt.NDArray[np.bool_]:
    """Filter 1: FILTER is PASS or `.`. The column must be there, even when every value is `.`,
    so that "nothing to filter" is said by the file, not assumed (build-inputs §2)."""
    if block.filter is None:
        raise InputError(
            f"panel {panel}: the pvar has no FILTER column; convert with "
            "`--make-pgen pvar-cols=filter` (build-inputs §2, §3)"
        )
    return np.array([value in PASSING_FILTERS for value in block.filter[start:stop]])


def _info(block: Records, start: int, stop: int, s: Settings) -> npt.NDArray[np.bool_]:
    """Filter 3: INFO > `info_min`, from the pvar INFO subfield R2 (Minimac) or INFO (IMPUTE).
    With `info_min` set, a pvar without them is refused: a sequenced panel passes None."""
    if s.info_min is None:
        return np.ones(stop - start, dtype=np.bool_)
    if block.info is None:
        raise InputError(
            f"panel {s.panel}: info_min = {s.info_min} needs an INFO column with R2 or INFO, and "
            "the pvar has none; a sequenced panel sets info_min to None (build-inputs §6)"
        )
    scores = [_info_score(block.info[at], block.ids[at], s.panel) for at in range(start, stop)]
    return np.array(scores, dtype=np.float64) > s.info_min


def _info_score(info: str, record: str, panel: str) -> float:
    """A record's INFO score: its R2 or its INFO subfield, exactly one of them."""
    fields = dict(item.split("=", 1) for item in info.split(";") if "=" in item)
    found = [key for key in _INFO_KEYS if key in fields]
    if len(found) != 1:
        raise InputError(
            f"panel {panel}: record {record}'s INFO must hold one of R2 or INFO, not "
            f"{' and '.join(found) or 'neither'} (build-inputs §6)"
        )
    return _number(fields[found[0]], f"record {record}'s {found[0]}", panel)


def _number(text: str, what: str, panel: str) -> float:
    try:
        return float(text)
    except ValueError as bad:
        raise InputError(f"panel {panel}: {what} is not a number: {text!r}") from bad


def _close_pending(state: Pass) -> None:
    """Turn the chromosome being read into its Variant Set (identity, one record per ALID,
    the engine's order) and add it, with its PC Check windows when the build scans."""
    pending, state.pending = state.pending, None
    if pending is None or not pending.ids:
        return
    oriented = [
        _orient(record, i, pending, state.panel) for i, record in enumerate(pending.alleles)
    ]
    kept = _one_per_alid(pending, oriented, state)
    order = sorted(kept, key=lambda i: alid_sort_key(oriented[i].alid))
    found = _chromosome_of(pending, oriented, order, state.samples)
    state.chromosomes.append(found)
    if state.settings.scan:
        _scan(state, pending, found, order)


def _scan(state: Pass, pending: _Pending, found: Chromosome, order: list[int]) -> None:
    """The PC Check's windows of the chromosome `found`, reading each window's genotypes from
    the open fileset when it is scanned, so no chromosome's genotypes are held (ogld#25)."""
    reader = state.reader
    if reader is None:  # pragma: no cover - the pass opens the fileset before it closes a run
        raise RuntimeError("the PC Check scans a chromosome while no fileset is open")
    common = np.concatenate(pending.common)[order]

    def read(rows: npt.NDArray[np.intp]) -> npt.NDArray[np.float32]:
        return reader.rows(found.rows[rows]).astype(np.float32)

    state.windows[found.name] = pc_check.stream_windows(
        found.name,
        found.position,
        common,
        read,
        state.samples,
        threads=state.settings.threads,
    )


def _chromosome_of(
    pending: _Pending, oriented: list[SortedIdentity], order: list[int], samples: int
) -> Chromosome:
    """The records `order` picks, in that order, as a chromosome's Variant Set."""
    frequency = np.concatenate(pending.alt_sum)[order] / (2 * samples)
    is_alt = np.array([oriented[i].a1_is_alt for i in order], dtype=np.bool_)
    return Chromosome(
        pending.name,
        pending.fileset,
        np.concatenate(pending.rows)[order].astype(np.int64),
        [pending.ids[i] for i in order],
        [oriented[i] for i in order],
        np.where(is_alt, frequency, 1 - frequency),
    )


def _orient(
    record: tuple[str, int, str, str], i: int, pending: _Pending, panel: str
) -> SortedIdentity:
    try:
        return sorted_identity(*record)
    except VariantNormalisationError as refused:
        raise InputError(f"panel {panel}: record {pending.ids[i]}: {refused}") from refused


def _one_per_alid(pending: _Pending, oriented: list[SortedIdentity], state: Pass) -> list[int]:
    """The records kept when several share a sorted ALID: of a reversed-indel pair, the one
    whose REF the ALID spells first (`resolve_sorted_reversed_indels`, the sorted spelling's rule;
    build-inputs §6), the other counted as `duplicate_reversed_indel`. Any other sharing is
    refused, naming them."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, o in enumerate(oriented):
        groups[o.alid].append(i)
    shared = [(alid, members) for alid, members in groups.items() if len(members) > 1]
    dropped = {i for alid, members in shared for i in _resolve(alid, members, pending, state)}
    state.dropped["duplicate_reversed_indel"] += len(dropped)
    return [i for i in range(len(oriented)) if i not in dropped]


def _resolve(alid: str, members: list[int], pending: _Pending, state: Pass) -> list[int]:
    """Which of `members`, records sharing `alid`, the sorted spelling's rule drops."""
    try:
        _, lost = resolve_sorted_reversed_indels([pending.alleles[i] for i in members])
    except VariantNormalisationError as refused:
        names = ", ".join(pending.ids[i] for i in members)
        raise InputError(
            f"panel {state.panel}: records {names} share ALID {alid} and are not a "
            f"reversed-indel pair: {refused} (build-inputs §6)"
        ) from refused
    return [members[i] for i in lost]
