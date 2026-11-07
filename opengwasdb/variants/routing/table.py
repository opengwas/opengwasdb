"""Pass 2's lookup: each source key's place on the Variant Index, fork-safe.

A `RouteTable` is built once, in the parent, from a routing's site map and
the axes a build stores. It holds numpy arrays and one ``bytes`` only -- each
Routing Key as a sorted ``uint64`` (`IntegerKeyCodec`), a parallel target,
axis and re-orientation array, and the full keys of the entries whose integer
is a hash -- so a forked Pass 2 worker searches it without touching the
per-element reference counts a dict or an object array would copy page by
page (issue 043 item 2). A hash only narrows the search: a hashed row is
matched to an entry only when their full Routing Keys are equal (issue #316).

`RouteTable.route_column` streams one Analysis's associations in fixed-size
batches, so a worker never holds a genome-wide source as Python lists, and
returns each axis's rows with each index's duplicates resolved
(`collapse_duplicates`): rows that agree collapse to the first, and rows that
disagree are all dropped, since no row says which is right (ADR 0072, D5). An
indel's ALID spells its reference allele first (ADR 0079), so an insertion and
a deletion at one position reach two places.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from opengwasdb.readers import orientation
from opengwasdb.readers.interface import ImputationScoreStatus, ReaderAssociation
from opengwasdb.variants.routing import keys as routing_keys
from opengwasdb.variants.routing.keys import (
    IntegerKeyCodec,
    IntegerKeys,
    KeyHash,
    SourceKey,
    is_hashed,
    routing_key,
)

log = logging.getLogger(__name__)

# Streamed in fixed-size batches so a worker never materialises a whole
# (genome-wide) source as Python lists at once -- peak per-worker memory is one
# batch of columns plus the accumulated matched rows (issue 043).
_RESOLVE_BATCH = 250_000

#: Sites encoded at a time while a table is built, so the parent never holds
#: a genome's sites as four more lists at once.
_ENCODE_BATCH = 1 << 20

#: The seven per-association lists one batch is read into: chromosome,
#: position, ref, alt, z, se and EAF (NaN where the source reports none).
_Columns = tuple[list[str], list[int], list[str], list[str], list[float], list[float], list[float]]


def _empty_index() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


def _empty_stat() -> np.ndarray:
    return np.empty(0, dtype=np.float32)


@dataclass(frozen=True)
class AssociationBatch:
    """One batch of a Source Reader's associations, before routing.

    In the reader's orientation: ``z`` and ``eaf`` are the source's own, not
    yet re-oriented by a reverse-strand lift. ``eaf`` and
    ``imputation_score`` are NaN where the source reports none. What an
    ``admit`` callable judges, so a row it refuses is never matched.
    """

    chromosome: Sequence[str]
    position: Sequence[int]
    ref: Sequence[str]
    alt: Sequence[str]
    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray
    imputation_score: np.ndarray
    imputation_status: np.ndarray

    def __post_init__(self) -> None:
        # One row filtered against another batch's mask either raises or, worse,
        # silently keeps the wrong rows, so a ragged batch is refused (stores #175).
        lengths = {len(values) for values in vars(self).values()}
        if len(lengths) != 1:
            raise ValueError(
                f"association batch columns hold {sorted(lengths)} row(s); every "
                "column must hold the batch's rows"
            )

    def __len__(self) -> int:
        return len(self.z)


#: Says which rows of one batch to route; True keeps a row.
Admit = Callable[[AssociationBatch], np.ndarray]


@dataclass(frozen=True)
class Placed:
    """One axis's rows for one Analysis, each index's duplicates resolved.

    ``index`` is a position on that axis, in the stream order of the kept
    rows. ``z`` and ``eaf`` follow the stored alleles; ``se`` is the
    source's. An index more than one association reached keeps the first of
    them when all agree on z and SE, and ``collapsed`` counts the rest. When
    they disagree none is kept: ``conflicting`` holds those indices, and
    ``dropped`` the index of every row dropped for them, in stream order.
    """

    index: np.ndarray
    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray
    conflicting: np.ndarray = field(default_factory=_empty_index)
    dropped: np.ndarray = field(default_factory=_empty_index)
    collapsed: int = 0


@dataclass(frozen=True)
class Unmatched:
    """The associations no axis holds, by routing key, in stream order.

    Kept only by a table built with ``keep_unmatched``; otherwise empty, and
    only counted. Not deduplicated, and in the reader's orientation.
    ``conflicting`` names the keys more than one kept association reached
    with a different z or SE, by `collapse_duplicates`' rule, as
    `Placed.conflicting` names an axis's indices; all their rows are still
    here.
    """

    keys: Sequence[str]
    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray
    conflicting: Sequence[str] = ()


@dataclass(frozen=True)
class RoutedColumn:
    """One Analysis routed: a `Placed` per axis, and what no axis holds."""

    placed: tuple[Placed, ...]
    unmatched: Unmatched
    n_unmatched: int


@dataclass(frozen=True)
class Duplicates:
    """How one stream's rows for repeated places resolve.

    ``keep`` holds the positions kept, in stream order: each place's first row,
    unless the place conflicts. ``conflicting`` holds, sorted, the places whose
    rows disagree on z or SE, and ``dropped`` the position of every row at
    one, in stream order. ``collapsed`` counts the rows that agreed with the
    row their place kept and so are not kept as rows of their own.
    """

    keep: np.ndarray
    conflicting: np.ndarray
    dropped: np.ndarray
    collapsed: int


def collapse_duplicates(place: np.ndarray, z: np.ndarray, se: np.ndarray) -> Duplicates:
    """Resolve the rows that share a place: agreeing ones collapse, conflicting ones go.

    Rows agree when z and SE are equal, NaN agreeing with NaN; the frequency is
    annotation, not the finding (ADR 0036), so a pair that differs only there
    is one association, and its first row's frequency is kept. A place whose
    rows disagree keeps none of them, because nothing in the rows says which
    is right (ADR 0072, D5). An insertion and a deletion at one position are
    two places, since an indel's ALID spells its reference allele first (ADR
    0079).
    """
    # One stable sort finds the repeated places; only their rows are compared,
    # so a column with few duplicates costs what last-wins cost, not a gather
    # of every row (#357: ~340 MB a worker on a 21 M-row FinnGen column).
    n = len(place)
    order = np.argsort(place, kind="stable")
    ordered = place[order]
    repeats = ordered[1:] == ordered[:-1]
    if not repeats.any():
        return Duplicates(np.arange(n), place[:0], np.empty(0, dtype=np.intp), 0)
    shared = np.zeros(n, dtype=bool)
    shared[1:] |= repeats
    shared[:-1] |= repeats
    at = np.flatnonzero(shared)
    rows, places = order[at], ordered[at]
    starts = np.concatenate(([True], places[1:] != places[:-1]))
    group = np.cumsum(starts) - 1
    firsts = rows[starts]  # the stable sort keeps each place's rows in stream order
    conflicted = _disagreeing(group, rows, firsts, z, se)
    dropped = np.sort(rows[conflicted[group]])
    keep = np.ones(n, dtype=bool)
    keep[rows] = False
    keep[firsts[~conflicted]] = True
    collapsed = len(rows) - len(dropped) - int((~conflicted).sum())
    return Duplicates(np.flatnonzero(keep), places[starts][conflicted], dropped, collapsed)


def _disagreeing(
    group: np.ndarray, rows: np.ndarray, firsts: np.ndarray, z: np.ndarray, se: np.ndarray
) -> np.ndarray:
    """Per place, whether any of its rows disagrees with its first on z or SE."""
    origin = firsts[group]
    agrees = _equal(z[rows], z[origin]) & _equal(se[rows], se[origin])
    disagreeing = np.zeros(len(firsts), dtype=bool)
    disagreeing[group[~agrees]] = True
    return disagreeing


def _equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    same: np.ndarray = (a == b) | (np.isnan(a) & np.isnan(b))
    return same


def _placed(index: np.ndarray, z: np.ndarray, se: np.ndarray, eaf: np.ndarray) -> Placed:
    """One axis's rows with each index's duplicates resolved, and the conflicts named."""
    if len(index) == 0:
        return Placed(index, z, se, eaf)
    found = collapse_duplicates(index, z, se)
    keep = found.keep
    return Placed(
        index[keep], z[keep], se[keep], eaf[keep],
        found.conflicting, index[found.dropped], found.collapsed,
    )


class _Collected:
    """One column's matched parts per axis, and its unmatched rows."""

    def __init__(self, n_axes: int, keep_unmatched: bool) -> None:
        self.parts: list[list[tuple[np.ndarray, ...]]] = [[] for _ in range(n_axes)]
        self.keep_unmatched = keep_unmatched
        self.unmatched_keys: list[str] = []
        self.unmatched_parts: list[tuple[np.ndarray, ...]] = []
        self.n_unmatched = 0

    def add_placed(self, axis: int, part: tuple[np.ndarray, ...]) -> None:
        if len(part[0]):
            self.parts[axis].append(part)

    def add_unmatched(self, keys: list[str], part: tuple[np.ndarray, ...]) -> None:
        self.unmatched_keys.extend(keys)
        if len(part[0]):
            self.unmatched_parts.append(part)

    def column(self) -> RoutedColumn:
        statistics = (_empty_stat(), _empty_stat(), _empty_stat())
        placed = tuple(
            _placed(*_joined(parts, (_empty_index(), *statistics))) for parts in self.parts
        )
        z, se, eaf = _joined(self.unmatched_parts, statistics)
        keys = self.unmatched_keys
        return RoutedColumn(
            placed=placed,
            unmatched=Unmatched(keys, z, se, eaf, _conflicting_keys(keys, (z, se))),
            n_unmatched=self.n_unmatched,
        )


def _conflicting_keys(keys: list[str], statistics: tuple[np.ndarray, np.ndarray]) -> list[str]:
    """The repeated keys whose rows disagree, by the rule placed indices meet,
    in the order each first appeared.

    A column's unmatched keys are nearly always distinct, so the common case
    costs one set; only a repeat numbers the keys to compare their rows.
    """
    if len(set(keys)) == len(keys):
        return []
    number: dict[str, int] = {}
    codes = np.fromiter(
        (number.setdefault(key, len(number)) for key in keys), dtype=np.int64, count=len(keys)
    )
    by_code = list(number)
    conflicting = collapse_duplicates(codes, *statistics).conflicting
    return [by_code[int(code)] for code in conflicting]


def _joined(
    parts: list[tuple[np.ndarray, ...]], empty: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, ...]:
    """Join a column's parts array by array, or ``empty`` when there are none."""
    if not parts:
        return empty
    return tuple(np.concatenate(column) for column in zip(*parts, strict=True))


def _retain_rows(columns: Sequence[list[Any]], keep: np.ndarray) -> None:
    """Keep ``keep``'s rows in every list of one batch, in place.

    The lists are positionally parallel, so a refused row leaves all of them
    together: a filter that kept a coordinate and dropped its z would route
    an association the caller refused.
    """
    if bool(keep.all()):
        return
    indices = np.flatnonzero(keep).tolist()
    for values in columns:
        values[:] = [values[index] for index in indices]


def _admitted(
    admit: Admit, columns: _Columns, scores: list[float | None], statuses: list[Any]
) -> np.ndarray:
    """``admit``'s keep mask for one batch: a boolean array with one entry per row.

    Nothing else is coerced into one. A NaN or a number is not a decision to
    keep a row, so a mask that is not boolean is refused rather than read.
    ``scores`` and ``statuses`` are empty when no score was collected, and the
    batch then carries every row's score as undeclared.
    """
    chroms, poss, refs, alts, zs, ses, eafs = columns
    if statuses:
        imputation_score = np.asarray(scores, dtype=np.float64)
        imputation_status = np.asarray(statuses, dtype=object)
    else:
        imputation_score = np.full(len(zs), np.nan)
        imputation_status = np.full(len(zs), ImputationScoreStatus.UNDECLARED, dtype=object)
    batch = AssociationBatch(
        chromosome=chroms,
        position=poss,
        ref=refs,
        alt=alts,
        z=np.asarray(zs, dtype=np.float64),
        se=np.asarray(ses, dtype=np.float64),
        eaf=np.asarray(eafs, dtype=np.float64),
        imputation_score=imputation_score,
        imputation_status=imputation_status,
    )
    keep = admit(batch)
    if not isinstance(keep, np.ndarray) or keep.dtype != np.bool_:
        got = keep.dtype if isinstance(keep, np.ndarray) else type(keep).__name__
        raise TypeError(f"admit must return a boolean array, not {got}")
    if keep.shape != (len(batch),):
        raise ValueError(
            f"admit returned a mask of shape {keep.shape} for a batch of {len(batch)} row(s)"
        )
    return keep


class _KeyIndex:
    """A table's Routing Keys as sorted integers (`IntegerKeyCodec`), and the
    full keys of the entries whose integer is a hash.

    The full keys are one ``bytes`` and its offsets, in table order, rather than
    an object a key: a forked worker comparing them would otherwise raise every
    key's reference count, and copy each page it sits on (issue 043 item 2).
    """

    def __init__(
        self,
        codec: IntegerKeyCodec,
        values: np.ndarray,
        hashed_at: np.ndarray,
        hashed_keys: bytes,
        offsets: np.ndarray,
    ) -> None:
        self.codec = codec
        self.values = values
        self._hashed_at = hashed_at
        self._hashed_keys = hashed_keys
        self._offsets = offsets

    def __len__(self) -> int:
        return len(self.values)

    def full_keys(self, positions: np.ndarray) -> list[bytes]:
        """The full Routing Keys of the hashed entries at ``positions``."""
        slot = np.searchsorted(self._hashed_at, positions)
        held = slot < len(self._hashed_at)
        if not (held.all() and np.array_equal(self._hashed_at[slot], positions)):
            raise RuntimeError("route table: a hashed key's entry holds no full key")
        data = self._hashed_keys
        starts, stops = self._offsets[slot].tolist(), self._offsets[slot + 1].tolist()
        return [data[start:stop] for start, stop in zip(starts, stops, strict=True)]

    def find(self, encoded: IntegerKeys) -> tuple[np.ndarray, np.ndarray]:
        """Whether each row's key is in the table, and its position there."""
        values = self.values
        at = np.minimum(np.searchsorted(values, encoded.values), len(values) - 1)
        matched = encoded.known & (values[at] == encoded.values)
        if len(encoded.hashed):
            self._confirm(encoded, at, matched)
        return matched, at

    def _confirm(self, encoded: IntegerKeys, at: np.ndarray, matched: np.ndarray) -> None:
        """Match each hashed row only to an entry whose full Routing Key is its own.

        The entries sharing a row's integer are adjacent, so each row walks
        them from the first until one holds its key, or none is left. Two
        keys a hash sends to one integer therefore never trade rows, and a key
        whose integer some other key holds stays unmatched.
        """
        rows = encoded.hashed
        value = encoded.values[rows]
        place = at[rows]
        pending = matched[rows]
        found = np.zeros(len(rows), dtype=bool)
        last = len(self.values) - 1
        while pending.any():
            walking = np.flatnonzero(pending)
            stored = self.full_keys(place[walking])
            same = np.fromiter(
                (key == encoded.full[i] for key, i in zip(stored, walking.tolist(), strict=True)),
                dtype=bool,
                count=len(walking),
            )
            found[walking[same]] = True
            rest = walking[~same]
            more = place[rest] < last
            place[rest] += more
            pending[:] = False
            pending[rest] = more & (self.values[place[rest]] == value[rest])
        matched[rows] = found
        at[rows] = place


class RouteTable:
    """Source routing keys -> a place on one of a build's axes.

    Built by `VariantRouting.table`. Every array is numpy, so a fork-pool
    worker inherits the table without copying it.
    """

    def __init__(
        self,
        keys: _KeyIndex,
        target: np.ndarray,
        axis: np.ndarray | None,
        swaps: np.ndarray,
        *,
        n_axes: int,
        keep_unmatched: bool,
    ) -> None:
        self._keys = keys
        self._target = target
        self._axis = axis
        self._swaps = swaps
        self._n_axes = n_axes
        self._keep_unmatched = keep_unmatched

    @classmethod
    def build(
        cls,
        site_alid: Mapping[SourceKey, str],
        reoriented: Collection[SourceKey],
        axes: Sequence[Sequence[str]],
        *,
        keep_unmatched: bool,
        key_hash: KeyHash | None = None,
    ) -> RouteTable:
        """Place every routed site on the first axis that holds its ALID.

        A site whose ALID no axis holds is left out, so its associations are
        unmatched. Two sites one routing key spells -- one site written in two
        cases -- must reach the same place with the same orientation: otherwise
        which one a lookup finds would depend on insertion order, and an
        association would be misplaced in silence. ``key_hash`` hashes the
        keys the integer cannot spell exactly; `keys.key_hash` by default.
        """
        if not axes or max(len(axis) for axis in axes) >= np.iinfo(np.int32).max:
            raise ValueError("a route table needs one or more axes, each shorter than int32")
        indices = [{alid: i for i, alid in enumerate(axis)} for axis in axes]
        sites, targets, axis_ids, swapped = _placed_sites(site_alid, reoriented, indices)
        del indices
        keys, order = _indexed(sites, routing_keys.key_hash if key_hash is None else key_hash)
        target, axis, swaps = _in_key_order(order, targets, axis_ids, swapped)
        _refuse_folded_conflicts(
            keys, target, axis, swaps, lambda at: routing_key(sites[order[at]]).encode()
        )
        table = cls(keys, target, axis, swaps, n_axes=len(axes), keep_unmatched=keep_unmatched)
        hashed = int(is_hashed(keys.values).sum())
        log.info(
            "Pass 2 lookup: %d variant keys, %d of them hashed and matched on the full key, "
            "%d re-oriented by the lift",
            len(keys),
            hashed,
            int(swaps.sum()),
        )
        return table

    def __len__(self) -> int:
        return len(self._keys)

    def integer_keys(self, sites: Sequence[SourceKey]) -> IntegerKeys:
        """``sites`` as the integer keys this table is searched by."""
        return self._keys.codec.encode(*_columns_of(sites))

    def route_column(
        self,
        associations: Iterable[ReaderAssociation],
        *,
        admit: Admit | None = None,
        imputation_scores: bool = True,
    ) -> RoutedColumn:
        """Route one Analysis's associations, batch by batch, onto the axes.

        ``admit``, given, judges each batch before it is matched and returns a
        keep mask; a refused row reaches no axis and is not counted unmatched.
        Without it every row is routed. ``imputation_scores=False`` says the
        Analysis declared no score, so its rows' scores are not collected and
        each batch carries them as undeclared, which is what such a reader
        yields anyway. A key a reverse-strand lift re-oriented has its z
        negated and its EAF complemented, so both follow the stored alleles
        (issue #284, spec §4-5).
        """
        columns: _Columns = ([], [], [], [], [], [], [])
        chroms, poss, refs, alts, zs, ses, eafs = columns
        scores: list[float | None] = []
        statuses: list[ImputationScoreStatus] = []
        collected = _Collected(self._n_axes, self._keep_unmatched)
        batch_size = _RESOLVE_BATCH
        # Collecting each row's score costs a Pass 2 worker about a fifth of its
        # per-row routing time, so it is done only for a batch that can use it.
        collect_scores = admit is not None and imputation_scores
        for assoc in associations:
            chroms.append(assoc.chromosome)
            poss.append(assoc.position)
            refs.append(assoc.ref)
            alts.append(assoc.alt)
            zs.append(assoc.z)
            ses.append(assoc.se)
            eafs.append(float("nan") if assoc.eaf is None else assoc.eaf)
            if collect_scores:
                scores.append(assoc.imputation_score.value)
                statuses.append(assoc.imputation_score.status)
            if len(zs) >= batch_size:
                self._flush(columns, scores, statuses, admit, collected)
        self._flush(columns, scores, statuses, admit, collected)
        return collected.column()

    def _flush(
        self,
        columns: _Columns,
        scores: list[float | None],
        statuses: list[ImputationScoreStatus],
        admit: Admit | None,
        collected: _Collected,
    ) -> None:
        """Admit, match and collect one batch, then empty every buffer."""
        try:
            if not columns[4]:
                return
            if admit is not None:
                keep = _admitted(admit, columns, scores, statuses)
                _retain_rows(columns, keep)
                if not columns[4]:
                    return
            self._match(columns, collected)
        finally:
            for values in columns:
                values.clear()
            scores.clear()
            statuses.clear()

    def _match(self, columns: _Columns, collected: _Collected) -> None:
        """Look one batch up and add its rows to ``collected``, order kept."""
        chroms, poss, refs, alts, zs, ses, eafs = columns
        if len(self._keys) == 0:
            matched = np.zeros(len(zs), dtype=bool)
            position = _empty_index()
        else:
            matched, place = self._keys.find(self._keys.codec.encode(chroms, poss, refs, alts))
            position = place[matched]
        z = np.array(zs, dtype=np.float32)
        se = np.array(ses, dtype=np.float32)
        eaf = np.array(eafs, dtype=np.float32)
        matched_rows = (z[matched], se[matched], eaf[matched])
        self._collect_matched(position, matched_rows, collected)
        unmatched = ~matched
        collected.n_unmatched += int(unmatched.sum())
        if self._keep_unmatched:
            at = np.flatnonzero(unmatched).tolist()
            keys = [routing_key((chroms[j], poss[j], refs[j], alts[j])) for j in at]
            collected.add_unmatched(keys, (z[unmatched], se[unmatched], eaf[unmatched]))

    def _collect_matched(
        self,
        position: np.ndarray,
        statistics: tuple[np.ndarray, np.ndarray, np.ndarray],
        collected: _Collected,
    ) -> None:
        """Re-orient the matched rows a lift swapped, then split them by axis.

        A swap is signed by the Source Reader's own rule (`swap`, ADR 0076).
        """
        z, se, eaf = statistics
        swapped = self._swaps[position]
        if swapped.any():
            z, eaf = orientation.swap(swapped, z, eaf)
        target = self._target[position].astype(np.int64)
        if self._axis is None:
            collected.add_placed(0, (target, z, se, eaf))
            return
        axis = self._axis[position]
        for axis_id in range(self._n_axes):
            on_axis = axis == axis_id
            collected.add_placed(
                axis_id, (target[on_axis], z[on_axis], se[on_axis], eaf[on_axis])
            )


def _in_key_order(
    order: np.ndarray,
    targets: list[int],
    axis_ids: list[int] | None,
    swaps: list[bool],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """The table's parallel arrays, in its keys' order.

    Each list is emptied once its array exists, so a genome-scale table does
    not hold both forms at its peak.
    """
    target = np.array(targets, dtype=np.int32)
    targets.clear()
    swap = np.array(swaps, dtype=bool)
    swaps.clear()
    axis = None if axis_ids is None else np.array(axis_ids, dtype=np.int8)[order]
    return target[order], axis, swap[order]


def _columns_of(
    sites: Sequence[SourceKey],
) -> tuple[list[str], list[int], list[str], list[str]]:
    """Sites as their four columns: chromosome, position, ref and alt."""
    if not sites:
        return [], [], [], []
    chroms, poss, refs, alts = (list(column) for column in zip(*sites, strict=True))
    return chroms, poss, refs, alts


def _indexed(sites: list[SourceKey], key_hash: KeyHash) -> tuple[_KeyIndex, np.ndarray]:
    """The sites' keys, sorted, and the order that sorts them (a stable sort).

    Encoded a batch at a time; the hashed entries' full keys are then laid
    out in table order, so the entry at a position finds its key by one
    search of the hashed positions.
    """
    codec = IntegerKeyCodec((site[0] for site in sites), key_hash)
    values = np.empty(len(sites), dtype=np.uint64)
    hashed_rows: list[np.ndarray] = []
    full: list[bytes] = []
    for start in range(0, len(sites), _ENCODE_BATCH):
        batch = sites[start : start + _ENCODE_BATCH]
        encoded = codec.encode(*_columns_of(batch))
        values[start : start + len(batch)] = encoded.values
        hashed_rows.append(encoded.hashed + start)
        full.extend(encoded.full)
    order = np.argsort(values, kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    hashed_at = rank[np.concatenate(hashed_rows)] if hashed_rows else _empty_index()
    del rank
    by_place = np.argsort(hashed_at)
    keys = [full[i] for i in by_place.tolist()]
    del full
    offsets = np.zeros(len(keys) + 1, dtype=np.int64)
    np.cumsum(np.fromiter(map(len, keys), dtype=np.int64, count=len(keys)), out=offsets[1:])
    return _KeyIndex(codec, values[order], hashed_at[by_place], b"".join(keys), offsets), order


def _refuse_folded_conflicts(
    keys: _KeyIndex,
    target: np.ndarray,
    axis: np.ndarray | None,
    swaps: np.ndarray,
    key_at: Callable[[int], bytes],
) -> None:
    """Refuse two entries of one Routing Key that reach different places or
    orientations.

    Equal integers that spell their alleles exactly are one key, so adjacent
    ones are compared directly. Equal hashed integers may be different keys,
    so each run of them is compared by its full keys.
    """
    values = keys.values
    same = values[1:] == values[:-1]
    if not same.any():
        return
    differ = (target[1:] != target[:-1]) | (swaps[1:] != swaps[:-1])
    if axis is not None:
        differ |= axis[1:] != axis[:-1]
    hashed = is_hashed(values[1:])
    exact = np.flatnonzero(same & ~hashed & differ)
    if len(exact):
        _refuse(key_at(int(exact[0]) + 1))
    pairs = np.flatnonzero(same & hashed)
    for run in np.split(pairs, np.flatnonzero(np.diff(pairs) != 1) + 1):
        if len(run):
            _refuse_hashed_run(keys, int(run[0]), int(run[-1]) + 2, target, axis, swaps)


def _refuse_hashed_run(
    keys: _KeyIndex,
    start: int,
    stop: int,
    target: np.ndarray,
    axis: np.ndarray | None,
    swaps: np.ndarray,
) -> None:
    """`_refuse_folded_conflicts` for the entries ``[start, stop)``, which share
    one hashed integer."""
    first: dict[bytes, int] = {}
    at = np.arange(start, stop)
    for position, key in zip(at.tolist(), keys.full_keys(at), strict=True):
        earlier = first.setdefault(key, position)
        if (
            target[earlier] != target[position]
            or swaps[earlier] != swaps[position]
            or (axis is not None and axis[earlier] != axis[position])
        ):
            _refuse(key)


def _refuse(key: bytes) -> None:
    raise ValueError(f"source key {key!r} maps to conflicting variants after case folding")


def _placed_sites(
    site_alid: Mapping[SourceKey, str],
    reoriented: Collection[SourceKey],
    indices: list[dict[str, int]],
) -> tuple[list[SourceKey], list[int], list[int] | None, list[bool]]:
    """Each placeable site, its target, axis (with more than one) and swap."""
    sites: list[SourceKey] = []
    targets: list[int] = []
    axis_ids: list[int] | None = [] if len(indices) > 1 else None
    swaps: list[bool] = []
    for site, alid in site_alid.items():
        axis_id = 0
        target = None
        for index in indices:
            target = index.get(alid)
            if target is not None:
                break
            axis_id += 1
        if target is None:
            continue
        sites.append(site)
        targets.append(target)
        if axis_ids is not None:
            axis_ids.append(axis_id)
        swaps.append(site in reoriented)
    return sites, targets, axis_ids, swaps
