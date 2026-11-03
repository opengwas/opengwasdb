"""Pass 2's lookup: each source key's place on the Variant Index, fork-safe.

A `RouteTable` is built once, in the parent, from a routing's site map and
the axes a build stores. It holds numpy arrays only -- a sorted object array
of byte keys and a parallel target, axis and re-orientation array -- so a
forked Pass 2 worker binary-searches it without touching the per-element
reference counts a dict would copy page by page (issue 043 item 2). The keys
are Python bytes rather than a fixed-width ``S`` array, so one rare 546-byte
indel does not pad every key to 546 bytes.

`RouteTable.route_column` streams one Analysis's associations in fixed-size
batches, so a worker never holds a genome-wide source as Python lists, and
returns each axis's rows with the last stream occurrence of each index kept:
when two source keys reach one index, the later one wins, as it would were
the whole source matched at once.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from opengwasdb.readers.interface import ImputationScoreStatus, ReaderAssociation
from opengwasdb.variants.routing.keys import SourceKey, encode_routing_keys, routing_key

log = logging.getLogger(__name__)

# Streamed in fixed-size batches so a worker never materialises a whole
# (genome-wide) source as Python lists at once -- peak per-worker memory is one
# batch of columns plus the accumulated matched rows (issue 043).
_RESOLVE_BATCH = 250_000

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
    """One axis's rows for one Analysis, the last stream occurrence of each index kept.

    ``index`` is a position on that axis, in the stream order of the kept
    rows. ``z`` and ``eaf`` follow the stored alleles; ``se`` is the
    source's. ``conflicting`` holds the indices more than one association
    reached with different statistics, of which only the last is kept.
    """

    index: np.ndarray
    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray
    conflicting: np.ndarray = field(default_factory=_empty_index)


@dataclass(frozen=True)
class Unmatched:
    """The associations no axis holds, by routing key, in stream order.

    Kept only by a table built with ``keep_unmatched``; otherwise empty, and
    only counted. Not deduplicated, and in the reader's orientation.
    ``conflicting`` names the keys more than one kept association reached
    with different statistics, as `Placed.conflicting` names an axis's
    indices; all their rows are still here.
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


def last_wins(index: np.ndarray) -> np.ndarray:
    """The positions to keep: each index's last occurrence, in stream order.

    ``np.unique`` over the reversed indices returns the first position in
    reversed order, which is the last occurrence in stream order.
    """
    if len(index) == 0:
        return np.empty(0, dtype=np.intp)
    _, first_in_reverse = np.unique(index[::-1], return_index=True)
    return np.sort(len(index) - 1 - first_in_reverse)


def _placed(index: np.ndarray, z: np.ndarray, se: np.ndarray, eaf: np.ndarray) -> Placed:
    """One axis's rows with each index's last occurrence kept, and the conflicts named."""
    if len(index) == 0:
        return Placed(index, z, se, eaf)
    keep = last_wins(index)
    conflicting = _empty_index()
    if len(keep) < len(index):
        conflicting = _conflicting_indices(index, keep, (z, se, eaf))
    return Placed(index[keep], z[keep], se[keep], eaf[keep], conflicting)


def _conflicting_indices(
    index: np.ndarray, keep: np.ndarray, statistics: tuple[np.ndarray, ...]
) -> np.ndarray:
    """The duplicated indices whose rows disagree; NaN agrees with NaN."""
    earlier = np.ones(len(index), dtype=bool)
    earlier[keep] = False
    rows = np.flatnonzero(np.isin(index, np.unique(index[earlier])))
    rows = rows[np.argsort(index[rows], kind="stable")]
    grouped = index[rows]
    same = grouped[1:] == grouped[:-1]
    for values in statistics:
        ordered = values[rows]
        a, b = ordered[1:], ordered[:-1]
        same &= (a == b) | (np.isnan(a) & np.isnan(b))
    return np.unique(grouped[1:][(grouped[1:] == grouped[:-1]) & ~same])


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
        empty = (_empty_index(), _empty_stat(), _empty_stat(), _empty_stat())
        placed = tuple(_placed(*_joined(parts, empty)) for parts in self.parts)
        z, se, eaf = _joined(self.unmatched_parts, (_empty_stat(), _empty_stat(), _empty_stat()))
        keys = self.unmatched_keys
        return RoutedColumn(
            placed=placed,
            unmatched=Unmatched(keys, z, se, eaf, _conflicting_keys(keys, (z, se, eaf))),
            n_unmatched=self.n_unmatched,
        )


def _conflicting_keys(keys: list[str], statistics: tuple[np.ndarray, ...]) -> list[str]:
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
    conflicting = _conflicting_indices(codes, last_wins(codes), statistics)
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
    """
    chroms, poss, refs, alts, zs, ses, eafs = columns
    batch = AssociationBatch(
        chromosome=chroms,
        position=poss,
        ref=refs,
        alt=alts,
        z=np.asarray(zs, dtype=np.float64),
        se=np.asarray(ses, dtype=np.float64),
        eaf=np.asarray(eafs, dtype=np.float64),
        imputation_score=np.asarray(scores, dtype=np.float64),
        imputation_status=np.asarray(statuses, dtype=object),
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


class RouteTable:
    """Source routing keys -> a place on one of a build's axes.

    Built by `VariantRouting.table`. Every array is numpy, so a fork-pool
    worker inherits the table without copying it.
    """

    def __init__(
        self,
        keys: np.ndarray,
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
    ) -> RouteTable:
        """Place every routed site on the first axis that holds its ALID.

        A site whose ALID no axis holds is left out, so its associations are
        unmatched. Two sites one routing key spells -- one site written in two
        cases -- must reach the same place with the same orientation: otherwise
        which one a lookup finds would depend on insertion order, and an
        association would be misplaced in silence.
        """
        if not axes or max(len(axis) for axis in axes) >= np.iinfo(np.int32).max:
            raise ValueError("a route table needs one or more axes, each shorter than int32")
        indices = [{alid: i for i, alid in enumerate(axis)} for axis in axes]
        placed = _placed_sites(site_alid, reoriented, indices)
        del indices
        arrays = _sorted_by_key(*placed)
        del placed
        table = cls(*arrays, n_axes=len(axes), keep_unmatched=keep_unmatched)
        table._refuse_folded_conflicts()
        log.info(
            "Pass 2 lookup: %d variant keys, max key length %d bytes, %d re-oriented by the lift",
            len(table._keys),
            max((len(k) for k in table._keys), default=0),
            int(table._swaps.sum()),
        )
        return table

    def __len__(self) -> int:
        return len(self._keys)

    def _refuse_folded_conflicts(self) -> None:
        keys = self._keys
        duplicate = keys[1:] == keys[:-1]
        conflicting = duplicate & (
            (self._target[1:] != self._target[:-1]) | (self._swaps[1:] != self._swaps[:-1])
        )
        if self._axis is not None:
            conflicting |= duplicate & (self._axis[1:] != self._axis[:-1])
        if np.any(conflicting):
            first = keys[1:][conflicting][0]
            raise ValueError(
                f"source key {first!r} maps to conflicting variants after case folding"
            )

    def route_column(
        self, associations: Iterable[ReaderAssociation], *, admit: Admit | None = None
    ) -> RoutedColumn:
        """Route one Analysis's associations, batch by batch, onto the axes.

        ``admit``, given, judges each batch before it is matched and returns a
        keep mask; a refused row reaches no axis and is not counted unmatched.
        Without it every row is routed. A key a reverse-strand lift re-oriented
        has its z negated and its EAF complemented, so both follow the stored
        alleles (issue #284, spec §4-5).
        """
        columns: _Columns = ([], [], [], [], [], [], [])
        chroms, poss, refs, alts, zs, ses, eafs = columns
        scores: list[float | None] = []
        statuses: list[ImputationScoreStatus] = []
        collected = _Collected(self._n_axes, self._keep_unmatched)
        batch_size = _RESOLVE_BATCH
        for assoc in associations:
            chroms.append(assoc.chromosome)
            poss.append(assoc.position)
            refs.append(assoc.ref)
            alts.append(assoc.alt)
            zs.append(assoc.z)
            ses.append(assoc.se)
            eafs.append(float("nan") if assoc.eaf is None else assoc.eaf)
            if admit is not None:
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
                _retain_rows(columns, _admitted(admit, columns, scores, statuses))
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
            query = encode_routing_keys(chroms, poss, refs, alts)
            clipped = np.minimum(np.searchsorted(self._keys, query), len(self._keys) - 1)
            matched = self._keys[clipped] == query
            position = clipped[matched]
        z = np.array(zs, dtype=np.float32)
        se = np.array(ses, dtype=np.float32)
        eaf = np.array(eafs, dtype=np.float32)
        self._collect_matched(position, (z[matched], se[matched], eaf[matched]), collected)
        unmatched = ~matched
        collected.n_unmatched += int(unmatched.sum())
        if self._keep_unmatched:
            keys = [
                routing_key((chroms[j], poss[j], refs[j], alts[j]))
                for j in np.flatnonzero(unmatched)
            ]
            collected.add_unmatched(keys, (z[unmatched], se[unmatched], eaf[unmatched]))

    def _collect_matched(
        self,
        position: np.ndarray,
        statistics: tuple[np.ndarray, np.ndarray, np.ndarray],
        collected: _Collected,
    ) -> None:
        """Re-orient the matched rows a lift swapped, then split them by axis."""
        z, se, eaf = statistics
        swapped = self._swaps[position]
        if swapped.any():
            z = np.where(swapped, -z, z)
            eaf = np.where(swapped, np.float32(1.0) - eaf, eaf)
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


def _sorted_by_key(
    keys_list: list[bytes], targets: list[int], axis_ids: list[int] | None, swaps: list[bool]
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """The table's parallel arrays, in sorted key order (a stable sort).

    Each list is emptied once its array exists, before the sort, so a
    genome-scale table does not hold both forms at its peak.
    """
    keys = np.array(keys_list, dtype=object)
    keys_list.clear()
    target = np.array(targets, dtype=np.int32)
    targets.clear()
    swap = np.array(swaps, dtype=bool)
    swaps.clear()
    axis = None if axis_ids is None else np.array(axis_ids, dtype=np.int8)
    order = np.argsort(keys, kind="stable")
    return keys[order], target[order], None if axis is None else axis[order], swap[order]


def _placed_sites(
    site_alid: Mapping[SourceKey, str],
    reoriented: Collection[SourceKey],
    indices: list[dict[str, int]],
) -> tuple[list[bytes], list[int], list[int] | None, list[bool]]:
    """Each placeable site's key, target, axis (with more than one) and swap."""
    keys: list[bytes] = []
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
        keys.append(routing_key(site).encode())
        targets.append(target)
        if axis_ids is not None:
            axis_ids.append(axis_id)
        swaps.append(site in reoriented)
    return keys, targets, axis_ids, swaps
