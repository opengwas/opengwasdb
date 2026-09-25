"""Distinct off-reference key runs and their memory-bounded merge (ticket #222).

Resolving Pass 2's off-reference keys starts by reducing every column's keys
to one build-wide set of distinct keys, each with the assemblies that declared
it. A :class:`KeyRun` is a sorted, distinct slice of that set, and the
reduction is a stream of pairwise run merges -- one column's run into a
worker's chunk, and each worker's chunk into the parent's -- so no process ever
holds more than a few runs, and each is released as soon as it is merged.

A run holds only numbers: about 13 bytes a key. The hashed keys' raw strings
stay in the per-column side files, because a Python string costs 60-100 bytes
and on OGS-00011 a third of the distinct keys are hashed; carrying them made a
51-column worker chunk several GB. The raw strings are read once, after the
merge, for the distinct keys that need them (``_fetch_hashed_raw`` in the
build), from the lowest column that declared each one.

The build-wide hash guarantee from #218 still holds without the strings: every
hashed key carries ``hashed_check``, a 64-bit hash of its raw string that is
independent of the one that made its value. Two different raw keys that share a
value almost surely differ in check (both hashes would have to collide), so a
merge that meets one value with two checks raises :class:`HashedKeyCollision`
naming the two columns, and the build reads both raw keys from them to name
them in the error.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.layouts.hybrid.unknown_keys import UnknownKeyEncodingError

__all__ = [
    "HG19",
    "HG38",
    "HashedKeyCollision",
    "KeyRun",
    "column_run",
    "merge_key_stream",
    "merge_runs",
    "read_run",
    "sorted_distinct",
    "write_run",
]

# One bit per source assembly, OR-ed when a key is declared more than once.
# A key declared under both has no single physical locus and is dropped.
HG19 = 1
HG38 = 2


class HashedKeyCollision(UnknownKeyEncodingError):
    """Two different raw keys encode to one hashed value.

    Raised by a merge, which holds no raw strings: it carries the value and the
    two columns whose side files hold the two keys, so the build can read them
    and name both.
    """

    def __init__(self, value: int, columns: tuple[int, int]) -> None:
        super().__init__(
            f"hash collision at off-reference value {value} between the keys of "
            f"columns {columns[0]} and {columns[1]}; refusing to merge them"
        )
        self.value = value
        self.columns = columns

    def __reduce__(self) -> tuple[Any, ...]:
        # Raised in a pool worker and re-raised in the parent: pickle the
        # fields, not the formatted message.
        return (HashedKeyCollision, (self.value, self.columns))


@dataclass(frozen=True)
class KeyRun:
    """Sorted distinct off-reference keys, with no raw strings.

    ``values`` is sorted and distinct; ``assembly_bits`` parallels it. The
    hash-region keys are repeated in ``hashed_values`` (sorted, distinct) with
    ``hashed_check`` -- a hash of the raw string independent of the value's --
    and ``hashed_origin``, the lowest column index whose side file holds it.
    """

    values: np.ndarray
    assembly_bits: np.ndarray
    hashed_values: np.ndarray
    hashed_check: np.ndarray
    hashed_origin: np.ndarray

    @property
    def size(self) -> int:
        return len(self.values)


def _empty_run() -> KeyRun:
    return KeyRun(
        values=np.empty(0, dtype=np.uint64),
        assembly_bits=np.empty(0, dtype=np.int8),
        hashed_values=np.empty(0, dtype=np.uint64),
        hashed_check=np.empty(0, dtype=np.uint64),
        hashed_origin=np.empty(0, dtype=np.int32),
    )


def _run_starts(sorted_values: np.ndarray) -> np.ndarray:
    """A bool mask marking the first element of each equal-value run."""
    starts = np.empty(len(sorted_values), dtype=bool)
    if len(sorted_values) == 0:
        return starts
    starts[0] = True
    np.not_equal(sorted_values[1:], sorted_values[:-1], out=starts[1:])
    return starts


def sorted_distinct(values: np.ndarray) -> np.ndarray:
    """``values``' distinct entries, ascending. Sorts ``values`` in place.

    ``np.unique`` allocates a sorted copy plus its own temporaries: about 0.9 GB
    of peak for one 20-million-key (160 MB) OGS-00011 column, in every worker
    at once. Sorting the freshly loaded array in place costs one bool mask, and
    returns ``values`` itself when it holds no repeats.
    """
    values.sort()
    starts = _run_starts(values)
    if bool(starts.all()):
        # A Pass 2 spill is already deduplicated per column, so this is the
        # usual case: no copy at all.
        return values
    distinct: np.ndarray = values[starts]
    return distinct


def column_run(
    keys: np.ndarray,
    hashed_values: np.ndarray,
    hashed_check: np.ndarray,
    *,
    column: int,
    assembly_bit: int,
) -> KeyRun:
    """One column's run. Sorts ``keys`` in place.

    ``hashed_values``/``hashed_check`` pair the column's hashed rows with the
    check hash of each row's raw key, in any order; a value named twice with
    two checks is a collision inside the column.
    """
    values = sorted_distinct(keys)
    order = np.argsort(hashed_values, kind="stable")
    ordered, checks = hashed_values[order], hashed_check[order]
    starts = _run_starts(ordered)
    first_check = checks[starts][np.cumsum(starts) - 1]
    clash = np.flatnonzero(checks != first_check)
    if len(clash):
        raise HashedKeyCollision(int(ordered[clash[0]]), (column, column))
    distinct = ordered[starts]
    return KeyRun(
        values=values,
        assembly_bits=np.full(len(values), assembly_bit, dtype=np.int8),
        hashed_values=distinct,
        hashed_check=checks[starts],
        hashed_origin=np.full(len(distinct), column, dtype=np.int32),
    )


def _locate(into: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insertion position of each query in sorted ``into``, and whether it is there."""
    position = np.searchsorted(into, queries)
    present = np.zeros(len(queries), dtype=bool)
    inside = position < len(into)
    present[inside] = into[position[inside]] == queries[inside]
    return position, present


def _merge_hashed(a: KeyRun, b: KeyRun) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The union of two runs' hashed keys, refusing one value with two checks."""
    position, present = _locate(a.hashed_values, b.hashed_values)
    shared = position[present]
    clash = np.flatnonzero(a.hashed_check[shared] != b.hashed_check[present])
    if len(clash):
        at = int(clash[0])
        raise HashedKeyCollision(
            int(a.hashed_values[shared[at]]),
            (int(a.hashed_origin[shared[at]]), int(b.hashed_origin[present][at])),
        )
    origin = a.hashed_origin.copy()
    origin[shared] = np.minimum(origin[shared], b.hashed_origin[present])
    fresh = ~present
    at_fresh = position[fresh]
    return (
        np.insert(a.hashed_values, at_fresh, b.hashed_values[fresh]),
        np.insert(a.hashed_check, at_fresh, b.hashed_check[fresh]),
        np.insert(origin, at_fresh, b.hashed_origin[fresh]),
    )


def merge_runs(a: KeyRun, b: KeyRun) -> KeyRun:
    """The union of two runs: assemblies OR-ed, the lower origin column kept.

    A linear merge -- ``searchsorted`` plus ``np.insert`` -- so the temporaries
    are the output and a few index arrays, not a concatenate-and-argsort of
    both inputs.
    """
    position, present = _locate(a.values, b.values)
    bits = a.assembly_bits.copy()
    bits[position[present]] |= b.assembly_bits[present]
    fresh = ~present
    at_fresh = position[fresh]
    values = np.insert(a.values, at_fresh, b.values[fresh])
    bits = np.insert(bits, at_fresh, b.assembly_bits[fresh])
    hashed_values, hashed_check, hashed_origin = _merge_hashed(a, b)
    return KeyRun(
        values=values,
        assembly_bits=bits,
        hashed_values=hashed_values,
        hashed_check=hashed_check,
        hashed_origin=hashed_origin,
    )


def merge_key_stream(runs: Iterable[KeyRun]) -> KeyRun:
    """Fold a stream of runs into one, holding a bounded stack of partial runs.

    Each arriving run is pushed and merged down while the run beneath it is no
    more than twice its size, so the stack's sizes more than double going down:
    it holds at most ``log2`` of the total runs, and at most about twice the
    largest one, while the work stays ``O(total log n)``. Columns overlap
    heavily in practice (50 OGS-00011 Analyses of ~20 M keys each share 38.7 M
    distinct keys), which keeps the stack at one or two runs; a plain binary
    counter would instead hold a near-full-size run at every level. Calling
    ``list(runs)`` first would hold every input at once (ticket #222 review
    round 1).
    """
    stack: list[KeyRun] = []
    for run in runs:
        stack.append(run)
        # The loop name must not outlive the push: the stream's next item is
        # produced while this frame still holds it.
        del run
        _settle(stack)
    while len(stack) > 1:
        _merge_top(stack)
    return stack[0] if stack else _empty_run()


def _settle(stack: list[KeyRun]) -> None:
    while len(stack) > 1 and stack[-2].size <= 2 * stack[-1].size:
        _merge_top(stack)


def _merge_top(stack: list[KeyRun]) -> None:
    """Replace the stack's top two runs with their merge, releasing both."""
    top = stack.pop()
    below = stack.pop()
    stack.append(merge_runs(below, top))


def write_run(run: KeyRun, path: Path) -> None:
    """Spill a run so a pool worker returns a path, not the arrays.

    Returning the arrays would pickle them in the worker and park the
    unpickled copy in the parent until ``ordered_map`` yields it -- up to its
    whole in-flight window of results at once.
    """
    np.savez(
        path,
        values=run.values,
        assembly_bits=run.assembly_bits,
        hashed_values=run.hashed_values,
        hashed_check=run.hashed_check,
        hashed_origin=run.hashed_origin,
    )


def read_run(path: Path) -> KeyRun:
    """Load a ``write_run`` spill, refusing arrays that do not pair up."""
    with np.load(path) as data:
        run = KeyRun(
            values=data["values"],
            assembly_bits=data["assembly_bits"],
            hashed_values=data["hashed_values"],
            hashed_check=data["hashed_check"],
            hashed_origin=data["hashed_origin"],
        )
    if len(run.assembly_bits) != run.size or not (
        len(run.hashed_values) == len(run.hashed_check) == len(run.hashed_origin)
    ):
        raise UnknownKeyEncodingError(
            f"key run {path} has {run.size} value(s) with {len(run.assembly_bits)} "
            f"assembly bit(s), and {len(run.hashed_values)} hashed value(s) with "
            f"{len(run.hashed_check)} check(s) and {len(run.hashed_origin)} origin(s); "
            "refusing to merge it"
        )
    return run
