"""`StoreCodec` -- the encoders, and the side tables and baselines a build writes.

`StoreCodec` is the Store Reader's `StoreDecoder`
(`opengwasdb.store_reader.encoding.codec`, the one place stored bytes become
physical values) with the encoders added: `encode_z`, `quantise_z`,
`encode_se` and `encode_eaf` code values under the same plan, scale and
markers the decoder reads them with (#464). This module also builds the sparse
side tables an integer plane's out-of-range cells go to (the `*Builder`s,
`write_exact_table`), and the per-variant EAF baselines a residual `eaf` plane
is coded against.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar, cast

import numpy as np

from opengwasdb.store.arrays import create_array
from opengwasdb.store_reader.encoding.codec import (
    EafExceptionTable,
    PositionSource,
    SeExceptionTable,
    SparseExactTable,
    StoreDecoder,
    ZOverflowTable,
    expit,
    logit,
    positions_for,
    se_float16,
    se_from_log,
    se_log_prediction,
)
from opengwasdb.store_reader.encoding.plan import (
    EAF_ABSENT,
    EAF_CODE_MAX,
    EAF_CODE_MIN,
    EAF_EXCEPTION,
    N_MAX,
    N_MISSING,
    SE_CODE_MAX,
    SE_CODE_MIN,
    SE_EXCEPTION,
    SE_MISSING,
    Z_CODE_MAX,
    Z_MISSING,
    Z_OVERFLOW,
    EafBaselineError,
    StoreEncoding,
)
from opengwasdb.store_reader.layout import EXCEPTION_TABLE_CHUNK, ArrayRole

# Sparse tables can still contain millions of exact values at the accepted
# 2% ceiling.  Keep their physical chunks bounded so both the writer and the
# compressed-size decision can stream them.  This *is* the seam's
# `EXCEPTION_TABLE_CHUNK` -- the `EXCEPTION_TABLE` role's default layout -- so
# the measurement in `se.py` and the array the seam writes cannot disagree.
EXACT_TABLE_CHUNK = EXCEPTION_TABLE_CHUNK

#: The largest SE a `float16` plane holds (65,504). An association whose SE is
#: above it is not stored, in z and SE together, whatever the plan (spec §15,
#: issue #338, ADR 0068): `storable` leaves it out before any plane is written.
SE_FLOAT16_MAX: float = float(np.finfo(np.float16).max)


def se_above_bound(values: Any) -> np.ndarray:
    """Which of `values` are SEs above `SE_FLOAT16_MAX`, `+inf` included; a missing
    one is not. The one comparison with the bound (issue #338): `storable` leaves
    out what it marks, the encoders refuse it, and every reader that asks whether a
    stored or source cell is above the bound asks it here.
    """
    with np.errstate(invalid="ignore"):
        over: np.ndarray = np.asarray(values) > SE_FLOAT16_MAX
    return over


class SeOutOfRange(ValueError):
    """An SE above `SE_FLOAT16_MAX` reached an encoder (issue #338).

    `storable` leaves such an association out, so one here was handed to an
    encoder by a path that did not ask `storable` first. A `float16` plane would
    hold it as 65,504 or `+inf` beside a finite z, and a residual plane as an
    exact exception no Store of this format may hold.
    """


def require_se_in_range(values: np.ndarray, *, where: str) -> None:
    """Refuse an SE above `SE_FLOAT16_MAX`, naming `where`, the count and the first.

    The encoders' guard: what makes a path that forgot `storable` fail loudly
    instead of writing `+inf`, where numpy's overflow warning was the only signal.
    """
    v = np.asarray(values)
    over = se_above_bound(v)
    if not over.any():
        return
    first = tuple(int(axis[0]) for axis in np.nonzero(over))
    raise SeOutOfRange(
        f"{where}: {int(over.sum()):,} standard error(s) above {SE_FLOAT16_MAX:,.0f}, the "
        f"largest a float16 plane holds (first {float(v[first])!r} at {first}). Such an "
        "association is not stored (spec §15, issue #338): the writer must hold what "
        "`storable` returns, never the cells it was handed"
    )


#: Below this decoded `log(se)` no code can read above `SE_FLOAT16_MAX`: `exp` is
#: monotone, and the margin is far wider than its rounding or the float32 one.
_LOG_SE_BOUND_FLOOR = float(np.log(SE_FLOAT16_MAX)) - 1e-6


def se_residual_codes(
    values: np.ndarray,
    eaf: np.ndarray,
    analysis_index: np.ndarray,
    coefficients: np.ndarray,
    step: float,
    *,
    exact: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Code SEs into the int8 residual plane: the one exception decision.

    Encoding, the candidate-range measurement, the Dense rewrite's exception
    count and the Ragged fitter all come through here, so a Store cannot be
    *measured* or *allocated* under one rule and *written* under another. A
    present SE is an exact exception when:

    * its residual from the model is not finite or falls outside the int8 range
      (a zero SE, an unpredictable cell, a residual beyond the range);
    * it is `exact` -- carried decoded from another Store, so it must decode to
      itself (issue #301) -- and its code would not;
    * its code decodes above `SE_FLOAT16_MAX` (issue #338). A code is the nearest
      step, so 65,504 can read half a step above itself; and a reader predicts
      from the frequency narrowed to `float32`, which under a steep model moves
      the prediction by whole units, so a far smaller SE can read above it too.

    The codes are measured from the prediction for the frequency as given, so no
    build's bytes move; the last two rules judge the value a reader decodes, from
    the `float32` frequency, every kept cell of it. Returns the raw codes and the
    mask of cells whose exact value goes in the side table.
    """
    v = np.asarray(values, dtype=np.float64)
    f = np.broadcast_to(np.asarray(eaf), v.shape)
    ai = np.broadcast_to(np.asarray(analysis_index, dtype=np.int64), v.shape)
    prediction = se_log_prediction(f, ai, coefficients)
    present = np.isfinite(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        codes = np.rint((np.log(v) - prediction) / step)
    held = present & (v > 0) & np.isfinite(codes) & (codes >= SE_CODE_MIN) & (codes <= SE_CODE_MAX)
    read = (  # what `decode_se` predicts from: the frequency narrowed to float32
        prediction
        if f.dtype == np.float32
        else se_log_prediction(f.astype(np.float32), ai, coefficients)
    )
    with np.errstate(invalid="ignore"):
        decoded_log = read + codes * step
    if exact is not None:
        judged = held & np.broadcast_to(np.asarray(exact, dtype=bool), v.shape)
        held[judged] = se_from_log(decoded_log[judged]) == v[judged].astype(np.float32)
    near = held & (decoded_log > _LOG_SE_BOUND_FLOOR)
    held[near] = ~se_above_bound(se_from_log(decoded_log[near]))
    exceptional = present & ~held
    raw = np.full(v.shape, SE_MISSING, dtype=np.int8)
    raw[exceptional] = SE_EXCEPTION
    raw[held] = codes[held].astype(np.int8)
    return raw, exceptional


def require_parallel(named: Sequence[tuple[str, Any]], *, where: str) -> None:
    """Refuse arrays that are not parallel, naming each one's length.

    Parallel means one shape: the arrays are alternative readings of the same
    cells, so an index into one is an index into all of them. `named` is
    `(name, array)` pairs in the order the refusal should name them, `None`
    where a plane is absent.

    One rule for every writer, because the failure it prevents is the same
    everywhere: an array shorter (or longer) than the cells it describes is
    either broadcast to the cells' length, which stores one cell's value
    under another's variant, or indexed with a mask of the wrong length, which
    crashes after the planes are open. Neither says which array was wrong.
    It lives here, below the codecs and the plane writers that all import
    this module, so every one of them can apply it without an upward import.
    """
    present = [(name, np.asarray(array)) for name, array in named if array is not None]
    if not present:
        return
    _, first = present[0]
    if all(array.shape == first.shape for _, array in present):
        return
    raise ValueError(f"{where}: {_lengths(present)} must be parallel")


def _lengths(named: Sequence[tuple[str, np.ndarray]]) -> str:
    """`named` as a readable list, a sequence's length or an array's shape."""
    described = [
        f"{name} of length {array.shape[0]}"
        if array.ndim == 1
        else f"{name} of shape {array.shape}"
        for name, array in named
    ]
    if len(described) == 1:
        return described[0]
    return ", ".join(described[:-1]) + " and " + described[-1]



class SparseExactBuilder:
    """Accumulates side-table cells during a write, in any order."""

    table_type: ClassVar[type[SparseExactTable]] = SparseExactTable

    def __init__(self) -> None:
        self._index: list[np.ndarray] = []
        self._value: list[np.ndarray] = []

    def add(self, positions: np.ndarray, values: np.ndarray) -> None:
        """Accumulate one batch: one exact value per flat position.

        Judged parallel here, where a table is handed the cells it holds,
        because `table()` sorts the values by the positions: a short value array
        would leave entries without a value, or values without an entry, in the
        table a reader decodes a cell from (issue #392).
        """
        require_parallel(
            (("positions", positions), ("values", values)),
            where=f"the {self.table_type.what} table",
        )
        self._index.append(np.asarray(positions, dtype=np.int64))
        self._value.append(np.asarray(values, dtype=np.float32))

    def __len__(self) -> int:
        return int(sum(len(part) for part in self._index))

    def table(self) -> Any:
        """Sort, and reject a plane that recorded one cell twice."""
        cls = self.table_type
        if not self._index:
            return cls.empty()
        index = np.concatenate(self._index)
        value = np.concatenate(self._value)
        order = np.argsort(index, kind="stable")
        index, value = index[order], value[order]
        if len(index) > 1 and np.any(index[1:] == index[:-1]):
            duplicate = int(index[1:][index[1:] == index[:-1]][0])
            raise ValueError(
                f"{cls.what} table has two entries for flat position {duplicate}; "
                "a cell was written twice with different values"
            )
        return cls(index=index, value=value)


_Table = TypeVar("_Table", bound=SparseExactTable)


class ZOverflowBuilder(SparseExactBuilder):
    """Accumulates out-of-range z cells during a write, in any order."""

    table_type: ClassVar[type[SparseExactTable]] = ZOverflowTable

    def table(self) -> ZOverflowTable:
        return cast(ZOverflowTable, super().table())


class EafExceptionBuilder(SparseExactBuilder):
    """Accumulates exception cells during a write, in any order."""

    table_type: ClassVar[type[SparseExactTable]] = EafExceptionTable

    def table(self) -> EafExceptionTable:
        return cast(EafExceptionTable, super().table())


def table_end_to_end(table_type: type[_Table], runs: Sequence[_Table]) -> _Table:
    """One table from runs of entries, each a contiguous range of flat positions,
    laid end to end in the order given -- a Dense row band's, a Ragged region's.

    Not sorted: the order the runs are given in is the table, so a caller that
    assembled them in the order its workers *finished* is refused by the table's
    own rule (its index would no longer ascend) instead of writing a table no
    lookup can read (#321, as `eaf_exception_table` for the Dense build).
    """
    if not runs:
        return table_type(index=np.empty(0, dtype=np.int64), value=np.empty(0, dtype=np.float32))
    return table_type(
        index=np.concatenate([run.index for run in runs]),
        value=np.concatenate([run.value for run in runs]),
    )


def eaf_exception_table(bands: Sequence[EafExceptionTable]) -> EafExceptionTable:
    """One table from a grid encode's per-band ones, in the order they arrive.

    A Dense row band holds a contiguous run of flat positions, so the bands'
    runs, laid end to end in row order, are already the sorted, unique table
    the format requires -- and are taken as given rather than sorted again,
    because sorting would hide the one thing that makes the bytes depend on
    order: which band a parent put first. A caller that assembled them in the
    order its workers *finished* is refused by the table's own rule (the index
    would no longer ascend), instead of writing a table no lookup can read.
    """
    return table_end_to_end(EafExceptionTable, bands)


class SeExceptionBuilder(SparseExactBuilder):
    table_type: ClassVar[type[SparseExactTable]] = SeExceptionTable

    def table(self) -> SeExceptionTable:
        return cast(SeExceptionTable, super().table())


@dataclass(frozen=True)
class SideTables:
    """One batch's exact side-table entries, one table per plane that has one."""

    z_overflow: ZOverflowTable
    eaf_exceptions: EafExceptionTable
    se_exceptions: SeExceptionTable

    def decoder(self, encoding: StoreEncoding) -> StoreCodec:
        """A codec that decodes the batch these entries came from."""
        return StoreCodec(
            encoding,
            z_overflow=self.z_overflow,
            eaf_exceptions=self.eaf_exceptions,
            se_exceptions=self.se_exceptions,
        )


# ── EAF baselines ───────────────────────────────────────────────────────────
#
# The baseline is a per-variant `float32`: the within-store representative
# *observed* frequency, against which every cell at that variant is coded as a
# residual. It is computed in logit space, because that is the space the
# residual lives in and a median taken in one space is not the median in the
# other once ties are averaged.


def _as_baseline(median_logit: np.ndarray) -> np.ndarray:
    """Median logits -> the `float32` baseline that is actually stored.

    A variant whose frequencies are all within about 6e-8 of 0 or 1 has a
    median logit whose `expit` rounds to exactly 0 or 1 in `float32`, and
    neither has a logit to code a residual against. Such a variant gets **no**
    baseline, so its cells go to the exception table and are held exactly --
    rather than being nudged to the nearest representable neighbour, which
    would move a MAF of 1e-8 to one of 6e-8 without saying so.
    """
    with np.errstate(invalid="ignore"):
        stored = np.where(np.isfinite(median_logit), expit(median_logit), np.nan).astype(np.float32)
        usable = np.isfinite(stored) & (stored > np.float32(0.0)) & (stored < np.float32(1.0))
    return np.where(usable, stored, np.float32(np.nan)).astype(np.float32)


def eaf_baseline_from_grid(values: np.ndarray) -> np.ndarray:
    """Per-row baselines for a Dense `(n_variants, n_analyses)` EAF block.

    A row with no frequency in the open interval (0, 1) -- no Analysis
    reported one, or every Analysis reported a monomorphic 0 or 1 -- gets NaN,
    and every EAF-bearing cell in it becomes an exception. That is deliberate:
    there is no honest baseline to code against, and inventing one (0.5, say,
    or the store-wide mean) is the kind of plausible-looking default this
    format refuses.
    """
    block = np.asarray(values, dtype=np.float64)
    if block.ndim != 2:
        raise EafBaselineError(f"expected a 2-D EAF block, got shape {block.shape}")
    with np.errstate(invalid="ignore"):
        transformed = logit(block)
        all_missing = np.all(~np.isfinite(transformed), axis=1)
        medians = np.full(block.shape[0], np.nan, dtype=np.float64)
        if not np.all(all_missing):
            medians[~all_missing] = np.nanmedian(transformed[~all_missing], axis=1)
    return _as_baseline(medians)


def eaf_baseline_from_pairs(
    variant_index: np.ndarray, values: np.ndarray, n_variants: int
) -> np.ndarray:
    """Per-variant baselines for a CSR plane, in one vectorised pass.

    `variant_index` and `values` are parallel and in any order. Sorting by
    (variant, logit) puts each variant's usable values in a contiguous, sorted
    run, so the median is two gathers rather than a Python loop over up to ten
    million variants.

    Every intermediate here is as long as the plane, so peak memory is a fixed
    multiple of the cell count: fine for a plane held as one array, unaffordable
    for a Ragged Overflow with billions of cells (issue #226). A caller that
    already holds its cells as per-Analysis runs sorted by variant -- which is
    what the Overflow CSR writer holds -- should use
    `eaf_baseline_from_sorted_runs` instead, which is exact and bounded.
    """
    variant_index = np.asarray(variant_index, dtype=np.int64)
    transformed = logit(values)
    if variant_index.shape != transformed.shape:
        raise EafBaselineError(
            f"variant_index shape {variant_index.shape} does not match values shape "
            f"{transformed.shape}"
        )
    baseline = np.full(int(n_variants), np.nan, dtype=np.float64)
    usable = np.isfinite(transformed)
    if not np.any(usable):
        return _as_baseline(baseline)
    _fill_median_logits(baseline, variant_index[usable], transformed[usable])
    return _as_baseline(baseline)


#: Cells gathered at once by `eaf_baseline_from_sorted_runs`. Peak working set is
#: about this many cells of `float64` logit plus their indices (~12 B/cell), and
#: does not grow with the plane: 2**25 cells is roughly 400 MiB.
DEFAULT_BASELINE_CELL_BUDGET = 1 << 25


def eaf_baseline_from_sorted_runs(
    index_runs: Sequence[np.ndarray],
    value_runs: Sequence[np.ndarray],
    n_variants: int,
    *,
    cell_budget: int = DEFAULT_BASELINE_CELL_BUDGET,
) -> np.ndarray:
    """Per-variant baselines from per-Analysis runs, in bounded blocks.

    Each run is one Analysis's `(variant_index, eaf)` **sorted ascending by
    variant index** -- what the Overflow CSR writer already holds, because
    every column is sorted before it is appended. That ordering is what makes
    this bounded: a contiguous range of variants is a contiguous *slice* of
    every run, located by one `searchsorted` per run, so a block's cells can be
    gathered without touching the rest of the plane.

    Blocks are cut on variant boundaries, never through one. A variant's median
    is not computable from a subset of its values, so each block must hold all
    the cells of every variant it covers for its medians to be final.

    Bit-identical to `eaf_baseline_from_pairs` over the concatenated runs: the
    logit is taken in the same `float64`, and each variant's median is the same
    arithmetic over the same sorted values. Peak memory is `cell_budget`-bounded
    rather than proportional to the plane (issue #226), so it is the path a
    billion-cell Ragged Overflow can actually take.
    """
    if len(index_runs) != len(value_runs):
        raise EafBaselineError(
            f"{len(index_runs)} index runs does not match {len(value_runs)} value runs"
        )
    baseline = np.full(int(n_variants), np.nan, dtype=np.float64)
    runs = _checked_runs(index_runs, value_runs)
    if not runs or n_variants <= 0:
        return _as_baseline(baseline)
    edges = _variant_block_edges(runs, int(n_variants), max(1, int(cell_budget)))
    # One searchsorted per run for every edge at once: the per-block slice
    # bounds are then lookups, not a binary search per block per run.
    bounds = np.empty((len(runs), len(edges)), dtype=np.int64)
    for position, (run_index, _) in enumerate(runs):
        bounds[position] = np.searchsorted(run_index, edges, side="left")
    for block in range(len(edges) - 1):
        vi, lv = _gather_block(runs, bounds[:, block], bounds[:, block + 1])
        if len(vi):
            _fill_median_logits(baseline, vi, lv)
    return _as_baseline(baseline)


def _checked_runs(
    index_runs: Sequence[np.ndarray], value_runs: Sequence[np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The non-empty runs, each verified parallel and sorted by variant index.

    Sortedness is the contract the whole bounded walk rests on, and a run that
    quietly broke it would return wrong baselines rather than fail, so it is
    checked rather than assumed. Each run is one Analysis, so the check is a
    diff over thousands of elements, not over the plane.
    """
    runs: list[tuple[np.ndarray, np.ndarray]] = []
    for position, (run_index, run_values) in enumerate(zip(index_runs, value_runs, strict=True)):
        run_index = np.asarray(run_index)
        run_values = np.asarray(run_values)
        if run_index.shape != run_values.shape:
            raise EafBaselineError(
                f"run {position}: variant_index shape {run_index.shape} does not match "
                f"values shape {run_values.shape}"
            )
        if run_index.size == 0:
            continue
        if np.any(np.diff(run_index) < 0):
            raise EafBaselineError(f"run {position} is not sorted by variant index")
        runs.append((run_index, run_values))
    return runs


def _variant_block_edges(
    runs: Sequence[tuple[np.ndarray, np.ndarray]], n_variants: int, cell_budget: int
) -> np.ndarray:
    """Variant-aligned block edges, each block holding about `cell_budget` cells.

    Cell counts are read off a fixed grid over the variant axis rather than per
    variant, so choosing the edges costs one `searchsorted` per run and no pass
    over the cells. A block is therefore bounded by the budget *or* by the
    densest single grid interval, whichever is larger -- with a grid this fine
    that interval is a few thousand variants, so the bound holds in practice
    while the edge choice stays cheap.
    """
    grid = np.unique(np.linspace(0, n_variants, min(n_variants, 4096) + 1).astype(np.int64))
    # cells_before[i] = cells whose variant index is below grid[i]
    cells_before = np.zeros(len(grid), dtype=np.int64)
    for run_index, _ in runs:
        cells_before += np.searchsorted(run_index, grid, side="left")
    edges = [int(grid[0])]
    anchor = 0
    for i in range(1, len(grid)):
        if cells_before[i] - cells_before[anchor] >= cell_budget:
            edges.append(int(grid[i]))
            anchor = i
    if edges[-1] != int(grid[-1]):
        edges.append(int(grid[-1]))
    return np.array(edges, dtype=np.int64)


def _gather_block(
    runs: Sequence[tuple[np.ndarray, np.ndarray]], starts: np.ndarray, stops: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One block's `(variant_index, logit)` for the cells with a usable logit."""
    index_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for (run_index, run_values), start, stop in zip(runs, starts, stops, strict=True):
        if stop > start:
            index_parts.append(run_index[start:stop])
            value_parts.append(run_values[start:stop])
    if not index_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    vi = np.concatenate(index_parts)
    lv = logit(np.concatenate(value_parts))
    usable = np.isfinite(lv)
    return vi[usable], lv[usable]


def _fill_median_logits(baseline: np.ndarray, vi: np.ndarray, lv: np.ndarray) -> None:
    """Write each variant's median logit into `baseline`, in place.

    Sorting by (variant, logit) puts each variant's values in a contiguous
    sorted run, so the median is two gathers: the middle value when the count is
    odd, the mean of the two middle ones when it is even -- `numpy.median`'s
    arithmetic, on the values `numpy.median` would have seen.
    """
    order = np.lexsort((lv, vi))
    vi, lv = vi[order], lv[order]
    starts = np.flatnonzero(np.concatenate(([True], vi[1:] != vi[:-1])))
    counts = np.diff(np.concatenate((starts, [len(vi)])))
    lower = lv[starts + (counts - 1) // 2]
    upper = lv[starts + counts // 2]
    baseline[vi[starts]] = (lower + upper) / 2.0


def encode_sample_sizes(values: np.ndarray) -> np.ndarray:
    """Per-association sample sizes as the `uint32` codes the `n` plane stores.

    NaN is missing and becomes `N_MISSING`. Anything that is not a whole number
    from 1 to `N_MAX` is refused rather than rounded or clipped: a sample size
    of 0, a fraction, or one that would collide with the reserved code is a
    broken input, not a nearly-right count (ADR 0067). A module function, not a
    codec method, because the plane has one kind and the Ragged writer encodes
    each Analysis as it arrives, before the plan exists.
    """
    v = np.asarray(values, dtype=np.float64)
    missing = np.isnan(v)
    present = v[~missing]
    bad = ~np.isfinite(present) | (present < 1) | (present > N_MAX) | (present != np.floor(present))
    if np.any(bad):
        raise ValueError(
            f"cannot store sample size {float(present[bad][0])!r}: an n cell holds a whole "
            f"number from 1 to {N_MAX}, or is missing"
        )
    out = np.full(v.shape, N_MISSING, dtype=np.uint32)
    out[~missing] = present.astype(np.uint32)
    return out


# ── Flat-position helpers ───────────────────────────────────────────────────
#
# A cell's position is the only thing that identifies it to the overflow
# table, so every read and write of a fixed-point plane says where its values
# came from. These build the four shapes that actually occur.


class StoreCodec(StoreDecoder):
    """Encode and decode a store's statistic planes under a declared plan.

    One instance per plane-bearing component: a Hybrid release's Dense
    Component and its Ragged Overflow each carry their own overflow table
    while sharing one `StoreEncoding`. Decoding is the Store Reader's
    `StoreDecoder` (#464); the encoders here are its inverse, and code with the
    same plan, scale and markers it decodes with.
    """

    @staticmethod
    def _require_residual_inputs(v: np.ndarray, eaf: np.ndarray) -> None:
        """Refuse SEs a residual plane cannot code: negative or non-finite, or
        present where the frequency is not."""
        missing = np.isnan(v)
        invalid = ~missing & (~np.isfinite(v) | (v < 0))
        if np.any(invalid):
            raise ValueError("cannot store negative or non-finite standard errors")
        # A residual plane is defined over a complete-EAF store (#118,
        # #138-#140). A finite `se` whose cell has no frequency used to become
        # an exact exception, which stored the value but put the plane outside
        # that contract -- and a decode-time refusal would then reject data this
        # method had just written. It fails here instead, where the caller still
        # has the source value and can fall back to `float16` (issue #159).
        if np.any(~missing & ~np.isfinite(np.asarray(eaf, dtype=np.float64))):
            raise ValueError(
                "residual SE needs a finite EAF for every cell carrying a standard error; "
                "a store with frequencies missing where `se` is present must use float16"
            )

    def encode_se(
        self,
        values: np.ndarray,
        *,
        eaf: np.ndarray,
        analysis_index: np.ndarray,
        coefficients: np.ndarray,
        positions: PositionSource = None,
        exceptions: SeExceptionBuilder | None = None,
        exact: np.ndarray | None = None,
    ) -> np.ndarray:
        """Code standard errors under the plan; `float16`, or residuals with exceptions.

        `exact` marks cells whose value must decode to exactly itself -- a value
        decoded from another Store and carried into this one (issue #301). Under
        a residual plan such a cell whose code would decode to anything else is
        an exact exception; a cell already on a code's value keeps the code.
        An SE above `SE_FLOAT16_MAX` raises `SeOutOfRange` under every plan; under
        a residual one the exceptions are `se_residual_codes`'s (#338).
        """
        v = np.asarray(values, dtype=np.float64)
        require_se_in_range(v, where=f"encode_se ({self.encoding.se.kind})")
        if not self.encoding.se.is_residual:
            return se_float16(v)
        self._require_se_model(eaf, analysis_index, coefficients)
        if np.shape(eaf) != v.shape:
            raise ValueError(f"EAF shape {np.shape(eaf)} does not match se shape {v.shape}")
        self._require_residual_inputs(v, eaf)
        out, exceptional = se_residual_codes(
            v, eaf, analysis_index, coefficients, self.encoding.se.step, exact=exact
        )
        if np.any(exceptional):
            if exceptions is None:
                raise ValueError("se exceptions need a SeExceptionBuilder")
            exceptions.add(positions_for(exceptional, positions, what="encode_se"), v[exceptional])
        return out

    def encode_z(
        self,
        values: np.ndarray,
        *,
        positions: PositionSource = None,
        overflow: ZOverflowBuilder | None = None,
    ) -> np.ndarray:
        """Float z-scores -> stored bytes.

        NaN becomes the missing marker; a value outside the representable
        range becomes the overflow marker and is handed to `overflow` exactly.
        A non-finite value that is not NaN (±inf, or a malformed statistic)
        **fails the build**: it is neither a value nor an absence, and there is
        no honest way to store it.
        """
        v = np.asarray(values, dtype=np.float64)
        if not self.encoding.z.is_fixed_point:
            return v.astype(np.float16)

        missing = np.isnan(v)
        broken = ~missing & ~np.isfinite(v)
        if np.any(broken):
            example = v[broken].ravel()[:3].tolist()
            raise ValueError(
                f"cannot store non-finite z {example} -- a statistic that is neither a "
                "value nor a recorded absence is malformed at source, and must fail the "
                "build rather than be encoded as something (ADR 0037 §1)"
            )
        scaled, in_range = self._scale(v, missing)
        codes = np.where(in_range, scaled, np.float64(Z_MISSING)).astype(np.int64)
        codes[~missing & ~in_range] = Z_OVERFLOW

        out_of_range = ~missing & ~in_range
        if np.any(out_of_range):
            if overflow is None:
                raise ValueError(
                    "z values outside the representable range "
                    f"[{self.encoding.z.min_representable:.6g}, "
                    f"{self.encoding.z.max_representable:.6g}] need an overflow table to "
                    "hold them exactly, but no ZOverflowBuilder was supplied"
                )
            overflow.add(
                positions_for(out_of_range, positions, what="encode_z"),
                v[out_of_range].astype(np.float32),
            )
        return codes.astype(np.int16)

    def _scale(self, v: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`(round(v * scale), which cells the plane can hold)`.

        Shared by `encode_z` and `quantise_z` so the two cannot disagree about
        where the representable range ends -- which is the boundary that
        decides whether a cell needs an overflow entry.
        """
        scale = self.encoding.z.scale
        assert scale is not None
        scaled = np.rint(np.where(missing, 0.0, v) * scale)
        in_range = ~missing & (scaled >= Z_OVERFLOW + 1) & (scaled <= Z_CODE_MAX)
        return scaled, in_range

    def quantise_z(self, values: np.ndarray) -> np.ndarray:
        """What a query will read back for `values`, without storing them.

        For the top-hit harvest and the source-fidelity check, both of which
        must compare against the *stored* value rather than the source one --
        thresholding on an unrounded z produces an index the store itself
        contradicts (issue #046).
        """
        v = np.asarray(values, dtype=np.float64)
        if not self.encoding.z.is_fixed_point:
            return np.asarray(v.astype(np.float16), dtype=np.float32)
        missing = np.isnan(v)
        scaled, in_range = self._scale(v, missing)
        out = np.where(
            in_range,
            scaled.astype(np.float32) / np.float32(self.encoding.z.scale),
            v.astype(np.float32),
        ).astype(np.float32)
        out[missing] = np.float32(np.nan)
        return out

    def encode_eaf(
        self,
        values: np.ndarray,
        *,
        baseline: np.ndarray | None = None,
        positions: PositionSource = None,
        exceptions: EafExceptionBuilder | None = None,
    ) -> np.ndarray:
        """Frequencies -> stored bytes.

        `baseline` is the per-*cell* baseline -- the variant's `eaf_baseline`
        already gathered onto the cells being written -- because the codec does
        not know how a layout maps cells to variants, and a codec that guessed
        would be the crossover bug of issue #99 in a new place.

        NaN becomes the absent code. A frequency of exactly 0 or 1, a cell at a
        variant with no usable baseline, and a residual outside the declared
        range all become the exception code and are handed to `exceptions`
        exactly. A frequency outside `[0, 1]` fails: it is not a frequency.
        """
        v = np.asarray(values, dtype=np.float64)
        absent = np.isnan(v)
        # Checked before the kind is consulted: an out-of-range frequency is
        # malformed whether the plane that would hold it is `float32` or a
        # residual, and a `float32` plane that accepted it would carry it all
        # the way to a query (spec §6a).
        invalid = ~absent & (~np.isfinite(v) | (v < 0.0) | (v > 1.0))
        if np.any(invalid):
            example = v[invalid].ravel()[:3].tolist()
            raise ValueError(
                f"cannot store {example} as an effect allele frequency: a value outside "
                "[0, 1] is not a frequency, and must fail the build rather than be "
                "encoded as something (ADR 0036 §9)"
            )
        if not self.encoding.eaf.is_residual:
            return v.astype(np.float32)

        if baseline is None:
            raise EafBaselineError(
                "encoding an int8_residual eaf plane needs the per-cell eaf_baseline; "
                "without it there is nothing for the residual to be relative to"
            )
        b = np.asarray(baseline, dtype=np.float32).astype(np.float64)
        if b.shape != v.shape:
            raise EafBaselineError(
                f"baseline shape {b.shape} does not match eaf values shape {v.shape}"
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            residual = logit(v) - logit(b)
            codes = np.rint(residual / self.encoding.eaf.step)
        codable = ~absent & np.isfinite(codes) & (codes >= EAF_CODE_MIN) & (codes <= EAF_CODE_MAX)
        out = np.where(codable, codes, np.float64(EAF_EXCEPTION))
        out[absent] = EAF_ABSENT
        exceptional = ~absent & ~codable
        if np.any(exceptional):
            if exceptions is None:
                raise ValueError(
                    "eaf values that the residual coding cannot express need an exception "
                    "table to hold them exactly, but no EafExceptionBuilder was supplied"
                )
            exceptions.add(
                positions_for(exceptional, positions, what="encode_eaf"),
                v[exceptional].astype(np.float32),
            )
        return out.astype(np.int8)


def write_exact_table(
    table: SparseExactTable,
    group: Any,
    *,
    compressor: Any = None,
    role: ArrayRole | None = None,
) -> None:
    """Write `table` beside its plane, replacing any existing one.

    Written even when empty, so "this plane is integer-coded" and "this
    plane has a table" are the same statement and validation can check it
    without a special case for the common store that overflows nothing.

    The `role` is explicit for the same reason as `write_eaf_baseline`'s:
    a Ragged CSR component opened directly at `data.zarr/ragged` reports
    `path == ""`, so the caller that knows its component names the Ragged
    role (issue #248, #247 review round 1).  The default is the Dense one.
    """
    for name, data, dtype in (
        (table.index_name, table.index, "int64"),
        (table.value_name, table.value, "float32"),
    ):
        create_array(
            group,
            name,
            role if role is not None else ArrayRole.EXCEPTION_TABLE,
            data=np.asarray(data, dtype=dtype),
            dtype=dtype,
            compressor=compressor,
            hint=EXACT_TABLE_CHUNK,
            overwrite=True,
        )
