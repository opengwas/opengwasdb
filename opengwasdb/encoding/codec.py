"""`StoreCodec` -- the only place stored statistic bytes become physical values.

Every builder, completion pass, query adapter and validation rule that touches
`z` goes through this module. The point is not the arithmetic, which is three
lines; it is that there is **one site**. Encoding multiplies the number of
places a plane's bytes are interpreted, and every silent defect this release
stage has closed was a single call site out of step with its siblings.

`decode_z` is public because z is independently meaningful -- Rho and the
top-hit harvest need nothing else to interpret it. There is deliberately no
public `decode_se(raw_se)`: once #118 codes `se` as a residual it cannot be
decoded without EAF, and a function that looks decodable without it would
invite exactly the half-done read that returns silent nonsense.

Cells an integer plane cannot hold live in a sparse side table keyed by the
cell's **flat position** in its plane -- row × n_analyses + column for a Dense
grid, the association's ordinal for a CSR. `z` calls its table an *overflow*
table (a value beyond the fixed-point range) and `eaf` an *exception* table (a
residual beyond the range, a frequency of 0 or 1, or a variant with no usable
baseline), but they are the same structure and the same lookup. Decoding such a
cell needs to know where it is, which is why the read helpers take a position
resolver rather than just the bytes.

`eaf` decoding needs two things beyond the bytes: the variant's baseline, and
-- on a Reference-Completed release that carries one -- the imputed mask and
the per-variant reference frequency. `decode_eaf` requires them rather than
defaulting them, because the failure mode of getting this half-right is a
panel frequency silently standing in for a cohort's own, which real data says
can be wrong by 3000x (ADR 0037 §4).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Self, cast

import numpy as np

from opengwasdb.encoding.plan import (
    EAF_ABSENT,
    EAF_CODE_MAX,
    EAF_CODE_MIN,
    EAF_EXCEPTION,
    Z_CODE_MAX,
    Z_MISSING,
    Z_OVERFLOW,
    EafBaselineError,
    StoreEncoding,
)

#: Where a plane's overflow table lives, in the same zarr group as the plane.
Z_OVERFLOW_INDEX = "z_overflow_index"
Z_OVERFLOW_VALUE = "z_overflow_value"

#: Where an `eaf` plane's exception table and per-variant baseline live.
EAF_EXCEPTION_INDEX = "eaf_exception_index"
EAF_EXCEPTION_VALUE = "eaf_exception_value"
EAF_BASELINE = "eaf_baseline"

#: The per-variant reference-panel frequency for a component's imputed cells.
EAF_REFERENCE = "eaf_reference"

#: Flat positions of some subset of a plane's cells: either an array parallel
#: to the values, or a callable taking the boolean mask of the cells that need
#: resolving and returning their positions. The callable form exists so a read
#: of a 10M-cell column does not have to materialise 10M positions to resolve
#: the two that overflowed.
PositionSource = np.ndarray | Callable[[np.ndarray], np.ndarray] | None


def _positions_for(mask: np.ndarray, positions: PositionSource, *, what: str) -> np.ndarray:
    if positions is None:
        raise ValueError(
            f"{what}: this plane has out-of-range (overflow) z cells, whose exact values "
            "live in the plane's overflow table keyed by flat position, but no positions "
            "were supplied to resolve them"
        )
    if callable(positions):
        return np.asarray(positions(mask), dtype=np.int64)
    resolved = np.asarray(positions)
    if resolved.shape != mask.shape:
        raise ValueError(
            f"{what}: positions shape {resolved.shape} does not match values shape {mask.shape}"
        )
    return np.asarray(resolved[mask], dtype=np.int64)


@dataclass(frozen=True)
class SparseExactTable:
    """Exact `float32` values for the cells an integer plane cannot hold.

    Sorted by flat position so a lookup is a binary search over a contiguous
    `int64` array. Tiny by construction -- 6,346 cells of 24.8e9 for `z` on
    `ukb-b`, a 74 KB side table against a multi-gigabyte store (ADR 0037).

    Subclasses supply only the array names and the word they use for the cells
    they hold, so `z` and `eaf` cannot drift apart in how they sort, dedupe,
    look up or refuse an entry that is not there.
    """

    index: np.ndarray  # int64, sorted, unique
    value: np.ndarray  # float32

    #: Overridden per plane.
    index_name: ClassVar[str] = ""
    value_name: ClassVar[str] = ""
    what: ClassVar[str] = ""

    @classmethod
    def empty(cls) -> Self:
        return cls(index=np.empty(0, dtype=np.int64), value=np.empty(0, dtype=np.float32))

    def __len__(self) -> int:
        return int(len(self.index))

    def lookup(self, positions: np.ndarray) -> np.ndarray:
        """Exact values at `positions`. A position with no entry is an error:
        a store whose plane says "look in the table" and whose table does not
        say what the value was has lost the cell, and it must not read as
        anything.
        """
        positions = np.asarray(positions, dtype=np.int64)
        slot = np.searchsorted(self.index, positions)
        found = slot < len(self.index)
        hit = np.zeros(len(positions), dtype=bool)
        hit[found] = self.index[slot[found]] == positions[found]
        if not np.all(hit):
            missing = positions[~hit][:5].tolist()
            raise ValueError(
                f"{self.what} table has no entry for cell(s) at flat "
                f"position(s) {missing}; the store's plane and its table disagree"
            )
        return np.asarray(self.value[slot], dtype=np.float32)

    @classmethod
    def read(cls, group: Any) -> Self:
        """Read the table beside its plane; empty when the store has none."""
        if cls.index_name not in group or cls.value_name not in group:
            return cls.empty()
        return cls(
            index=np.asarray(group[cls.index_name][:], dtype=np.int64),
            value=np.asarray(group[cls.value_name][:], dtype=np.float32),
        )

    def write(self, group: Any) -> None:
        """Write the table beside its plane, replacing any existing one.

        Written even when empty, so "this plane is integer-coded" and "this
        plane has a table" are the same statement and validation can check it
        without a special case for the common store that overflows nothing.
        """
        for name, data, dtype in (
            (self.index_name, self.index, "int64"),
            (self.value_name, self.value, "float32"),
        ):
            if name in group:
                del group[name]
            group.create_dataset(
                name, data=np.asarray(data, dtype=dtype), chunks=(max(1, len(self.index)),),
                dtype=dtype,
            )


class SparseExactBuilder:
    """Accumulates side-table cells during a write, in any order."""

    table_type: ClassVar[type[SparseExactTable]] = SparseExactTable

    def __init__(self) -> None:
        self._index: list[np.ndarray] = []
        self._value: list[np.ndarray] = []

    def add(self, positions: np.ndarray, values: np.ndarray) -> None:
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


@dataclass(frozen=True)
class ZOverflowTable(SparseExactTable):
    """Exact `float32` z for the cells the fixed-point plane cannot hold.

    Not a clip and not a build failure: the `ukb-b` survey found 568 genuine
    associations above |z| = 64 (ADR 0037 §1).
    """

    index_name: ClassVar[str] = Z_OVERFLOW_INDEX
    value_name: ClassVar[str] = Z_OVERFLOW_VALUE
    what: ClassVar[str] = "z overflow"


class ZOverflowBuilder(SparseExactBuilder):
    """Accumulates out-of-range z cells during a write, in any order."""

    table_type: ClassVar[type[SparseExactTable]] = ZOverflowTable

    def table(self) -> ZOverflowTable:
        return cast(ZOverflowTable, super().table())


@dataclass(frozen=True)
class EafExceptionTable(SparseExactTable):
    """Exact `float32` EAF for the cells the residual coding cannot express.

    Three kinds of cell land here, and all three are stored exactly rather
    than approximated: a residual outside the declared range, a frequency of
    exactly 0 or 1 (which has no logit), and a cell at a variant with no
    usable baseline (ADR 0037 §2).
    """

    index_name: ClassVar[str] = EAF_EXCEPTION_INDEX
    value_name: ClassVar[str] = EAF_EXCEPTION_VALUE
    what: ClassVar[str] = "eaf exception"


class EafExceptionBuilder(SparseExactBuilder):
    """Accumulates exception cells during a write, in any order."""

    table_type: ClassVar[type[SparseExactTable]] = EafExceptionTable

    def table(self) -> EafExceptionTable:
        return cast(EafExceptionTable, super().table())


# ── EAF baselines ───────────────────────────────────────────────────────────
#
# The baseline is a per-variant `float32`: the within-store representative
# *observed* frequency, against which every cell at that variant is coded as a
# residual. It is computed in logit space, because that is the space the
# residual lives in and a median taken in one space is not the median in the
# other once ties are averaged.


def logit(f: np.ndarray) -> np.ndarray:
    """`log(f / (1 - f))`, NaN outside the open interval (0, 1)."""
    f = np.asarray(f, dtype=np.float64)
    out = np.full(f.shape, np.nan, dtype=np.float64)
    usable = np.isfinite(f) & (f > 0.0) & (f < 1.0)
    out[usable] = np.log(f[usable]) - np.log1p(-f[usable])
    return out


def expit(x: np.ndarray) -> np.ndarray:
    """The inverse of `logit`, evaluated without overflowing at either tail."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(x.shape, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    tail = np.exp(x[~positive])
    out[~positive] = tail / (1.0 + tail)
    return out


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
        stored = np.where(np.isfinite(median_logit), expit(median_logit), np.nan).astype(
            np.float32
        )
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
    vi, lv = variant_index[usable], transformed[usable]
    order = np.lexsort((lv, vi))
    vi, lv = vi[order], lv[order]
    starts = np.flatnonzero(np.concatenate(([True], vi[1:] != vi[:-1])))
    counts = np.diff(np.concatenate((starts, [len(vi)])))
    lower = lv[starts + (counts - 1) // 2]
    upper = lv[starts + counts // 2]
    baseline[vi[starts]] = (lower + upper) / 2.0
    return _as_baseline(baseline)


class StoreCodec:
    """Encode and decode a store's statistic planes under a declared plan.

    One instance per plane-bearing component: a Hybrid release's Dense
    Component and its Ragged Overflow each carry their own overflow table
    while sharing one `StoreEncoding`.
    """

    def __init__(
        self,
        encoding: StoreEncoding,
        *,
        z_overflow: ZOverflowTable | None = None,
        eaf_exceptions: EafExceptionTable | None = None,
    ) -> None:
        self.encoding = encoding
        self.z_overflow = z_overflow
        self.eaf_exceptions = eaf_exceptions

    # ---- z ---------------------------------------------------------------

    @property
    def z_dtype(self) -> str:
        return self.encoding.z.dtype

    @property
    def z_fill_value(self) -> Any:
        """The value a freshly created z plane is filled with: "missing"."""
        return Z_MISSING if self.encoding.z.is_fixed_point else float("nan")

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
                _positions_for(out_of_range, positions, what="encode_z"),
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

    def decode_z(self, raw: np.ndarray, *, positions: PositionSource = None) -> np.ndarray:
        """Stored bytes -> `float32` z-scores, with NaN for missing cells."""
        codes = np.asarray(raw)
        if not self.encoding.z.is_fixed_point:
            # The guard matters more in this direction than the other: reading
            # a fixed-point plane as `float16` does not fail, it returns
            # z-scores a thousand times too large.
            if codes.dtype.kind in "iu":
                raise ValueError(
                    f"z plane has dtype {codes.dtype}, but this store declares "
                    f"{self.encoding.z.kind}; the store and its manifest disagree"
                )
            return np.asarray(codes, dtype=np.float32)

        if codes.dtype != np.int16:
            raise ValueError(
                f"z plane has dtype {codes.dtype}, but this store declares "
                f"{self.encoding.z.kind}; the store and its manifest disagree"
            )
        scale = self.encoding.z.scale
        assert scale is not None
        out = codes.astype(np.float32) / np.float32(scale)
        out[codes == Z_MISSING] = np.float32(np.nan)
        out_of_range = codes == Z_OVERFLOW
        if np.any(out_of_range):
            if self.z_overflow is None:
                raise ValueError(
                    "z plane has out-of-range (overflow) cells but this codec was built "
                    "without the plane's overflow table"
                )
            out[out_of_range] = self.z_overflow.lookup(
                _positions_for(out_of_range, positions, what="decode_z")
            )
        return out

    def missing_mask(self, raw: np.ndarray) -> np.ndarray:
        """Which cells of a stored z plane are missing (spec §15).

        A plane's missing marker is defined by its declared encoding, not by
        its dtype: NaN for a float plane, the reserved sentinel for an integer
        one. Callers checking missingness -- the paired-NaN invariant, the
        imputed-mask rules -- ask this rather than assuming NaN, which is what
        lets an integer plane express a contract it cannot hold a NaN for.
        """
        codes = np.asarray(raw)
        if not self.encoding.z.is_fixed_point:
            return np.asarray(np.isnan(codes), dtype=bool)
        return np.asarray(codes == Z_MISSING, dtype=bool)

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

    # ---- eaf --------------------------------------------------------------

    @property
    def eaf_dtype(self) -> str:
        return self.encoding.eaf.dtype

    @property
    def eaf_fill_value(self) -> Any:
        """The value a freshly created `eaf` plane is filled with: "no EAF"."""
        return EAF_ABSENT if self.encoding.eaf.is_residual else float("nan")

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
        codable = (
            ~absent
            & np.isfinite(codes)
            & (codes >= EAF_CODE_MIN)
            & (codes <= EAF_CODE_MAX)
        )
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
                _positions_for(exceptional, positions, what="encode_eaf"),
                v[exceptional].astype(np.float32),
            )
        return out.astype(np.int8)

    def decode_eaf(
        self,
        raw: np.ndarray,
        *,
        baseline: np.ndarray | None = None,
        positions: PositionSource = None,
        imputed: np.ndarray | None = None,
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """Stored bytes -> `float32` frequencies, with NaN where there is none.

        On a release whose plan declares `eaf_reference`, `imputed` and
        `reference` are **required**, not optional: an imputed cell's frequency
        is the panel's, and a decode that quietly skipped the substitution
        would return NaN for every imputed cell in a store that has the value.
        An observed cell never falls back to the panel -- FinnGen's frequencies
        differ from the EUR panel by up to 3000x (ADR 0037 §4).
        """
        codes = np.asarray(raw)
        if not self.encoding.eaf.is_residual:
            if codes.dtype.kind in "iu":
                raise ValueError(
                    f"eaf plane has dtype {codes.dtype}, but this store declares "
                    f"{self.encoding.eaf.kind}; the store and its manifest disagree"
                )
            out = np.asarray(codes, dtype=np.float32)
            return self._apply_eaf_reference(out, imputed=imputed, reference=reference)

        if codes.dtype != np.int8:
            raise ValueError(
                f"eaf plane has dtype {codes.dtype}, but this store declares "
                f"{self.encoding.eaf.kind}; the store and its manifest disagree"
            )
        if baseline is None:
            raise EafBaselineError(
                "decoding an int8_residual eaf plane needs the per-cell eaf_baseline; "
                "an int8 residual plane is meaningless without it (ADR 0037)"
            )
        b = np.asarray(baseline, dtype=np.float32).astype(np.float64)
        if b.shape != codes.shape:
            raise EafBaselineError(
                f"baseline shape {b.shape} does not match eaf plane shape {codes.shape}"
            )
        with np.errstate(invalid="ignore"):
            out = expit(logit(b) + codes.astype(np.float64) * self.encoding.eaf.step)
        out = out.astype(np.float32)
        out[codes == EAF_ABSENT] = np.float32(np.nan)
        exceptional = codes == EAF_EXCEPTION
        if np.any(exceptional):
            if self.eaf_exceptions is None:
                raise ValueError(
                    "eaf plane has exception cells but this codec was built without the "
                    "plane's exception table"
                )
            out[exceptional] = self.eaf_exceptions.lookup(
                _positions_for(exceptional, positions, what="decode_eaf")
            )
        return self._apply_eaf_reference(out, imputed=imputed, reference=reference)

    def _apply_eaf_reference(
        self,
        decoded: np.ndarray,
        *,
        imputed: np.ndarray | None,
        reference: np.ndarray | None,
    ) -> np.ndarray:
        """Substitute panel frequencies on this component's imputed cells."""
        if not self.encoding.eaf.reference:
            return decoded
        if imputed is None or reference is None:
            raise EafBaselineError(
                "this release declares reference EAF for its imputed cells, so decoding "
                "needs the imputed mask and the per-cell eaf_reference; without them "
                "every imputed cell would read as NaN in a store that holds the value "
                "(ADR 0037 §4)"
            )
        mask = np.asarray(imputed).astype(bool)
        panel = np.asarray(reference, dtype=np.float32)
        if mask.shape != decoded.shape or panel.shape != decoded.shape:
            raise EafBaselineError(
                f"imputed mask {mask.shape} and reference {panel.shape} must both match "
                f"the decoded eaf shape {decoded.shape}"
            )
        decoded = decoded.copy()
        decoded[mask] = panel[mask]
        return decoded

    def eaf_absent_mask(self, raw: np.ndarray) -> np.ndarray:
        """Which cells of a stored `eaf` plane carry no frequency.

        Read through the plane's declared encoding, like `missing_mask` for
        `z`: NaN for a float plane, the reserved code for a residual one.
        """
        codes = np.asarray(raw)
        if not self.encoding.eaf.is_residual:
            return np.asarray(np.isnan(codes), dtype=bool)
        return np.asarray(codes == EAF_ABSENT, dtype=bool)


# ── Flat-position helpers ───────────────────────────────────────────────────
#
# A cell's position is the only thing that identifies it to the overflow
# table, so every read and write of a fixed-point plane says where its values
# came from. These build the four shapes that actually occur.


def positions_flat(start: int) -> Callable[[np.ndarray], np.ndarray]:
    """A contiguous 1-D run of a CSR plane starting at flat position `start`."""

    def resolve(mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(mask).astype(np.int64) + start

    return resolve


def positions_at(indices: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """An arbitrary gather from a 1-D plane at flat positions `indices`."""
    flat = np.asarray(indices, dtype=np.int64)

    def resolve(mask: np.ndarray) -> np.ndarray:
        return np.asarray(flat[mask], dtype=np.int64)

    return resolve


def positions_row_band(
    row_start: int, n_analyses: int
) -> Callable[[np.ndarray], np.ndarray]:
    """A full-width row band of a Dense grid, rows `row_start...`."""
    offset = int(row_start) * int(n_analyses)

    def resolve(mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(mask).astype(np.int64) + offset

    return resolve


def positions_rows_cols(
    rows: Sequence[int] | np.ndarray, cols: Sequence[int] | np.ndarray, n_analyses: int
) -> Callable[[np.ndarray], np.ndarray]:
    """A Dense block at the cross product of `rows` x `cols`, in that order."""
    row_idx = np.asarray(rows, dtype=np.int64)
    col_idx = np.asarray(cols, dtype=np.int64)

    def resolve(mask: np.ndarray) -> np.ndarray:
        i, j = np.divmod(np.flatnonzero(mask).astype(np.int64), len(col_idx))
        return row_idx[i] * n_analyses + col_idx[j]

    return resolve


def positions_pairs(rows: np.ndarray, cols: np.ndarray, n_analyses: int) -> np.ndarray:
    """Elementwise Dense cells `(rows[i], cols[i])`."""
    return np.asarray(rows, dtype=np.int64) * int(n_analyses) + np.asarray(cols, dtype=np.int64)
