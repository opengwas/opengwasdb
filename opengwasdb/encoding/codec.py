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

Out-of-range cells (|z| above the plane's representable edge) live in a sparse
`ZOverflowTable` keyed by the cell's **flat position** in its plane -- row ×
n_analyses + column for a Dense grid, the association's ordinal for a CSR.
Decoding such a cell therefore needs to know where it is, which is why the
read helpers take a position resolver rather than just the bytes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from opengwasdb.encoding.plan import (
    Z_CODE_MAX,
    Z_MISSING,
    Z_OVERFLOW,
    StoreEncoding,
)

#: Where a plane's overflow table lives, in the same zarr group as the plane.
Z_OVERFLOW_INDEX = "z_overflow_index"
Z_OVERFLOW_VALUE = "z_overflow_value"

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
class ZOverflowTable:
    """Exact `float32` z for the cells the fixed-point plane cannot hold.

    Sorted by flat position so a lookup is a binary search. Tiny by
    construction -- 6,346 cells of 24.8e9 on `ukb-b`, a 74 KB side table
    against a multi-gigabyte store (ADR 0037).
    """

    index: np.ndarray  # int64, sorted, unique
    value: np.ndarray  # float32

    @classmethod
    def empty(cls) -> ZOverflowTable:
        return cls(index=np.empty(0, dtype=np.int64), value=np.empty(0, dtype=np.float32))

    def __len__(self) -> int:
        return int(len(self.index))

    def lookup(self, positions: np.ndarray) -> np.ndarray:
        """Exact values at `positions`. A position with no entry is an error:
        a store whose plane says "overflow" and whose table does not say what
        the value was has lost the association, and must not read as anything.
        """
        positions = np.asarray(positions, dtype=np.int64)
        slot = np.searchsorted(self.index, positions)
        found = slot < len(self.index)
        hit = np.zeros(len(positions), dtype=bool)
        hit[found] = self.index[slot[found]] == positions[found]
        if not np.all(hit):
            missing = positions[~hit][:5].tolist()
            raise ValueError(
                "z overflow table has no entry for out-of-range cell(s) at flat "
                f"position(s) {missing}; the store's z plane and its overflow table disagree"
            )
        return np.asarray(self.value[slot], dtype=np.float32)

    @classmethod
    def read(cls, group: Any) -> ZOverflowTable:
        """Read the table beside a z plane; empty when the store has none."""
        if Z_OVERFLOW_INDEX not in group or Z_OVERFLOW_VALUE not in group:
            return cls.empty()
        return cls(
            index=np.asarray(group[Z_OVERFLOW_INDEX][:], dtype=np.int64),
            value=np.asarray(group[Z_OVERFLOW_VALUE][:], dtype=np.float32),
        )

    def write(self, group: Any) -> None:
        """Write the table beside its z plane, replacing any existing one.

        Written even when empty, so "this plane is fixed-point" and "this
        plane has a table" are the same statement and validation can check it
        without a special case for the common store that overflows nothing.
        """
        for name, data, dtype in (
            (Z_OVERFLOW_INDEX, self.index, "int64"),
            (Z_OVERFLOW_VALUE, self.value, "float32"),
        ):
            if name in group:
                del group[name]
            group.create_dataset(
                name, data=np.asarray(data, dtype=dtype), chunks=(max(1, len(self.index)),),
                dtype=dtype,
            )


class ZOverflowBuilder:
    """Accumulates out-of-range cells during a write, in any order."""

    def __init__(self) -> None:
        self._index: list[np.ndarray] = []
        self._value: list[np.ndarray] = []

    def add(self, positions: np.ndarray, values: np.ndarray) -> None:
        self._index.append(np.asarray(positions, dtype=np.int64))
        self._value.append(np.asarray(values, dtype=np.float32))

    def __len__(self) -> int:
        return int(sum(len(part) for part in self._index))

    def table(self) -> ZOverflowTable:
        """Sort, and reject a plane that recorded one cell twice."""
        if not self._index:
            return ZOverflowTable.empty()
        index = np.concatenate(self._index)
        value = np.concatenate(self._value)
        order = np.argsort(index, kind="stable")
        index, value = index[order], value[order]
        if len(index) > 1 and np.any(index[1:] == index[:-1]):
            duplicate = int(index[1:][index[1:] == index[:-1]][0])
            raise ValueError(
                f"z overflow table has two entries for flat position {duplicate}; "
                "a cell was written twice with different values"
            )
        return ZOverflowTable(index=index, value=value)


class StoreCodec:
    """Encode and decode a store's statistic planes under a declared plan.

    One instance per plane-bearing component: a Hybrid release's Dense
    Component and its Ragged Overflow each carry their own overflow table
    while sharing one `StoreEncoding`.
    """

    def __init__(
        self, encoding: StoreEncoding, *, z_overflow: ZOverflowTable | None = None
    ) -> None:
        self.encoding = encoding
        self.z_overflow = z_overflow

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
