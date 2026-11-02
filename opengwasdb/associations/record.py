"""The association record every reader returns, and the presence rule (spec §15).

A cell is present exactly when neither its Z nor its SE carries its plane's
missing marker. Finiteness plays no part: an SE above a `float16` plane's range
is stored as `+inf` beside a finite Z, and that cell is an association like any
other (issues #337, #338). The rule is `states()`, here and nowhere else; every
reader, `validate` and `check-observed-view` classify through it (ADR 0070).

A decoded plane reads NaN exactly where it holds its marker -- a float plane's
marker is NaN, and an integer plane's codec decodes its sentinel to NaN -- so
`classify()` applies the rule to decoded values, and `states()` to markers read
straight from the stored codes.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from typing import Any

import numpy as np


class Status(IntEnum):
    """Association Status (spec §15), as a record's int8 `status` holds it."""

    MISSING = 0
    OBSERVED = 1
    IMPUTED = 2


#: The `association_status` labels, indexed by `Status`.
STATUS_LABELS = np.array(["missing", "observed", "imputed"], dtype=object)


class State(IntEnum):
    """A cell under spec §15: one of the three Statuses, or an invalid state.

    The first three equal their `Status`, so a valid cell's state is its status.
    """

    MISSING = 0
    OBSERVED = 1
    IMPUTED = 2
    #: One of Z and SE carries its missing marker and the other does not.
    PARTIAL = 3
    #: The imputed mask holds a value other than 0 and 1.
    FLAG_INVALID = 4
    #: The imputed mask says imputed, and both Z and SE are missing.
    IMPUTED_MISSING = 5
    #: A float plane's marker is a NaN other than the canonical one.
    NON_CANONICAL = 6


_INVALID_STATES = (
    State.PARTIAL,
    State.FLAG_INVALID,
    State.IMPUTED_MISSING,
    State.NON_CANONICAL,
)

#: The one NaN a float plane's missing marker is written as: numpy's `np.nan`
#: in that dtype. A decoded `float16` plane is `float32` with its payload kept,
#: so `0x7e00` arrives as `0x7fc00000` and any other NaN as something else.
_CANONICAL_NAN = {2: 0x7E00, 4: 0x7FC00000, 8: 0x7FF8000000000000}
_BITS = {2: np.uint16, 4: np.uint32, 8: np.uint64}


def non_canonical(values: np.ndarray, missing: np.ndarray | None = None) -> np.ndarray:
    """Which cells hold a NaN whose bits are not the canonical marker (spec §15).

    An integer plane's marker is a sentinel, so nothing there can be
    non-canonical. `missing` is `isnan(values)` when the caller has it already.
    """
    values = np.asarray(values)
    if values.dtype.kind != "f":
        return np.zeros(values.shape, dtype=bool)
    nan = np.isnan(values) if missing is None else missing
    size = values.dtype.itemsize
    odd: np.ndarray = nan & (values.view(_BITS[size]) != _CANONICAL_NAN[size])
    return odd

#: Cells `Tally.of` counts at a time (4 MiB of int8 states).
_TALLY_WINDOW = 1 << 22


def states(
    z_missing: np.ndarray,
    se_missing: np.ndarray,
    flags: np.ndarray | None = None,
    odd_marker: np.ndarray | None = None,
) -> np.ndarray:
    """Each cell's `State` (int8) from its markers and its imputed flag.

    `flags` is the imputed mask as stored; None means the component holds none,
    so every present cell is observed. A cell whose markers disagree is
    `PARTIAL` whatever its flag says, and one whose marker is a non-canonical NaN
    (`odd_marker`, from `non_canonical`) is `NON_CANONICAL` whatever else it is.
    """
    z_missing = np.asarray(z_missing, dtype=bool)
    se_missing = np.asarray(se_missing, dtype=bool)
    out = np.where(z_missing, np.int8(State.MISSING), np.int8(State.OBSERVED)).astype(np.int8)
    if flags is not None:
        flags = np.asarray(flags)
        one = flags == 1
        out[one & ~z_missing] = State.IMPUTED
        out[one & z_missing] = State.IMPUTED_MISSING
        out[(flags != 0) & ~one] = State.FLAG_INVALID
    out[z_missing != se_missing] = State.PARTIAL
    if odd_marker is not None:
        out[np.asarray(odd_marker, dtype=bool)] = State.NON_CANONICAL
    return out


def present(z: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Which decoded cells are present: neither Z nor SE is missing, whatever the flag."""
    found: np.ndarray = states(np.isnan(z), np.isnan(se)) == State.OBSERVED
    return found


def classify(z: np.ndarray, se: np.ndarray, flags: np.ndarray | None = None) -> np.ndarray:
    """Each decoded cell's `State`: a decoded plane reads NaN exactly at its marker."""
    z_missing, se_missing = np.isnan(z), np.isnan(se)
    return states(z_missing, se_missing, flags, _odd_markers(z, se, z_missing, se_missing))


def _odd_markers(
    z: np.ndarray, se: np.ndarray, z_missing: np.ndarray, se_missing: np.ndarray
) -> np.ndarray:
    odd: np.ndarray = non_canonical(z, z_missing) | non_canonical(se, se_missing)
    return odd


def require_mask(required: bool, present: bool, source: str) -> None:
    """Refuse a Reference-Completed component that carries no imputed mask.

    Its mask is what says which associations are imputed; read without one,
    every present cell would pass for observed (spec §15).
    """
    if required and not present:
        raise ValueError(
            f"{source}: a Reference-Completed component carries no imputed mask, so "
            "which associations are imputed cannot be read (spec §15)"
        )


@dataclass(frozen=True)
class Tally:
    """How many cells fall in each `State`, and the overlaps §15's rules name.

    `states()` gives one state per cell, so a partial cell flagged imputed is
    counted as `PARTIAL`; the four overlap counts keep what that hides, because
    `validate` reports each one by its own message.
    """

    by_state: tuple[int, ...] = (0,) * len(State)
    #: Z carries its marker and SE does not.
    z_only_missing: int = 0
    #: SE carries its marker and Z does not.
    se_only_missing: int = 0
    #: Flagged imputed where Z is missing.
    imputed_z_missing: int = 0
    #: Flagged imputed where SE is missing.
    imputed_se_missing: int = 0
    #: Flags other than 0 and 1, whatever the cell's markers.
    flags_not_binary: int = 0

    @classmethod
    def of_decoded(
        cls, z: np.ndarray, se: np.ndarray, flags: np.ndarray | None = None
    ) -> Tally:
        """The counts over decoded cells, their markers' payloads included."""
        z_missing, se_missing = np.isnan(z), np.isnan(se)
        return cls.of(z_missing, se_missing, flags, _odd_markers(z, se, z_missing, se_missing))

    @classmethod
    def of(
        cls,
        z_missing: np.ndarray,
        se_missing: np.ndarray,
        flags: np.ndarray | None = None,
        odd_marker: np.ndarray | None = None,
    ) -> Tally:
        """The counts over every cell, taken a window at a time.

        A window bounds the working set -- the per-cell states and the counting
        copy -- whatever the length: `validate` tallies a whole Ragged sequence
        at once, 127 million rows on OGS-00027.
        """
        z_missing = np.asarray(z_missing, dtype=bool).ravel()
        se_missing = np.asarray(se_missing, dtype=bool).ravel()
        raw = None if flags is None else np.asarray(flags).ravel()
        odd = None if odd_marker is None else np.asarray(odd_marker, dtype=bool).ravel()
        total = cls()
        for lo in range(0, len(z_missing), _TALLY_WINDOW):
            hi = lo + _TALLY_WINDOW
            total = total + cls._of_window(
                z_missing[lo:hi],
                se_missing[lo:hi],
                None if raw is None else raw[lo:hi],
                None if odd is None else odd[lo:hi],
            )
        return total

    @classmethod
    def _of_window(
        cls,
        z_missing: np.ndarray,
        se_missing: np.ndarray,
        flags: np.ndarray | None,
        odd_marker: np.ndarray | None,
    ) -> Tally:
        counts = np.bincount(
            states(z_missing, se_missing, flags, odd_marker), minlength=len(State)
        )
        imputed: np.ndarray | None = None
        not_binary = 0
        if flags is not None:
            imputed = flags == 1
            not_binary = int(np.count_nonzero(~imputed & (flags != 0)))
        return cls(
            by_state=tuple(int(c) for c in counts),
            z_only_missing=int(np.count_nonzero(z_missing & ~se_missing)),
            se_only_missing=int(np.count_nonzero(se_missing & ~z_missing)),
            imputed_z_missing=0 if imputed is None else int(np.count_nonzero(imputed & z_missing)),
            imputed_se_missing=(
                0 if imputed is None else int(np.count_nonzero(imputed & se_missing))
            ),
            flags_not_binary=not_binary,
        )

    def __add__(self, other: Tally) -> Tally:
        return Tally(
            by_state=tuple(a + b for a, b in zip(self.by_state, other.by_state, strict=True)),
            z_only_missing=self.z_only_missing + other.z_only_missing,
            se_only_missing=self.se_only_missing + other.se_only_missing,
            imputed_z_missing=self.imputed_z_missing + other.imputed_z_missing,
            imputed_se_missing=self.imputed_se_missing + other.imputed_se_missing,
            flags_not_binary=self.flags_not_binary + other.flags_not_binary,
        )

    def count(self, state: State) -> int:
        return self.by_state[state]

    @property
    def invalid(self) -> int:
        """Cells in a state spec §15 forbids."""
        return sum(self.by_state[s] for s in _INVALID_STATES)


class InvalidAssociations(ValueError):
    """A read met cells spec §15 forbids. Admission refuses them at build time
    (#357), so a Store holding one is damaged; the read stops rather than drop
    or relabel them.
    """

    def __init__(
        self, source: str, variant_index: np.ndarray, analysis_index: np.ndarray, state: np.ndarray
    ) -> None:
        state = np.asarray(state)
        bad = np.flatnonzero(np.isin(state, _INVALID_STATES))
        counts = ", ".join(
            f"{s.name.lower()} {n:,}"
            for s in _INVALID_STATES
            if (n := int(np.count_nonzero(state[bad] == s)))
        )
        first = ", ".join(
            f"(variant {int(variant_index[i])}, analysis {int(analysis_index[i])}: "
            f"{State(int(state[i])).name.lower()})"
            for i in bad[:5]
        )
        self.count = len(bad)
        super().__init__(
            f"{source}: {len(bad):,} association cell(s) violate spec §15 ({counts}); "
            f"first: {first}"
        )


def _index_column(values: Any, length: int) -> np.ndarray:
    if np.ndim(values) == 0:
        return np.full(length, values, dtype=np.int32)
    return np.asarray(values, dtype=np.int32)


@dataclass(frozen=True, eq=False)
class Associations(Mapping[str, np.ndarray]):
    """Decoded associations as parallel arrays, one row each.

    Read as a mapping it is the query result contract (ADR 0020, ADR 0070):
    ``variant_index``, ``analysis_index``, ``z``, ``se``, ``eaf``, ``n`` when
    the component carries the plane, and ``association_status`` as labels.

    Every row is valid under spec §15: a constructor that is handed an invalid
    cell raises `InvalidAssociations`. `z` and `se` are NaN exactly where
    `status` is `MISSING`; `±inf` is a value. `eaf` is NaN where there is no
    frequency, and the panel's on an imputed cell of a release carrying one.
    """

    variant_index: np.ndarray
    analysis_index: np.ndarray
    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray
    status: np.ndarray
    n: np.ndarray | None = None

    def __post_init__(self) -> None:
        fields = {
            "variant_index": np.int32,
            "analysis_index": np.int32,
            "z": np.float32,
            "se": np.float32,
            "eaf": np.float32,
            "status": np.int8,
        }
        for name, dtype in fields.items():
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=dtype))
        if self.n is not None:
            object.__setattr__(self, "n", np.asarray(self.n, dtype=np.float64))
        lengths = {len(getattr(self, k)) for k in self._column_names()}
        if len(lengths) > 1:
            raise ValueError(f"association columns differ in length: {sorted(lengths)}")

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def empty(cls, *, with_n: bool = False) -> Associations:
        none = np.empty(0)
        return cls(none, none, none, none, none, none, none if with_n else None)

    @classmethod
    def from_decoded(
        cls,
        variant_index: Any,
        analysis_index: Any,
        z: np.ndarray,
        se: np.ndarray,
        eaf: np.ndarray | None = None,
        imputed: np.ndarray | None = None,
        n: np.ndarray | None = None,
        *,
        source: str = "associations",
    ) -> Associations:
        """A record from columns already decoded, every row kept.

        Classifies each row by the presence rule and raises `InvalidAssociations`
        for any row spec §15 forbids. A scalar index is repeated down the column.
        `eaf` None means no frequency anywhere; `imputed` None, no imputed mask.
        """
        z = np.asarray(z, dtype=np.float32)
        se = np.asarray(se, dtype=np.float32)
        vi = _index_column(variant_index, len(z))
        ai = _index_column(analysis_index, len(z))
        state = classify(z, se, imputed)
        if np.any(state > State.IMPUTED):
            raise InvalidAssociations(source, vi, ai, state)
        if eaf is None:
            eaf = np.full(len(z), np.nan, dtype=np.float32)
        return cls(vi, ai, z, se, eaf, state, n)

    # ── the mapping view ────────────────────────────────────────────────────

    def _column_names(self) -> tuple[str, ...]:
        n = ("n",) if self.n is not None else ()
        return ("variant_index", "analysis_index", "z", "se", "eaf", *n, "status")

    def _keys(self) -> tuple[str, ...]:
        n = ("n",) if self.n is not None else ()
        return ("variant_index", "analysis_index", "z", "se", "eaf", *n, "association_status")

    @cached_property
    def labels(self) -> np.ndarray:
        """`status` as the object array of labels a result carries."""
        out: np.ndarray = STATUS_LABELS[self.status]
        return out

    def __getitem__(self, key: str) -> np.ndarray:
        if key == "association_status":
            return self.labels
        if key in ("variant_index", "analysis_index", "z", "se", "eaf") or (
            key == "n" and self.n is not None
        ):
            value: np.ndarray = getattr(self, key)
            return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys())

    def __len__(self) -> int:
        return len(self._keys())

    @property
    def count(self) -> int:
        """How many associations the record holds."""
        return len(self.z)

    # ── derived records ─────────────────────────────────────────────────────

    def take(self, index: Any) -> Associations:
        """The rows `index` selects -- a boolean mask, an order, or a slice."""
        return Associations(
            self.variant_index[index],
            self.analysis_index[index],
            self.z[index],
            self.se[index],
            self.eaf[index],
            self.status[index],
            None if self.n is None else self.n[index],
        )

    def where(self, keep: np.ndarray) -> Associations:
        return self.take(np.asarray(keep, dtype=bool))

    def head(self, limit: int | None) -> Associations:
        return self if limit is None else self.take(slice(0, limit))

    def observed_only(self, apply: bool = True) -> Associations:
        """Without the imputed rows (Observed-Only mode, spec §19); missing rows stay."""
        return self.where(self.status != Status.IMPUTED) if apply else self

    def present(self) -> Associations:
        """Without the missing rows."""
        return self.where(self.status != Status.MISSING)

    def remap_variants(self, to_shared: np.ndarray) -> Associations:
        """The same rows with `variant_index` translated through `to_shared`."""
        if not self.count:
            return self
        vi = np.asarray(to_shared)[self.variant_index.astype(np.int64)]
        return Associations(
            vi, self.analysis_index, self.z, self.se, self.eaf, self.status, self.n
        )

    def in_genomic_order(self) -> Associations:
        """Sorted by `(analysis_index, variant_index)`, ties in their current order."""
        return self.take(np.lexsort((self.variant_index, self.analysis_index)))

    @staticmethod
    def concat(parts: Sequence[Associations], *, with_n: bool = False) -> Associations:
        """The parts' rows in order. `with_n` says whether the result carries `n`;
        a part without it cannot join a result that does."""
        if with_n and any(p.n is None for p in parts):
            raise ValueError("a record without n cannot join a result carrying n")
        if not parts:
            return Associations.empty(with_n=with_n)
        return Associations(
            variant_index=_cat(parts, "variant_index"),
            analysis_index=_cat(parts, "analysis_index"),
            z=_cat(parts, "z"),
            se=_cat(parts, "se"),
            eaf=_cat(parts, "eaf"),
            status=_cat(parts, "status"),
            n=_cat(parts, "n") if with_n else None,
        )


@dataclass(frozen=True, eq=False)
class Inspection:
    """Every cell of a region, decoded, with its `State`; invalid cells included.

    For the checks that count what a read would refuse -- `validate` and
    `check-observed-view`. `flags` is the imputed mask as stored, or None when
    the component holds none.
    """

    variant_index: np.ndarray
    analysis_index: np.ndarray
    state: np.ndarray
    flags: np.ndarray | None
    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray
    n: np.ndarray | None = None

    @classmethod
    def of(
        cls,
        variant_index: Any,
        analysis_index: Any,
        z: np.ndarray,
        se: np.ndarray,
        eaf: np.ndarray,
        flags: np.ndarray | None,
        n: np.ndarray | None = None,
    ) -> Inspection:
        z = np.asarray(z, dtype=np.float32)
        return cls(
            variant_index=_index_column(variant_index, len(z)),
            analysis_index=_index_column(analysis_index, len(z)),
            state=classify(z, se, flags),
            flags=None if flags is None else np.asarray(flags),
            z=z,
            se=np.asarray(se, dtype=np.float32),
            eaf=np.asarray(eaf, dtype=np.float32),
            n=n,
        )

    def take(self, index: Any) -> Inspection:
        """The cells `index` selects, every column with them."""

        def pick(values: np.ndarray | None) -> np.ndarray | None:
            return None if values is None else values[index]

        return Inspection(
            variant_index=self.variant_index[index],
            analysis_index=self.analysis_index[index],
            state=self.state[index],
            flags=pick(self.flags),
            z=self.z[index],
            se=self.se[index],
            eaf=self.eaf[index],
            n=pick(self.n),
        )

    @staticmethod
    def concat(parts: Sequence[Inspection], *, with_n: bool) -> Inspection:
        """The parts' cells in order; a part without a mask contributes zeros."""
        if not parts:
            none = np.empty(0)
            return Inspection.of(none, none, none, none, none, None, none if with_n else None)
        return Inspection(
            variant_index=_cat(parts, "variant_index"),
            analysis_index=_cat(parts, "analysis_index"),
            state=_cat(parts, "state"),
            flags=_joined_flags(parts),
            z=_cat(parts, "z"),
            se=_cat(parts, "se"),
            eaf=_cat(parts, "eaf"),
            n=_cat(parts, "n") if with_n else None,
        )

    def tally(self) -> Tally:
        return Tally.of_decoded(self.z, self.se, self.flags)

    @property
    def present(self) -> np.ndarray:
        """Which cells are present: neither marker set, whatever the flag says."""
        mask: np.ndarray = ~(np.isnan(self.z) | np.isnan(self.se))
        return mask

    def associations(self, source: str) -> Associations:
        """Every cell as a record; raises `InvalidAssociations` if any is invalid."""
        if np.any(self.state > State.IMPUTED):
            raise InvalidAssociations(source, self.variant_index, self.analysis_index, self.state)
        return Associations(
            self.variant_index, self.analysis_index, self.z, self.se, self.eaf, self.state, self.n
        )


def _cat(parts: Sequence[Associations] | Sequence[Inspection], name: str) -> np.ndarray:
    """One column of every part, end to end."""
    joined: np.ndarray = np.concatenate([getattr(p, name) for p in parts])
    return joined


def _joined_flags(parts: Sequence[Inspection]) -> np.ndarray | None:
    """The parts' imputed masks end to end; None when no part has one."""
    if all(p.flags is None for p in parts):
        return None
    joined: np.ndarray = np.concatenate(
        [np.zeros(len(p.z), np.uint8) if p.flags is None else p.flags for p in parts]
    )
    return joined
