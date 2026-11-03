"""How a component stores associations: what it may hold, and how it encodes them.

`storable` is the one judgement a writer makes before any plane is written:
every cell must be one Association Admission would admit (ADR 0072), and an SE
above what a `float16` plane can represent is counted per Analysis. Admission
guarantees the first for every source row, so a cell that breaks it here was
produced after admission -- a completion fill, a merged pass, a re-encode --
and the write is refused, naming Admission's rule. Nothing is filtered here.

`AssociationCodec` encodes a batch under one component's plan, in the one
order the planes allow: `storable`, then z, then EAF against each cell's
variant baseline, then SE against the EAF a reader will decode for the cell
(the panel's on an imputed one). `Placement` says where the batch sits, so the
baseline, the SE model and the side tables are each looked up by the right
index; `round_trip` is what a reader would read back, without a write.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from opengwasdb.encoding.codec import (
    EAF_BASELINE,
    EAF_REFERENCE,
    EafExceptionBuilder,
    PositionSource,
    SeExceptionBuilder,
    SideTables,
    StoreCodec,
    ZOverflowBuilder,
    positions_flat,
    positions_pairs,
    positions_row_band,
)
from opengwasdb.encoding.plan import StoreEncoding
from opengwasdb.encoding.planes import SE_COEFFICIENTS

log = logging.getLogger(__name__)

__all__ = [
    "SE_FLOAT16_MAX",
    "AssociationCodec",
    "CellValues",
    "Encoded",
    "Placement",
    "Storable",
    "log_se_out_of_range",
    "storable",
]

#: The largest SE a `float16` plane holds as itself (65,504). A larger one is
#: stored as `+inf`, which is still an association (spec §15, issue #337).
SE_FLOAT16_MAX: Final = float(np.finfo(np.float16).max)

#: Cells `storable` checks at a time, so a Dense band costs window-sized masks.
_WINDOW = 1 << 22


@dataclass(frozen=True)
class CellValues:
    """A batch of association cells' statistics, before they are encoded.

    Parallel arrays of one shape, a column, a cell list or a row band. NaN is
    missing in `z` and `se`, and absent in `eaf`; an SE of `+inf` is a value.
    `eaf` is None when the batch carries no frequency plane, and `imputed`
    (the 0/1 mask) when every present cell is observed.
    """

    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray | None = None
    imputed: np.ndarray | None = None


@dataclass(frozen=True)
class Storable:
    """What a Store holds of a batch, and what the batch counted."""

    #: The cells to encode.
    cells: CellValues
    #: Which cells hold an association once stored.
    stored: np.ndarray
    #: Per Analysis, how many stored SEs exceed `SE_FLOAT16_MAX`, `+inf` included.
    se_out_of_range: np.ndarray


def storable(
    cells: CellValues, analyses: np.ndarray | int, n_analyses: int, *, where: str
) -> Storable:
    """What a Store holds of `cells`, whose Analysis indices are `analyses`.

    `analyses` broadcasts against the cells' shape: one Analysis's index for a
    column, one per cell for a cell list, one per column for a row band.
    `where` names the writer, for the refusal. Pure and idempotent: the plan
    plays no part, so a writer calls it before the SE plan exists.

    Raises `opengwasdb.admission.NotAdmitted`, through Admission's own check,
    when a cell is one Admission would not admit (one of z and SE missing, an
    infinite z, an SE not positive, an imputed flag that is not 0/1 or sits on
    a missing cell). An SE of `+inf` is not refused: it is a stored value.
    """
    z, se, imputed = _parallel(cells, where)
    owner = np.broadcast_to(np.asarray(analyses, dtype=np.int64), z.shape)
    over = np.zeros(n_analyses, dtype=np.int64)
    for window in _windows(z.shape):
        z_w, se_w = z[window], se[window]
        imputed_w = None if imputed is None else imputed[window]
        if _breaks_admission(z_w, se_w, imputed_w):
            _refuse(z_w, se_w, imputed_w, where=_where(where, window, z.shape))
        over += _out_of_range(se_w, owner[window], n_analyses)
    return Storable(cells=cells, stored=~np.isnan(z), se_out_of_range=over)


@dataclass(frozen=True)
class Placement:
    """Where a batch of cells sits in one component's planes.

    `rows` and `analyses` broadcast against the batch's shape; `positions` is
    each cell's flat position, which keys every side table (spec §6a).
    """

    #: Each cell's variant row: what gathers its EAF baseline and panel EAF.
    rows: np.ndarray
    #: Each cell's Analysis index: what gathers its SE model.
    analyses: np.ndarray
    positions: PositionSource
    n_analyses: int

    @classmethod
    def band(cls, r0: int, r1: int, n_analyses: int) -> Placement:
        """Rows `[r0:r1)` of a Dense grid, every Analysis: a `(r1 - r0, n_analyses)` batch."""
        return cls(
            rows=np.arange(r0, r1, dtype=np.int64)[:, None],
            analyses=np.arange(n_analyses, dtype=np.int64)[None, :],
            positions=positions_row_band(r0, n_analyses),
            n_analyses=n_analyses,
        )

    @classmethod
    def pairs(cls, rows: np.ndarray, cols: np.ndarray, n_analyses: int) -> Placement:
        """The Dense cells `(rows[i], cols[i])`, as `Cells.pairs` names them."""
        rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
        return cls(rows, cols, positions_pairs(rows, cols, n_analyses), n_analyses)

    @classmethod
    def sequence(
        cls,
        start: int,
        variant_index: np.ndarray,
        analysis_index: np.ndarray | int,
        n_analyses: int,
    ) -> Placement:
        """A run of an association sequence from flat position `start`."""
        return cls(
            np.asarray(variant_index, dtype=np.int64),
            np.asarray(analysis_index, dtype=np.int64),
            positions_flat(start),
            n_analyses,
        )


@dataclass(frozen=True)
class Encoded:
    """A batch's codes in the plan's dtypes, and the side-table entries it owes."""

    z: np.ndarray
    se: np.ndarray
    #: None when the batch carries no frequency plane.
    eaf: np.ndarray | None
    #: The imputed mask as stored, or None when the batch carries none.
    imputed: np.ndarray | None
    #: Exactly the batch's exceptional cells, keyed by `Placement.positions`.
    tables: SideTables
    storable: Storable


class AssociationCodec:
    """How one component stores associations under one plan (ADR 0037, spec §6a, §15).

    `eaf_baseline` and `eaf_reference` are per variant row of the component;
    `se_coefficients` is `(n_analyses, 2)`, needed by a residual SE plan only.
    A preliminary plan whose SE is still `float16` round-trips EAF before the
    SE model exists.
    """

    def __init__(
        self,
        plan: StoreEncoding,
        *,
        eaf_baseline: np.ndarray | None = None,
        eaf_reference: np.ndarray | None = None,
        se_coefficients: np.ndarray | None = None,
    ) -> None:
        self.plan = plan
        self._codec = StoreCodec(plan)
        self._baseline = _float32_or_none(eaf_baseline)
        self._reference = _float32_or_none(eaf_reference)
        self._coefficients = _float32_or_none(se_coefficients)

    @classmethod
    def of_component(cls, group: Any, plan: StoreEncoding) -> AssociationCodec:
        """The codec for a written component: its baseline, panel EAF and SE model."""

        def held(name: str) -> np.ndarray | None:
            return np.asarray(group[name][:], dtype=np.float32) if name in group else None

        return cls(
            plan,
            eaf_baseline=held(EAF_BASELINE),
            eaf_reference=held(EAF_REFERENCE),
            se_coefficients=held(SE_COEFFICIENTS) if plan.se.is_residual else None,
        )

    def encode(self, cells: CellValues, at: Placement, *, where: str) -> Encoded:
        """`storable`, then each plane's codes, in the order the planes allow.

        No I/O and deterministic. Raises `NotAdmitted` (`storable`),
        `EafBaselineError` (a residual EAF plan with no baseline), and
        `ValueError` for what the plan cannot hold: a frequency outside
        [0, 1], a frequency where the plan declares no EAF plane, or a
        residual SE with no model or no decoded frequency (issue #159).
        """
        kept = storable(cells, at.analyses, at.n_analyses, where=where)
        values = kept.cells
        z_overflow = ZOverflowBuilder()
        z = self._codec.encode_z(values.z, positions=at.positions, overflow=z_overflow)
        eaf_exceptions = EafExceptionBuilder()
        eaf = self._encode_eaf(values, at, eaf_exceptions, where)
        se_exceptions = SeExceptionBuilder()
        se = self._encode_se(values, at, eaf, eaf_exceptions, se_exceptions)
        imputed = None if values.imputed is None else np.asarray(values.imputed, dtype=np.uint8)
        tables = SideTables(z_overflow.table(), eaf_exceptions.table(), se_exceptions.table())
        return Encoded(z, se, eaf, imputed, tables, kept)

    def round_trip(self, cells: CellValues, at: Placement, *, where: str) -> CellValues:
        """What a reader decodes for `cells` once stored at `at`; nothing is written."""
        encoded = self.encode(cells, at, where=where)
        decoder = encoded.tables.decoder(self.plan)
        shape = np.shape(encoded.z)
        eaf = self._decoded_eaf(decoder, encoded.eaf, encoded.imputed, at, shape)
        if self.plan.se.is_residual:
            se = decoder.decode_se(
                encoded.se,
                eaf=eaf,
                analysis_index=np.broadcast_to(at.analyses, shape),
                coefficients=self._model(),
                positions=at.positions,
            )
        else:
            se = decoder.decode_se(
                encoded.se,
                eaf=eaf,
                analysis_index=np.zeros(shape, np.int64),
                coefficients=np.zeros((0, 2), np.float32),
            )
        return CellValues(
            decoder.decode_z(encoded.z, positions=at.positions),
            se,
            None if encoded.eaf is None else eaf,
            cells.imputed,
        )

    # ── the planes, in order ────────────────────────────────────────────────

    def _encode_eaf(
        self, values: CellValues, at: Placement, exceptions: EafExceptionBuilder, where: str
    ) -> np.ndarray | None:
        if values.eaf is None:
            return None
        if self.plan.eaf.is_absent:
            held = int(np.count_nonzero(~np.isnan(np.asarray(values.eaf))))
            if held:
                raise ValueError(
                    f"{where}: {held:,} cell(s) carry a frequency, but the plan declares "
                    "no EAF plane to hold one"
                )
            return None
        return self._codec.encode_eaf(
            values.eaf,
            baseline=self._per_cell(self._baseline, at, np.shape(values.eaf)),
            positions=at.positions,
            exceptions=exceptions,
        )

    def _encode_se(
        self,
        values: CellValues,
        at: Placement,
        eaf_codes: np.ndarray | None,
        eaf_exceptions: EafExceptionBuilder,
        exceptions: SeExceptionBuilder,
    ) -> np.ndarray:
        shape = np.shape(values.se)
        if not self.plan.se.is_residual:
            return self._codec.encode_se(
                values.se,
                eaf=np.empty(shape, np.float32),
                analysis_index=np.zeros(shape, np.int64),
                coefficients=np.zeros((0, 2), np.float32),
            )
        decoder = StoreCodec(self.plan, eaf_exceptions=eaf_exceptions.table())
        imputed = None if values.imputed is None else np.asarray(values.imputed, dtype=np.uint8)
        eaf = self._decoded_eaf(decoder, eaf_codes, imputed, at, shape)
        return self._codec.encode_se(
            values.se,
            eaf=eaf,
            analysis_index=np.broadcast_to(at.analyses, shape),
            coefficients=self._model(),
            positions=at.positions,
            exceptions=exceptions,
        )

    def _decoded_eaf(
        self,
        decoder: StoreCodec,
        codes: np.ndarray | None,
        imputed: np.ndarray | None,
        at: Placement,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        """The EAF a reader decodes at these cells: the panel's on an imputed one."""
        reference = self._per_cell(self._reference, at, shape) if self.plan.eaf.reference else None
        flags = np.zeros(shape, np.uint8) if imputed is None else imputed
        if codes is None:
            # No frequency plane: NaN, unless the panel's stands in on an imputed cell.
            no_plane = StoreEncoding(z=self.plan.z, se=self.plan.se).with_eaf_reference(
                self.plan.eaf.reference
            )
            blank = np.full(shape, np.nan, np.float32)
            return StoreCodec(no_plane).decode_eaf(blank, imputed=flags, reference=reference)
        return decoder.decode_eaf(
            codes,
            baseline=self._per_cell(self._baseline, at, shape),
            positions=at.positions,
            imputed=flags,
            reference=reference,
        )

    def _model(self) -> np.ndarray:
        if self._coefficients is None:
            raise ValueError("residual SE needs se_coefficients, and this codec was given none")
        return self._coefficients

    @staticmethod
    def _per_cell(
        per_variant: np.ndarray | None, at: Placement, shape: tuple[int, ...]
    ) -> np.ndarray | None:
        if per_variant is None:
            return None
        return np.broadcast_to(per_variant[at.rows], shape)


def _float32_or_none(values: np.ndarray | None) -> np.ndarray | None:
    return None if values is None else np.asarray(values, dtype=np.float32)


def log_se_out_of_range(counts: np.ndarray, analysis_ids: Sequence[str], *, where: str) -> None:
    """Log, as one warning, each Analysis's SEs above `SE_FLOAT16_MAX`.

    Such an SE is stored as `+inf` and stays an association; the count is the
    one issue #338 makes each Store record. Nothing is logged for none.
    """
    counts = np.asarray(counts, dtype=np.int64)
    found = np.flatnonzero(counts)
    if not len(found):
        return
    named = ", ".join(f"{analysis_ids[i]}: {int(counts[i]):,}" for i in found.tolist())
    log.warning(
        "%s: %s SE value(s) above %s, stored as +inf (spec §15): %s",
        where,
        f"{int(counts.sum()):,}",
        f"{SE_FLOAT16_MAX:,.0f}",
        named,
    )


def _parallel(cells: CellValues, where: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    z, se = np.asarray(cells.z), np.asarray(cells.se)
    imputed = None if cells.imputed is None else np.asarray(cells.imputed)
    if se.shape != z.shape or (imputed is not None and imputed.shape != z.shape):
        raise ValueError(
            f"{where}: z {z.shape}, se {se.shape} and imputed "
            f"{None if imputed is None else imputed.shape} must be parallel"
        )
    return z, se, imputed


def _out_of_range(se: np.ndarray, owner: np.ndarray, n_analyses: int) -> np.ndarray:
    """Per Analysis, the SEs above `SE_FLOAT16_MAX` among these cells."""
    with np.errstate(invalid="ignore"):
        over = se > SE_FLOAT16_MAX
    return np.bincount(owner[over], minlength=n_analyses).astype(np.int64)


def _windows(shape: tuple[int, ...]) -> list[tuple[slice, ...]]:
    """Leading-axis slices of at most `_WINDOW` cells (one row at least)."""
    if not shape:
        return [()]
    width = int(np.prod(shape[1:], dtype=np.int64)) or 1
    step = max(1, _WINDOW // width)
    return [(slice(i, min(i + step, shape[0])),) for i in range(0, max(shape[0], 1), step)]


def _breaks_admission(z: np.ndarray, se: np.ndarray, imputed: np.ndarray | None) -> bool:
    """Whether any cell breaks Admission's guarantee, `+inf` SE allowed; in the
    cells' own dtype, so a band is not copied to `float64` to find nothing."""
    z_missing, se_missing = np.isnan(z), np.isnan(se)
    if np.any(z_missing != se_missing):
        return True
    present = ~z_missing
    with np.errstate(invalid="ignore"):
        if np.any(present & (~np.isfinite(z) | ~(se > 0))):
            return True
    if imputed is not None:
        flagged = imputed != 0
        if np.any(flagged & ((imputed != 1) | z_missing)):
            return True
    return False


def _refuse(z: np.ndarray, se: np.ndarray, imputed: np.ndarray | None, *, where: str) -> None:
    """Raise Admission's own refusal for these cells.

    An SE narrowed to `+inf` is handed on as `SE_FLOAT16_MAX`, so the check
    judges every other rule and names the cell that broke one.
    """
    # Imported on refusal only: Admission's package imports the association
    # readers, whose Ragged adapter imports the writer that imports this module.
    from opengwasdb.admission.cells import require_admitted

    finite_se = np.where(np.isposinf(se), np.float32(SE_FLOAT16_MAX), se)
    require_admitted(z, finite_se, imputed=imputed, where=where)
    raise AssertionError(f"{where}: a cell broke Admission's guarantee, yet its check passed")


def _where(where: str, window: tuple[slice, ...], shape: tuple[int, ...]) -> str:
    """`where`, naming the window when the batch spans more than one."""
    if not window or (window[0].start == 0 and window[0].stop == shape[0]):
        return where
    return f"{where} (from index {window[0].start:,} of the batch's first axis)"
