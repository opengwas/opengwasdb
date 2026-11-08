"""What an LD Store says about a pair, and which requests it answers (spec §8, §9).

These rules are the format's promise to its readers, so they are pure functions of the
store's recorded numbers: t_far and the distances it measured (spec §7). The reader
(ogld#4) and the queries (ogld#4, ogld#6) call them; none re-derives a boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt


class AnswerKind(StrEnum):
    """The Pair Answer for two Variant Set variants (spec §8, `CONTEXT.md`).

    Absent is not here: it is a variant's answer, not a pair's, and a reader reports it
    per variant.
    """

    STORED = "stored"
    """The Stored r: every pair within near_bp, and a farther one with r̂² ≥ t_far."""

    BOUND = "bound"
    """"r² < t_far": a pair more than near_bp and at most far_bp apart that is not stored."""

    NOT_MEASURED = "not_measured"
    """More than far_bp apart, or on different chromosomes: the store holds nothing."""


def answer_kind(
    *, same_chromosome: bool, distance: int, r_hat: float, t_far: float, near_bp: int, far_bp: int
) -> AnswerKind:
    """The kind of answer a store gives for a pair whose Stored r would be `r_hat` (spec §8).

    `distance` is |p_j − p_i| in bp. Boundaries are inclusive: d ≤ near_bp is the Near
    Band, near_bp < d ≤ far_bp is answered when r̂² ≥ t_far, and a larger d is Not
    Measured. The builder keeps a pair beyond near_bp exactly when this is STORED.
    """
    if distance < 0:
        raise ValueError(f"a distance is |p_j − p_i|, so never negative: {distance}")
    if not -1.0 <= r_hat <= 1.0:
        raise ValueError(f"r̂ must be a Stored r in [−1, 1]: {r_hat}")
    if not same_chromosome or distance > far_bp:
        return AnswerKind.NOT_MEASURED
    if distance <= near_bp or r_hat * r_hat >= t_far:
        return AnswerKind.STORED
    return AnswerKind.BOUND


def answer_kinds(
    r: npt.NDArray[np.float64],
    *,
    chrom: npt.NDArray[np.str_],
    position: npt.NDArray[np.integer],
    t_far: float,
    near_bp: int,
    far_bp: int,
) -> npt.NDArray[np.str_]:
    """The AnswerKind value of every cell of an unfilled signed-r matrix (spec §8).

    This is how an unstored pair's meaning is recovered from the matrix alone, as the
    owner decided on 7 November 2026: from the rows' chromosomes and positions and the
    store's t_far. A NaN more than `near_bp` and at most `far_bp` apart on one chromosome
    is the Bound; any other NaN is Not Measured. It reads only a matrix without a `fill`,
    and checks every cell against what the store can hold, so it labels nothing it
    cannot vouch for: a NaN within `near_bp` (the Near Band is complete), a number where
    the store holds nothing, and a number beyond `near_bp` below t_far (no stored pair
    there is) each raise, as a fill or a broken read.
    """
    same = chrom[:, None] == chrom[None, :]
    distance = np.abs(position[:, None].astype(np.int64) - position[None, :].astype(np.int64))
    near = same & (distance <= near_bp)
    far = same & (distance > near_bp) & (distance <= far_bp)
    unstored = np.isnan(r)
    if (unstored & near).any():
        raise ValueError("a NaN within near_bp: the Near Band is complete, so the read is broken")
    if (~unstored & ~near & ~far).any():
        raise ValueError(
            "a number where the store holds nothing (beyond far_bp, or across chromosomes): "
            "a fill or a broken read, not a Pair Answer (spec §8)"
        )
    with np.errstate(invalid="ignore"):
        below = ~unstored & far & (r * r < t_far)
    if below.any():
        raise ValueError(
            "a number beyond near_bp below t_far, where no pair is stored: a fill or a broken "
            "read, not a Pair Answer (spec §8)"
        )
    out = np.full(r.shape, AnswerKind.NOT_MEASURED.value, dtype="<U12")
    out[near | (far & ~unstored)] = AnswerKind.STORED.value
    out[far & unstored] = AnswerKind.BOUND.value
    return out


class RefusalReason(StrEnum):
    """Why a request is refused (spec §9). Each names the Pair Answer the store lacks."""

    INVALID = "invalid"
    """r2 outside [0, 1], a window or radius that is negative or not finite."""

    WINDOW_OVER_FAR = "window_over_far"
    """The window reaches past far_bp, where pairs are Not Measured."""

    THRESHOLD_BELOW_T_FAR = "threshold_below_t_far"
    """A window past near_bp with r2 < t_far: the store holds only the Bound there."""

    REGION_OVER_FAR = "region_over_far"
    """A region wider than far_bp (owner, 7 November 2026: radius ≤ 5 Mb)."""


class Refusal(Exception):
    """A request the LD Store declines rather than answer approximately (spec §9).

    It carries its `reason`. It is never replaced by a partial or approximate answer:
    that is the wrong answer that looks like a right one (ADR 0001).
    """

    def __init__(self, reason: RefusalReason, detail: str) -> None:
        super().__init__(f"refused ({reason.value}): {detail}")
        self.reason = reason
        self.detail = detail


class InputError(ValueError):
    """An input ogld cannot use, named: a missing call, a PC File without a kept sample, a
    BuildInputs that contradicts itself (`docs/spec/build-inputs.md`).

    Distinct from a fault inside ogld, such as `quantise` refusing r = 1.3: the CLI exits 2
    for an InputError or a Refusal, and lets any other exception propagate.
    """


@dataclass(frozen=True)
class Decision:
    """Whether a request is answered, and if not why (spec §9).

    `window_bp` is the request's window in bp, W = kb × 1,000, or a region's width.
    """

    refusal: RefusalReason | None
    detail: str
    window_bp: float

    @property
    def answered(self) -> bool:
        return self.refusal is None

    def raise_if_refused(self) -> None:
        """Raise the `Refusal` this decision names; return quietly when it is answered."""
        if self.refusal is not None:
            raise Refusal(self.refusal, self.detail)


def judge_request(*, r2: float, kb: float, t_far: float, near_bp: int, far_bp: int) -> Decision:
    """Answered or refused, for `clump` and `proxies` (spec §9).

    The pairs considered are those with d ≤ W = kb × 1,000 bp, and one reaches the
    threshold when r̂² ≥ r2. Answered when W ≤ near_bp, at any r2, because the Near Band
    is complete; and when near_bp < W ≤ far_bp with r2 ≥ t_far, because every pair the
    store left out there has r̂² < t_far ≤ r2. r2 = t_far exactly is answered. The
    distances and t_far are the store's recorded ones.
    """
    if not (math.isfinite(r2) and 0.0 <= r2 <= 1.0 and math.isfinite(kb) and kb >= 0):
        return Decision(
            RefusalReason.INVALID,
            f"r2 must be in [0, 1] and kb finite and ≥ 0: r2={r2}, kb={kb}",
            math.nan,
        )
    window = kb * 1000
    if window > far_bp:
        detail = (
            f"a {window:,.0f} bp window reaches past {far_bp:,} bp, where pairs are Not Measured"
        )
        return Decision(RefusalReason.WINDOW_OVER_FAR, detail, window)
    if window > near_bp and r2 < t_far:
        detail = (
            f"r2={r2} is below t_far={t_far}: beyond {near_bp:,} bp the store holds only the Bound "
            f"r² < t_far for unstored pairs, which does not decide r̂² ≥ r2"
        )
        return Decision(RefusalReason.THRESHOLD_BELOW_T_FAR, detail, window)
    return Decision(None, "answered", window)


def judge_region(*, radius: int, far_bp: int) -> Decision:
    """Answered or refused, for an LD matrix of a region (owner, 7 November 2026).

    The region is every variant within `radius` bp of its centre, so its width is
    2 × radius, and it is answered when that is at most the store's `far_bp`. Then no
    pair inside it is Not Measured: each is a Stored r or the Bound.
    """
    if radius < 0:
        return Decision(
            RefusalReason.INVALID, f"a radius is a distance, so never negative: {radius}", math.nan
        )
    width = 2 * radius
    if width > far_bp:
        detail = f"a region {width:,} bp wide is wider than the {far_bp:,} bp the store measured"
        return Decision(RefusalReason.REGION_OVER_FAR, detail, width)
    return Decision(None, "answered", width)
