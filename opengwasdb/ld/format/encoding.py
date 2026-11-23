"""How r is stored, and which pairs beyond the Near Band an LD Store keeps (ld-store §3).

An LD Store records its encoding (ld-store §7), and `REncoding` is that record as a value: the one
owner of the dtype, the scale, the reserved code and the rounding, and so of quantising,
dequantising, the step and the far code floor (ADR 0090). The builder (ogld#5) quantises with
the encoding it was given, the reader (ogld#4) decodes with the one the store records, and
validation recomputes the floor with it. No other module holds a scale, a dtype or a reserved
code of its own, so no component can round, decode or threshold at another store's scale.

t_far and the distances are not the encoding's: they depend on the panel and on ADR 0087, and a
store records them beside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Final, TypeGuard

import numpy as np
import numpy.typing as npt

from opengwasdb.ld.format.provenance import Encoding
from opengwasdb.ld.format.semantics import InputError

#: The χ²₁ upper quantile at p = 5×10⁻⁶ in t_far = max(0.001, q / (n − k − 1)) (ADR 0087).
T_FAR_Q: Final = 20.84

#: t_far's floor: MR's default r² threshold (ADR 0087).
T_FAR_MIN: Final = 0.001

#: The builder's distances (ADR 0087): the Near Band is complete up to NEAR_BP, and the
#: Far Pairs and Partner Rows reach FAR_BP. A store records the distances it measured (ld-store §7)
#: and readers use those, never these.
NEAR_BP: Final = 3_000_000
FAR_BP: Final = 10_000_000

# A dot product of standardised vectors can land a few ulps past ±1. Further than this is
# not rounding error but a builder fault, and clamping it would store a plausible r = ±1.
_R_SLACK: Final = 1e-6

#: An array of codes, in the dtype every admitted encoding's container has (ADR 0090).
type Codes = npt.NDArray[np.int16]


@dataclass(frozen=True)
class REncoding:
    """One LD Store's encoding of r: r̂ = code / `scale`, in an int16 container (ld-store §3).

    Every admitted encoding shares the container (owner, 10 November 2026; ADR 0090): int16
    codes, −32768 reserved and every `r` array's fill value, and rounding half to even. Only
    the scale differs, a whole number from 1 to 32767, so the scale is symmetric: negating a
    code never overflows, r = ±1 is exact, and a lost chunk still reads as a code that is never
    a correlation. The HGDP+1kGP stores are × 511 and UK Biobank EUR × 2047. There is no
    default: a build is given its scale and a store records it.
    """

    scale: int

    #: The container's dtype, the same for every admitted encoding.
    dtype: ClassVar[np.dtype[np.int16]] = np.dtype(np.int16)
    #: Never a correlation. Every `r` array's fill value, so a chunk that was never written
    #: cannot read as r = 0 (ld-store §3, §6).
    reserved_code: ClassVar[int] = int(np.iinfo(np.int16).min)
    #: The largest scale the container holds symmetrically: the other end from the reserved code.
    largest_scale: ClassVar[int] = int(np.iinfo(np.int16).max)
    #: How r × scale becomes a code: `numpy.rint`, which rounds half to even.
    rounding: ClassVar[str] = "half_to_even"

    def __post_init__(self) -> None:
        if not self.admits(self.scale):
            raise InputError(
                f"an r scale is a whole number from 1 to {self.largest_scale}, not "
                f"{self.scale!r} (ld-store §3)"
            )

    @classmethod
    def admits(cls, scale: object) -> bool:
        """Whether `scale` is one the container holds: a whole number from 1 to 32767."""
        return _whole(scale) and 1 <= scale <= cls.largest_scale

    @classmethod
    def of(cls, record: object) -> REncoding:
        """The encoding a store's `encoding` record names (ld-store §7), or an `InputError` naming
        the record when this package does not implement it (ld-store §13): another dtype, reserved
        code or rounding, a scale outside 1–32767, or a key it does not know."""
        if not cls._implements(record):
            raise InputError(
                f"the encoding {record!r} is not one this ogld implements: exactly the keys "
                f"{sorted(Encoding.__required_keys__)}, an int16 container with "
                f"{cls.reserved_code} reserved, rounded {cls.rounding}, at a whole scale from 1 to "
                f"{cls.largest_scale} (ld-store §3, §13)"
            )
        return cls(record["r_scale"])

    @classmethod
    def _implements(cls, record: object) -> TypeGuard[dict[str, Any]]:
        # Exactly the record's keys, as `Encoding` types them: a key this package does not know
        # could change what a code means, so a record with one is refused, not ignored.
        if not isinstance(record, dict) or set(record) != Encoding.__required_keys__:
            return False
        container = (record["r_dtype"], record["r_reserved_code"], record["rounding"])
        expected = (cls.dtype.name, cls.reserved_code, cls.rounding)
        return container == expected and cls.admits(record["r_scale"])

    def record(self) -> Encoding:
        """The store's `encoding` provenance record of this encoding (ld-store §7)."""
        return {
            "r_dtype": self.dtype.name,
            "r_scale": self.scale,
            "r_reserved_code": self.reserved_code,
            "rounding": self.rounding,
        }

    @property
    def step(self) -> float:
        """One code's width in r: 1 / scale."""
        return 1 / self.scale

    @property
    def max_error(self) -> float:
        """Half a step, 1 / (2 × scale): |r̂ − r| never exceeds it (ld-store §3)."""
        return 1 / (2 * self.scale)

    def quantise(self, r: npt.ArrayLike) -> Codes:
        """The code of each r: clamp(rint(r × scale), −scale, +scale) (ld-store §3).

        `numpy.rint` rounds half to even, as the spec requires. A value that is not finite,
        or whose magnitude exceeds 1 by more than float error, raises: r is undefined only
        for a zero-variance variant, which the builder drops (ld-store §3), so such a value means
        the computation went wrong, and a clamped code would hide it.
        """
        values = np.asarray(r, dtype=np.float64)
        bad = ~np.isfinite(values) | (np.abs(values) > 1 + _R_SLACK)
        if bad.any():
            raise ValueError(
                f"not a correlation, so it has no code: {values[bad].ravel()[:5].tolist()} "
                "(ld-store §3)"
            )
        codes = np.clip(np.rint(values * self.scale), -self.scale, self.scale)
        return codes.astype(self.dtype)

    def holds(self, codes: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Where `codes` are codes of this encoding, |code| ≤ scale: the reserved code never is,
        and neither is a code past the scale, which would decode to |r̂| > 1 (ld-store §12)."""
        # Compared in the codes' own dtype: every scale fits int16, and a validation block of 8 Mi
        # codes would otherwise hold a 64 MiB copy (ld-store §12).
        values = np.asarray(codes)
        return np.asarray((values >= -self.scale) & (values <= self.scale))

    def require_codes(self, codes: npt.ArrayLike) -> None:
        """Raise ValueError naming the values of `codes` this encoding never writes. The
        reserved code −32768 is what an unwritten chunk reads as, so taking it for a number is
        exactly the silent zero it exists to prevent; a code past the scale would be an r beyond
        ±1. The reader refuses both before it decodes, and the writer before it stores."""
        values = np.asarray(codes)
        held = self.holds(values)
        if not held.all():
            found = values[~held].ravel()[:5].tolist()
            raise ValueError(
                f"codes {found} are not codes of r at × {self.scale}: {self.reserved_code} is "
                "reserved for an unwritten chunk, and any other is past the scale (ld-store §3)"
            )

    def dequantise(self, codes: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """The Stored r of each code: code / scale (ld-store §3). A code this encoding never writes
        raises, naming it (`require_codes`)."""
        values = np.asarray(codes)
        self.require_codes(values)
        return values.astype(np.float64) / self.scale

    def far_code_floor(self, t_far: float) -> int:
        """The smallest code c ≥ 0 with (c / scale)² ≥ t_far, in float64 (ld-store §3).

        A pair more than `near_bp` apart is a Far Pair exactly when |code| reaches it, which is
        the same as r̂² ≥ t_far. Computed from an estimate and then stepped to
        the exact boundary, because the float64 comparison, not the real-number one, is the
        rule.
        """
        if not 0.0 <= t_far <= 1.0:
            raise ValueError(f"t_far must be an r² in [0, 1]: {t_far}")
        code = max(0, math.floor(math.sqrt(t_far) * self.scale) - 2)
        while (code / self.scale) ** 2 < t_far:
            code += 1
        return code

    def partner_code_floor(self, t_partner: float, t_far: float) -> int:
        """The Partner Floor: the smallest code c ≥ 0 with (c / scale)² ≥ t_partner, in float64,
        the same rule as the far code floor at the Partner Threshold (ld-store §3; partner-rows §3,
        §5.3).

        `t_partner` is a build input with no default (ADR 0097, D3), and it is never above the
        store's t_far: a partner threshold above it would leave a pair between the two in neither
        component, and a request at r2 ≥ t_far could not be answered (partner-rows §3). So a
        `t_partner` above `t_far` is refused here, naming both, before anything is written.
        """
        if not 0.0 <= t_partner <= t_far:
            raise InputError(
                f"t_partner = {t_partner!r} is above t_far = {t_far!r}: the Partner Threshold is "
                "never above the Far Threshold, or a pair between the two within near_bp would "
                "be in neither component (ld-store §3; partner-rows §3)"
            )
        return self.far_code_floor(t_partner)

    def least_r(self, code: int) -> float:
        """A bound below every |r| whose code reaches |code|: (code − 1) / scale. The exact edge
        is (code − ½) / scale; half a step lower, float error in r × scale cannot lose a pair,
        so a builder may judge only the r above it."""
        return (code - 1) / self.scale

    def stored_mask(
        self,
        distance: npt.ArrayLike,
        codes: npt.ArrayLike,
        *,
        near_bp: int,
        far_bp: int,
        far_code_floor: int,
    ) -> npt.NDArray[np.bool_]:
        """Whether each pair is Stored at its distance, from its code (ld-store §8).

        Every pair within `near_bp` is Stored; beyond it, one is Stored exactly when it is
        within `far_bp` and its |code| reaches `far_code_floor`. The builder, the Partner Rows
        emitter and the Genotype Store reader all judge with this one predicate, so the Bound
        and Not Measured cannot drift between them (owner, 21 November 2026).
        """
        gap = np.abs(np.asarray(distance, dtype=np.int64))
        magnitude = np.abs(np.asarray(codes, dtype=np.int32))
        near = gap <= near_bp
        far = (gap <= far_bp) & (magnitude >= far_code_floor)
        return near | far

    def partner_mask(
        self,
        distance: npt.ArrayLike,
        codes: npt.ArrayLike,
        *,
        near_bp: int,
        far_bp: int,
        partner_code_floor: int,
        far_code_floor: int,
    ) -> npt.NDArray[np.bool_]:
        """Whether each pair is in a Partner Rows row, from its code (partner-rows §3, §7).

        A pair within `near_bp` is kept when its |code| reaches the Partner Floor, and a pair
        beyond it, out to `far_bp`, when its |code| reaches the far code floor. This is the
        floor choice the Partner Rows emitter makes and the Genotype Store reader answers
        `proxies` and `clump` with, held once (owner, 21 November 2026).

        Pass each pair's real `distance`. The Partner Rows emitter instead passes its two
        already-split groups' boundary: `near_bp` for the Near Band's codes, every one of which
        is within `near_bp`, and `far_bp` for the Far Pairs, every one of which is beyond
        `near_bp` and within `far_bp`. The answer is the same for every such pair, and
        `tests/ld/test_pair_masks.py` holds the emitter's bytes to this rule, pair by pair.
        """
        gap = np.abs(np.asarray(distance, dtype=np.int64))
        magnitude = np.abs(np.asarray(codes, dtype=np.int32))
        floor = np.where(gap <= near_bp, partner_code_floor, far_code_floor)
        return (gap <= far_bp) & (magnitude >= floor)


def _whole(value: object) -> TypeGuard[int]:
    """A whole number as JSON holds one: an int, never a bool or a float."""
    return isinstance(value, int) and not isinstance(value, bool)


def t_far(n: int, k: int) -> float:
    """The Far Threshold: max(0.001, q / (n − k − 1)), in float64 (ADR 0087, ld-store §3).

    `n` is the samples the LD was computed from and `k` the covariates adjusted for,
    counted by name in provenance (ld-store §7). A store records this value and every reader
    uses the record; the builder and validation are the only callers. A panel with
    n − k − 1 ≤ q would have t_far ≥ 1, which no pair can reach, so it raises (§12).
    """
    if k < 0:
        raise ValueError(f"k counts covariates, so it cannot be negative: n={n}, k={k}")
    dof = n - k - 1
    if dof <= T_FAR_Q:
        raise ValueError(
            f"n − k − 1 = {dof} must exceed q = {T_FAR_Q} for t_far < 1: n={n}, k={k} (ld-store §12)"
        )
    return max(T_FAR_MIN, T_FAR_Q / dof)
