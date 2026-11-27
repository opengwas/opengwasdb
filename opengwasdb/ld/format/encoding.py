"""How a build quantises r, and which pairs beyond the Near Band an LD Store keeps (ld-store §3).

An LD Store records its encoding (ld-store §7), and `REncoding` is that record as a value. Its read
side -- the container, which encodings a reader implements (`of`), `holds`, `require_codes`,
dequantising, and the quantising and the two masks a Genotype Store's reader judges pairs by
(`quantise`, `stored_mask`, `partner_mask`) -- is `ogquery.ld.format.encoding.REncoding`, the one
owner of the dtype, the scale, the reserved code and the rounding (ADR 0090; opengwas/ogquery#2
and #11, ogquery ADR 0002). This module extends it with the build side, which ogquery leaves out:
the record a build writes, the step and its error, the far code floor, the Partner Floor and
`least_r`. The builder quantises and masks with the encoding it was given, validation recomputes
the floors with the one the store records, and the reader (ogquery's) decodes, quantises and
masks with that one too, so no component can round, decode or threshold at another store's scale.

opengwasdb's `REncoding` is a subclass, so it is an `ogquery.ld.format.REncoding` wherever a reader
takes one, and `REncoding.of` and its constructor give this class. ogquery's readers construct
their own, which has the read members but none of the build ones above: a build or validation step
handed one raises `AttributeError` at its first build member (`far_code_floor`, `record`, `step`)
rather than compute a floor at a guessed scale. Construct `REncoding(encoding.scale)` to cross
over.

t_far and the distances are not the encoding's: they depend on the panel and on ADR 0087, and a
store records them beside it. A reader uses the t_far, floors and distances a store records.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from ogquery.ld.format.encoding import Codes as Codes
from ogquery.ld.format.encoding import REncoding as ReadEncoding
from ogquery.ld.format.provenance import Encoding
from ogquery.ld.format.semantics import InputError

#: The χ²₁ upper quantile at p = 5×10⁻⁶ in t_far = max(0.001, q / (n − k − 1)) (ADR 0087).
T_FAR_Q: Final = 20.84

#: t_far's floor: MR's default r² threshold (ADR 0087).
T_FAR_MIN: Final = 0.001

#: The builder's distances (ADR 0087): the Near Band is complete up to NEAR_BP, and the
#: Far Pairs and Partner Rows reach FAR_BP. A store records the distances it measured (ld-store §7)
#: and readers use those, never these.
NEAR_BP: Final = 3_000_000
FAR_BP: Final = 10_000_000


@dataclass(frozen=True)
class REncoding(ReadEncoding):
    """One LD Store's encoding of r, as a build quantises with it: ogquery's `REncoding` (the
    container, `of`, `holds`, `require_codes`, `dequantise`) and the build side ogquery leaves out
    (ld-store §3; ADR 0090). It adds no field, so it equals another of its class at the same scale.
    """

    @classmethod
    def of(cls, record: object) -> REncoding:
        """The encoding a store's `encoding` record names, as this class, so a build or
        validation step quantises and computes floors with it; refused as ogquery's `of` refuses
        an encoding it does not implement (ld-store §3, §13)."""
        encoding = super().of(record)
        # ogquery's `of` constructs `cls`; were it to construct its own class, this class's
        # members would be missing, and a build would fail later and further from the cause.
        assert isinstance(encoding, REncoding), type(encoding)
        return encoding

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
            f"n − k − 1 = {dof} must exceed q = {T_FAR_Q} for t_far < 1: n={n}, k={k} "
            "(ld-store §12)"
        )
    return max(T_FAR_MIN, T_FAR_Q / dof)
