"""`StoreEncoding` -- the persisted plan that says how a store's statistic
planes are encoded (ADR 0037, issue #119).

Four properties, each load-bearing:

1. **Decided in exactly one function.** `decide()` is the tree. Nothing else
   branches on measured data properties to pick an encoding, so a rule change
   is a one-line diff with one test file rather than an audit of every writer.
2. **Persisted in `manifest.json`, never re-derived on read.** If a reader
   re-ran the tree, changing a threshold would silently change how existing
   stores decode. `from_manifest` is the only way a reader gets a plan.
3. **Authoritative over array presence.** Asking `if "z" in root` and
   inferring from dtype is how a store comes to disagree with itself; the plan
   says what should be there and validation checks that it is.
4. **Unknown kinds fail loudly** (`UnsupportedEncoding`, spec §21). A reader
   meeting an encoding it does not implement rejects the release rather than
   guessing.

Scope note: `z`, `se` and `eaf`. `se` residual coding is still #118's, and
until it lands `SeEncoding` has one kind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

#: `z` is missing. Decodes to NaN, and the paired `se` must be NaN too
#: (spec §15). An integer plane cannot hold NaN, which is why the contract is
#: stated per codec rather than per dtype (ADR 0038 §6).
Z_MISSING = -32768

#: `z` is outside the representable range. The exact `float32` lives in the
#: plane's sparse overflow table. Not a clip and not a build failure: the
#: `ukb-b` survey found 568 genuine associations above |z| = 64 (ADR 0037).
Z_OVERFLOW = -32767

#: The largest code that carries a value; the two below it are reserved.
Z_CODE_MAX = 32767

#: Steps per unit for the `int16` fixed-point `z` plane. 1/1024 gives a
#: representable range of -31.998 .. +31.999 with a worst-case p error of 1.6%
#: at the edge and 0.24% at z = 5, against `float16`'s 26.4% at z = 30. The
#: 6,346 `ukb-b` cells outside the range cost a 74 KB side table.
DEFAULT_Z_SCALE = 1024

#: Version of the plan's own schema, distinct from `format_version`: it
#: identifies the shape of the `encoding` block, not the store format. Version
#: 1 declared `z` and `se`; version 2 adds `eaf` (issue #116).
ENCODING_VERSION = 2


class UnsupportedEncoding(Exception):
    """A store declares a statistic encoding this build does not implement."""


@dataclass(frozen=True)
class ZEncoding:
    """How the `z` plane's stored bytes relate to z-scores."""

    kind: str
    scale: int | None = None

    @property
    def is_fixed_point(self) -> bool:
        """Whether this plane stores scaled integers, with the reserved codes
        and the overflow table that go with them."""
        return self.kind == "int16_fixed"

    @property
    def dtype(self) -> str:
        return "int16" if self.is_fixed_point else "float16"

    @property
    def max_representable(self) -> float:
        """Largest z the plane itself holds; anything above goes to the table.

        Stated as an exact endpoint rather than "±32": signed fixed point is
        asymmetric, and reserving `Z_MISSING`/`Z_OVERFLOW` makes it more so.
        """
        if not self.is_fixed_point:
            return float("inf")
        assert self.scale is not None
        return Z_CODE_MAX / self.scale

    @property
    def min_representable(self) -> float:
        if not self.is_fixed_point:
            return float("-inf")
        assert self.scale is not None
        return (Z_OVERFLOW + 1) / self.scale

    def to_manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.scale is not None:
            payload["scale"] = self.scale
        return payload

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> ZEncoding:
        kind = str(data["kind"])
        if kind == "float16":
            return cls(kind="float16")
        if kind == "int16_fixed":
            scale = int(data["scale"])
            if scale <= 0:
                raise UnsupportedEncoding(f"z encoding int16_fixed has invalid scale {scale}")
            return cls(kind="int16_fixed", scale=scale)
        raise UnsupportedEncoding(
            f"z encoding kind {kind!r} is not implemented by this build; "
            "this release cannot be read (spec §21)"
        )


@dataclass(frozen=True)
class SeEncoding:
    """How the `se` plane's stored bytes relate to standard errors.

    `float16` is the *right* encoding here, for the opposite reason it is the
    wrong one for `z`: `se` spans 3.2 decades and needs relative precision,
    which a float exponent already provides (ADR 0037). Recorded so it is not
    later "fixed" by analogy with `z`.
    """

    kind: str = "float16"

    @property
    def dtype(self) -> str:
        return "float16"

    def to_manifest(self) -> dict[str, Any]:
        return {"kind": self.kind}

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> SeEncoding:
        kind = str(data["kind"])
        if kind != "float16":
            raise UnsupportedEncoding(
                f"se encoding kind {kind!r} is not implemented by this build; "
                "this release cannot be read (spec §21)"
            )
        return cls(kind=kind)


#: `eaf` is absent for this cell -- the Analysis reported no frequency at this
#: variant. Decodes to NaN, and is *not* a residual of zero: a store that
#: cannot tell "no frequency" from "the same frequency as its neighbours"
#: reports a made-up number for every EAF-less cohort (ADR 0037 §2).
EAF_ABSENT = -128

#: `eaf` is not representable as a residual -- the exact `float32` lives in the
#: plane's exception table. Covers a residual outside the range, a frequency of
#: exactly 0 or 1 (no logit), and a variant with no usable baseline.
EAF_EXCEPTION = -127

#: The residual codes, inclusive. 254 levels, not 253: an `int8` has 256 codes
#: and two are reserved. The accuracy table in ADR 0037 §2 was computed for 254.
EAF_CODE_MIN = -126
EAF_CODE_MAX = 127

#: Half of the code span, and the divisor that turns a range into a step. The
#: span is deliberately asymmetric -- `[-126 * step, +127 * step]` -- for the
#: same reason `z`'s is: reserving codes off one end of a signed integer makes
#: it so, and stating exact endpoints is honest where "±range" is not.
EAF_CODE_HALF = 127

#: The residual ranges `decide()` may choose between, in logit units.
EAF_RANGE_CANDIDATES = (0.5, 1.0, 2.0)

#: The largest share of EAF-bearing cells `decide()` will send to the exception
#: table in exchange for a smaller (more accurate) range. An exception costs 12
#: raw bytes -- an `int64` position and a `float32` value -- against the
#: plane's 1 byte per cell, so 2% caps the side table at 24% of the plane it
#: sits beside. Within that budget the smaller range is always preferred: it is
#: 2-4x more accurate on every cell that is *not* an exception, and an
#: exception is stored exactly either way.
EAF_EXCEPTION_BUDGET = 0.02

#: Raw bytes an exception-table entry costs (`int64` index + `float32` value).
EAF_EXCEPTION_BYTES = 12


class EafBaselineError(Exception):
    """A baseline array is missing, or does not describe the cells given."""


@dataclass(frozen=True)
class EafEncoding:
    """How the `eaf` plane's stored bytes relate to effect allele frequencies.

    Three kinds, and the plan says which:

    - `absent` -- no Analysis in this release reports a frequency, so there is
      no plane. Distinct from a plane of all-absent cells, and distinct from
      "the array happens not to be on disk", which is what the code used to
      ask.
    - `float32` -- ADR 0036's plane, kept as the escape valve. Exact, and the
      right answer whenever the residual coding would not actually be smaller
      (a store with roughly one EAF-bearing cell per variant pays more for the
      per-variant baseline than the `int8` cell saves).
    - `int8_residual` -- ADR 0037 §2. A per-variant `float32` baseline plus a
      per-cell quantised **logit** residual.
    - `float32_optional` -- what a `format_version` 1.0 release is in, and the
      only kind this build reads but never writes. ADR 0036 made the plane's
      *presence* the statement that a release had frequencies, so a reader of
      such a release still has to look. Naming that weaker contract is what
      lets every other kind mean exactly what it says, and lets validation
      hold the newer ones to plan-vs-arrays agreement without exempting the
      older ones by accident.

    The transform is the logit, `log(f / (1 - f))`, and not the log of the
    frequency. Both leave residuals of the same width on real data (measured:
    FinnGen sd 0.0170 against 0.0169), but only the logit's error is bounded
    on *both* sides of the frequency: a residual error of `d` moves `f` by a
    relative `(1 - f) * d` and `1 - f` by a relative `f * d`, so neither the
    effect allele frequency nor the minor allele frequency users filter on can
    be wrong by more than `d`. `log(f)` is blind to the minor side, which for
    a frequency near 1 is the only side anyone reads.
    """

    kind: str
    residual_range: float | None = None

    #: Whether this component carries a per-variant `eaf_reference` array for
    #: its imputed cells (ADR 0037 §4). Per component, not per store: a Hybrid
    #: release's Dense Component has imputed cells where its Ragged Overflow
    #: does not, and each declares its own plan in its own manifest.
    reference: bool = False

    @property
    def is_absent(self) -> bool:
        return self.kind == "absent"

    @property
    def is_residual(self) -> bool:
        return self.kind == "int8_residual"

    @property
    def is_optional_plane(self) -> bool:
        """Whether the plane's presence, rather than the plan, says if this
        release has frequencies. True only for `format_version` 1.0 releases."""
        return self.kind == "float32_optional"

    @property
    def dtype(self) -> str:
        return "int8" if self.is_residual else "float32"

    @property
    def step(self) -> float:
        """Logit units per code."""
        if not self.is_residual:
            raise EafBaselineError(f"eaf encoding {self.kind!r} has no residual step")
        assert self.residual_range is not None
        return self.residual_range / EAF_CODE_HALF

    @property
    def max_residual(self) -> float:
        """Largest residual the plane holds; above it a cell is an exception."""
        return EAF_CODE_MAX * self.step

    @property
    def min_residual(self) -> float:
        return EAF_CODE_MIN * self.step

    @property
    def worst_case_relative_error(self) -> float:
        """Half a step: the most a decoded EAF -- or its complement -- can be
        wrong by, relative, on a cell that is not an exception."""
        return self.step / 2.0

    def to_manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.residual_range is not None:
            payload["residual_range"] = self.residual_range
        if self.reference:
            payload["reference"] = True
        return payload

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> EafEncoding:
        kind = str(data["kind"])
        reference = bool(data.get("reference", False))
        if kind in ("absent", "float32", "float32_optional"):
            return cls(kind=kind, reference=reference)
        if kind == "int8_residual":
            residual_range = float(data["residual_range"])
            if not math.isfinite(residual_range) or residual_range <= 0:
                raise UnsupportedEncoding(
                    f"eaf encoding int8_residual has invalid residual_range {residual_range}"
                )
            return cls(kind=kind, residual_range=residual_range, reference=reference)
        raise UnsupportedEncoding(
            f"eaf encoding kind {kind!r} is not implemented by this build; "
            "this release cannot be read (spec §21)"
        )


@dataclass(frozen=True)
class EafMeasurements:
    """What the tree is allowed to know about a build's frequencies.

    Counted over every component the plan will cover -- a Hybrid release's
    Dense Component and its Ragged Overflow are summed, because one `decide()`
    serves both and each writes its own baseline array.
    """

    #: Cells the plane spans: the Dense grid's area, or the CSR's length.
    n_cells: int = 0
    #: Of those, cells carrying a finite frequency.
    n_eaf_cells: int = 0
    #: Length of the variant axis (or axes) a baseline array would cover. This,
    #: not `n_analyses`, is what decides whether the baseline amortises: a
    #: Ragged store whose variants appear in one Analysis each pays 4 bytes of
    #: baseline per 1-byte cell, and is better off in `float32`.
    n_variants: int = 0
    #: Share of the EAF-bearing cells that would fall outside each candidate
    #: range, measured on the build's own data. Keys are the candidates.
    exception_fraction: dict[float, float] = field(default_factory=dict)

    @property
    def has_eaf(self) -> bool:
        return self.n_eaf_cells > 0


@dataclass(frozen=True)
class EncodingMeasurements:
    """The one summary the decision tree is allowed to look at.

    Computed once per build, from the data rather than from the layout.
    Only the fields the current tree consults are here; #118 adds the
    per-Analysis `se`~MAF fit when there is a rule that reads it.
    """

    n_analyses: int
    eaf: EafMeasurements | None = None


@dataclass(frozen=True)
class StoreEncoding:
    """The plan: one decision, recorded in `manifest.json`, read everywhere."""

    z: ZEncoding
    se: SeEncoding
    eaf: EafEncoding = EafEncoding(kind="absent")
    version: int = ENCODING_VERSION

    @classmethod
    def decide(cls, measurements: EncodingMeasurements) -> StoreEncoding:
        """The decision tree, and the only site that runs it (ADR 0037).

        R1 -- `z` is `int16` fixed point, unconditionally: it is bounded and
        needs uniform precision, and the sparse overflow table means no data
        property can make that choice wrong.
        R2 -- `se` is `float16`, unconditionally until #118 measures the
        `se`~MAF fit and earns the residual coding.
        R3..R5 -- `eaf` is decided by `_decide_eaf` below, from measured
        residual spread and measured bytes.
        """
        return cls(
            z=ZEncoding(kind="int16_fixed", scale=DEFAULT_Z_SCALE),
            se=SeEncoding(kind="float16"),
            eaf=_decide_eaf(measurements.eaf),
        )

    @classmethod
    def legacy(cls) -> StoreEncoding:
        """The plan a release that declares none is in: `float16` throughout.

        Every store up to `format_version` 0.1. Derived here rather than
        guessed at each read site, so "no declaration" is one plan rather than
        an absence every caller handles differently.
        """
        return cls(
            z=ZEncoding(kind="float16"),
            se=SeEncoding(kind="float16"),
            eaf=EafEncoding(kind="float32_optional"),
            version=0,
        )

    @property
    def is_legacy(self) -> bool:
        return self.version == 0

    def with_eaf_reference(self, present: bool) -> StoreEncoding:
        """This plan, with the component's `eaf_reference` presence set.

        Reference EAF is the one part of the plan that legitimately differs
        between a Hybrid release's two components, so it is set per component
        rather than decided by the tree (ADR 0037 §4).
        """
        if present and self.eaf.is_absent:
            raise EafBaselineError(
                "a component cannot carry reference EAF for its imputed cells while "
                "declaring no eaf plane at all: the two say different things about "
                "whether this release has frequencies"
            )
        return StoreEncoding(
            z=self.z,
            se=self.se,
            eaf=EafEncoding(
                kind=self.eaf.kind,
                residual_range=self.eaf.residual_range,
                reference=present,
            ),
            version=self.version,
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "z": self.z.to_manifest(),
            "se": self.se.to_manifest(),
            "eaf": self.eaf.to_manifest(),
        }

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> StoreEncoding:
        """Read a declared plan. Never re-derives it, and never falls back."""
        version = int(data.get("version", ENCODING_VERSION))
        if version > ENCODING_VERSION:
            raise UnsupportedEncoding(
                f"encoding block version {version} is newer than this build "
                f"implements ({ENCODING_VERSION}); this release cannot be read"
            )
        # A `format_version` 1.0 release declares `z` and `se` but not `eaf`:
        # its frequencies are ADR 0036's `float32` plane, present or absent on
        # disk. That is a real encoding, and naming it here is what keeps the
        # "never infer from array presence" rule true for those releases too.
        eaf = (
            EafEncoding.from_manifest(data["eaf"])
            if "eaf" in data
            else EafEncoding(kind="float32_optional")
        )
        return cls(
            z=ZEncoding.from_manifest(data["z"]),
            se=SeEncoding.from_manifest(data["se"]),
            eaf=eaf,
            version=version,
        )


def eaf_residual_bytes(measurements: EafMeasurements, residual_range: float) -> int:
    """Raw bytes the residual coding would occupy at `residual_range`."""
    fraction = measurements.exception_fraction.get(residual_range, 0.0)
    exceptions = math.ceil(fraction * measurements.n_eaf_cells)
    return (
        measurements.n_cells
        + 4 * measurements.n_variants
        + EAF_EXCEPTION_BYTES * exceptions
    )


def eaf_float32_bytes(measurements: EafMeasurements) -> int:
    """Raw bytes ADR 0036's `float32` plane would occupy."""
    return 4 * measurements.n_cells


def _decide_eaf(measurements: EafMeasurements | None) -> EafEncoding:
    """Pick the `eaf` encoding from measured spread and measured bytes.

    Two questions, in order, and neither is answered from the layout:

    1. **Which range?** The smallest candidate whose measured exception
       fraction fits `EAF_EXCEPTION_BUDGET`, because a smaller range is more
       accurate on every cell it holds and an exception is exact either way.
       If none fits, the widest candidate -- a store whose frequencies really
       do disagree by more than a factor of `e^2` between Analyses is not made
       better by a narrower range, only more expensive.
    2. **Is it worth it?** Raw bytes against ADR 0036's `float32` plane. This
       is where the baseline's economics are decided rather than assumed: the
       saving is per *cell* and the cost is per *variant*, so a store with few
       EAF-bearing cells per variant is left in `float32`. Compressed bytes
       would be the truer comparison but cannot be measured before the data is
       written; the raw comparison is conservative in the safe direction,
       since the `int8` plane compresses substantially better than the
       `float32` one it would replace (measured: 0.26 against 1.52 B/cell on
       FinnGen chr1).
    """
    if measurements is None or not measurements.has_eaf:
        return EafEncoding(kind="absent")
    chosen = next(
        (
            candidate
            for candidate in EAF_RANGE_CANDIDATES
            if measurements.exception_fraction.get(candidate, 1.0) <= EAF_EXCEPTION_BUDGET
        ),
        EAF_RANGE_CANDIDATES[-1],
    )
    if eaf_residual_bytes(measurements, chosen) >= eaf_float32_bytes(measurements):
        return EafEncoding(kind="float32")
    return EafEncoding(kind="int8_residual", residual_range=chosen)
