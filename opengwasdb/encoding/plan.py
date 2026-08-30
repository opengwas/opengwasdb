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

Scope note: `z` and `se` only. EAF's encoding is unchanged by #114 -- it stays
the `float32` plane ADR 0036 shipped, present or absent per source -- and joins
the plan with #116, which is the change that gives it something to declare.
"""

from __future__ import annotations

from dataclasses import dataclass
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
#: identifies the shape of the `encoding` block, not the store format.
ENCODING_VERSION = 1


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


@dataclass(frozen=True)
class EncodingMeasurements:
    """The one summary the decision tree is allowed to look at.

    Computed once per build, from the data rather than from the layout.
    Only the fields the current tree consults are here; #116/#118 add the EAF
    residual spread and the per-Analysis `se`~MAF fit when there are rules
    that read them.
    """

    n_analyses: int


@dataclass(frozen=True)
class StoreEncoding:
    """The plan: one decision, recorded in `manifest.json`, read everywhere."""

    z: ZEncoding
    se: SeEncoding
    version: int = ENCODING_VERSION

    @classmethod
    def decide(cls, measurements: EncodingMeasurements) -> StoreEncoding:
        """The decision tree, and the only site that runs it (ADR 0037).

        R1 -- `z` is `int16` fixed point, unconditionally: it is bounded and
        needs uniform precision, and the sparse overflow table means no data
        property can make that choice wrong.
        R2 -- `se` is `float16`, unconditionally until #118 measures the
        `se`~MAF fit and earns the residual coding.

        Both rules being unconditional is why `measurements` is not consulted
        here yet. It is in the signature because the *point* of this function
        is that encoding decisions are made from measured data in one place:
        the rules that read the summary arrive with #116 and #118, and they
        arrive by editing this function rather than by giving it a new shape.
        """
        _ = measurements
        return cls(
            z=ZEncoding(kind="int16_fixed", scale=DEFAULT_Z_SCALE),
            se=SeEncoding(kind="float16"),
        )

    @classmethod
    def legacy(cls) -> StoreEncoding:
        """The plan a release that declares none is in: `float16` throughout.

        Every store up to `format_version` 0.1. Derived here rather than
        guessed at each read site, so "no declaration" is one plan rather than
        an absence every caller handles differently.
        """
        return cls(z=ZEncoding(kind="float16"), se=SeEncoding(kind="float16"), version=0)

    @property
    def is_legacy(self) -> bool:
        return self.version == 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "z": self.z.to_manifest(),
            "se": self.se.to_manifest(),
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
        return cls(
            z=ZEncoding.from_manifest(data["z"]),
            se=SeEncoding.from_manifest(data["se"]),
            version=version,
        )
