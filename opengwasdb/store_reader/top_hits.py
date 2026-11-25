"""The Top-Hit Index, as a reader addresses it: its tiers, their keys and their rows.

A Top-Hit Index is a `top_hits/` group of one tier per significance threshold
(`p_5e_08`, ...). A tier holds the associations whose two-sided p clears that
threshold, as parallel arrays ordered by ``(analysis_index, variant_index)`` with
an ``analysis_offsets`` array addressing one Analysis's slice (spec, "Top-Hit
Index"). Writing and checking a tier are `opengwasdb.top_hits`'s; the
|z| a threshold cuts at is `opengwasdb.store_reader.stats.z_critical`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

import numpy as np
import zarr

#: The group a Store's data.zarr holds the Top-Hit Index in.
TOP_HIT_GROUP = "top_hits"

#: The thresholds a Store Release indexes, loosest last. Positionally paired with
#: `TOP_HIT_COUNT_COLUMNS`, the `analyses.tsv` columns the tiers' counts land in.
TOP_HIT_THRESHOLDS = (5e-8, 5e-6, 5e-4)


class UnstoredThresholdError(ValueError):
    """A Dense or Hybrid query named a threshold the release stores no Top-Hit tier for.

    A Dense or Hybrid release answers `top_hits` from its tiers and never scans the grid
    (#483): at OGS-00016's scale a scan is hours. The error names the thresholds
    the release does store, so the caller can ask one of them. `fields` is what a reader
    of the Store Reader contract compares (#474); the message is for people.
    """

    def __init__(self, threshold: float, stored: list[float]) -> None:
        self.threshold = threshold
        self.stored = list(stored)
        named = ", ".join(f"{t:g}" for t in self.stored) or "none"
        super().__init__(
            f"threshold {threshold:g} is not a stored top-hit tier of this release "
            f"(stored: {named}); a Dense or Hybrid release answers top_hits at a "
            "stored tier only"
        )

    @property
    def fields(self) -> dict[str, object]:
        return {"threshold": self.threshold, "stored": self.stored}


def threshold_key(threshold: float) -> str:
    """Stable Zarr group key for a p-value threshold."""

    return f"p_{threshold:.0e}".replace("-", "_").replace("+", "")


def tier_threshold(key: str, group: Any, *, missing: float | None = None) -> float:
    """The p-value threshold a tier was written for.

    Its ``threshold`` attribute; when the tier has none, `missing` if the caller
    names its own fallback, else the threshold its key spells -- the inverse of
    `threshold_key` (``p_5e_08`` -> 5e-08).
    """
    if "threshold" in group.attrs:
        return float(group.attrs["threshold"])
    if missing is not None:
        return missing
    return float(key.replace("p_", "").replace("_", "-"))


def has_index(root: Any) -> bool:
    """Whether a Store's data group holds a Top-Hit Index."""
    return TOP_HIT_GROUP in root


def index_group(root: Any) -> Any:
    """The Top-Hit Index group of a Store's data group, one tier per key."""
    return root[TOP_HIT_GROUP]


def analysis_hit_counts(root: Any, threshold: float, n_analyses: int) -> list[int]:
    """How many associations the `threshold` tier holds for each Analysis, in
    ``analysis_index`` order.

    Read from the tier's ``analysis_offsets``. An index that predates that array
    (ukb-b, and other stores built before issue #22) is counted from its
    ``analysis_index`` array instead, over `n_analyses` Analyses. A store with no
    Top-Hit Index, or no tier for `threshold`, is a KeyError, not a zero count.
    """
    group = index_group(root)[threshold_key(threshold)]
    if "analysis_offsets" in group:
        offsets = np.asarray(group["analysis_offsets"], dtype=np.int64)
        counts: list[int] = (offsets[1:] - offsets[:-1]).tolist()
        return counts
    analysis_index = np.asarray(group["analysis_index"], dtype=np.int64)
    counted: list[int] = np.bincount(analysis_index, minlength=n_analyses)[:n_analyses].tolist()
    return counted


def tier_arrays(group: Any) -> dict[str, Any]:
    """One tier's index arrays, in their compact contract dtypes.

    ``Array[:]`` already returns an ndarray. ``np.asarray`` therefore avoids
    ``astype``'s unconditional full-size copy on a valid tier while retaining
    the coercion that lets the validation logic diagnose malformed contents.
    Keeping the uint32 coordinates compact is material: OGS-00010's nominal
    tier alone has 45.7 million hits (issue #254). The tier must not be missing
    any of the arrays its format carries (`opengwasdb.top_hits.missing_arrays`).
    `abs_z` is None on a tier that does not store it: one of format 0.3.0, which
    drops it (ADR 0102, #471).

    What the validator's Top-Hit rules read: they check the tier's structure and
    cross-check its values against the planes, so they want every array as stored,
    in these dtypes, rather than `TopHitTier.read`'s rows (#473).
    """
    return {
        "rows": np.asarray(group["variant_index"][:], dtype=np.uint32),
        "cols": np.asarray(group["analysis_index"][:], dtype=np.uint32),
        "z_values": np.asarray(group["z"][:], dtype=np.float32),
        "se_values": np.asarray(group["se"][:], dtype=np.float32),
        "abs_z": np.asarray(group["abs_z"][:], dtype=np.float32) if "abs_z" in group else None,
        "offsets": np.asarray(group["analysis_offsets"][:], dtype=np.uint64),
        "imputed_values": (
            np.asarray(group["imputed"][:], dtype=np.uint8) if "imputed" in group else None
        ),
        "eaf_values": (
            np.asarray(group["eaf"][:], dtype=np.float32) if "eaf" in group else None
        ),
    }


class TierRows(NamedTuple):
    """The rows of one tier (or one Analysis's slice of it), in index order.

    `se`, `eaf` and `imputed` are None when the index predates the array; the
    caller decides what an older index derives them from.
    """

    variant_index: np.ndarray
    analysis_index: np.ndarray
    z: np.ndarray
    se: np.ndarray | None
    eaf: np.ndarray | None
    imputed: np.ndarray | None


#: The Clumps of the 5e-8 tier (spec §18.1, ADR 0106): a `clumps/` group inside the tier,
#: holding the one stored Clumping Configuration, `default/`, with `lead` and `r` parallel
#: to the tier and the provenance in its `clumps` attribute.
CLUMPS_GROUP = "clumps"
CLUMPS_CONFIGURATION = "default"
CLUMPS_ATTRIBUTE = "clumps"
#: Each array of the configuration, with the dtype §18.1 gives it.
CLUMPS_ARRAYS: Mapping[str, np.dtype[Any]] = {
    "lead": np.dtype(np.int32), "r": np.dtype(np.int16),
}
#: Every key the `clumps` attribute holds (§18.1), and every key of an `ld_stores` entry.
CLUMPS_ATTRIBUTE_KEYS = frozenset({
    "configuration", "threshold", "tier_rows", "score", "ties", "kb", "join", "markers",
    "ld_stores", "not_clumped", "opengwasdb", "created_at",
})
CLUMPS_ENTRY_KEYS = frozenset({
    "name", "kind", "format_version", "samples_sha256", "provenance_sha256", "location",
    "reference_assembly", "n", "k", "t_far", "t_partner", "near_bp", "far_bp", "r_scale", "r2",
    "analyses", "rows_clumped", "rows_absent", "rows_ambiguous", "indel_rows_by_sorted_spelling",
})


class MalformedClumps(ValueError):
    """A tier's `clumps/` group that is not laid out as spec §18.1 says, named with why."""


class TierClumps(NamedTuple):
    """The Clumps of a tier (or of one Analysis's slice of it), as stored (spec §18.1).

    `lead` is each row's Clump Lead as an offset in its Analysis's clump slice, or a
    negative marker; `r` the LD Store's own code of its signed r with that lead; `attribute`
    the configuration's `clumps` attribute, which names the LD Store each ancestry was
    clumped against and its `r_scale`.
    """

    lead: np.ndarray
    r: np.ndarray
    attribute: dict[str, Any]


class TopHitTier:
    """Address one threshold tier without exposing its physical arrays.

    Each of the tier's arrays is opened at most once per tier object (#244).
    zarr-python 3 reads an array's metadata from the store on every
    ``group[name]`` and every ``name in group`` -- about 1 ms each on a local
    store -- and one top-hit query touches up to seven arrays, so a reader that
    reopened them per field spent more time on metadata than on hits (OGS-00009:
    21 ms as it was, 5.8 ms opened once). A Store Release does not change while
    it is queried, so an opened array stays valid for the tier's lifetime.
    """

    def __init__(self, group: zarr.Group):
        self.group = group
        self._arrays: dict[str, zarr.Array | None] = {}

    def _array(self, name: str) -> zarr.Array | None:
        """The tier's array `name`, or None when this index predates it."""
        if name not in self._arrays:
            try:
                self._arrays[name] = self.group[name]
            except KeyError:
                self._arrays[name] = None
        return self._arrays[name]

    def _required(self, name: str) -> zarr.Array:
        array = self._array(name)
        if array is None:
            raise KeyError(f"top-hit tier {self.group.path!r} has no {name!r} array")
        return array

    def has(self, name: str) -> bool:
        """Whether this tier's index carries the array `name`."""
        return self._array(name) is not None

    @property
    def has_offsets(self) -> bool:
        """Whether an Analysis's slice can be addressed by offset (older indexes cannot)."""
        return self.has("analysis_offsets")

    def _bounds(self, analysis_index: int | None) -> tuple[int, int]:
        if analysis_index is None:
            return 0, int(self._required("z").shape[0])
        offsets = self._required("analysis_offsets")
        if analysis_index < 0 or analysis_index + 1 >= int(offsets.shape[0]):
            return 0, 0
        pair = offsets[analysis_index : analysis_index + 2]
        return int(pair[0]), int(pair[1])

    def _read(self, name: str, bounds: tuple[int, int], dtype: str) -> np.ndarray:
        start, stop = bounds
        return np.asarray(self._required(name)[start:stop], dtype=dtype)

    def _read_if_present(self, name: str, bounds: tuple[int, int], dtype: str) -> np.ndarray | None:
        return self._read(name, bounds, dtype) if self.has(name) else None

    def has_clumps(self) -> bool:
        """Whether the tier holds a `clumps/` group, well formed or not."""
        return CLUMPS_GROUP in self.group

    def clumps(self, analysis_index: int | None = None) -> TierClumps | None:
        """The tier's stored Clumps, or one Analysis's slice of them; None where the tier
        holds no `clumps/` group (spec §18.1).

        A group that is not laid out as §18.1 says is refused, `MalformedClumps`, naming
        what is wrong: a member besides `default/`, a missing array or attribute, an array of
        another dtype or length than the tier's, or an attribute missing a key. Whether the
        release declares the Clumps is its manifest's to say (`clumps`, §2), not the tier's.
        """
        if not self.has_clumps():
            return None
        default = self._clumps_configuration()
        n_rows = int(self._required("z").shape[0])
        arrays = {name: self._clumps_array(default, name, dtype, n_rows)
                  for name, dtype in CLUMPS_ARRAYS.items()}
        attribute = _clumps_attribute(self.group.path, default)
        start, stop = self._bounds(analysis_index)
        return TierClumps(
            lead=np.asarray(arrays["lead"][start:stop], dtype=np.int32),
            r=np.asarray(arrays["r"][start:stop], dtype=np.int16),
            attribute=attribute,
        )

    def _clumps_configuration(self) -> zarr.Group:
        group = self.group[CLUMPS_GROUP]
        names = sorted(name for name, _ in group.members())
        if names != [CLUMPS_CONFIGURATION]:
            raise MalformedClumps(
                f"{self.group.path}/{CLUMPS_GROUP} holds {names}; {CLUMPS_CONFIGURATION!r} is "
                "the one Clumping Configuration a release holds (spec §18.1)"
            )
        configuration: zarr.Group = group[CLUMPS_CONFIGURATION]
        return configuration

    def _clumps_array(
        self, default: zarr.Group, name: str, dtype: np.dtype[Any], n_rows: int
    ) -> zarr.Array:
        where = f"{self.group.path}/{CLUMPS_GROUP}/{CLUMPS_CONFIGURATION}/{name}"
        if name not in default:
            raise MalformedClumps(f"{where} is missing (spec §18.1)")
        array: zarr.Array = default[name]
        if array.dtype != dtype or array.shape != (n_rows,):
            raise MalformedClumps(
                f"{where} is {array.dtype} of shape {array.shape}; the tier's Clumps are "
                f"{dtype}, one per its {n_rows} rows (spec §18.1)"
            )
        return array

    def read(
        self, analysis_index: int | None = None, *, require: Iterable[str] = ()
    ) -> TierRows:
        """The tier's rows, or one Analysis's slice of them.

        An `analysis_index` outside the index addresses an empty slice. Naming an
        optional array in `require` makes its absence a KeyError rather than a
        None, for a caller (a Ragged read) that has no fallback for it. A tier
        with no `analysis_offsets` answers one Analysis by filtering its rows on
        `analysis_index`: still the pre-built tier, and no grid is read (#483).
        """
        for name in require:
            self._required(name)
        if analysis_index is not None and not self.has_offsets:
            whole = self.read(None, require=require)
            keep = whole.analysis_index == analysis_index
            return TierRows(
                variant_index=whole.variant_index[keep],
                analysis_index=whole.analysis_index[keep],
                z=whole.z[keep],
                se=None if whole.se is None else whole.se[keep],
                eaf=None if whole.eaf is None else whole.eaf[keep],
                imputed=None if whole.imputed is None else whole.imputed[keep],
            )
        bounds = self._bounds(analysis_index)
        return TierRows(
            variant_index=self._read("variant_index", bounds, "int32"),
            analysis_index=self._read("analysis_index", bounds, "int32"),
            z=self._read("z", bounds, "float32"),
            se=self._read_if_present("se", bounds, "float32"),
            eaf=self._read_if_present("eaf", bounds, "float32"),
            imputed=self._read_if_present("imputed", bounds, "uint8"),
        )


def _clumps_attribute(where: str, default: zarr.Group) -> dict[str, Any]:
    """The configuration's `clumps` attribute, every key §18.1 requires present."""
    attribute = default.attrs.get(CLUMPS_ATTRIBUTE)
    if not isinstance(attribute, Mapping):
        raise MalformedClumps(
            f"{where}/{CLUMPS_GROUP}/{CLUMPS_CONFIGURATION} has no {CLUMPS_ATTRIBUTE!r} "
            "attribute (spec §18.1)"
        )
    missing = sorted(CLUMPS_ATTRIBUTE_KEYS - set(attribute))
    if missing:
        raise MalformedClumps(
            f"{where}/{CLUMPS_GROUP}/{CLUMPS_CONFIGURATION}'s {CLUMPS_ATTRIBUTE!r} attribute "
            f"lacks {missing} (spec §18.1)"
        )
    return dict(attribute)


class TopHitTiers:
    """A store's top-hit threshold tiers, each opened once and kept (#244).

    A query facade holds one, so a repeated top-hit query reuses the tier group
    and the arrays its `TopHitTier` already opened instead of reading their
    metadata from the store again.
    """

    def __init__(self, root: zarr.Group):
        self._root = root
        self._tiers: dict[str, TopHitTier | None] = {}

    def tier(self, threshold: float) -> TopHitTier | None:
        """The tier for `threshold`, or None when the store has no such tier."""
        key = threshold_key(threshold)
        if key not in self._tiers:
            try:
                self._tiers[key] = TopHitTier(self._root[f"{TOP_HIT_GROUP}/{key}"])
            except KeyError:
                self._tiers[key] = None
        return self._tiers[key]

    def stored_thresholds(self) -> list[float]:
        """The thresholds this store has a tier for, tightest first."""
        try:
            group = self._root[TOP_HIT_GROUP]
        except KeyError:
            return []
        return sorted(tier_threshold(key, group[key]) for key in group)

    def require(self, threshold: float) -> TopHitTier:
        """The tier for `threshold`; `UnstoredThresholdError` when there is none."""
        tier = self.tier(threshold)
        if tier is None:
            raise UnstoredThresholdError(threshold, self.stored_thresholds())
        return tier
