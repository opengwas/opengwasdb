"""The Top-Hit Index: its format, and the one place that writes, reads, counts
and checks it, for every layout (issue #53).

A Top-Hit Index is a `top_hits/` group of one tier per significance threshold
(`p_5e_08`, ...). A tier holds the associations whose two-sided p clears that
threshold, as parallel arrays ordered by ``(analysis_index, variant_index)`` with
an ``analysis_offsets`` array addressing one Analysis's slice. A Dense Store, a
Ragged Store and a Hybrid Store's Ragged Overflow Component write the same tier;
what differs between them is how the candidate cells are *harvested*, which stays
in ``layouts/*/top_hits.py``. This module owns everything after the harvest and
everything a reader needs, so the layouts cannot drift apart on the format.

The on-disk format is specified in ``docs/spec/store-format.md``; nothing here
changes it.
"""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import zarr
from numcodecs import Blosc

from opengwasdb.encoding.timing import log_phase
from opengwasdb.model.analyses import TOP_HIT_COUNT_COLUMNS, Analysis
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store.arrays import ArrayRole

log = logging.getLogger(__name__)

# scipy ships no type stubs; importing by name keeps the build's one scipy use
# from needing a per-site `type: ignore`.
_special = importlib.import_module("scipy.special")

#: The group a Store's data.zarr holds the Top-Hit Index in.
TOP_HIT_GROUP = "top_hits"

#: The thresholds a Store Release indexes, loosest last. Positionally paired with
#: `TOP_HIT_COUNT_COLUMNS`, the `analyses.tsv` columns the tiers' counts land in.
TOP_HIT_THRESHOLDS = (5e-8, 5e-6, 5e-4)

#: Re-exported from the seam, which owns the role's default layout, so the
#: writer default and the `TOP_HIT_INDEX` policy cannot disagree.
TOP_HIT_CHUNK_SIZE = store_arrays.TOP_HIT_CHUNK_SIZE

# A dict, not a zip over the caller's own `thresholds`, so `read_top_hit_counts`
# accepts any subset of TOP_HIT_THRESHOLDS rather than silently requiring three.
_THRESHOLD_COLUMNS = dict(zip(TOP_HIT_THRESHOLDS, TOP_HIT_COUNT_COLUMNS, strict=True))

#: Every tier carries these, whatever layout wrote it. `imputed` and `eaf` are
#: optional: an index that predates them, or a Store that has no such plane.
TOP_HIT_TIER_ARRAYS = frozenset(
    {"analysis_offsets", "variant_index", "analysis_index", "abs_z", "z", "se", "p_value"}
)

# The dtype each tier array is written as. One table, so an optional array cannot
# acquire a different dtype from the required ones by being written elsewhere.
_TIER_DTYPES = {
    "variant_index": "uint32",
    "analysis_index": "uint32",
    "abs_z": "float32",
    "z": "float32",
    "se": "float32",
    "p_value": "float64",
    "imputed": "uint8",
    "eaf": "float32",
}


def threshold_key(threshold: float) -> str:
    """Stable Zarr group key for a p-value threshold."""

    return f"p_{threshold:.0e}".replace("-", "_").replace("+", "")


def z_critical(threshold: float) -> float:
    """|z| cutoff equivalent to two-sided p <= threshold.

    The stored p-value is ``p = erfc(|z| / sqrt(2))`` (see the query facade and
    the old scalar ``p_value_from_z``), so ``p <= threshold`` is exactly
    ``|z| >= sqrt(2) * erfcinv(threshold)``. Thresholding on ``|z|`` lets the
    hot path avoid computing a p-value for every cell. The index build, the
    Ragged scan fallback and the validator all cut here and nowhere else.
    """

    return math.sqrt(2.0) * float(_special.erfcinv(threshold))


# ── write ─────────────────────────────────────────────────────────────────────


def write_index(
    root: zarr.Group,
    columns: dict[str, np.ndarray],
    thresholds: tuple[float, ...],
    n_analyses: int,
    chunk_size: int = TOP_HIT_CHUNK_SIZE,
    compressor: Blosc | None = None,
) -> dict[str, int]:
    """Write one tier per threshold into `root`'s Top-Hit Index group (created if
    absent -- callers pass the Store's data group, never the index's own); return
    each tier's hit count.

    `columns` holds every parallel candidate array the caller has -- the required
    ``variant_index``, ``analysis_index``, ``z`` and ``se`` and the optional
    ``imputed`` and ``eaf`` -- for every cell clearing at least the loosest
    threshold (extra cells are harmless: each tier re-filters by its own
    `z_critical`). Each array is filtered, ordered and written by the same three
    lines, so an array cannot be dropped from one of those steps and not the
    others.
    """
    top = store_arrays.require_group(root, TOP_HIT_GROUP)
    comp = compressor if compressor is not None else store_arrays.compressor()
    abs_z = np.abs(columns["z"]).astype("float32")
    written: dict[str, int] = {}
    for threshold in thresholds:
        key = threshold_key(threshold)
        with log_phase(log, f"Top-hit tier {key}"):
            written[key] = _write_tier(top, threshold, columns, abs_z, n_analyses, chunk_size, comp)
        log.info("%s: %d hits", key, written[key])
    top.attrs["thresholds"] = list(thresholds)
    return written


def _write_tier(
    top: zarr.Group,
    threshold: float,
    columns: dict[str, np.ndarray],
    abs_z: np.ndarray,
    n_analyses: int,
    chunk_size: int,
    compressor: Blosc,
) -> int:
    group = store_arrays.create_group(top, threshold_key(threshold))

    keep = abs_z >= z_critical(threshold)
    kept = {name: values[keep] for name, values in columns.items()}
    kept["abs_z"] = abs_z[keep]
    kept["p_value"] = _special.erfc(kept["abs_z"].astype("float64") / math.sqrt(2.0))

    order = np.lexsort((kept["variant_index"], kept["analysis_index"]))
    kept = {name: values[order] for name, values in kept.items()}

    offsets = np.empty(n_analyses + 1, dtype="uint64")
    offsets[0] = 0
    np.cumsum(
        np.bincount(kept["analysis_index"], minlength=n_analyses),
        dtype=np.uint64,
        out=offsets[1:],
    )
    store_arrays.create_array(
        group,
        "analysis_offsets",
        ArrayRole.TOP_HIT_ANALYSIS_OFFSETS,
        data=offsets,
        dtype="uint64",
        compressor=compressor,
    )
    for name, values in kept.items():
        store_arrays.create_array(
            group,
            name,
            ArrayRole.TOP_HIT_INDEX,
            data=values,
            dtype=_TIER_DTYPES[name],
            compressor=compressor,
            hint=chunk_size,
        )
    group.attrs["threshold"] = threshold
    group.attrs["order"] = "analysis_index,variant_index"
    return int(len(kept["variant_index"]))


# ── read ──────────────────────────────────────────────────────────────────────


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
        """Whether an Analysis's slice can be addressed (older indexes cannot)."""
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

    def read(
        self, analysis_index: int | None = None, *, require: Iterable[str] = ()
    ) -> TierRows:
        """The tier's rows, or one Analysis's slice of them.

        An `analysis_index` outside the index addresses an empty slice. Naming an
        optional array in `require` makes its absence a KeyError rather than a
        None, for a caller (a Ragged read) that has no fallback for it.
        """
        for name in require:
            self._required(name)
        bounds = self._bounds(analysis_index)
        return TierRows(
            variant_index=self._read("variant_index", bounds, "int32"),
            analysis_index=self._read("analysis_index", bounds, "int32"),
            z=self._read("z", bounds, "float32"),
            se=self._read_if_present("se", bounds, "float32"),
            eaf=self._read_if_present("eaf", bounds, "float32"),
            imputed=self._read_if_present("imputed", bounds, "uint8"),
        )


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


# ── counts ────────────────────────────────────────────────────────────────────


def read_top_hit_counts(
    store_path: str | Path,
    n_analyses: int,
    thresholds: tuple[float, ...] = TOP_HIT_THRESHOLDS,
) -> dict[str, list[int]]:
    """Per-Analysis hit counts for each threshold tier, from an already-built
    top-hit index (every layout's builder writes the same schema). Keyed by the
    ``analyses.tsv`` column each tier persists to (ADR 0032), in
    ``analysis_index`` order.

    ``n_analyses`` is needed for the fallback path below, not just for
    sizing a zero-filled result: real pre-issue-#22-era stores (e.g. ukb-b)
    have a top-hit index that predates the ``analysis_offsets`` array this
    function otherwise reads directly, so it falls back to counting the
    flat ``analysis_index`` array by hand for those.
    """
    root = store_arrays.open_group(Path(store_path) / "data.zarr")
    top = root[TOP_HIT_GROUP]
    counts: dict[str, list[int]] = {}
    for threshold in thresholds:
        column = _THRESHOLD_COLUMNS[threshold]
        group = top[threshold_key(threshold)]
        if "analysis_offsets" in group:
            offsets = np.asarray(group["analysis_offsets"], dtype=np.int64)
            counts[column] = (offsets[1:] - offsets[:-1]).tolist()
        else:
            analysis_index = np.asarray(group["analysis_index"], dtype=np.int64)
            counts[column] = np.bincount(analysis_index, minlength=n_analyses)[:n_analyses].tolist()
    return counts


def add_hit_counts(store_path: str | Path, analyses: list[Analysis]) -> list[Analysis]:
    """Add per-Analysis Top-Hit Counts from ``store_path``'s already-built
    top-hit index onto whatever each ``Analysis`` already carries (ADR 0032).

    Blank ``n_hits_*`` fields act as zero, so this both sets counts from
    scratch (a fresh build's analyses start blank) and accumulates a Hybrid
    store's Dense Component and Ragged Overflow Component contributions onto
    the same Analysis when called once per component -- the two partition an
    Analysis's associations disjointly (CONTEXT.md, Query Component).
    """
    counts = read_top_hit_counts(store_path, len(analyses))
    return [
        replace(
            a,
            n_hits_5e8=str(int(a.n_hits_5e8 or 0) + counts["n_hits_5e8"][i]),
            n_hits_5e6=str(int(a.n_hits_5e6 or 0) + counts["n_hits_5e6"][i]),
            n_hits_5e4=str(int(a.n_hits_5e4 or 0) + counts["n_hits_5e4"][i]),
        )
        for i, a in enumerate(analyses)
    ]


# ── structural rules ──────────────────────────────────────────────────────────
#
# What a tier must satisfy before a layout's validator compares it with the
# matrix or the CSR it was harvested from. The cross-checks themselves differ per
# layout and stay with the validator; these do not, and used to be written twice.


def missing_arrays(group: Any) -> list[str]:
    """The required arrays a tier lacks, sorted."""
    return sorted(TOP_HIT_TIER_ARRAYS.difference(group.keys()))


def load_tier_arrays(group: Any) -> dict[str, Any]:
    """One tier's index arrays, in their compact contract dtypes.

    ``Array[:]`` already returns an ndarray. ``np.asarray`` therefore avoids
    ``astype``'s unconditional full-size copy on a valid tier while retaining
    the coercion that lets the validation logic diagnose malformed contents.
    Keeping the uint32 coordinates compact is material: OGS-00010's nominal
    tier alone has 45.7 million hits (issue #254). The tier must not be missing
    any of `TOP_HIT_TIER_ARRAYS` (see `missing_arrays`).
    """
    return {
        "rows": np.asarray(group["variant_index"][:], dtype=np.uint32),
        "cols": np.asarray(group["analysis_index"][:], dtype=np.uint32),
        "z_values": np.asarray(group["z"][:], dtype=np.float32),
        "se_values": np.asarray(group["se"][:], dtype=np.float32),
        "abs_z": np.asarray(group["abs_z"][:], dtype=np.float32),
        "offsets": np.asarray(group["analysis_offsets"][:], dtype=np.uint64),
        "imputed_values": (
            np.asarray(group["imputed"][:], dtype=np.uint8) if "imputed" in group else None
        ),
        "eaf_values": (
            np.asarray(group["eaf"][:], dtype=np.float32) if "eaf" in group else None
        ),
    }


def _lengths_ok(a: dict[str, Any], group: Any) -> bool:
    """Whether every parallel array in the tier is the same length."""
    n = len(a["rows"])
    optional = [x for x in (a["imputed_values"], a["eaf_values"]) if x is not None]
    # `a`'s entries are numpy arrays, `group["se"]`/`group["p_value"]` are Zarr
    # arrays; zarr 3 removed ``len(Array)``, so every length is the leading
    # axis (which is what ``len`` meant for the numpy ones too).
    return all(
        int(x.shape[0]) == n
        for x in [a["cols"], a["z_values"], a["abs_z"], group["se"], group["p_value"], *optional]
    )


def _offsets_ok(offsets: np.ndarray, n_rows: int, n_analyses: int) -> bool:
    """Whether the per-analysis offsets partition the tier's rows."""
    return bool(
        len(offsets) == n_analyses + 1
        and offsets[0] == 0
        and not np.any(offsets[:-1] > offsets[1:])
        and offsets[-1] == n_rows
    )


def slices_are_ordered(
    ais: np.ndarray, vis: np.ndarray, analysis_offsets: np.ndarray, n_analyses: int
) -> bool:
    """Whether each analysis's slice holds only its own hits, in genomic order."""
    for analysis_index in range(n_analyses):
        start = int(analysis_offsets[analysis_index])
        end = int(analysis_offsets[analysis_index + 1])
        if np.any(ais[start:end] != analysis_index):
            return False
        if end - start > 1 and np.any(vis[start : end - 1] >= vis[start + 1 : end]):
            return False
    return True


def structural_problem(
    a: dict[str, Any], group: Any, n_analyses: int, expects_imputed: bool
) -> str | None:
    """The first thing wrong with a tier before its cells are cross-checked.

    `expects_imputed` is whether the Store the tier indexes records completion
    status, in which case the tier must too. The order is the order the checks
    were written in: a structural fault is reported ahead of a value fault,
    because the second is not worth reading while the first makes the arrays
    untrustworthy.
    """
    if expects_imputed and a["imputed_values"] is None:
        return "is missing imputed completion status"
    if not _lengths_ok(a, group):
        return "has inconsistent array lengths"
    if not _offsets_ok(a["offsets"], len(a["rows"]), n_analyses):
        return "has invalid analysis offsets"
    if not slices_are_ordered(a["cols"], a["rows"], a["offsets"], n_analyses):
        return "has incorrect or non-genomic analysis slices"
    imputed = a["imputed_values"]
    if imputed is not None and not np.all((imputed == 0) | (imputed == 1)):
        return "has invalid imputed completion status"
    # Strictly increasing rows inside slices that each hold only their named
    # Analysis already prove every (row, column) pair unique. Rebuilding those
    # pairs as int64 and sorting them with np.unique duplicated hundreds of MB
    # on production Top-Hit Indexes without checking another invariant (#254).
    return None
