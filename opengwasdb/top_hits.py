"""The Top-Hit Index: its format, and the one place that writes, reads, counts
and checks it, for every layout (issue #53).

A Top-Hit Index is a `top_hits/` group of one tier per significance threshold
(`p_5e_08`, ...). A tier holds the associations whose two-sided p clears that
threshold, as parallel arrays ordered by ``(analysis_index, variant_index)`` with
an ``analysis_offsets`` array addressing one Analysis's slice. A Dense Store, a
Ragged Store and a Hybrid Store's Ragged Overflow Component write the same tier;
what differs between them is how the candidate cells are *harvested*, which stays
in ``layouts/*/top_hits.py``. This module owns everything after the harvest, so
the layouts cannot drift apart on the format; what a reader needs -- the tiers,
their keys and rows, and the |z| a threshold cuts at -- is the Store Reader's
(`opengwasdb.store_reader.top_hits` and `.stats`, #464). Of those, only
`z_critical` is re-exported here.

The on-disk format is specified in ``docs/spec/store-format.md``; nothing here
changes it.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc

from opengwasdb.encoding.codec import require_parallel
from opengwasdb.encoding.timing import log_phase
from opengwasdb.model.analyses import TOP_HIT_COUNT_COLUMNS, Analysis
from opengwasdb.store import arrays as store_arrays

# Re-exported: the Store Reader owns the cut (#464). It is called through the
# module, so a test that moves the cut moves every caller with it.
from opengwasdb.store_reader import layout as store_layout
from opengwasdb.store_reader import stats as reader_stats
from opengwasdb.store_reader.arrays import array_length
from opengwasdb.store_reader.format import carries_stored_p
from opengwasdb.store_reader.layout import ArrayRole
from opengwasdb.store_reader.release import release_arrays
from opengwasdb.store_reader.stats import z_critical as z_critical
from opengwasdb.store_reader.top_hits import (
    TOP_HIT_GROUP,
    TOP_HIT_THRESHOLDS,
    analysis_hit_counts,
    index_group,
    threshold_key,
)
from opengwasdb.store_reader.top_hits import has_index as reader_has_index

log = logging.getLogger(__name__)

#: Re-exported from the seam, which owns the role's default layout, so the
#: writer default and the `TOP_HIT_INDEX` policy cannot disagree.
TOP_HIT_CHUNK_SIZE = store_layout.TOP_HIT_CHUNK_SIZE

# A dict, not a zip over the caller's own `thresholds`, so `read_top_hit_counts`
# accepts any subset of TOP_HIT_THRESHOLDS rather than silently requiring three.
_THRESHOLD_COLUMNS = dict(zip(TOP_HIT_THRESHOLDS, TOP_HIT_COUNT_COLUMNS, strict=True))

#: Every tier carries these, whatever layout wrote it. `imputed` and `eaf` are
#: optional: an index that predates them, or a Store that has no such plane.
TOP_HIT_TIER_ARRAYS = frozenset({"analysis_offsets", "variant_index", "analysis_index", "z", "se"})

#: What a tier before format 0.3.0 also stores: `abs_z`, its z's magnitude, and
#: `p_value`, the two-sided p that z implies. Nothing but the validator read them, and
#: it recomputes both from `z` (`opengwasdb.store_reader.stats`), so 0.3.0 drops them
#: (ADR 0102, #471). This build writes neither.
STORED_P_ARRAYS = frozenset({"abs_z", "p_value"})

# The dtype each tier array is written as. One table, so an optional array cannot
# acquire a different dtype from the required ones by being written elsewhere.
_TIER_DTYPES = {
    "variant_index": "uint32",
    "analysis_index": "uint32",
    "z": "float32",
    "se": "float32",
    "imputed": "uint8",
    "eaf": "float32",
}


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

    The arrays are judged parallel before the index group is created (issue
    #392). One of another length was refused by numpy inside the first tier's
    hit mask (`values[keep]`), after that tier's group had been created: the
    call wrote no tier's arrays, and the tiers a refused call leaves behind are
    the ones an earlier call wrote. No array was named.
    """
    require_parallel(tuple(columns.items()), where="write_index")
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

    # The cut is on |z|; neither it nor the p it implies is stored (ADR 0102, #471).
    keep = reader_stats.clears_cutoff(abs_z, threshold)
    kept = {name: values[keep] for name, values in columns.items()}

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


def write_stored_p(root: zarr.Group, compressor: Blosc | None = None) -> None:
    """Give every tier of `root`'s Top-Hit Index the `abs_z` and `p_value` a release
    before format 0.3.0 stores (`STORED_P_ARRAYS`), recomputed from its `z`.

    `write_index` writes 0.3.0's tiers, which store neither (ADR 0102, #471). A
    rebuild of a 0.2 release's index (`ogdb build-dense-top-hits`,
    `build-ragged-top-hits`) adds them after, so the release keeps its own format
    (#313). The values are `recomputed_abs_z_and_p`'s, the validator's.
    """
    comp = compressor if compressor is not None else store_arrays.compressor()
    index = root[TOP_HIT_GROUP]
    for key in list(index):
        tier = index[key]
        abs_z, p_value = recomputed_abs_z_and_p(tier["z"][:])
        for name, values, dtype in (("abs_z", abs_z, "float32"), ("p_value", p_value, "float64")):
            store_arrays.create_array(
                tier,
                name,
                ArrayRole.TOP_HIT_INDEX,
                data=values,
                dtype=dtype,
                compressor=comp,
                hint=TOP_HIT_CHUNK_SIZE,
                overwrite=True,
            )


# ── read ──────────────────────────────────────────────────────────────────────


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

    ``n_analyses`` is needed for the fallback, not just for sizing a
    zero-filled result: real pre-issue-#22-era stores (e.g. ukb-b) have a
    top-hit index that predates the ``analysis_offsets`` array the count is
    otherwise read from, so the Store Reader's `analysis_hit_counts` counts the
    flat ``analysis_index`` array by hand for those.
    """
    root = release_arrays(store_path)
    return {
        _THRESHOLD_COLUMNS[threshold]: analysis_hit_counts(root, threshold, n_analyses)
        for threshold in thresholds
    }


def summed_hit_counts(
    store_paths: list[Path], n_analyses: int
) -> dict[str, list[int]]:
    """Per-Analysis Top-Hit Counts over several components' indexes, summed: what the
    `analyses.tsv` beside them records (ADR 0032, #509). A Hybrid root's counts both its
    Dense Component's tiers and its own (the Ragged Overflow's); every other table, its
    own release's. A component whose tier counts other than `n_analyses` Analyses is a
    `ValueError`, and one with no Top-Hit Index or no tier a `KeyError`, never a zero."""
    summed = {column: [0] * n_analyses for column in TOP_HIT_COUNT_COLUMNS}
    for path in store_paths:
        for column, counts in read_top_hit_counts(path, n_analyses).items():
            if len(counts) != n_analyses:
                raise ValueError(
                    f"{path}: the {column} tier counts {len(counts)} Analyses, the "
                    f"analyses.tsv beside it holds {n_analyses}"
                )
            summed[column] = [a + b for a, b in zip(summed[column], counts, strict=True)]
    return summed


def misshapen_tier(store_path: str | Path, n_analyses: int) -> str | None:
    """The first tier of the release at `store_path` whose `analysis_offsets` do not
    partition its rows over `n_analyses` Analyses, or None. A count read through such
    offsets is not a count. This is `structural_problem`'s offsets test without loading
    the tier, and `top_hits.structure` reports the tier itself. A tier that predates
    the offsets is counted from `analysis_index`, and is not judged here."""
    tiers = index_group(release_arrays(store_path))
    for threshold in TOP_HIT_THRESHOLDS:
        key = threshold_key(threshold)
        if key not in tiers or "analysis_offsets" not in tiers[key]:
            continue
        offsets = np.asarray(tiers[key]["analysis_offsets"][:], dtype=np.int64)
        if not _offsets_ok(offsets, array_length(tiers[key]["variant_index"]), n_analyses):
            return f"{store_path}: tier {key} has invalid analysis offsets"
    return None


def has_top_hit_index(store_path: str | Path) -> bool:
    """Whether the release at `store_path` carries a Top-Hit Index."""
    return bool(reader_has_index(release_arrays(store_path)))


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


def tier_array_names(format_version: str) -> frozenset[str]:
    """The arrays a tier of a release of `format_version` must carry: `STORED_P_ARRAYS`
    as well before 0.3.0 (`carries_stored_p`)."""
    if carries_stored_p(format_version):
        return TOP_HIT_TIER_ARRAYS | STORED_P_ARRAYS
    return TOP_HIT_TIER_ARRAYS


def missing_arrays(group: Any, format_version: str) -> list[str]:
    """The required arrays a tier of a release of `format_version` lacks, sorted."""
    return sorted(tier_array_names(format_version).difference(group.keys()))


def dropped_arrays(group: Any, format_version: str) -> list[str]:
    """The arrays a tier carries that its release's `format_version` dropped, sorted:
    `STORED_P_ARRAYS` from 0.3.0, which no writer of that format stores."""
    if carries_stored_p(format_version):
        return []
    return sorted(STORED_P_ARRAYS.intersection(group.keys()))


def _lengths_ok(a: dict[str, Any], group: Any) -> bool:
    """Whether every parallel array in the tier is the same length."""
    n = len(a["rows"])
    optional = [x for x in (a["imputed_values"], a["eaf_values"], a["abs_z"]) if x is not None]
    # `a`'s entries are numpy arrays, `group["se"]`/`group["p_value"]` are Zarr
    # arrays; zarr 3 removed ``len(Array)``, so every length is the leading
    # axis (which is what ``len`` meant for the numpy ones too).
    stored_p = [group["p_value"]] if "p_value" in group else []
    return all(
        int(x.shape[0]) == n for x in [a["cols"], a["z_values"], group["se"], *stored_p, *optional]
    )


def recomputed_abs_z_and_p(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """What a 0.2 tier stored beside `z`: its magnitude, as `float32`, and the two-sided
    p it implies, through `opengwasdb.store_reader.stats` (the one derivation)."""
    abs_z = np.abs(np.asarray(z, dtype=np.float32))
    return abs_z, reader_stats.p_values_from_z(abs_z)


def stored_p_agrees(abs_z: np.ndarray, p_value: np.ndarray, z: np.ndarray) -> bool:
    """Whether a 0.2 tier's stored `abs_z` and `p_value` are what its `z` implies.

    `abs_z` within the tolerance its validator rule has always used (1e-3, relative and
    absolute); `p_value` within 1e-3 relative, and absolutely only within float64's
    smallest normal number. A floor any wider would wave through a wrong p of 5e-8; one
    narrower would fail every p below it, where the build's scipy `erfc` flushed to 0.0
    what `store_reader.stats` keeps as a subnormal: both say p underflows, |z| from
    37.5 (226 rows a tier on OGS-00006). The validator's `top_hits.abs_z` and the
    rewrite to 0.3.0, which drops both arrays, ask this one question (ADR 0102, #471).
    """
    abs_z_now, p_now = recomputed_abs_z_and_p(z)
    stored_p = np.asarray(p_value, dtype=np.float64)
    return bool(
        np.allclose(np.asarray(abs_z), abs_z_now, rtol=1e-3, atol=1e-3)
        and np.allclose(stored_p, p_now, rtol=1e-3, atol=_SMALLEST_NORMAL_P)
    )


#: Below this a float64 p is subnormal, or 0.0 where an `erfc` flushes it (`stored_p_agrees`).
_SMALLEST_NORMAL_P = float(np.finfo(np.float64).tiny)


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
