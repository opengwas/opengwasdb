"""`overview.html` — a generated, human-browsable rendering of a Store
Release's Analytical Metadata (store-format spec §1/§7a, ADR 0030/0032).

Build-time-derived, like `analyses.tsv` itself: written once by the builder
(or the migration script, issue #24) and never independently authored or
edited. Self-contained (no external assets, no network requests) so it opens
correctly from a downloaded store directory with no network access. Visual
identity (palette, typography) matches `docs/opengwasdb-storage-format.html`,
so OpenGWASDB's generated HTML artifacts read as one family (ADR 0032).

A single file, tab-switched client-side rather than split across sibling
files (ADR 0032): AC1/AC2 (store-format spec §20, ADR 0030) both name
`overview.html` specifically as the file a user browses Analyses in, and
tabs keep the "no external assets, no network" self-containment property
trivially true for the whole page at once (issue #37).

Two steps rather than one (issue #57). `gather_store_overview()` reads
everything the page shows out of an opened Store Release — release identity,
the `analyses.tsv` table, the Guide's directory scan, the Rho Matrix's summary
and its histogram bins and hierarchical clustering, the Indexed Variant
Subsets — into a `StoreOverview` value, and `render_overview_html()` turns that
value into HTML touching nothing at all. The split is what lets this page
survive a partial release: `overview.html` is regenerable presentation
(issue #23 AC3), so a missing manifest degrades to blank header fields, an
absent Rho group to no Rho tab, and a damaged Indexed Variant Subset to a
flagged row, rather than any of them aborting the page. That deliberate
degradation now lives in one gather step's failure handling instead of being
re-decided by each accessor, and a group that exists but cannot be read still
raises — a silently shorter Rho Matrix is a wrong answer that looks like a
right one (ADR 0030, CONTRIBUTING). The renderer is pure, so the markup can be
tested against a hand-built value instead of a built-and-rho-estimated store.
"""

from __future__ import annotations

import html
import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from opengwasdb.layouts.dense.indexed_subsets import (
    IndexedSubsetError,
    indexed_subset_namespace_path,
    open_indexed_subset,
)
from opengwasdb.layouts.dense.rho import DenseRhoReader
from opengwasdb.model.analyses import (
    ANCESTRY_PROP_PREFIX,
    AnalysesTable,
    analysis_label,
)
from opengwasdb.store.arrays import open_group
from opengwasdb.store.open import UnsupportedFormatVersion

# scipy publishes no type stubs, so mypy sees nothing of `scipy.cluster.
# hierarchy` or `scipy.spatial.distance` -- the same reason `layouts/dense/
# rho.py` and `stats.py` import scipy behind a bare suppression. Left at that,
# the heatmap's clustering would be an unchecked boundary: a misspelt keyword
# or a wrongly shaped argument would surface as a `TypeError` while a Store
# Release's page was being written, not as a finding here. These Protocols
# declare exactly what this module calls, and the modules are bound to them
# with `cast`, so mypy checks every scipy call below. Widen a Protocol when a
# call needs more of scipy; anything outside them is deliberately unavailable.


class _HierarchyModule(Protocol):
    """`scipy.cluster.hierarchy`, as far as the Rho heatmap uses it."""

    def linkage(self, y: np.ndarray, *, method: str) -> np.ndarray:
        """Cluster a condensed float distance vector, hierarchically."""

    def leaves_list(self, z: np.ndarray) -> np.ndarray:
        """The leaf order of the tree `linkage` returned."""


class _DistanceModule(Protocol):
    """`scipy.spatial.distance`, as far as the Rho heatmap uses it."""

    def squareform(self, x: np.ndarray, *, checks: bool = ...) -> np.ndarray:
        """A symmetric square float matrix as its condensed vector."""


_hierarchy = cast(_HierarchyModule, importlib.import_module("scipy.cluster.hierarchy"))
_spatial_distance = cast(
    _DistanceModule, importlib.import_module("scipy.spatial.distance")
)

# ── The value the renderer takes (issue #57) ────────────────────────────────


@dataclass(frozen=True)
class FileGuideEntry:
    """One Guide-tab row: a top-level store entry, and what it is."""

    name: str
    kind: str  # "human" or "internal"
    description: str


@dataclass(frozen=True)
class RhoHistogram:
    """The Rho distribution, already counted into bins by the gather step.

    `counts` and `edges` are `np.histogram`'s own outputs (one more edge than
    there are counts), not a rendering of them: the bin count is a decision
    about how much of the distribution the page shows, so it belongs with the
    data, not with the markup.
    """

    counts: tuple[int, ...]
    edges: tuple[float, ...]
    n_finite_pairs: int


@dataclass(frozen=True)
class RhoPairRow:
    """One Analysis pair, already ranked and labelled for display."""

    analysis_a: str
    label_a: str
    analysis_b: str
    label_b: str
    rho: float
    n_null: int


@dataclass(frozen=True)
class RhoHeatmap:
    """The hierarchically clustered submatrix, in display order.

    `rho` and `n_null` are row-major `k x k` sequences, `rho` carrying None for
    a NaN pair — exactly the payload the page embeds as JSON. The ordering is
    `scipy.cluster.hierarchy`'s answer, computed once in the gather step.
    """

    k: int
    labels: tuple[str, ...]
    rho: tuple[float | None, ...]
    n_null: tuple[int, ...]


@dataclass(frozen=True)
class RhoTraitRow:
    """One Analysis's own rho relationships — blank fields when it has none."""

    analysis: str
    label: str
    mean_abs_rho: float | None
    max_abs_rho: float | None
    partner_id: str | None
    partner_label: str
    n_null: int | None


@dataclass(frozen=True)
class RhoOverview:
    """The Rho Matrix tab, or None on a Store Release carrying no Rho group."""

    summary: tuple[tuple[str, str], ...]
    n_analyses: int
    histogram: RhoHistogram
    top_positive: tuple[RhoPairRow, ...]
    top_negative: tuple[RhoPairRow, ...]
    heatmap: RhoHeatmap | None  # None when there are too few Analyses to cluster
    traits: tuple[RhoTraitRow, ...]


@dataclass(frozen=True)
class IndexedSubsetsOverview:
    """The Indexed Variant Subsets tab (ADR 0061), or None when there is none.

    `problems` holds every named directory that is not trusted as a published
    subset; those are surfaced rather than rendered as valid rows, and are
    excluded from `rows` and `total_bytes` so a partial group cannot inflate
    the totals.
    """

    rows: tuple[dict[str, str], ...]
    total_bytes: int
    problems: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class StoreOverview:
    """Everything `overview.html` shows, read out of one Store Release.

    Carries no store path: what the page needs from the filesystem — the header
    summary, the Guide's inventory, the Rho Matrix, the Indexed Variant
    Subsets — is resolved here, so `render_overview_html()` cannot reach back
    into the store it came from (issue #57).
    """

    store_id: str
    release_id: str
    completion_state: str
    table: AnalysesTable
    files: tuple[FileGuideEntry, ...]
    rho: RhoOverview | None = None
    indexed_subsets: IndexedSubsetsOverview | None = None


# ── Gather: an opened Store Release -> a StoreOverview value ────────────────

# (kind, description) for each known top-level store entry -- "human" files
# are safe to open directly, "internal" ones are read by the query API and
# should not be hand-edited. Matched by filename against a directory scan
# (issue #38), not hand-maintained per layout, so it can never silently omit
# a file: an unrecognised entry still appears, via the fallback below.
_FILE_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "manifest.json": ("human", "Store identity, format version, and provenance. Plain JSON."),
    "analyses.tsv": (
        "human",
        "Analytical Metadata -- the source of truth for the Analyses tab. "
        "Plain tab-delimited text.",
    ),
    "overview.html": ("human", "This page."),
    "index.sqlite": ("internal", "Variant alias lookup index. Read via the query API."),
    "variants.tsv.gz": ("internal", "Tabix-indexed variant table. Read via the query API."),
    "variants.tsv.gz.tbi": ("internal", "Tabix index for variants.tsv.gz."),
    "variant_offsets.npy": ("internal", "Variant axis row-offset index."),
    "variant_alid_bytes.npy": ("internal", "Variant axis identifier index."),
    "variant_alid_rows.npy": ("internal", "Variant axis identifier index."),
    "data.zarr": (
        "internal",
        "Association statistics (Z/SE) and top-hit indexes, chunked and compressed. "
        "Read via the query API.",
    ),
    "dense": (
        "internal",
        "Dense Component -- a nested Dense store holding the Hybrid layout's shared "
        "reference-panel variants (has its own manifest.json/analyses.tsv/overview.html).",
    ),
    "dense_to_shared.npy": ("internal", "Dense Component row -> shared variant table index map."),
}
_FALLBACK_DESCRIPTION = ("internal", "Internal store file.")


def _scan_store_files(store_path: Path) -> tuple[FileGuideEntry, ...]:
    """`(name, kind, description)` for every top-level entry actually present
    in the store, sorted by name. A live directory scan rather than a
    hand-maintained per-layout list, so it can't drift as Dense/Hybrid or
    Observed-Only/Reference-Completed stores add or omit files (issue #38).
    """
    entries_by_name: dict[str, tuple[str, str]] = (
        {
            path.name: _FILE_DESCRIPTIONS.get(path.name, _FALLBACK_DESCRIPTION)
            for path in store_path.iterdir()
        }
        if store_path.is_dir()
        else {}
    )
    # overview.html is being written by this very call on a fresh build (it
    # doesn't exist on disk yet at scan time), or already exists on a
    # regeneration (issue #39) -- list it either way so the Guide never omits
    # itself.
    entries_by_name.setdefault("overview.html", _FILE_DESCRIPTIONS["overview.html"])
    return tuple(
        FileGuideEntry(name, kind, description)
        for name, (kind, description) in sorted(entries_by_name.items())
    )


# ── Gather failure policy: the one place an unusable artifact is tolerated ──
#
# `overview.html` is regenerable presentation (issue #23 AC3), so a release
# whose manifest, Rho group or Indexed Variant Subset cannot be read must still
# render -- with blank header fields, no Rho tab, or a flagged subset row. That
# tolerance is decided here once, for every accessor below, rather than
# re-implemented per accessor: a gather step names the failures that mean "this
# artifact is absent or damaged", and anything else propagates, because treating
# an unexpected failure as an absent artifact is how a page that looks complete
# gets shipped over a store that could not be read (issue #57).

#: Failures that mean an optional artifact -- the Rho group, an Indexed Variant
#: Subset -- is absent, damaged or in a format this build no longer reads: a
#: missing path or group (`OSError`, which zarr's `GroupNotFoundError` and
#: `PathNotFoundError` subclass), corrupt JSON metadata (`ValueError`), a node
#: of the wrong type (`TypeError`), a missing array (`KeyError`), a subset that
#: fails its structural read seam, and a pre-release format (ADR 0041).
_PROBE_ERRORS: tuple[type[Exception], ...] = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    IndexedSubsetError,
    UnsupportedFormatVersion,
)

#: Failures that mean a release has no usable `manifest.json`: the file is
#: missing or unreadable, or it is not valid JSON. Deliberately narrower than
#: `_PROBE_ERRORS` -- the same policy with this artifact's own failures, because
#: a manifest that parses to something that is not an object is a store worth
#: looking at, not a header worth inventing (`AttributeError` surfaces).
_MANIFEST_ERRORS: tuple[type[Exception], ...] = (OSError, json.JSONDecodeError)


def _probe(
    action: Callable[[], Any], *, errors: tuple[type[Exception], ...] = _PROBE_ERRORS
) -> tuple[Any, Exception | None]:
    """Run a probe, returning its result or the failure it is allowed to hide."""
    try:
        return action(), None
    except errors as error:
        return None, error


def _degrade(
    action: Callable[[], Any],
    default: Any,
    *,
    errors: tuple[type[Exception], ...] = _PROBE_ERRORS,
) -> Any:
    """`action()`'s result, or `default` when an optional artifact is unusable.

    The gather step's one degrade-gracefully seam. `errors` is the artifact's
    own list of failures-that-mean-absent (`_PROBE_ERRORS` for the Rho group and
    the Indexed Variant Subsets, `_MANIFEST_ERRORS` for the header); anything
    else is a bug and propagates.
    """
    value, error = _probe(action, errors=errors)
    return default if error is not None else value


def _manifest_summary(store_path: Path) -> tuple[str, str, str]:
    """`(store_id, release_id, completion_state)`, best-effort from
    `manifest.json`.

    `overview.html` is regenerable from a store's already-persisted data
    (issue #23 AC3), so a missing or unreadable manifest degrades to blank
    header fields rather than failing the whole render -- through the same
    policy as every other optional artifact, with `_MANIFEST_ERRORS` as its own
    list of tolerable failures.
    """
    manifest = _degrade(
        lambda: json.loads((store_path / "manifest.json").read_text(encoding="utf-8")),
        {},
        errors=_MANIFEST_ERRORS,
    )
    return (
        str(manifest.get("store_id", "")),
        str(manifest.get("release_id", "")),
        str(manifest.get("completion_state", "")),
    )


def _open_optional_group(path: Path) -> Any | None:
    """A Zarr group opened read-only, or None when it is absent/unreadable.

    Rho and Indexed Variant Subsets are opt-in derived artifacts; overview.html
    is regenerable presentation and must degrade gracefully rather than fail to
    render when one is missing or damaged (issues #23, #267).
    """
    return _degrade(lambda: open_group(path), None)


def _load_rho_group(store_path: Path) -> Any | None:
    """The optional `data.zarr/rho` group (ADR 0025), or None if absent/unreadable.

    Rho is an opt-in, add-in-place artifact (`build-dense-rho`) -- most stores
    won't have one, and `overview.html` must degrade gracefully rather than
    fail to render (issue #23 AC3 applies here too).
    """
    root = _open_optional_group(store_path / "data.zarr")
    return root["rho"] if root is not None and "rho" in root else None


def _finite_pair_values(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(row_index, col_index, value)` for every finite strict-lower-triangle
    cell -- the one unique pair per Analysis pair, matching `rho.py`'s own
    packed-triangle convention (`np.tril_indices(n, k=-1)`)."""
    rows, cols = np.tril_indices(mat.shape[0], k=-1)
    vals = mat[rows, cols]
    finite = np.isfinite(vals)
    return rows[finite], cols[finite], vals[finite]


# A k x k submatrix embeds as k**2 JSON numbers plus a k*cell_px canvas.
# 300 (matching/exceeding the reference `pleiodb` report's 200) costs ~1MB of
# embedded JSON at ukb-b scale (2,514 Analyses) -- real page weight, but this
# is a store artifact opened deliberately, not a page loaded incidentally, so
# that trade favours coverage. Still fully self-contained/offline (ADR 0025
# "no external assets, no network requests" -- overview.py module docstring):
# the cost is bytes, not a new dependency or a network round-trip.
_RHO_HEATMAP_TRAITS = 300
_RHO_TOP_PAIRS = 30
_RHO_HISTOGRAM_BINS = 201


def _rho_histogram(rho_mat: np.ndarray) -> RhoHistogram:
    """Bin the finite pair values over the whole [-1, 1] domain.

    The full domain rather than the observed range, so two releases' pages can
    be compared: a distribution squeezed into [0.02, 0.08] and one spanning
    [-0.6, 0.9] must not both fill their plot area.
    """
    _, _, finite = _finite_pair_values(rho_mat)
    counts, edges = np.histogram(finite, bins=_RHO_HISTOGRAM_BINS, range=(-1.0, 1.0))
    return RhoHistogram(
        counts=tuple(int(count) for count in counts),
        edges=tuple(float(edge) for edge in edges),
        n_finite_pairs=int(len(finite)),
    )


def _rank_rho_pairs(
    rho_mat: np.ndarray,
    n_null_mat: np.ndarray,
    ids: list[str],
    labels: list[str],
    *,
    ascending: bool,
) -> tuple[RhoPairRow, ...]:
    """The `_RHO_TOP_PAIRS` strongest (or weakest) finite pairs, ranked here so
    the renderer only formats them."""
    rows_idx, cols_idx, vals = _finite_pair_values(rho_mat)
    order = np.argsort(vals if ascending else -vals)[:_RHO_TOP_PAIRS]
    return tuple(
        RhoPairRow(
            analysis_a=ids[i],
            label_a=labels[i],
            analysis_b=ids[j],
            label_b=labels[j],
            rho=float(vals[k]),
            n_null=int(n_null_mat[i, j]),
        )
        for k in order
        for i, j in ((int(rows_idx[k]), int(cols_idx[k])),)
    )


def _rho_heatmap(
    rho_mat: np.ndarray,
    n_null_mat: np.ndarray,
    labels: list[str],
    *,
    k: int = _RHO_HEATMAP_TRAITS,
) -> RhoHeatmap | None:
    """Cluster the `k` most entangled Analyses and return the display payload.

    None when there are fewer than two Analyses to cluster at all -- the page
    says so rather than rendering a degenerate 1x1 matrix.
    """
    n = rho_mat.shape[0]
    k = min(k, n)
    if k < 2:
        return None

    abs_mat = np.abs(np.nan_to_num(rho_mat, nan=0.0))
    np.fill_diagonal(abs_mat, 0.0)
    top = np.argsort(-abs_mat.max(axis=1))[:k]

    sub_rho = rho_mat[np.ix_(top, top)]
    dist = 1.0 - np.nan_to_num(sub_rho, nan=0.0)
    dist = (dist + dist.T) / 2.0  # exact symmetry against float round-off
    np.fill_diagonal(dist, 0.0)
    linkage_matrix = _hierarchy.linkage(
        _spatial_distance.squareform(dist, checks=False), method="average"
    )
    order = top[_hierarchy.leaves_list(linkage_matrix)]

    ordered_rho = rho_mat[np.ix_(order, order)]
    ordered_n = n_null_mat[np.ix_(order, order)]
    return RhoHeatmap(
        k=k,
        labels=tuple(labels[i] for i in order),
        rho=tuple(None if not np.isfinite(v) else round(float(v), 4) for v in ordered_rho.flat),
        n_null=tuple(int(v) for v in ordered_n.flat),
    )


def _rho_trait_rows(
    rho_mat: np.ndarray, n_null_mat: np.ndarray, ids: list[str], labels: list[str]
) -> tuple[RhoTraitRow, ...]:
    """One row per Analysis: its mean |rho| and its single strongest
    relationship (partner + rho + support) -- "which traits are most
    entangled", complementing the pair-level tables."""
    n = rho_mat.shape[0]
    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(rho_mat) & off_diagonal
    abs_rho = np.abs(rho_mat)

    rows = []
    for i in range(n):
        row_mask = finite[i]
        if not row_mask.any():
            rows.append(
                RhoTraitRow(
                    analysis=ids[i],
                    label=labels[i],
                    mean_abs_rho=None,
                    max_abs_rho=None,
                    partner_id=None,
                    partner_label="",
                    n_null=None,
                )
            )
            continue
        ranked = np.where(row_mask, abs_rho[i], -1.0)
        j = int(np.argmax(ranked))
        rows.append(
            RhoTraitRow(
                analysis=ids[i],
                label=labels[i],
                mean_abs_rho=float(abs_rho[i][row_mask].mean()),
                max_abs_rho=float(abs_rho[i][j]),
                partner_id=ids[j],
                partner_label=labels[j],
                n_null=int(n_null_mat[i, j]),
            )
        )
    return tuple(rows)


def _rho_summary(
    group: Any, n_analyses: int, rho_mat: np.ndarray
) -> tuple[tuple[str, str], ...]:
    """The run's own recorded provenance, plus the NaN-pair count the page
    reports -- the only number here derived from the stored matrix rather than
    read from the group's attributes."""
    n_pairs = n_analyses * (n_analyses - 1) // 2
    rows_idx, cols_idx = np.tril_indices(n_analyses, k=-1)
    n_nan = int(np.isnan(rho_mat[rows_idx, cols_idx]).sum())
    return (
        ("Method", str(group.attrs.get("method", ""))),
        ("Analyses", f"{n_analyses:,}"),
        ("Analysis pairs", f"{n_pairs:,}"),
        ("Pairs below min_nulls (NaN)", f"{n_nan:,}"),
        ("z_thresh", str(group.attrs.get("z_thresh", ""))),
        ("min_nulls", str(group.attrs.get("min_nulls", ""))),
        ("window_bp", str(group.attrs.get("window_bp", ""))),
        ("Variants used (thinned)", f"{int(group.attrs.get('n_variants_used', 0)):,}"),
        ("Observed only", str(group.attrs.get("observed_only", ""))),
    )


def _gather_rho(store_path: Path, table: AnalysesTable) -> RhoOverview | None:
    """The Rho Matrix tab's data (ADR 0025), or None when the release has no
    Rho group -- or an empty one.

    Only the group's *presence* degrades (`_load_rho_group`): a group that is
    there but whose arrays cannot be read raises, because a half-read Rho
    Matrix rendered as a shorter, plausible one is exactly the wrong answer
    this project treats as the worst outcome.
    """
    group = _load_rho_group(store_path)
    if group is None:
        return None
    n_analyses = int(group.attrs.get("n_analyses", 0))
    if n_analyses <= 0:
        return None

    rho_mat, n_null_mat = DenseRhoReader(group, n_analyses).matrix(None)
    by_index = {int(row["analysis_index"]): row for row in table.rows if row.get("analysis_index")}
    ids = [by_index.get(i, {}).get("analysis_id", str(i)) for i in range(n_analyses)]
    labels = [analysis_label(by_index.get(i, {})) for i in range(n_analyses)]

    return RhoOverview(
        summary=_rho_summary(group, n_analyses, rho_mat),
        n_analyses=n_analyses,
        histogram=_rho_histogram(rho_mat),
        top_positive=_rank_rho_pairs(rho_mat, n_null_mat, ids, labels, ascending=False),
        top_negative=_rank_rho_pairs(rho_mat, n_null_mat, ids, labels, ascending=True),
        heatmap=_rho_heatmap(rho_mat, n_null_mat, labels),
        traits=_rho_trait_rows(rho_mat, n_null_mat, ids, labels),
    )


# ── Indexed Variant Subsets (issue #267, ADR 0061) ──────────────────────────

#: The group's own recorded facts, rendered in the order the store-format spec
#: §10b names them. Overview presents them; it never recomputes a value.
_INDEXED_SUBSET_FIELDS: tuple[str, ...] = (
    "Subset",
    "Profile",
    "Requested",
    "Resolved",
    "Absent",
    "Overflow-only",
    "Analyses",
    "Physical bytes",
    "Input SHA-256",
    "Reference assembly",
    "Source store",
    "Source release",
    "Source format",
    "Builder",
    "Created (UTC)",
)

#: What an absent attribute reads as, where a decision defines its absence. A
#: group written before `overflow_only_count` existed records neither it nor its
#: sample, and its count is zero by construction rather than a count never taken
#: (ADR 0065). An attribute no decision defines is absent from here and reads
#: blank.
_ABSENT_ATTRIBUTE_VALUES: dict[str, str] = {"overflow_only_count": "0"}

#: Attribute names whose values are counts, and so are digit-grouped when shown.
_COUNT_SUFFIXES: tuple[str, ...] = ("_count", "_analyses", "_variants")


def _human_bytes(count: int) -> str:
    """A binary-unit rendering of a byte count for the summary line."""
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{int(value):,} B" if unit == "B" else f"{value:,.1f} {unit}"
        value /= 1024.0
    return f"{count:,} B"  # unreachable; keeps mypy's return check happy


def _physical_bytes(path: Path) -> int:
    """Allocated bytes for `path` and everything under it (matches `du -s -B1`).

    Allocated rather than apparent size, and directories included: the index is
    bricked into tens of thousands of small chunk files inside deep directories,
    and it is the bytes on disk that the storage-cost decision rests on.
    """
    total = path.stat().st_blocks * 512
    for entry in path.rglob("*"):
        total += entry.stat().st_blocks * 512
    return total


def _read_subset_attrs(path: Path) -> dict[str, Any] | None:
    """A published subset's recorded attributes, or None when unreadable.

    Overview presents the group's own recorded facts and never recomputes a
    value; a damaged group is omitted rather than shown with invented values.
    """
    group = _open_optional_group(path)
    return dict(group.attrs) if group is not None else None


def _indexed_subset_value(attrs: dict[str, Any], key: str) -> str:
    """One recorded attribute as a table cell, with its absence decided.

    Where a decision gives an absent attribute a meaning, that is what is shown;
    `overflow_only_count` is zero by construction for a group written before it
    existed, because that builder could only refuse an Overflow-only ALID
    (ADR 0065). An absence no decision defines reads blank, how the page shows
    every other unrecorded value, never as an invented number. A recorded count
    is digit-grouped only when it is one; a group's own recorded text is shown as
    recorded rather than forced through a number format.
    """
    if key not in attrs:
        return _ABSENT_ATTRIBUTE_VALUES.get(key, "")
    value = attrs[key]
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return str(value)
    if isinstance(value, int) and key.endswith(_COUNT_SUFFIXES):
        return f"{value:,}"
    return str(value)


def _indexed_subset_row(path: Path, attrs: dict[str, Any], size: int) -> dict[str, str]:
    """One display row. The directory name is the identity, never the attrs name:
    a group whose recorded name disagrees is flagged, not silently relabelled.

    `size` is measured once by the caller so the total below is the sum of the
    rows' sizes rather than a second walk of every chunk file (#263 review).
    """
    return {
        "Subset": path.name,
        "Profile": _indexed_subset_value(attrs, "statistic_profile"),
        "Requested": _indexed_subset_value(attrs, "requested_count"),
        "Resolved": _indexed_subset_value(attrs, "resolved_count"),
        "Absent": _indexed_subset_value(attrs, "absent_count"),
        "Overflow-only": _indexed_subset_value(attrs, "overflow_only_count"),
        "Analyses": _indexed_subset_value(attrs, "n_analyses"),
        "Physical bytes": f"{size:,} B ({_human_bytes(size)})",
        "Input SHA-256": _indexed_subset_value(attrs, "input_sha256"),
        "Reference assembly": _indexed_subset_value(attrs, "reference_assembly"),
        "Source store": _indexed_subset_value(attrs, "source_store_id"),
        "Source release": _indexed_subset_value(attrs, "source_release_id"),
        "Source format": _indexed_subset_value(attrs, "source_format_version"),
        "Builder": _indexed_subset_value(attrs, "builder_version"),
        "Created (UTC)": _indexed_subset_value(attrs, "created_at"),
    }


def _published_subset_paths(store_path: Path) -> list[Path]:
    """The published subset groups under a release, sorted by name.

    A leading `.` is the module's staging/replacement convention, so those
    entries are skipped rather than rendered as a named subset.
    """
    # A Hybrid release keeps its subsets inside the nested Dense Component, so
    # the namespace is located through the same resolver the writer and reader
    # use rather than by hand (#271).
    namespace = indexed_subset_namespace_path(store_path)
    if not namespace.is_dir():
        return []
    return [
        entry
        for entry in sorted(namespace.iterdir())
        if entry.is_dir() and not entry.name.startswith(".")
    ]


def _subset_problem(
    store_path: Path, path: Path, attrs: dict[str, Any]
) -> dict[str, str] | None:
    """A reason a named directory must be surfaced rather than rendered as valid.

    The recorded name must match the directory, and the group must pass the
    Indexed Variant Subset **structural read seam** (`open_indexed_subset`):
    schema, profile, order, declared counts, exact array set, plane dtypes and
    shapes, and Variant Indices in bounds. It deliberately does not decode or
    compare values -- that is `validate`'s expensive job -- so overview stays a
    cheap presentation of a structurally complete group.
    """
    recorded = str(attrs.get("indexed_subset_name", ""))
    if recorded != path.name:
        return {
            "name": path.name,
            "issue": (
                f"recorded name {recorded!r} disagrees with its directory name; "
                "not trusted as a published subset"
            ),
        }
    _, error = _probe(lambda: open_indexed_subset(store_path, path.name))
    if error is None:
        return None
    if isinstance(error, IndexedSubsetError):
        return {
            "name": path.name,
            "issue": str(error).removeprefix(f"store {store_path}: "),
        }
    return {
        "name": path.name,
        "issue": (
            "structural check could not read the store or group "
            f"({type(error).__name__}: {error})"
        ),
    }


def _collect_indexed_subsets(
    store_path: Path,
) -> tuple[list[dict[str, str]], int, list[dict[str, str]]]:
    """Valid rows, their total physical bytes, and every invalid entry.

    A named directory that cannot be read, whose recorded name disagrees with
    its directory, or that fails the structural read seam is reported in
    `problems` and is **not** also rendered or counted as valid -- an entry that
    is both listed and flagged would let a partial group inflate the totals.
    """
    rows: list[dict[str, str]] = []
    total = 0
    problems: list[dict[str, str]] = []
    for path in _published_subset_paths(store_path):
        attrs = _read_subset_attrs(path)
        if attrs is None:
            problems.append({"name": path.name, "issue": "unreadable Zarr group"})
            continue
        problem = _subset_problem(store_path, path, attrs)
        if problem is not None:
            problems.append(problem)
            continue
        size = _physical_bytes(path)
        rows.append(_indexed_subset_row(path, attrs, size))
        total += size
    return rows, total, problems


def _gather_indexed_subsets(store_path: Path) -> IndexedSubsetsOverview | None:
    """The Indexed Variant Subsets tab (ADR 0061), or None when the release has
    neither a published subset nor an invalid named entry -- in which case the
    tab is absent rather than empty."""
    rows, total, problems = _collect_indexed_subsets(store_path)
    if not rows and not problems:
        return None
    return IndexedSubsetsOverview(
        rows=tuple(rows),
        total_bytes=total,
        problems=tuple(problems),
    )


def gather_store_overview(store_path: str | Path, table: AnalysesTable) -> StoreOverview:
    """Read everything `overview.html` shows out of the release at `store_path`.

    `table` is the release's `analyses.tsv`, already read by the caller (the
    builder wrote it from records it holds; `regenerate-overview` reads the
    file). Everything the page takes from the filesystem, and every decision
    about how much of it to show, is resolved here.
    """
    store_path = Path(store_path)
    store_id, release_id, completion_state = _manifest_summary(store_path)
    return StoreOverview(
        store_id=store_id,
        release_id=release_id,
        completion_state=completion_state,
        table=table,
        files=_scan_store_files(store_path),
        rho=_gather_rho(store_path, table),
        indexed_subsets=_gather_indexed_subsets(store_path),
    )

# ── Render: a StoreOverview value -> the page (no I/O at all) ───────────────

_STYLE = """
:root {
  --paper: #f1f2ef;
  --surface: #fbfbf9;
  --ink: #14181c;
  --ink-soft: #4b5459;
  --muted: #8b9095;
  --hairline: #dcdcd6;
  --hairline-strong: #c3c4be;
  --accent: #1f5fd1;
  --accent-soft: #dce6fa;
  --accent-wash: #eef3fc;
  --accent-neg: #c2410c;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, "Liberation Mono", monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
header {
  padding: 28px 32px 20px;
  border-bottom: 1px solid var(--hairline);
  background: var(--surface);
}
.eyebrow {
  font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 8px; display: flex; align-items: center; gap: 8px;
}
.eyebrow::before {
  content: ""; width: 7px; height: 7px; background: var(--accent); border-radius: 1px;
  display: inline-block;
}
header h1 { margin: 0; font-size: 22px; font-weight: 650; letter-spacing: -0.01em; }
header .meta { margin-top: 6px; font-size: 13px; color: var(--ink-soft); font-family: var(--mono); }
.tabs {
  display: flex; gap: 4px; padding: 0 32px; background: var(--surface);
  border-bottom: 1px solid var(--hairline);
}
.tabs button {
  font: inherit; font-size: 13px; padding: 10px 14px; border: none; background: none; cursor: pointer;
  color: var(--ink-soft); border-bottom: 2px solid transparent; margin-bottom: -1px;
}
.tabs button:hover { color: var(--ink); }
.tabs button.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }
main { padding: 20px 32px 60px; }
.tab-panel.hidden { display: none; }
input[type="text"] {
  font: inherit; padding: 0.5rem 0.75rem; width: 24rem; max-width: 100%; margin-bottom: 0.9rem;
  border: 1px solid var(--hairline-strong); border-radius: 6px; background: var(--surface); color: var(--ink);
}
input[type="text"]:focus { outline: 2px solid var(--accent-soft); border-color: var(--accent); }
.table-scroll {
  overflow-x: auto; border: 1px solid var(--hairline); border-radius: 8px; background: var(--surface);
}
table { border-collapse: collapse; font-size: 12.5px; width: max-content; min-width: 100%; }
th, td {
  padding: 6px 10px; text-align: left; white-space: nowrap; border-bottom: 1px solid var(--hairline);
}
th {
  cursor: pointer; background: var(--surface); position: sticky; top: 0; z-index: 2;
  color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.03em;
  border-bottom: 1px solid var(--hairline-strong);
}
th.sorted-asc::after { content: " \\25B2"; color: var(--accent); }
th.sorted-desc::after { content: " \\25BC"; color: var(--accent); }
tbody tr:nth-child(even) { background: var(--paper); }
td.blank { color: var(--muted); }
th.sticky-col, td.sticky-col {
  position: sticky; left: 0; z-index: 1; background: var(--surface); box-shadow: 1px 0 0 var(--hairline-strong);
}
th.sticky-col { z-index: 3; }
tbody tr:nth-child(even) td.sticky-col { background: var(--paper); }
.bar-wrap {
  display: inline-block; width: 60px; height: 0.65em; background: var(--hairline); border-radius: 2px;
  overflow: hidden; vertical-align: middle; margin-right: 6px;
}
.bar-fill { display: block; height: 100%; background: var(--accent); }
.guide-intro { max-width: 62ch; color: var(--ink-soft); font-size: 13.5px; margin: 0 0 16px; }
table.guide { border-collapse: collapse; width: 100%; max-width: 760px; font-size: 13px; }
table.guide td { padding: 8px 4px; border-bottom: 1px solid var(--hairline); white-space: normal; }
table.guide td.fname { font-family: var(--mono); font-size: 12.5px; white-space: nowrap; padding-right: 24px; }
table.guide td.fdesc { color: var(--ink-soft); }
.badge {
  font-family: var(--mono); font-size: 10px; padding: 1px 7px; border-radius: 999px; margin-left: 8px;
  letter-spacing: 0.02em;
}
.badge.human { background: var(--accent-wash); color: var(--accent); }
.badge.internal { background: var(--hairline); color: var(--ink-soft); }
.rho-summary { border-collapse: collapse; width: 100%; max-width: 520px; font-size: 13px; margin-bottom: 20px; }
.rho-summary td { padding: 6px 4px; border-bottom: 1px solid var(--hairline); }
.rho-summary td:first-child { color: var(--ink-soft); padding-right: 24px; white-space: nowrap; }
.rho-section-heading { font-size: 14px; font-weight: 650; margin: 28px 0 8px; }
.rho-heatmap-wrap { position: relative; display: inline-block; }
.rho-heatmap { border: 1px solid var(--hairline); border-radius: 4px; cursor: crosshair; }
.rho-heatmap-tooltip {
  position: absolute; pointer-events: none; font-size: 12px; font-family: var(--mono);
  background: var(--ink); color: var(--surface); padding: 6px 9px; border-radius: 6px;
  white-space: nowrap; transform: translate(-50%, -110%); z-index: 5;
}
.rho-heatmap-tooltip.hidden { display: none; }
"""

_SCRIPT = """
function initTable(table) {
  const search = document.getElementById(table.dataset.searchable);
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);

  search.addEventListener("input", () => {
    const q = search.value.toLowerCase();
    for (const row of rows) {
      row.style.display = row.textContent.toLowerCase().includes(q) ? "" : "none";
    }
  });

  table.tHead.querySelectorAll("th").forEach((th, colIndex) => {
    let ascending = true;
    th.addEventListener("click", () => {
      table.tHead.querySelectorAll("th").forEach(
        (h) => h.classList.remove("sorted-asc", "sorted-desc")
      );
      const sorted = rows.slice().sort((a, b) => {
        const av = a.cells[colIndex].textContent;
        const bv = b.cells[colIndex].textContent;
        const an = parseFloat(av), bn = parseFloat(bv);
        const cmp = (!isNaN(an) && !isNaN(bn)) ? an - bn : av.localeCompare(bv);
        return ascending ? cmp : -cmp;
      });
      sorted.forEach((row) => tbody.appendChild(row));
      th.classList.add(ascending ? "sorted-asc" : "sorted-desc");
      ascending = !ascending;
    });
  });
}
document.querySelectorAll("table[data-searchable]").forEach(initTable);

document.querySelectorAll(".tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.remove("hidden");
  });
});

function rhoColor(v) {
  if (v === null) return "#dcdcd6";
  const t = Math.max(-1, Math.min(1, v));
  const mix = (a, b, f) => Math.round(a + (b - a) * f);
  // Canvas can't read CSS custom properties -- keep these RGB triples in
  // sync with --surface / --accent / --accent-neg in _STYLE by hand.
  const white = [251, 251, 249];
  const end = t >= 0 ? [31, 95, 209] : [194, 65, 12];
  const f = Math.abs(t);
  const [r, g, b] = [mix(white[0], end[0], f), mix(white[1], end[1], f), mix(white[2], end[2], f)];
  return `rgb(${r},${g},${b})`;
}

function initRhoHeatmap(wrap) {
  const canvas = wrap.querySelector("canvas.rho-heatmap");
  const tooltip = wrap.querySelector(".rho-heatmap-tooltip");
  const data = JSON.parse(wrap.querySelector(".rho-heatmap-data").textContent);
  const n = data.k;
  const cell = canvas.width / n;
  const ctx = canvas.getContext("2d");
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      ctx.fillStyle = rhoColor(data.rho[i * n + j]);
      ctx.fillRect(j * cell, i * cell, cell, cell);
    }
  }
  canvas.addEventListener("mousemove", (event) => {
    const rect = canvas.getBoundingClientRect();
    const i = Math.floor((event.clientY - rect.top) / cell);
    const j = Math.floor((event.clientX - rect.left) / cell);
    if (i < 0 || i >= n || j < 0 || j >= n) {
      tooltip.classList.add("hidden");
      return;
    }
    const rho = data.rho[i * n + j];
    const rhoText = rho === null ? "NaN" : rho.toFixed(4);
    const nn = data.n_null[i * n + j];
    tooltip.textContent = `${data.labels[i]} x ${data.labels[j]}: rho=${rhoText}, n_null=${nn}`;
    tooltip.style.left = `${event.clientX - rect.left}px`;
    tooltip.style.top = `${event.clientY - rect.top}px`;
    tooltip.classList.remove("hidden");
  });
  canvas.addEventListener("mouseleave", () => tooltip.classList.add("hidden"));
}
document.querySelectorAll(".rho-heatmap-wrap").forEach(initRhoHeatmap);
"""

_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{style}</style>
</head>
<body>
<header>
  <div class="eyebrow">OpenGWASDB Store Release</div>
  <h1>{title}</h1>
  <div class="meta">{meta}</div>
</header>
<nav class="tabs">
{tab_buttons}
</nav>
<main>
{tab_sections}
</main>
<script>{script}</script>
</body>
</html>
"""



def _display_fieldnames(table: AnalysesTable) -> tuple[str, ...]:
    """`analysis_id` first (the sticky identity column), then every other
    column in `analyses.tsv`'s own order (store-format spec §7a) -- a
    display-only reordering; the on-disk column order is unchanged.
    """
    if "analysis_id" not in table.fieldnames:
        return table.fieldnames
    rest = [name for name in table.fieldnames if name != "analysis_id"]
    return ("analysis_id", *rest)


def _ancestry_prop_columns(fieldnames: tuple[str, ...]) -> list[str]:
    return sorted(name for name in fieldnames if name.startswith(ANCESTRY_PROP_PREFIX))


def _ancestry_fieldnames(table: AnalysesTable) -> tuple[str, ...]:
    """`analysis_id`, `assigned_ancestry`, then every `ancestry_prop_<population>`
    column present -- the same Ancestry Composition data already in
    `analyses.tsv`, just a narrower column set than the Analyses tab
    (issue #37)."""
    return ("analysis_id", "assigned_ancestry", *_ancestry_prop_columns(table.fieldnames))


def _bar_html(value: str) -> str:
    """An inline width-bar for a 0..1 proportion, followed by its value --
    an at-a-glance composition view with no charting library (issue #37)."""
    try:
        fraction = float(value)
    except ValueError:
        return html.escape(value)
    pct = max(0.0, min(1.0, fraction)) * 100
    return (
        f'<span class="bar-wrap"><span class="bar-fill" style="width:{pct:.1f}%"></span></span>'
        f"{html.escape(value)}"
    )


def _cell_classes(name: str, value: str, *, sticky_column: str | None) -> str:
    """A cell's CSS classes: the sticky identity column, and the muted blank
    an absent value gets (absence renders as an em dash, never as a zero)."""
    return " ".join(
        classes
        for classes in (
            "sticky-col" if name == sticky_column else "",
            "" if value else "blank",
        )
        if classes
    )


def _cell_body(name: str, value: str, *, bar_columns: frozenset[str]) -> str:
    """A cell's content: a proportion's inline bar, else the escaped value."""
    if value and name in bar_columns:
        return _bar_html(value)
    return html.escape(value) if value else "—"


def _render_row(
    row: dict[str, str],
    fieldnames: tuple[str, ...],
    *,
    sticky_column: str | None,
    bar_columns: frozenset[str],
) -> str:
    cells = []
    for name in fieldnames:
        value = row.get(name, "")
        classes = _cell_classes(name, value, sticky_column=sticky_column)
        class_attr = f' class="{classes}"' if classes else ""
        body = _cell_body(name, value, bar_columns=bar_columns)
        cells.append(f"<td{class_attr}>{body}</td>")
    return "<tr>" + "".join(cells) + "</tr>"


def _render_table_section(
    *,
    table_id: str,
    search_id: str,
    search_placeholder: str,
    fieldnames: tuple[str, ...],
    rows: tuple[dict[str, str], ...],
    sticky_column: str | None = None,
    bar_columns: frozenset[str] = frozenset(),
) -> str:
    header_cells = "".join(
        f'<th class="sticky-col">{html.escape(name)}</th>'
        if name == sticky_column
        else f"<th>{html.escape(name)}</th>"
        for name in fieldnames
    )
    body_rows = "\n".join(
        _render_row(row, fieldnames, sticky_column=sticky_column, bar_columns=bar_columns)
        for row in rows
    )
    return (
        f'<input id="{search_id}" type="text" placeholder="{html.escape(search_placeholder)}">\n'
        '<div class="table-scroll">\n'
        f'<table id="{table_id}" data-searchable="{search_id}">\n'
        f"<thead><tr>{header_cells}</tr></thead>\n"
        f"<tbody>\n{body_rows}\n</tbody>\n"
        "</table>\n"
        "</div>"
    )


def _render_analyses_section(table: AnalysesTable) -> str:
    """The Analyses tab: every `analyses.tsv` column, searchable and sortable."""
    return _render_table_section(
        table_id="analyses",
        search_id="search-analyses",
        search_placeholder="Filter rows...",
        fieldnames=_display_fieldnames(table),
        rows=table.rows,
        sticky_column="analysis_id",
    )


def _render_ancestry_section(table: AnalysesTable) -> str:
    """The Ancestry tab: composition as inline width-bars (issue #37)."""
    return _render_table_section(
        table_id="ancestry",
        search_id="search-ancestry",
        search_placeholder="Filter analyses...",
        fieldnames=_ancestry_fieldnames(table),
        rows=table.rows,
        sticky_column="analysis_id",
        bar_columns=frozenset(_ancestry_prop_columns(table.fieldnames)),
    )


def _render_guide_section(files: tuple[FileGuideEntry, ...]) -> str:
    """The Guide tab: every top-level entry of this release, and what it is."""
    rows = "\n".join(
        "<tr>"
        f'<td class="fname">{html.escape(entry.name)}'
        f'<span class="badge {html.escape(entry.kind)}">{html.escape(entry.kind)}</span></td>'
        f'<td class="fdesc">{html.escape(entry.description)}</td>'
        "</tr>"
        for entry in files
    )
    return (
        '<p class="guide-intro">Files in this Store Release, generated at build time. '
        '"human" files are safe to open directly; "internal" files are read by the '
        "query API and should not be edited by hand.</p>\n"
        f"<table class=\"guide\">\n<tbody>\n{rows}\n</tbody>\n</table>"
    )


_HIST_TICKS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _rho_histogram_svg(histogram: RhoHistogram) -> str:
    """The distribution plot, drawn from the gathered bins."""
    counts, edges = histogram.counts, histogram.edges
    width, height = 640, 190
    left, right, top, bottom = 48, 16, 12, 34  # plot-area margins for axis labels
    plot_w, plot_h = width - left - right, height - top - bottom
    max_count = max(max(counts), 1) if counts else 1
    bar_w = plot_w / max(len(counts), 1)

    def x_of(rho: float) -> float:
        return left + (rho - edges[0]) / ((edges[-1] - edges[0]) or 1.0) * plot_w

    bars = []
    for i, count in enumerate(counts):
        bar_h = (count / max_count) * plot_h
        title = f"{edges[i]:.2f} to {edges[i + 1]:.2f}: {count:,} pairs"
        bars.append(
            f'<rect x="{x_of(edges[i]):.1f}" y="{top + plot_h - bar_h:.1f}" '
            f'width="{bar_w * 0.85:.1f}" height="{bar_h:.1f}" fill="var(--accent)">'
            f"<title>{html.escape(title)}</title></rect>"
        )

    axes = (
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" '
        'stroke="var(--hairline-strong)"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        'stroke="var(--hairline-strong)"/>'
        f'<line x1="{x_of(0.0):.1f}" y1="{top}" x2="{x_of(0.0):.1f}" y2="{top + plot_h}" '
        'stroke="var(--hairline-strong)" stroke-dasharray="3,3"/>'
    )
    x_ticks = "".join(
        f'<text x="{x_of(t):.1f}" y="{top + plot_h + 16}" font-size="10" fill="var(--muted)" '
        f'text-anchor="middle" font-family="var(--mono)">{t:g}</text>'
        for t in _HIST_TICKS
    )
    y_ticks = (
        f'<text x="{left - 6}" y="{top + plot_h + 4}" font-size="10" fill="var(--muted)" '
        f'text-anchor="end" font-family="var(--mono)">0</text>'
        f'<text x="{left - 6}" y="{top + 8}" font-size="10" fill="var(--muted)" '
        f'text-anchor="end" font-family="var(--mono)">{max_count:,}</text>'
    )
    x_label = (
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 4}" font-size="11" '
        'fill="var(--ink-soft)" text-anchor="middle">rho</text>'
    )
    y_mid = top + plot_h / 2
    y_label = (
        f'<text x="12" y="{y_mid:.1f}" font-size="11" fill="var(--ink-soft)" '
        f'text-anchor="middle" transform="rotate(-90 12 {y_mid:.1f})">Analysis pairs</text>'
    )
    return (
        f'<svg width="{width}" height="{height}" role="img" '
        f'aria-label="Distribution of {histogram.n_finite_pairs:,} finite rho values">'
        f"{axes}{''.join(bars)}{x_ticks}{y_ticks}{x_label}{y_label}</svg>"
    )


def _rho_pairs_table(rows: tuple[RhoPairRow, ...], *, table_id: str) -> str:
    """One ranked-pairs table, reusing the page's searchable/sortable table."""
    fieldnames = ("Analysis A", "Label A", "Analysis B", "Label B", "Rho", "Support (n_null)")
    return _render_table_section(
        table_id=table_id,
        search_id=f"search-{table_id}",
        search_placeholder="Filter pairs...",
        fieldnames=fieldnames,
        rows=tuple(
            {
                "Analysis A": row.analysis_a,
                "Label A": row.label_a,
                "Analysis B": row.analysis_b,
                "Label B": row.label_b,
                "Rho": f"{row.rho:.4f}",
                "Support (n_null)": f"{row.n_null:,}",
            }
            for row in rows
        ),
    )


def _rho_clustered_heatmap(heatmap: RhoHeatmap | None, n_analyses: int) -> str:
    """The clustered submatrix: a canvas the page paints from the embedded
    payload the gather step prepared."""
    if heatmap is None:
        return "<p>Not enough Analyses to render a heatmap.</p>"
    k = heatmap.k
    payload = {
        "k": k,
        "labels": list(heatmap.labels),
        "rho": list(heatmap.rho),
        "n_null": list(heatmap.n_null),
    }
    cell_px = max(4, min(12, 640 // k))
    return (
        f'<p class="guide-intro">Hierarchically clustered submatrix of the {k} Analyses with the '
        "strongest pairwise |rho| relationship elsewhere in the store. Hover a cell for its pair "
        "and support.</p>\n"
        '<div class="rho-heatmap-wrap">\n'
        f'<canvas class="rho-heatmap" width="{k * cell_px}" height="{k * cell_px}"></canvas>\n'
        '<div class="rho-heatmap-tooltip hidden"></div>\n'
        f'<script type="application/json" class="rho-heatmap-data">{json.dumps(payload)}</script>\n'
        "</div>"
    )


def _rho_partner_cell(row: RhoTraitRow) -> str:
    """An Analysis's strongest partner, labelled when its label is non-blank."""
    if row.partner_id is None:
        return ""
    return f"{row.partner_id} ({row.partner_label})" if row.partner_label else row.partner_id


def _rho_trait_summary_table(rows: tuple[RhoTraitRow, ...]) -> str:
    """One row per Analysis: its mean |rho| and its single strongest
    relationship (partner + rho + support) -- "which traits are most
    entangled", complementing the pair-level tables above."""
    fieldnames = ("Analysis", "Label", "Mean |rho|", "Max |rho|", "Strongest partner", "Support")
    return _render_table_section(
        table_id="rho-trait-summary",
        search_id="search-rho-trait-summary",
        search_placeholder="Filter Analyses...",
        fieldnames=fieldnames,
        rows=tuple(
            {
                "Analysis": row.analysis,
                "Label": row.label,
                "Mean |rho|": "" if row.mean_abs_rho is None else f"{row.mean_abs_rho:.4f}",
                "Max |rho|": "" if row.max_abs_rho is None else f"{row.max_abs_rho:.4f}",
                "Strongest partner": _rho_partner_cell(row),
                "Support": "" if row.n_null is None else f"{row.n_null:,}",
            }
            for row in rows
        ),
        sticky_column="Analysis",
    )


def _render_rho_section(rho: RhoOverview) -> str:
    """The Rho Matrix tab (ADR 0025): summary, distribution, top pairs, and a
    clustered heatmap -- present only when the release carries a Rho group."""
    summary_body = "\n".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(value)}</td></tr>"
        for key, value in rho.summary
    )
    heatmap_k = rho.heatmap.k if rho.heatmap is not None else rho.n_analyses

    return (
        '<p class="guide-intro">Rho (ADR 0025) is the correlation between two Analyses\' '
        "statistics under the null -- sample overlap x phenotypic correlation -- estimated by "
        "the pleiodb conditional-MLE method on shared non-significant Z-scores.</p>\n"
        '<div class="rho-section-heading">Run summary</div>\n'
        f'<table class="rho-summary"><tbody>{summary_body}</tbody></table>\n'
        '<div class="rho-section-heading">Distribution of rho values</div>\n'
        f"{_rho_histogram_svg(rho.histogram)}\n"
        '<div class="rho-section-heading">Top 30 positive rho pairs</div>\n'
        f"{_rho_pairs_table(rho.top_positive, table_id='rho-top-pos')}\n"
        '<div class="rho-section-heading">Top 30 negative rho pairs</div>\n'
        f"{_rho_pairs_table(rho.top_negative, table_id='rho-top-neg')}\n"
        '<div class="rho-section-heading">Clustered heatmap (top '
        f'{heatmap_k} Analyses by |rho|)</div>\n'
        f"{_rho_clustered_heatmap(rho.heatmap, rho.n_analyses)}\n"
        '<div class="rho-section-heading">Trait-level rho summary</div>\n'
        f"{_rho_trait_summary_table(rho.traits)}"
    )


def _render_subset_problems(problems: tuple[dict[str, str], ...]) -> str:
    rows = "\n".join(
        "<tr>"
        f'<td class="fname">{html.escape(problem["name"])}</td>'
        f'<td class="fdesc">{html.escape(problem["issue"])}</td>'
        "</tr>"
        for problem in problems
    )
    return (
        '<p class="guide-intro"><strong>Invalid or unreadable entries</strong> — '
        "these named directories are not trusted as published subsets and their values "
        "are not invented.</p>\n"
        f'<table class="guide"><tbody>\n{rows}\n</tbody></table>\n'
    )


def _render_indexed_subsets_section(subsets: IndexedSubsetsOverview) -> str:
    """The Indexed Variant Subsets tab (ADR 0061): one row per published name,
    read from the group's own metadata."""
    parts = [
        '<p class="guide-intro">Optional Indexed Variant Subsets (ADR 0061) are derived, '
        "rebuildable query indexes over a caller-supplied variant list. Counts, checksum, "
        "Reference Assembly and build provenance are read from each group's recorded "
        "metadata; physical size is computed from the group's files at render time.</p>\n"
    ]
    if subsets.problems:
        parts.append(_render_subset_problems(subsets.problems))
    if subsets.rows:
        parts.append(
            f'<p class="guide-intro">Total physical size: {subsets.total_bytes:,} B '
            f"({_human_bytes(subsets.total_bytes)}).</p>\n"
        )
        parts.append(
            _render_table_section(
                table_id="indexed-subsets",
                search_id="search-indexed-subsets",
                search_placeholder="Filter subsets...",
                fieldnames=_INDEXED_SUBSET_FIELDS,
                rows=subsets.rows,
                sticky_column="Subset",
            )
        )
    else:
        parts.append('<p class="guide-intro">No readable published subsets.</p>\n')
    return "".join(parts)


def _overview_tabs(overview: StoreOverview) -> list[tuple[str, str, str]]:
    """The store's tabs in display order.

    Rho and Indexed Variant Subsets are opt-in derived artifacts, so each is
    present only when the release actually carries one; the Guide is always
    last and always present (issue #38).
    """
    tabs: list[tuple[str, str, str]] = [
        ("analyses", "Analyses", _render_analyses_section(overview.table)),
        ("ancestry", "Ancestry", _render_ancestry_section(overview.table)),
    ]
    if overview.rho is not None:
        tabs.append(("rho", "Rho Matrix", _render_rho_section(overview.rho)))
    if overview.indexed_subsets is not None:
        subsets = _render_indexed_subsets_section(overview.indexed_subsets)
        tabs.append(("indexed-subsets", "Indexed Subsets", subsets))
    tabs.append(("guide", "Guide", _render_guide_section(overview.files)))
    return tabs


def _header_meta(overview: StoreOverview) -> str:
    """The header line: who the release is, how it was completed, and how many
    Analyses it carries. Blank parts are left out rather than shown empty."""
    parts = [
        part
        for part in (
            overview.store_id,
            f"release {overview.release_id}" if overview.release_id else "",
            overview.completion_state,
        )
        if part
    ]
    parts.append(f"{len(overview.table.rows)} Analyses")
    return html.escape(" · ".join(parts))


def _render_tab_buttons(tabs: list[tuple[str, str, str]]) -> str:
    """The tab strip, the first tab active so the page opens on Analyses."""
    return "\n".join(
        f'<button data-tab="{key}"{active}>{html.escape(label)}</button>'
        for i, (key, label, _section) in enumerate(tabs)
        for active in (' class="active"' if i == 0 else "",)
    )


def _render_tab_sections(tabs: list[tuple[str, str, str]]) -> str:
    """Every tab's panel; only the first is visible until a tab is clicked."""
    return "\n".join(
        f'<section id="tab-{key}" class="tab-panel{hidden}">\n{section}\n</section>'
        for i, (key, _label, section) in enumerate(tabs)
        for hidden in ("" if i == 0 else " hidden",)
    )


def render_overview_html(overview: StoreOverview, *, title: str = "OpenGWASDB Analyses") -> str:
    """Render `overview` as the whole `overview.html` document.

    Pure: every file, group and statistic it shows was resolved by
    `gather_store_overview()`, so this touches no store, opens no Zarr group
    and reads no directory (issue #57).
    """
    tabs = _overview_tabs(overview)
    return _TEMPLATE.format(
        title=html.escape(title),
        style=_STYLE,
        meta=_header_meta(overview),
        tab_buttons=_render_tab_buttons(tabs),
        tab_sections=_render_tab_sections(tabs),
        script=_SCRIPT,
    )


def write_overview_html(
    output_path: str | Path, table: AnalysesTable, *, title: str = "OpenGWASDB Analyses"
) -> Path:
    """Render `table` (an `analyses.tsv` read) as `overview.html` at `output_path`."""
    output_path = Path(output_path)
    out_path = output_path / "overview.html"
    out_path.write_text(
        render_overview_html(gather_store_overview(output_path, table), title=title),
        encoding="utf-8",
    )
    return out_path
