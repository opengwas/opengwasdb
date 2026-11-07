"""Layout-independent query facade (ADR-0006, ADR-0020, ADR-0033).

Three adapter classes share one result contract but not one method set --
`query_store()` / `OpenGWASDBStore.query()` dispatch to the right one from
the store manifest: `StoreQuery` (Dense), `RaggedStoreQuery` (Ragged),
`HybridStoreQuery` (Hybrid). See ADR-0033 for the full rationale; in short:

Result shape -- every association-returning method returns an
`opengwasdb.associations.Associations` record. Read as a mapping it is the
contract ADR 0020 set: ``{"variant_index", "analysis_index", "z", "se", "eaf",
"association_status"}`` as parallel arrays (int32, int32, float32, float32,
float32, object). A Ragged release carrying per-association sample sizes adds
``"n"`` (float64, NaN where the build was given no N for the pair) to every
result, empty ones included, and a release without the plane never has the key
(ADR 0067).

Ordering -- `analysis()`, `phewas()`, `range_phewas()`, `range_by_analysis()`,
and `lookup()` make no ordering guarantee beyond grouping (rows for one scan
target are contiguous, not sorted); `lookup()` is Analysis-major on every
layout, its rows grouped by Analysis in the order the request first names each.
`top_hits()` returns genomic order -- sorted by `(analysis_index, variant_index)`
-- on every adapter and every internal path (indexed and full-scan fallback
alike), matching the `group.attrs["order"]` the top-hit index itself is built
with.

observed_only / limit -- both apply as filter-then-limit everywhere they are
accepted, on every internal path: `observed_only` narrows the result set
first, `limit` then caps the already-filtered rows.

Presence (ADR 0070, superseding ADR 0033 §7) -- every read goes through
`opengwasdb.associations`, so one rule decides presence on every layout and
every method, `top_hits()` included: an association is present exactly when
neither its Z nor its SE carries its plane's missing marker (spec §15). An
infinite SE is a value, returned with its Status. What a read returns follows
what each layout stores: a Dense grid stores no association at a missing cell,
so Dense results omit them (ADR 0020); a CSR stores a row for every attempted
pair, so Ragged and Ragged Overflow results return such a row labelled
`missing`. A cell spec §15 forbids -- one marker without the other, an imputed
flag outside {0, 1}, a flag on a missing cell, or a non-canonical NaN marker --
is refused with `InvalidAssociations` naming the component and the cell, and a
Reference-Completed component without its imputed mask is refused on opening.

Method availability -- `variants_table()`/`analyses_table()` and
`__enter__`/`__exit__` are present on all three adapters.
`analyses_table()` returns the same shape on every adapter -- every column
of `analyses.tsv` (ADR-0034's unified schema every layout shares), keyed by
`analysis_index`. Ragged rows populate the molecular-QTL columns (tissue,
context, and gene identity carried via `analysis_label`/`trait_ontology_id`,
ADR 0035) that Dense/Hybrid rows mostly leave blank, and leave Dense/Hybrid's
other Trait-identity/effect-scale columns mostly blank in turn; a caller
grouping Ragged Analyses by a shared gene filters this table on
`trait_ontology_id` rather than through a separate lookup. `analysis_label`
reaches a caller as the stored column, blank included -- a release that
supplied no label says so, rather than having one invented for it; a *display*
label is `opengwasdb.model.analyses.analysis_label()`'s job, the one
implementation of that fallback (issue #57). `rho()`/`rho_row()`/
`rho_matrix()` (ADR-0025, a Dense storage artifact) are exposed
on `StoreQuery` and on `HybridStoreQuery` (delegated to its Dense Component);
`RaggedStoreQuery` has no Rho Matrix format. `range_by_analysis()` (query by
probe/TSS position) is Ragged-only: it scans `AnalysesIndex`'s already
store-open-time-loaded rows for `trait_chr`/`trait_bp`, which only
Ragged/molecular-QTL releases populate.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import NoReturn, Self

import numpy as np

from opengwasdb import top_hits
from opengwasdb.associations import (
    Associations,
    Cells,
    CsrReader,
    DenseGridReader,
    SubsetReader,
)
from opengwasdb.index import AnalysesIndex
from opengwasdb.layouts.dense.indexed_subsets import (
    IndexedSubsetError,
    IndexedSubsetLayoutError,
    rescope_indexed_subset_error,
)
from opengwasdb.layouts.dense.rho import DenseRhoReader
from opengwasdb.layouts.hybrid.layout import dense_component_path, dense_to_shared_path
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.model.enums import CompletionState, PrimaryStorageLayout
from opengwasdb.query.resolve import resolve_rows
from opengwasdb.store.open import OpenGWASDBStore, open_store
from opengwasdb.top_hits import TierRows, TopHitTier, TopHitTiers
from opengwasdb.variants import VariantAxis


def _ragged_tier_record(
    tier: TopHitTier,
    analysis_index: int | None,
    csr: RaggedCSRReader,
    *,
    with_n: bool,
    source: str,
) -> Associations:
    """A Ragged top-hit tier's rows (all of it, or one Analysis's slice) as a record.

    `se` is required -- a Ragged index always wrote it. A field the tier lacks is
    resolved back to each hit's CSR cell: `eaf` from the frequency plane, `n`
    always (the tiers carry none, ADR 0067), and the imputed flags from the
    sequence's own mask. A tier without `imputed` reads all-observed only where
    the sequence holds no mask, which an opened reader allows only on a
    component that is observed by construction (`CsrReader`'s `requires_mask`).
    """
    rows = tier.read(analysis_index, require=("se",))
    assert rows.se is not None
    vi, ai = rows.variant_index, rows.analysis_index
    eaf = rows.eaf if rows.eaf is not None else csr.eaf_pairs(vi, ai)
    imputed = rows.imputed
    if imputed is None and csr.has_imputed:
        imputed = csr.imputed_at(_tier_positions(csr, vi, ai, source))
    return Associations.from_decoded(
        vi,
        ai,
        rows.z,
        rows.se,
        eaf,
        imputed,
        csr.n_pairs(vi, ai) if with_n else None,
        source=source,
    )


def _tier_positions(
    csr: RaggedCSRReader, variant_index: np.ndarray, analysis_index: np.ndarray, source: str
) -> np.ndarray:
    """Each tier hit's flat CSR position; a hit the sequence does not hold is refused."""
    slots, positions = csr.pair_positions(variant_index, analysis_index)
    if len(slots) != len(variant_index):
        raise LookupError(
            f"{source}: {len(variant_index) - len(slots):,} top hit(s) name a cell the "
            "association sequence does not hold"
        )
    out = np.empty(len(variant_index), dtype=np.int64)
    out[slots] = positions
    return out


def _empty_rho_result() -> dict[str, np.ndarray]:
    return {
        "analysis_id_a": np.empty(0, dtype=object),
        "analysis_id_b": np.empty(0, dtype=object),
        "rho": np.empty(0, dtype="float32"),
        "n_null": np.empty(0, dtype="int64"),
    }


def _empty_rho_row_result() -> dict[str, np.ndarray]:
    return {
        "analysis_id": np.empty(0, dtype=object),
        "rho": np.empty(0, dtype="float32"),
        "n_null": np.empty(0, dtype="int64"),
    }


def _empty_rho_matrix_result() -> dict[str, np.ndarray]:
    return {
        "analysis_id": np.empty(0, dtype=object),
        "rho": np.empty((0, 0), dtype="float32"),
        "n_null": np.empty((0, 0), dtype="int64"),
    }


def _variants_table(variant_axis: VariantAxis) -> dict[int, dict]:
    """Return all variants keyed by variant_index.

    The Store Variant Table's shape is layout-independent, so all three
    adapters share this projection rather than each repeating it.
    """
    return {
        r.variant_index: {
            "alid": r.alid,
            "chromosome": r.chromosome,
            "position": r.position,
            "effect_allele": r.effect_allele,
            "other_allele": r.other_allele,
            "rsid": r.rsid,
        }
        for r in variant_axis.all()
    }


def _is_completed(store: OpenGWASDBStore) -> bool:
    """Whether a release is Reference-Completed, and so must say per cell which are imputed."""
    return store.manifest.completion_state is CompletionState.REFERENCE_COMPLETED


class _StoreQueryBase:
    """What every adapter answers the same way: its tables, result resolution,
    and the context manager. `analyses_table()` returns every `analyses.tsv`
    column keyed by `analysis_index` (ADR-0034's unified schema every layout
    shares); a Hybrid reads the *shared* table (see `HybridStoreQuery`)."""

    _analyses: AnalysesIndex
    _variant_axis: VariantAxis

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def variants_table(self) -> dict[int, dict]:
        """Return all variants keyed by variant_index."""
        return _variants_table(self._variant_axis)

    def analyses_table(self) -> dict[int, dict]:
        """Return all analyses keyed by analysis_index -- every analyses.tsv column."""
        return self._analyses.all()

    def resolve(
        self, result: Mapping[str, np.ndarray], *, include_variant_info: bool = False
    ) -> Iterator[dict[str, object]]:
        """Resolve a raw (index-keyed) result to human-readable rows (issue #104);
        see `opengwasdb.query.resolve.resolve_rows`."""
        return resolve_rows(
            self._analyses, self._variant_axis, result, include_variant_info=include_variant_info
        )


class StoreQuery(_StoreQueryBase):
    """Public query object that hides the physical store layout — Dense stores."""

    def __init__(self, store: OpenGWASDBStore):
        self.store = store
        self._connection = store.index_connection()
        self._analyses = AnalysesIndex(store.path)
        self._root = store.arrays(mode="r")
        self._variant_axis = VariantAxis(store.path, self._connection)
        # Every association read goes through the grid reader: the store's
        # declared encoding (ADR 0037) and the presence rule (ADR 0070) are
        # applied there, once, rather than at each result site.
        self._grid = DenseGridReader.open(
            self._root,
            store.manifest.encoding,
            source=f"Dense Store Release {store.path}",
            requires_mask=_is_completed(store),
        )
        self._rho_reader: DenseRhoReader | None = (
            DenseRhoReader(self._root["rho"], self._grid.n_analyses)
            if "rho" in self._root
            else None
        )
        self._top_hits = TopHitTiers(self._root)

    def close(self) -> None:
        self._variant_axis.close()
        self._connection.close()


    def _analysis_index(self, analysis_id: str) -> int | None:
        analysis = self._analyses.by_id(analysis_id)
        return None if analysis is None else int(analysis["analysis_index"])

    def analysis(
        self,
        analysis_id: str,
        *,
        observed_only: bool = False,
        # Dense may carry one; decode it instead of the primary planes (#265).
        indexed_subset: str | None = None,
    ) -> Associations:
        """Return every present association of one analysis.

        With `indexed_subset`, read the Analysis's full statistics from the
        named Indexed Variant Subset (ADR 0061, issue #265) instead of the
        primary planes. The result is identical to filtering the ordinary
        result to the subset's Variant Indices; an unknown, incomplete, stale or
        invalid subset raises rather than falling back to the slow path.
        """
        if indexed_subset is not None:
            return self._indexed_analysis(
                analysis_id, indexed_subset, observed_only=observed_only
            )
        col = self._analysis_index(analysis_id)
        if col is None:
            return Associations.empty()
        return self._grid.read(Cells.analysis(col), observed_only=observed_only)

    def _indexed_analysis(
        self, analysis_id: str, subset_name: str, *, observed_only: bool
    ) -> Associations:
        """One Analysis's full statistics over a named Indexed Variant Subset.

        Precedence: the named subset is resolved and structurally validated
        **before** the Analysis ID is looked up, so a typo'd, stale or corrupt
        selector always fails even when the Analysis is also unknown -- the
        caller explicitly asked for an indexed answer, and an unknown ID does
        not excuse the selector.  A *valid* subset with an unknown Analysis ID
        keeps the ordinary empty-result semantics.
        """
        subset = SubsetReader.open(self.store.path, subset_name)
        col = self._analysis_index(analysis_id)
        if col is None:
            return Associations.empty()
        return subset.read(Cells.analysis(col), observed_only=observed_only)

    def phewas(self, identifier: str, *, observed_only: bool = False) -> Associations:
        """Return one variant across all analyses."""
        variant = self._variant_axis.by_identifier(identifier)
        if variant is None:
            return Associations.empty()
        return self._grid.read(
            Cells.variants([variant.variant_index]), observed_only=observed_only
        )

    def range_phewas(
        self, chromosome: str, start: int, end: int, *, observed_only: bool = False
    ) -> Associations:
        """Return present associations for all variants in a genomic range (regional PheWAS)."""
        rows = self._variant_axis.range_indices(chromosome, start, end)
        if len(rows) == 0:
            return Associations.empty()
        return self._grid.read(Cells.variants(rows), observed_only=observed_only)

    def lookup(
        self, identifiers: list[str], analysis_ids: list[str], *, observed_only: bool = False
    ) -> Associations:
        """Return present associations for a specific variant × analysis set, Analysis-major."""
        # Row-index resolution only -- no VariantRecord materialisation, no
        # per-identifier variants.tsv.gz open (issue #3).
        rows = self._variant_axis.indices_by_identifiers(identifiers)
        cols = [i for aid in analysis_ids if (i := self._analysis_index(aid)) is not None]
        if not len(rows) or not cols:
            return Associations.empty()
        # Surgical orthogonal read: only the chunks intersecting the requested
        # rows × cols, not the full analysis width per row (issue 052).
        return self._grid.read(Cells.cross(rows, cols), observed_only=observed_only)

    def top_hits(
        self, *, analysis_id: str | None = None, threshold: float = 5e-8,
        limit: int | None = None, observed_only: bool = False,
    ) -> Associations:
        """Return genomic-order top hits, optionally for one analysis."""
        tier = self._top_hits.tier(threshold)
        if tier is None:
            return Associations.empty()
        analysis_index: int | None = None
        if analysis_id is not None:
            analysis_index = self._analysis_index(analysis_id)
            if analysis_index is None or not tier.has_offsets:
                return Associations.empty()
        record = self._top_hit_record(tier.read(analysis_index))
        return record.observed_only(observed_only).head(limit)

    def _top_hit_record(self, rows: TierRows) -> Associations:
        """A tier's rows as a record, each field read once.

        A current index carries `se`, `eaf` and, on a release with an imputed
        mask, `imputed`, and the grid is not touched. An older one lacks some,
        and the grid answers them at the hits' own cells in one read, shared
        between SE decoding and the `eaf` column (#253).
        """
        se, eaf, imputed = rows.se, rows.eaf, rows.imputed
        if se is None or eaf is None or (imputed is None and self._grid.has_imputed):
            cells = self._grid.inspect(Cells.pairs(rows.variant_index, rows.analysis_index))
            se = cells.se if se is None else se
            eaf = cells.eaf if eaf is None else eaf
            imputed = cells.flags if imputed is None else imputed
        return Associations.from_decoded(
            rows.variant_index,
            rows.analysis_index,
            rows.z,
            se,
            eaf,
            imputed,
            source=f"Top-Hit Index of {self.store.path}",
        )

    def rho(self, *ids: str) -> dict[str, np.ndarray]:
        """Long-format pairwise Rho for a set of Analysis IDs (positional, or a
        single iterable of IDs); self-pairs excluded. Empty when the store has
        no Rho Matrix (opt-in, ADR 0025) or no ID resolves."""
        if len(ids) == 1 and not isinstance(ids[0], str):
            ids = tuple(ids[0])
        if self._rho_reader is None:
            return _empty_rho_result()
        resolved = [
            (aid, int(a["analysis_index"]))
            for aid in ids
            if (a := self._analyses.by_id(aid)) is not None
        ]
        out_a: list[str] = []
        out_b: list[str] = []
        out_rho: list[float] = []
        out_n: list[int] = []
        for x in range(len(resolved)):
            aid_a, idx_a = resolved[x]
            for y in range(x + 1, len(resolved)):
                aid_b, idx_b = resolved[y]
                if idx_a == idx_b:
                    continue
                r, n = self._rho_reader.pair(idx_a, idx_b)
                out_a.append(aid_a)
                out_b.append(aid_b)
                out_rho.append(r)
                out_n.append(n)
        return {
            "analysis_id_a": np.array(out_a, dtype=object),
            "analysis_id_b": np.array(out_b, dtype=object),
            "rho": np.array(out_rho, dtype="float32"),
            "n_null": np.array(out_n, dtype="int64"),
        }

    def rho_row(self, analysis_id: str) -> dict[str, np.ndarray]:
        """One Analysis's Rho and support against every other Analysis."""
        analysis = self._analyses.by_id(analysis_id)
        if self._rho_reader is None or analysis is None:
            return _empty_rho_row_result()
        idx = int(analysis["analysis_index"])
        rho_vals, n_vals = self._rho_reader.row(idx)
        id_by_index = self._analyses.all()
        others = [i for i in range(self._rho_reader.n_analyses) if i != idx]
        return {
            "analysis_id": np.array([id_by_index[i]["analysis_id"] for i in others], dtype=object),
            "rho": rho_vals[others].astype("float32"),
            "n_null": n_vals[others].astype("int64"),
        }

    def rho_matrix(self, ids: list[str] | None = None) -> dict[str, np.ndarray]:
        """Wide-format Rho: the full symmetric matrix (diagonal 1.0), or the
        dense submatrix for a given vector of Analysis IDs, in that order."""
        if self._rho_reader is None:
            return _empty_rho_matrix_result()
        if ids is None:
            id_by_index = self._analyses.all()
            ordered_ids = [
                id_by_index[i]["analysis_id"] for i in range(self._rho_reader.n_analyses)
            ]
            rho_mat, n_mat = self._rho_reader.matrix(None)
        else:
            resolved = [
                (aid, int(a["analysis_index"]))
                for aid in ids
                if (a := self._analyses.by_id(aid)) is not None
            ]
            ordered_ids = [aid for aid, _ in resolved]
            rho_mat, n_mat = self._rho_reader.matrix([idx for _, idx in resolved])
        return {
            "analysis_id": np.array(ordered_ids, dtype=object),
            "rho": rho_mat.astype("float32"),
            "n_null": n_mat.astype("int64"),
        }


def _analysis_major(found: Associations, requested: list[int]) -> Associations:
    """`found`'s rows grouped by Analysis, in the order `requested` first names each.

    Within an Analysis the rows keep their order, so a Hybrid lookup lists the
    Dense Component's rows before the Overflow's -- each component's own order --
    and every layout's `lookup()` is Analysis-major (ADR 0070).
    """
    if not found.count:
        return found
    order = list(dict.fromkeys(requested))
    rank = np.full(max(order, default=0) + 1, len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return found.take(np.argsort(rank[found.analysis_index], kind="stable"))


def _refuse_indexed_subset(store_path: Path, layout: str, subset_name: str) -> NoReturn:
    """Refuse a subset selector on a layout that cannot carry one (ADR 0061).

    A caller who explicitly names a subset must never be answered from the
    primary path (issue #265): silently ignoring the selector would look like a
    successful indexed query while reading the slow matrix underneath.
    """
    raise IndexedSubsetLayoutError(
        f"release {store_path} has {layout} layout; an Indexed Variant Subset "
        f"({subset_name!r}) is Dense-only and cannot be used here"
    )


class RaggedStoreQuery(_StoreQueryBase):
    """Public query object that hides the physical store layout — Ragged stores."""

    def __init__(self, store: OpenGWASDBStore):
        self.store = store
        self._variant_axis = VariantAxis(store.path)
        self._reader = CsrReader.open(
            store.path,
            store.manifest.encoding,
            source=f"Ragged Store Release {store.path}",
            requires_mask=_is_completed(store),
        )
        # Absence is an error: a release without the variant index is refused
        # with an instruction to build it (ADR 0060).
        self._reader.open_variant_index(self._variant_axis.n_variants)
        self._csr = self._reader.structure
        self._analyses = AnalysesIndex(store.path)
        self._top_hits = TopHitTiers(store.arrays(mode="r"))
        # Every result carries `n` exactly when the release has the plane, so a
        # caller can tell "no sample sizes here" from "no rows" (ADR 0067).
        self._has_n = self._reader.carries_n

    def _empty(self) -> Associations:
        return Associations.empty(with_n=self._has_n)

    def close(self) -> None:
        self._variant_axis.close()

    def _resolve_analysis_id(self, analysis_id: str) -> int | None:
        row = self._analyses.by_id(analysis_id)
        return None if row is None else int(row["analysis_index"])


    def analysis(
        self,
        analysis_id: str,
        *,
        observed_only: bool = False,
        # Ragged carries no Indexed Variant Subsets; a selector is refused (#265).
        indexed_subset: str | None = None,
    ) -> Associations:
        """All associations for one analysis (analysis_id lookup).

        Indexed Variant Subsets are Dense-only (ADR 0061), so a
        selector naming one is refused rather than ignored (#265).
        """
        if indexed_subset is not None:
            _refuse_indexed_subset(self.store.path, "Ragged", indexed_subset)
        idx = self._resolve_analysis_id(analysis_id)
        if idx is None:
            return self._empty()
        return self._reader.read(Cells.analysis(idx), observed_only=observed_only)

    def range_phewas(
        self, chromosome: str, start: int, end: int, *, observed_only: bool = False
    ) -> Associations:
        """All associations where the variant falls in [start, end] (regional PheWAS).

        Read from the by-variant index (ADR 0060): the block spanning the
        range's variants, filtered to them when they are not contiguous.
        """
        variant_indices = self._variant_axis.range_indices(chromosome, start, end)
        if len(variant_indices) == 0:
            return self._empty()
        return self._reader.read(Cells.variants(variant_indices), observed_only=observed_only)

    def _analysis_indices_in_range(self, chromosome: str, start: int, end: int) -> list[int]:
        """Analysis indices whose Trait position falls in [start, end] --
        analyses.tsv's own trait_chr/trait_bp columns are the sole source of
        truth for this (ADR-0034, issue #69); this scans the already
        store-open-time-loaded AnalysesIndex rather than a second,
        independently-shaped tabix-indexed position file."""
        matches = []
        for index, row in self._analyses.items():
            bp = row.get("trait_bp") or ""
            if row.get("trait_chr") == chromosome and bp and start <= int(bp) <= end:
                matches.append(index)
        return matches

    def range_by_analysis(
        self, chromosome: str, start: int, end: int, *, observed_only: bool = False
    ) -> Associations:
        """All associations for analyses whose probe/TSS falls in [start, end]."""
        analysis_indices = self._analysis_indices_in_range(chromosome, start, end)
        if not analysis_indices:
            return self._empty()
        return self._reader.read(Cells.analyses(analysis_indices), observed_only=observed_only)

    def phewas(self, identifier: str, *, observed_only: bool = False) -> Associations:
        """All analyses that have an association for a given variant identifier."""
        variant = self._variant_axis.by_identifier(identifier)
        if variant is None:
            return self._empty()
        return self._reader.read(
            Cells.variants([variant.variant_index]), observed_only=observed_only
        )

    def top_hits(
        self, *, analysis_id: str | None = None, threshold: float = 5e-8,
        limit: int | None = None, observed_only: bool = False,
    ) -> Associations:
        """Associations passing a significance threshold, in genomic order
        (analysis_index, then variant_index) -- the "analysis_index,
        variant_index" order the top-hit index itself is built in (see
        ``group.attrs["order"]`` in ``layouts/*/top_hits.py``). Both the
        indexed fast path and the full-scan fallback apply observed_only
        before limit, so ``limit`` caps the returned (post-filter) rows.

        Uses the precomputed top-hit index when available (fast path);
        falls back to a full CSR scan otherwise. The two paths return the
        same shape of answer for the same call.
        """
        tier = self._top_hits.tier(threshold)
        if tier is None:
            return self._top_hits_by_scan(
                analysis_id=analysis_id,
                threshold=threshold,
                limit=limit,
                observed_only=observed_only,
            )
        analysis_index = None
        if analysis_id is not None:
            analysis_index = self._resolve_analysis_id(analysis_id)
            if analysis_index is None or not tier.has_offsets:
                return self._empty()
        record = _ragged_tier_record(
            tier,
            analysis_index,
            self._csr,
            with_n=self._has_n,
            source=f"Top-Hit Index of {self.store.path}",
        )
        return record.observed_only(observed_only).head(limit)

    def _scan_threshold_hits(self, region: Cells | None, threshold: float) -> np.ndarray:
        """Flat CSR positions of `region`'s rows -- every row, with None -- clearing
        `threshold`.

        The scan reads Z a window at a time and keeps only what clears the cutoff,
        so peak memory is a window, not the association count (#252): at
        OGS-00011's 3,085,080,783-cell Overflow a whole-array decode held 12.3 GB
        of `variant_index` and 3.1 GB of mask at once. The cutoff is
        `z_critical`, the one the index is built with, so a scan and an index
        agree on the boundary. The reader refuses a non-canonical marker the
        cutoff would otherwise drop.
        """
        cutoff = top_hits.z_critical(threshold)
        hits: list[np.ndarray] = []
        for window in self._reader.windows(region):
            assert window.pos is not None
            mask = np.abs(self._reader.z(window)) >= cutoff
            if mask.any():
                hits.append(window.pos[mask])
        if not hits:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(hits)

    def _top_hits_by_scan(
        self,
        *,
        analysis_id: str | None,
        threshold: float,
        limit: int | None,
        observed_only: bool,
    ) -> Associations:
        """Top hits with no index to read: scan the CSR and threshold it.

        Returns the same shape of answer as the indexed path, applying
        `observed_only` before `limit` exactly as that path does. `analysis_id`
        is resolved here too, so a caller passing one gets a filtered result on
        both paths rather than an unfiltered store-wide scan on this one. The
        hits are read at their flat positions, which ascend in genomic order.
        """
        region = None
        if analysis_id is not None:
            analysis_index = self._resolve_analysis_id(analysis_id)
            if analysis_index is None:
                return self._empty()
            region = Cells.analysis(analysis_index)
        hit_positions = self._scan_threshold_hits(region, threshold)
        if len(hit_positions) == 0:
            return self._empty()
        found = self._reader.read(Cells.positions(hit_positions), observed_only=observed_only)
        return found.head(limit)

    def lookup(
        self, identifiers: list[str], analysis_ids: list[str], *, observed_only: bool = False
    ) -> Associations:
        """Associations for a specific variant × analysis set, Analysis-major.

        Each requested Analysis's sorted segment is binary-searched for the
        wanted variants (O(log) chunk reads), rather than the Analysis being
        decoded whole and `np.isin`-ed. The cost is the requested variants and
        Analyses, not the store and not the requested Analyses' sizes (#252).
        Requested Analysis order and duplicates carry through; a request that
        resolves nothing yields the empty result.
        """
        variants = [
            v for id_ in identifiers if (v := self._variant_axis.by_identifier(id_)) is not None
        ]
        if not variants:
            return self._empty()
        wanted = np.array(sorted({v.variant_index for v in variants}), dtype=np.int32)
        analyses = [
            idx for aid in analysis_ids if (idx := self._resolve_analysis_id(aid)) is not None
        ]
        return self._reader.read(Cells.cross(wanted, analyses), observed_only=observed_only)


class HybridStoreQuery(_StoreQueryBase):
    """Query facade for a Hybrid store (ADR 0026).

    A thin integration layer: it dispatches to the nested Dense Component's
    ``StoreQuery`` (on-panel variants) and to the Ragged Overflow reader
    (off-panel variants), remaps the Dense Component's panel-local
    ``variant_index`` onto the shared variant index space, and concatenates. A
    variant is in exactly one component (on-panel xor off-panel), so results are
    a plain union with no dedup.
    """

    def __init__(self, store: OpenGWASDBStore):
        self.store = store
        self._dense_store = open_store(dense_component_path(store.path))
        self._dense = StoreQuery(self._dense_store)
        self._dense_to_shared = np.load(dense_to_shared_path(store.path)).astype("int32")
        # The Overflow at store/data.zarr/ragged.
        # The Overflow is observed by construction (ADR 0026): it never carries an
        # imputed mask, even in a Reference-Completed release, so it reads without one.
        self._overflow = CsrReader.open(
            store.path,
            store.manifest.encoding,
            source=f"Ragged Overflow of {store.path}",
            requires_mask=False,
        )
        self._csr = self._overflow.structure
        self._connection = store.index_connection()
        self._analyses = AnalysesIndex(store.path)  # shared analyses.tsv
        self._variant_axis = VariantAxis(store.path, self._connection)  # shared union table
        self._overflow.open_variant_index(self._variant_axis.n_variants)
        self._top_hits = TopHitTiers(store.arrays(mode="r"))  # overflow's own tiers

    def close(self) -> None:
        self._dense.close()
        self._variant_axis.close()
        self._connection.close()

    # ── shared tables ────────────────────────────────────────────────────────
    # `analyses_table()`, `variants_table()` and `resolve()` are the base
    # class's, over the *shared* tables this facade opened -- deliberately not
    # the Dense Component's. Hybrid builds write `analyses.tsv` twice: once
    # under `dense/` counting only that component's on-panel top hits, once at
    # the shared root where `add_hit_counts()` has additionally counted the
    # Ragged Overflow Component's, so the Dense Component's own copy undercounts
    # `n_hits_*` for any Analysis with off-panel hits (issue #107). The
    # Analytical Metadata columns are identical in both; only the counts
    # differ, which is exactly what makes the wrong one easy to miss. Results
    # resolve against the shared variant table, the `variant_index` space they
    # are remapped into.

    # ── Rho Matrix (ADR 0025, Dense-only artifact) ───────────────────────────
    # Delegated to the Dense Component: Rho is opt-in, built against a Dense
    # store's own variant axis. A Hybrid release's Dense Component is a
    # self-contained Dense Store Release, so if Rho was built against it these
    # just work; otherwise they return the same empty result StoreQuery
    # returns for a Dense store with no Rho Matrix.
    def rho(self, *ids: str) -> dict[str, np.ndarray]:
        return self._dense.rho(*ids)

    def rho_row(self, analysis_id: str) -> dict[str, np.ndarray]:
        return self._dense.rho_row(analysis_id)

    def rho_matrix(self, ids: list[str] | None = None) -> dict[str, np.ndarray]:
        return self._dense.rho_matrix(ids)

    # ── dispatch helpers ─────────────────────────────────────────────────────
    def _remap_dense(self, result: Associations) -> Associations:
        """A Dense Component result with its panel-local variant_index made shared."""
        return result.remap_variants(self._dense_to_shared)

    def _on_panel_mask(self, shared_indices: np.ndarray) -> np.ndarray:
        """Which shared variant indices are rows of the Dense Component's panel.

        One vectorised `searchsorted` over the panel map, with the needles cast
        to the map's own dtype. The per-variant Python form cast the whole
        13.4 M-entry `int32` map on every call (22.6 ms against 0.004 ms for an
        `int32` scalar on OGS-00011), and `range_phewas` made one call per
        shared variant -- 58,006 of them for a 1 Mb TCF7L2 window (#252).
        Searching that map with `int64` needles still promoted and copied it on
        every call (review round 1), so the needles are narrowed instead; a
        needle outside the map's range cannot be a panel row and is masked out
        before the cast, so narrowing cannot wrap it into range.
        """
        shared_indices = np.asarray(shared_indices)
        if shared_indices.size == 0:
            return np.zeros(shared_indices.shape, dtype=bool)
        panel = self._dense_to_shared
        if panel.size == 0:
            return np.zeros(shared_indices.shape, dtype=bool)
        low, high = int(panel[0]), int(panel[-1])
        result = np.zeros(shared_indices.shape, dtype=bool)
        in_range = (shared_indices >= low) & (shared_indices <= high)
        if in_range.any():
            needles = shared_indices[in_range].astype(panel.dtype, copy=False)
            pos = np.searchsorted(panel, needles)
            safe = np.minimum(pos, panel.size - 1)
            result[in_range] = panel[safe] == needles
        return result

    def _shared_is_on_panel(self, shared_idx: int) -> bool:
        return bool(self._on_panel_mask(np.array([shared_idx], dtype=np.int64))[0])

    def _overflow_read(self, cells: Cells, *, observed_only: bool) -> Associations:
        """The Overflow's rows for a region, without the `n` a Hybrid never reports."""
        return self._overflow.read(cells, observed_only=observed_only)

    # ── public query surface ─────────────────────────────────────────────────
    def analysis(
        self,
        analysis_id: str,
        *,
        observed_only: bool = False,
        # Delegated to the nested Dense Component's index; never the overflow (#271).
        indexed_subset: str | None = None,
    ) -> Associations:
        """All associations for one analysis (analysis_id lookup) across both components.

        With a selector, the answer comes from the nested Dense Component's
        Indexed Variant Subset and covers its on-panel variants only: the index
        is built against that self-contained Dense release (issue #271), its
        panel-local `variant_index` is remapped through `dense_to_shared.npy`,
        and the Ragged Overflow is never read or concatenated. An unknown,
        stale, incomplete, corrupt or unsupported selector raises rather than
        falling back to either ordinary path, and the failure names the Hybrid
        root the caller passed -- not the internal `<root>/dense` -- while
        preserving the exception subclass.
        """
        if indexed_subset is not None:
            try:
                return self._remap_dense(
                    self._dense.analysis(
                        analysis_id, observed_only=observed_only, indexed_subset=indexed_subset
                    )
                )
            except IndexedSubsetError as exc:
                # The nested Dense facade scopes the failure to `<root>/dense`
                # because that is the Store it opened; re-point it at the
                # Hybrid root the operator named, replacing rather than
                # prepending the prefix (issue #271 review).
                raise rescope_indexed_subset_error(
                    self.store.path, self._dense.store.path, exc
                ) from exc
        dense = self._remap_dense(self._dense.analysis(analysis_id, observed_only=observed_only))
        analysis = self._analyses.by_id(analysis_id)
        if analysis is None:
            return dense
        overflow = self._overflow_read(
            Cells.analysis(int(analysis["analysis_index"])), observed_only=observed_only
        )
        return Associations.concat([dense, overflow])

    def phewas(self, identifier: str, *, observed_only: bool = False) -> Associations:
        variant = self._variant_axis.by_identifier(identifier)
        if variant is None:
            return Associations.empty()
        if self._shared_is_on_panel(variant.variant_index):
            return self._remap_dense(self._dense.phewas(variant.alid, observed_only=observed_only))
        return self._overflow_read(
            Cells.variants([variant.variant_index]), observed_only=observed_only
        )

    def range_phewas(
        self, chromosome: str, start: int, end: int, *, observed_only: bool = False
    ) -> Associations:
        dense = self._remap_dense(
            self._dense.range_phewas(chromosome, start, end, observed_only=observed_only)
        )
        shared_idx = np.asarray(self._variant_axis.range_indices(chromosome, start, end))
        off_panel = shared_idx[~self._on_panel_mask(shared_idx)]
        if not len(off_panel):
            return dense
        overflow = self._overflow_read(Cells.variants(off_panel), observed_only=observed_only)
        return Associations.concat([dense, overflow])

    def lookup(
        self, identifiers: list[str], analysis_ids: list[str], *, observed_only: bool = False
    ) -> Associations:
        dense = self._remap_dense(
            self._dense.lookup(identifiers, analysis_ids, observed_only=observed_only)
        )
        off_panel = self._off_panel_indices(identifiers)
        if not off_panel:
            return dense
        # Each requested Analysis's sorted segment is binary-searched for the
        # wanted variants, in ascending Analysis order (#252).
        columns = sorted(
            {i for aid in analysis_ids if (i := self._analysis_index(aid)) is not None}
        )
        overflow = self._overflow_read(
            Cells.cross(off_panel, columns), observed_only=observed_only
        )
        requested = [i for aid in analysis_ids if (i := self._analysis_index(aid)) is not None]
        return _analysis_major(Associations.concat([dense, overflow]), requested)

    def _off_panel_indices(self, identifiers: list[str]) -> list[int]:
        """The shared indices of the identifiers that resolve off the panel, ascending."""
        records = [
            rec for id_ in identifiers if (rec := self._variant_axis.by_identifier(id_)) is not None
        ]
        if not records:
            return []
        shared = np.array([r.variant_index for r in records], dtype=np.int64)
        return sorted({int(i) for i in shared[~self._on_panel_mask(shared)]})

    def _analysis_index(self, analysis_id: str) -> int | None:
        analysis = self._analyses.by_id(analysis_id)
        return None if analysis is None else int(analysis["analysis_index"])

    def _overflow_top_hits(
        self, threshold: float, analysis_index: int | None = None
    ) -> Associations:
        tier = self._top_hits.tier(threshold)
        if tier is None:
            return Associations.empty()
        if analysis_index is not None and not tier.has_offsets:
            return Associations.empty()
        return _ragged_tier_record(
            tier,
            analysis_index,
            self._csr,
            with_n=False,
            source=f"Ragged Overflow Top-Hit Index of {self.store.path}",
        )

    def top_hits(
        self, *, analysis_id: str | None = None, threshold: float = 5e-8,
        limit: int | None = None, observed_only: bool = False,
    ) -> Associations:
        analysis_index = None
        if analysis_id is not None:
            analysis = self._analyses.by_id(analysis_id)
            if analysis is None:
                return Associations.empty()
            analysis_index = int(analysis["analysis_index"])
        dense = self._remap_dense(
            self._dense.top_hits(
                analysis_id=analysis_id, threshold=threshold, observed_only=observed_only
            )
        )
        overflow = self._overflow_top_hits(threshold, analysis_index)
        merged = Associations.concat([dense, overflow.observed_only(observed_only)])
        return merged.in_genomic_order().head(limit)


def query_store(path: str | Path) -> StoreQuery | RaggedStoreQuery | HybridStoreQuery:
    """Open a store and return the layout-independent query facade."""
    store = open_store(path)
    if store.manifest.primary_layout is PrimaryStorageLayout.RAGGED:
        return RaggedStoreQuery(store)
    if store.manifest.primary_layout is PrimaryStorageLayout.HYBRID:
        return HybridStoreQuery(store)
    return StoreQuery(store)
