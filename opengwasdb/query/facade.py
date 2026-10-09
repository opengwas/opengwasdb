"""Layout-independent query facade (ADR-0006, ADR-0020, ADR-0033).

Three adapter classes share one result contract but not one method set --
`query_store()` / `OpenGWASDBStore.query()` dispatch to the right one from
the store manifest: `StoreQuery` (Dense), `RaggedStoreQuery` (Ragged),
`HybridStoreQuery` (Hybrid). See ADR-0033 for the full rationale; in short:

Result shape -- every association-returning method returns
``{"variant_index", "analysis_index", "z", "se", "eaf", "association_status"}`` as
parallel arrays (int32, int32, float32, float32, object; ADR-0020).

Ordering -- `analysis()`, `phewas()`, `range_phewas()`, `range_by_analysis()`,
and `lookup()` make no ordering guarantee beyond grouping (rows for one scan
target are contiguous, not sorted). `top_hits()` returns genomic order --
sorted by `(analysis_index, variant_index)` -- on every adapter and every
internal path (indexed and full-scan fallback alike), matching the
`group.attrs["order"]` the top-hit index itself is built with.

observed_only / limit -- both apply as filter-then-limit everywhere they are
accepted, on every internal path: `observed_only` narrows the result set
first, `limit` then caps the already-filtered rows.

Finiteness vs "missing" (point-query methods only -- `analysis()`,
`phewas()`, `range_phewas()`, `lookup()`; `top_hits()` is a separate case,
below) -- `StoreQuery` only ever returns finite `(z, se)` cells; a
non-finite Dense grid cell (untested, or an attempted-but-failed completion
-- ADR-0013, ADR-0022) is silently absent from the result rather than
returned with `association_status="missing"`. This keeps point queries
against a mostly-empty Dense grid sparse (ADR-0020). `RaggedStoreQuery`
never filters for finiteness: a CSR entry only exists for a variant x
analysis pair someone attempted (observed, or a completion attempt), so a
non-finite entry is already a small, deliberate set, and is returned with
`association_status="missing"` via `_status_array`. `HybridStoreQuery` is
split by construction, not a third uniform behaviour: on-panel results are
delegated to its Dense Component (`StoreQuery`) and inherit Dense's
drop-non-finite behaviour; off-panel (Ragged Overflow) results are read the
same unfiltered way as `RaggedStoreQuery`, though the Overflow Component is
documented as always-observed (ADR-0026), so a non-finite overflow cell
would be an anomaly rather than an expected outcome.

`top_hits()` sits outside the point-query finiteness contract above, on
every adapter: candidacy is decided at build time by
`|z| >= z_critical(threshold)` (`layouts/*/top_hits.py`), which excludes NaN
`z` (a NaN comparison is always false) but does not itself guarantee a
finite paired `se` -- no separate `isfinite(se)` filter is applied at query
time.

Method availability -- `variants_table()`/`analyses_table()` and
`__enter__`/`__exit__` are present on all three adapters.
`analyses_table()` returns the same shape on every adapter -- every column
of `analyses.tsv` (ADR-0034's unified schema every layout shares), keyed by
`analysis_index`. Ragged rows populate the molecular-QTL columns (tissue,
context, and gene identity carried via `analysis_label`/`trait_ontology_id`,
ADR 0035) that Dense/Hybrid rows mostly leave blank, and leave Dense/Hybrid's
other Trait-identity/effect-scale columns mostly blank in turn; a caller
grouping Ragged Analyses by a shared gene filters this table on
`trait_ontology_id` rather than through a separate lookup. `rho()`/
`rho_row()`/`rho_matrix()` (ADR-0025, a Dense storage artifact) are exposed
on `StoreQuery` and on `HybridStoreQuery` (delegated to its Dense Component);
`RaggedStoreQuery` has no Rho Matrix format. `range_by_analysis()` (query by
probe/TSS position) is Ragged-only: it scans `AnalysesIndex`'s already
store-open-time-loaded rows for `trait_chr`/`trait_bp`, which only
Ragged/molecular-QTL releases populate.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import zarr

from opengwasdb.encoding import DenseEafPlane, DenseSePlane, DenseZPlane, EafRead
from opengwasdb.index import AnalysesIndex
from opengwasdb.layouts.dense.rho import DenseRhoReader
from opengwasdb.layouts.dense.top_hits import DenseTopHitReader, TopHitTiers, z_critical
from opengwasdb.layouts.hybrid.layout import dense_component_path, dense_to_shared_path
from opengwasdb.layouts.ragged.by_variant import ByVariantReader, has_variant_index
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.model.enums import CompletionState, PrimaryStorageLayout
from opengwasdb.query.resolve import resolve_rows
from opengwasdb.store.open import OpenGWASDBStore, open_store
from opengwasdb.variants import VariantAxis


def _empty_result() -> dict[str, np.ndarray]:
    return {
        "variant_index": np.empty(0, dtype="int32"),
        "analysis_index": np.empty(0, dtype="int32"),
        "z": np.empty(0, dtype="float32"),
        "se": np.empty(0, dtype="float32"),
        "eaf": np.empty(0, dtype="float32"),
        "association_status": np.empty(0, dtype=object),
    }


def _status_array(imputed_flags: np.ndarray, z_vals: np.ndarray, se_vals: np.ndarray) -> np.ndarray:
    """Derive association_status strings from imputed mask, z, and se (ADR-0013:
    finite Z and SE means observed/imputed; NaN Z *or* SE means missing)."""
    out = np.where(imputed_flags == 1, "imputed", "observed").astype(object)
    out[~(np.isfinite(z_vals) & np.isfinite(se_vals))] = "missing"
    return out


def _top_hits_result(
    variant_index: np.ndarray,
    analysis_index: np.ndarray,
    z: np.ndarray,
    se: np.ndarray,
    eaf: np.ndarray,
    imputed: np.ndarray,
    *,
    observed_only: bool,
    limit: int | None,
) -> dict[str, np.ndarray]:
    """Shared indexed top-hit finalizer: filter, then cap, then assemble.

    The Dense and Ragged indexed paths resolve their parallel arrays their own
    way, then both finish identically (ADR 0033): `observed_only` runs before
    `limit` so an imputed leading hit can never crowd an observed one out of
    the cap, and the six parallel arrays are filtered and sliced together so a
    misaligned array can never slip into the returned result.
    """
    if observed_only:
        keep = imputed == 0
        variant_index = variant_index[keep]
        analysis_index = analysis_index[keep]
        z = z[keep]
        se = se[keep]
        eaf = eaf[keep]
        imputed = imputed[keep]
    if limit is not None:
        variant_index = variant_index[:limit]
        analysis_index = analysis_index[:limit]
        z = z[:limit]
        se = se[:limit]
        eaf = eaf[:limit]
        imputed = imputed[:limit]
    return {
        "variant_index": variant_index,
        "analysis_index": analysis_index,
        "z": z,
        "se": se,
        "eaf": eaf,
        "association_status": _status_array(imputed, z, se),
    }


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


def _open_by_variant(store: OpenGWASDBStore, variant_axis: VariantAxis) -> ByVariantReader | None:
    """The component's variant index reader, or `None` when it carries none (ADR 0060).

    Absence is not an error: every variant-side query falls back to the scan.
    """
    if not has_variant_index(store.path):
        return None
    return ByVariantReader(
        store.path, store.manifest.encoding, n_axis=variant_axis.n_variants
    )


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


class StoreQuery:
    """Public query object that hides the physical store layout — Dense stores."""

    def __init__(self, store: OpenGWASDBStore):
        self.store = store
        self._connection = store.index_connection()
        self._analyses = AnalysesIndex(store.path)
        self._root = store.arrays(mode="r")
        self._variant_axis = VariantAxis(store.path, self._connection)
        self._is_completed = store.manifest.completion_state is CompletionState.REFERENCE_COMPLETED
        self._imputed: zarr.Array | None = (
            self._root["imputed"] if self._is_completed and "imputed" in self._root else None
        )
        # Every z and eaf read goes through its plane: the store's declared
        # encoding (ADR 0037) is applied in one place rather than at each
        # result site, and the eaf plane gathers the per-variant baseline and
        # the reference frequency for imputed cells with it.
        self._z = DenseZPlane.open(self._root, store.manifest.encoding)
        # Open the frequency plane lazily: a current top-hit index carries its
        # decoded EAF values and must not touch the much larger source plane.
        self.__eaf: DenseEafPlane | None = None
        self._encoding = store.manifest.encoding
        self._se = DenseSePlane.open(self._root, self._encoding)
        self._rho_reader: DenseRhoReader | None = (
            DenseRhoReader(self._root["rho"], self._z.n_analyses) if "rho" in self._root else None
        )
        self._top_hits = TopHitTiers(self._root)

    @property
    def _eaf(self) -> DenseEafPlane:
        if self.__eaf is None:
            self.__eaf = DenseEafPlane.open(self._root, self._encoding)
        return self.__eaf

    def _imputed_pairs(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Imputed flags for elementwise (row, col) pairs; all-zeros when not a completed store."""
        if self._imputed is None or len(rows) == 0:
            return np.zeros(len(rows), dtype=np.uint8)
        return self._imputed.vindex[rows, cols].astype(np.uint8)

    def _eaf_pairs(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Decoded EAF for elementwise (row, col) pairs (ADR 0036, ADR 0037).

        All-NaN when the store declares no `eaf` plane, and per cell when this
        Analysis reported no frequency there. Per-cell NaN already means "no
        EAF here", so an absent plane and an absent cell read the same way and
        callers need no separate has-EAF check. On a Reference-Completed
        release carrying reference EAF, an imputed cell reads the panel's
        frequency and an observed cell never does (ADR 0037 §4).
        """
        return self._eaf.points(rows, cols)

    def _shared_cell_result(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        z_vals: np.ndarray,
        se_vals: np.ndarray,
        eaf_read: EafRead,
        mask: np.ndarray,
        *,
        observed_only: bool,
    ) -> dict[str, np.ndarray]:
        """The finite cells of a shared EAF read, and the finished result.

        The one place a shared region is cut. Indexing the decoded frequencies
        with the query's own finite mask is what #253's alignment trap is about:
        a mask that did not match the region would return a neighbouring cell's
        frequency, plausibly and silently. SE decoding was handed the same
        unmasked array, so both consumers agree cell for cell.
        """
        eaf_vals = np.asarray(eaf_read.values)[mask]
        if eaf_read.imputed is None:
            imputed = np.zeros(len(eaf_vals), dtype=np.uint8)
        else:
            imputed = np.asarray(eaf_read.imputed, dtype=np.uint8)[mask]
        return self._cell_result(
            rows,
            cols,
            z_vals,
            se_vals,
            observed_only=observed_only,
            eaf_vals=eaf_vals,
            imputed=imputed,
        )

    def close(self) -> None:
        self._variant_axis.close()
        self._connection.close()

    def __enter__(self) -> StoreQuery:
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
        self, result: dict[str, np.ndarray], *, include_variant_info: bool = False
    ) -> Iterator[dict[str, object]]:
        """Resolve a raw (index-keyed) result to human-readable rows (issue #104);
        see `opengwasdb.query.resolve.resolve_rows`."""
        return resolve_rows(
            self._analyses, self._variant_axis, result, include_variant_info=include_variant_info
        )

    def _cell_result(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        z_vals: np.ndarray,
        se_vals: np.ndarray,
        *,
        observed_only: bool,
        eaf_vals: np.ndarray | None = None,
        imputed: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """The six parallel arrays every index-keyed result carries.

        Assembling them in one place is what keeps the shapes of `analysis`,
        `phewas`, `range_phewas` and `lookup` identical: a caller cannot tell
        which one produced a result, and `observed_only` cannot filter five of
        the six arrays in one method and six in another.

        `eaf_vals` and `imputed` let a caller that already read those cells in
        bulk hand them over rather than gather the same cells a second time.
        """
        if imputed is None:
            imputed = self._imputed_pairs(rows, cols)
        if observed_only:
            keep = imputed == 0
            rows, cols, z_vals, se_vals, imputed = (
                rows[keep],
                cols[keep],
                z_vals[keep],
                se_vals[keep],
                imputed[keep],
            )
            if eaf_vals is not None:
                eaf_vals = eaf_vals[keep]
        if eaf_vals is None:
            eaf_vals = self._eaf_pairs(rows, cols)
        return {
            "variant_index": rows,
            "analysis_index": cols,
            "z": z_vals,
            "se": se_vals,
            "eaf": eaf_vals,
            "association_status": _status_array(imputed, z_vals, se_vals),
        }

    def analysis(self, analysis_id: str, *, observed_only: bool = False) -> dict[str, np.ndarray]:
        """Return all finite associations for one analysis."""
        analysis = self._analyses.by_id(analysis_id)
        if analysis is None:
            return _empty_result()
        col = int(analysis["analysis_index"])
        z_col = self._z.column(col)
        eaf_read = self._eaf.read_column(col, want_imputed=True)
        se_col = self._se.column(col, eaf=eaf_read.values)
        mask = np.isfinite(z_col) & np.isfinite(se_col)
        rows = np.where(mask)[0].astype("int32")
        cols = np.full(len(rows), col, dtype="int32")
        return self._shared_cell_result(
            rows, cols, z_col[mask], se_col[mask], eaf_read, mask, observed_only=observed_only
        )

    def phewas(self, identifier: str, *, observed_only: bool = False) -> dict[str, np.ndarray]:
        """Return one variant across all analyses."""
        variant = self._variant_axis.by_identifier(identifier)
        if variant is None:
            return _empty_result()
        row = variant.variant_index
        z_row = self._z.row(row)
        eaf_read = self._eaf.read_row(row, want_imputed=True)
        se_row = self._se.row(row, eaf=eaf_read.values)
        mask = np.isfinite(z_row) & np.isfinite(se_row)
        cols = np.where(mask)[0].astype("int32")
        rows = np.full(len(cols), row, dtype="int32")
        return self._shared_cell_result(
            rows, cols, z_row[mask], se_row[mask], eaf_read, mask, observed_only=observed_only
        )

    def range_phewas(
        self, chromosome: str, start: int, end: int, *, observed_only: bool = False
    ) -> dict[str, np.ndarray]:
        """Return finite associations for all variants in a genomic range (regional PheWAS)."""
        row_indices = self._variant_axis.range_indices(chromosome, start, end)
        if len(row_indices) == 0:
            return _empty_result()
        z_block = self._z.rows(row_indices)
        eaf_read = self._eaf.read_rows(row_indices, want_imputed=True)
        se_block = self._se.rows(row_indices, eaf=eaf_read.values)
        mask = np.isfinite(z_block) & np.isfinite(se_block)
        rows_rel, cols = np.where(mask)
        rows = row_indices[rows_rel].astype("int32")
        cols = cols.astype("int32")
        return self._shared_cell_result(
            rows, cols, z_block[mask], se_block[mask], eaf_read, mask, observed_only=observed_only
        )

    def lookup(
        self,
        identifiers: list[str],
        analysis_ids: list[str],
        *,
        observed_only: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return finite associations for a specific variant × analysis set."""
        # Row-index resolution only -- no VariantRecord materialisation, no
        # per-identifier variants.tsv.gz open (issue #3).
        row_indices = self._variant_axis.indices_by_identifiers(identifiers).tolist()
        analyses = [a for aid in analysis_ids if (a := self._analyses.by_id(aid)) is not None]
        if not row_indices or not analyses:
            return _empty_result()
        col_indices = [int(a["analysis_index"]) for a in analyses]
        # Surgical orthogonal read: fetch only the chunks intersecting the
        # requested rows × cols, not the full analysis width per row. Under a
        # narrow analysis chunk this reads far fewer chunks (issue 052).
        z_block = self._z.block(row_indices, col_indices)
        eaf_read = self._eaf.read_block(row_indices, col_indices, want_imputed=True)
        se_block = self._se.block(row_indices, col_indices, eaf=eaf_read.values)
        mask = np.isfinite(z_block) & np.isfinite(se_block)
        rows_rel, cols_rel = np.where(mask)
        rows = np.array([row_indices[r] for r in rows_rel], dtype="int32")
        cols = np.array([col_indices[c] for c in cols_rel], dtype="int32")
        return self._shared_cell_result(
            rows, cols, z_block[mask], se_block[mask], eaf_read, mask, observed_only=observed_only
        )

    def top_hits(
        self,
        *,
        analysis_id: str | None = None,
        threshold: float = 5e-8,
        limit: int | None = None,
        observed_only: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return genomic-order top hits, optionally for one analysis."""
        reader = self._top_hits.reader(threshold)
        if reader is None:
            return _empty_result()
        analysis_index: int | None = None
        if analysis_id is not None:
            analysis = self._analyses.by_id(analysis_id)
            if analysis is None or not reader.has("analysis_offsets"):
                return _empty_result()
            analysis_index = int(analysis["analysis_index"])
        bounds = reader.bounds(analysis_index)
        variant_indices = reader.read("variant_index", bounds, "int32")
        analysis_indices = reader.read("analysis_index", bounds, "int32")
        z_values = reader.read("z", bounds, "float32")
        se_values, eaf, imp = self._top_hit_fields(
            reader, bounds, variant_indices, analysis_indices
        )
        return _top_hits_result(
            variant_indices,
            analysis_indices,
            z_values,
            se_values,
            eaf,
            imp,
            observed_only=observed_only,
            limit=limit,
        )

    def _top_hit_fields(
        self,
        reader: DenseTopHitReader,
        bounds: tuple[int, int],
        variant_indices: np.ndarray,
        analysis_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`se`, `eaf` and `imputed` for the indexed top hits, each read once.

        A current index carries all three decoded. An older one carries none,
        so SE decoding and the result's `eaf` column derive them from the
        planes; the one region read is shared between them (#253), where the
        two fallbacks used to read it twice.
        """
        has_se = reader.has("se")
        has_eaf = reader.has("eaf")
        has_imputed = reader.has("imputed")
        eaf_read = (
            self._eaf.read_points(
                variant_indices.astype("int64"),
                analysis_indices.astype("int64"),
                want_imputed=not has_imputed,
            )
            if not (has_se and has_eaf)
            else None
        )
        if has_se:
            se_values = reader.read("se", bounds, "float32")
        else:
            assert eaf_read is not None
            se_values = self._se.points(
                variant_indices.astype("int64"),
                analysis_indices.astype("int64"),
                eaf=eaf_read.values,
            )
        eaf = reader.read("eaf", bounds, "float32") if has_eaf else eaf_read.values
        if has_imputed:
            imp = reader.read("imputed", bounds, "uint8")
        elif eaf_read is not None and eaf_read.imputed is not None:
            imp = np.asarray(eaf_read.imputed, dtype="uint8")
        else:
            imp = self._imputed_pairs(variant_indices, analysis_indices)
        return se_values, eaf, imp

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


def _chunk_windows(lo: int, hi: int, chunk: int) -> Iterator[tuple[int, int]]:
    """Half-open windows of `chunk` covering `[lo, hi)`.

    The variant-side scans read in these windows so peak memory is a window,
    not the association count (#252): at OGS-00011's 3,085,080,783-cell Overflow
    a whole-array decode held 12.3 GB of `variant_index` and 3.1 GB of mask at
    once. The window is the association arrays' own inner chunk, so no chunk is
    read twice.
    """
    start = int(lo)
    stop = int(hi)
    step = max(1, int(chunk))
    while start < stop:
        yield start, min(start + step, stop)
        start += step


def _filter_block_to_variants(
    block: dict[str, np.ndarray], wanted: np.ndarray, low: int, high: int
) -> dict[str, np.ndarray]:
    """A by-variant block filtered to `wanted` when the range is not contiguous.

    A variant index is position-ordered, so a range query's variants are usually
    every index in `[low, high]`.  When they are not -- the Hybrid case, where a
    window's off-panel variants interleave with on-panel ones that have no
    overflow rows -- the block's rows for the un-wanted variants are dropped.
    """
    if len(wanted) == high - low + 1:
        return block
    keep = np.isin(block["variant_index"], wanted)
    return {name: values[keep] for name, values in block.items()}


def _by_variant_block_result(
    block: dict[str, np.ndarray], *, observed_only: bool
) -> dict[str, np.ndarray]:
    """The six result arrays for a decoded by-variant block (ADR 0060).

    A block already holds whole variants, so this is the Ragged and Hybrid
    indexed paths' one finalizer -- the same one the step-3 scan finalizes
    through, so the two routes cannot assemble a result differently.
    """
    return _top_hits_result(
        block["variant_index"],
        block["analysis_index"],
        block["z"],
        block["se"],
        block["eaf"],
        block["imputed"],
        observed_only=observed_only,
        limit=None,
    )


def _eaf_at_positions(
    reader: RaggedCSRReader, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One frequency read at flat CSR positions: `(values, imputed mask)` (#252).

    The values carry the panel substitution on imputed cells and the mask they
    were substituted under comes back with them, so SE decoding and Association
    Status cannot be aligned to different reads (#253).
    """
    read = reader.eaf_at_read(positions, want_imputed=True)
    values = np.asarray(read.values, dtype=np.float32)
    if read.imputed is None:
        return values, np.zeros(len(positions), dtype=np.uint8)
    return values, np.asarray(read.imputed, dtype=np.uint8)


class RaggedStoreQuery:
    """Public query object that hides the physical store layout — Ragged stores."""""

    def __init__(self, store: OpenGWASDBStore):
        self.store = store
        self._csr = RaggedCSRReader(store.path)
        self._variant_axis = VariantAxis(store.path)
        self._by_variant = _open_by_variant(store, self._variant_axis)
        self._analyses = AnalysesIndex(store.path)
        self._top_hits = TopHitTiers(store.arrays(mode="r"))

    def close(self) -> None:
        self._variant_axis.close()

    def __enter__(self) -> RaggedStoreQuery:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _resolve_analysis_id(self, analysis_id: str) -> int | None:
        row = self._analyses.by_id(analysis_id)
        return None if row is None else int(row["analysis_index"])

    def _decoded_slice(
        self, start: int, end: int, analysis_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """`(z, se, eaf, imputed)` for one Analysis's CSR slice.

        The one read of the slice's frequencies is shared by SE decoding and
        the returned `eaf` column (#253); `imputed` is the mask that read
        carried, or zeros when the component holds none.
        """
        z = self._csr.z_slice(start, end)
        eaf_read = self._csr.eaf_slice_read(start, end, want_imputed=True)
        se = self._csr.se_slice(start, end, analysis_index=analysis_index, eaf=eaf_read.values)
        imputed = (
            eaf_read.imputed
            if eaf_read.imputed is not None
            else np.zeros(len(z), dtype=np.uint8)
        )
        return z, se, eaf_read.values, imputed

    def variants_table(self) -> dict[int, dict]:
        """Return all variants keyed by variant_index."""
        return _variants_table(self._variant_axis)

    def analyses_table(self) -> dict[int, dict]:
        """Return all analyses keyed by analysis_index -- every analyses.tsv column.

        Same shape StoreQuery/HybridStoreQuery return (ADR-0034's unified
        schema). A caller grouping Ragged Analyses by a shared gene filters
        this table on trait_ontology_id rather than through a separate
        lookup.
        """
        return self._analyses.all()

    def resolve(
        self, result: dict[str, np.ndarray], *, include_variant_info: bool = False
    ) -> Iterator[dict[str, object]]:
        """Resolve a raw (index-keyed) result to human-readable rows (issue #104);
        see `opengwasdb.query.resolve.resolve_rows`."""
        return resolve_rows(
            self._analyses, self._variant_axis, result, include_variant_info=include_variant_info
        )

    def analysis(self, analysis_id: str, *, observed_only: bool = False) -> dict[str, np.ndarray]:
        """All associations for one analysis (analysis_id lookup)."""
        idx = self._resolve_analysis_id(analysis_id)
        if idx is None:
            return _empty_result()
        offsets = self._csr._offsets[idx : idx + 2]
        start, end = int(offsets[0]), int(offsets[1])
        if start == end:
            return _empty_result()
        vi = self._csr._variant_index[start:end].astype("int32")
        # One frequency read for the whole Analysis, shared by SE decoding and
        # the result's `eaf` column (#253) -- the plane used to be read once by
        # `se_slice` and again by `eaf_slice`.
        z, se, eaf, imp = self._decoded_slice(start, end, idx)
        if observed_only:
            mask = imp == 0
            vi, z, se, eaf, imp = vi[mask], z[mask], se[mask], eaf[mask], imp[mask]
        status = _status_array(imp, z, se)
        return {
            "variant_index": vi,
            "analysis_index": np.full(len(z), idx, dtype="int32"),
            "z": z,
            "se": se,
            "eaf": eaf,
            "association_status": status,
        }

    def range_phewas(
        self,
        chromosome: str,
        start: int,
        end: int,
        *,
        observed_only: bool = False,
    ) -> dict[str, np.ndarray]:
        """All associations where the variant falls in [start, end] (regional PheWAS)."""
        variant_indices = self._variant_axis.range_indices(chromosome, start, end)
        if len(variant_indices) == 0:
            return _empty_result()

        # With the variant-centric index (ADR 0060) the window's rows are one
        # contiguous block of the index: read exactly them, not every
        # Analysis's `variant_index`.  Without it, search each Analysis's
        # sorted segment for the window's variants, in chunk-sized windows
        # (#252).  Neither path materialises a whole plane.
        if self._by_variant is not None:
            return self._indexed_variant_result(
                np.asarray(variant_indices, dtype=np.int32), observed_only=observed_only
            )
        hit_positions = self._csr.variant_positions(np.asarray(variant_indices, dtype=np.int32))
        return self._hit_rows_result(
            hit_positions,
            self._csr.variant_index_at(hit_positions),
            observed_only=observed_only,
        )

    def _indexed_variant_result(
        self, variant_indices: np.ndarray, *, observed_only: bool
    ) -> dict[str, np.ndarray]:
        """The by-variant index's rows for `variant_indices` (ADR 0060).

        The index is contiguous and variant-ordered, so a (possibly gapped)
        set of variants is the block spanning its minimum and maximum, filtered
        to the set when the set is not itself contiguous.
        """
        assert self._by_variant is not None
        low, high = int(variant_indices.min()), int(variant_indices.max())
        first, last = self._by_variant.rows_for_variant_range(low, high)
        if first == last:
            return _empty_result()
        block = self._by_variant.decode(low, high, first, last)
        block = _filter_block_to_variants(
            block, np.asarray(variant_indices, dtype=np.int32), low, high
        )
        return _by_variant_block_result(block, observed_only=observed_only)

    def _hit_rows_result(
        self,
        hit_positions: np.ndarray,
        variant_indexes: np.ndarray,
        *,
        observed_only: bool,
        limit: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Decode flat CSR hit positions into the six-array result shape.

        The decode `range_phewas` and `phewas` share (issue #130): each
        resolves a set of flat CSR positions and a parallel `variant_indexes`
        array its own way -- range reads every hit's own variant from the CSR
        column, phewas fills the single target variant, and the two are
        deliberately not unified -- but a flat position then decodes the same
        in both: to its Analysis through the CSR offsets, to its imputed flag,
        through the `observed_only` filter, then to z/se/eaf and status. A
        hand-maintained copy of this block previously lived in both methods;
        a divergence between them would be silent, so the decode lives here
        once and the parity tests pin both methods to it.
        """
        if len(hit_positions) == 0:
            return _empty_result()
        hit_positions = np.asarray(hit_positions, dtype=np.int64)
        # The full `offsets` array is one int64 per Analysis (not per
        # association), so this is the one array the scan may hold whole.
        offsets = np.asarray(self._csr._offsets[:], dtype=np.int64)
        analysis_indices = np.searchsorted(offsets[1:], hit_positions, side="right").astype("int32")
        # One frequency read at the hit positions -- not `z_all()`/`se_all()`,
        # which decoded the whole z and se planes (and, through residual SE, the
        # whole eaf plane) to keep a handful of rows (#252). The same decoded
        # EAF feeds SE decoding and the `eaf` column through `_top_hits_result`.
        eaf_values, imp = _eaf_at_positions(self._csr, hit_positions)
        if observed_only:
            keep = imp == 0
            hit_positions = hit_positions[keep]
            analysis_indices = analysis_indices[keep]
            variant_indexes = variant_indexes[keep]
            imp = imp[keep]
            eaf_values = eaf_values[keep]

        z_out = self._csr.z_at(hit_positions)
        se_out = self._csr.se_at(hit_positions, eaf=eaf_values)
        return _top_hits_result(
            variant_indexes,
            analysis_indices,
            z_out,
            se_out,
            eaf_values,
            imp,
            observed_only=observed_only,
            limit=limit,
        )

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
        self,
        chromosome: str,
        start: int,
        end: int,
        *,
        observed_only: bool = False,
    ) -> dict[str, np.ndarray]:
        """All associations for analyses whose probe/TSS falls in [start, end]."""
        analysis_indices = self._analysis_indices_in_range(chromosome, start, end)
        if not analysis_indices:
            return _empty_result()

        all_vi: list[np.ndarray] = []
        all_ai: list[np.ndarray] = []
        all_z: list[np.ndarray] = []
        all_se: list[np.ndarray] = []
        all_eaf: list[np.ndarray] = []
        all_status: list[np.ndarray] = []

        for ai in analysis_indices:
            offsets = self._csr._offsets[ai : ai + 2]
            s, e = int(offsets[0]), int(offsets[1])
            if s == e:
                continue
            vi = self._csr._variant_index[s:e].astype("int32")
            z, se, eaf, imp = self._decoded_slice(s, e, ai)
            if observed_only:
                keep = imp == 0
                vi, z, se, eaf, imp = vi[keep], z[keep], se[keep], eaf[keep], imp[keep]
            if len(z) == 0:
                continue
            all_vi.append(vi)
            all_ai.append(np.full(len(z), ai, dtype="int32"))
            all_z.append(z)
            all_se.append(se)
            all_eaf.append(eaf)
            all_status.append(_status_array(imp, z, se))

        if not all_vi:
            return _empty_result()
        return {
            "variant_index": np.concatenate(all_vi),
            "analysis_index": np.concatenate(all_ai),
            "z": np.concatenate(all_z),
            "se": np.concatenate(all_se),
            "eaf": np.concatenate(all_eaf),
            "association_status": np.concatenate(all_status),
        }

    def phewas(self, identifier: str, *, observed_only: bool = False) -> dict[str, np.ndarray]:
        """All analyses that have an association for a given variant identifier.

        With the variant-centric index (ADR 0060) this reads one variant's
        contiguous row block; without it the step-3 scan reads every Analysis's
        `variant_index` in chunk-sized windows and decodes z/se/eaf only at the
        hits, so peak memory is a window rather than the store (#252).
        """
        variant = self._variant_axis.by_identifier(identifier)
        if variant is None:
            return _empty_result()

        # With the index, one variant's rows are a contiguous block: read them
        # and stop, instead of scanning every Analysis's `variant_index`
        # (#252, ADR 0060).  Without it the scan below is the answer, bounded
        # to a window rather than 4N bytes.
        if self._by_variant is not None:
            target_variant = int(variant.variant_index)
            first, last = self._by_variant.rows_for_variant(target_variant)
            if first == last:
                return _empty_result()
            block = self._by_variant.decode(target_variant, target_variant, first, last)
            return _by_variant_block_result(block, observed_only=observed_only)
        target_vi = np.int32(variant.variant_index)

        # The scan is O(total associations) and stays so on an unindexed
        # release, but it reads in chunk-sized windows, so peak memory is a
        # window rather than 4N bytes of `variant_index` (#252).
        hit_positions = self._csr.variant_positions(np.array([target_vi], dtype=np.int32))
        return self._hit_rows_result(
            hit_positions,
            np.full(len(hit_positions), target_vi, dtype="int32"),
            observed_only=observed_only,
        )

    def top_hits(
        self,
        *,
        analysis_id: str | None = None,
        threshold: float = 5e-8,
        limit: int | None = None,
        observed_only: bool = False,
    ) -> dict[str, np.ndarray]:
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
        reader = self._top_hits.reader(threshold)
        if reader is not None:
            analysis_index = None
            if analysis_id is not None:
                analysis_index = self._resolve_analysis_id(analysis_id)
                if analysis_index is None or not reader.has("analysis_offsets"):
                    return _empty_result()
            bounds = reader.bounds(analysis_index)
            vi = reader.read("variant_index", bounds, "int32")
            ai = reader.read("analysis_index", bounds, "int32")
            z = reader.read("z", bounds, "float32")
            se = reader.read("se", bounds, "float32")
            imp = reader.read_or(
                "imputed", bounds, "uint8", lambda: np.zeros(len(vi), dtype=np.uint8)
            )
            eaf = reader.read_or("eaf", bounds, "float32", lambda: self._csr.eaf_pairs(vi, ai))
            return _top_hits_result(
                vi, ai, z, se, eaf, imp, observed_only=observed_only, limit=limit
            )

        return self._top_hits_by_scan(
            analysis_id=analysis_id,
            threshold=threshold,
            limit=limit,
            observed_only=observed_only,
        )

    def _scan_threshold_hits(self, lo: int, hi: int, threshold: float) -> np.ndarray:
        """Flat CSR positions in `[lo, hi)` clearing `threshold`.

        The scan reads z in chunk-sized windows and keeps only what clears the
        cutoff, so the whole-plane `z_all()` decode (plus the whole
        `variant_index` and `imputed` arrays) is never held (#252). The cutoff
        is `z_critical`, the one the index is built with, so a scan and an
        index agree on the boundary.
        """
        cutoff = z_critical(threshold)
        windows: list[np.ndarray] = []
        for start, stop in _chunk_windows(lo, hi, self._csr.scan_window):
            mask = np.abs(self._csr.z_slice(start, stop)) >= cutoff
            if mask.any():
                windows.append(np.where(mask)[0].astype(np.int64) + start)
        if not windows:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(windows)

    def _top_hits_by_scan(
        self,
        *,
        analysis_id: str | None,
        threshold: float,
        limit: int | None,
        observed_only: bool,
    ) -> dict[str, np.ndarray]:
        """Top hits with no index to read: scan the CSR and threshold it.

        Returns the same shape of answer as the indexed path, applying
        `observed_only` before `limit` exactly as that path does. `analysis_id`
        is resolved here too -- the indexed path resolves it via `bounds()` --
        so a caller passing one gets a filtered result on both paths rather
        than an unfiltered store-wide scan on this one. The hits are then
        decoded by `_hit_rows_result`, the same finalizer `phewas` and
        `range_phewas` use.
        """
        if analysis_id is not None:
            analysis_index = self._resolve_analysis_id(analysis_id)
            if analysis_index is None:
                return _empty_result()
        else:
            analysis_index = None

        offsets = np.asarray(self._csr._offsets[:], dtype=np.int64)
        if analysis_index is None:
            lo, hi = 0, self._csr.n_associations
        else:
            lo, hi = int(offsets[analysis_index]), int(offsets[analysis_index + 1])
        hit_positions = self._scan_threshold_hits(lo, hi, threshold)
        if len(hit_positions) == 0:
            return _empty_result()
        return self._hit_rows_result(
            hit_positions,
            self._csr.variant_index_at(hit_positions),
            observed_only=observed_only,
            limit=limit,
        )

    def lookup(
        self,
        identifiers: list[str],
        analysis_ids: list[str],
        *,
        observed_only: bool = False,
    ) -> dict[str, np.ndarray]:
        """Associations for a specific variant × analysis set.

        Each requested Analysis's sorted segment is binary-searched for the
        wanted variants (`segment_positions`, O(log) chunk reads), rather than
        the Analysis being decoded whole and `np.isin`-ed. The cost is the
        requested variants and Analyses, not the store and not the requested
        Analyses' sizes (#252). Requested Analysis order and duplicates carry
        through; a request that resolves nothing yields the empty result.
        """
        variants = [
            v for id_ in identifiers if (v := self._variant_axis.by_identifier(id_)) is not None
        ]
        if not variants:
            return _empty_result()

        wanted = np.array(sorted({v.variant_index for v in variants}), dtype=np.int32)
        parts = []
        for aid in analysis_ids:
            idx = self._resolve_analysis_id(aid)
            if idx is None:
                continue
            positions = self._csr.segment_positions(wanted, analysis_index=idx)
            if len(positions) == 0:
                continue
            parts.append(
                self._hit_rows_result(
                    positions,
                    self._csr.variant_index_at(positions),
                    observed_only=observed_only,
                )
            )
        return _concat_results(parts)


def _concat_results(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate query-result dicts (each a plain component read)."""
    parts = [p for p in parts if len(p["z"])]
    if not parts:
        return _empty_result()
    return {
        "variant_index": np.concatenate([p["variant_index"] for p in parts]).astype("int32"),
        "analysis_index": np.concatenate([p["analysis_index"] for p in parts]).astype("int32"),
        "z": np.concatenate([p["z"] for p in parts]).astype("float32"),
        "se": np.concatenate([p["se"] for p in parts]).astype("float32"),
        "eaf": np.concatenate([p["eaf"] for p in parts]).astype("float32"),
        "association_status": np.concatenate([p["association_status"] for p in parts]),
    }


class HybridStoreQuery:
    """Query facade for a Hybrid store (ADR 0026).

    A thin integration layer: it dispatches to the nested Dense Component's
    ``StoreQuery`` (on-panel variants) and to the Ragged Overflow CSR (off-panel
    variants), remaps the Dense Component's panel-local ``variant_index`` onto the
    shared variant index space, and concatenates. A variant is in exactly one
    component (on-panel xor off-panel), so results are a plain union with no dedup.
    """

    def __init__(self, store: OpenGWASDBStore):
        self.store = store
        self._dense_store = open_store(dense_component_path(store.path))
        self._dense = StoreQuery(self._dense_store)
        self._dense_to_shared = np.load(dense_to_shared_path(store.path)).astype("int32")
        self._csr = RaggedCSRReader(store.path)  # overflow at store/data.zarr/ragged
        self._connection = store.index_connection()
        self._analyses = AnalysesIndex(store.path)  # shared analyses.tsv
        self._variant_axis = VariantAxis(store.path, self._connection)  # shared union table
        self._by_variant = _open_by_variant(store, self._variant_axis)
        self._top_hits = TopHitTiers(store.arrays(mode="r"))  # overflow's own tiers

    def close(self) -> None:
        self._dense.close()
        self._variant_axis.close()
        self._connection.close()

    def __enter__(self) -> HybridStoreQuery:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── shared-table tables ──────────────────────────────────────────────────
    def analyses_table(self) -> dict[int, dict]:
        """Every column of the *shared* `analyses.tsv`, keyed by analysis_index.

        Deliberately not delegated to the Dense Component. Hybrid builds write
        `analyses.tsv` twice -- once under `dense/` counting only that
        component's on-panel top hits, once at the shared root where
        `add_hit_counts()` has additionally counted the Ragged Overflow
        Component's -- so the Dense Component's own copy undercounts
        `n_hits_*` for any Analysis with off-panel hits (issue #107). The
        Analytical Metadata columns are identical in both; only the counts
        differ, which is exactly what makes the wrong one easy to miss.
        """
        return self._analyses.all()

    def resolve(
        self, result: dict[str, np.ndarray], *, include_variant_info: bool = False
    ) -> Iterator[dict[str, object]]:
        """Resolve a raw (index-keyed) result to human-readable rows (issue #104);
        see `opengwasdb.query.resolve.resolve_rows`. Uses the shared
        analyses/variant tables (not the Dense Component's panel-local
        ones), matching the shared `variant_index` space `resolve()`'s
        results are already remapped into."""
        return resolve_rows(
            self._analyses, self._variant_axis, result, include_variant_info=include_variant_info
        )

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

    def variants_table(self) -> dict[int, dict]:
        return _variants_table(self._variant_axis)

    # ── dispatch helpers ─────────────────────────────────────────────────────
    def _remap_dense(self, result: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Translate a Dense Component result's panel-local variant_index to shared."""
        if len(result["variant_index"]):
            result["variant_index"] = self._dense_to_shared[
                result["variant_index"].astype("int64")
            ].astype("int32")
        return result

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

    def _overflow_for_analysis(self, col: int) -> dict[str, np.ndarray]:
        offsets = self._csr._offsets[col : col + 2]
        s, e = int(offsets[0]), int(offsets[1])
        if s == e:
            return _empty_result()
        vi = self._csr._variant_index[s:e].astype("int32")
        z = self._csr.z_slice(s, e)
        eaf_read = self._csr.eaf_slice_read(s, e)
        se = self._csr.se_slice(s, e, analysis_index=col, eaf=eaf_read.values)
        return {
            "variant_index": vi,
            "analysis_index": np.full(len(z), col, dtype="int32"),
            "z": z,
            "se": se,
            "eaf": eaf_read.values,
            "association_status": _status_array(np.zeros(len(z), dtype=np.uint8), z, se),
        }

    def _overflow_rows(self, positions: np.ndarray) -> dict[str, np.ndarray]:
        """The six arrays for Overflow cells at flat CSR positions.

        The Overflow is always observed (ADR 0026), so Association Status is
        `observed`/`missing` from z and se alone. The frequency is read once at
        the hit positions and handed to SE decoding as well as returned (#252):
        `z_at`/`se_at`/`eaf_at` each used to read the plane, and `se_at` read it
        again to predict a residual.
        """
        positions = np.asarray(positions, dtype=np.int64)
        if len(positions) == 0:
            return _empty_result()
        offsets = np.asarray(self._csr._offsets[:], dtype=np.int64)
        analysis_indices = np.searchsorted(offsets[1:], positions, side="right").astype("int32")
        eaf_values = np.asarray(self._csr.eaf_at(positions), dtype=np.float32)
        z = self._csr.z_at(positions)
        se = self._csr.se_at(positions, eaf=eaf_values)
        return {
            "variant_index": self._csr.variant_index_at(positions),
            "analysis_index": analysis_indices,
            "z": z,
            "se": se,
            "eaf": eaf_values,
            "association_status": _status_array(np.zeros(len(positions), dtype=np.uint8), z, se),
        }

    def _overflow_by_variants(
        self, shared_indices: set[int], wanted_analyses: set[int] | None = None
    ) -> dict[str, np.ndarray]:
        """All overflow associations whose (off-panel) variant is in the set.

        With `wanted_analyses` given, each requested Analysis's sorted segment
        is **binary-searched** for the wanted variants (`segment_positions`, O(log)
        chunk reads): its rows are sorted by variant_index, so no new index is
        needed to locate a (variant, Analysis) pair, and a lookup costs its
        requested variants and Analyses and not the requested Analyses' sizes
        or the store (#252). Without it, where every Analysis may hold the
        variant, a whole-store windowed scan is used instead: one zarr read per
        scan window is much cheaper than one per Analysis segment, and neither
        holds the array whole.

        Either route returns flat CSR order (Analysis ascending, variant
        ascending within each), the order the whole-store scan this replaces
        produced, so answers are identical.
        """
        if not shared_indices:
            return _empty_result()
        wanted = np.array(sorted(shared_indices), dtype=np.int32)
        if wanted_analyses is None:
            if self._by_variant is not None:
                return self._overflow_indexed(wanted)
            positions = self._csr.variant_positions(wanted)
            return self._overflow_rows(positions)
        parts: list[dict[str, np.ndarray]] = []
        for col in sorted(wanted_analyses):
            positions = self._csr.segment_positions(wanted, analysis_index=int(col))
            if len(positions) == 0:
                continue
            parts.append(self._overflow_rows(positions))
        return _concat_results(parts)

    def _overflow_indexed(self, wanted: np.ndarray) -> dict[str, np.ndarray]:
        """The Overflow's by-variant index rows for `wanted` (ADR 0060).

        The index holds one contiguous block per variant, so a set of wanted
        off-panel variants is the block spanning its minimum and maximum,
        filtered to the set when on-panel variants interleave.  Reads the
        answer, not the Overflow's `variant_index`.
        """
        assert self._by_variant is not None
        low, high = int(wanted.min()), int(wanted.max())
        first, last = self._by_variant.rows_for_variant_range(low, high)
        if first == last:
            return _empty_result()
        block = self._by_variant.decode(low, high, first, last)
        block = _filter_block_to_variants(block, wanted, low, high)
        return {
            "variant_index": block["variant_index"].astype("int32"),
            "analysis_index": block["analysis_index"].astype("int32"),
            "z": block["z"],
            "se": block["se"],
            "eaf": block["eaf"],
            "association_status": _status_array(block["imputed"], block["z"], block["se"]),
        }

    # ── public query surface ─────────────────────────────────────────────────
    def analysis(self, analysis_id: str, *, observed_only: bool = False) -> dict[str, np.ndarray]:
        dense = self._remap_dense(self._dense.analysis(analysis_id, observed_only=observed_only))
        analysis = self._analyses.by_id(analysis_id)
        overflow = (
            self._overflow_for_analysis(int(analysis["analysis_index"]))
            if analysis is not None
            else _empty_result()
        )
        return _concat_results([dense, overflow])

    def phewas(self, identifier: str, *, observed_only: bool = False) -> dict[str, np.ndarray]:
        variant = self._variant_axis.by_identifier(identifier)
        if variant is None:
            return _empty_result()
        if self._shared_is_on_panel(variant.variant_index):
            return self._remap_dense(self._dense.phewas(variant.alid, observed_only=observed_only))
        return self._overflow_by_variants({int(variant.variant_index)})

    def range_phewas(
        self, chromosome: str, start: int, end: int, *, observed_only: bool = False
    ) -> dict[str, np.ndarray]:
        dense = self._remap_dense(
            self._dense.range_phewas(chromosome, start, end, observed_only=observed_only)
        )
        shared_idx = self._variant_axis.range_indices(chromosome, start, end)
        on_panel = self._on_panel_mask(shared_idx)
        off_panel = {int(i) for i in np.asarray(shared_idx)[~on_panel].tolist()}
        overflow = self._overflow_by_variants(off_panel)
        return _concat_results([dense, overflow])

    def lookup(
        self,
        identifiers: list[str],
        analysis_ids: list[str],
        *,
        observed_only: bool = False,
    ) -> dict[str, np.ndarray]:
        dense = self._remap_dense(
            self._dense.lookup(identifiers, analysis_ids, observed_only=observed_only)
        )
        # Off-panel identifiers: resolve on the shared table, keep off-panel ones.
        records = [
            rec for id_ in identifiers if (rec := self._variant_axis.by_identifier(id_)) is not None
        ]
        off_shared: set[int] = set()
        if records:
            shared = np.array([r.variant_index for r in records], dtype=np.int64)
            for rec, off in zip(records, ~self._on_panel_mask(shared), strict=True):
                if off:
                    off_shared.add(int(rec.variant_index))
        wanted_cols = {
            int(a["analysis_index"])
            for aid in analysis_ids
            if (a := self._analyses.by_id(aid)) is not None
        }
        overflow = self._overflow_by_variants(off_shared, wanted_cols)
        return _concat_results([dense, overflow])

    def _overflow_top_hits(
        self, threshold: float, analysis_index: int | None = None
    ) -> dict[str, np.ndarray]:
        reader = self._top_hits.reader(threshold)
        if reader is None:
            return _empty_result()
        if analysis_index is not None and not reader.has("analysis_offsets"):
            return _empty_result()
        bounds = reader.bounds(analysis_index)
        vi = reader.read("variant_index", bounds, "int32")
        ai = reader.read("analysis_index", bounds, "int32")
        z = reader.read("z", bounds, "float32")
        se = reader.read("se", bounds, "float32")
        eaf = reader.read_or("eaf", bounds, "float32", lambda: self._csr.eaf_pairs(vi, ai))
        return {
            "variant_index": vi,
            "analysis_index": ai,
            "z": z,
            "se": se,
            "eaf": eaf,
            "association_status": _status_array(np.zeros(len(z), dtype=np.uint8), z, se),
        }

    def top_hits(
        self,
        *,
        analysis_id: str | None = None,
        threshold: float = 5e-8,
        limit: int | None = None,
        observed_only: bool = False,
    ) -> dict[str, np.ndarray]:
        analysis_index = None
        if analysis_id is not None:
            analysis = self._analyses.by_id(analysis_id)
            if analysis is None:
                return _empty_result()
            analysis_index = int(analysis["analysis_index"])
        dense = self._remap_dense(
            self._dense.top_hits(
                analysis_id=analysis_id, threshold=threshold, observed_only=observed_only
            )
        )
        overflow = self._overflow_top_hits(threshold, analysis_index)  # overflow is always observed
        merged = _concat_results([dense, overflow])
        if len(merged["z"]):
            order = np.lexsort((merged["variant_index"], merged["analysis_index"]))
            merged = {k: v[order] for k, v in merged.items()}
        if limit is not None:
            merged = {k: v[:limit] for k, v in merged.items()}
        return merged


def query_store(path: str | Path) -> StoreQuery | RaggedStoreQuery | HybridStoreQuery:
    """Open a store and return the layout-independent query facade."""
    store = open_store(path)
    if store.manifest.primary_layout is PrimaryStorageLayout.RAGGED:
        return RaggedStoreQuery(store)
    if store.manifest.primary_layout is PrimaryStorageLayout.HYBRID:
        return HybridStoreQuery(store)
    return StoreQuery(store)
