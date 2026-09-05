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

from opengwasdb.encoding import DenseEafPlane, DenseZPlane
from opengwasdb.index import AnalysesIndex
from opengwasdb.layouts.dense.rho import DenseRhoReader
from opengwasdb.layouts.dense.top_hits import DenseTopHitReader, threshold_key
from opengwasdb.layouts.hybrid.layout import dense_component_path, dense_to_shared_path
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


class StoreQuery:
    """Public query object that hides the physical store layout — Dense stores."""

    def __init__(self, store: OpenGWASDBStore):
        self.store = store
        self._connection = store.index_connection()
        self._analyses = AnalysesIndex(store.path)
        self._root = store.arrays(mode="r")
        self._variant_axis = VariantAxis(store.path, self._connection)
        self._is_completed = (
            store.manifest.completion_state is CompletionState.REFERENCE_COMPLETED
        )
        self._imputed: zarr.Array | None = (
            self._root["imputed"] if self._is_completed and "imputed" in self._root else None
        )
        # Every z and eaf read goes through its plane: the store's declared
        # encoding (ADR 0037) is applied in one place rather than at each
        # result site, and the eaf plane gathers the per-variant baseline and
        # the reference frequency for imputed cells with it.
        self._z = DenseZPlane.open(self._root, store.manifest.encoding)
        self._eaf = DenseEafPlane.open(self._root, store.manifest.encoding)
        self._rho_reader: DenseRhoReader | None = (
            DenseRhoReader(self._root["rho"], self._z.n_analyses)
            if "rho" in self._root
            else None
        )

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

    @staticmethod
    def _contiguous_row_slice(row_indices: np.ndarray) -> slice | None:
        if len(row_indices) == 0:
            return None
        start = int(row_indices[0])
        stop = int(row_indices[-1]) + 1
        if stop - start != len(row_indices):
            return None
        if not np.array_equal(row_indices, np.arange(start, stop, dtype=row_indices.dtype)):
            return None
        return slice(start, stop)

    @classmethod
    def _read_row_block(
        cls, array: zarr.Array, row_indices: np.ndarray, dtype: str | np.dtype
    ) -> np.ndarray:
        row_slice = cls._contiguous_row_slice(row_indices)
        if row_slice is not None:
            return array[row_slice, :].astype(dtype)
        return array.oindex[row_indices, :].astype(dtype)

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

    def analysis(self, analysis_id: str, *, observed_only: bool = False) -> dict[str, np.ndarray]:
        """Return all finite associations for one analysis."""
        analysis = self._analyses.by_id(analysis_id)
        if analysis is None:
            return _empty_result()
        col = int(analysis["analysis_index"])
        z_col = self._z.column(col)
        se_col = self._root["se"][:, col].astype("float32")
        mask = np.isfinite(z_col) & np.isfinite(se_col)
        rows = np.where(mask)[0].astype("int32")
        cols = np.full(len(rows), col, dtype="int32")
        z_vals = z_col[mask]
        se_vals = se_col[mask]
        imp = self._imputed_pairs(rows, cols)
        if observed_only:
            keep = imp == 0
            rows, cols, z_vals, se_vals, imp = (
                rows[keep], cols[keep], z_vals[keep], se_vals[keep], imp[keep]
            )
        return {
            "variant_index": rows,
            "analysis_index": cols,
            "z": z_vals,
            "se": se_vals,
            "eaf": self._eaf_pairs(rows, cols),
            "association_status": _status_array(imp, z_vals, se_vals),
        }

    def phewas(self, identifier: str, *, observed_only: bool = False) -> dict[str, np.ndarray]:
        """Return one variant across all analyses."""
        variant = self._variant_axis.by_identifier(identifier)
        if variant is None:
            return _empty_result()
        row = variant.variant_index
        z_row = self._z.row(row)
        se_row = self._root["se"][row, :].astype("float32")
        mask = np.isfinite(z_row) & np.isfinite(se_row)
        cols = np.where(mask)[0].astype("int32")
        rows = np.full(len(cols), row, dtype="int32")
        z_vals = z_row[mask]
        se_vals = se_row[mask]
        imp = self._imputed_pairs(rows, cols)
        if observed_only:
            keep = imp == 0
            rows, cols, z_vals, se_vals, imp = (
                rows[keep], cols[keep], z_vals[keep], se_vals[keep], imp[keep]
            )
        return {
            "variant_index": rows,
            "analysis_index": cols,
            "z": z_vals,
            "se": se_vals,
            "eaf": self._eaf_pairs(rows, cols),
            "association_status": _status_array(imp, z_vals, se_vals),
        }

    def range_phewas(
        self, chromosome: str, start: int, end: int, *, observed_only: bool = False
    ) -> dict[str, np.ndarray]:
        """Return finite associations for all variants in a genomic range (regional PheWAS)."""
        row_indices = self._variant_axis.range_indices(chromosome, start, end)
        if len(row_indices) == 0:
            return _empty_result()
        z_block = self._z.rows(row_indices)
        se_block = self._read_row_block(self._root["se"], row_indices, "float32")
        mask = np.isfinite(z_block) & np.isfinite(se_block)
        rows_rel, cols = np.where(mask)
        rows = row_indices[rows_rel].astype("int32")
        cols = cols.astype("int32")
        z_vals = z_block[mask]
        se_vals = se_block[mask]
        row_slice = self._contiguous_row_slice(row_indices)
        if row_slice is not None:
            eaf_vals = self._eaf.band(row_slice.start, row_slice.stop)[mask]
        else:
            eaf_vals = self._eaf_pairs(rows, cols)
        if self._imputed is None:
            imp = np.zeros(len(z_vals), dtype=np.uint8)
        else:
            imp_block = self._read_row_block(self._imputed, row_indices, np.uint8)
            imp = imp_block[mask]
        if observed_only:
            keep = imp == 0
            rows, cols, z_vals, se_vals, eaf_vals, imp = (
                rows[keep], cols[keep], z_vals[keep], se_vals[keep], eaf_vals[keep], imp[keep]
            )
        return {
            "variant_index": rows,
            "analysis_index": cols,
            "z": z_vals,
            "se": se_vals,
            "eaf": eaf_vals,
            "association_status": _status_array(imp, z_vals, se_vals),
        }

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
        analyses = [
            a
            for aid in analysis_ids
            if (a := self._analyses.by_id(aid)) is not None
        ]
        if not row_indices or not analyses:
            return _empty_result()
        col_indices = [int(a["analysis_index"]) for a in analyses]
        # Surgical orthogonal read: fetch only the chunks intersecting the
        # requested rows × cols, not the full analysis width per row. Under a
        # narrow analysis chunk this reads far fewer chunks (issue 052).
        z_block = self._z.block(row_indices, col_indices)
        se_block = self._root["se"].oindex[row_indices, col_indices].astype("float32")
        mask = np.isfinite(z_block) & np.isfinite(se_block)
        rows_rel, cols_rel = np.where(mask)
        rows = np.array([row_indices[r] for r in rows_rel], dtype="int32")
        cols = np.array([col_indices[c] for c in cols_rel], dtype="int32")
        z_vals = z_block[mask]
        se_vals = se_block[mask]
        imp = self._imputed_pairs(rows, cols)
        if observed_only:
            keep = imp == 0
            rows, cols, z_vals, se_vals, imp = (
                rows[keep], cols[keep], z_vals[keep], se_vals[keep], imp[keep]
            )
        return {
            "variant_index": rows,
            "analysis_index": cols,
            "z": z_vals,
            "se": se_vals,
            "eaf": self._eaf_pairs(rows, cols),
            "association_status": _status_array(imp, z_vals, se_vals),
        }

    def top_hits(
        self,
        *,
        analysis_id: str | None = None,
        threshold: float = 5e-8,
        limit: int | None = None,
        observed_only: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return genomic-order top hits, optionally for one analysis."""
        key = threshold_key(threshold)
        path = f"top_hits/{key}"
        if path not in self._root:
            return _empty_result()
        group = self._root[path]
        analysis_index: int | None = None
        if analysis_id is not None:
            analysis = self._analyses.by_id(analysis_id)
            if analysis is None or "analysis_offsets" not in group:
                return _empty_result()
            analysis_index = int(analysis["analysis_index"])
        reader = DenseTopHitReader(group)
        bounds = reader.bounds(analysis_index)
        variant_indices = reader.read("variant_index", bounds, "int32")
        analysis_indices = reader.read("analysis_index", bounds, "int32")
        z_values = reader.read("z", bounds, "float32")
        if "se" in group:
            se_values = reader.read("se", bounds, "float32")
        else:
            # Pointwise (coordinate) read: fetch se at exactly the index cells,
            # not the full analysis width per row (issue 052).
            se_values = self._root["se"].vindex[
                variant_indices.astype("int64"), analysis_indices.astype("int64")
            ].astype("float32")
        if "imputed" in group:
            imp = reader.read("imputed", bounds, "uint8")
        else:
            imp = self._imputed_pairs(variant_indices, analysis_indices)
        if observed_only:
            keep = imp == 0
            variant_indices, analysis_indices, z_values, se_values, imp = (
                variant_indices[keep], analysis_indices[keep],
                z_values[keep], se_values[keep], imp[keep],
            )
        if limit is not None:
            variant_indices = variant_indices[:limit]
            analysis_indices = analysis_indices[:limit]
            z_values = z_values[:limit]
            se_values = se_values[:limit]
            imp = imp[:limit]
        return {
            "variant_index": variant_indices,
            "analysis_index": analysis_indices,
            "z": z_values,
            "se": se_values,
            "eaf": self._eaf_pairs(variant_indices, analysis_indices),
            "association_status": _status_array(imp, z_values, se_values),
        }

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


class RaggedStoreQuery:
    """Public query object that hides the physical store layout — Ragged stores."""

    def __init__(self, store: OpenGWASDBStore):
        self.store = store
        self._csr = RaggedCSRReader(store.path)
        self._variant_axis = VariantAxis(store.path)
        self._analyses = AnalysesIndex(store.path)
        self._is_completed = (
            store.manifest.completion_state is CompletionState.REFERENCE_COMPLETED
        )
        # Load imputed mask when present (reference-completed stores).
        ragged_path = store.data_path / "ragged"
        self._imputed: zarr.Array | None = None
        if self._is_completed:
            try:
                _root = zarr.open_group(str(ragged_path), mode="r")
                if "imputed" in _root:
                    self._imputed = _root["imputed"]
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        self._variant_axis.close()

    def __enter__(self) -> RaggedStoreQuery:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _resolve_analysis_id(self, analysis_id: str) -> int | None:
        row = self._analyses.by_id(analysis_id)
        return None if row is None else int(row["analysis_index"])

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

    def _get_imputed_slice(self, start: int, end: int) -> np.ndarray:
        """Return imputed mask slice [start:end]; all-zeros if not a completed store."""
        if self._imputed is not None:
            return self._imputed[start:end].astype(np.uint8)
        return np.zeros(end - start, dtype=np.uint8)

    def analysis(self, analysis_id: str, *, observed_only: bool = False) -> dict[str, np.ndarray]:
        """All associations for one analysis (analysis_id lookup)."""
        idx = self._resolve_analysis_id(analysis_id)
        if idx is None:
            return _empty_result()
        offsets = self._csr._offsets[idx: idx + 2]
        start, end = int(offsets[0]), int(offsets[1])
        if start == end:
            return _empty_result()
        vi = self._csr._variant_index[start:end].astype("int32")
        z = self._csr.z_slice(start, end)
        se = self._csr._se[start:end].astype("float32")
        eaf = self._csr.eaf_slice(start, end)
        imp = self._get_imputed_slice(start, end)
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
        variant_set = set(
            self._variant_axis.range_indices(chromosome, start, end).tolist()
        )
        if not variant_set:
            return _empty_result()

        offsets = self._csr._offsets[:]
        vi_all = self._csr._variant_index[:]
        z_all = self._csr.z_all()
        se_all = self._csr._se[:]

        mask = np.isin(vi_all, np.array(sorted(variant_set), dtype=np.int32))
        hit_positions = np.where(mask)[0]
        if len(hit_positions) == 0:
            return _empty_result()

        analysis_indices = np.searchsorted(offsets[1:], hit_positions, side="right").astype("int32")
        imp = (
            self._imputed[hit_positions].astype(np.uint8)
            if self._imputed is not None
            else np.zeros(len(hit_positions), dtype=np.uint8)
        )
        if observed_only:
            keep = imp == 0
            hit_positions = hit_positions[keep]
            analysis_indices = analysis_indices[keep]
            imp = imp[keep]

        z_out = z_all[hit_positions].astype("float32")
        se_out = se_all[hit_positions].astype("float32")
        return {
            "variant_index": vi_all[hit_positions].astype("int32"),
            "analysis_index": analysis_indices,
            "z": z_out,
            "se": se_out,
            "eaf": self._csr.eaf_at(hit_positions),
            "association_status": _status_array(imp, z_out, se_out),
        }

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
            offsets = self._csr._offsets[ai: ai + 2]
            s, e = int(offsets[0]), int(offsets[1])
            if s == e:
                continue
            vi = self._csr._variant_index[s:e].astype("int32")
            z = self._csr.z_slice(s, e)
            se = self._csr._se[s:e].astype("float32")
            eaf = self._csr.eaf_slice(s, e)
            imp = self._get_imputed_slice(s, e)
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

        O(n_total_associations) scan — acceptable for exploratory use; add a
        variant-centric CSR index (issue deferred) for production phewas.
        """
        variant = self._variant_axis.by_identifier(identifier)
        if variant is None:
            return _empty_result()
        target_vi = np.int32(variant.variant_index)

        offsets = self._csr._offsets[:]
        vi_all = self._csr._variant_index[:]
        z_all = self._csr.z_all()
        se_all = self._csr._se[:]

        hit_positions = np.where(vi_all == target_vi)[0]
        if len(hit_positions) == 0:
            return _empty_result()

        analysis_indices = np.searchsorted(offsets[1:], hit_positions, side="right").astype("int32")
        imp = (
            self._imputed[hit_positions].astype(np.uint8)
            if self._imputed is not None
            else np.zeros(len(hit_positions), dtype=np.uint8)
        )
        if observed_only:
            keep = imp == 0
            hit_positions = hit_positions[keep]
            analysis_indices = analysis_indices[keep]
            imp = imp[keep]

        z_out = z_all[hit_positions].astype("float32")
        se_out = se_all[hit_positions].astype("float32")
        return {
            "variant_index": np.full(len(hit_positions), target_vi, dtype="int32"),
            "analysis_index": analysis_indices,
            "z": z_out,
            "se": se_out,
            "eaf": self._csr.eaf_at(hit_positions),
            "association_status": _status_array(imp, z_out, se_out),
        }

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
        key = threshold_key(threshold)
        root = self.store.arrays(mode="r")
        path = f"top_hits/{key}"
        if path in root:
            group = root[path]
            analysis_index = None
            if analysis_id is not None:
                analysis_index = self._resolve_analysis_id(analysis_id)
                if analysis_index is None or "analysis_offsets" not in group:
                    return _empty_result()
            reader = DenseTopHitReader(group)
            bounds = reader.bounds(analysis_index)
            vi = reader.read("variant_index", bounds, "int32")
            ai = reader.read("analysis_index", bounds, "int32")
            z = reader.read("z", bounds, "float32")
            se = reader.read("se", bounds, "float32")
            imp = (
                reader.read("imputed", bounds, "uint8")
                if "imputed" in group else np.zeros(len(vi), dtype=np.uint8)
            )
            if observed_only:
                keep = imp == 0
                vi, ai, z, se, imp = vi[keep], ai[keep], z[keep], se[keep], imp[keep]
            if limit is not None:
                vi, ai, z, se = vi[:limit], ai[:limit], z[:limit], se[:limit]
                imp = imp[:limit]
            return {
                "variant_index": vi, "analysis_index": ai, "z": z, "se": se,
                "eaf": self._csr.eaf_pairs(vi, ai),
                "association_status": _status_array(imp, z, se),
            }

        # Fallback: full CSR scan. analysis_id is resolved and applied here
        # too (the indexed path resolves it via `bounds()`) so a caller
        # passing analysis_id gets a filtered result on both paths, not an
        # unfiltered store-wide scan on the fallback.
        import math
        analysis_index = None
        if analysis_id is not None:
            analysis_index = self._resolve_analysis_id(analysis_id)
            if analysis_index is None:
                return _empty_result()

        offsets = self._csr._offsets[:]
        vi_all = self._csr._variant_index[:]
        z_all = self._csr.z_all()
        se_all = self._csr._se[:]

        sqrt2 = math.sqrt(2.0)
        lo, hi, mid = 0.0, 40.0, 0.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if math.erfc(mid / sqrt2) > threshold:
                lo = mid
            else:
                hi = mid
        z_thresh = float(mid)
        z_f32 = z_all.astype("float32")
        mask = np.abs(z_f32) >= z_thresh
        if analysis_index is not None:
            start, stop = int(offsets[analysis_index]), int(offsets[analysis_index + 1])
            segment_mask = np.zeros(len(vi_all), dtype=bool)
            segment_mask[start:stop] = True
            mask &= segment_mask
        # CSR segments are analysis-major and variant_index-ascending within
        # each analysis -- a build-time invariant (build_besd.py,
        # build_ssf.py, complete.py all re-sort each analysis's segment by
        # variant_index) -- so np.where(mask) already yields positions in
        # "analysis_index,variant_index" order with no re-sort needed: the
        # same ordering contract the top-hit index itself is built in.
        hit_positions = np.where(mask)[0]

        if len(hit_positions) == 0:
            return _empty_result()

        analysis_indices = np.searchsorted(offsets[1:], hit_positions, side="right").astype("int32")
        imp_all = (
            self._imputed[:].astype(np.uint8) if self._imputed is not None
            else np.zeros(len(vi_all), dtype=np.uint8)
        )
        imp_hits = imp_all[hit_positions]
        if observed_only:
            keep = imp_hits == 0
            hit_positions = hit_positions[keep]
            analysis_indices = analysis_indices[keep]
            imp_hits = imp_hits[keep]
        if limit is not None:
            hit_positions = hit_positions[:limit]
            analysis_indices = analysis_indices[:limit]
            imp_hits = imp_hits[:limit]

        z_out = z_f32[hit_positions]
        se_out = se_all[hit_positions].astype("float32")
        return {
            "variant_index": vi_all[hit_positions].astype("int32"),
            "analysis_index": analysis_indices,
            "z": z_out,
            "se": se_out,
            "eaf": self._csr.eaf_at(hit_positions),
            "association_status": _status_array(imp_hits, z_out, se_out),
        }

    def lookup(
        self,
        identifiers: list[str],
        analysis_ids: list[str],
        *,
        observed_only: bool = False,
    ) -> dict[str, np.ndarray]:
        """Associations for a specific variant × analysis set."""
        variants = [
            v
            for id_ in identifiers
            if (v := self._variant_axis.by_identifier(id_)) is not None
        ]
        if not variants:
            return _empty_result()

        target_vi = {v.variant_index for v in variants}
        all_vi, all_ai, all_z, all_se, all_eaf, all_status = [], [], [], [], [], []

        for aid in analysis_ids:
            idx = self._resolve_analysis_id(aid)
            if idx is None:
                continue
            offsets_pair = self._csr._offsets[idx: idx + 2]
            s, e = int(offsets_pair[0]), int(offsets_pair[1])
            if s == e:
                continue
            vi = self._csr._variant_index[s:e].astype("int32")
            z = self._csr.z_slice(s, e)
            se = self._csr._se[s:e].astype("float32")
            eaf = self._csr.eaf_slice(s, e)
            imp = self._get_imputed_slice(s, e)
            sub_mask = np.isin(vi, np.array(sorted(target_vi), dtype=np.int32))
            if not sub_mask.any():
                continue
            vi, z, se, eaf, imp = (
                vi[sub_mask], z[sub_mask], se[sub_mask], eaf[sub_mask], imp[sub_mask]
            )
            if observed_only:
                keep = imp == 0
                vi, z, se, eaf, imp = vi[keep], z[keep], se[keep], eaf[keep], imp[keep]
            if len(z) == 0:
                continue
            all_vi.append(vi)
            all_ai.append(np.full(len(z), idx, dtype="int32"))
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

    def _shared_is_on_panel(self, shared_idx: int) -> bool:
        pos = int(np.searchsorted(self._dense_to_shared, shared_idx))
        return pos < len(self._dense_to_shared) and int(self._dense_to_shared[pos]) == shared_idx

    def _overflow_for_analysis(self, col: int) -> dict[str, np.ndarray]:
        offsets = self._csr._offsets[col: col + 2]
        s, e = int(offsets[0]), int(offsets[1])
        if s == e:
            return _empty_result()
        vi = self._csr._variant_index[s:e].astype("int32")
        z = self._csr.z_slice(s, e)
        se = self._csr._se[s:e].astype("float32")
        return {
            "variant_index": vi,
            "analysis_index": np.full(len(z), col, dtype="int32"),
            "z": z,
            "se": se,
            "eaf": self._csr.eaf_slice(s, e),
            "association_status": _status_array(np.zeros(len(z), dtype=np.uint8), z, se),
        }

    def _overflow_by_variants(self, shared_indices: set[int]) -> dict[str, np.ndarray]:
        """All overflow associations whose (off-panel) variant is in the set."""
        if not shared_indices:
            return _empty_result()
        offsets = self._csr._offsets[:]
        vi_all = self._csr._variant_index[:]
        wanted = np.fromiter(shared_indices, dtype=np.int32, count=len(shared_indices))
        mask = np.isin(vi_all, wanted)
        hits = np.where(mask)[0]
        if len(hits) == 0:
            return _empty_result()
        analysis_indices = np.searchsorted(offsets[1:], hits, side="right").astype("int32")
        z = self._csr.z_at(hits)
        se = self._csr._se[:][hits].astype("float32")
        eaf = self._csr.eaf_at(hits)
        return {
            "variant_index": vi_all[hits].astype("int32"),
            "analysis_index": analysis_indices,
            "z": z,
            "se": se,
            "eaf": eaf,
            "association_status": _status_array(np.zeros(len(hits), dtype=np.uint8), z, se),
        }

    # ── public query surface ─────────────────────────────────────────────────
    def analysis(self, analysis_id: str, *, observed_only: bool = False) -> dict[str, np.ndarray]:
        dense = self._remap_dense(self._dense.analysis(analysis_id, observed_only=observed_only))
        analysis = self._analyses.by_id(analysis_id)
        overflow = (
            self._overflow_for_analysis(int(analysis["analysis_index"]))
            if analysis is not None else _empty_result()
        )
        return _concat_results([dense, overflow])

    def phewas(self, identifier: str, *, observed_only: bool = False) -> dict[str, np.ndarray]:
        variant = self._variant_axis.by_identifier(identifier)
        if variant is None:
            return _empty_result()
        if self._shared_is_on_panel(variant.variant_index):
            return self._remap_dense(
                self._dense.phewas(variant.alid, observed_only=observed_only)
            )
        return self._overflow_by_variants({int(variant.variant_index)})

    def range_phewas(
        self, chromosome: str, start: int, end: int, *, observed_only: bool = False
    ) -> dict[str, np.ndarray]:
        dense = self._remap_dense(
            self._dense.range_phewas(chromosome, start, end, observed_only=observed_only)
        )
        shared_idx = self._variant_axis.range_indices(chromosome, start, end)
        off_panel = {
            int(i) for i in shared_idx.tolist() if not self._shared_is_on_panel(int(i))
        }
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
        off_shared: set[int] = set()
        for id_ in identifiers:
            rec = self._variant_axis.by_identifier(id_)
            if rec is not None and not self._shared_is_on_panel(rec.variant_index):
                off_shared.add(int(rec.variant_index))
        wanted_cols = {
            int(a["analysis_index"])
            for aid in analysis_ids
            if (a := self._analyses.by_id(aid)) is not None
        }
        overflow = self._overflow_by_variants(off_shared)
        if len(overflow["z"]) and wanted_cols:
            keep = np.isin(overflow["analysis_index"], np.fromiter(
                wanted_cols, dtype="int32", count=len(wanted_cols)))
            overflow = {k: v[keep] for k, v in overflow.items()}
        elif not wanted_cols:
            overflow = _empty_result()
        return _concat_results([dense, overflow])

    def _overflow_top_hits(
        self, threshold: float, analysis_index: int | None = None
    ) -> dict[str, np.ndarray]:
        key = threshold_key(threshold)
        root = self.store.arrays(mode="r")
        path = f"top_hits/{key}"
        if path not in root:
            return _empty_result()
        group = root[path]
        if analysis_index is not None and "analysis_offsets" not in group:
            return _empty_result()
        reader = DenseTopHitReader(group)
        bounds = reader.bounds(analysis_index)
        vi = reader.read("variant_index", bounds, "int32")
        ai = reader.read("analysis_index", bounds, "int32")
        z = reader.read("z", bounds, "float32")
        se = reader.read("se", bounds, "float32")
        return {
            "variant_index": vi,
            "analysis_index": ai,
            "z": z,
            "se": se,
            "eaf": self._csr.eaf_pairs(vi, ai),
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
        dense = self._remap_dense(self._dense.top_hits(
            analysis_id=analysis_id, threshold=threshold, observed_only=observed_only
        ))
        overflow = self._overflow_top_hits(
            threshold, analysis_index
        )  # overflow is always observed
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
