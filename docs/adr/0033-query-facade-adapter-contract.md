# Query facade stays three adapters behind one documented contract

Supersedes nothing; amends ADR-0006 the same way ADR-0020 does.

## Context

ADR-0006 promised a layout-independent query engine: callers query a Store
Release without knowing whether it is Dense, Ragged, or Hybrid. In practice
`StoreQuery`, `RaggedStoreQuery`, and `HybridStoreQuery` (`opengwasdb/query/facade.py`)
had drifted apart in ways the ADR never made explicit:

- `variants_table()`/`analyses_table()` and `__enter__`/`__exit__` existed on
  some adapters and not others, with no documented reason.
- Ragged `top_hits()`'s full-scan fallback returned descending-|z| order and
  ignored `analysis_id`, while the indexed fast path (and Dense/Hybrid
  `top_hits()`) return genomic order; the fallback also applied `limit`
  before `observed_only` where every other path applies it after.
- Dense's point-query methods (`analysis()`, `phewas()`, `range_phewas()`,
  `lookup()`) silently drop non-finite `(z, se)` cells; Ragged never filters
  for finiteness and instead labels non-finite cells
  `association_status="missing"`. Same result shape, undocumented difference
  in what counts as a result.

Issue #51 asked whether the fix is to collapse the three classes into one
adapter interface with capability detection pushed behind the boundary, or to
pin the contract down and keep the three-class split.

## Decision

**Keep the three adapter classes**, selected by `query_store()` /
`OpenGWASDBStore.query()` from the store manifest, and make the contract each
one implements explicit rather than implicit:

1. **Result shape** is identical across all three: `dict[str, np.ndarray]`
   with `variant_index` (int32), `analysis_index` (int32), `z` (float32),
   `se` (float32), `association_status` (object) as parallel arrays
   (ADR-0020).

2. **`top_hits()` ordering is genomic order** — sorted by
   `(analysis_index, variant_index)` — on every adapter and every internal
   path, matching the `group.attrs["order"]` the top-hit index itself is
   built with (`opengwasdb/top_hits.py`, which every layout's builder writes
   through; it was `layouts/dense/top_hits.py` and `layouts/ragged/top_hits.py`
   until issue #53).
   The Ragged full-scan fallback previously sorted by descending `|z|` and
   ignored `analysis_id`; both are fixed in `facade.py` so the same call
   returns the same shape of answer regardless of which physical path served
   it. CSR segments are analysis-major and variant-index-ascending within
   each analysis by construction (`build_besd.py`, `build_ssf.py`,
   `complete.py` all re-sort each analysis's segment), so the fallback needs
   no explicit re-sort to match this order — it only needed to stop
   re-sorting by significance.

3. **`observed_only` is applied before `limit`**, everywhere `limit` is
   accepted: `limit` caps the already-filtered result, never the candidate
   set. The Ragged fallback previously did the opposite; fixed alongside (2).

4. **`variants_table()`/`analyses_table()` and context-manager support**
   (`__enter__`/`__exit__`) are present on all three adapters. Previously
   `RaggedStoreQuery` had neither — there was no layout reason for the gap,
   just an implementation gap, so both are added. `analyses_table()`'s row
   shape still differs by layout: Dense/Hybrid rows carry the
   phenotype/effect-scale fields `analyses.tsv` stores; Ragged rows carry
   the molecular-QTL fields (probe/gene, tissue, context) the `analyses`
   SQLite table stores, because a Ragged Analysis is a QTL probe, not a
   GWAS phenotype — `analysis_id` is the one field both shapes share. This
   is a genuine metadata-schema difference, not a facade inconsistency.

5. **`rho()`/`rho_row()`/`rho_matrix()`** (ADR-0025) are storage artifacts of
   a Dense Store Release. They are exposed on `StoreQuery` directly and on
   `HybridStoreQuery` by delegating to its Dense Component (itself a
   self-contained Dense Store Release); they return the same empty result a
   Dense store with no Rho Matrix returns when the Dense Component has none.
   `RaggedStoreQuery` has no Rho Matrix format and does not expose these
   methods — a genuine capability difference, not an inconsistency.

6. **`range_by_analysis()` stays Ragged-only.** It queries by probe/TSS
   position via `TraitsAxisReader`, which only Ragged/molecular-QTL releases
   populate. Dense and Hybrid releases have no probe/TSS axis to query by.

7. **Finiteness handling stays intentionally different for the point-query
   methods (`analysis()`, `phewas()`, `range_phewas()`, `lookup()`), and is
   now documented in `facade.py`'s module docstring:** `StoreQuery` drops
   non-finite `(z, se)` cells outright — they never appear in results, not
   even as `"missing"` — because the Dense grid is mostly-empty by
   construction (most variant×analysis cells were never tested) and
   returning every such cell would defeat the sparse-array contract
   (ADR-0020) that motivated the array API in the first place.
   `RaggedStoreQuery` never filters for finiteness: a CSR entry exists only
   for a pair someone attempted (observed, or a completion attempt per
   ADR-0013/ADR-0022), so a non-finite entry is already a small, deliberate
   set, and surfacing it as `association_status="missing"` costs nothing
   extra. Callers who need Dense's aggregate missing counts already have
   `completion_n_missing_total` (`analyses.tsv`, ADR-0011) without paying for
   a full-grid scan. `HybridStoreQuery` does not have one uniform behaviour
   here: on-panel results are delegated to its Dense Component and inherit
   Dense's drop-non-finite behaviour; off-panel (Ragged Overflow) results are
   read the same unfiltered way `RaggedStoreQuery` reads its CSR, though the
   Overflow Component is documented as always-observed (ADR-0026), so a
   non-finite overflow cell would be an anomaly rather than an expected
   outcome.

8. **`top_hits()` is not covered by the point-query finiteness contract, on
   any adapter.** Candidacy is decided at build time by
   `|z| >= z_critical(threshold)` (`layouts/*/top_hits.py`), which excludes
   NaN `z` (a NaN comparison is always false) but does not itself guarantee a
   finite paired `se` — no `isfinite(se)` filter is applied at query time.
   This was already true before this change; it is recorded here because the
   point-query finiteness rule in (7) does not extend to it.

## Alternatives considered

**One generic `StoreQuery` class with internal layout branching.** Rejected:
this is exactly the shape ADR-0006 already rejected — it pushes layout
selection down into every method body instead of behind an adapter boundary.
The twelve `"imputed" in root` / `"rho" in root` / `"analysis_offsets" not in
group` branches issue #51 flagged are capability checks *within* a layout
(does this particular store instance have this optional artifact), not
layout dispatch, and stay where they are.

**A formal typed capability `Protocol`/ABC callers introspect at runtime.**
Rejected for now: there are only three adapters and one caller surface
(`query_store()`); the method-availability differences above (5 and 6) are
few enough to document rather than encode as a runtime-checkable type.
Revisit if a fourth layout, or a caller outside this repo needing runtime
capability discovery, appears.

## Consequences

- `opengwasdb/query/facade.py` carries a module docstring stating the four
  contract axes (result shape, ordering, observed_only/limit, finiteness) so
  a caller can predict method availability and semantics without knowing the
  physical layout, other than the two documented capability differences
  (Rho, `range_by_analysis`).
- Ragged `top_hits()`'s full-scan fallback now matches the indexed path's
  contract for the same call, verified by a test that forces both paths.
- `RaggedStoreQuery` gained `variants_table()`, `analyses_table()`,
  `__enter__`/`__exit__`; `HybridStoreQuery` gained `rho()`, `rho_row()`,
  `rho_matrix()`. No existing method's behaviour changed for callers already
  relying on it, except the Ragged `top_hits()` full-scan fallback, which was
  producing an inconsistent answer.
