# ADR-0020: Sparse flat array query contract

## Status

Accepted

## Context

The chr1 × 100-analysis benchmark (see `docs/benchmark-output/opengwasdb_vcf_ukb_chr1_benchmark.json`)
showed the following query slowdowns vs the besdq zarr prototype:

| Pattern       | Ratio vs besdq |
|---------------|----------------|
| regional      | 561×           |
| phewas        | 6×             |
| tophits       | 234×           |
| bulk          | 28×            |
| random_lookup | 12×            |

Root-cause analysis confirmed that zarr reads were fast; the bottleneck was
constructing O(result_count) Python `AssociationResult` objects per query.
The bulk pattern materialised 762 080 objects in one call, taking ~14 minutes.

The existing `analysis_arrays()` companion method demonstrated that returning
raw numpy arrays avoided the bottleneck entirely, but it was not a complete
replacement: it returned a dense column array with NaN for absent cells, and it
had no equivalent for the range, phewas, lookup, or top-hits patterns.

## Decision

Replace the row-materialisation query API with a **sparse flat array** contract.
All five public methods (`analysis`, `phewas`, `range`, `lookup`, `top_hits`)
return a `dict[str, np.ndarray]` with four parallel arrays of length *k*,
where *k* is the number of finite cells in the result set:

```python
{
    "variant_index": np.ndarray,  # int32
    "analysis_index": np.ndarray, # int32
    "z": np.ndarray,              # float32
    "se": np.ndarray,             # float32
}
```

This format is the prior art already used by the internal top-hit index
(written by `opengwasdb/top_hits.py`; it was `layouts/dense/top_hits.py` until
issue #53), so it is a natural fit.

Two metadata accessors are added:
- `variants_table() → dict[int, dict]` keyed by `variant_index`
- `analyses_table() → dict[int, dict]` keyed by `analysis_index`

Callers who need derived fields (`beta`, `p_value`) compute them from `z` and
`se` using numpy vectorised operations rather than per-row Python arithmetic.

The following items are **removed**:
- `AssociationResult` frozen dataclass (entire `opengwasdb/query/results.py`)
- `analysis_arrays()` companion method (superseded by `analysis()`)
- `variant()` method (alias of `phewas()`, removed for API clarity)
- `range(analysis_id=...)` filter parameter (caller can filter by index)

This amends the facade contract defined in **ADR-0006** (layout-independent
query engine). The layout-independence guarantee is preserved: `StoreQuery`
continues to hide physical layout details behind a uniform interface.

## Alternatives considered

**Keep row methods alongside array methods**: Two APIs for the same data with no
clear ownership boundary. Rejected because there are no external callers at this
stage, so the cost of a clean break is zero.

**Deprecation wrappers returning `AssociationResult`**: Adds code complexity
and defers the performance benefit. Rejected for the same reason.

**Dense matrix return** (the `analysis_arrays()` approach extended to all
methods): Requires the caller to unpack NaN-padded arrays. Sparse arrays are
more memory-efficient for queries that return a small fraction of the full matrix
(e.g. range and phewas), and are the format already used by the top-hit index.

## Consequences

- All callers must be updated (CLI, tests, benchmarks).
- No external callers exist at v0.1, so migration cost is zero.
- Downstream code that needs human-readable variant/analysis metadata calls
  `variants_table()` and `analyses_table()` once and joins by index.
- The performance improvement is expected to be largest for `bulk` and `regional`
  queries, where result counts are in the hundreds of thousands.
