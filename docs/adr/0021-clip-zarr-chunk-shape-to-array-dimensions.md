# ADR-0021: Clip zarr chunk shape to actual array dimensions at write time

> **Superseded by [ADR 0058](./0058-dense-chunk-and-shard-shapes.md) as the
> statement of a Dense grid's chunk.** The clipping rule below is unchanged and
> still holds; what changed is the hint it clips. `DEFAULT_CHUNK_SHAPE =
> (1000, 1000)` is no longer the Analysis-axis width a Dense plane is written
> with: ADR 0058 (#246) decided `[1000, 64]` inner chunks with a
> `[100000, 1024]` shard. Read ADR 0058 for the shapes and the measurements
> behind them.

## Status

Accepted

## Context

The dense store builds in `build.py` and `build_vcf.py` used a fixed
`DEFAULT_CHUNK_SHAPE = (1000, 1000)` for both row and column axes.  When the
number of analyses is smaller than 1000 (e.g. the chr1 × 100-analysis UKB
dataset), the declared chunk shape exceeded the array's column dimension.

zarr v2 stores the declared chunk shape in `.zarray` metadata and allocates a
decompression buffer of that declared size on every chunk read.  Profiling
showed:

| Array | Declared chunks | Effective cols | Buffer per chunk | Chunks for full col |
|---|---|---|---|---|
| besdq  | (1000, 100) | 100 | 200 KB | 763 |
| ours (old) | (1000, 1000) | 100 | **2 MB** | 763 |
| ours (new) | (1000, 100) | 100 | 200 KB | 763 |

The mismatch caused zarr to allocate 10× more memory per chunk read despite the
physical chunk files being identical in size (mean 167 KB compressed).  This
produced a 3× wall-clock slowdown on single-column reads (`[:, col]`) used by
the `analysis` (bulk) and `regional` query patterns.

Direct zarr timing on the chr1 store confirmed the hypothesis:

| Operation | Before fix | After fix | besdq baseline |
|---|---|---|---|
| `z[:, col]` raw zarr read | 1403 ms | ~420 ms (rechunk test) | 462 ms |
| full `analysis()` (bulk) query | 1764 ms | **824 ms** | 490 ms |
| `range()` regional query | 98 ms | **78 ms** | 5 ms |
| `phewas()` query | 2.65 ms | **1.42 ms** | 0.9 ms |
| `top_hits()` query | 1.30 ms | **1.13 ms** | 0.7 ms |
| `lookup()` random query | 224 ms | **114 ms** | 30 ms |

The remaining 1.68× gap on bulk vs besdq is expected: besdq reads one array
(beta), opengwasdb reads two (z+se) then applies an `np.isfinite` mask.
Reading two columns costs ~2× the single-column time; the mask overhead is
negligible (measured at < 3%).  For equivalent work (two columns) opengwasdb
matches or slightly beats besdq.

## Decision

In `_write_zarr` (both `build.py` and `build_vcf.py`), clip the declared chunk
shape to the actual array dimensions before calling `zarr.create_dataset`:

```python
effective_chunks = (min(chunk_shape[0], z.shape[0]), min(chunk_shape[1], z.shape[1]))
```

`DEFAULT_CHUNK_SHAPE = (1000, 1000)` is kept as the *maximum* hint.  For arrays
with fewer analyses the effective chunk covers all columns in one tile, matching
besdq's layout.  For arrays with more than 1000 analyses the column chunks are
naturally bounded by the declared value.

## Alternatives considered

**Change DEFAULT_CHUNK_SHAPE to (1000, 100)**: Only correct for 100-analysis
stores; would produce wrong chunk layout for stores with different analysis
counts.

**Dynamic chunk shape = (1000, n_analyses)**: Equivalent to the chosen approach
for small n_analyses, but requires threading n_analyses through the call stack.
Clipping at write time is simpler and achieves the same result.

## Consequences

- New stores are written with the correct chunk shape and read ~3× faster on
  column-oriented queries.
- Existing stores (already built) retain the oversized chunk declaration and
  remain slower until rebuilt.  No migration is performed automatically.
- The `chunk_shape` stored in zarr root attrs reflects the effective (clipped)
  value, not the original hint.
- The test `test_dense_build_writes_standard_envelope_and_metadata` was updated
  to assert `chunks == (3, 2)` for a 3-variant × 2-analysis fixture, confirming
  the clip behaviour.
