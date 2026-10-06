# A variant-centric index for the Ragged and Hybrid Overflow components

Ticket #252, epic #240. The format-free fixes of #252 (step 3) bound the
variant-side scans' memory and stop them decoding whole planes, and make
`lookup` proportional to the request. They do **not** make off-axis PheWAS or
region queries proportional to the answer: without a variant-side index, those
must read every Analysis's `variant_index` to find the rows. This ADR decides
the index's on-disk shape, compatibly with ADR 0057 (format 0.2.0) and ADR 0058
(the decided chunk and shard shapes). **Building it is separate work** (#252
step 5); the epic can close without it, and a release that carries no index
queries correctly through the scan it replaces.

## Context

The Ragged Layout is analysis-major: `offsets` makes a per-Analysis read O(1),
but a per-variant question has no index. On OGS-00011's Hybrid Overflow
(3,085,080,783 associations over 150.6 M off-axis variants, 3,317 Analyses) the
measured cost of the variant-side shapes is in #252's table: off-axis PheWAS
17.1 s / 25.2 GB, a 1 Mb region 630 s / 27.8 GB, a 10 × 100 lookup 38.6 s /
50.2 GB. The scan is behaving as written; the store is 25× the eQTLGen store
whose PheWAS measured 688 ms.

#252 step 3 replaced the whole-plane decodes with at-position reads in
chunk-sized windows, vectorised the on-axis test, and made `lookup` binary-
search each requested Analysis's segment. Those changes make the Overflow's
columns readable from the chunks a hit touches, but they leave the two shapes
that ask "every Analysis at this variant" or "every Analysis in this region"
reading the whole `variant_index` (12.34 GB at OGS-00011) in windows. Time
stays O(N); #250's OGS-00011 benchmark would be dominated by it.

The Overflow's rows are sorted by `variant_index` within each Analysis (every
builder argsorts, now asserted in `RaggedCSRWriter.add_analysis`), so a
per-variant question *can* binary-search an Analysis's segment. That bounds a
lookup, but not a scan over every Analysis: the number of segment searches is
the number of Analyses, each costing O(log rows) chunk reads.

## Decision

**A `by_variant/` CSR duplicate** inside each Ragged component (a pure Ragged
store's `data.zarr/ragged/`, and a Hybrid's Overflow `data.zarr/ragged/`). It
holds one row per existing association, ordered by `(variant_index,
analysis_index)`, so a variant's rows are contiguous and a variant range's rows
are contiguous. **No Analysis-sorted array moves**: the duplicate sits beside
them and is written by the same builders that write them.

### Arrays and roles

| array | dtype | contents | role (ADR 0058) |
|---|---|---|---|
| `by_variant/offsets` | int64 | `n_off_axis_variants + 1` row offsets, shared axis | `PER_VARIANT`, `component_chunk=1000` |
| `by_variant/analysis_index` | int32 | the Analysis of each row | `ASSOCIATION_SEQUENCE` |
| `by_variant/z` | the component's `z` dtype (int16 at OGS-00011) | the same codes, re-keyed | `ASSOCIATION_SEQUENCE` |
| `by_variant/z_overflow_index` / `_value` | int64 / float32 | the re-keyed `z` overflow table | `EXCEPTION_TABLE` |
| `by_variant/se` | the component's `se` dtype (float16) | the same codes | `ASSOCIATION_SEQUENCE` |
| `by_variant/eaf` | the component's `eaf` dtype (int8 residual) | the same codes, re-keyed; the per-variant `eaf_baseline` is **shared**, not duplicated | `ASSOCIATION_SEQUENCE` |
| `by_variant/eaf_exception_index` / `_value` | int64 / float32 | the re-keyed EAF exception table | `EXCEPTION_TABLE` |
| `by_variant/se_exception_*`, `se_coefficients` | as the Analysis-sorted side | shared where the plan's keys are per Analysis or per variant; re-keyed where the key is the flat position | `EXCEPTION_TABLE` / `SE_COEFFICIENTS` |

Every array takes its layout from the existing role table; the index adds **no
new role and no new shard policy**. It therefore satisfies ADR 0058's
constraint to #252 ("it must survive the same `shard_layout` role table rather
than choosing its own") and, because the Analysis-sorted arrays are untouched,
never re-shards them. The per-cell columns are `ASSOCIATION_SEQUENCE` (inner
chunk 200,000, shard ~1,000,000 elements); the per-variant offsets are
`PER_VARIANT` with the component plane's variant-axis chunk, no coarser than
1,000, so one per-variant lookup reads one 8 KB inner chunk; the exception and
overflow tables are `EXCEPTION_TABLE` (200,000, one shard). A 0.2.0 conversion
writes each role through `shard_layout` exactly as #245's converter does today.

`analysis_index` and not `variant_index` is the per-row key: the row's variant
is implied by the offsets, the Analysis is not.

### Ordering and the query contract

Rows within a variant are ascending by `analysis_index`, so a PheWAS result is
in the same Analysis-ascending order the current scan returns; a region query
returns variant-ascending, Analysis-ascending within each. The query facade
keeps its documented "no ordering guarantee beyond grouping" contract; the
index is a faster route to the same grouping, and the parity tests in
`tests/test_variant_side_scans.py` can be pointed at an indexed fixture to pin
that.

### Where it is read

- `RaggedStoreQuery.phewas` and `HybridStoreQuery.phewas` (off-axis): read
  `offsets[v]`, then the variant's row block.
- `RaggedStoreQuery.range_phewas` and `HybridStoreQuery.range_phewas`:
  `offsets[lo:hi]` over the variant range (in `variant_index` order — the
  shared axis is sorted), then the rows of the variants the range selects.
- `RaggedStoreQuery.lookup` / `HybridStoreQuery.lookup`: unchanged; step 3's
  per-Analysis segment search is already proportional to the request, and it
  works on an unindexed release.
- `top_hits` and the Analysis-side reads are unchanged.

An absent index is not an error: the facade falls back to the step-3 scan. A
release therefore remains readable and correct without the index, and the index
is optional build output rather than a `format_version` change. (Whether the
index's *presence* is recorded in the manifest is left to the builder ticket;
it is a provenance fact, not a new format.)

### Storage at OGS-00011 scale

From the Overflow's own array metadata (measured, `data.zarr/ragged`):

| array | shape | dtype | raw |
|---|---:|---|---:|
| `variant_index` | 3,085,080,783 | int32 | 12.34 GB |
| `z` | 3,085,080,783 | int16 | 6.17 GB |
| `se` | 3,085,080,783 | float16 | 6.17 GB |
| `eaf` | 3,085,080,783 | int8 | 3.085 GB |
| `eaf_baseline` | 164,051,296 | float32 | 0.656 GB |
| `eaf_exception_index` / `_value` | 180,396,687 | int64 / float32 | 2.17 GB |

The Overflow is **18 GB on disk** for ~30.6 GB of values, a 1.7× compression.
The duplicate drops `variant_index` per cell and stores `analysis_index`
instead; it copies `z`, `se` and `eaf` and shares the per-variant
`eaf_baseline`. Its raw size is therefore about
12.34 + 6.17 + 6.17 + 3.085 (the four per-cell columns)
+ 1.21 (`by_variant/offsets`) + 2.17 (the re-keyed EAF exception table)
+ a small re-keyed `z` overflow table ≈ **31.2 GB**, and at the Overflow's own
measured 1.7× it is **about +18 GB on disk — close to doubling the 18 GB
Overflow**, matching #252's estimate of "about +19 GB here".

### Build cost

No comparison sort is needed. The Overflow is analysis-major, so the build is a
counting sort by variant: one pass to count rows per variant
(`np.bincount(variant_index, minlength=n_off_axis_variants)`, 164 M int64 =
1.3 GB), a prefix sum to `offsets`, and one pass to scatter each cell into its
variant's block while copying the codes and re-keying the overflow/exception
tables. Two passes over ~30 GB of values, single-threaded zarr decode at the
rate #244 measured for these arrays; **estimated 30–60 min** on this node,
parallelisable by variant band because the destination offsets are known after
the counting pass. The streamed Overflow writers #228/#233 established
(`write_eaf_plane`/`flush_se`, region by region) are the model: the index is
written in the same bounded regions, in the same phases, so the build's peak is
unchanged.

### Query cost

At OGS-00011, a per-variant PheWAS reads one `by_variant/offsets` inner chunk
(8 KB) and the variant's row block: at 3,317 Analyses × ~9 bytes a cell that is
about 30 KB, one inner chunk of each of `analysis_index`, `z`, `se` and `eaf`.
That is five or so chunk reads against Dense PheWAS's measured 30.7 ms. A 1 Mb
region reads `offsets[lo:hi]` and the range's contiguous blocks, so its cost is
the answer's size (the TCF7L2 window held 8.3 M rows ≈ 75 MB of values) rather
than the store's. Both meet the acceptance target: off-axis PheWAS within a
small factor of its Dense equivalent, and a 1 Mb region well under a second.

## Considered options

- **A permutation index** (per-variant `offsets` plus int32 positions into the
  existing Analysis-sorted arrays). Cheaper on disk (offsets 1.21 GB + 12.34 GB
  of positions ≈ 13.5 GB, no statistic duplicated) and it needs no statistic
  copy, but its reads are scattered: answering one variant's PheWAS gathers its
  Analyses' rows from across the whole CSR. With 3,317 Analyses spread over the
  Overflow's ~15,425 `z` chunks, the expected number of distinct chunks touched
  is ~3,000 — about 19% of the plane per query, and the same again for `se` and
  `eaf`. That is the whole-plane read #252 step 3 removed, reintroduced as a
  gather. A region query of 8.3 M rows is worse still. Rejected.
- **An analysis-list index only** (per-variant "which Analyses hold this
  variant", no statistics). It answers "is this variant in the store and
  where", and `lookup`-style segment search then resolves the statistics. But
  the number of segment searches is the number of Analyses holding the variant
  (`n_analyses_hit × O(log rows)` chunk reads), which for a pleiotropic variant
  at 3,317 Analyses is thousands of chunk reads and hundreds of MB. It does not
  meet "proportional to the answer, within a small factor of Dense". Rejected.
- **A delta- or varint-encoded `analysis_index`** to shrink the duplicate's
  dominant column. Worth measuring, but it is a change to the index's *codes*,
  not its shape, and it can land later behind the same role policy. Deferred,
  not rejected.
- **Extending the Dense Component's `eaf_baseline`/variant axis to carry the
  off-axis variants** (a dense-of-union axis). ADR 0026 rejected this for the
  component partition, and it would make the un-imputable tail dense; the same
  reasoning applies to the index. Rejected.
- **Do nothing; rely on `lookup`'s segment search.** Rejected: it cannot make a
  whole-store PheWAS or region query sublinear in N, which is the acceptance
  criterion the numbers above fail.

## Consequences

- **The index is optional and additive.** No Analysis-sorted array changes, no
  `format_version` changes, an unindexed release still answers every query
  (through the step-3 scan), and a 0.1.0 or 0.2.0 release can gain or lose it
  by a build, not a conversion.
- **A `by_variant/` group roughly doubles the Overflow's bytes on disk** at
  OGS-00011 (~+18 GB). That is the price of making per-variant work
  proportional to the answer; the numbers are above so an operator can decide
  per release.
- **The Hybrid top-hit index and the streamed Overflow writers are affected**:
  the builder writes the duplicate in the same phases and the same bounded
  regions as the Analysis-sorted arrays (ADR 0026, issues #228/#233), and the
  Hybrid's `dense_to_shared` map is unchanged because the index is keyed on the
  shared axis.
- **Building and wiring it is #252 step 5, out of scope for #252 and not
  required to finish epic #240.** It is separate work: a builder phase and a
  query-facade route with its own tests, parity-checked against the step-3 scan
  on an indexed fixture and on OGS-00011.
- **ADR 0058's role table is not extended.** If a future measurement shows the
  `PER_VARIANT` offsets chunk too fine or too coarse, that is an ADR 0058
  amendment, not a private choice made here.

## References

- The ticket: opengwas/opengwasdb#252 (steps 3 and 4); the epic: #240.
- ADR 0026 (Hybrid Layout), ADR 0037 (statistic encodings), ADR 0057 (format
  0.2.0), ADR 0058 (Dense chunk and shard shapes, and the constraint to #252).
- The format-free fixes whose scans this index replaces: #252 step 3,
  `tests/test_variant_side_scans.py`.
- The Overflow's array metadata and on-disk size: OGS-00011
  `store.opengwasdb/data.zarr/ragged`.
