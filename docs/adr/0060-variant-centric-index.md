# A variant-centric index for the Ragged and Hybrid Overflow components

Ticket #252, epic #240. The format-free fixes of #252 (step 3) bound the
variant-side scans' memory, stop them decoding whole planes, and make `lookup`
binary-search each requested Analysis's sorted segment. They do **not** make
off-axis PheWAS or region queries proportional to the answer: without a
variant-side index, those must read every Analysis's `variant_index` to find
the rows. This ADR decides the index's on-disk shape, compatibly with ADR 0057
(format 0.2.0), ADR 0058 (the decided Dense chunk and shard shapes) and #248's
ADR 0059 (the Ragged roles and shard policies). **It was built and measured
as #252 step 5**; the storage, build and query numbers in the sections below
are from the committed artifacts, not estimates. A release that carries no
index still queries correctly through the scan it replaces.

## Context

The Ragged Layout is analysis-major: `offsets` makes a per-Analysis read O(1),
but a per-variant question has no index. On OGS-00011's Hybrid Overflow
(3,085,080,783 associations over 150.6 M off-axis variants, 3,317 Analyses) the
measured cost of the variant-side shapes is in #252's artifact
(`docs/benchmark-output/opengwasdb_ogs00011_hybrid_252_ab.json`): off-axis
PheWAS 26.4 s / 2.5 GB, a 1 Mb region 107.9 s / 5.4 GB, the two random lookups
1.6–1.9 s / 2.5–4.5 GB. #252 step 3 replaced the whole-plane decodes with
at-position reads in windows, vectorised the on-axis test, and made `lookup`
binary-search each requested Analysis's segment; those changes bound memory and
make a lookup proportional to the request, but the two shapes that ask "every
Analysis at this variant" or "every Analysis in this region" still read the
whole `variant_index` (12.34 GB at OGS-00011) in windows. Time stays O(N).

The Overflow's rows are sorted by `variant_index` within each Analysis (every
builder argsorts, now asserted in `RaggedCSRWriter.add_analysis`), so a
per-variant question *can* binary-search an Analysis's segment. That bounds a
lookup, but not a scan over every Analysis: the number of segment searches is
the number of Analyses, each costing O(log rows) chunk reads.

## Decision

**A `by_variant/` CSR duplicate** inside each Ragged component — a pure Ragged
store's `data.zarr/ragged/by_variant/`, and a Hybrid's Overflow
`data.zarr/ragged/by_variant/`. It holds one row per existing association,
ordered by `(variant_index, analysis_index)`, so a variant's rows are contiguous
and a variant range's rows are contiguous. **No Analysis-sorted array moves**:
the duplicate sits beside them and is written by the same builders that write
them.

### Arrays, roles and shards

Every array takes its layout from the authoritative role table in
`opengwasdb/store/arrays.py`; **the index adds no role and no shard policy**.
The roles are #248's Ragged ones (ADR 0059), so the index shards with the
Ragged arrays and not with #246's Dense shapes.

| array | dtype | contents | role (ADR 0059) | inner chunk | shard |
|---|---|---|---|---|---|
| `by_variant/offsets` | int64 | `n_axis + 1` row offsets, see below | `RAGGED_PER_VARIANT` | **1,000** (explicit hint) | 10,000,000 elements |
| `by_variant/analysis_index` | int32 | the Analysis of each row | `ASSOCIATION_SEQUENCE` | 200,000 | 50,000,000 elements |
| `by_variant/z` | the component's `z` dtype (int16 at OGS-00011; `int16` or `float16` per plan) | the same codes, re-keyed | `ASSOCIATION_SEQUENCE` | 200,000 | 50,000,000 elements |
| `by_variant/se` | the component's `se` dtype (float16 at OGS-00011; `float16` or `int8_residual` per plan) | the same codes | `ASSOCIATION_SEQUENCE` | 200,000 | 50,000,000 elements |
| `by_variant/eaf` | the component's `eaf` dtype (int8 residual at OGS-00011; `absent`, `float32` or `int8_residual` per plan) | the same codes, re-keyed; the per-variant `eaf_baseline` is **shared** with the Analysis-sorted plane, not duplicated | `ASSOCIATION_SEQUENCE` | 200,000 | 50,000,000 elements |
| `by_variant/imputed` | uint8 | the imputed mask, **when the component has one** (a completed standalone Ragged release) | `ASSOCIATION_SEQUENCE` | 200,000 | 50,000,000 elements |
| `by_variant/z_overflow_index` / `_value` | int64 / float32 | the re-keyed `z` overflow table | `RAGGED_EXCEPTION_TABLE` | 200,000 | 10,000,000 elements |
| `by_variant/eaf_exception_index` / `_value` | int64 / float32 | the re-keyed EAF exception table | `RAGGED_EXCEPTION_TABLE` | 200,000 | 10,000,000 elements |
| `by_variant/se_exception_index` / `_value` | int64 / float32 | the re-keyed SE exception table, **when the plan codes `se` as `int8_residual`** | `RAGGED_EXCEPTION_TABLE` | 200,000 | 10,000,000 elements |

The parenthesised dtypes are OGS-00011's; the plan decides them (`z` is `int16`
or `float16`, `se` is `float16` or `int8_residual`, and `eaf` is `absent`,
`float32` or `int8_residual` — §6a). `analysis_index`, `offsets` and `imputed`
are format-fixed.

`RAGGED_PER_VARIANT` applies `_per_variant` with a **1,000-element override**:
the offset array is read at a single variant, so one 8 KB inner chunk per
variant is the right read unit, and the role's 10 M-element shard keeps the file
count low. `analysis_index` and not `variant_index` is the per-row key: the
row's variant is implied by the offsets, the Analysis is not. The per-variant
`eaf_baseline`, `se_coefficients` and (if the plan has one) `eaf_reference` are
per-variant or per-Analysis artifacts of the component and are **shared, not
duplicated**; only arrays keyed on the flat cell position are re-keyed.

Three cell-keyed parts of the Ragged contract move with the rows and cannot be
recovered from the Analysis-sorted component, because the by-variant rows do
not preserve their original CSR ordinal:

- **`imputed`** (a completed standalone Ragged release) drives Association
  Status, `observed_only` and the reference-EAF substitution, so it is
  duplicated as a flat `uint8` sequence exactly as the Analysis-sorted one is.
- **`se_exception_index` / `se_exception_value`** are keyed by CSR ordinal
  (`docs/spec/store-format.md` §6a), so under `int8_residual` they are re-keyed
  to the by-variant ordinals like the Z overflow and EAF exceptions; a
  duplicate without them cannot reconstruct an exact SE.
- **`z_overflow_index` / `_value`** and **`eaf_exception_index` / `_value`** are
  already in the table and are re-keyed the same way.

A parity test on an indexed fixture must therefore compare `imputed` and exact
SE cells, not only `z`/`se`/`eaf` values: a duplicate that dropped the mask
would answer `observed` where the store holds `imputed` (silently), and one
that dropped SE exceptions would return the coded value instead of the exact
one.

The `ASSOCIATION_SEQUENCE` shard is a fixed **50 M elements** whatever the
dtype, so `analysis_index`'s shard is 200 MB where `z`'s is 100 MB and `se`'s
100 MB; that is #248's policy, not a choice made here. If a dtype-aware cap is
wanted, it is a #248 amendment that changes the role, not a private shard in
this index.

### Path and group mapping the converter and validation need

The index's paths are `ragged/by_variant/<leaf>` in both a Ragged and a Hybrid
release (a Hybrid has no separate path space; its Overflow is the same
`ragged/` group). The strict mapping in `opengwasdb/store/arrays.py` must learn
them before the builder lands, or a 0.2.0 conversion refuses the group. The
required additions are:

- **`is_recorded_group_path`** must accept `ragged/by_variant` as a recorded
  group. Today only `top_hits/<tier>` is allowed to have a second segment, so
  `ragged/by_variant` returns False and a conversion refuses the container with
  "unknown group". The `by_variant` group holds arrays only; no deeper group is
  defined, so the rule is exactly one extra segment under `ragged`.
- **`role_for_array_path`** must resolve `ragged/by_variant/<leaf>`. Today
  `_grouped_role("ragged", rest)` looks the whole remainder up in
  `_RAGGED_ROLES_BY_NAME`; `by_variant/offsets` is not a key, so it returns
  `None` and the array is refused. The nested table is:

  | leaf under `ragged/by_variant/` | role |
  |---|---|
  | `offsets` | `RAGGED_PER_VARIANT` (with the 1,000-element inner-chunk hint recorded alongside) |
  | `analysis_index`, `z`, `se`, `eaf`, `imputed` | `ASSOCIATION_SEQUENCE` |
  | `z_overflow_index`, `z_overflow_value`, `eaf_exception_index`, `eaf_exception_value`, `se_exception_index`, `se_exception_value` | `RAGGED_EXCEPTION_TABLE` |

  `imputed` and the `se_exception_*` pair are present only when the component
  has them (`imputed` on a completed standalone Ragged release; the SE exception
  table only under `int8_residual`). A release whose Analysis-sorted component
  carries one of them but whose `by_variant/` group does not is invalid, and so
  is the reverse.

  An unknown leaf under `by_variant/` must keep returning `None` (refused), as
  the other groups' leaf maps do.

- **Validation** reads the same table: the layout recorded for each
  `ragged/by_variant/...` array must agree with `chunk_layout`/`shard_layout`
  for the role (including the offsets array's explicit inner-chunk hint, which
  must be part of the recorded layout, or a converted array and a rebuilt one
  would disagree), and the group must be present exactly when the release
  declares the index. Whether the index's *presence* is recorded in the
  manifest is left to the builder ticket; it is a provenance fact, not a new
  `format_version`.

### The offset axis: `n_axis + 1` direct offsets

`by_variant/offsets` is indexed by the **component's own variant index**, not by
an off-axis rank. Its length is `n_axis + 1`, where `n_axis` is the component's
variant axis length — the shared axis for a Hybrid Overflow (164,051,296 at
OGS-00011) and the store's own axis for a pure Ragged component. An entry for a
variant the index does not cover (every on-panel variant of a Hybrid) is empty:
`offsets[v] == offsets[v + 1]`.

This is the correction to the first draft, which declared
`n_off_axis_variants + 1` while queries indexed it with the shared
`variant_index`. Panel and off-panel variants interleave on a Hybrid's shared
axis, so a shared index is not an off-axis rank: a high shared index would run
past an `n_off_axis_variants` array, and a lower one would address another
variant's block.

The alternative — a compact off-axis axis plus an explicit
shared-index-to-rank map — saves `n_shared - n_off_axis` entries (13.4 M
int64 = 0.107 GB here) at the cost of a second array and a rank lookup on every
query. Direct offsets are chosen: one array, no mapping, and the saving is
0.3 % of the index.

### Ordering and the query contract

Rows within a variant are ascending by `analysis_index`, so a PheWAS result is
in the same Analysis-ascending order the current scan returns; a region query
returns variant-ascending, Analysis-ascending within each. The facade keeps its
documented "no ordering guarantee beyond grouping" contract; the index is a
faster route to the same grouping, and the parity tests in
`tests/test_variant_side_scans.py` can be pointed at an indexed fixture to pin
that.

### Where it is read

- `RaggedStoreQuery.phewas` and `HybridStoreQuery.phewas` (off-axis):
  `offsets[v]` then the variant's row block.
- `RaggedStoreQuery.range_phewas` and `HybridStoreQuery.range_phewas`:
  `offsets[lo:hi]` over the variant range, then the rows of the variants the
  range selects.
- `RaggedStoreQuery.lookup` / `HybridStoreQuery.lookup`: unchanged; step 3's
  per-Analysis binary search is already proportional to the request and works
  on an unindexed release.
- `top_hits` and the Analysis-side reads are unchanged.

An absent index is not an error: the facade falls back to the step-3 scan.

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
`eaf_baseline`. Its raw size is the four per-cell columns
(12.34 + 6.17 + 6.17 + 3.085 = 27.8 GB) plus `by_variant/offsets` over the
shared axis (164,051,297 × int64 = **1.31 GB**), plus the re-keyed EAF
exception table (2.17 GB) and a small re-keyed `z` overflow table: about
**31.3 GB**.

**Measured (ruling c, `docs/benchmark-output/opengwasdb_252_index_cost.json`):
the finished `ragged/by_variant/` group is 16.678 GiB (17.91 GB) on disk**, so
`data.zarr/ragged` grew from 20 GB to 37 GB — within the ~+18 GB estimate above,
and not quite doubling the Overflow after compression. The estimate was made
from raw byte counts; the measured group is smaller because the duplicated
`z`/`se`/`eaf` codes compress like the Analysis-sorted planes they copy. The
same artifact's other rows: eQTLGen (127,331,910 rows) **+0.422 GiB**, OGS-00006
(58,054,212) **+0.158 GiB**, the pilots (86,373 and 202,803) under a megabyte.

OGS-00011's Overflow is observed-only and codes `se` as `float16`, so neither
`imputed` nor an SE exception table is duplicated there. A **completed
standalone Ragged** component adds `by_variant/imputed` (uint8, 3.085 GB raw at
this scale) and a component whose plan selects `int8_residual` adds a re-keyed
`se_exception_index` / `_value` pair, whose raw size is the Analysis-sorted
table's own — **12 bytes per exception in total** (an int64 index plus a
float32 value), before compression; both are proportional to the cells the
component already stores. The general accounting is therefore
`n_axis + 1` int64 offsets + one int32 and one `z` cell per association + one
`se` cell + one `eaf` cell + (one uint8 `imputed` cell when completed) + the
re-keyed overflow and exception tables.

### Build cost

No comparison sort is needed. The Overflow is analysis-major, so the build is a
counting sort by variant: one pass to count rows per variant over the shared
axis (`np.bincount(variant_index, minlength=n_shared)`, 164 M int64 = 1.31 GB),
a prefix sum to `offsets`, and one pass to scatter each cell into its variant's
block while copying the codes and re-keying the overflow and exception tables
(and the `imputed` mask and SE exceptions where the component carries them).

**Measured on OGS-00011 (ruling c): 2,035.8 s (33.9 min), peak RSS 9.914 GiB.**
The implementation is bounded, not a whole-component pass: a windowed count, a
band-partitioned spill (destinations are known after the counting pass), and
whole-shard writes, holding the `n_axis + 1` offsets (1.31 GB), one 50 M-cell
band, and the (small) exception tables. The first estimate was 30–60 min; the
measured build sits at the low end, and the same artifact's smaller rows scale
as expected — eQTLGen (127.3 M rows) 356.7 s / 2.73 GiB, OGS-00006 (58.1 M)
54.5 s / 2.11 GiB, the pilots sub-second. The build re-reads the component it
is duplicating; it never needs the cells resident, so its peak does not grow
with the store.

### Query cost

**Measured on OGS-00011 (`docs/benchmark-output/opengwasdb_ogs00011_252_variant_index_ab.json`),
each shape in a fresh process with the index present and absent** (the absence is
a rename of `ragged/by_variant/`, so the scan side is genuine; answers are
compared in canonical row order, and every shape's two answers matched):

| shape | scan | index | speed-up |
|---|---:|---:|---:|
| off-axis PheWAS (`phewas_off_axis`) | 31,833.3 ms | **44.7 ms** | 711× |
| 1 Mb region (`regional`) | 97,980.0 ms | **5,033.1 ms** | 19.5× |
| bulk, one Analysis genome-wide | 14,322.4 ms | 13,209.2 ms | control |
| PheWAS, on-axis variant (Dense) | 1,917.9 ms | 1,925.1 ms | control |
| region × one Analysis | 10,244.5 ms | 11,539.7 ms | control |
| top hits | 45.7 ms | 66.7 ms | control |
| random lookup, 10×100 | 3,062.7 ms | 3,094.9 ms | control |
| random lookup, 100×10 | 2,376.3 ms | 3,271.0 ms | control |
| bulk, largest Overflow | 26,808.6 ms | 29,223.7 ms | control |

Off-axis PheWAS lands in the tens of milliseconds the acceptance asks for,
against the same harness's on-axis Dense PheWAS of ~1.9 s. The 1 Mb region is
5.0 s, not the sub-second the first estimate hoped: most of that is the Dense
Component's own window read, which the index does not serve — the Overflow's
part is the range's contiguous blocks. A per-variant read is one 8 KB offsets
chunk plus the variant's ~30 KB block; step 2's phase split confirms the scan's
cost is its **match** phase (eQTLGen PheWAS: 1198 ms match / 14 ms read) while
the index's match is ~2 ms.

The index's re-keyed EAF exception table (180,396,687 entries, ~2.1 GiB) is read
once per query process, on the first decode (~2.4 s at OGS-00011). It is read
lazily: opening a store for an analysis-side shape that never decodes the index
pays nothing, and the benchmark warms it before timing a decoding shape so the
table read is not charged to the per-query number. The scan side pays the
equivalent table read when its reader opens.

## Considered options

- **A permutation index** (per-variant `offsets` plus int32 positions into the
  existing Analysis-sorted arrays). Cheaper on disk (offsets over the shared
  axis 1.31 GB + 12.34 GB of positions ≈ 13.7 GB, no statistic duplicated), but
  its reads are scattered: one 3,317-Analysis variant's rows are gathered from
  across the whole CSR, an expected ~3,000 of the Overflow's ~15,425 `z` chunks
  — about 19 % of the plane per query, and the same again for `se` and `eaf`.
  That is the whole-plane read step 3 removed, reintroduced as a gather, and a
  region query of 8.3 M rows is worse. Rejected.
- **A compact off-axis axis plus a rank map.** Saves 0.107 GB and costs a
  second array and a lookup on every query. Rejected (see the axis section).
- **An analysis-list index only** (per-variant "which Analyses hold this
  variant", no statistics). The number of segment searches is the number of
  Analyses holding the variant (`n_analyses_hit × O(log rows)` chunk reads),
  which for a pleiotropic variant at 3,317 Analyses is thousands of chunk reads
  and hundreds of MB. Rejected.
- **A delta- or varint-encoded `analysis_index`** to shrink the duplicate's
  dominant column. Worth measuring, but it changes the index's *codes*, not its
  role or shape, and can land later. Deferred, not rejected.
- **Extending the Dense Component's axis to carry the off-axis variants.**
  ADR 0026 rejected this for the component partition; the same reasoning
  applies to the index. Rejected.
- **Do nothing; rely on `lookup`'s binary search.** Cannot make a whole-store
  PheWAS or region query sublinear in N. Rejected.

## Consequences

- **The index is optional and additive.** No Analysis-sorted array changes, no
  `format_version` changes, an unindexed release still answers every query,
  and a 0.1.0 or 0.2.0 release can gain or lose it by a build, not a
  conversion.
- **A `ragged/by_variant/` group is a large share of the Overflow's bytes on
  disk** at OGS-00011 (measured **+16.678 GiB**, from 20 GB to 37 GB of
  `data.zarr/ragged`). That is the price of making per-variant work
  proportional to the answer.
- **The strict path and group mapper must be extended** (the `by_variant`
  group and its leaf→role table, and the offsets array's inner-chunk hint as
  part of its recorded layout) before a converted 0.2.0 release can carry the
  index; a conversion refuses the group until then, loudly.
- **The Hybrid top-hit index and the streamed Overflow writers are affected**:
  the builder writes the duplicate in the same phases and bounded regions as
  the Analysis-sorted arrays, and the Hybrid's `dense_to_shared` map is
  unchanged because the index is keyed on the shared axis.
- **Built and wired as #252 step 5.** The Ragged, SSF, BESD and Hybrid builders
  write it, Reference Completion rebuilds it, `ogdb build-variant-index` adds it
  to an existing 0.2.0 release, the query facade reads it (falling back to the
  scan), and `validate` checks it. Parity against the step-3 scan is asserted on
  indexed fixtures and on OGS-00011 (`
  docs/benchmark-output/opengwasdb_ogs00011_252_variant_index_ab.json`).

## References

- The ticket: opengwas/opengwasdb#252 (steps 3, 4 and 5); the epic: #240.
- ADR 0026 (Hybrid Layout), ADR 0037 (encodings), ADR 0057 (format 0.2.0),
  ADR 0058 (Dense chunk and shard shapes), ADR 0059 (#248: the Ragged roles,
  the 50 M-cell sequence and 10 M-cell side shards, and the strict path mapper).
- The format-free fixes this index replaces: #252 step 3,
  `tests/test_variant_side_scans.py`.
- The Overflow's array metadata and on-disk size: OGS-00011
  `store.opengwasdb/data.zarr/ragged`.
