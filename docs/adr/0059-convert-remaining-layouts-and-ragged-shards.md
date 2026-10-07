# Convert every layout to 0.2.0, and bound the Ragged shards

Issue #248, epic #240. This is a short successor to ADR 0057, which records what
format 0.2.0 is and why conversion — not a rebuild — is the migration route. It
records the two decisions #248 adds: that the converter handles **every** layout
rather than one, and how the Ragged 1-D arrays are sharded, which #246 does not
decide.

## Context

ADR 0057 landed a converter for one layout, Dense Observed-Only, as the tracer.
The rest of the epic needs the other layouts converted the same way, because a
store whose values are right and whose layout is old should not be rebuilt:

- **Ragged** (Observed-Only and Reference-Completed): a CSR group whose parallel
  sequences are read by Analysis. OGS-00001 and OGS-00002 are the pilots.
- **Hybrid**: a top-level release whose **nested Dense Component is a Store
  Release with its own manifest, `index.sqlite` and `data.zarr`** (spec §16).
  OGS-00004 and OGS-00005 are the pilots, and OGS-00011 (#250) is the large one.
- **Dense Reference-Completed**: the Dense tracer plus the imputed mask,
  `on_panel`, `eaf_reference` and the SE/EAF side tables. OGS-00010 is the pilot,
  13,549,988 variants × 2,024 Analyses.

#245's review left a recorded-layout gap it could not close: a Hybrid Dense
Component's manifest has no `provenance.dense.chunk_shape`; the layout lived
only in the **outer** `provenance.hybrid`, which the component validator never
sees.

The Ragged arrays force a shard decision #246 does not own. #246 benchmarks the
**Dense** Analysis-axis inner chunk and shard. The Ragged sequences are 1-D and
read by Analysis, and OGS-00011's overflow sequences are **3,085,080,783
entries** at a 200,000-element inner chunk. #245's proposed default — about one
million elements per shard — would make 3,086 files per sequence, and the
whole-array policy it used for exception tables would make OGS-00011's
180,396,687-entry Ragged `eaf_exception_index` a single 1.4 GB file.

## Decision

### 1. One converter handles every layout

`opengwasdb.store.convert` / `scripts/convert_store_to_0_2_0.py` no longer
refuses Dense Reference-Completed, Ragged or Hybrid. It walks every Zarr tree a
release carries — `data.zarr` for a standalone release, plus `dense/data.zarr`
for a Hybrid — and plans every array through the same role table. The refusals
that remain are the ones that protect a converted release from being a guess:
an array or group the role table cannot name, a source already at 0.2.0 on
**any** manifest (so a half-converted Hybrid is refused), and a layout the
format does not define.

A Hybrid is converted as one operation: one fresh `release_id` for both
manifests, both `data.zarr` trees rewritten, both `index.sqlite` `dense` blobs
re-pointed, and `verify_conversion` checking both trees. Publishing stays a
single staged rename, so a failure leaves neither half behind.

### 2. The Ragged shards are element caps, not parameters

| role | shard |
|---|---|
| Ragged association sequences (`ragged/z`, `se`, `variant_index`, `eaf`, `imputed`) | 50,000,000 elements |
| Ragged per-variant side arrays (`ragged/eaf_baseline`, `eaf_reference`) | 10,000,000 elements |
| Ragged exception / overflow tables (`ragged/z_overflow_*`, `ragged/eaf_exception_*`, `ragged/se_exception_*`) | 10,000,000 elements |

Each is rounded to a whole number of inner chunks and clipped to the array, so
the shard is always a whole multiple of the inner chunk.

The numbers are chosen so a shard is tens to low hundreds of MB and a converter
worker's block is bounded like a Dense one (100,000 × 1,024 int16 ≈ 205 MB):

- OGS-00011's overflow sequences become **62 files per array**, at ≤ 200 MB
  uncompressed for the widest 4-byte dtype, rather than 3,086;
- its Ragged `eaf_exception_index` becomes **19 files of 80 MB**, rather than one
  1.4 GB file;
- a sequence shorter than one shard is one file, so small Ragged stores are
  unchanged.

The sequence shard is bounded by **cells**, not by one Analysis's run. A
variant-side index (#252) is a new array beside the Analysis-sorted ones, and a
cell-bounded shard leaves the existing arrays shard-aligned, so adding it does
not require re-sharding them.

### 3. A Hybrid Dense Component records its own layout

The converter writes the effective `chunk_shape`, `shard_shape`, `compressor`
and `zarr_format` into the nested component's `provenance.dense`, and keeps the
outer release's `provenance.hybrid` in step. The recorded-layout rule (§20) then
covers the component through the component validator it already runs, closing
#245's gap. A component root with no Dense plane carries no Dense layout
attributes.

## Consequences

- **Every existing Store Release the seam can describe is convertible.** The
  #248 pilots are OGS-00001 and OGS-00002 (Ragged), OGS-00004 and OGS-00005
  (Hybrid) and OGS-00010 (Dense Reference-Completed); OGS-00011 follows in #250
  once #252's format-free fixes land. #248's worker report records the numbers
  and any pilot that could not be converted.
- **The Dense shapes stay #246's decision.** The Ragged roles are distinct from
  the Dense ones precisely so a Ragged shard can be sized here without moving the
  Dense numbers #246 is benchmarking.
- **A Ragged or Hybrid outer root has no Dense layout attributes**, and the
  verifier checks the root's rewritten keys only where a Dense plane exists.
- **The Ragged shard caps are not CLI flags.** They live in the seam's role
  table, one authority, so the converter and #247's builders cannot disagree.
- **A builder writes a Ragged sequence plane one whole shard at a time (#249).**
  The 50,000,000-element shard is one file, so flushing it in 4,194,304-cell
  regions was a read-modify-write of the whole shard, about twelve times per
  shard; `RaggedCSRWriter` now raises its write region to the shard.  The cost is
  a roughly 1.5 GB working set at the full shard (about 30 bytes a cell) against
  the 130 MiB a 4,194,304-cell region used.  The whole-shard write guard covers
  multi-shard 1-D arrays too, so this cannot regress silently; a 1-D array whose
  shard is the whole array (the Dense SE exception table, filled band by band)
  stays exempt.

## Alternatives rejected

- **Refuse everything but Dense Observed-Only and convert each layout in its own
  ticket.** It would fork the plan/verify/publish machinery per layout and leave
  the recorded-layout gap open; rejected.
- **Keep one million elements per Ragged shard (#245's proposed default).** 3,086
  files per OGS-00011 sequence is file-count overhead for no read benefit, since
  the inner chunk is what a query reads; rejected.
- **Shard a Ragged sequence by one Analysis's run.** It would make a shard
  Analysis-aligned and cheap to reset, but a variant-side index added later
  (#252) would need the arrays re-sharded around a different sort order;
  rejected in favour of a fixed cell count.
- **Keep the Ragged exception tables whole in one shard.** Bounds the file count,
  but OGS-00011's Ragged `eaf_exception_index` is a 1.4 GB single file and one
  converter worker holds it whole; rejected.
- **Record a Hybrid component's layout only in the outer `provenance.hybrid`.**
  That is the gap #245 reported: the component manifest describes arrays it does
  not describe. Rejected.
