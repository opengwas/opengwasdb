# Store format 0.2.0: Zarr v3 with sharding

> **Amended by the addendum at the end (#247).** Decision 2's interim —
> `CURRENT_FORMAT_VERSION` held at `0.1.0` "until #247" — is over: the builders
> write 0.2.0. The addendum records the one decision the ticket left open, that
> a build does not choose its shard. Decisions 1, 3, 4 and 5 stand as written.

Issues #237 and #239, epic #240, implemented by #245 (the Dense converter) and
#247 (the builders). This ADR records what 0.2.0 is, why the migration route is
**conversion** rather than rebuild, what it costs, and the rejected options.

## Context

Two problems meet here, and sharding is what lets one fix serve both.

**The Dense Analysis-axis chunk is too wide.** OGS-00009 is 9,847,701 variants ×
2,024 Analyses, `[1000, 1000]`, 119,118 files, `z`/`se`/`eaf` 29,545 files each
(#239). One Analysis genome-wide under zarr 3 with the #244 settings takes
21.9 s and 1.14 GB peak; under zarr 2.18 it took 27.6 s and 12.04 GB (ADR 0056,
#244 Stage B). PheWAS — one variant across every Analysis — reads 2,024
Analysis-columns' worth of 1,000-wide chunks to extract one column per chunk:
6.5 ms on 2.18, 21.2 ms on zarr 3. Narrowing the Analysis-axis chunk to 64 or
128 turns a PheWAS read into 16–32× fewer bytes at the same chunk count
multiplied the same way (#237).

**A narrower chunk multiplies the file count.** `[1000, 64]` is roughly 16× the
files of `[1000, 1000]`, and a release already holds 119,118. Sharding (Zarr v3's
`sharding_indexed` codec) breaks the tie: the **inner chunk** stays the unit a
query reads, and the **shard** — a bounded block of inner chunks — is the unit
stored as one file. Narrow reads get narrow chunks, and the file count is set by
the shard, not the chunk (#239).

zarr-python 3 also reads v2 stores, so a 0.1.0 release stays readable while
0.2.0 exists. The change from v2 to v3 is a **breaking** change under ADR
0038/ADR 0041 — `zarr.json` replaces `.zarray`/`.zgroup`, a required entry
restructured — so it is a new release series, `0.2`.

## Decision

### 1. 0.2.0 is Zarr v3 with the sharding codec; arrays, dtypes and encodings are unchanged

The arrays, their dtypes, their fill values and their encodings are exactly
those of 0.1.0 (spec §10, §6a). Only the physical layout changes: every array
carries a `zarr.json` whose codecs begin with `sharding_indexed`, whose inner
chain is the v3 spelling of the 0.1.0 compressor — Blosc zstd / clevel 3 /
bitshuffle — and whose shard shape is the `chunk_grid` configuration. An array
stored uncompressed in 0.1.0 stays uncompressed inside its shard. The spec
states this in §10a.

### 2. Readers read 0.1.0 and 0.2.0; builders will write only 0.2.0

`SUPPORTED_FORMAT_VERSIONS` gains the `(0, 2)` series. `CURRENT_FORMAT_VERSION`
**stays `0.1.0` until #247**, so every builder keeps writing 0.1.0 and the
converter is the only 0.2.0 writer in the interim. No package version is cut
between the converter landing and the builders switching (the epic's sequencing
note).

The consequence is deliberate and tested: `check_writable_format_version`
(ADR 0038 §4) now refuses to complete a 0.2.0 source, because completion writes
into its source's arrays and preserves its source's `format_version`. A converted
store is completed **before** conversion, or after #247. The guard was written
for exactly this state — "the moment a second readable version exists" — and
0.2.0 is where it becomes live. (The addendum at the end corrects that order for
the state #247 leaves: convert the `0.1.0` source to `0.2.0` first, then complete
the converted release.)

### 3. Conversion, not rebuild, is the migration route

A 0.2.0 release is derived from a 0.1.0 one by **rewriting the physical layout
and nothing else**: every array is read as stored codes and written as the same
codes into a v3 sharded array. The values are bit-identical — the converter
verifies this block by block, as raw bytes, before publishing — so a conversion
cannot change an answer, only the cost of producing it.

That is the contrast with ADR 0041's restamp: a restamp changes only the stamp
because the bytes already mean what the new version says; a conversion changes
the physical bytes and proves the values did not move. A rebuild would re-read
the source VCF (13h30m and 425 GB for `ukb-b`, #148) to produce the same values
under a different layout, with a full build's worth of new ways to get it
wrong. So:

- **Rebuild** remains the default for a store that should pick up build-time
  fixes, or whose sources are available cheaply.
- **Convert** is for a store whose values are right and whose layout is old.
  `scripts/convert_store_to_0_2_0.py` is the tool. It refuses a source it does
  not fully understand (any layout but Dense Observed-Only, by name; any format
  but 0.1.0), never writes its source, refuses an existing destination, derives
  a new release with a fresh `release_id` and `created_at`, records the source
  `release_id`, the installed commit and the new per-array layout in a
  `zarr_v3_conversion` provenance block, regenerates `overview.html`, and
  publishes by rename only after the staged copy is bit-exact against the source
  **and** validates with no errors.

  > **Superseded by [ADR 0059](0059-convert-remaining-layouts-and-ragged-shards.md) (#248).**
  > The converter now accepts every layout the role table can name — Dense
  > Reference-Completed, Ragged (Observed-Only and Reference-Completed) and Hybrid
  > as well as Dense Observed-Only — and refuses only an unmapped array or group,
  > a source already at 0.2.0, and an unknown layout. The rest of the passage
  > (source never written, destination refused, fresh identity, per-component
  > provenance, staged/validated/renamed publication) still holds. This ADR's
  > Dense Observed-Only scope is the state at #245.

### 4. Shards are bounded on both axes, and the shape is a parameter

#239 suggested a shard spanning a variant row block across every Analysis. That
does not fit the Dense VCF builder, which writes `[all variants × band]` column
bands: a shard is written whole only if every write covers it. Dense shards are
therefore `[V_s × A_s]` with `A_s` the band width. Completion and the SE rewrite
write row blocks, which is compatible.

The converter takes the inner Analysis-axis chunk (`--dense-analysis-chunk`,
default 64), the Dense shard (`--dense-shard`, default `100000x1024`) and the
top-hit shard width (`--top-hit-shard-chunks`, default 64 inner chunks) as
parameters. **#246 benchmarks the shapes and decides the defaults**; #245 only
fixes the mechanism and proposes the defaults. The inner chunk and shard are the
array seam's role policies (`chunk_layout`, `shard_layout`), one authority
#247's builders also read. `--top-hit-shard-chunks 1` gives a top-hit shard of
one inner chunk, the "effectively unsharded" variant #246 measures the top-hit
query against; it is still a v3 sharded array, so the "every array sharded" rule
is not relaxed.

### 5. The layout is recorded in three places and validated

`manifest.json` `provenance.dense` (`chunk_shape`, `shard_shape`, `compressor`,
`zarr_format`), the `index.sqlite` `dense` blob, and the `data.zarr` root
attributes all describe the Dense planes' layout. New validation rules require
each to agree with the arrays (clipped the way the role policy clips a hint), and
require the Zarr on-disk format to match `format_version` — 0.1.0 all v2, 0.2.0
all v3 and every array sharded. A half-converted release is invalid. The
per-variant chunking rule (issue #135) applies to the **inner** chunk, because
that is the unit a query reads.

## Consequences

- **A converted release is a new release.** It is not a Provenance Amendment:
  the arrays change, so it is outside the exception (spec §21.4). It keeps
  `store_id`, and changes `release_id` and `created_at`.
- **The converter is the only 0.2.0 writer until #247**, so no version is cut in
  between. `opengwasdb-stores` (the sibling repository) changes in the same cut:
  its query walkthrough, manifests and catalogue name format versions, and none
  of them fails when this package changes.
- **0.2.0 does not make reads faster on its own.** Sharding adds a codec hop;
  the win comes from the narrower inner chunk, and sharding is what keeps the
  file count from exploding. #246 measures the combination against set L and the
  whole-Analysis guard.
- **The file-count win is the shard's.** OGS-00009 at `[1000, 64]` inner and a
  `100000×1024` shard has 198 shards per plane rather than ~315,000 files.
- **A converted store carries new files and a new identity, so anything that
  pins a release by `release_id` must be re-pointed.** That is the normal cost of
  a derived release (#164).
- **Deleting the v2 reader is a later decision.** It stays until nothing needs
  it; #247 makes 0.2.0 current, and #248/#250 convert the remaining layouts and
  stores.

## Alternatives rejected

- **Stay on Zarr v2 with narrow chunks.** The file count and per-file overhead
  are the reason #239 exists: `[1000, 64]` on OGS-00009 is roughly 16× the files
  of `[1000, 1000]`, and top-hit / exception / per-variant arrays make it worse.
  Rejected.
- **Tarball or archive packaging instead of sharding.** It reduces inode count
  without changing the codec, but it replaces random access with an extraction
  step, needs a format of our own to describe offsets, and zarr v3 already ships
  the mechanism. Rejected.
- **A shard spanning a variant row block across every Analysis.** It fits a
  row-block writer but not the Dense VCF builder's `[all variants × band]`
  writes, so a shard would never be written whole and each band write would
  become a read-modify-write of the entire Analysis axis. Rejected in favour of
  bounded `[V_s × A_s]` shards.
- **In-place conversion.** Rewriting a published release's arrays in place
  breaks the immutability every reader relies on, leaves a half-converted store
  if it dies, and cannot mint the new `release_id` a derived release needs.
  Rejected: the converter stages, verifies and publishes by rename.
- **Rebuild every store.** Correct but disproportionate where the values are
  unchanged: `ukb-b` is 13h30m from 425 GB of VCF (#148), and a rebuild's only
  product here is a different layout. Rejected as the default; a rebuild is
  still the right answer for a store that should gain build-time fixes.
- **Keep writing 0.1.0 from the builders and make 0.2.0 read-only.** It defers
  the format change indefinitely while the file-count problem stays. Rejected:
  the epic's point is to adopt the layout, and the interim is explicitly
  temporary.

## Addendum (#247): the builders write 0.2.0, and a build cannot choose its shard

Decision 2 above held `CURRENT_FORMAT_VERSION` at `0.1.0` "until #247". This
addendum records what #247 did, and one decision the ticket left open.

**One version constant.** `CURRENT_FORMAT_VERSION` is `0.2.0` and is what the
converter *writes*; the converter's source version is its own
`SOURCE_FORMAT_VERSION = "0.1.0"`.  There is no second target constant.  A
built release and a converted release therefore declare the same format and
carry the same physical layout, which is what #249 checks.  The interim of
decision 2 — builders and converter writing different formats — is over, and no
package version was cut inside it.

**The builders take the shapes from the seam.** A Dense release's inner chunk is
`DENSE_CHUNK_SHAPE`, the ADR 0058 decision `[1000, 64]`; its shard is
`DENSE_SHARD_SHAPE`, `[100_000, 1024]`, clipped to the array; the top-hit index
is sharded at `TOP_HIT_SHARD_CHUNKS`; every other role's shard comes from the
same `_SHARD_LAYOUTS` table the converter reads. `create_array` computes the
shard from the role whenever it creates an array in a v3 group, so a builder
cannot write an unsharded 0.2.0 array, and the seam translates a numcodecs Blosc
codec to its v3 spelling rather than letting the writers keep two codec names.

**A build does not choose its shard.** `--chunk-variants` / `--chunk-analyses`
name the **inner chunk** and nothing else; there is no `--shard-variants` /
`--shard-analyses` pair. The reasons:

- the shard is a property of the **format**, decided once (#246) and recorded in
  three places that validation checks against the arrays. A build whose shard
  differed from the converter's would produce a release the format's own
  recorded-layout rule still accepts but #249's builder-vs-conversion identity
  does not, which is a divergence with no reader-visible symptom;
- a build that could set the shard could write a release the converter cannot
  reproduce, so the epic's stated equivalence ("a built store's layout equals
  what converting the 0.1.0 build would produce") would hold only by convention;
- a lower shard is a **memory** knob (ADR 0058 measured 256 at 10.1 GB ukb-b
  against 1024 at 40.3 GB), and the human accepted the 40.3 GB cost. If a build
  ever needs the narrower scratch, the change is to `DENSE_SHARD_SHAPE` — one
  edit in the seam, a new ADR superseding 0058, and every writer and the
  converter move together — not a flag.

The converter keeps `--dense-shard` and `--dense-analysis-chunk` because it
reproduces a chosen layout and records the choice in
`provenance.zarr_v3_conversion`, so a converted release says which shapes it
used. A builder always records the format's shapes.

**No build path writes a partial shard.** The Dense VCF band writer's band width
is the shard's Analysis width, and the row-block writers (the residual EAF
encode, the SE rewrite and its float16 narrowing, Dense Reference Completion, the
Hybrid Dense Component) write whole multiples of the shard's row count. A
test-time hook, `opengwasdb.store.arrays.require_whole_shard_writes` (also
enabled by `OPEN_GWASDB_REQUIRE_WHOLE_SHARD_WRITES=1`, which the real-data pilot
sets), refuses a write that covers part of a shard, so a later writer cannot
regress to the silent read-modify-write decision 4 warns about. It costs
production nothing: the hook is off unless entered.

**Completion of a `0.1.0` source is refused**, naming
`scripts/convert_store_to_0_2_0.py`. Decision 2 predicted this as the guard's
first real firing; it is now live and tested. The one executable order for a
converter-supported layout is: **convert the `0.1.0` source to `0.2.0`, then
complete the converted release** (or rebuild). Completion cannot come first — a
`0.1.0` source is not writable — and it cannot be skipped: the converted release
is a `0.2.0` release, and this build writes it.
