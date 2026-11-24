# Format 0.3.0: a variable-length ALID sidecar, and the rewrite that derives it

## Status

Accepted. Issue [#467](https://github.com/opengwas/opengwasdb/issues/467), the tracer of epic
[#450](https://github.com/opengwas/opengwasdb/issues/450)'s format tickets (#467-#472). **In
development:** format 0.3.0 is written on the epic's integration branch and released only once
#468-#472 are in, so it is one format version with one backfill (the owner's plan of record, 23 Nov
2026).

What this ADR changes in other documents:

- Spec [§1](../spec/store-format.md#1-store-release-envelope), §1.1, §4: the envelope names the
  sidecar's four files from 0.3.0 and the fixed-width pair before it; §4 states the layout.
- Spec §20.0: a new rule, `variant_table.alid_sidecar`.
- From #468 (§6): spec §1, §1.1 and §4 name the position sidecar's three files and how a range is
  answered; §20.0 adds `variant_table.position_sidecar`; §21.3 and §21.4 add it to 0.3.0 and to
  the rewrite.
- Spec §21.1-§21.4: 0.3.0 in each table, and `ogdb rewrite-release` among the migrations.
- From #471 (the dated section below): spec §1, §1.1, §10a, §18, §20 and §20.0 for what
  0.3.0 drops, and banners on [ADR 0001](0001-hybrid-sqlite-zarr-store-envelope.md) and
  [ADR 0057](0057-store-format-0-2-0-zarr-v3-with-sharding.md) §5.
- [ADR 0038](0038-store-format-versioning-and-migration-policy.md) and
  [ADR 0041](0041-reset-format-version-and-drop-pre-release-formats.md) stand: this applies them.

**Who decided.** The layout's bounds (plain `.npy`, row to ALID in O(1), a vectorised binary
search, every ALID indexed), the version, the rewrite's shape and its proof are the #450
orchestration's rulings for #467, under the owner's plan of record. The details below are this
ADR's.

## Context

Every 0.2 release indexes its ALIDs in two sorted arrays: `variant_alid_bytes.npy`, each ALID in a
fixed `|S32` slot, and `variant_alid_rows.npy`, each slot's row. A fixed width is what lets
`np.searchsorted` run over a memory map. Two costs follow:

1. **An ALID over 32 bytes is not in the index** (#127: casting it in truncated two indels onto one
   key). The Variant Axis finds it by a tabix scan of the BGZF Variant Table, which only pysam reads
   and which the Store Reader is to stop opening (#476). OGS-00010 holds 115,043 such ALIDs; the
   real releases measured here hold 238 (OGS-00007), 287 (OGS-00025) and 2,026 (OGS-00024).
2. **Naming a row inverts the index.** The pair maps ALID to row. Row to ALID, which every resolved
   result needs, builds the inverse permutation on first use, an allocation the size of the axis
   (#465).

## Decision

### 1. The sidecar

Four plain `.npy` files beside the Variant Table, each memory-mappable and read with numpy alone
(`opengwasdb.store_reader.alid_sidecar`, which imports nothing else):

| file | dtype | length | holds |
|---|---|---|---|
| `variant_alid_utf8.npy` | `uint8` | total bytes | every row's ALID, UTF-8, in row order |
| `variant_alid_offsets.npy` | `<u8` | rows + 1 | row `i` is `utf8[offsets[i]:offsets[i+1]]` |
| `variant_alid_keys.npy` | `\|S24` | rows | each ALID's first 24 bytes, in full-byte order |
| `variant_alid_order.npy` | `<i4` | rows | the row of each key |

- **Row to ALID** is two reads of `offsets` and a slice of `utf8`: nothing is inverted.
- **ALID to row** is a `searchsorted` of the keys, vectorised over a batch. A query shorter than 24
  bytes is its own key, and the one stored ALID with that key is its match. A longer one is
  compared on its full bytes against the stored ALIDs sharing its key, which run in full-byte order
  and are bisected.
- **Every ALID is indexed**, however long, and the writer refuses an empty ALID or one that names
  two rows.

The key is a search aid, never an identity. 24 bytes because ALID length is mean 14.7-14.9 and p99
18-21 across the pilots and the three releases measured; 0.13-0.5% of their ALIDs reach 24 bytes,
so nearly every search ends at the key.

The codec's `write_alid_sidecar` is the sidecar's one writer. A build or a completion reaches it
through `write_variant_axis`, the axis's one writer, which every build and completion path already
calls; the rewrite (§3) calls it directly, as it rewrites the ALID index and nothing else of the
axis. The Hybrid's nested Dense Component writes its own sidecar through `write_variant_axis`; #472
makes it share the root's.

### 2. Format 0.3.0, a new series

A 0.2 reader looks up an ALID in `variant_alid_bytes.npy`. A 0.3.0 release has none, so that reader
would scan or miss every lookup while treating the release as fully understood. Under ADR 0041
this is an incompatible change, so the series moves. A package without 0.3 refuses a 0.3.0 release
by name, and this package reads 0.1, 0.2 and 0.3 (`SUPPORTED_FORMAT_VERSIONS`).

0.3.0 is 0.2.3 with the sidecar: reference-first indels (ADR 0079) and the SE count (ADR 0068)
stay. `format_version_for` stamps 0.3.0 on a release that carries the sidecar. It refuses to stamp a
sidecar beside sorted indels or without the count, because no version describes that release.
Builds and completion stamp 0.3.0. A completion of a 0.2.2 or 0.2.3 source does too: it rewrites
the axis through `write_variant_axis`. The 0.1.0-to-0.2.0 converter still writes 0.2.0 and copies
its source's fixed-width pair.

The fixed-width pair is not written from 0.3.0. A 0.3.0 release carrying it fails its closed
envelope, and a 0.2 release carrying the sidecar fails too (`envelope_at`). A reader takes
whichever index the release holds.

### 3. The rewrite: `ogdb rewrite-release SOURCE DESTINATION`

ADR 0038 makes a transformation produce a new, immutable release. `opengwasdb.store.rewrite` derives
a 0.3.0 Staged Release from a 0.2.3 one:

1. It refuses before writing anything: a destination that exists, is the source or lies inside it,
   and every source but 0.2.3. That includes one already 0.3.0, and a Hybrid whose two manifests
   are not both 0.2.3.
2. It copies the source (`cp -a --reflink=auto`), so an unchanged array costs no space where the
   filesystem can share blocks.
3. It runs `STEPS` on the copy, in order. Each step is one named change. #467's step,
   `alid_sidecar`, writes each component's sidecar from its Variant Table. It does not use the
   fixed-width pair, which lacks the long ALIDs, and it refuses a table whose rows are out of order
   or disagree with `variant_offsets.npy`. Then it deletes the pair. #468-#472 each append a step.
4. It stamps every manifest 0.3.0 with one fresh `release_id` and `created_at`, keeps `store_id`,
   and records a `format_rewrite` provenance block: the source `release_id`, the from and to
   versions, the steps, the tool, the time and the commit. From #468 it also records the range
   probes the release could not pose (`range_questions_not_posed`, §6). ADR 0038 §5 settles the identity: a new
   `release_id`, with the source named in provenance. The 0.2.0 converter set the precedent for
   `store_id` and for one `release_id` across both Hybrid manifests.
5. **The proof.** It asks the source and the staged copy the same deterministic questions
   (`opengwasdb.store.query_digests`) and refuses on any difference in any array of any answer:
   - each Analysis (at most 8, spread);
   - a PheWAS by ALID and by rsID of 7 variants spread over the axis, and of two identifiers no
     release holds;
   - a range PheWAS and a lookup;
   - the top hits at five thresholds, overall and per Analysis;
   - all of these both Observed-Only and with imputed cells;
   - the `variants_table()` and `analyses_table()`, column by column;
   - the resolved rows of the lookup, the range and the widest top hits;
   - the **ALID probes**: a PheWAS of each of the release's 3 longest ALIDs and of the first ALID at
     each of 23, 24, 25, 32 and 33 bytes, where an index changes how it searches, and one lookup of
     them all. Spread variants are short (the longest asked on OGS-00007 is 15 bytes), so without
     the probes no question reaches the sidecar's full-byte compare.

   The step also searches for every ALID it wrote and refuses unless each names its own row, so
   a search fault the questions miss still stops the rewrite.

   That code moved from `tests/store_reader_digests.py` into production. The golden-digest test
   calls it, so the proof and the test have one owner.
6. It validates the copy and publishes it by rename only if it is clean. Any failure discards the
   staging directory, and the source is never written.

Staging, the destination refusals and the validation gate are the converter's helpers
(`opengwasdb.store.convert`).

**Only 0.2.3 is a source.** 0.3.0 keeps 0.2.3's features. A 0.2.0, 0.2.1 or 0.2.2 release lacks
reference-first indels, the SE count, or both. No mechanical step can supply them: the reference
allele of a sorted indel is not in the release, and a 0.2.2 release may hold SEs above 65,504 that
0.2.3 leaves out. Such a release is rebuilt. This matters for the backfill
(opengwasdb-stores#498): no hosted release is 0.2.3 today (see Consequences).

**What the rewrite is for (the owner, 23 Nov 2026, in the #450 run; opengwas/opengwasdb-stores#498).**
Hosted releases are not rewritten. Once every 0.3.0 format change (#467-#472) is in, every Store is
rebuilt from source at 0.3.0. The rewrite stays a 0.2.3-only tool: the digest-equality proof that
each format step changes no answer, run on the fixture releases and on small real ones, and the
frame #468-#472 extend.

**The proof is sized for fixtures and small releases.** Its questions are bounded in Analyses and
variants, not in rows, and on a large Dense release some cannot run. Measured on OGS-00016 (Dense,
27.9 M variants, 1,691 Analyses):
- each `analysis` call takes 57 s and 1 GB, and the set asks 16 per side;
- `range_phewas` covers chromosome 1 from 48,327 to 50,802,392: 499,400 variants by 1,691
  Analyses, 844 M cells, each resolved to a row. That is infeasible;
- `variants_table()` would need about 17 GB, extrapolated from OGS-00025.

No bounded variant of the proof is built: under the owner's decision no large release is
rewritten.

### 4. Validation

`variant_table.alid_sidecar` (spec §20.0) judges a 0.3.0 release from the one streamed read of its
Variant Table. It checks five things:

- each row's sidecar ALID is the table's;
- the offsets start at 0, rise with every row and end at the byte array's end;
- the order names every row once, so every ALID is indexed;
- each key is the first 24 bytes of its row's ALID;
- the keys run in the strict order of the full ALID bytes, so no ALID names two rows.

A missing sidecar file fails `envelope.required`.

### 5. One ALID index, two implementations

`opengwasdb.store_reader.alid_index` is the one place that knows there are two ALID indexes. Both
answer one interface:
- `search`: each ALID's row, `ABSENT`, or `UNINDEXED` for an ALID the 0.2 slot cannot hold;
- `alids`: each row's ALID, None where the index has none;
- `unindexed_rows`;
- `problems`: what the validator's two rules report.

**One key chooses it: the release's `format_version`** (`alid_index_kind`). The Variant Axis takes
the version from the manifest and opens that kind only (`open_alid_index`). The envelope, its
required entries and both validator rules (each with a version `when=`) ask the same function. The
files present choose nothing:
- a stray sidecar beside a 0.2.3 release's pair is never read, and `envelope.closed` reports it;
- a release whose version carries an index whose files are missing is refused by name. Every
  readable version carries one, so no release is read by scanning the table for every ALID. A
  0.1.0 release without the pair, such as the retired OGS-00010, is refused, as `validate` already
  failed it.

The Variant Axis and `check-observed-view` read the table only for an `UNINDEXED` ALID, which the
0.2 pair alone answers.

**Order.** The axis answers ALIDs in the order asked, then rsIDs, on either index; a 0.2 release
once put the scanned long ALIDs last. A Dense `lookup` keeps that order. A Ragged or Hybrid `lookup`
returns its associations in row order whatever it was asked, on both formats, and #467 does not
change that.

**What #476 deletes**, when the 0.2 series stops being read:
- `FixedWidthAlidIndex` and its entry in `ALID_INDEX_KINDS`;
- the 0.2 branch of `alid_index_kind`;
- `UNINDEXED`, with the axis's and `check-observed-view`'s table reads for it;
- the `variant_table.alid_index` rule and its `when=`, which then leave `variant_table.alid_sidecar`
  without a `when=`.

### 6. The position sidecar (#468, 23 Nov 2026)

**Context.** A 0.2 release answers `range_phewas` by a tabix fetch of its BGZF Variant Table. That
loads pysam, which the Store Reader is to stop importing (#464, #476), and it is the last query
shape that reads the table to find its rows (#465). Rows are already in genomic order: a Variant
Index is sorted by `alid_sort_key`, and tabix refuses a table out of order. So a sorted array of
positions answers a range by binary search.

**Decision.** Three plain `.npy` files beside the Variant Table
(`opengwasdb.store_reader.position_sidecar`, numpy and the standard library only):

| file | dtype | length | holds |
|---|---|---|---|
| `variant_position.npy` | `<u4` | rows | each row's position, in row order |
| `variant_position_chromosomes.npy` | `\|S<n>` | chromosomes | each chromosome once, as the table spells it, in row order |
| `variant_position_chromosome_offsets.npy` | `<u8` | chromosomes + 1 | chromosome `k`'s rows are `offsets[k]:offsets[k+1]` |

- **A range is two `searchsorted` calls** in its chromosome's slice: the first position at or after
  `start`, and the first after `end`. The rows between are the answer, ascending. The chromosome
  is normalised first, by the existing owner (`normalise_chromosome`, which the Variant Axis
  already applied before a tabix fetch).
- **The dtypes.** A position fits `uint32`: the longest human chromosome is 248,956,422 bases,
  and `<u4` holds 4,294,967,295. That is 4 bytes a row, against the 8 of `<i8`. The row offsets
  are `<u8`, as the ALID sidecar's are; there are a few dozen of them. The labels are stored, not
  implied by a fixed list, because a release may hold a contig outside `1`-`22`, `X`, `Y` and `MT`,
  which `chromosome_sort_key` sorts after them by its text.
- **The order is the canonical chromosome order**, the order the rows are in. The writer takes the
  order as an argument (`chromosome_key`), so the codec does not restate it, and refuses rows not
  in it: a chromosome whose rows are not contiguous, a position that falls within a chromosome, a
  position outside `1..2**32-1`, an empty label, or chromosomes out of `chromosome_sort_key`'s
  order.
- **One writer.** `write_position_sidecar` is it. `write_variant_axis` calls it, so every build and
  completion writes the sidecar, the Hybrid's nested Dense Component included; #472 removes that
  copy. The rewrite calls it directly, as its own step.
- **0.2.x is unchanged.** A 0.2 release answers a range by tabix, through the legacy seam, as it
  did. Like the ALID index, the version chooses (`carries_position_sidecar`): a 0.3.0 release
  missing the sidecar is refused (`MissingPositionSidecar`), not read by tabix. A stray sidecar
  beside a 0.2 release is not read, and `envelope.closed` reports it.
- **Edge cases match tabix.** `TabixVariantTable.fetch` returns no rows when tabix refuses, so a
  0.2 release answers an `end` before `start`, and a chromosome it lacks, with no rows. The
  sidecar answers both the same way. A `start` of 0 or less starts at the first position, as
  tabix's `max(0, start - 1)` does.
- **`VariantAxis.range`**, which returns records rather than rows, finds its rows in the sidecar and
  reads them with `by_indices`. `VariantAxis.tabix_range` keeps the tabix read, and the validator
  uses it, because the `.tbi` still ships as a convenience (spec §1.1).

**The rewrite step**, `position_sidecar`, writes each component's sidecar from its Variant Table,
streamed in row order and checked against `variant_offsets.npy`. It does not open the release's
search indexes, because an earlier step rewrites them. Before it ends, it reads the sidecar back
and refuses unless every row is at the table's chromosome and position.

**The proof's range probes.** `query_digests.range_questions` picks the ranges from the release's
own rows, so they are deterministic. Each is asked both Observed-Only and with imputed cells:
- `first`, `to_first`: the first variant's position, and from 1 to it;
- `last`, `from_last`: the last variant's position, and from it to 500,000,000. That bound is past
  every human position and inside the 2^29 a tabix index addresses;
- `boundary/before`, `boundary/after`: from the last variant before the first chromosome boundary
  to that chromosome's end, and from the next chromosome's start to its first variant;
- `empty/between`: between the first two variants of one chromosome with a position free between
  them;
- `empty/chromosome`: the first of `1`-`22`, `X`, `Y`, `MT` the release lacks;
- `shared`, `shared/from_before`, `shared/to_after`: the first position several variants share,
  alone, from the position before it, and to the position after it.

A release that cannot pose one leaves that question out. A release with one chromosome, for
example, has no boundary. The rewrite does not leave a question out silently. Its log line names
each question it could not pose, and `provenance.format_rewrite.range_questions_not_posed`
records them. The probes are their own section, `range_probes`, so the golden pins keep their
shapes.

The six `store_reader_digests` fixtures that the rewrite tests use leave out five or six questions:
- All six leave out `boundary/before`, `boundary/after`, `shared`, `shared/from_before` and
  `shared/to_after`. Each holds one chromosome and no shared position.
- `dense_observed` also leaves out `empty/between`. No position is free between two of its
  variants.

`tests/one_source_releases.LAYOUTS` builds a Dense, Ragged and Hybrid release over three
chromosomes that poses all eleven questions. The `range_releases` session fixture builds them once.
`test_rewrite_release.test_the_proof_asks_every_range_question_of_each_layout` sends the 0.2.3
form of each one through `rewrite_release`. It asserts that the source and the staged release are
both asked all eleven questions and answer them identically. That test covers every question the
digest fixtures leave out.

**A known cost: the rewrite streams each table twice.** The ALID step and the position step each
read the Variant Table on their own, and each holds a Python list per row. That doubles the
rewrite's main cost, which matters at ukb-b's scale. One pass could feed both steps. It is not
worth the change now: the rewrite is a proof, run on fixtures and small releases, and every hosted
Store is rebuilt at 0.3.0 instead (the owner's ruling, recorded in §3).

**Validation.** `variant_table.position_sidecar` (spec §20.0) judges a 0.3.0 release from the one
streamed read of its table. It checks that:
- each row's chromosome and position are the table's;
- each array has its dtype;
- the offsets start at 0, rise with every chromosome and end at the row count;
- each chromosome is named once, in the canonical order;
- no position is 0, and none falls within a chromosome.

`envelope.required` names each missing file. `variant_table.representatives` still checks the tabix
index on 0.3.0.

The reader refuses some sidecars at open (`MalformedPositionSidecar`): offsets that do not number
the chromosomes plus one, a chromosome named twice, and a label that is not UTF-8. Read as they
stand, these would answer one chromosome's ranges from another's rows, or from none. The check
reads only the chromosomes, a few dozen. The Variant Axis opens the sidecar on first use, so in the
validator such a sidecar fails `variant_table.position_sidecar` alone. The comparison of each row
against the table is `PositionSidecar.misplaced`, shared by the rule and the rewrite's read-back.

**Alternatives considered.**
- *A fixed chromosome list, offsets only.* It needs no label file, but it cannot hold a contig, and
  the canonical order would then be stated in the format as well as in `chromosome_sort_key`.
- *One `(chromosome rank, position)` key per row*, searched once. It costs 4 bytes more a row,
  and needs the rank in the codec.
- *`int64` positions.* 8 bytes a row for nothing: no position needs more than 32 bits.
- *Keep tabix on 0.3.0.* pysam would stay on the query path, and #476 could not remove it.

## Consequences

- **A range on 0.3.0 reads neither the Variant Table nor its tabix index** (#468):
  `range_phewas` answers in every layout without importing pysam, and exactly as the 0.2.3 form of
  the release does by tabix (`tests/test_range_phewas_0_3_0.py`). The position sidecar costs 4
  bytes a row, and a few bytes a chromosome.
- **The Store Reader resolves every ALID on 0.3.0 without the Variant Table.** `lookup` and `phewas`
  by an ALID over 32 bytes return its row in every layout without importing pysam
  (`tests/test_long_alid_0_3_0.py`). On 0.2.3 the same questions return the same rows by the scan.
- **Size.** The sidecar holds every ALID whole, so it is about 40% larger than the pair it
  replaces. On OGS-00024 (7.07 M rows) the pair is 254 MB and the sidecar 359 MB, of which 57 MB
  is the 8-byte offsets. #469 drops `variant_offsets.npy`, 8 bytes per row.
- **Nothing hosted is rewritten.** Of the 20 hosted releases under `/data/opengwasdb/stores`:
  - 11 are 0.2.0;
  - 5 are 0.2.1;
  - 3 are 0.2.2;
  - 1 is 0.1.0;
  - none is 0.2.3.

  By the owner's decision (§3), all of them are rebuilt from source at 0.3.0
  (opengwas/opengwasdb-stores#498).
- **Cost of the proof.** A small Ragged release is proved in seconds (OGS-00007, 171,006 variants:
  21 s in all); OGS-00025 (1.18 M variants, 1,038 Analyses) in 2 × 72 s. A large Dense release is
  out of its reach (§3).
- **Tests build 0.2.3 releases** by turning a 0.3.0 build back (`tests/format_0_2_3.py`). The nine
  completion pins, turned back so, match their previous pins in every entry but the Dense
  `overview.html`, which lists the files.

## 23 Nov 2026 (#471): what else 0.3.0 drops

[ADR 0102](0102-what-a-store-release-ships.md) classifies every component a release ships.
Issue [#471](https://github.com/opengwas/opengwasdb/issues/471) carries its drops into 0.3.0,
the same format version as the sidecar, so there is still one backfill. From 0.3.0:

- **`index.sqlite` holds only `completion_quality`**, and ships only where that table does: in
  a Reference-Completed release, a Hybrid's in its Dense Component. The envelope is therefore
  a function of layout, version and completion state (`envelope_at`, `ships_index`). An
  Observed-Only release, and a Hybrid's outer release, carrying one fails `envelope.closed`.
  An `index.sqlite` holding anything else fails `sqlite.provenance_only`.
- **The `metadata` table is gone**: its `dense` layout blob and the unread `schema_version`,
  `n_variants` and `n_analyses`. `manifest.json` is the one copy of the layout the blob held,
  and `layout.recorded` and `hybrid.recordings` read the manifest and the root attributes.
- **`variant_aliases` is gone.** On 0.3.0 the Store Reader never opens `index.sqlite`: an
  rsID the rsID index lacks is a miss, and SQLite is not loaded. On 0.2 the fallback stays
  until #476 retires it.
- **The Top-Hit tiers' `abs_z` and `p_value` are gone.** The validator recomputes both from
  `z` through `opengwasdb.store_reader.stats`. `top_hits.threshold` fails a tier row whose
  recomputed |z| does not clear its threshold, on every format. `top_hits.abs_z` holds a 0.2
  tier's stored pair to the recomputed one: `abs_z` within the 1e-3 it always used, and
  `p_value` within 1e-3 relative.

`completion_quality` and `on_panel` stay, as provenance.

Builds and completion write 0.3.0's tiers. Rebuilding the index of an existing 0.2 release
(`ogdb build-dense-top-hits`, `build-ragged-top-hits`) keeps that release's format, as #313
requires: its tiers get `abs_z` and `p_value` back, recomputed from `z`
(`top_hits.write_stored_p`).

**The rewrite** drops all four in two steps after `alid_sidecar` and `position_sidecar`.
`provenance_index` writes a Reference-Completed component a new `index.sqlite` holding only
`completion_quality`, copied with its schema. `tier_stored_p` drops each tier's pair. Neither copies what it drops. A step
names the paths it drops (`RewriteStep.leaves_out`), and the copy leaves them out, so a later
step that needed one would read it from the source. Each refuses a source whose drop would
change an answer or lose evidence:
- a `variant_aliases` rsID the rsID index lacks;
- a `dense` blob that disagrees with the manifest;
- a tier whose stored pair is not what its `z` implies;
- a table it does not know;
- a consolidated-metadata record over a tier.

The proof also asks about rsIDs no release names, by PheWAS and inside a lookup (its
`misses` section). On 0.2 those reach `variant_aliases`; on 0.3.0 they are misses.

On OGS-00007 (Ragged Observed-Only, 0.2.1, rewritten with the version refusal bypassed, as for
#467), the rewrite took 21.6 s. All 146 questions, 9 ALID probes and 3 misses matched. It left
out 71,412 B: `abs_z` 24,224 B and `p_value` 47,188 B over three tiers, and an empty
`index.sqlite`. ADR 0102's Consequences gives the same drops on the large releases: 575 MB of
tier arrays on OGS-00009, and 2.09 GB on OGS-00011's Dense Component.

## Alternatives considered

- **Widen the fixed slot** (`|S64`, `|S128`). It still leaves some ALID out (OGS-00007 holds one of
  87 bytes, and the fixture indel here is 607), and it costs the width on every row.
- **Hash long ALIDs into the slot.** A hash names no row on its own, so row to ALID would still
  need the table, and a collision is a wrong answer that looks right.
- **Keys of 16 or 32 bytes.** A 16-byte key is the whole ALID only for an ALID under 16 bytes; every
  ALID of 16 bytes or more is confirmed on its full bytes. That is 5.4% of OGS-00007's ALIDs and
  7.4% of OGS-00025's, against 0.50% and 0.13% at 24 bytes. A 32-byte key costs 8 bytes a row more
  to send 0.15% and 0.03% to that compare.
- **Arrow or Parquet.** A dependency the Store Reader does not otherwise need, for two arrays.
- **Rewrite in place.** ADR 0038 makes a release immutable. A derived release can be proved
  against its source, and an in-place edit cannot.
