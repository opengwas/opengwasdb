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
- From #469 (§7): spec §1, §1.1 and §4 drop `variant_offsets.npy` from 0.3.0 and say how a row's
  rsID and the variant count are read without the Variant Table, and give the `.tbi` rebuild
  command; §20.0 adds `variant_table.rsid_rows` and restates `rows` and `representatives`;
  §21.3 and §21.4 add the change to 0.3.0 and to the rewrite.
- From #470 (§8): spec §1, §1.1 and §4 name the `source_alid` sidecar's three files and say
  what a row's `source_alid` is; §20.0 adds `variant_table.source_alid_sidecar` and restates
  `variant_table.source_alid`; §21.3 and §21.4 add it to 0.3.0 and to the rewrite.
- From #471 (the dated section below): spec §1, §1.1, §10a, §18, §20 and §20.0 for what
  0.3.0 drops, and banners on [ADR 0001](0001-hybrid-sqlite-zarr-store-envelope.md) and
  [ADR 0057](0057-store-format-0-2-0-zarr-v3-with-sharding.md) §5.
- From #472 (the dated section below): spec §1, §1.1, §16, §20.0 and §21.4 for the Hybrid
  Dense Component that shares the variant sidecars, and `CONTEXT.md`'s **Dense Component**.
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

### 7. The Variant Table leaves the read path (#469, 24 Nov 2026)

**Context.** After §1-§6 a 0.3.0 query still read `variants.tsv.gz` for two things: the rsID of a
resolved row (`resolve(include_variant_info=True)`), and `variants_table()`. Both went through
pysam: a BGZF seek per row, located by `variant_offsets.npy`, or a whole-table scan. The variant
count was the length of `variant_offsets.npy`. ADR 0102 classes the table a convenience from 0.3.0
and drops the offsets.

**Decision.**

- **A row's identity comes from the sidecars.** Its ALID is the ALID sidecar's. Its chromosome,
  position and alleles are its ALID's, parsed by the existing owner (`identity_by_indices`). Its
  rsID is the rsID index's, inverted (`opengwasdb.store_reader.rsid_index`). The variant count is
  the ALID sidecar's rows. `VariantAxis.variants_table()` and `rsids_by_indices` answer from these
  on 0.3.0, and from the table on 0.2.x, unchanged.
- **How 0.2.x spells a row's rsID, kept.** The table gives a row one rsID cell. A blank cell (`.`)
  is `None` in `variants_table()` and `.` in a resolved row. A cell naming several (a GWAS-VCF
  `ID` such as `rs1;rs2`) is that cell, whole. The rsID index holds each row's cell whole, so its
  inversion gives the same. An rsID naming several rows (a multi-allelic site) names each.
- **The inversion is paid for once it would cost less than the scans.** Until it is inverted, a
  call asking for up to 32 rows scans `variant_rsid_rows.npy` in chunks, which allocates nothing
  the size of the axis. The index counts what its scans have cost and inverts, once, into one
  `int32` per row, kept for later calls, when the next scan would take the total past one
  inversion. A call asking more than 32 rows inverts at once. Measured on OGS-00004 (23,726,262 rows,
  20,008,480 named, page cache warm): the scan takes 0.04 s for 1 row, 0.07 s for 7 and 0.25 s
  for 32; the inversion takes 1.0 s and keeps 95 MB (184 MB at its peak), then any number of rows
  is a gather. Inverting the whole index to name one row would repeat #465's mistake.
- **What the table's rsID is, the index must say.** The reader names a row's rsID from the index
  alone. A new rule, `variant_table.rsid_rows`, checks that the inverted index names every row by
  the table's rsID, and a blank row by none. The 0.2 series left an rsID wider than the index's
  24-byte slot out of the index and kept it in the table. On 0.3.0 that rsID would vanish from
  every answer, so `write_variant_axis` refuses one (`UnindexableRsidError`). None of the six
  releases measured holds one. OGS-00003, OGS-00004, OGS-00007 and OGS-00025 hold 41,789,333 named
  rows of 46,400,721, every one `rs` and digits (`scratch/469/rsid_census.txt`); OGS-00009 and
  OGS-00016 (39.1 M rows) hold none over 12 bytes (`scratch/469-review/census.txt`). The owner
  chose the refusal (24 Nov, #469 review). The caveat: the 0.3.0 rebuild reads sources, not these
  tables, and the GWAS-VCF reader keeps an `ID` cell whole (`readers/gwas_vcf.py`), so a source
  whose cell names several rsIDs past 24 bytes (three of nine digits already are) is refused at
  build.
- **A 0.3.0 release without the rsID index is refused** (`MissingRsidIndex`). Read as it stands,
  it would name no row's rsID. A 0.2 release built before #109 still opens without one.
- **`variant_offsets.npy` is not written from 0.3.0.** A 0.3.0 release carrying it fails its
  closed envelope; a 0.2 Dense release missing it fails `envelope.required`, as before. The Dense
  builder's `provenance.dense.variant_axis.row_offsets` no longer names it, and the `index.sqlite`
  `dense` blob that also named it is gone from 0.3.0 (#471, the section below).
- **The record methods still read the table** (superseded by §8, #470, which moves them to
  the sidecars). `by_index`, `by_indices`, `all`, `records` and
  `range` return every column, `source_alid` included, and only the table holds `source_alid` until
  #470. The builders, completion and the validator call them; no query does. On 0.3.0 `by_index`
  fetches the row by tabix where the position sidecar places it, one fetch a row, so `by_indices`
  and `range` cost one fetch a row there too. Each says so in its docstring.
- **The CLI loads no pysam for a query.** `opengwasdb query-*` imports only what a query needs:
  Reference Completion, whose LD panel reader loads pysam, is imported inside the `complete-*`
  commands, and the variant-axis writer imports pysam when it writes.
- **The validator.** `variant_table.rows` counts against the ALID sidecar on 0.3.0.
  `variant_table.representatives` checks only the tabix index there: the position sidecar places
  the rows, and `variant_table.position_sidecar` judges that row by row. `hybrid.coverage`
  compares ALIDs from the sidecars on 0.3.0.
- **The `.tbi` can be rebuilt** with `tabix -s1 -b2 -e2 -c'#' variants.tsv.gz`. On OGS-00007 the
  rebuilt index decompresses to the bytes of the shipped one, and answers all 170,943 positions
  identically. So does `Rsamtools::indexTabix("variants.tsv.gz", seq = 1, start = 2, end = 2,
  comment = "#")` (Rsamtools 2.28.0), which spec §4 gives too.

**The rewrite step**, `variant_offsets`, fits #471's step frame (the section below). It leaves the
file out of the copy (`leaves_out`), so the steps that stream the staged Variant Table check its rows
against the source's offsets. Its `run` comes last: it refuses unless the staged ALID and position
sidecars each count the rows the source's `variant_offsets.npy` does, and drops the Dense
provenance's `row_offsets`. The `index.sqlite` `dense` blob, which named the file too, is dropped
with the rest of `metadata` by #471's `provenance_index` step. The proof already asked `variants_table()` and the resolved rows of the lookup, the range
and the widest top hits, with variant info, so it needs no new question.

**Evidence.** `tests/test_variant_table_off_read_path.py` gives every fixture rsIDs that plant a row
naming none, two rows naming one, and a cell naming several. These fixtures are the six digest
releases and the three one-source layouts. Each 0.3.0 release answers `variants_table()` and the
resolved rows of every Analysis, a lookup and the top hits at 5e-4 exactly as its 0.2.3 form, scanned
and inverted. In a fresh interpreter, with each table and `.tbi` made unreadable, every question of the
proof is answered without importing pysam, on each layout Observed-Only and Reference-Completed; so is
every `opengwasdb query-*` command, with `--variant-info`.

On OGS-00007 (Ragged, 171,006 variants, 0.2.1, read only, its version refusal bypassed as
for #467), the rewrite runs every step and its proof passes in 18.8 s. The proof asks 146
questions, 9 ALID probes and 22 range probes. The published copy carries no
`variant_offsets.npy`, and `variant_table.rsid_rows` passes on it. Each read below ran in a
fresh process, twice, with the same answer digest on both formats:

| read | 0.2.1 (table) | 0.3.0 (sidecars) | pysam loaded on 0.3.0 |
|---|---:|---:|---|
| `variants_table()`, 171,006 rows | 0.93 s | 0.54 s | no |
| `resolve(analysis, include_variant_info=True)`, 14,894 rows | 0.89 s | 0.12 s | no |

The spec review (`scratch/469-review/`) rewrote two more, read only. OGS-00024 (Ragged, 7,067,555
rows, 1,034 Analyses) published in 849 s, and its proof matched on all 146 questions. Its
`variants_table()` was identical column by column, and so were its resolved rows. With the table at
mode 0, `variants_table()` took 20.5 s on 0.3.0 against 34.0 s on 0.2.1. OGS-00002 holds 37,575 rows
with no rsID. Its proof passed, and its resolved rows carried the same count of `.` in each
Analysis.

**ogdb-api breaks on a 0.3.0 release.** It reads `variant_offsets.npy` and seeks into the
Variant Table through the axis's private attributes. Its owner moves its pin (#450, 23 Nov).

**Alternatives considered.**
- *Invert the whole index on first use, always.* It is simpler, but it costs 1.0 s and 95 MB on
  OGS-00004 to name one row.
- *Keep the rsID in a row-ordered sidecar.* It would make a row's rsID a slice, at the cost of a
  second copy of every name. The inversion's one-off second does not justify it.
- *Write `.` to the table for an over-wide rsID.* The table and the index would then agree, but
  the answer would differ from what 0.2.x gave. Refusing the build loses nothing silently.

### 8. The `source_alid` sidecar (#470, 24 Nov 2026)

**Context.** A row's `source_alid` is the ALID its source spelt it by: for a lifted row the
pre-liftover ALID, the one link from a GRCh38 row back to the dataset it came from. Every
build writes an unlifted row's as its own ALID: the VCF, SSF, BESD and Hybrid builds take
it from their routing, and `ogdb build-dense`, which lifts nothing, writes each row's own
ALID (from #470; before it, that build wrote none). A collision (#85) leaves a row none, and so
does a row Reference Completion added from the panel. After §7 only the Variant Table held
it, in its eighth column, so a 0.3.0 query could not return it, and completion of a 0.3.0
source, the record methods and the source-fidelity join still read the table for it (#449
made completion keep it).

**Decision.**

- **A state a row, and text for lifted rows only.** Three plain `.npy` files beside the
  Variant Table (`opengwasdb.store_reader.source_alid_sidecar`, numpy and the standard
  library only):

  | file | dtype | length | holds |
  |---|---|---|---|
  | `variant_source_alid_state.npy` | `\|u1` | rows | each row's state: 0, none; 1, its own ALID; 2, another ALID, held below |
  | `variant_source_alid_utf8.npy` | `uint8` | total bytes | each state-2 row's `source_alid`, UTF-8, in row order |
  | `variant_source_alid_offsets.npy` | `<u8` | state-2 rows + 1 | the `k`-th state-2 row's is `utf8[offsets[k]:offsets[k+1]]` |

  Why (the owner's ruling, 24 Nov 2026, on the sizes under **Size** below): a release built
  on its own assembly names its own ALID on every row, and a sidecar holding each row's
  text would repeat the ALID sidecar's, byte for byte, and outgrow the table it replaces.
  "Ship only what is read" (ADR 0102) rules out storing it twice. The state costs one byte
  a row; a lifted row's text is stored whole, however long.
- **A state-2 row's text is found by its rank.** The reader gathers the state-2 rows'
  numbers once, from the state, on first use (`SourceAlidSidecar.lifted_rows`), and
  binary-searches them. That costs one scan of a byte a row and an array the size of the
  lifted rows, which is nothing on a release that lifted nothing. A rank computed for every
  row would be an 8-byte array a row, the cost this layout avoids, and a second file of
  state-2 row numbers would restate the state. The first read also refuses a state outside
  0-2, or offsets that do not number the state-2 rows plus one
  (`MalformedSourceAlidSidecar`).
- **The codec does not read the ALID sidecar.** A state-1 row's value is its ALID, which the
  caller passes in: the Variant Axis gives its ALID index's `alids`. Nothing looks a row up
  by its `source_alid` except `validate --source`'s join: it finds a state-1 row by the ALID
  index's search, and scans the text for the rest, a window at a time, by length
  (`SourceAlidSidecar.rows_of`).
- **One writer.** `write_source_alid_sidecar` is it, and it decides each row's state by
  comparing its `source_alid` with its ALID. `write_variant_axis` calls it, before it
  writes the table, so every build and completion writes the sidecar and completion
  keeps each observed row's value through #449's `carried_source_alids`. The rewrite calls
  it directly, as its own step. It refuses a value the table could not hold as the sidecar
  does: `.`, the table's blank, which the table reads as none, and one holding a tab, a
  newline or NUL (`UnencodableSourceAlidError`).
- **The reads.** On 0.3.0 the `source_alid` of `variants_table()`, of a resolved row with
  variant info (`source_alids_by_indices`) and of every record comes from the sidecar.
  `by_index`, `by_indices`, `all`, `records` and `range` name their rows from the sidecars
  alone (`sidecar_records`), which retires §7's last exception: no 0.3.0 read of a row,
  query or not, opens `variants.tsv.gz`. The validator reads the table through
  `written_table_records`, which it checks the sidecars against, and through
  `tabix_range`. Every one of these reads makes the one version split, `reads_table`.
- **The source-fidelity join keeps its liftover fallback.** `indices_by_source_alid` finds
  a state-1 row through the ALID index and scans the lifted text for the rest. When no row
  has a `source_alid`, every state 0, it returns None, as it does for a 0.2 table that
  predates the column: a 0.3.0 release rewritten from such a table holds nothing to join
  on, and `validate --source` then lifts its samples itself (#473) rather than count them
  unmatched.
- **The column is new on every format.** `variants_table()` and resolved rows with variant
  info gain `source_alid`, as they name rsIDs: `None` in `variants_table()` and `.` in a
  resolved row where a row has none. A 0.2 release reads it from its table, so the
  rewrite's proof compares it, and a table written before the column gives none. The
  `opengwasdb query-* --variant-info` TSV names its columns and is unchanged.
- **The version chooses**, as for the other sidecars (`carries_source_alid_sidecar`): a
  0.3.0 release missing the files fails `envelope.required` and is refused at open
  (`MissingSourceAlidSidecar`), never read as naming none; a 0.2 release carrying them fails
  `envelope.closed`. A sidecar whose offsets do not start at 0 and end at the byte array's
  length, or whose arrays are not of their dtypes, is refused at open, and one with a
  state outside 0-2, or offsets that do not number the state-2 rows, at its first read
  (`MalformedSourceAlidSidecar`); the sidecar opens on first use, so in the validator it
  fails `variant_table.source_alid_sidecar` alone.
- **A Hybrid.** The sidecar lives at the root only. A 0.3.0 Dense Component carries no
  Variant Axis (#472): its three files fall out of `envelope_at`'s component envelope, so
  Hybrid completion drops the copy the Dense completion writes
  (`COMPLETED_DENSE_COMPONENT_COPIES`), with no list of its own. `PanelVariantAxis` answers
  each panel row's `source_alid` as its shared row's, in `source_alids_by_indices`,
  `variants_table()`, the records and `indices_by_source_alid`, so a Dense Analysis's
  resolved rows read the shared table's value, which completion keeps, and never the
  Dense copy's, which it left blank (opengwasdb-stores#499).

**The rewrite step**, `source_alid_sidecar`, runs after `position_sidecar`. It streams each
component's table, a Hybrid's Dense Component excepted (it gets no sidecar, and
`dense_shared_axis` drops its table), in row order, checked against the source's `variant_offsets.npy`, writes
the sidecar from its `source_alid` column and each row's ALID, so the states are the table's
(state 0 on every row of a seven-column table), and reads it back: it refuses unless every
row's state, and every lifted row's text, is the table's. That makes the rewrite
stream each table three times; §6's note on the cost stands. The proof needs no new question:
`variants_table()` compares every row's `source_alid`, and the resolved rows of the lookup,
the range and the widest top hits compare the rows those answers reach.

**Validation.** `variant_table.source_alid_sidecar` (spec §20.0) folds over the one streamed
read of the table: each row's state against the one its table `source_alid` and ALID give,
and each lifted row's text against the table's (`differing`, shared with the rewrite's
read-back); the dtypes; every state 0, 1 or 2; offsets that number the state-2 rows plus one
and rise with each, so text is held for those rows alone; and one row per row.
`variant_table.source_alid` (#449) counts sourced rows from the sidecar on 0.3.0.

**Evidence.** `tests/test_source_alid_sidecar.py` plants, in each of the six digest
releases (every layout, both completion states), a collision row, an unlifted row, a lifted
one, and lifted values of 24, 32, 33 and 130 bytes. Each 0.3.0 release returns them through
`variants_table()` and resolved rows, exactly as its 0.2.3 form does from its table. A lifted
Dense and a lifted Hybrid completion keep every observed row's value and give a panel row none.
The codec writes a collision row state 0, an unlifted row state 1 and a lifted row state 2
with its text, and a 169-byte value round-trips whole. The validator fails a sidecar that
makes a collision row lifted, gives a lifted row other text, is one row short, or holds a
state byte of 3. #469's
fresh-interpreter probe now also asks each row's `source_alid` and every record method, with
the table unreadable, and loads no pysam.

Two real releases were rewritten, read only, with the version refusal bypassed as for #467:
- **OGS-00007** (Ragged, 0.2.1, 171,006 rows) in 23.1 s. All 146 questions, 9 ALID probes,
  22 range probes and 3 misses matched, `source_alid` among them in `variants_table()` and the
  5 resolved shapes. Its table holds no `source_alid`, so every row is state 0 and the
  sidecar is 171,398 B, its state and three headers, against 2,492,367 B of
  `variants.tsv.gz`. Read back, all 171,006 rows equal the table's. With the table
  unreadable the rewritten release answered `variants_table()`, the resolved rows of an
  Analysis, `records` and the source-fidelity join in 1.6 s, without pysam.
- **OGS-00001** (Ragged BESD, 0.2.0, which records a lift of 86,376 variants) in 15.5 s. All
  146 questions, 4 ALID probes, 16 range probes and 3 misses matched. It was built before
  #449, so its table holds no `source_alid` either (86,768 B against 1,231,739 B), and
  `variant_table.source_alid` fails it, as it fails the source.

None of the eight hosted releases checked holds a lifted `source_alid`: OGS-00001, -00002,
-00003 and -00007 were read whole, and OGS-00004, -00009, -00015 and -00027 over their first
299,999 rows. Each row is blank or names its own ALID.

**Size.** A release built on its own assembly gives every row its own ALID. Stored as text a
row, the sidecar first written for #470 repeated the ALID sidecar's: on OGS-00003 (Dense,
21,327,043 rows, each its own ALID) it would have been 489,484,138 B, the ALID sidecar's
318,867,530 B of text byte for byte and 170,616,352 B of offsets. That is 23.0 B a row, and
1.34 times the 366,598,962 B `variants.tsv.gz`. OGS-00011's 47,920,778 shared rows would
have cost about 1.1 GB. With a state a row, OGS-00003's sidecar is 21,327,435 B (measured,
read only: every row state 1), 1.00 B a row and 5.8% of its table. OGS-00011's shared rows,
all state 1 (opengwasdb-stores#499), come to 47,921,162 B by the same count, 5.8% of its
823,175,062 B table. A lifted row adds its text and 8 bytes of offset.

**Alternatives considered.**
- *Each row's text, with row offsets* (#470 as first written). 23.0 B a row on OGS-00003,
  most of it the ALID sidecar's text again (**Size**).
- *Reuse the ALID sidecar's codec with keys.* A search nothing asks for, at 28 bytes a row.
- *A fixed-width `|S<n>` array.* An indel's `source_alid` runs past 100 bytes; a width that
  held it would cost it on every row, and a narrower one would truncate.
- *One row index into the ALID sidecar instead of the text.* Most rows of a lifted release
  name an ALID the release does not hold (the GRCh37 spelling), so the text is needed.
- *Keep `source_alid` in the table only.* Completion and the source-fidelity join would keep
  pysam, and a 0.3.0 query could not trace a row to its source.

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
  is the 8-byte offsets. #469 drops `variant_offsets.npy`, 8 bytes per row (§7).
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
`completion_quality`, copied with its schema. `tier_stored_p` drops each tier's pair, and has
no `run`: neither copies what it drops. A step names the paths it drops
(`RewriteStep.leaves_out`), and the copy leaves them out, so a later step that needed one
would read it from the source. Each step's `refuse` hook runs before anything is staged,
and refuses a source whose drop would change an answer or lose evidence:
- a `variant_aliases` rsID the rsID index lacks;
- a `dense` blob that disagrees with the manifest;
- a Reference-Completed component with no `completion_quality`;
- an `index.sqlite` table it does not know;
- a tier storing one of `abs_z` and `p_value` without the other;
- a tier whose stored pair is not what its `z` implies;
- a consolidated-metadata record over a tier.

The rewrite finds a Hybrid and its Dense Component from the root manifest's layout
(`Component`), not from directory names. #470 and #472 add their steps to this frame.

The proof also asks about rsIDs no release names, by PheWAS and inside a lookup (its
`misses` section). On 0.2 those reach `variant_aliases`; on 0.3.0 they are misses.

Two real releases were rewritten, each with the version refusal bypassed as for #467:

- **OGS-00007** (Ragged Observed-Only, 0.2.1) took 21.6 s, before #468's step. All 146
  questions, 9 ALID probes and 3 misses matched. It left out 71,412 B: `abs_z` 24,224 B and
  `p_value` 47,188 B over three tiers, and an empty `index.sqlite`.
- **OGS-00006** (Ragged Reference-Completed, 0.2.1) took 2,183 s, of which 1,976 s were the two
  sides of the proof. All 146 questions, 9 ALID probes, 22 range probes and 3 misses matched.
  It left out 19,012,514 B: `abs_z` 3,452,953 B, `p_value` 14,969,737 B, and the source's
  589,824 B `index.sqlite`. The new one holds the same 7,509 `completion_quality` rows.

On its first run the rewrite refused OGS-00006. In each tier, 226 rows (|z| 37.7 to 40)
stored `p_value` 0.0. Here `store_reader.stats` recomputes a subnormal, 4.1e-311 at |z| 37.7:
the build's scipy `erfc` flushed it to zero, and `math.erfc` does not. Both say p
underflows, so `stored_p_agrees` allows an absolute difference up to float64's smallest
normal number, and no more. Under that rule all 54 tiers agree, over the 18 hosted 0.2
components whose tiers total under 1.2 GB (|z| up to 200.8).

ADR 0102's Consequences gives the same drops on the large releases: 575 MB of tier arrays
on OGS-00009, and 2.09 GB on OGS-00011's Dense Component.

## 24 Nov 2026 (#472): the Hybrid Dense Component shares the variant sidecars

[ADR 0102](0102-what-a-store-release-ships.md) §4 removes the Hybrid duplication: a 0.2
Hybrid's Dense Component carries its own copy of the shared Variant Table's rows, of every
search index over them, and of `analyses.tsv`. Issue
[#472](https://github.com/opengwas/opengwasdb/issues/472) carries it into 0.3.0, the same
format version, so there is still one backfill.

**What the Dense Component carries from 0.3.0.** `manifest.json`, `data.zarr/`,
`dense_to_shared.npy` and, when Reference-Completed, `index.sqlite` with its
`completion_quality` (`SHARED_AXIS_DENSE_COMPONENT_ENVELOPE`, `envelope_at`). No Variant
Table, `.tbi`, ALID, position or rsID sidecar, `analyses.tsv` or `overview.html`. What is
dropped is derived from the envelopes, never listed again: `DENSE_COMPONENT_COPIES` is the
0.2.3 component's envelope less the 0.3.0 one's (what the rewrite leaves out), and
`COMPLETED_DENSE_COMPONENT_COPIES` a standalone 0.3.0 Dense release's less it (what
completion drops). Each would have been one more file per sidecar for #467, #468
and #470 to write twice, and on a completed Hybrid the copy is already wrong (#481).

**How it is read.** One function decides that a release is a Dense Component that shares
the axis, `shared_axis_root`: the version chooses, as everywhere in 0.3.0
(`dense_component_shares_axis`), and the component's place says it is one (`hybrid_root_of`:
the `dense/` directory of a release whose manifest says `hybrid`; a parent manifest that
exists but cannot be read is refused, `UnreadableHybridManifest`). `StoreRelease`, the
Variant Axis's guard, the rewrite and the validator ask it. It reads the filesystem, so it
changes when a staged root's manifest lands; completion writes that manifest before it
drops the copies, and nothing opens the component in between.

- `StoreRelease.variant_axis()` of such a component is `PanelVariantAxis`
  (`opengwasdb.store_reader.panel_axis`): the shared axis seen through the map. A panel row's
  variant is the shared row the map names; a shared row's panel row is a binary search of the
  map, `hybrid.panel_rows`, the one search of it, which the query facade asks too. It answers
  what a `VariantAxis` answers, in panel rows, without inheriting from it; readers are typed
  `AnyVariantAxis`, so each is checked against both. It refuses
  (`DenseComponentSharesAxis`) to hand out an index the component does not carry. Its
  `analyses_path` is the root's. So completion, the validator and the Indexed Variant Subset builder, which open the
  component as a release, read the shared copies without knowing it.
- `HybridStoreQuery` resolves an identifier and a range once, against the shared sidecars,
  and reaches Dense rows through the map. `phewas` and `lookup` already did (#465);
  `range_phewas` now does too. Its nested Dense query is handed the shared Analyses and a
  `PanelVariantAxis` over the facade's own shared axis, on every format, so it opens no axis
  of its own.

**What `dense/analyses.tsv` held that the root does not.** Two columns differ. Its Top-Hit
Counts count the Dense Component alone (#107); nothing reads them. Its
`n_missing_se_out_of_range` counts the associations left out of the Dense Component alone
(ADR 0068). Two things read that:
- Hybrid completion carried the source Overflow's count forward as the root's less the
  component's. On 0.3.0 the Dense completion reads the root's Analyses through the
  component, so it carries the root's count, both components', into the completed Dense
  Component's, and the source Overflow's count is already there: the completed root counts
  what a completion of the 0.2.3 form counts (`_root_analyses`).
- `analyses.se_out_of_range_partition` held the root's count at least the component's. On
  0.3.0 there is nothing to compare, and it does not judge the component.

The table rules (`analyses.*`) judge a 0.3.0 Hybrid's root `analyses.tsv`, its one copy of
the Analyses; before 0.3.0 they judged the Dense Component's copy and never the root's, and
they still do.

**Builds** write the map and no copy. The build's #255 read-back of the Dense Component's
table against its names becomes a read-back of the map against the shared ALID sidecar
just written (`require_panel_mapped`). The map's invariants are stated once,
`check_dense_to_shared` (in range, strictly ascending, one entry per Dense row), and the
comparison of each Dense row's ALID with its mapped shared row's once, `misnamed_panel_rows`:
the build's read-back, the rewrite's refusal and the validator ask both. The single-pass build's rewrite of the Dense table
when the global rsID rekey moved a panel name is gone with the table.

**Completion** runs the unchanged Dense completion on the component, as before. It writes
the completed component as a standalone Dense release, own Variant Table included, and the
Hybrid completion reads the completed panel from it. Once the shared table and the map are
written, the map is read back against the shared ALID sidecar, and the copies are dropped
(`_drop_dense_copies`). Writing them to drop them costs one Variant Table write of the
panel: 9.5 ms of a 3.0 s fixture completion, and on OGS-00011's 14,267,257-row panel 94 s
to write 1,371,261,555 B and 0.08 s to drop them, against that release's 73,388 s
completion (`scratch/472/copy_cost.py`). Not writing them would mean a second Dense completion path, which the module's
design (one Dense completion, reused unchanged) is there to avoid.

**The rewrite step**, `dense_shared_axis`, runs before `variant_offsets`, which stays last
to count the sidecars the others wrote. It leaves the Dense Component's copies out of the
copy, and the earlier steps write it no sidecar
(`Component.carries_axis`). Its `refuse` hook streams both Variant Tables of the source side
by side, once, and refuses unless the map names one shared row for each Dense row, in range,
strictly ascending, and holding that row's ALID. It does not compare the copies' rsID or
`source_alid` columns: on a completed release the Dense copy lost some of each (#481), and the
shared table, which every query already reads, kept them. The proof needs no new question:
every question it asks of a Hybrid now reaches the Dense Component through the map.

**Indexed Variant Subsets** (fix round 1). A subset records the release it was built
from, its `source_release_id` and `source_format_version`, so that a reader refuses one
built from another release (ADR 0061). The rewrite mints a new `release_id` and
version, so a subset copied unchanged was stale: OGS-00004's `hm3` failed
`subsets.indexed`, and the live OGS-00011's would have failed after its proof. ADR 0061
makes a subset a non-authoritative derived artifact, rebuildable from the release and the
variant list. The rewrite changes no Store Variant Index and no stored code, so the subset
holds the same cells of the new release, and its step `indexed_subsets` renames it rather
than rebuilding it: it writes the two attributes (`restamp_indexed_subsets`) and keeps
the rest. It refuses a source whose subset already names another release, and never
freshens a stale one. The proof now asks each subset the spread Analyses, Observed-Only
and not, on both sides, and `subsets.indexed`, which compares every decoded cell of the
subset with the primary planes, judges the result. Rebuilding was the alternative; it
costs a full build of each subset (956 s for `hm3` on the completed OGS-00011) and
proves nothing the comparison does not.

**Validation.** The `variant_table.*` rules do not judge a 0.3.0 Dense Component: it has no
table. `hybrid.dense_to_shared` judges the map against the shared axis (in range, strictly
ascending, no row twice, each by its own finding); `hybrid.coverage` against the Dense side,
which from 0.3.0 is only the panel, the rows of its grid: one map entry for each. Before
0.3.0 the Dense side is still the component's Variant Table, whose representative rows' ALIDs
are compared, as they were.

**Evidence.** `tests/test_hybrid_shared_axis_0_3_0.py`. Both Hybrid digest releases,
Observed-Only and Reference-Completed, carry only the four entries above, validate clean,
and refuse a stray copy (`envelope.closed`). Their Dense axis answers, row by row, what the
shared axis answers at the mapped row. Each corruption of the map fails one rule by itself:
a row out of range, a row named twice and a descending pair fail `hybrid.dense_to_shared`,
a map one row short fails `hybrid.coverage`. A Hybrid completed from its 0.3.0 form, from
its 0.2.3 form, and rewritten from the 0.2.3 form of that completion, answer every
question of the proof identically. The rewrite drops the copies, refuses three maps that
misname a Dense row, and publishes a source whose Dense copy has lost every rsID, naming
each row by the shared table's. After fix round 1 the module holds 26 tests; on the code
before #472, 22 fail and the 3 sharing one rewrite fixture error in it. The one that passes,
a 0.2 map naming another variant, holds behaviour that was already there.
`test_rewrite_release` proves both Hybrid digest releases through the rewrite, as before.

**OGS-00004** (Hybrid Observed-Only, 0.2.2, 23,726,262 shared rows, 13,718,605 panel
rows, 10 Analyses) was rewritten read only, its version refusal bypassed as for #467
(`scratch/472/real_rewrite_472.py`). The proof asked 142 questions, 9 ALID probes, 20
range probes and 3 rsID misses, and every answer matched; it took 715 s on the source and
487 s on the copy, and 1,662 s in all, at 20 GB peak RSS. The copy's
`hybrid.dense_to_shared`, `hybrid.coverage` and closed envelope pass; it fails only what
the bypass brings (a 0.2.2 `analyses.tsv` has no `n_missing_se_out_of_range`) and its
`hm3` Indexed Variant Subset, which named the source's `release_id` and version; the
`indexed_subsets` step now carries it over (above). Rewritten again with it (fix round 2,
`scratch/472/real_rewrite_472_v3.py`, 1,264 s), the proof matched on every question, the
16 subset questions included; the whole validator, `subsets.indexed` included, passed the
copy but for the `n_missing_se_out_of_range` column the 0.2.2 source lacks; and `hm3`
named the new `release_id` and 0.3.0 with its `builder_version` and `created_at` kept. In
the same run the package before #472 (the integration tip, `34fa74c5`, in its own
interpreter, its path asserted) digested the source: it and this package reading the copy
agree on all 142 questions, both `range_phewas` included, the 9 ALID probes, the 20 range
probes, the 3 misses and both tables (`scratch/472/OGS-00004.tip-vs-472.txt`). That, and
the unchanged golden digests of the two Hybrid fixtures, which the pre-#472 reader pinned,
are the comparison of the old read path with the new. The Dense Component's
0.2 copies came to 1,098,940,214 B, the figure ADR 0102 §4 gives. A 0.3.0 component that
kept its copy would carry 1,265,761,783 B: its table, `.tbi`, rsID index, `analyses.tsv`
and `overview.html` as the source holds them, and the ALID and position sidecars sized
from the shared ones at the mapped rows (`scratch/472/dense_bytes.py`).

**OGS-00011 as completed** (Hybrid Reference-Completed, 0.2.0, the release #481 describes,
now under `replaced/20261122T224238Z`; 48,069,918 shared rows, 14,267,257 panel rows,
3,169 Analyses) was rewritten the same way, read only. Its whole proof is too large for
it (ADR 0105 §3): the one broad `range_phewas`, from the first variant of chromosome 1 to
the last variant's position, would read about 600,000 panel rows by 3,169 Analyses on each
side. So the comparison ran outside the proof, as the #472 ruling preferred to a named
allowance (`scratch/472/real_rewrite_472_v2.py`): every other question was asked of both
sides, 140 questions, 9 ALID probes, 20 range probes and 3 rsID misses, in 3,693 s and
2,448 s at 57 GB peak RSS, and **no answer changed**. The rewrite then published, and the
rules it touches (the envelope, `hybrid.*`, `variant_table.*`, `sqlite.provenance_only`,
the Analysis ids) pass on the copy; a whole validation of this release takes 9.7 h on one
worker (its `completion-validate` record), so the rest was not run. The Dense Component's
0.2 copies came to 1,175,803,588 B, ADR 0102 §4's figure; a 0.3.0 component keeping its
copy would carry 1,347,866,437 B.

**No digest changed on the rsID lookups #481 lost**, because the miss was already gone
from the read path: #465 made `HybridStoreQuery.lookup` resolve identifiers against the
shared axis, and `phewas` always had. The loss is still in the 0.2 Dense copy, and the
rewrite drops it. Of the 850,273 Dense rows whose copy names no rsID where the shared
table does, 200 spread over the axis were asked by rsID and by ALID, as a PheWAS and as a
lookup over five of their Analyses, of the source read with this package and of the
rewritten copy (`scratch/472/rsid_481_check.py`). All 200 gave the same answers on both,
and by rsID the same as by ALID; `rs761832497`, #481's example, gives 3,131 rows to the
PheWAS and 5 to the lookup either way.

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
