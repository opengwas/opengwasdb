# Validation Rules over shared scans

Issue [#55](https://github.com/opengwas/opengwasdb/issues/55), part of epic #58.
Records how `validate` is organised: as named rules, judged against a release's
targets, with the rules that read association values sharing one streamed read.

## Context

`validate` was three per-layout orchestrators, Dense, Ragged and Hybrid, in one
module of 4,434 lines, each a sequence of checks that appended messages to
one shared error list. That shape caused three problems, and none of them raised
an error.

- **No rule could be named.** A test asserted on message substrings, so it
  could not tell "this rule failed" from "some rule with a similar message
  failed", nor see a collateral failure beside the intended one. A caller
  could not run one rule.
- **One failure hid others.** 29 checks returned early when the shared list was
  non-empty. A bad vocabulary skipped the Dense band pass, the Top-Hit pass and
  rho. Any error skipped the z overflow-table rule. A Dense Component error
  skipped the Overflow. A release with two independent defects reported one.
- **Planes were read and decoded several times.** A full validate read a Dense
  grid's `se`, `eaf` and per-variant arrays three times and its imputed mask
  six. On a Ragged release it decoded the EAF twice (once to predict SE, once to
  range-check it), each time gathering `eaf_baseline` per association from
  Zarr. It also read `variant_index` once per check that needed it. The
  same rule was written once per layout, and the copies drifted. The Ragged
  Top-Hit check sampled 1,000 cells where the Dense one was exhaustive, the
  Overflow's frequencies were never checked, and the Ragged release's Store
  Variant Table was never validated.

## Decision

**A Validation Rule is a named judgement of one kind of target**
(`validation/rule.py`):

- A `Rule` has a stable `name`, the `kinds` of target it judges (`RELEASE`,
  `GRID`, `CSR`), a `when` predicate, the rules it runs `after`, and a
  severity.
- A `Check` reads what it needs and yields messages.
- A `Fold` is judged from a shared `Scan`. It `measure`s each block, the measures
  are added, and it `judge`s the total.
- `Skip` says a rule cannot judge this target. It is recorded with its reason,
  and it is not a pass.

**A release is judged as targets** (`validate.release_targets`):
- a Dense release is a release and its grid;
- a Ragged release is a release and its association sequence;
- a Hybrid release is its outer release, its Ragged Overflow, and, when it
  opens as a Dense release, its Dense Component's release and grid.

A prerequisite is looked up on the target, then on the targets it belongs to. A
prerequisite that does not apply to a target constrains nothing.

**The runner** (`validation/run.py`) runs the rules in registration order on
every target each applies to:
- A rule whose prerequisite failed or was skipped is skipped, with the names that
  blocked it. A selection runs the named rules and their prerequisites, and an
  unknown name is refused before anything is read.
- An exception fails the rule that raised it, as `validation failed: …`, after
  whatever it found. It fails no other rule. That holds for a rule's `when` too,
  which is asked only once the rule's prerequisites passed: it may read what a
  failed prerequisite found broken. A rule blocked there is recorded as skipped,
  whether or not its `when` would have accepted the target. A prerequisite on a
  fold's own scan is the exception, below: it is decided after the read, and the
  fold's `when` is asked before it.
- Findings come out in rule order, then target order.
- A scan is read once per target, for every fold on it that can run there. A read
  that fails part-way is reported by the first fold still measuring, and the others
  are skipped. A plane decoded lazily that cannot be decoded (`Unreadable`) is
  reported by the first fold that reads it, and blocks only the folds that read it
  after. So an undecodable SE no longer hides the side table that explains it. A
  plane whose reason another fold on the scan already reports names that fold, and
  its readers are skipped behind it: an undecodable EAF is reported by
  `side_tables.eaf_exceptions`, which names the cell, and `associations.eaf_range`
  is skipped behind it. Selected without that fold, the reader reports the plane.
- A fold's `after` may name an earlier fold on the same scan, and the registry
  refuses a later one. That prerequisite is decided when the pass ends. Every fold
  measures in the one read; then the folds are judged in registration order, and
  one standing on a fold that failed or was skipped is skipped behind it, its
  `judge` not called. So a prerequisite means one thing everywhere: a selection
  takes it, and its failure skips what stands on it.
  `variant_table.representatives` runs after `variant_table.rows` this way: its
  rows are chosen from the offsets, so it stands on the row count.
- A rule with nothing to judge raises `Skip`: `associations.eaf_range` on a
  component holding no frequencies records "no frequencies", not a pass.
- `fidelity.source` runs after every error rule before it, and is stopped only by
  one that failed (`only_failures_block`), as source fidelity ran only on a release
  with no error before. Each skip among its prerequisites stands behind a failure
  that is also one, or behind nothing to judge.

**Three scans** stream what the value rules judge:
- `ASSOCIATIONS` reads a Dense grid in row bands and a sequence in windows of
  consecutive cells, under one 1 GiB decoded budget (#254). Each block reads
  every plane's codes once. It decodes EAF through the plane classes, given the
  block in memory. SE is decoded from the block's own EAF, and Z and SE only on
  first use.
- `VARIANT_TABLE` streams the Store Variant Table once.
- `BY_VARIANT` reads a variant index's `analysis_index` once.

A `Block` is the seam: `dense_bands`, `sequence_windows` and `Block.of` for tests.

**One rule for both layouts.** Where a Dense rule and its Ragged twin judged the
same thing, they are one rule, and each layout keeps the words it has always
reported in. "Ragged Overflow se contains negative finite values" and "se
contains negative finite values" are one rule, `associations.se_negative`.

**The registry is the contract.** Spec §20.0 lists every rule by name, in the
order it runs, and `tests/test_validation_registry.py` holds the two together.
`validate_store(path, rules=…)` and `ValidationResult.failed()`, `of()` and
`skipped` reach the names from Python.

**`validate --format json` did not change (D1).** Its keys stayed `ok`,
`errors` and `warnings`, and its messages stayed byte-identical: opengwasdb-stores'
`register.validate_verdict` refuses any other key. Emitting rule names there was a
two-repository follow-up: `register` first accepts an optional `findings` key,
then this package emits it. That is #378, below.

**The follow-up (#378) is this repository's half of a two-repository change.**
`validate --format json` emits a fourth key, `findings`, naming each record's rule,
its severity (`error`, `warning` or `skip`), the target the rule judged, and the
message it always had — or, for a skip, the reason recorded above. `ok`, `errors`
and `warnings` still say what they said. `ValidationResult` keeps one ordered record
list with `errors`, `warnings`, `skips` and `skipped` as projections, so what the CLI
prints and what Python sees cannot drift apart. Spec §20.1 fixes the shape.

**The stores half is a dependency, not a fact here.** opengwasdb-stores'
`register.validate_verdict` accepts exactly `ok`, `errors` and `warnings` at the
revision this landed, so it refuses a record an engine with this change writes. It
must gain `findings` — accepting a record with or without the key, and checking the
key's items and their agreement with `errors` and `warnings` when it is present —
before such an engine is pinned there. The two halves are ordered: the reader first,
then the pin. Until then, this package's own readers are the only ones that see the
key.

**Clarification: the CLI did not list skips until #378, and its "valid" stays
truthful.** `validate` printed `ok`, `errors` and `warnings`, and no skips; listing
them was a follow-up, and #378 lists them. In a full run, the only run the CLI makes, a rule
is skipped with no error recorded only where it has nothing to judge. A skip is
recorded in these places:
- `_Run.ready`, behind a prerequisite, and `_Run.conclude`, behind one on the
  fold's own scan. A prerequisite that failed recorded an error. One that was
  skipped is one of these cases in turn.
- `_Pass.drive`, behind a read that failed part-way. The first fold still
  measuring reports that read as an error.
- `_Pass.unreadable`, behind the fold that reports an undecodable plane as an
  error. That fold is the first reader, or `side_tables.eaf_exceptions`, which
  reports every band it measured that could not be decoded.
- `_contained`, where a rule raises `Skip` before reporting anything. Three do:
  - `associations.eaf_range`, "no frequencies": the component has no `eaf`
    plane and no reference panel.
  - `by_variant.present`: a format 0.1.0 Ragged component without the
    variant-centric index, which spec §11.1 permits. The eight `by_variant.*`
    rules after it are skipped with it; there is no index for them to judge.
  - `sqlite.variant_aliases`: an `index.sqlite` with no alias table, which
    spec §20 lists as a skip.

A `when` that declines a target records no skip: the target is not one the rule
judges. `fidelity.source` is stopped only by a failure, not by a skip. A
selection takes every prerequisite of what it names, so the same holds for a
selection: `variant_table.representatives`, selected alone, brings
`variant_table.rows`, which reports a row-count mismatch and skips it.
`tests/test_validation_skips.py` holds the rules that raise `Skip`, and the
rules their skips stop, to this list.

**Admission's witnesses.** Six rules check, after the fact, what Association
Admission (ADR 0072) guarantees of every cell it admits. Each records the
Admission `Reason`s whose drops keep a cell it would fail out of a Store
(`Rule.witnesses`):
- `associations.markers_paired`: `se_unusable` and `effect_missing`;
- `associations.markers_canonical` and `completion.imputed_present`:
  `z_nonfinite` and `se_unusable`;
- `associations.se_negative`: `se_unusable`;
- `csr.segment_order`, its repeat case: `duplicate_collapsed` and
  `duplicate_conflicting` (and `duplicate_reversed_indel`, until #316 retired
  it);
- `completion.mask_binary`: none. It witnesses the cell guard's flag rule, which
  no row's `Reason` names.

A rule naming a reason Admission lacks is refused at import: the reasons are read
from Admission's `Reason` enum, and `tests/test_admission_witnesses.py` holds the
witness list to it. The rules and Admission deliberately do not
share predicates where Admission is stricter than published Stores. Admission
requires SE > 0 and EAF in (0, 1); `validate` accepts SE ≥ 0, warns on `+inf`,
and accepts EAF in [0, 1].

## Consequences

- **Rules run after unrelated failures (D4).** Only failing Stores change.
  Every published Store's `validate --format json` was byte-identical across #55's
  change, except #380's three 0.1.0 releases (below). #378's key is a later change,
  and the JSON object gained a key there: the three keys #55 left alone still say
  what they said, and a reader that enumerates the keys must accept `findings`.
- **Coverage gained (D3).** These checks held on one layout only, and now hold
  on all:
  - a Ragged release's Store Variant Table;
  - a sequence's Top-Hit Index, every cell, completeness included, where it was
    a sample of 1,000;
  - a completed Ragged release's declaration against its mask, and its
    `completion_n_missing_total`;
  - an Overflow's frequencies.
  Every published Store passes them: OGS-00001, -00002, -00006, -00007,
  -00027, and OGS-00011's Overflow.
- **A 0.1.0 Ragged component need not carry the variant-centric index (#380).**
  Spec §11.1 requires it in 0.2.0. OGS-00004, -00005 and -00007 now pass.
- **Reads.** The value rules read each plane once per target, and tests hold
  them to it by element count. On a completed Ragged release, a full validate
  read `z`, `se` and `eaf` 2.17 times, `imputed` 3.17 times and `eaf_baseline`
  3 times, and now reads them 1.17, 1.17 and 1.0 times. The remainder is
  `by_variant.sample`'s cells.
- **Time and memory**, measured against the release before. RSS is read from
  the validating process alone. Wall times are from a shared node.
  - OGS-00006, a Ragged release, two runs each, alternating: 52–53 s against
    55 s, and peak RSS from 3.8 GiB to 1.6 GiB.
  - OGS-00008, a Dense release, two runs each, alternating: 41–44 s against
    72 s, and peak RSS from 0.76–0.80 GiB to 0.56 GiB.
  - OGS-00011, a Hybrid release, side by side and without the Indexed Variant
    Subset check, which did not change: 13,821 s against 13,230 s, and peak
    RSS from 113.8 GiB to 46.0 GiB. That check is 30,209 s of an instrumented
    45,132 s run. A full run's peak is not measured here.
  - OGS-00016, a Dense release, full runs side by side, with the kernel's
    exact high-water mark read as each phase starts and ends: 16,826 s against
    18,814 s, and peak RSS from 14.56 GiB to 14.32 GiB. The Indexed Variant
    Subset check sets the peak in both, starting from 3.07 GiB and 2.38 GiB.
    Before it, the highest point rose from 9.42 GiB to 10.52 GiB. Both are the
    same whole read of the eaf exception table, 6.4 GiB above where it starts.
    It starts higher, 4.11 GiB against 3.02 GiB, because the association
    scan's folds are judged before the 1.6 GiB the scan holds, chiefly the
    Top-Hit Index's tiers, is released.
- **SE decoding.** ADR 0070's list of `validate`'s own SE decodes is now two:
  `_dense_band_se` and `_sequence_window_se` in `validation/scans.py`, one decode
  per block, shared by every rule. The whole-sequence Ragged decode is gone. The
  guard (`tests/test_association_reads_guard.py`) checks every module under
  `validation/` that is not a reader, so a new rule module cannot decode SE
  unseen.
- **One message is kept that a JSON change could drop.** A completed Ragged
  release with an imputed cell missing Z reports "missing z rows have imputed=1
  (inconsistent)" after "imputed=1 rows have missing z-scores", the same count
  in other words. Dropping it changes `validate --format json`, which #378 can
  do.
- **What stays as it was.** `subsets.indexed` wraps the Indexed Variant Subset
  validation the build's publication gate runs over a sample (#270); each subset
  is now judged on its own findings. Source fidelity is `fidelity.source`, its
  read untouched in one function (#56). `check-observed-view` is not a rule set
  here (#375).

## Amendment: no rule reads the Dense EAF exception table whole (issue #387)

Three rules read that table, and each read it whole or once per band. The table is
one Zarr v3 shard of hundreds of millions of entries -- 237,027,522 on OGS-00016,
1,136,991,041 on OGS-00011 -- so `[:]` is 2.7 GiB and 12.7 GiB of `int64` plus
`float32`.

- `encoding.eaf` loaded it to check that positions pair with values, are sorted and
  unrepeated, and hold frequencies.
- `subsets.indexed` loaded it to prove the index's copy was the primary's,
  remapped, and `_remap_side_table` held several copies of it at once; the value
  comparison then had the plane consult the table again for every band.
- `side_tables.eaf_exceptions` held the flat position of every marked cell for the
  whole association scan -- 1.14 billion of them, 8.5 GiB -- and its judge read
  the whole table beside them, sorted them and set-differenced the two.

**No consumer reads it whole.** The shard's inner chunks are 200,000 entries and
Zarr reads them as byte ranges, so each rule streams what it needs and holds a
page at most -- about 96 MB for the subset check, one page of 262,144 entries for
the plan rule and the scan (all inside the 1 GiB budget above):

- `encoding.eaf` walks the table a page at a time, carrying the last position
  across the boundary, so sortedness, uniqueness and the value range are judged
  where they were and its three findings keep their order.
- `subsets.indexed` keeps the cells whose variant the subset holds -- 21,157,534
  of them on OGS-00011's HapMap3 index -- compares that set with the index's own
  table, and hands it to the value comparison keyed by the primary's flat
  positions, so the table is not consulted again per band: 1,128 bands on
  OGS-00016's HapMap3 subset, 2,114 on OGS-00011's.
- `side_tables.eaf_exceptions` compares each block's marked cells against the span
  of the table that block must occupy, as the scan reaches it. A block's cells are
  one contiguous range of flat positions and the table is sorted by them, so its
  entries are one contiguous span starting where the previous block's ended:
  nothing but the running counts, the first position of each kind and one page's
  tail is held.

A table that is not sorted and unique, or whose values are not parallel with its
positions, is the subset's finding rather than an exception that ends the rule
before the entries after it are judged.

**A pass still streams the table several times** (superseded by the amendment for
issue #314 below, which reads it once a pass), which is not the same as reading
it whole: once in `encoding.eaf`, once in `side_tables.eaf_exceptions`, once in
`subsets.indexed`, and once per band in the association scan's own decode. Each
stream is bounded, and each rule needs its own view. Merging them into one pass is
#314's, which moves the subset comparison onto the scan's bands; the subset check's
own read is once per Indexed Variant Subset today, and a release carries one.

The per-band consult the association scan makes when it decodes a band stays: a
Dense band's exception cells are one contiguous span of the sorted table, so a
band's lookup is one read of it already. A Dense grid reads the table in 4,756
bands on OGS-00011, about 239,000 exception cells a band.

A shared scan's folds are also judged after, not before, the 1.60 GiB the plain
association scan holds for them is released -- chiefly the Top-Hit Index's tiers --
so a judge words its findings from its total and from each tier's key and cell
count, which are metadata, and reports exactly what it did before.

Measured, each side's peak read from its own `VmHWM`:

| | base `df000b4` | head, round 1 | head, round 2 |
|---|---|---|---|
| OGS-00016's HapMap3 subset, the subset rule's high-water mark | 12.12 GiB | 0.60 GiB | |
| OGS-00011's HapMap3 subset, the subset rule's high-water mark | 57.37 GiB | 1.60 GiB | |
| OGS-00011's HapMap3 subset, one comparison band | 26.4 s | 14.6 s | |
| OGS-00016, a full `validate`, wall time | 21,056 s | 20,436 s | |
| OGS-00016, a full `validate`, peak RSS | 14.55 GiB | 3.92 GiB | |
| OGS-00011, a full `validate`, wall time | 40,998 s | 33,219 s | 32,279 s |
| OGS-00011, a full `validate`, peak RSS | 68.53 GiB | 18.15 GiB | 8.30 GiB |

The two full runs on each Store were started at the same moment, so they saw the
same load; the round-2 run on OGS-00011 is head alone, against a base figure
measured on the same Store. Base's OGS-00016 peak rises from 4.88 GiB at 16:59:16 to 14.55 GiB at 16:59:46, a
window that holds both the side-table judge's whole read and the subset check's —
16:59:27 to 16:59:39, 12.12 GiB on its own — so the samples cannot say which of
the two took it over the other. Round 1's peaks are reads of the table: 3.92 GiB
is the plan rule's read of the 2.65 GiB table on OGS-00016, and 18.15 GiB its read
of the 12.7 GiB one on OGS-00011. After round 2 pages that read, nothing head does
reads the table whole: its 8.30 GiB is the Dense association scan's own working
set, 8.10 GiB of it reached two minutes into that scan and 0.2 GiB gained over the
four hours that followed.

`validate`'s report is the same to the byte across base and head on OGS-00011 and on
every published Store and fixture this change was run over; on OGS-00016 it differs
by one record, 355-build's `side_tables.se_exceptions_rounded` warning, which
`df000b4` predates. **The one report that did change is a release no earlier
version could judge.** A release whose own EAF exception table is disordered,
duplicated or unpaired used to abort the rule, in `EafExceptionTable.__post_init__`,
with `validation failed: eaf exception table index must be sorted and unique` or
`… index and value must be parallel 1-D arrays`, and `_eaf_reference` never ran. It
now gets a finding — `… eaf exception table is not sorted by position, or repeats
one`, or `… eaf exception table has 3 positions but 2 values` — and the entries
after it are still judged, which is round 1's "a corrupt table is a finding, not an
abort".

The paragraphs in "Time and memory" above that call the subset check's table read
the peak of a full run, and that call the scan's held folds the reason its later
stage starts higher, describe the code before this amendment.

## Amendment: a check rides the association scan, and the scan reads the table once (issue #314)

Two passes crossed a Dense grid's statistic planes: the association scan, which every
value fold and the Top-Hit Index verification share, and the Indexed Variant Subset
comparison, which read the primary planes again for each unit it compared -- an index
write band by a tile of Analyses. On OGS-00009 the comparison was 11,030.5 s of a
16,973.9 s validation and the scan 5,886.3 s (issue #308's phase report). Beside them
the release's EAF exception table was streamed four times a pass: by the scan's
per-band lookup, `side_tables.eaf_exceptions`, `encoding.eaf` and the subset check.

**A check may ride a scan** (`Ride`). It boards before the pass over a target, is
handed each block after the folds, and is judged as a check in its own place in the
registry, from what the pass left on the target. A ride that fails, and a check that no
pass carried -- a selection without the scan's folds, or a target on which no fold is
ready -- reads for itself, exactly as before. A ride that hands a check the data itself,
as `encoding.eaf`'s does, saves a read and changes no finding. `subsets.indexed`'s
hands it digests of the planes instead, and its guarantee is probabilistic, below.

**`subsets.indexed` rides the association scan.** The pass hands each band to every
published subset's ride, which folds the subset's cells of the band into one 64-bit
digest per comparison unit: per cell, MurmurHash3's finaliser over the cell's key and
then each compared plane's value -- decoded Z, SE and EAF and the stored imputed mask,
with every NaN and -0 folded to one pattern, as `np.array_equal(..., equal_nan=True)`
compares them -- summed over the unit. The rule then reads only the index, digests each
unit the same way, and compares. A unit whose digests differ, that either side could
not decode, or whose index is wrong on its own (Z and SE missing apart, a mask that is
no mask) is compared cell by cell by the comparison it always used, in the order it
always walked, so the finding, its words and the unit it comes from are unchanged. The
pass also hands each ride the release's exception table, page by page, of which it
keeps the subset's cells, and the subset's share of `eaf_baseline` and `eaf_reference`.

**The cost is a collision, so the guarantee is probabilistic.** Each step of a cell's
digest is a bijection, so a change to one value of one cell always changes its unit's
sum and is always found. A unit in which several cells or planes differ is missed only
if the two sums are equal, about 2^-64 per unit for an accidental corruption. MurmurHash3's
finaliser is public and invertible, so under a fixed key a colliding corruption could be
built in advance (the review built a two-cell one in 13 s); the key is therefore drawn at
random once per run (`run_key`), and no finding depends on it. **The owner ruled on
12 Nov 2026 to keep the full digest**: every subset cell stays covered by its unit's
digest, with no sampling and no exact hold of the index, keyed per run, and the
comparison logs, per subset, how many units it compared cell by cell and why, so a
digest that never matches shows as a number rather than only as lost time. That was
weighed against the two exact designs:

- *Hold one side.* The index is Analysis-major in inner chunks of `[1, 65,536]`, and the
  scan reads primary bands of a few thousand rows across every Analysis, so pairing
  cells exactly means holding a whole index chunk-column for every Analysis: 2,514 x
  65,536 cells x 5 bytes of codes, 824 MB, on OGS-00009, and 1.04 GB on OGS-00011's
  Dense Component, beside a decoded band of 1 GiB.
- *Re-read the index per band.* Every index chunk would be decoded once per band its
  slots fall in, about 150 times on OGS-00009.

**The table is read once a pass** (`TablePages`). Each block takes its span of the
sorted table as it decodes -- every entry its cells can name, read a page of 262,144
entries at a time -- in place of the per-band lookup, which also read the two 200,000-
entry inner chunks it bisected to. `side_tables.eaf_exceptions` reconciles the block
against that span, and `encoding.eaf` and every subset's ride watch the pages go by;
the entries past the last block are read for them once the blocks are.

**`encoding.eaf` waits for the pass.** It is registered before the scan's first fold
and decides whether the value folds run, so where the table is there to walk it is
decided when the read ends, before any fold is judged. A fold standing on it is judged
as if it had been decided first: one it would have stopped measures provisionally and
is skipped behind it with every reason it would have named, in the order of its
`after`, and nothing the fold met in the read is reported; one already stopped by
another prerequisite, or whose `when` declines or raises, is recorded once it is
decided, as it would have been. On a target no fold reads, its own read is the same
page walk.

**Measured**, base `723cac0` against head `a3c7355`, each pair started at the same
moment on a shared host, `uptime` sampled every five minutes and each process's own
`VmHWM` every 30 seconds. OGS-00009's pair ran from 04:32 to 07:58 UTC, inside the
nightly backup window (load average 1.5 to 83 in its five-minute samples), so its
absolute times are inflated by that load; base and head ran side by side and saw the
same load, so their ratio stands, and #322 re-measures validation end to end:

| | base | head |
|---|---|---|
| OGS-00009 (11,191,070 variants x 2,514 Analyses), a full `validate` | 12,342 s | 4,836 s |
| its `subsets.indexed` | 8,230.2 s | 621.4 s |
| its Dense association scan | 4,069.8 s | 4,149.7 s |
| its peak RSS | 2.61 GiB | 2.58 GiB |
| OGS-00015 (11,650,615 x 241), a full `validate` | 1,243 s | 407 s |
| its `subsets.indexed` | 896.0 s | 63.1 s |
| its Dense association scan | 310.2 s | 306.9 s |
| its EAF exception table, bytes read in the pass | 191,843,168 in 1,525 reads | 375,672 in 2 reads |

Head's `subsets.indexed` is the read of the index alone; the planes are read once, by
the scan, which the digests cost 2% on OGS-00009. OGS-00015's 31,306-entry table is
one inner chunk, so base's per-band lookups read it whole for every band: 511 times
its size. Head reads each entry once, as it reads OGS-00009's 8,621 (103,452 bytes).
Both peaks are the association scan's working set, reached in its first half hour.
`validate --format json` is the same to the byte on both Stores, on the seven published
Stores #387 compared (OGS-00001 to -00006 and -00008, text and JSON), and on #387's ten
Indexed Variant Subset fixtures, clean and corrupted, Dense and Hybrid.

Measured again once the key was drawn per run, head `05d1f52` against the integration
tip `b49c1cb`, side by side from 14:09 UTC on 12 Nov (load average 4 to 43), sampled as
above, with a counter of every read of the release's EAF exception table:

| | tip | head |
|---|---|---|
| OGS-00016 (27,929,380 x 1,691), a full `validate` | 18,080 s | 8,951 s |
| its `subsets.indexed` | 9,313.6 s | 318.8 s |
| its Dense association scan | 8,650.8 s | 8,526.2 s |
| its peak RSS | 3.03 GiB | 3.01 GiB |
| its EAF exception table, bytes read in the pass | 35,615,471,496 in 32,796 reads | 2,844,330,264 in 1,810 reads |
| OGS-00011's Dense Component (14,267,257 x 3,169), head alone | | 16,250 s |
| its `subsets.indexed` | 16,904.3 s at #387's head | 874.6 s |
| its Dense association scan | 14,677.1 s at #387's head | 14,497.2 s |
| its peak RSS | 8.30 GiB at #387's head | 8.24 GiB |
| its EAF exception table, bytes read in the pass | | 13,643,892,492 in 8,676 reads |

OGS-00016's table, 237,027,522 entries, is read once in 905 pages, where the tip read it
12.5 times over; OGS-00011's 1,136,991,041 are read once in 4,338. On both, every unit's
digests agreed (0 of 1,128 and 0 of 2,114 compared cell by cell). OGS-00011's ride keeps
its HapMap3 subset's 21,157,534 exception cells, about 254 MB, through the whole scan
rather than after it, and its peak, reached as the scan ends, is #387's. Both JSON reports
are byte-identical: OGS-00016's tip and head (661 bytes), OGS-00011's head and #387's
(1,687 bytes). On OGS-00011's Overflow the pass reads its 124,961,622-entry table once
too; `by_variant.sample` still looks entries up in it for its sample, outside the pass.

A corrupted real subset is found as before: in scratch copies of OGS-00015, one Z cell
of `hm3` bumped by a code, at a slot whose primary row the scan reads in its 138th band,
and separately two Z cells of one unit exchanged, each give the tip's finding, its JSON
byte-identical; head compares one of the three units it reaches cell by cell, the one
whose digests disagreed.

## Amendment: a Dense grid's band pass is spread across forked workers (issue #315)

With the planes read once (#314), the association scan was 86% of OGS-00009's
validation, one process decoding 2,798 bands in turn. `validate --n-workers N` spreads
those bands over the shared ordered fork pool the SE and Top-Hit Index phases use
(#221), and the report is the one a single process makes.

**One band a task, added in band order.** Every block is measured by one function
wherever it is read (`_measured`), into what it adds to the pass: each fold's measure,
or its failure, or a plane it could not read, each worded where the exception was
raised; and each riding check's part. The driver adds those in band order through the
same code a single process adds them with, so a fold's total is the same sum in the
same order, the first failure reported is the first in band order, a plane reported
undecodable is reported by the fold that meets it first in band order, and a read that
stops part-way stops at the same band, with every band before it added. Workers do not
know which folds the driver has stopped since it submitted their band; what they
measure past a fold's failure is discarded.

**The driver reads the first band.** What the folds load on their first block -- a
Top-Hit Index's tiers, the most of what a full-size pass holds (#387) -- is then loaded
once, before the pool forks, and every worker inherits it rather than loading a copy
of its own.

**A riding check rides in two steps** (`Rider`): `part`, where the block is read, says
what the block adds without changing the check; `join`, in band order where the pass
is driven, adds it. `subsets.indexed`'s ride makes each band's unit sums in the worker,
under the key the driver drew once for the run (`run_key`) when it boarded the ride,
and the join refuses rows that do not follow the last it added. A ride a part breaks,
on either side, leaves the pass and reads for itself, as before.

**The EAF exception table stays read once a pass.** The driver cuts each band's span
from its one paged read of the table as it hands the band out (`GridSplit.units`), so
`encoding.eaf` and every subset's ride still watch each page go by there, in order, and
a worker is handed its band's span and nothing else. A worker reading its own span
would first have to find where the span starts -- a search of the index -- and would
read the page at each band's edge twice.

**Workers open what they read after the fork.** A worker opens the grid's group again
(`Target.reopen`), and through it the planes and the side tables its codecs look
exceptions up in -- the SE exceptions, the Z overflow, the EAF exceptions where a band
has no span -- so no read in a worker goes through a handle the driver opened. Reopening
is not what keeps a forked reader from hanging: the hazard is process-wide. With the
fused pipeline's thread pool on, a worker whose first read spans no more shards than
the driver's reads left idle permits in the inherited pool queues its chunks to threads
the fork did not copy, and never returns, whatever handle it reads through. The array
seam's `max_workers = 1` (ADR 0056) is what prevents it. Reproduced at three workers on
4,000 x 2 cells in shards of 2,000 x 1, whose bands, like OGS-00009's, read two shards
side by side within one shard row: finished in 1.3 s and 1.8 s at `max_workers` 1,
killed at 45 s, twice, at 8. One shard a band, or bands crossing a shard row, finish
either way, so the test's fixture is shaped like the real one.

**A failure of the pool fails the run.** A worker that fails to read its own band
reports it as that band's outcome, a finding like any other. Anything that escapes the
pool itself is the run's failure, not the Store's (`PoolFailure`): a worker killed
mid-band -- for memory, say -- breaks the pool (`WorkerLost`), and a host out of
processes refuses the fork. The pass names the band whose result it was waiting for --
the oldest still out, not necessarily the one a dead worker held -- and how many more
were in flight, and `validate` ends with that error. It does not report a Store finding
about a read that never finished.

**Memory.** A worker holds one band at a time: the decoded-band budget (#254), at 96
bytes a cell, and 512 MiB beside it (`WORKER_FLOOR_BYTES`). The pool is sized by
`encoding.memory.pool_workers` (#321): no more workers than a quarter of the host's
available memory holds beside the driver and the outcomes it holds in flight, no more
than the bands left, and one on a host whose memory cannot be read.

A worker's budget is held against its private memory, with its PSS beside it (ruled on
13 Nov 2026). Its `VmHWM` is not its own cost: it counts every page the worker maps,
and a forked worker maps the driver's -- the Top-Hit tiers above all, 1.8 to 2.2 GiB on
OGS-00009 and 6.5 to 6.9 GiB on OGS-00011 -- which it shares with the driver rather than
holds, and which the pool's sizing counts once, as the driver's.

The budget holds with a margin because of a fix. On OGS-00011 a worker first reached
1.81 GiB (PSS) beside a band budgeted at 0.85 GiB, and that was not the band.
`Block.locate` searched a Top-Hit tier's `uint32` rows with Python-int bounds, so numpy
copied the whole tier to `int64` on every call: 1.12 GB for OGS-00011's 140,478,063-cell
tier, several times a band, in the serial pass as well. The bounds now take the rows'
own dtype. Measured band by band under tracemalloc, every fold in turn, the Top-Hit
folds had reached 137.2 bytes a cell on OGS-00011 and 56.2 on OGS-00009; with the fix a
band peaks in its own read, at 54.7 on OGS-00011 and 52.9 on OGS-00009, and between
bands a worker holds 0.25 to 0.27 GiB of its own.

**Still one process:** an association sequence's scan -- a Ragged release's, a Hybrid's
Overflow -- whose windows each carry the cell before them; the Store Variant Table's and
`by_variant`'s scans; and every check of `analyses.tsv`, the envelope and the side
tables. `--n-workers 1` reads every band in process.

**Measured** at the head, OGS-00009 at one worker and at 32 started at the same moment,
side by side, from 00:35 to 01:40 UTC on 13 Nov 2026 (load average 4.8 to 17.9,
available memory never below 913 GB; from 01:06 the runs were inside the hours of the
backup window recorded on 22 Oct, whether or not it still runs). Each round read every
process's `/proc/<pid>/status` and `smaps_rollup`: each driver about every 1.1 s, and
each worker every 1.96 s on average. OGS-00011 ran at 32 workers the night before with
the same code path (`c4402e7`; the head adds only the pool-failure handling), each
worker sampled every 2.79 s on average:

| | OGS-00009, 1 worker | OGS-00009, 32 workers | OGS-00011, 32 workers |
|---|---|---|---|
| a full `validate` | 3,857 s | 942 s | 1,955 s |
| its Dense association scan | 2,904.6 s, 2,798 bands | 140.9 s | 274.9 s, 4,756 bands |
| its `subsets.indexed`, read in the driver | 905.3 s | 757.8 s | 906.3 s |
| the driver's peak `VmHWM` | 2.55 GiB | 2.24 GiB | 7.32 GiB |
| a worker's budget | | 1,432 MiB | 1,382 MiB |
| a worker's largest private memory, the budget's measure | | 0.66 GiB | 0.77 GiB |
| a worker's largest PSS | | 0.71 GiB | 0.95 GiB |
| a worker's largest `VmHWM`, the driver's pages included | | 2.29 GiB | 7.03 GiB |

OGS-00009's two reports are byte-identical to each other and to #314's head's, and
OGS-00011's to #314's; 0 of their 1,676 and 2,114 subset units were compared cell by
cell, and the driver read OGS-00011's 1,136,991,041-entry EAF exception table once, in
4,338 pages. #314's runs of OGS-00011 at one worker took 16,250 s, its Dense scan
14,497.2 s. At 32 workers `subsets.indexed` is 757.8 s of OGS-00009's 942 s.

Before the `locate` fix (`fef3914`, 12 Nov from 20:06 UTC, sampled every 30 s),
OGS-00016 ran at 32 workers in 745 s, its Dense scan 351.0 s against #314's 8,526.2 s at
one, its 237,027,522-entry table read once in 905 pages, and its report byte-identical
to #314's. #314's two corrupted copies of OGS-00015 -- one Z cell of `hm3` bumped where
the scan reads its 138th band, and two cells of one unit exchanged -- each gave #314's
finding at 32 workers, their reports byte-identical. Those runs also found OGS-00011's
workers over budget, which led to the `locate` fix above.

## Alternatives considered

- **`check(release)` over a caching release** (the ticket's sketch). Simple
  rules, but whole decoded planes cached, about 25 GB resident on OGS-00011's
  1.75 billion-cell Overflow, or a summary struct every Dense rule edits.
  Rejected for memory.
- **Rule identity on today's seams.** A `report.error(rule, message)` in place
  of the error list, and the orchestrators kept. Cheap, but a rule stays welded
  to its traversal, a selection only filters findings after all the work, and
  the repeated reads stay. Rejected: it names the rules and fixes nothing else.
