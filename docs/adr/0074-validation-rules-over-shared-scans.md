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

**A pass still streams the table several times**, which is not the same as reading
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
measured on the same Store. Base's OGS-00016 peak is the scan's last minute, where
the side-table judge read the whole table beside every marked position. Round 1's
peaks are reads of the table: 3.92 GiB is the plan rule's read of the 2.65 GiB
table on OGS-00016 and 18.15 GiB its read of the 12.7 GiB one on OGS-00011. After
round 2 pages that read, nothing head does reads the table whole, and its 8.30 GiB
is the Dense association scan's own working set, reached as that scan finished at
14:45:28 rather than by any read of the table.

The paragraphs in "Time and memory" above that call the subset check's table read
the peak of a full run, and that call the scan's held folds the reason its later
stage starts higher, describe the code before this amendment.

## Alternatives considered

- **`check(release)` over a caching release** (the ticket's sketch). Simple
  rules, but whole decoded planes cached, about 25 GB resident on OGS-00011's
  1.75 billion-cell Overflow, or a summary struct every Dense rule edits.
  Rejected for memory.
- **Rule identity on today's seams.** A `report.error(rule, message)` in place
  of the error list, and the orchestrators kept. Cheap, but a rule stays welded
  to its traversal, a selection only filters findings after all the work, and
  the repeated reads stay. Rejected: it names the rules and fixes nothing else.
