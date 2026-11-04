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
  whatever it found. It fails no other rule.
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

**`validate --format json` does not change (D1).** Its keys stay `ok`,
`errors` and `warnings`, and its messages stay byte-identical: opengwasdb-stores'
`register.validate_verdict` refuses any other key. Emitting rule names there is a
two-repository follow-up: `register` first accepts an optional `findings` key,
then this package emits it.

**Admission's witnesses.** Six rules check, after the fact, what Association
Admission (ADR 0072) guarantees of every cell it admits. Each records the
Admission `Reason`s whose drops keep a cell it would fail out of a Store
(`Rule.witnesses`):
- `associations.markers_paired`: `se_unusable` and `effect_missing`;
- `associations.markers_canonical` and `completion.imputed_present`:
  `z_nonfinite` and `se_unusable`;
- `associations.se_negative`: `se_unusable`;
- `csr.segment_order`, its repeat case: `duplicate_collapsed` and
  `duplicate_conflicting`;
- `completion.mask_binary`: none. It witnesses the cell guard's flag rule, which
  no row's `Reason` names.

A rule naming a reason Admission lacks is refused at import, and
`tests/test_admission_witnesses.py` holds the two lists together. Until #357
merges, the reasons are a copy of its `Reason` values, and the test compares the
copy with the enum once it exists. The rules and Admission deliberately do not
share predicates where Admission is stricter than published Stores. Admission
requires SE > 0 and EAF in (0, 1); `validate` accepts SE ≥ 0, warns on `+inf`,
and accepts EAF in [0, 1].

## Consequences

- **Rules run after unrelated failures (D4).** Only failing Stores change.
  Every published Store's `validate --format json` is byte-identical across the
  change, except #380's three 0.1.0 releases (below).
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
- **Time and memory**, measured against the release before, with RSS sampled:
  - OGS-00006, a Ragged release, back to back: 74–82 s against 76–81 s, and
    peak RSS from 3.9–4.0 GiB to 1.6–1.8 GiB. Its new variant-table rules cost
    about 40 s. The single decode, and `by_variant.sample` reading each column
    once rather than a cell at a time, saved as much.
  - OGS-00008, a Dense release, back to back: 59 s against 89–91 s, and peak
    RSS from 0.79 GiB to 0.56–0.57 GiB.
  - OGS-00011, a Hybrid release, side by side and without the Indexed Variant
    Subset check, which did not change: 13,962 s against 14,192 s, and peak
    RSS from 110 GiB to 48 GiB. That check is 30,209 s of an instrumented
    45,132 s run. A full run still peaks at 113 GiB, in that check.
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

## Alternatives considered

- **`check(release)` over a caching release** (the ticket's sketch). Simple
  rules, but whole decoded planes cached, about 25 GB resident on OGS-00011's
  1.75 billion-cell Overflow, or a summary struct every Dense rule edits.
  Rejected for memory.
- **Rule identity on today's seams.** A `report.error(rule, message)` in place
  of the error list, and the orchestrators kept. Cheap, but a rule stays welded
  to its traversal, a selection only filters findings after all the work, and
  the repeated reads stay. Rejected: it names the rules and fixes nothing else.
