# Association Admission: one rule set, one gate, one record

Issue [#357](https://github.com/opengwas/opengwasdb/issues/357), part of epic
#58. Records which of an Analysis's source rows a Store admits, where that is
decided, and where what it dropped is written down. Amends ADR 0054, whose one
row-admission rule this generalises, and ADR 0053's checkpoint record.

## Context

Every builder decided for itself which source rows to store, and each decided
differently. The inventory (`reports/357-proposal.md` §1) found 19 rules, nine
of which behaved differently by layout:

- an unusable SE was substituted by one reader, dropped silently by three,
  stored as NaN or `inf` by the VCF reader and raised on by `build-dense`;
- a duplicate (variant, Analysis) kept the last row in Dense and Hybrid builds,
  collapsed agreeing rows in Ragged SSF, and raised in `build-dense`;
- a declared INFO or MAF threshold filtered only the Hybrid build;
- nothing counted the rows a reader skipped, or the ones routing could not place.

None of these raised an error. OGS-00027 lost 7,444 associations with an unusable
SE, and GCST90295971 (OGS-00011) stored the last of seven conflicting duplicates,
one of them with the opposite sign, with no record of either.

## Decision

**Association Admission is one module, `opengwasdb.admission`.**

- **The rule set** (`rules.py`) judges a batch of an Analysis's rows and gives
  each dropped row exactly one `Reason`. First the rules that decide whether a row
  is an association at all, then the filters the Analysis declared. The declared
  filters therefore judge and count only associations, which is the population a
  Store's `info_score` and `maf` provenance has always described.
- **A `Gate`** is opened once per Analysis. It returns each batch's keep mask, adds
  the batch to the Analysis's `AdmissionCounts`, and keeps the source key and reason
  of every row dropped for a logged reason. Builders hand it rows before they are
  matched to the Variant Index: through Variant Routing's `admit=` (Dense VCF,
  Hybrid), or directly (Ragged SSF, Ragged BESD, `build-dense`). The resolver uses
  `counting_gate`, which only counts.
- **An `AdmissionRecord`** is the build-level record, written beside the Store
  Release and never inside it, so a Store's bytes and `format_version` do not
  depend on it. A gate writes its Analysis's part (`parts/<index>.json`, and
  `dropped/<index>.tsv.gz` when it logged anything) temp-then-rename, from
  whichever process judged the Analysis. `finish()` refuses a record missing a part
  and writes `admission.json`, in Manifest Row order. The record is therefore the
  same at any worker count, and a resumed Hybrid build rereads its counts from the
  parts. Checkpoint format 3 records the record's path instead of
  `info_counts.json`, and refuses a format-2 checkpoint.
- **What is logged (D8).** A row dropped by a declared INFO or MAF filter is
  counted only: it is reproducible from the source and the policy, and a per-row
  log would cost more than the Store (an estimated 51–54 GB for OGS-00011). So is
  a row collapsed into an agreeing duplicate, whose association the Store holds.
  Every other reason's rows are also logged, by source key, because nothing but
  the source can say which rows those were.
- **Completion** admits an imputed candidate by the same cell rule (`admissible`).
  It refuses a source cell with one of z and SE missing, rather than pairing an
  imputed value with an observed one (D1). What it leaves unimputed is already
  counted in each Store's `completion_quality`, so a completion writes no record.
- **`require_admitted`** checks Admission's guarantee on cells a writer is handed.
  Every builder calls it on the z and SE it is about to store, so a half-missing
  pair (spec §15) or any other cell a gate should have dropped refuses the build,
  naming where (D1). No source row can reach it through a working gate. It checks:
  - a present cell has a finite z and a finite, positive SE;
  - z is missing exactly where SE is (spec §15, ADR 0070's `states()`);
  - a frequency is NaN or inside (0, 1);
  - an imputed flag is 0 or 1, and 1 only on a present cell.

### The rules, in precedence order

| Reason | A row is dropped when | Logged | Since |
|---|---|---|---|
| `se_unusable` | its reported SE is missing, not positive or not finite | yes | slice 4 (Ragged), D3 (every reader) |
| `effect_missing` | it has a usable SE but no effect | yes | slice 5 |
| `z_nonfinite` | its z is infinite, beside a usable SE | yes | D4 |
| `info_below_threshold` | a usable score is below a `FILTERED` INFO threshold | no | ADR 0054; every builder since D2 |
| `maf_missing` | a `FILTERED` MAF policy finds no usable frequency | no | ADR 0054; every builder since D2 |
| `maf_below_threshold` | MAF is below a `FILTERED` threshold | no | ADR 0054; every builder since D2 |
| `duplicate_collapsed` | it agrees on z and SE with an earlier row for its (variant, Analysis), whose association it is | no | D5 |
| `duplicate_conflicting` | the rows for its (variant, Analysis) disagree on z or SE: all are dropped | yes | D5 |
| `off_axis` | no axis of the Store holds its variant, and nothing will place it later | yes | D6 |

The duplicate rule is Variant Routing's `collapse_duplicates`, applied after
the rules above. Each builder uses it: the Dense VCF and Hybrid builds through
routing (an axis's rows and the Hybrid's off-reference keys), and Ragged SSF and
`build-dense` on their own rows. It replaces routing's last-wins, Ragged SSF's
own rule and `build-dense`'s refusal. The fold's merge of an off-reference entry
with an existing Overflow one still keeps the later (issue #223).

`off_axis` is a Dense VCF row its axis lacks (a lift failure, a variant a
Variant Reference omits), a two-pass Hybrid row no axis holds (it used to be
spilled and deleted unread), and a BESD row whose ESI entry never made the
axis. A single-pass Hybrid's unmatched rows are not dropped here: the fold
places them in the Ragged Overflow, and the rows the fold cannot place are its
own to count.

A frequency is never a reason to drop a row (D7). One a source reported
outside (0, 1) -- exactly 0 or 1, beyond the interval, not finite, or not a
number -- is absent in the Store, by the rule `parse_af` applies in every reader
(ADR 0036). `build-dense`, which stored a record's frequency as given, applies
it too. The record counts them per Analysis (`eaf_made_absent`), over the rows
the Source Reader read, so it can say how much of a source's frequency column
the Store does not hold. A `.esi` frequency is a variant's, which every probe
shares, so `build-ragged-besd` logs the file's count once instead.

`admission.json`'s `"rules"` names the rule set's version, and each decision
below that changes what a rule admits moves it.

## Consequences

- One place answers "why is this association not in the Store": the record names
  each drop's reason, and its log names the row.
- A Source Reader yields a row with no effect, with a NaN z, where it used to skip
  it. Admission drops it, so every consumer of `stream_associations` must too: the
  phenotype-SD pipeline does.
- A record sits beside every build's Store. Every builder command requires
  `--admission-record DIR`, and exits before writing anything without it. A
  default beside `OUTPUT_PATH` would follow a staging name: the
  `opengwasdb-stores` workflow builds `store.opengwasdb.partial` and renames only
  that, which would leave `store.opengwasdb.partial.admission` beside no Store.
  Refusing a missing flag fails louder than deriving a final name the package
  cannot know. The Python API keeps its default, `OUTPUT_PATH.admission`.
- Routing collects an imputation score only for an Analysis that declared one
  (`route_column(imputation_scores=False)`). Collecting it for every row cost the
  Dense VCF Pass 2 8–12%.
