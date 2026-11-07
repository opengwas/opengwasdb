# Association Admission: one rule set, one gate, one record

> **Amended by [ADR 0079](0079-indel-alids-spell-the-reference-allele-first.md)**
> (issue #381). An indel's ALID spells its reference allele first, so an
> insertion and a deletion at one position are two variants, and the
> reversed-indel rule below can no longer fire: no build drops a row as
> `duplicate_reversed_indel`, which stays readable in a record written before.
> An indel whose source declares no reference is a third malformed identity,
> dropped and logged as `indel_reference_undeclared` after `allele_invalid`.
> The rule set's version is `admission/12`.

Issue [#357](https://github.com/opengwas/opengwasdb/issues/357), part of epic
#58. Records which of an Analysis's source rows a Store admits, where that is
decided, and where what it dropped is written down. Amends ADR 0054, whose one
row-admission rule this generalises, and ADR 0053's checkpoint record.

## Context

Every builder decided for itself which source rows to store, and each decided
differently. Issue #357's inventory found 19 rules, nine of which behaved
differently by layout:

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

**Association Admission is one module, `opengwasdb.admission`.** D1 to D8 below
name the decisions issue #357 took on the rules that differed.

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
  It refuses a source cell with one of z and SE missing (`require_paired`, D1),
  rather than pairing an imputed value with an observed one or carrying the
  cell as it stands. Every completion checks every cell it carries: the Dense
  bands, a Ragged Analysis's observed rows (including one no LD block
  touches), and the Hybrid's Ragged Overflow. What it leaves unimputed is
  already counted in each Store's `completion_quality`, so a completion writes
  no record.
  - *Clarified 5 Nov 2026.* A completion that reads its source through the
    Association Reader (ADR 0070) refuses such a cell by the reader's presence
    rule instead, and a missing marker other than the canonical NaN with it:
    `InvalidAssociations` names the source, where `require_paired` raised
    `NotAdmitted`. Both are `ValueError`s, but code that catches `NotAdmitted`
    no longer catches the refusal. The completions that read this way:
    - Hybrid, its source's Ragged Overflow (#391).
    - Ragged, each source Analysis (#393).
    - Dense, its source's bands and each LD block's rows (#393). With that,
      no completion calls `require_paired`, and it is removed.
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
| `chromosome_missing` | its chromosome is an empty cell or a missing-value token (`.`, `NA`, `NaN`, `None`) | yes | #365 |
| `allele_invalid` | an allele, upper-cased (ADR 0075), is empty or holds a character other than A, C, G or T | yes | #365 |
| `indel_reference_undeclared` | its two alleles differ in length and its source declares neither the reference, so no ALID can spell it reference first (ADR 0079) | yes | #381 |
| `se_unusable` | its reported SE is missing, not positive or not finite | yes | slice 4 (Ragged), D3 (every reader) |
| `effect_missing` | it has a usable SE but no effect | yes | slice 5 |
| `z_nonfinite` | its z is infinite, beside a usable SE | yes | D4 |
| `info_below_threshold` | a usable score is below a `FILTERED` INFO threshold | no | ADR 0054; every builder since D2 |
| `maf_missing` | a `FILTERED` MAF policy finds no usable frequency | no | ADR 0054; every builder since D2 |
| `maf_below_threshold` | MAF is below a `FILTERED` threshold | no | ADR 0054; every builder since D2 |
| `duplicate_collapsed` | it agrees on z and SE with an earlier row for its (variant, Analysis), whose association it is | no | D5 |
| `duplicate_reversed_indel` | it is an indel whose declared reference allele its ALID spells second, at a (variant, Analysis) whose distinct rows, once agreeing copies collapse, are this one and one spelling it first, which is kept; no longer fires since ADR 0079 | yes | D5, reversed indels |
| `duplicate_conflicting` | the rows for its (variant, Analysis) disagree on z or SE: all are dropped | yes | D5 |
| `off_axis` | no axis of the Store holds its variant, and nothing will place it later | yes | D6 |
| `lift_unmapped` | its source is lifted onto the Store's assembly and the chain places its site nowhere | yes | #386 |
| `lift_ambiguous` | its Routing Key is declared by a lifted source and by a source already on the Store's assembly: two loci sharing a string, so no row of it is placed | yes | #386 |

`build-ragged-ssf` lifts a source declaring another build, and counts what the
lift drops under its own two reasons rather than `off_axis`: a record then says
the chain, not an axis, lost the row. Records written before #386 hold neither
reason and read as before. The rule set's version moved to `admission/11`.

The duplicate rule is Variant Routing's `collapse_duplicates`, applied after
the rules above, wherever two rows can reach one (variant, Analysis):
- the Dense VCF and Hybrid builds, through routing: an axis's rows, and the
  Hybrid's off-reference keys;
- the Hybrid's fold, where two raw keys resolve to one ALID, or an
  off-reference entry reaches an index the column already holds;
- Ragged SSF and `build-dense`, on their own rows, and Ragged BESD, on a probe
  that names one SNP twice;
- Reference Completion, on a source Store's repeated cell and on a variant
  two LD blocks fill.

It replaces routing's last-wins, the fold's later-wins (issue #223), Ragged
SSF's own rule, BESD's keeping both rows, completion's last-written map entry
and `build-dense`'s refusal. The fold runs after Pass 2's gates have written
their parts, so it writes its own: `parts/<index>.fold.json` and
`dropped/<index>.fold.tsv.gz`, added to the Analysis when the record is
finished. A fold run again for a column replaces them. A completion writes no
record, so its log names every row it drops, a source Store's duplicate cells
and the fills two blocks disagree on, one line each with the row's variant, z,
SE and reason (D8).

**Clarified 5 Nov 2026 (epic #58's final review): a variant two LD blocks fill
has an owner.**

- **The rule.** Blocks are half-open, `[start, end)`. A variant at a position
  one block ends at and the next starts at belongs to the block that starts
  there, and only that block fills it. It stays in both blocks' LD context
  (`completion.ld_panel.shared_endpoints`). A block that no other block follows
  keeps its end: a chromosome's last block, or, in a Ragged completion, a
  block whose successor does not complete that Analysis. Each such cell is
  filled once, and counted once in `completion_quality`. Where the owner's fit
  fails for an Analysis, the cell stays missing, even where the other block's
  fit passes. Of the published Stores, the Dense OGS-00015 holds 254 cells
  only the other block filled, and OGS-00017 10; the rule leaves each missing.
  The Ragged OGS-00002 holds 8, but leaves 1 missing: for the other 7 the
  owning block does not complete that Analysis, so the ending block keeps its
  end. Completed again from OGS-00001, it holds 106,652 imputed cells against
  106,653.
- **The guard.** The duplicate rule stays in the writer of every completion.
  A fill that still reaches a cell another block filled is resolved by it, and
  each dropped fill is logged. That happens where two blocks overlap away from
  an endpoint, and no panel completion reads has such an overlap. The dropped
  fill stays counted in its block's `completion_quality` row.
- **The cause.** `opengwasdb-stores`' panel builder (`construct_block.py`)
  cuts each block with plink2's inclusive `--from-bp`/`--to-bp`. Adjacent rows
  of its region table share an endpoint, so every boundary variant is in both
  blocks. Measured from the block files, the variants two blocks hold, each
  at an endpoint the two share, are:

  | Panel | Variants |
  |---|---|
  | `reference/ld` EUR | 1,022 |
  | `hgdp1kgp-hg38` EUR | 985 |
  | `hgdp1kgp-hg38` AFR | 1,330 |
  | `hgdp1kgp-hg38` EAS | 866 |
  | `hgdp1kgp-hg38-amr` AMR | 1,198 |

  No block holds a variant twice.
- **What was published.** The completed Store Releases published by 5 Nov
  2026 hold 1,500,055 cells that both blocks filled, 798,831 of them in
  OGS-00009, and each stores whichever fill completion wrote last. In 16
  recomputed cells the two fills differed in both z and SE, and 7 differed in
  sign. Issue #396 repairs the Stores.

Two conflicting rows can be two variants. An insertion and a deletion at one
position share an ALID, because the ALID sorts its alleles: BMI_IRN's
`1:827587` `G>GGGGCC` (alt frequency 0.83%) and `GGGGCC>G` (0.05%) are both
`1:827587:G:GGGGCC`. They are two alleles of a repeat, not one variant read
from opposite alleles, and every conflicting pair in OGS-00003 is of this kind.
When the source declares which allele is the reference, the row whose
reference allele the ALID spells first is kept: for an anchored indel, the
insertion, which is the variant the ALID reads as. The other row is dropped and
logged (`duplicate_reversed_indel`). The rule judges distinct rows. Agreeing
copies collapse first, as any agreeing duplicate does, and the exception holds
only when two distinct rows remain, one spelling the reference first and one
second. Two distinct insertions, or two distinct deletions, leave no row to
keep, and every row at the place is `duplicate_conflicting`. The rule judges
ref and alt, not effect and other: FinnGen's effect allele is its alt, so its
insertion is the row whose z is negated onto the ALID. A source that declares no reference has
its pair dropped as conflicting, and so is every other conflicting duplicate,
SNVs included, since two SNV rows at one ALID are one variant seen from
opposite alleles. Dropping both rows was rejected because it loses a cell
base kept; this rule keeps it wherever the source declares the reference.

Which formats declare the reference allele (`ReferenceAllele`, carried on each
`ReaderAssociation` and through routing as one byte a row):
- GWAS-VCF's `REF`, FinnGen r13's `ref`, and a Column Map with
  `ref_alt_effect_is_alt`, whose rows are the FinnGen reader's, do.
- GWAS-SSF names an effect and an other allele, and its documentation does not
  make the other allele the reference, harmonised or not. Nor do a
  `build-dense` record's alleles, a BESD `.esi` row's A1 and A2, or an
  `effect_and_other` Column Map. None declares one.
- A `ref_alt_and_effect_allele` Column Map (dbGaP MVP) is treated as declaring
  none: nothing this package holds documents that the deposit's `ref` is the
  genome's reference allele.

This is an interim rule. The ALID still spells a reversed pair as one variant:
#381 gives an indel a ref-first ALID, which is an ADR and a format change, and
#382 corrects the published cells that hold a deletion's statistics under the
insertion's ALID.

`off_axis` is a Dense VCF row its axis lacks (a lift failure, a variant a
Variant Reference omits), a two-pass Hybrid row no axis holds (it used to be
spilled and deleted unread), and a BESD row whose ESI entry never made the
axis. A single-pass Hybrid's unmatched rows are not dropped in Pass 2: the
fold places them in the Ragged Overflow. A row whose off-reference key the
fold cannot resolve (a failed lift, a key the resolution drops) is `off_axis`
too, counted and logged by raw key in the record's `fold` stage. A two-pass
build folds nothing, so it refuses a `.unk` spill it finds rather than delete
rows no stage has counted.

The two identity rules come first, because a row they drop names no variant:
no ALID can spell it, so no other rule has anything to judge. The Source Reader
finds them as it forms the row's identity (`variants.normalise`: a
`MalformedIdentity` raised by `normalise_chromosome` or `normalise_allele`),
leaves the row out of both of its streams, so no axis holds its variant, and
names it in the stream's `StreamTally`; the gate then counts and logs it
(#365). Every reader applies them the same way: the GWAS-VCF build streams,
which validated no allele and kept `NA` as a contig, the tabular readers, which
dropped such rows unseen, `build-dense`, which refused the source, and BESD,
whose `.esi` rows they judge once for every probe. An alt, random or unplaced
contig is a chromosome (`11_KI270927v1_alt`). A row the readers drop for
another identity reason (an unparseable position, equal alleles, an
unsupported chromosome number) is still the reader's own.

A frequency is never a reason to drop a row (D7). One a source reported
outside (0, 1) -- exactly 0 or 1, beyond the interval, not finite, or not a
number -- is absent in the Store, by the rule `parse_af` applies in every reader
(ADR 0036). `build-dense`, which stored a record's frequency as given, applies
it too. The record counts them per Analysis (`eaf_made_absent`), over the rows
the Source Reader read, so it can say how much of a source's frequency column
the Store does not hold. A `.esi` frequency is a variant's, which every probe
shares, so a `build-ragged-besd` probe counts the rows it reads at a variant
whose reported frequency is absent, when the build stores the frequency. An
infinite `.esi` frequency is one of them; the `.esi` reader used to refuse
the file for it.

`admission.json`'s `"rules"` names the rule set's version, `admission/11`. A
change to what a rule admits, or to what the record counts, moves it.

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
