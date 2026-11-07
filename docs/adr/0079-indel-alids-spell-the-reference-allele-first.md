# An indel's ALID spells its reference allele first

Issue [#381](https://github.com/opengwas/opengwasdb/issues/381), under epic #367,
built in epic #307's run because #316's integer key join keys on the ALID it
defines. The owner decided the rule and where the reference comes from on 4 Nov
2026, and the two points left open then, and the format version, on 7 Nov.
Format `0.2.1`. Amends ADR 0072 (its reversed-indel rule and its identity
rules) and ADR 0036 §2 (its statement of the ALID convention).

## Context

An ALID, `chr:pos:A1:A2`, sorted its two alleles. For an SNV that is right: the
genome holds one base there, so `A>G` and `G>A` are one variant seen from
either allele. For an indel it is wrong. A deletion written sorted is spelt
exactly like the insertion of the same sequence, and at a tandem repeat both
are real. FinnGen R13 BMI_IRN at `1:827587`:

| ref → alt | rsid | alt frequency | beta | se |
|---|---|---|---|---|
| `G` → `GGGGCC` | – | 0.0083 | −0.0012 | 0.0136 |
| `GGGGCC` → `G` | rs755160762 | 0.0005 | +0.0226 | 0.0506 |

Both were `1:827587:G:GGGGCC`, and the deletion read as the insertion flipped.
The same file holds 872,658 insertions and 1,071,490 deletions. 96,412 of the
deletions share an ALID with an insertion in the same Analysis. The other
975,078 are stored under an ALID spelt like an insertion, and another Analysis
reporting the insertion at that site joins them as one variant. None of this
raised an error.

#357's interim rule (ADR 0072, D5) kept the insertion of such a pair, where the
source declares its reference, and dropped the deletion as
`duplicate_reversed_indel`. Lone deletions kept the insertion's spelling.

A reference genome does not settle it. Where the genome reads `GGGGCC`, both
readings fit it. Left alignment normalises a ref/alt pair the source already
gave; it cannot say which allele is the reference. Only the source's own
declaration tells the two variants apart.

## Decision

**1. The ALID.** Where the two alleles are one length — an SNV, or a
substitution of several bases — `A1` is the alphabetically first and `A2` the
other, as before. Where their lengths differ, `A1` is the **reference allele**
and `A2` the alternate (spec §4). BMI_IRN's pair is `1:827587:G:GGGGCC` and
`1:827587:GGGGCC:G`. An anchored indel's shorter allele sorts first, so an
insertion's ALID is unchanged and a deletion's is reversed. A statistic is
signed for `A1`, as every statistic is, so a deletion's z is now signed for its
reference allele: FinnGen's deletion, whose beta is for `G`, is stored negated.

The rule is `opengwasdb.variants.normalise.alid_alleles`. Every place an ALID is
formed goes through it: `canonical_alid`, `orient_to_canonical` and its
hot-path form `oriented_alid`, which a contract test holds to one answer. That
covers each Source Reader's rows and projections, routing's sites and keys, the
lift (`statistics_swap`), Variant Reference extraction and reading, the Hybrid
fold's key table, the Indexed Variant Subset list reader, the phenotype-SD
pipeline and source fidelity.

**2. Where the reference comes from.** Only from the source's declaration
(`orient_to_canonical(..., other_is_reference=)`, required wherever an ALID is
formed from a row):

- declared: GWAS-VCF's `REF`, FinnGen r13's `ref`, and a
  `ref_alt_effect_is_alt` Column Map's `ref`;
- not declared: GWAS-SSF, harmonised or not (see 7); a BESD `.esi`'s A1 and A2;
  a `build-dense` association TSV; an `effect_and_other` Column Map; and a
  `ref_alt_and_effect_allele` (dbGaP MVP) Column Map, whose `ref` nothing in
  this package documents as the genome's.

An indel whose source declares no reference names no ALID. It is a malformed
identity, like a missing chromosome (#365): the reader leaves it out of both of
its streams and names it in the stream's tally (`UndeclaredReference`), and
Association Admission counts and logs it per Analysis as
`indel_reference_undeclared`, after `allele_invalid` in precedence (ADR 0072).
It is never stored under a guessed order. Its SNVs and substitutions are stored
as before. The rule set's version moves to `admission/12`.

**3. After a lift**, "reference" is the release assembly's (GRCh38): the
source's declared reference projected onto it, complemented on a
reverse-strand chain block. A lift therefore never re-orients an indel.

**4. Equal-length multi-base substitutions stay sorted**, like SNVs: the
genome holds one sequence there, so their two orders are one variant.

**5. `format_version` steps to `0.2.1`**, the compatible remainder (spec
§21.1). The arrays are 0.2.0's; only an indel's ALID and the sign of its
statistic change.

- **A 0.2.0 Store is read as before.** Its indel ALIDs keep their sorted
  spelling, and nothing rewrites them. A source-fidelity check spells a
  source's indels sorted against such a Store, and signs them with it
  (`validate --source`).
- **The converter writes 0.2.0**, not the current version: a 0.1.0 Store
  sorts its indels and says nowhere which allele is the reference, so its copy
  cannot honestly be stamped 0.2.1 (`TARGET_FORMAT_VERSION`).
- **Completion refuses a 0.2.0 or 0.1.0 source.** It writes into its source's
  arrays and keeps its format (ADR 0038 §4), and completing such a source would
  add panel ALIDs spelt by this rule beside the source's own deletions spelt
  like insertions. The source is rebuilt, then completed.
- **A new Validation Rule, `variant_table.indel_order`**, judges a Store from
  0.2.1 on: each indel row's ALID spells its alleles as the row records them,
  its `effect_allele` then `other_allele`, and a lifted variant's
  `source_alid` spells the same allele first, a lift never changing an
  allele's length. A 0.2.0 or 0.1.0 Store is not judged by it. What it cannot
  see is a same-assembly Store whose builder spelt a deletion sorted
  throughout, table columns included: nothing in such a Store records the
  reference apart from the ALID itself.
- **#338's reference branch also used 0.2.1.** If it lands after this, it
  takes the next step, `0.2.2`.

**6. LD Reference Panels place no indel.** A panel names each variant by an id
with its alleles sorted, and every panel on the IEU host does so for its
indels: 648,392 in UKB EUR and 2.1–3.2 M in each HGDP+1kGP panel, none unsorted
(13,124,104 in all). UKB's own `.afreq` marks all 9,848,829 rows
`PROVISIONAL_REF?=Y`. Against GRCh38 evidence (the HGDP+1kGP pvars, gnomAD v3.1.2's
REF/ALT), UKB's id-first allele is the reference for 30.8% of 586,364 indels the
evidence reads one way and its `OA` for 69.2%, so neither column declares it.
`canonical_panel_alid` names no indel, so completion neither matches nor fills
one: a fill there counts as missing, and an observed indel stays observed. The
HGDP+1kGP AMR panel does carry a `REF` column (it matches the genome REF at all
1,989,545 sites read one way), but a panel deletion's `A1` would then not be
the allele its LD is coded on (`EA`, the sorted first), and completion has no
step that re-signs a statistic between the two. Placing panel indels waits on
that and on a ruling (issue #381's report).

**7. GWAS-SSF declares no reference until the owner rules on the harmoniser
code.** The 4 Nov decision read code 11 as declaring `hm_other_allele`, from 26
files of the layout that keeps harmonised alleles in `hm_*` columns. The
implementation checked the layout that holds them in the main columns beside
`hm_code`, which 4,595 of the host's 6,078 harmonised files use. At FinnGen R13
indel sites read one way, over 197 such files:

| `hm_code` | rows | `other_allele` = FinnGen `ref` |
|---|---:|---:|
| 10 | 37,894,632 | 99.38% |
| 11 | 19,267,774 | 98.32% |

There code 10 is not "no reference" and code 11 not the more reliable one, and
neither is reliable file by file: code 11 is below 95% in 51 of 131 files (for
example GCST90837174's code-11 deletions, 412 of 76,810). The project's 58
`swap_alleles` rescued copies keep `hm_code` while exchanging the allele
columns, so a code-keyed rule reads the wrong allele on every indel row there.
The per-row declaration is in place (`DeclaredReference`, a column and the
codes that declare); which rows of GWAS-SSF it names is the ruling.

**8. #357's reversed-indel rule no longer fires.** An insertion and a deletion
at one position are two variants, and an indel whose source declares no
reference never reaches a place, so no build drops a row as
`duplicate_reversed_indel`. The reason stays readable in records written
before. Routing still carries each row's declaration (`REFERENCE_*`), now
`REFERENCE_FIRST` for every declared indel.

## Considered

- **Sorted ALIDs with a flag recording the orientation.** Keeps 0.2.0's
  spellings, but every reader and joiner would have to read the flag to tell a
  deletion from an insertion, and one that did not would join them in silence:
  the defect this ADR removes.
- **Read the reference from a genome FASTA.** None is on the host, and at a
  repeat both readings fit the genome.
- **Read the panel's reference from its SNP id, or from `OA`.** Each is wrong
  for 30.8% or 69.2% of UKB's indels the evidence reads one way.
- **Keep an undeclared indel under its sorted spelling.** That is the
  ambiguity itself, now in a format that claims to have removed it.
- **GWAS-SSF code 11 as decided on 4 Nov**, or codes 10 and 11 together, for
  the main-column layout. Both store whole files' indels against the wrong
  reference (7); left for the owner's ruling.

## Consequences

- Until the published Releases are repaired or rebuilt (#367, #382), a
  published Store and a new Store spell a deletion differently, and a lone
  deletion's z has the opposite sign in the two. A query by `1:827587:G:GGGGCC`
  finds the insertion in a new Store and, in a published one, whichever row the
  build kept.
- Rebuilt from its sources, the FinnGen R13 pilot OGS-00003 holds 96,428 more
  variants: each a deletion that now has its own row beside its insertion.
  Its other 975,140 deletions are respelt, their z negated and EAF
  complemented. Every other cell keeps its z and EAF, but 198,820,410 of
  them decode an SE up to one `int8_residual` step (0.39%) away: each
  Analysis's `se_coefficients` are fitted over the cells it stores, and the
  new deletion rows move them by up to 1.9e-5.
- Stores built from GWAS-SSF, BESD or `build-dense` sources hold no indels.
  In the 197 GWAS Catalog files checked that is 96,420,588 indel rows, each
  logged in its build's Admission record.
- A completion imputes no indel and matches none to the panel, so a completed
  Store's indels are the observed ones.
- A package that knows only 0.2.0 reads a 0.2.1 Store correctly and warns that
  it is newer; a completion it ran would spell its panel's indels sorted beside
  the Store's reference-first deletions. Only a package that knows 0.2.1
  completes one.
- A Variant Reference or variant list written before 0.2.1 names an indel by
  its sorted spelling, which this build reads as the insertion: a source's
  deletion at that site is off the reference.
- Breaking for Python callers: `orient_to_canonical`, `oriented_alid`,
  `orient_or_drop`, `identity_fault`, `project_source_variant` and
  `oriented_variant` take a required `other_is_reference`, and the tabular
  projection columns a required `reference`.
