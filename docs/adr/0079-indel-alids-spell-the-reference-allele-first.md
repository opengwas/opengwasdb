# An indel's ALID spells its reference allele first

Issue [#381](https://github.com/opengwas/opengwasdb/issues/381), under epic #367,
built in epic #307's run because #316's integer key join keys on the ALID it
defines. The owner decided the rule and where the reference comes from on 4 Nov
2026, and the two points left open then, and the format version, on 7 Nov.
Format `0.2.1`. Amends ADR 0072 (its reversed-indel rule and its identity
rules) and ADR 0036 §2 (its statement of the ALID convention). Points 6 and 7
record the owner's rulings of 7 Nov on panel and GWAS-SSF indels.

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
- declared row by row: a GWAS-SSF file whose harmonised alleles are its main
  columns, on a row whose harmoniser code is 10 or 11 (see 7);
- not declared: every other GWAS-SSF row (see 7); a BESD `.esi`'s A1 and A2;
  a `build-dense` association TSV; an `effect_and_other` Column Map; and a
  `ref_alt_and_effect_allele` (dbGaP MVP) Column Map, whose `ref` nothing in
  this package documents as the genome's.

An indel whose source declares no reference names no ALID. It is a malformed
identity, like a missing chromosome (#365): the reader leaves it out of both of
its streams and names it in the stream's tally (`UndeclaredReference`), and
Association Admission counts and logs it per Analysis as
`indel_reference_undeclared`, after `allele_invalid` in precedence (ADR 0072),
or, for GWAS-SSF, under a reason naming its layout (see 7). It is never stored
under a guessed order. Its SNVs and substitutions are stored as before. The
rule set's version moves to `admission/13`.

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

**6. An LD Reference Panel places an indel by a declared reference, settled
by GRCh38 evidence where the panel declares none** (owner, 7 Nov). A panel
names each variant by an id with its alleles sorted, and every panel on the IEU
host does so for its indels: 648,392 in UKB EUR and 2.1–3.2 M in each
HGDP+1kGP panel, none unsorted (13,124,104 in all). UKB's own `.afreq` marks all
9,848,829 rows `PROVISIONAL_REF?=Y`. Against GRCh38 evidence (the HGDP+1kGP
pvars, gnomAD v3.1.2's REF/ALT), UKB's id-first allele is the reference for
30.8% of 586,364 indels the evidence reads one way and its `OA` for 69.2%, so
neither column declares it. So:

- **A panel with a `REF` column** (the HGDP+1kGP AMR panel, whose `REF` matches
  the genome's at all 1,989,545 sites read one way) places each indel by it.
- **A panel without one** (UKB EUR, HGDP+1kGP AFR/EAS/EUR/SAS) places it by the
  indel-reference table beside its blocks, `ld_dir/{ancestry}/
  indel_reference.tsv.gz` (tabix-indexed), with `indel_reference.json` beside
  it: the evidence files and their checksums, the script's revision, and the
  counts by outcome. `scripts/build_panel_indel_reference.py` writes both and
  never replaces either. It settles an indel by the HGDP+1kGP pvars and
  FinnGen R13's declared `ref`, only where a source reads the key one way and
  no source reads it otherwise; a key a source holds both ways (the insertion
  and the deletion both exist), one the sources read differently, and one no
  source holds are left unsettled, each counted.
- **A panel with neither** places no indel, and says so in completion's log
  and its record.

`canonical_panel_alid` names an unsettled indel no ALID, so completion neither
matches nor fills it: a fill there counts as missing, and an observed indel
stays observed. A panel codes its eigenvectors and `EAF` on each row's `EA`
(the sorted-first allele, `construct_block.py`), and a statistic is signed for
the ALID's `A1`, so a row whose `A1` is not its `EA` -- every settled panel
deletion -- has its eigenvector row negated and its frequency complemented
when its block is read (`LDBlock.flipped`). The same rule holds an SNP whose
`EA` were not its `A1`; on the host's panels none is. A completion records how
its panel's indels took their reference (`provenance.completion.panel_indels`:
`reference_from`, each table read with its checksum and counts, the indels
read and placed).

**7. GWAS-SSF declares its reference by layout and harmoniser code** (owner,
7 Nov; it supersedes the 4 Nov rule for code 11). The reader tells the layout
from the header:

- **harmonised alleles in the main columns, `hm_code` beside them** (4,595 of
  the host's 6,078 harmonised files): a row whose code is 10 or 11 declares its
  `other_allele` the reference; any other code drops its indels
  (`indel_ssf_hm_code`);
- **the study's alleles in the main columns, harmonised ones in
  `hm_other_allele`/`hm_effect_allele`** (1,483 files): every indel is dropped
  (`indel_ssf_hm_columns`);
- **no harmoniser column** (a filtered file, for example OGS-00006's): every
  indel is dropped (`indel_ssf_unharmonised`), as decided on 4 Nov for a file
  not harmonised;
- **any other mix of harmoniser columns** fits neither layout, and nothing
  says which columns hold its harmonised alleles, so the file is refused,
  never read under a default.

Each record counts, per Analysis, the indels the code rule kept
(`indels_kept_by_hm_code`, over the rows the reader read) beside those each
reason dropped, so a Release says how many indels it took on this rule.

The evidence, at FinnGen R13 indel sites read one way (FinnGen
`finngen_R13_AB1_GONOCOCCAL`'s declared `ref`). In the main-column layout,
over 197 files (132 with indels, 57,162,435 matched rows):

| `hm_code` | FinnGen site | rows | `other_allele` = `ref` |
|---|---|---:|---:|
| 10 | insertion | 15,153,474 | 99.58% |
| 10 | deletion | 22,741,158 | 99.25% |
| 11 | insertion | 10,358,080 | 99.09% |
| 11 | deletion | 8,909,694 | 97.43% |

Pooled, 99.38% of code-10 and 98.32% of code-11 rows. File by file it is less
even: code 11 is below 95% in 51 of 131 files and code 10 in 19 of 130 (for
example GCST90837174's code-11 deletions, 412 of 76,810), mostly the deletions
of each file's minority code. The owner accepted that residual error on 7 Nov.
In the `hm_*` layout, over 35 files, `hm_other_allele` is the reference for
99.57% of code-11 insertions but 87.42% of code-11 deletions (99.00% and
97.16% for code 10); the 4 Nov evidence, from 26 such files, reproduced on
GCST90061408 but did not generalise (GCST90000293 and GCST90054782 show the
reverse failure), so that layout places no indel.

**Rescued copies are an open contract with opengwasdb-stores.** The 58
`swap_alleles` copies under `derived/ebi-gwas-catalog/` (opengwasdb-stores #155)
exchange the allele columns but keep `hm_code`, so in them `effect_allele` is
the reference and this rule reads the wrong allele on every indel row. Nothing
a build reads says a file is such a copy -- the decision is in the stores
repository's `allele-check` sidecars and the acquisition overlay, and only the
path tells -- so the reader does not guess from the path. No build manifest
names a rescued copy today; one that does needs a declared signal first.

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
- **GWAS-SSF code 11 alone, as decided on 4 Nov.** In the main-column layout
  code 10 is the more reliable code and 69% of indel rows; dropping it would
  lose them for no accuracy (7).
- **Dropping every GWAS-SSF indel**, which no code-keyed error can reach. The
  owner chose to keep codes 10 and 11 and accept the residual error (7).
- **Reading a rescued copy's reference from its path.** That is a contract
  between two repositories, and needs one (7).

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
- Stores built from BESD or `build-dense` sources, or from GWAS-SSF files in
  the `hm_*` layout or with no harmoniser column, hold no indels; each is
  logged in its build's Admission record. GWAS-SSF files in the main-column
  layout keep their code-10 and code-11 indels: all but 371 of the 96,420,588
  indel rows of the 197 files checked.
- A completion places a panel indel only by the panel's `REF` column or its
  indel-reference table, and fills nothing at the rest.
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
