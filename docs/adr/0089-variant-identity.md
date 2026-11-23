# Variant identity: a tested copy of the engine's function, not an import

Issue [#3](https://github.com/opengwas/ogld/issues/3), part of epic [#1](https://github.com/opengwas/ogld/issues/1).
The engine is opengwas/opengwasdb at `1ab2b09`.

**Who decided.** The copy-not-import choice is the ticket's implementer's, made under the
ticket's brief ("import the engine's function, or keep a tested copy held to it by a conformance
test") and framed by epic #1's decision comments. It is not an owner ruling, and the owner may
reverse it. The rulings it rests on are the epic's: ogld's runtime dependencies are numpy, zarr
and numcodecs, and the format stores the engine's ALID exactly (`ld-store.md` §2).

## Decision

`ogld.identity` holds a **copy** of the engine's identity rules (`orient_to_canonical`,
`normalise_chromosome`, `normalise_allele`), and `tests/data/identity/` holds the **engine's own
answers** for awkward inputs, which a default-tier test holds the copy to. ogld does not import
the engine.

**Amended 23 November 2026 (#456).** The copy is gone. The owner ruled on #456 that
`opengwasdb.variants` owns both ALID spellings. LD Stores use its sorted spelling
(`opengwasdb.variants.sorted_spelling`) until #462 moves them to the engine's reference-first
one (ADR 0079). The sorted spelling is judged by the engine's own `orient_to_canonical`. The
committed fixtures are frozen, and `tests/ld/test_sorted_spelling_conformance.py` still holds
the spelling to them. A lookup given a deletion's reference-first ALID refuses
(`ReferenceFirstIndel`). The reasons below stand as the history of the copy.

## Why

- The engine's runtime dependencies (pysam, scipy, pandas, scikit-learn, pyliftover, typer) are
  far heavier than ogld's. Importing it would put them in every install.
- The rules are small: upper-case, require A/C/G/T, sort the two alleles, a canonical chromosome
  label.
- A copy can drift, and drift is silent: it flips the sign of r for some variants. The fixture
  prevents it. `orient.tsv`, `chromosomes.tsv` and `reversed.tsv` were written by
  `tests/data/identity/make_fixture.py` running the engine's functions from the engine's pixi
  environment. The script refuses an engine checkout that is dirty, or that is not the one
  `opengwasdb` was imported from, and each file names the engine commit on its first line.
  `tests/test_identity_conformance.py` needs no engine at test time.
- Refusals are held to the engine's message and `reason`, and the fixture has rows with two
  faults, so the engine's order is tested: the chromosome, then **ALT before REF** (the engine's
  effect allele is the ALT), then the position, then identical alleles. Moving any check fails a
  test.

## When the engine changes

**ogld follows the engine.** `ld-store.md` §2 says the `alid` is "exactly as the engine's
identity function forms it", and a Store whose ALIDs differ from the engine's is wrong in a way
no reader sees. So an engine change to identity (a new commit that changes
`opengwasdb.variants.normalise`, or issue 012) is a cross-repo contract change. It becomes an
ogld ticket that re-runs `make_fixture.py` on the new commit, reads the TSV diff, and changes
`ogld/identity` and the fixture together. It does not wait for a failing build, because nothing
fails: the copy still agrees with the fixture it was held to.

**Hashed long ALIDs.** The engine's issue 012 and its ADR 0017 would hash a long allele into a
compact ALID and keep the full alleles beside it. If the engine adopts that, ogld follows, and
§2's "none is ever truncated" changes with it, in that ticket. Until then the engine forms a
full-string ALID at any length, and so does ogld (`orient` never truncates an allele).

## Trimming and left alignment: the engine does neither

The engine's glossary says an ALID is formed "after upper-casing, trimming and left alignment".
Its code does not. `canonical_alid` and `orient_to_canonical` upper-case, validate and sort. No
reader runs `bcftools norm` or any trimming, and the engine's
`issues/012-production-variant-normalisation.md`, which adds both, has every acceptance box
unchecked. So **ogld forms exactly what the engine forms: it neither trims nor left-aligns.** The
fixture holds indels that either step would change (`CAA`/`CA`, `ACA`/`AA`, `TCA`/`TGA`,
`AC`/`GC`) to show the alleles kept as spelt.

What the build does about alignment: `build-inputs.md` §3 runs `bcftools norm -m -any -f` only on
a multi-allelic source, to split and left-align it. The v2 conversion command has no `norm`, and
v2 holds no multi-allelic record, so v2's alleles reach the ALID as the callset spells them.

| source | records | indels | with a common suffix, or a common prefix beyond the anchor base |
|---|---|---|---|
| HGDP+1kGP v2, chr21 | 1,054,083 | 130,543 | 0 |
| HGDP+1kGP v2, chr22 | 1,093,149 | 129,069 | 0 |
| Test Panel | 1,061 | 145 | 0 |

No record has a lower-case base or a base outside A, C, G, T. **Trimming changes no record. Left
alignment could not be counted:** it needs a GRCh38 FASTA, and the IEU compute nodes have none.
The v2 callset is GATK-normalised upstream, which makes an unaligned record unlikely.

## Two records, one ALID: reversed indels

An insertion and a deletion at one position form one ALID, because the ALID sorts its alleles
(`T/TA` and `TA/T` are both `22:16029124:T:TA`, with opposite `a1_is_alt`). On v2 chr22, 11,848
ALIDs are shared this way, and 1,156 of those have both records at EUR MAC ≥ 20. `ld-store.md`
§12 requires no repeated ALID, so a build must choose.

**It follows the engine's rule** (opengwas/opengwasdb ADR 0072, `duplicate_reversed_indel`): the
record whose REF the ALID spells first (REF = A1) is kept, and the other is dropped and counted in
`variants_dropped.duplicate_reversed_indel`. The engine admits associations by this rule, so an
LD Store row means the same variant as the engine's association at that ALID. A deletion is kept
where its REF sorts first (`AT/T` over `T/AT`). `ogld.identity.resolve_reversed_indels` is the
pure rule, tested against rows of `reversed.tsv` (the engine's own BMI_IRN pair, a v2 pair, a pair
with the deletion kept, a lower-case pair). The builder (#5) applies it to the records that pass
the variant filters; the Test Store (#12) applies it through the same helper. The Test Panel has no
such pair.

Two records with identical REF and ALT, and two SNVs sharing an ALID, cannot come from a split of
one site. The helper refuses them (`IdentityError`, reason `duplicate_record`), and a build stops
on them, where the engine would drop both as conflicting.

## Behaviour inherited from the engine

- Refusals raise `ogld.identity.IdentityError(ValueError)`, carrying the engine's `reason`
  (`chromosome_missing`, `allele_invalid`; empty for the engine's other refusals). The CLI exits 2
  for it, so a bad query chromosome is not a traceback.
- `normalise_chromosome` refuses a missing label (`""`, `.`, `NA`, `NaN`, `None`, a bare `chr`) and
  a digit string outside `1`–`26` (`0`, `27`, `022`). It passes everything else through after
  dropping a `chr` prefix, as the engine does: a contig (`HLA-A`, `chrUn_gl000220`), `-1`, `+22`,
  `22.0` and `chr 22` (which keeps its space and forms `" 22:…"`). The build's autosomes-only
  rule (build-inputs §2) refuses those, not identity.
- A position that is not an integer raises the engine's `invalid position 'abc'`.
- `alid_locus` is ogld's own; the engine parses ALIDs only to sort them. It never raises, and
  returns a locus only for a canonical ALID: a chromosome of the format (`1`–`22`, `X`, `Y`, `MT`),
  a position of at most 10 ASCII digits that is positive, unpadded and at most 2³¹ − 1, and two
  distinct alleles over A, C, G, T with A1 sorted first. So `22:100:G:A`, `foo:100:A:G`, `chr22:…`,
  `22:0100:A:G` and a 5,000-digit position are all Absent, and so is the ALID of `-1`, `+22`,
  `22.0` or `chr 22`, though `orient` forms one for them.
