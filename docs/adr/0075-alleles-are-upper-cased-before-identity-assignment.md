# Alleles are upper-cased before identity assignment

> **Amended by [ADR 0072](0072-association-admission.md)** (issue #357). Where
> one Analysis gives both spellings of a site, routing no longer keeps the
> later. The two are resolved as any two associations on one index are, by
> Association Admission's duplicate rule: agreeing ones collapse to the first,
> and conflicting ones are both dropped and reported (`Placed.conflicting`).
> The rest stands.

Issue [#360](https://github.com/opengwas/opengwasdb/issues/360), part of epic
#58. Records the case rule every build path and extraction follow, where it
lives, and why it is not applied in the readers.

## Context

VCF and GWAS-SSF bases are case-insensitive: a source writing `a`/`g` reports
the variant a source writing `A`/`G` reports. Until this change each path
treated the lower-case spelling its own way, which #352's stage 1
characterisation found:

- A Dense build kept `1:101000:a:g` as an ALID of its own beside
  `1:101000:A:G`, and a Dense build from a Variant Reference dropped the
  lower-case row's associations as unmatched, silently.
- Extraction wrote the lower-case ALID, and the Variant Reference reader then
  refused the artifact as naming one variant twice.
- A Hybrid two-pass build refused the sources. A Hybrid single-pass build
  (`RoutingKey.CASE_FOLDED`) stored them on the upper-case ALID.
- The Ragged builders identify variants through `normalise_allele`, which
  already upper-cased.

Orientation was split as well. The GWAS-VCF reader signed z on the raw order
(`alt > ref`), and the FinnGen, GWAS-SSF and Column Map readers on the
upper-cased one (`orient_to_canonical`). For a mixed-case pair such as `a`/`G`,
a Dense build named the variant `G:a`, so the statistic of one reader or the
other was stored against the wrong A1.

The exposure, counted before the change:

- No published Store holds a lower-case allele: none of the 295,979,114 rows
  of the 17 releases' Store Variant Tables. ALIDs containing a lower-case letter
  in the chromosome, such as `11_KI270927v1_alt` (1,271 rows in OGS-00008 and
  OGS-00009, 1,565 in OGS-00027), are contigs, not alleles.
- OGS-00011's sources do. 133 of its 3,169 GWAS-SSF files spell alleles in
  lower case, 1,516,243,599 of 39,538,830,263 rows. 212 of those rows are a
  mixed-case pair whose order upper-casing changes. Its Hybrid single-pass
  build folds them, and all 97,211 of its disputed rsid keys (96,361 variants
  it leaves unnamed under `--on-rsid-conflict drop`) are a site's two
  spellings named differently, which only a comparison across spellings can
  see (ADR 0066).
- OGS-00007's ten Ragged GWAS-SSF sources spell every allele in lower case
  (202,803 rows), and its Store holds none, because the Ragged builder already
  upper-cased.
- No other source a Hybrid or Ragged release was built from holds one. That is
  OGS-00001/02/04/05/06/27 in full, and OGS-00003 and OGS-00008 as a check on
  the Dense releases. A Dense release's own ALIDs show its sources' case,
  because a Dense build kept the raw spelling, and none holds a lower-case
  allele.

## Decision

**Alleles are upper-cased before identity assignment, on every build path and
in extraction** (spec §4). Every ALID, every routing key, and every choice of
A1 — which sets the sign a statistic is stored with — is made from upper-cased
alleles.

- The rule is `opengwasdb.variants.normalise.identity_allele`. Its batch form,
  `identity_alleles`, calls it for Pass 2's per-association keys. Its one fast
  form, `identity_base`, upper-cases a reader's one-base cell with a byte mask
  on the tabular readers' per-row path. A contract test holds all three to one
  answer over every printable ASCII byte and two non-ASCII letters. An ALID is
  formed from raw alleles only by `canonical_alid`.
- The places a source allele becomes identity call them: routing's keys
  (`routing_key`, `encode_routing_keys`, `source_alid`); Pass 1's hg38
  placement; the lift's ALID and its re-orientation (`build/liftover.py`);
  extraction's two window writers; the Hybrid fold's name for an off-reference
  key; the GWAS-VCF reader's orientation; and `normalise_allele`, through which
  the FinnGen, GWAS-SSF and Column Map readers and every Ragged builder already
  oriented.
- A Source Reader keeps the source's spelling in `ref` and `alt`. It survives
  only as provenance — a Variant Reference's source keys — and in the rsid
  harvest.
- `RoutingKey` is removed, since its two modes now spell every key alike.
  Routing's functions take no `key`. What a harvest's refusals call the build
  is the caller's to say (`route_variants(label=...)`).

## Considered: upper-casing in the readers

The readers already normalise chromosome labels, and upper-casing the alleles
there too would make every spelling downstream upper-case, with no fold
anywhere. It was rejected because it erases the spelling the rsid harvest
compares. OGS-00011's 97,211 disputes would vanish. A rebuild would then name
its 96,361 withheld variants by the first-non-empty rule, which ADR 0066
decided against, and `--on-rsid-conflict` would have nothing left to decide.
Applied where identity is formed instead, the rule changes no published Store.

## Consequences

- Dense, Hybrid two-pass and extraction agree with the Hybrid single-pass
  build: a site's two spellings are one variant, holding both spellings'
  associations. Two Analyses, one spelling each, are both reached by one ALID.
  One Analysis giving both keeps the later, and routing reports the
  disagreement (`Placed.conflicting`), as for any two associations on one
  index.
- Because one routing key now spells both cases on every build, routing's case
  checks apply to every build. An identity both assemblies declare is two loci,
  however each spells it, and every spelling of it is dropped from both, as an
  exact tuple both declared always was. A Hybrid build used to refuse a pair
  whose case differed; nothing is refused for that now. Kept, any spelling
  would draw one locus's associations onto the other. The union's placement
  and extraction's windows apply this through one function,
  `cross_assembly_collisions`, so a Variant Reference drops exactly what the
  union does, and the Hybrid fold already drops a key both assemblies
  declare. A route table refuses two spellings that reach one key with
  different places or orientations.
- rsids. A single-pass harvest compares the names a site's spellings carry on
  every build; `build-dense-vcf` has no `--on-rsid-conflict` and refuses. A
  two-pass build keeps the first non-empty name, as it does for any two
  sources naming one variant differently. A Hybrid two-pass build used to
  refuse such sources, but only because their spellings became two ALIDs on one
  key. The two rules now differ for a site's two spellings, just as they
  already differed for one spelling two sources name differently, which Pass 1's
  union resolves first-wins. Whether names should be compared across sources
  everywhere, or nowhere, is left open.
- The characterisation trees match the base for every case whose inputs carry
  no lower-case allele, and for the Hybrid single-pass cases whose inputs do.
  The seven cases that changed are now cases `since="#360"`: Dense two-pass
  and ALID-list over the `hg38` manifest, its extraction, and the Hybrid
  two-pass build. Every cell the base held, they hold once its ALID is
  upper-cased. The ALID-list build also stores the two lower-case
  associations the base dropped, and the Hybrid build no longer refuses.
- On four OGS-00011 Analyses, two of them lower-case, a Hybrid single-pass
  build with the release's own reference and `drop` writes the base's 612
  Store files byte for byte, with the same 3,989 disputed keys. A Dense build
  of the same four stores 21,270,680 variants where the base stored
  31,218,470, 20,806,441 of them lower-case. Each Analysis's cells are
  unchanged once moved to the upper-case ALID.
- `format_version` does not move: no published Store's content changes.
- Breaking for Python callers: `RoutingKey` is gone, and `route_variants` and
  `RouteTable.build` take no `key=`.
