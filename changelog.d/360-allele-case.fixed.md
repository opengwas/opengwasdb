- **A lower-case allele is the same base as an upper-case one, on every build
  path and in extraction (issue #360, spec §4, ADR 0075).** Alleles are
  upper-cased before identity assignment, so a source spelling `a`/`g` stores
  its associations under the ALID a source spelling `A`/`G` reaches, signed
  against the upper-cased pair's A1. Before, a Dense build kept `1:101000:a:g`
  as a variant of its own, and a Dense build from a Variant Reference dropped
  that row's associations as unmatched, silently; extraction wrote the
  lower-case ALID, which the reference reader then refused; and a Hybrid
  two-pass build refused the sources. A mixed-case pair such as `a`/`G` was
  also named `G:a` by a Dense build, against a statistic the GWAS-VCF reader
  signed on `G` and the GWAS-SSF and FinnGen readers on `A`; every reader now
  signs on the upper-cased pair. No published Store holds a lower-case allele,
  and a Hybrid single-pass build routes and names exactly as before, so
  OGS-00011, whose sources hold 133 lower-case Analyses, rebuilds unchanged. A
  single-pass harvest now compares the names a site's two spellings carry on a
  Dense build too, and `build-dense-vcf`, which has no `--on-rsid-conflict`,
  refuses names that disagree. And an identity declared under both hg38 and
  hg19 is now dropped in every spelling, by the union and by extraction alike,
  as an exact tuple both declared always was. Kept, an hg38 source's `g`/`a`
  beside the shared `G`/`A` drew the hg19 locus's associations onto the hg38
  one, a misattribution a Hybrid build could already make. A Hybrid build no
  longer refuses a pair of the two assemblies whose case differs.
