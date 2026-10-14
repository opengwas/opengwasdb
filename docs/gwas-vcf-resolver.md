# Resolving GWAS-VCF Analysis metadata

`opengwasdb resolve-analyses` supports `source_reader_capability=opengwasdb.gwas-vcf`
through the same bounded resolver API as GWAS-SSF. For each Analysis it performs
one streaming `bcftools query` over source records, after a header capability
probe, and produces ancestry and phenotype-SD evidence without reopening the
records for reference-AF fallback.

## Required assembly declarations

Every manifest row must provide `source_genome_build`. `source_assembly` is a
compatibility alias; if both are populated, their normalized builds must agree.
`--source-build` is an explicit fallback only for rows that omit both. The
ancestry reference build is always supplied with `--reference-build`.

Same-build inputs proceed without transformation. A GRCh37/hg19 GWAS-VCF used
with a GRCh38/hg38 reference requires `--chain-file`. Each lifted SNP must have
one mapping; a reverse-strand mapping complements both source alleles before
canonical orientation, so AF and signed effects still name the transformed
effect allele. Ambiguous mappings and indels (whose target alleles a single
position lift cannot establish) fail rather than choosing or inventing Variant
Identity. Other build directions and cross-build Source Formats without an
implemented liftover projection produce a controlled failure before source
records are read. Builds are not inferred from filenames, coordinates, or VCF
headers.

Source build, reference build, chain path, chain SHA-256, chain size and chain
mtime are stored in each resolution record and its resume fingerprint. Source
identity continues to use the manifest-recorded checksum together with the
observed file size and mtime; the resolver does not hash every source body.

## Supported VCF statistics

The VCF projection reads one sample's FORMAT fields:

- `EZ` is the preferred z score when finite;
- otherwise a usable `ES / SE` supplies z;
- resolver beta is exactly `z * SE`;
- `AF` is the ALT/effect-allele frequency and is reoriented with the allele;
- `SE` must be finite and positive.

The existing ancestry rule is unchanged: a canonical, non-palindromic row needs
usable AF and SE, but not an effect. Phenotype-SD evidence retains the existing
association eligibility, so a missing effect excludes that row from SD evidence
without excluding an otherwise usable ancestry row. A missing SE excludes the
row from ancestry evidence too. Multiallelic and unnormalisable identities keep
the existing reader exclusions.

The reader holds at most one metrics chunk (50,000 rows by default); ancestry
and SD evidence remain bounded by the resolver's extraction panel and evidence
sample. A non-zero `bcftools` exit or structurally malformed query output is a
controlled per-Analysis failure, never a successful short EOF. When a resolver
bound closes a still-running query intentionally, cleanup terminates it, waits
for at most five seconds, then kills and bounded-waits again. A process already
known to have failed is never treated as that intentional cancellation.

## Explicit limitations

GWAS-VCF resolution does not support a declared INFO/imputation-score filter.
The manifest parser refuses that configuration rather than silently ignoring
it. The path does not choose method tiers, alter ancestry gates or formulas,
decide Store membership, or execute a Store build. Those remain caller-owned
policy as described by ADR 0044.
