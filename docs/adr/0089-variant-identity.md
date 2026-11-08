# Variant identity: a tested copy of the engine's function, not an import

Issue [#3](https://github.com/opengwas/ogld/issues/3), part of epic [#1](https://github.com/opengwas/ogld/issues/1).
The engine is opengwas/opengwasdb at `1ab2b09`.

## Decision

`ogld.identity` holds a **copy** of the engine's identity rules (`orient_to_canonical`,
`normalise_chromosome`, `identity_allele`, `normalise_allele`), and `tests/data/identity/` holds
the **engine's own answers** for awkward inputs, which a default-tier test holds the copy to.
ogld does not import the engine.

## Why

- The engine's runtime dependencies (pysam, scipy, pandas, scikit-learn, pyliftover, typer) are
  far heavier than ogld's (numpy, zarr, numcodecs). Importing it would put them in every install,
  and `import ogld` must work with none of them.
- The rules are small and stable: upper-case, require A/C/G/T, sort the two alleles, canonical
  chromosome label. A copy is about 60 lines.
- A copy can drift, and drift here is silent: it flips the sign of r for some variants. The
  fixture is what prevents it. `orient.tsv` (70 records) and `chromosomes.tsv` (88 labels) were
  written by `tests/data/identity/make_fixture.py` running the engine's functions from the
  engine's pixi environment; both files name the engine commit (`1ab2b09`) on their first line.
  `tests/test_identity_conformance.py` needs no engine at test time. Refusals are held to the
  engine's message as well as to `ValueError`.
- Checked by mutation: with A1 set to REF always, the conformance test fails on the fixture's
  rows where ALT sorts first, and 21 tests fail in all.

To follow the engine: re-run the script on the engine's new commit, read the diff of the two
TSVs, and change `ogld/identity` until the test passes. A diff is a decision, not a refresh.

## Trimming and left alignment: the engine does neither

The engine's glossary says ALID is formed "after upper-casing, trimming and left alignment".
Its code does not. `canonical_alid` and `orient_to_canonical` upper-case, validate and sort.
No reader runs `bcftools norm` or any trimming, and engine issue `issues/012-production-variant-normalisation.md`,
which adds both (and ADR 0017's long-allele hashing), has every acceptance box unchecked. So
**ogld forms exactly what the engine forms: it neither trims nor left-aligns**, and the fixture
holds indels that either step would change (`CAA`/`CA`, `ACA`/`AA`, `TCA`/`TGA`, `AC`/`GC`) to
show the alleles kept as spelt.

Counts, on the data ogld builds from:

| source | records | indels | with a common suffix, or a common prefix beyond the anchor base |
|---|---|---|---|
| HGDP+1kGP v2, chr21 | 1,054,083 | 130,543 | 0 |
| HGDP+1kGP v2, chr22 | 1,093,149 | 129,069 | 0 |
| Test Panel | 1,061 | 145 | 0 |

No record has a lower-case base, a multi-allelic ALT or a base outside A, C, G, T either. **Trimming
therefore changes no record. Left alignment could not be counted:** it needs a GRCh38 FASTA and
there is none on the IEU compute nodes (`bcftools norm -f` has nothing to read). The v2
callset is GATK-normalised upstream, which makes an unaligned record unlikely, not impossible.

Where this stops being true: when the engine implements issue 012, its ALID for an indel the
current rule leaves alone will change, and so will ogld's if it follows. Build-inputs §3 already
has the user left-align with `bcftools norm` before conversion, so an LD Store built on aligned
input will agree with that engine as well.

## Other behaviour inherited from the engine

- A non-numeric label that is no alias (`HLA-A`, `chrUn_gl000220`) passes `normalise_chromosome`
  unchanged, as in the engine. The builder's autosomes-only rule (build-inputs §2) refuses it, not
  identity. `"0"`, `"27"` and any other number refused, as are `"" . NA NaN None` and a bare `chr`.
- `orient` judges the chromosome, then the alleles, then the position, then identical alleles.
- `alid_locus` is ogld's own, since the engine parses ALIDs only for sorting (`alid_sort_key`
  raises on a malformed one). It returns a locus only for a canonical, unpadded ALID.
