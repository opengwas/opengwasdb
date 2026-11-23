# Test Panel

The **Test Panel** (`CONTEXT.md`) is a real slice of HGDP+1kGP EUR, committed so that every
ogld test that needs genotypes reads the same data: 1,061 variants on chr21 and chr22 for
all 771 EUR samples, with their PC File and independent reference answers. Load it with
`tests/ld/testdata.py`; a test never parses these files itself.

Rows are **Test Panel rows**: the 0-based record order of `panel.pvar`, header excluded. Every
per-variant file is in that order. ALIDs are ogld#3's to form, so nothing here is keyed by one, and
every r is signed to the **ALT** allele; a builder re-signs it to the engine A1.

## Source, terms and the owner's decisions

- **Genotypes:** the phased HGDP+1kGP v2 callset (Koenig et al. 2024, Genome Research), the
  owner's choice of genotype source (7 November 2026, epic opengwas/ogld#1; ADR 0088):
  `gs://gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr{21,22}.filtered.SNV_INDEL.phased.shapeit5.bcf`,
  downloaded and checked against the bucket's CRC32C. All 771 members of
  `/data/opengwasdb/reference/hgdp1kgp-hg38/members/EUR.keep` are in it. It has no missing
  calls, so no imputation is involved: the build input rule is that genotypes are complete.
- **PCs:** our own 20 PCs, from gnomAD's subcontinental recipe run on v2 with the Long-Range LD
  Regions excluded (`scripts/ld/make_pcs.py`): the PCs the EUR build uses. Declared source:
  "ogld scripts/make_pcs.py on HGDP+1kGP v2 EUR (ADR 0002)".
- **Terms:** gnomAD's primary data "are available free of restrictions under the Creative Commons
  Zero Public Domain Dedication", and "All users of gnomAD data agree to not attempt to reidentify
  participants" (gnomAD Terms of Use, https://gnomad.broadinstitute.org/terms, read 7 November
  2026). The paper's Data access says "All data from this study are freely available".
- **Committing it:** the owner approved committing this public slice on 7 November 2026, choosing
  a real committed subset over a synthetic generator (ADR 0088). "Genotypes never leave the host"
  (ADR 0087) is about panels whose genotypes may not be published, such as UK Biobank.

## Settings

- **FILTER:** PASS or `.` only, from the source's own FILTER column, which the conversion keeps in `panel.pvar`.
- **MAC >= 20**, counted on the 771 EUR samples.
- **Complete genotypes:** a missing call stops the run, as it stops a build.
- **PC Adjustment** at k = 0, 10; at k = 10, t_far = 0.0274211.

Each is a parameter of `scripts/ld/make_test_panel.py`; a changed decision is one run with the new
value, which regenerates every file here.

## Files

| file | holds | bytes |
|---|---|---|
| `panel.pgen` | genotypes: the documented conversion's own output | 149,059 |
| `panel.pvar` | `#CHROM POS ID REF ALT FILTER`, one line per Test Panel row | 37,521 |
| `panel.psam` | the 771 samples | 8,926 |
| `panel.keep` | `#IID`, the 771 samples | 6,609 |
| `pcs.eigenvec` | the PC File: `#IID PC1 ... PC20` | 173,618 |
| `variants.tsv` | per row: FILTER, MAC, `passes_filters`, `drop_reason`, `components` | 44,956 |
| `reference/r.npz` | `row_i`, `row_j`, `r_k0`, `r_k10`: every same-chromosome pair of passing rows at most 10 Mb apart, row_i < row_j | 1,715,626 |
| `plink2/r_unphased_k0.npz` | `r`: plink2 --r-unphased, float64, aligned with `reference/r.npz` | 838,217 |
| `plink2/clump_r2_0.01.clumps` | plink2 --clump --clump-unphased at r2 0.01, 3,000 kb; IDs are `r<row>`, most significant lead first | 585 |
| `plink2/clump_r2_0.001.clumps` | the same at r2 0.001 | 515 |
| `gwas/z.tsv` | per row: the simulated GWAS z, `nan` where the row fails the filters | 24,805 |
| `dosages.raw` | plink2 --export A (ALT counted) for every Far Pair, Bound and band-edge row: the independent genotypes `tests/ld/test_testdata.py` recomputes r from | 49,189 |

`manifest.json` records every parameter, input, tool version and file SHA-256; `tests/ld/test_testdata.py` checks the hashes.

## How to regenerate

On an IEU compute node, from the repository root, after `scripts/ld/make_pcs.py` has written
`/data/opengwasdb/work/ogld/pcs/EUR.eigenvec`:

```bash
OPENGWASDB_LD_PLINK2=<plink2> OPENGWASDB_LD_BCFTOOLS=<bcftools> OPENGWASDB_LD_BGZIP=<bgzip>     pixi run -e dev python scripts/ld/make_test_panel.py --work <scratch dir>
```

Inputs come from documented roots (`scripts/ld/test_panel/roots.py`): `$OPENGWASDB_LD_PHASED_V2`
for the BCFs, `$OPENGWASDB_LD_PCS` for the PC File and `$OPENGWASDB_LD_REFERENCE` for the keep
file. `manifest.json` records each by root, name and hash, never by host path. A cached
conversion, genotype count or row-ID pvar is reused only when the stamp beside it records the
same inputs.

It converts chr21 and chr22 for the EUR samples with the documented command
(`docs/spec/build-inputs.md`), chooses the records below, cuts exactly those records out of the
BCFs with `bcftools view -R`, and converts them again with the same command into `panel.*`. It
checks that both conversions give identical genotypes, then computes every answer from
`panel.pgen`. It is deterministic: no timestamps, a fixed seed, and every choice is
first-by-position or strongest-first. The conversion that made `panel.*`:

```text
plink2 --vcf <work>/test_panel.vcf.gz --keep $OPENGWASDB_LD_REFERENCE/hgdp1kgp-hg38/members/EUR.keep --max-alleles 2 --set-all-var-ids @:#:$r:$a --new-id-max-allele-len 1000 --make-pgen pvar-cols=filter --out <work>/panel
```

## What the filters leave

1061 rows; **1056 pass** (chr21 468, chr22 588). Dropped, by the first filter failed: FILTER 0, MAC 5, zero variance 0. The reference holds 146,593 pairs.

**Not in v2, so built by hand in unit tests** (ADR 0088): a record that fails FILTER (v2's FILTER
column is empty), a missing call (v2 has none, and a build refuses one), and a site with no
genotype variance that passes MAC >= 20 (none on chr21 or chr22 in EUR). The builder's unit
tests (ogld#5) make each one.

## PC Adjustment's covariates

The PC Check (`docs/spec/build-inputs.md`, section 8) on the Test Panel's own common variants (33 windows of 4 Mb) drops no PC, so k = 10 regresses out PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10; `reference_r(10)` and the GWAS use exactly these. The largest window share is 0.084, of PC1 in 22:22021005-26021004. `manifest.json` holds the scan, `pc_check`.

## The edge cases, and where they are

The Far Pairs, the Bound and the band-edge pairs are also in `dosages.raw`, and `tests/ld/test_testdata.py` recomputes their r at k = 0 and 10 from it with `numpy.linalg.lstsq`, independently of `reference/r.npz`.

| edge case | rows | what is there |
|---|---|---|
| proxy_cluster_chr21 | 60-209 | 150 rows, 11,175 pairs; r2 >= 0.8 in 5,514 at k = 0 and 5,511 at k = 10 |
| proxy_cluster_chr22 | 869-1018 | 150 rows, 11,175 pairs; r2 >= 0.8 in 534 at k = 0 and 534 at k = 10 |
| gwas_region_chr21 | 250-455 | 200 rows, 19,900 pairs; r2 >= 0.8 in 78 at k = 0 and 77 at k = 10 |
| gwas_region_chr22 | 495-804 | 300 rows, 44,850 pairs; r2 >= 0.8 in 346 at k = 0 and 342 at k = 10 |
| backbone_chr21 | 0-467 | 109 rows, 3,036 pairs; r2 >= 0.8 in 0 at k = 0 and 0 at k = 10 |
| backbone_chr22 | 471-1060 | 116 rows, 3,266 pairs; r2 >= 0.8 in 0 at k = 0 and 0 at k = 10 |
| pair 10,000,000 bp apart | 475, 821 | positions 16,029,456 and 26,029,456; r_k0 0.031737, r_k10 0.024265 |
| pair 10,000,001 bp apart | 479, 823 | positions 16,231,175 and 26,231,176; not in the reference: over 10 Mb |
| pair 3,000,000 bp apart | 478, 491 | positions 16,190,010 and 19,190,010; r_k0 -0.025219, r_k10 -0.034553 |
| pair 3,000,001 bp apart | 472, 490 | positions 16,022,568 and 19,022,569; r_k0 -0.083324, r_k10 -0.080913 |
| real Far Pair at k = 10 | 6, 39 | 9,198,849 bp; r_k10 0.172795, r2 0.029858 >= t_far 0.0274211 |
| real Far Pair at k = 10 | 35, 216 | 8,517,708 bp; r_k10 0.185670, r2 0.034473 >= t_far 0.0274211 |
| real Far Pair at k = 10 | 44, 221 | 7,734,483 bp; r_k10 0.170150, r2 0.028951 >= t_far 0.0274211 |
| real Far Pair at k = 10 | 848, 1032 | 8,793,156 bp; r_k10 0.203443, r2 0.041389 >= t_far 0.0274211 |
| real Far Pair at k = 10 | 854, 1020 | 4,001,684 bp; r_k10 0.193093, r2 0.037285 >= t_far 0.0274211 |
| real Far Pair at k = 10 | 856, 1025 | 4,954,402 bp; r_k10 0.210428, r2 0.044280 >= t_far 0.0274211 |
| 3-10 Mb pair below t_far (a Bound) | 0, 11 | 3,000,282 bp, r_k10 -0.021593; 37,374 such pairs, 7 Far Pairs in all |
| same chromosome, over 10 Mb | 0, 36 | Not Measured; 135,263 such pairs |
| chr21 with chr22 | 0, 471 | Not Measured; 275,184 such pairs |
| split multi-allelic site | 476, 477 | position 16,109,998, alleles [['G', 'A'], ['G', 'T']]; the pair at distance 0 has r_k0 -0.060778 |
| indel with a long ALID | 806 | 22:21,715,659, REF 183 bp, ALT 1 bp: `chrom:pos:ref:alt` is 197 bytes, over the engine's 32-byte ALID slot |
| fails mac | 468, 469, 470, 473, 474 | row 468 MAC 0; row 469 MAC 5; row 470 MAC 1; row 473 MAC 19; row 474 MAC 10 |

## The simulated GWAS

One phenotype from 8 causal variants (rows 287, 353, 421, 537, 612, 693, 774, 798), each explaining 6% of its variance, the rest Gaussian noise from seed 20261107. Every passing row's z is its t statistic in y ~ 1 + PC1..PC10 + genotype; only z is committed. The largest |z| is 8.92. plink2 --clump with --clump-unphased, because an LD Store's r is unphased and v2's pgen holds phase (p1 = p2 = 5e-08, 3,000 kb, p = erfc(|z| / sqrt 2)), gives lead rows [612, 421, 774, 537, 353, 798, 287, 730] at r2 0.01 and [612, 421, 774, 353, 287, 703] at r2 0.001, most significant first.

## Size

The data files hold 3,049,626 bytes; this README and `manifest.json` add about 26 kB, and `tests/ld/test_testdata.py` holds the whole directory under the 5 MB budget. The reference's float64 r is most of it: r at k = 0, r at k = 10 and plink2's r, 24 bytes a pair. Pairs grow with the square of variant density, so about 2,000 variants at these clusters' density would hold roughly four times the pairs, near 13 MB. The trade made: dense clusters of 150-300 variants, where LD is, and a 300 kb backbone elsewhere, rather than float32 r, which would move a stored code near a rounding edge.
