# Build Inputs Specification

Status: draft

This document defines what an LD Store build reads and the rules it applies before computing r:
the genotype fileset, how to convert other formats to it, the keep file, the PC File, the variant
filters, missing genotypes, and the PC Check. The decisions behind it are in
[ADR 0002](../adr/0002-build-inputs.md); the LD Store it writes is
[`docs/spec/ld-store.md`](./ld-store.md); the terms are in `CONTEXT.md`.

Normative language: **MUST** is a requirement a build enforces by refusing; **SHOULD** is a
recommended default. A refusal names the panel and, where there is one, the variant or sample,
and is never replaced by a quiet fix (CONTRIBUTING.md, "Fail loudly over degrading quietly").

Measured numbers are from the prototype branch `proto/1-build-inputs`, cited as
`proto/1-build-inputs@<sha>:<path>`; `…/` stands for `prototypes/build-inputs/`.

## 1. `BuildInputs`, the builder's input record

`ogld.BuildInputs` (ogld#5) holds everything a build reads and every setting it applies, and each
field is recorded in the store's provenance (`ld-store.md` §7).

| field | type | default | meaning |
|---|---|---|---|
| `pgen` | tuple of paths | — | one or more `.pgen` files (§2), each with its `.pvar` (or `.pvar.zst`) and `.psam` beside it |
| `keep` | path | — | the keep file (§4); n is its sample count |
| `pcs` | path or None | — | the PC File (§5); None only with k = 0 |
| `pc_source` | string or None | — | the PC File's declared source, recorded beside its SHA-256; None only with k = 0 |
| `panel_id` | string | — | the Source Panel's identifier, such as `hgdp1kgp-v2-EUR` |
| `ancestry` | string | — | the ancestry label, such as `EUR` |
| `genotype_source` | string | — | what the genotypes are: the release and its files |
| `k` | int | 10 | the most PCs regressed out: the first k the PC Check keeps (§8). If fewer than k survive, the build uses those, as long as at least 5 do. t_far's k is the number used |
| `mac_min` | int | 20 | the minor allele count floor (§6) |
| `info_min` | float or None | 0.8 | the INFO floor (§6); None for a sequenced panel, explicitly |

There is **no call-rate field**: genotypes must be complete (§7). A missing call is a refusal,
not a filter, so there is nothing to tune.

## 2. The pgen

A build reads PLINK 2 pgen filesets only (ADR 0002, decision 1).

- **Assembly.** GRCh38. Positions are 1-based. The builder cannot check the assembly from the
  file; `genotype_source` names it, and an LD Store records `reference_assembly` GRCh38.
- **Filesets.** One fileset per chromosome, or one for the genome. A chromosome MUST NOT appear in
  two filesets. Every fileset's `.psam` MUST hold every keep-file sample; extra samples are
  ignored, and order does not matter.
- **Records.** Biallelic: one ALT per record. A split multi-allelic site is several records at
  one position, each its own variant, and their pair at distance 0 is an ordinary pair. REF and
  ALT MUST be strings of `A`, `C`, `G`, `T` or `N`, in either case; the builder passes them
  unchanged to the engine's identity function (ogld#3), which upper-cases and left-aligns them.
  A record with `*`, a symbolic allele or `.` is refused.
- **Chromosome names.** `#CHROM` is mapped to the engine's labels: a leading `chr` (any case) is
  removed; `1`–`22` are kept. **Only autosomes are built.** A record on `X`, `Y`, `XY`, a PAR,
  `M`/`MT` (or plink2's codes `23`–`26`) is refused, naming the chromosome: the diploid dosage
  model has no rule yet for hemizygous genotypes, and the HGDP+1kGP psam files carry no sex.
- **Columns.** The `.pvar` MUST have `#CHROM POS ID REF ALT FILTER`; INFO is read only when
  `info_min` is set (§6). The FILTER column MUST be present even when every value is `.`, so that
  "nothing to filter" is said by the file, not assumed from a missing column.
- **IDs.** Set at conversion to `chrom:pos:REF:ALT` (§3). The builder never keys by ID, which
  can repeat; it names a variant by ID in a refusal.
- **Genotypes.** Hard calls, or dosages where the pgen carries them; they MUST be complete (§7).
  Phase is ignored: r is the correlation of unphased dosages, as in `plink2 --r-unphased`.

## 3. Converting to pgen

Every conversion MUST use these plink2 flags, which the Test Panel's own conversion exercises
(`tests/data/test-panel/manifest.json`, `conversion_command`). plink2 v2.0.0-a.6.9 cannot split a
multi-allelic record while importing, so a source with multi-allelic records is split first.

| flag | why |
|---|---|
| `--max-alleles 2` | drops a record that is still multi-allelic, rather than read it as biallelic |
| `--set-all-var-ids @:#:$r:$a --new-id-max-allele-len 1000` | names every variant `chrom:pos:REF:ALT`; HGDP+1kGP v2's IDs are all `.`. An allele over 1,000 bases stops the conversion instead of truncating an ID into a duplicate |
| `--make-pgen pvar-cols=filter` | writes `#CHROM POS ID REF ALT FILTER`, forcing FILTER even when it is empty (§2); add `,info` for an imputed panel (§6) |
| `--keep <file>` | optional: restricts the samples at conversion |

**VCF or BCF of hard calls**, such as HGDP+1kGP v2:

```bash
plink2 --bcf hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf \
       --max-alleles 2 --set-all-var-ids '@:#:$r:$a' --new-id-max-allele-len 1000 \
       --make-pgen pvar-cols=filter --out v2.chr22
```

Use `--vcf` for a `.vcf.gz`. Without `dosage=`, plink2 imports GT as hard calls.

**Imputed VCF** (dosages): add `dosage=DS` (Minimac's DS; `dosage=HDS` for phased dosages) and
keep INFO, `--make-pgen pvar-cols=filter,info`. plink2 regenerates hard calls from the dosages; the
builder reads the dosages.

**Multi-allelic records**: split and left-align with bcftools before plink2:
`bcftools norm -m -any -f GRCh38.fa in.vcf.gz -Ob -o split.bcf`. Left alignment matters to the
ALID (ogld#3).

**PLINK 1 bed**: `plink2 --bfile in --fa GRCh38.fa --ref-from-fa force --set-all-var-ids
'@:#:$r:$a' --new-id-max-allele-len 1000 --make-pgen pvar-cols=filter --out out`. A bed's A1/A2 are
not REF/ALT; `--ref-from-fa` sets REF from the reference. The sign of r does not depend on it (r is
signed to the engine A1, `ld-store.md` §3), but REF matters to left alignment. Indels coded `I`/`D` are
refused (§2).

**BGEN** (dosages): `plink2 --bgen in.bgen ref-first --sample in.sample --max-alleles 2
--set-all-var-ids '@:#:$r:$a' --new-id-max-allele-len 1000 --make-pgen pvar-cols=filter --out out`.
`ref-first` when the first allele is REF, as in UK Biobank's imputation; `ref-last` otherwise. A
BGEN carries no INFO: apply the INFO floor before conversion with `--extract` from the
imputation's per-variant table, set `info_min` to None, and say so in `genotype_source`.

## 4. The keep file

Text, one sample per line, header `#IID`. n, the Source Panel's sample count, is its line count
after the header. Every listed sample MUST be in every fileset's `.psam` and have a PC File row;
a duplicate IID is refused. The keep file's SHA-256 of sorted IDs is `panel.samples_sha256`
(`ld-store.md` §7).

## 5. The PC File

plink2's `.eigenvec` layout: tab-separated text, a header `#IID PC1 PC2 … PCm` (plink2 also
writes `#FID IID PC1 …`, which is accepted: samples are matched by IID), and one row per sample.

- **PCs used:** PC1…PCm in order; the covariates are the first k of them the PC Check keeps (§8),
  or all it keeps if fewer, and at least 5 MUST survive. The HGDP+1kGP PC Files have m = 20.
- **Samples:** every keep-file sample MUST have a row; extra rows are ignored; a duplicate IID is
  refused.
- **Values:** finite; a NaN or infinity is refused, naming the sample and PC.
- **Provenance** records the file's SHA-256 and its declared source (`pc_source`), and the PC
  Check's result (§8).

A PC File holds one row per sample, so it is individual-level and never published or committed
outside the Test Panel.

## 6. Variant filters

Applied in this order on the keep-file samples; a dropped variant is counted under the first one
it fails, in `variants_dropped.<reason>`, and every filter's value is in `variant_filters`
(`ld-store.md` §7).

| order | filter | default | reason recorded |
|---|---|---|---|
| 1 | FILTER is `PASS` or `.` | always | `filter` |
| 2 | MAC ≥ `mac_min`, counted on the kept samples | 20 | `mac` |
| 3 | INFO > `info_min`, from the pvar INFO subfield `R2` (Minimac) or `INFO` (IMPUTE); a pgen with neither, with `info_min` set, is refused; with both, refused | 0.8, or None | `info` |
| 4 | nonzero genotype variance on the kept samples (after PC Adjustment, nonzero residual variance) | always | `zero_variance` |

Biallelic records are the conversion's job (§3). There is no call-rate filter (§7).

MAC ≥ 20 is Pan-UKB's floor. A site where every kept sample is heterozygous passes it and has no
variance; ADR 0001's prototype found 9 such sites on AFR chr22 of the unfiltered VCF. v2's pooled
ExcHet filter removes most; EUR chr21 and chr22 have none at MAC ≥ 20.

**What the filters leave in HGDP+1kGP v2.** EUR chr22 has 1,093,149 records; 124,449 pass (all
FILTER `.`, none without variance), 50.9% of the 244,656 that ADR 0001's sizes counted from the
unfiltered gnomAD VCF. Those counts included non-PASS sites and applied no call-rate floor, for
every panel but AMR. **Rough:** Near Band bytes go roughly with the square of variant density, so
EUR's would be about 26% of the 308 GB in `ld-store.md`, near 80 GB; ogld#7 measures it.

## 7. Missing genotypes and dosages

**Rule: genotypes MUST be complete.** A build refuses an input with any missing genotype among
the kept samples, naming the number of missing genotypes, the first variant by ID and the panel.
No genotype is imputed and there is no call-rate floor (ADR 0002, decision 2).

**Why.** `plink2 --r-unphased` computes each pair's r on the samples called at both variants
(pairwise-complete): on 2.74 M EUR chr22 pairs of the gnomAD v3.1.2 VCF its float64 r equals
numpy's pairwise-complete r to 1.1×10⁻¹⁶
(`proto/1-build-inputs@96312d8:…/results/missing_window_by_call_rate.tsv`). Pairwise deletion
cannot be combined with regressing out PCs, which needs one complete vector per variant; mean
imputation can, but matches plink2 only when both variants are fully called. One missing call in
771 moves r by up to 0.0235, and no call-rate floor below 1.0 brings it within the quantisation
step (`proto/1-build-inputs@96312d8:…/results/missing_window_floors.tsv`; ADR 0002's table). With
complete genotypes, every r equals plink2's, so **the epic's k = 0 criterion holds for every pair**.

**Dosages.** Where a pgen carries dosages, r is computed on them, as plink2 does ("Dosages are
used when present"); a missing dosage is a missing genotype. Phase is ignored.

**HGDP+1kGP.** The gnomAD v3.1.2 VCFs fail this rule (60.1% of EUR chr22's MAC ≥ 20 sites have a
missing call). The phased v2 callset passes it: SHAPEIT5 filled missing calls from haplotypes, and
it holds no missing or unphased call (§10).

## 8. The PC Check

The builder runs the PC Check on every PC File before PC Adjustment, so it judges any PC File,
whatever made it (ADR 0002, decision 5). It asks whether a PC is a **local** axis, one stretch
of genome such as an inversion or the MHC, rather than ancestry.

**The rule.** PC j is local, and dropped from the covariates, only when

1. one 4 Mb window explains at least 10% of it (`window_share` ≥ 0.10), **and**
2. the rest of the genome explains it no better than chance (`rest_share` ≤ the null's 99th
   percentile).

**The covariates** are the first k = 10 PCs of the PC File that are not local, in PC order. If
fewer than 10 survive, the build uses all that do, as long as at least 5 do; with fewer than 5 it
is refused, naming the dropped PCs. t_far's k is the number of covariates actually used, and
provenance records it (owner, 7 November 2026; ADR 0002, decision 5).

**The scan.**

- **Variants:** the Variant Set's variants with MAF ≥ 0.05 on the kept samples.
- **Windows:** 4,000,000 bp, stepped by 2,000,000 bp from each chromosome's first such variant,
  so every stretch of up to 2 Mb lies wholly in one window. A window's variants are thinned evenly
  to at most 5,000; a window with fewer than 5 is skipped.
- **A window's basis:** its top min(5, m) sample-space principal components: the leading
  eigenvectors of XᵀX, X the window's genotypes standardised per variant.
- **Share:** a PC's share in a window is the fraction of its centred variance in the span of
  the window's basis.
- **Per PC** (`PcScan` in provenance): `window`, the window with the largest share, as
  `chrom:start-end`; `window_share`, that share; `rest_share`, the mean share over every window
  that does not overlap it; `local`, the verdict; and `rest_null_q99`, the threshold it was judged
  against.
- **The null:** `rest_share` of the same PC with its values permuted across samples, 200 times
  (seed 405). A permuted vector carries no genotype structure, so its shares are chance: about
  5 / n per window. `rest_share` ≤ the permutations' 99th percentile is "no better than chance".

**Provenance** (`adjustment.pc_check`, `ogld.format.PcCheck`): `window_bp` (4,000,000),
`min_share` (0.10), `scan` (one `PcScan` per PC of the file, in order), `dropped` (the local PCs'
names), and `null` (how the null was made: kind, permutations, seed, quantile).
`adjustment.covariates` names the PCs used and `adjustment.k` counts them.

**What it found on the HGDP+1kGP PC Files** (`proto/1-build-inputs@c658b36:…/results/pcs_v2_summary.tsv` and `…/results/pc_check_v2_scan.tsv`, from `scripts/make_pcs.py`'s reports):

| panel | samples (PCA + projected) | pruned SNVs | windows | local PCs | kept | covariates used | largest window share among PC1-PC10; rest of genome against null q99 |
|---|---|---|---|---|---|---|---|
| EUR | 771 (665 + 106) | 86,240 | 1,398 | none | 20 of 20 | 10 | PC1, 16:84030204-88030203: 0.312; rest 0.0733 against 0.0070 |
| AFR | 801 (608 + 193) | 210,291 | 1,399 | none | 20 of 20 | 10 | PC1, 9:39516-4039515: 0.731; rest 0.1984 against 0.0077 |
| EAS | 816 (724 + 92) | 85,818 | 1,399 | none | 20 of 20 | 10 | PC1, 14:102242093-106242092: 0.472; rest 0.1334 against 0.0074 |
| SAS | 782 (672 + 110) | 94,812 | 1,398 | none | 20 of 20 | 10 | PC1, 12:128011565-132011564: 0.466; rest 0.1603 against 0.0079 |
| AMR | 345 (341 + 4) | 75,296 | 1,398 | none | 20 of 20 | 10 | PC1, 16:6021628-10021627: 0.655; rest 0.4547 against 0.0237 |

- **No PC is local in any panel**, so every build regresses out PC1-PC10 and t_far's k is 10. Each panel's largest window share belongs to an ancestry axis that the rest of the genome also explains, 10 to 26 times the null's 99th percentile.
- **EUR PC2 is kept.** Its top window is LCT, 2:134013241-138013240, at 0.195, over the 10% bar; the rest of the genome explains 0.0204 of it against a null 99th percentile of 0.0070. gnomAD's EUR PC2 gives the same verdict (LCT 0.209, rest 0.020)
  (`proto/1-build-inputs@28b5380:…/results/local_pc_rule.tsv`).
- **Two implementations agree.** The prototype's `m10_local_pc_rule.py`, written separately, gives the same windows, shares and rest shares on EUR to every printed digit.
- **Caveat: every PC beats the null a little.** A PC from a genome-wide PCA loads on every window, so even EUR PC20's rest share (0.0085) exceeds the permutation null's 99th percentile (0.0071). A real inversion PC from such a PCA may therefore also clear the null and be kept. None of these panels has one, so the rule's power on a real local PC is untested; ogld#5's test builds a purely local PC, which it drops.

**The rule it replaces.** The first rule failed a PC when one of four named regions explained
≥ 10% of it and ≥ 2× the most any of 47 random 4 Mb windows did. It failed EUR PC2 on LCT, with
gnomAD's PCs (0.217 against 0.047) and with our own region-free ones (0.204 against 0.054)
(`proto/1-build-inputs@fb3f26c:…/results/pc_check_verdict.tsv`, `pc_check_verdict_own.tsv`). Those
windows explain up to 5.4% of PC2, far above chance: PC2 is EUR's north–south ancestry axis, and
LCT is differentiated along it. The local-PC rule asks the question that matters, whether the
rest of the genome explains the PC, and keeps it.

## 9. Long-Range LD Regions

[`long-range-ld-regions-grch38.tsv`](./long-range-ld-regions-grch38.tsv) lists the regions in
GRCh38, 1-based inclusive, each with its source and its source coordinates:

- the 24 autosomal regions of Price et al. (2008, _AJHG_ 83:132) and Weale (2010), from the GRCh37
  table of the University of Michigan Genome Wiki, "Regions of high linkage disequilibrium (LD)"
  (read 7 November 2026), as `ld-store.md` §7 recommends;
- the 17q21.31 inversion region, chr17:45,761,253–46,765,892, as Soto-Beasley et al. (2020,
  _Parkinsonism Relat Disord_, PMC7686230) give it in GRCh38;
- the MHC (chr6:28,477,797–33,448,354 in GRCh37, the GRC's MHC) and the 8p23.1 inversion
  (chr8:8,055,789–11,980,649 in GRCh37), as Pan-UKB's EUR PCA excluded them
  (atgu/ukbb_pan_ancestry `pre_process_saige_data.py`). Both lie inside a Price region; they are
  listed for their narrower sources.

Each GRCh37 endpoint is lifted alone with UCSC's `hg19ToHg38.over.chain.gz`, so a region is never
broken into fragments (the failure `ld-store.md` §7 describes in the widely copied GRCh38 table).
No endpoint fell in a chain gap. Price's LCT region, chr2:134.5–138 Mb in GRCh37, is
chr2:133,742,429–137,242,430.

`scripts/make_pcs.py` excludes every region widened by 2 Mb on each side. The margin is measured:
outside LCT and the MHC, mean r² with gnomAD's EUR PC2 per 250 kb bin stays raised to 1.5 Mb past
an edge (0.0079 against 0.0031 more than 4 Mb away), and no further
(`proto/1-build-inputs@fb3f26c:…/results/margin_EUR_pc2.tsv`); 2 Mb covers that and one 500 kb
pruning window. The regions also define "outside the excluded regions" for ogld#5's
"real LD survives" diagnostic (`ld-store.md` §7).

## 10. HGDP+1kGP inputs

- **Genotypes:** the phased HGDP+1kGP v2 callset (Koenig et al. 2024), 22 autosome BCFs, 19.1 GB,
  `https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr{1..22}.filtered.SNV_INDEL.phased.shapeit5.bcf`
  with `.csi`. Downloaded to `/data/opengwasdb/work/ogld/phased_v2/` on the IEU nodes; each file
  is checked against the bucket's CRC32C (the BCFs are composite objects, so the bucket has no
  MD5) and has a `.complete` record (`proto/1-build-inputs@28b5380:…/m09_download_v2.py`).
  ogld#7 decides their permanent home.
- **Samples:** the members files under `/data/opengwasdb/reference/hgdp1kgp-hg38*/members/`; the
  v2 counts are EUR 771, AFR 801, EAS 816, SAS 782, AMR 345 (ADR 0002 lists the five absent).
- **PC Files:** `scripts/make_pcs.py` writes `/data/opengwasdb/work/ogld/pcs/<panel>.eigenvec`
  and `<panel>.json`; declared source "ogld scripts/make_pcs.py on HGDP+1kGP v2" with its commit.
- **Conversion:** §3's BCF command, one fileset per chromosome.

## 11. The Test Panel

The Test Panel, `tests/data/test-panel/`, is v2 chr21 and chr22 for the 771 EUR samples, with the
EUR PC File and independent reference answers. Its README lists what it holds, where each edge case
is, and how `scripts/make_test_panel.py` regenerates it; `tests/testdata.py` is the only way a test
reads it.
