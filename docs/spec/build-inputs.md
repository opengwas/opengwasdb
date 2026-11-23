# Build Inputs Specification

Status: draft

This document defines what an LD Store build reads and the rules it applies before computing r:
the genotype fileset, how to convert other formats to it, the keep file, the PC File, the variant
filters, missing genotypes, and the PC Check. The decisions behind it are in
[ADR 0088](../adr/0088-build-inputs.md); the LD Store it writes is
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
| `r_scale` | int | — | the scale of the store's encoding of r, r̂ = code / r_scale, a whole number from 1 to 32767 (`ld-store.md` §3): 511 for the HGDP+1kGP panels, 2047 for UK Biobank EUR (ADR 0090). There is no default: a build without one is refused, naming it. Provenance records it as `encoding.r_scale` |
| `k` | int | 10 | the target: the first `k` PCs the PC Check keeps are regressed out (§8). If fewer survive, the build uses those. Provenance records it as `k_target` |
| `k_min` | int | 5 | the fewest kept PCs a build accepts; below it the build is refused, naming the dropped PCs (§8). Provenance records it, and `k`, the number used, which is t_far's k. It applies only when PCs are used: with `k` = 0 no PC File is read, nothing is refused for want of PCs, and provenance records k = 0 and `k_min` as null |
| `mac_min` | int | 20 | the minor allele count floor (§6) |
| `info_min` | float or None | 0.8 | the INFO floor (§6); None for a sequenced panel, explicitly |
| `t_partner` | float | — | **From format 0.2.0**, the r² a pair within `near_bp` (3 Mb) must reach to be in the store's **Partner Rows**: the **Partner Threshold** (`ld-store.md` §5, §7; `docs/spec/partner-rows.md` §3). There is no default: a build without one is refused, naming it. It MUST be ≤ t_far, and a build given a larger one is refused, naming both values. 0.01 for the five HGDP+1kGP panels; 0.001 for UK Biobank EUR, where it equals t_far. Provenance records it as `t_partner` |

There is **no call-rate field**: genotypes must be complete (§7). A missing call is a refusal,
not a filter, so there is nothing to tune.

**`t_partner` is set per Source Panel, as `t_far` is** (owner, 18 November 2026; ADR 0097).
`t_far` follows from n and k (ADR 0087); `t_partner` does not, because it decides how much LD a
row must hold to be worth storing for `clump` and `proxies`, and a small panel keeps more pairs by
chance at the same threshold: at n = 341, 7.2% of AMR's unlinked pairs pass r² 0.01, against EUR's
1.05% at n = 665. So from format 0.2.0 a build must be told it, and a build without it is refused
rather than defaulted. It also MUST NOT exceed `t_far`: a pair between the two thresholds within
`near_bp` would then be in neither the Near Band's partner component nor anywhere else, and a
request at r2 ≥ t_far within `near_bp` could not be answered exactly
(`docs/spec/partner-rows.md` §3).

## 2. The pgen

A build reads PLINK 2 pgen filesets only (ADR 0088, decision 1).

- **Assembly.** GRCh38. Positions are 1-based. The builder cannot check the assembly from the
  file; `genotype_source` names it, and an LD Store records `reference_assembly` GRCh38.
- **Filesets.** One fileset per chromosome, or one for the genome. A chromosome MUST NOT appear in
  two filesets, and within a fileset its records MUST be one contiguous run. No two filesets may
  share a file name, since provenance names each file by its name. Every fileset's `.psam` MUST
  hold every keep-file sample; extra samples are ignored, and order does not matter: the build
  puts every fileset's samples, and the PC File's, in one order, the first fileset's.
- **Records.** Biallelic: one ALT per record. A split multi-allelic site is several records at
  one position, each its own variant, and their pair at distance 0 is an ordinary pair. REF and
  ALT MUST be strings of `A`, `C`, `G` or `T`, in either case; the builder passes them unchanged
  to the engine's identity function (ogld#3), which upper-cases them, requires nothing outside
  A, C, G, T (an `N`, an empty allele or any other character is refused, naming the record) and
  sorts the pair. It does not trim or left-align (ADR 0089): the ALID spells the alleles as the
  pgen holds them.
  A record with `*`, a symbolic allele or `.` is refused.
- **Chromosome names.** `#CHROM` is mapped to the engine's labels: a leading `chr` (any case) is
  removed; `1`–`22` are kept. **Only autosomes are built** (owner, 8 November 2026,
  epic opengwas/ogld#1). A record on `X`, `Y`, `XY`, a PAR, `M`/`MT` (or plink2's codes `23`–`26`) is refused,
  naming the chromosome. The reason: a male's X and Y genotypes are hemizygous, and the diploid
  dosage model has no rule for coding them, and the HGDP+1kGP psam files carry no sex to apply one.
  `ld-store.md` already names `X`, `Y` and `MT`, so the format is ready when the owner rules.
- **Columns.** The `.pvar` (plain, or zstd-compressed as plink2's `.pvar.zst`) MUST have
  `#CHROM POS ID REF ALT FILTER`, found by name, with one record per pgen variant; INFO is read
  only when `info_min` is set (§6). The FILTER column MUST be present even when every value is
  `.`, so that "nothing to filter" is said by the file, not assumed from a missing column.
- **IDs.** Set at conversion to `chrom:pos:REF:ALT` (§3). The builder never keys by ID, which
  can repeat; it names a variant by ID in a refusal.
- **Genotypes.** Hard calls, or dosages where the pgen carries them; they MUST be complete (§7).
  Phase is ignored: r is the correlation of unphased dosages, as in `plink2 --r-unphased`.

## 3. Converting to pgen

Every conversion MUST use these plink2 flags, which the Test Panel's own conversion exercises
(`tests/ld/data/test-panel/manifest.json`, `conversion_command`). plink2 v2.0.0-a.6.9 cannot split a
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
`bcftools norm -m -any -f GRCh38.fa in.vcf.gz -Ob -o split.bcf`. This is the only normalisation a
conversion does, and only for a source that needs the split: the v2 command above has no `norm`,
and v2 holds no multi-allelic record. ogld itself neither trims nor left-aligns (§2, ADR 0089), so
the ALID spells the alleles as that step left them.

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

**A Source Panel's samples are unrelated** (owner, 10 November 2026; ADR 0088, decision 11): its
keep file holds no first- or second-degree pair. Relatives share long haplotypes on every
chromosome, so they put LD between unlinked variants that n does not predict: with the
HGDP+1kGP members files' 1kGP trios (197 first-degree pairs among EUR's 771), the share of
seeded cross-chromosome pairs at r² ≥ 0.01 was 1.76 times what sampling predicts at k = 0, and
1.07 times among the 665 unrelated. The builder cannot judge relatedness from genotypes it does
not otherwise model, so the rule is the keep file's: §10 says how the HGDP+1kGP ones are made.

## 5. The PC File

plink2's `.eigenvec` layout: tab-separated text, a header `#IID PC1 PC2 … PCm` (plink2 also
writes `#FID IID PC1 …`, which is accepted: samples are matched by IID), and one row per sample.

- **PCs used:** PC1…PCm in order, and the header MUST name them so; the covariates are the first k of them the PC Check keeps (§8),
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
it fails, in `variants_dropped.<reason>`. Every setting is in `variant_filters` (`ld-store.md` §7):
`filter` (`["PASS", "."]`), `mac_min`, `info_min` (null when none was applied), and
`autosomes_only`, for §2's chromosome rule, which refuses a record rather than drop it.

| order | filter | default | reason recorded |
|---|---|---|---|
| 1 | FILTER is `PASS` or `.` | always | `filter` |
| 2 | MAC ≥ `mac_min`, counted on the kept samples | 20 | `mac` |
| 3 | INFO > `info_min`, from the pvar INFO subfield `R2` (Minimac) or `INFO` (IMPUTE); a pgen with neither, with `info_min` set, is refused; with both, refused | 0.8, or None | `info` |
| 4 | nonzero genotype variance on the kept samples (after PC Adjustment, nonzero residual variance: a residual sum of squares above 10⁻¹² of the centred dosages', judged once the PC Check has chosen the covariates, so after filter 5) | always | `zero_variance` |
| 5 | one record per ALID: of a reversed-indel pair, the record whose REF the ALID spells first | always | `duplicate_reversed_indel` |

Biallelic records are the conversion's job (§3). There is no call-rate filter (§7).

**Reversed indels (filter 5).** An insertion and a deletion at one position form one ALID, because
the ALID sorts its alleles: `T/TA` and `TA/T` are both `22:16029124:T:TA`, with opposite
`a1_is_alt`. They are two alleles of a repeat, not one variant, and §2 of `ld-store.md` needs one
Variant Set row per ALID. The builder follows the engine (opengwas/opengwasdb ADR 0072,
`duplicate_reversed_indel`): of two records that passed filters 1–4 and share an ALID, **it keeps
the record whose REF is spelled first in the ALID (REF = A1) and drops the other**, counting it in
`variants_dropped.duplicate_reversed_indel`. The engine admits associations by this rule, so an
LD Store row means the same variant as the engine's association at that ALID. The rule is
`ogld.identity.resolve_reversed_indels`. Two records that no split of one site gives are refused
as a malformed input, naming both: two with identical REF and ALT, and two SNVs sharing an ALID.
On v2 chr22, 11,848 ALIDs are shared by such a pair, 1,156 of them with both records at EUR
MAC ≥ 20 (spec review of ogld#3); the Test Panel has none.

MAC ≥ 20 is Pan-UKB's floor. A site where every kept sample is heterozygous passes it and has no
variance; ADR 0087's prototype found 9 such sites on AFR chr22 of the unfiltered VCF. v2's pooled
ExcHet filter removes most; EUR chr21 and chr22 have none at MAC ≥ 20.

**What the filters leave in HGDP+1kGP v2.** EUR chr22 has 1,093,149 records; 124,449 pass (all
FILTER `.`, none without variance), 50.9% of the 244,656 that ADR 0087's sizes counted from the
unfiltered gnomAD VCF. Those counts included non-PASS sites and applied no call-rate floor, for
every panel but AMR. Near Band bytes go roughly with the square of variant density, so the builds
came out well below what ADR 0087's counts predicted: ogld#7 measured EUR's Near Band at 69.4 GB,
and the five stores at 532 GB in all (`ld-store.md` §6).

## 7. Missing genotypes and dosages

**Rule: genotypes MUST be complete.** A build refuses an input with any missing genotype among
the kept samples, naming the number of missing genotypes, the first variant by ID and the panel.
No genotype is imputed and there is no call-rate floor (ADR 0088, decision 2). A build reads the
records in blocks of 65,536, in file order, and refuses the first fault it meets, in this order:

- **A malformed pvar line** (fields that do not match the header, a POS that is not an integer)
  is refused as its block is read, naming the line, whether or not a missing call came before.
- **Missing calls.** It checks each block's genotypes for missing calls before it applies the
  other rules to that block, and once it has found one it only counts missing calls in the rest,
  so the refusal counts every one from the first.
- **A block before the first missing call is judged in full:** a fault there, such as a missing
  FILTER column or a record on chrX, is refused first, and the missing calls after it are never
  counted.
- **A pvar whose record count differs from its pgen's** is refused when that fileset has been
  read, before the missing calls are, since the counts would describe the wrong records.

So an input whose first block has a missing call, as the shipped gnomAD v3.1.2 pgens' do, is
refused for its missing calls even though it has no FILTER column either.

**Why.** `plink2 --r-unphased` computes each pair's r on the samples called at both variants
(pairwise-complete): on 2.74 M EUR chr22 pairs of the gnomAD v3.1.2 VCF its float64 r equals
numpy's pairwise-complete r to 1.1×10⁻¹⁶
(`proto/1-build-inputs@96312d8:…/results/missing_window_by_call_rate.tsv`). Pairwise deletion
cannot be combined with regressing out PCs, which needs one complete vector per variant; mean
imputation can, but matches plink2 only when both variants are fully called. One missing call in
771 moves r by up to 0.0235, and no call-rate floor below 1.0 brings it within the quantisation
step (`proto/1-build-inputs@96312d8:…/results/missing_window_floors.tsv`; ADR 0088's table). With
complete genotypes, every r equals plink2's, so **the epic's k = 0 criterion holds for every pair**.

**Dosages.** Where a pgen carries dosages, r is computed on them, as plink2 does ("Dosages are
used when present"); a missing dosage is a missing genotype. Phase is ignored.

**HGDP+1kGP.** The gnomAD v3.1.2 VCFs fail this rule (60.1% of EUR chr22's MAC ≥ 20 sites have a
missing call). The phased v2 callset passes it: SHAPEIT5 filled missing calls from haplotypes, and
it holds no missing or unphased call (§10).

## 8. The PC Check

The builder runs the PC Check on every PC File before PC Adjustment, so it judges any PC File,
whatever made it (ADR 0088, decision 5). It asks whether a PC is a **local** axis, one stretch
of genome such as an inversion or the MHC, rather than ancestry.

**The rule** (the owner's, 7 November 2026). PC j is local, and dropped from the covariates,
when all three hold:

1. one 4 Mb window explains at least 10% of it (`window_share` ≥ 0.10);
2. that share is above the window-side null, the 99th percentile of the largest window share a
   vector with no genotype structure reaches (`window_share` > `window_null_q99`);
3. the rest of the genome explains at most 0.05 of what that window does
   (`ratio` = `rest_share` / `window_share` ≤ 0.05).

The ratio replaced "the rest of the genome explains it no better than chance", which kept EUR's
17q21.31 karyotype axis (below). `rest_null_q99` is still reported, as a diagnostic, and decides
nothing.

**Its limit.** By chance alone a window holds about 5 / n of any vector's variance, so a PC's
ratio cannot fall much below 5 / (n × `window_share`): below about 100 samples the rule cannot
drop any PC. Every HGDP+1kGP panel keeps at least 341 samples, AMR's unrelated ones (§10).

**What the PC Check holds in memory.** Not a chromosome's genotypes (ogld#25). The variant pass
keeps, for each variant that passes, whether it is common (`pc_check.common`, the one frequency
rule, on the float32 dosages). When a chromosome closes, each window's variants are thinned by
position alone, so only those, at most 5,000, are read again from the pgen, `threads` windows at
a time. The scan holds those windows' genotypes, about `threads` × 5,000 × n × 25 B, and the
genome's bases, 5 × n × 8 B for each of about 1,400 windows. That is bounded by a window, not by
a chromosome's rows, but it grows linearly with the samples n: at n = 49,766 about 25 GB on 4
threads and 50 GB on 8, besides the bases' 2.8 GB (measured at n = 1,000 and 4,000: 0.53 and
2.00 GB). The scan is the one a whole genotype matrix gives, window for window, and
`pc_check.windows` runs it. The check before the tiles, that some pair within 100 kb lies outside
the Long-Range LD Regions, lists a tile of rows' pairs at a time. On EUR chr6 at k = 10 on 8
threads (n = 665) the pass peaks at 1.4 GB, against 6.1 GB, and the build's peak before the tiles
is 2.0 GB, against 10.4 GB.

**The covariates** are the first `k` = 10 (`BuildInputs.k`) PCs of the PC File that are not local,
in PC order. If fewer than 10 survive, the build uses all that do, as long as at least
`BuildInputs.k_min` = 5 do; with fewer than 5 it is refused, naming the dropped PCs. t_far's k is
the number of covariates actually used (owner, 7 November 2026; ADR 0088, decision 5).

**The scan.**

- **Variants:** the Variant Set's variants with MAF ≥ 0.05 on the kept samples.
- **Windows:** 4,000,000 bp, stepped by 2,000,000 bp from each chromosome's first such variant,
  so every stretch of up to 2 Mb lies wholly in one window. A window's variants are thinned evenly
  to at most 5,000; a window with fewer than 5 is skipped.
- **A window's basis:** its top min(5, m) sample-space principal components: the leading
  eigenvectors of XᵀX, X the window's genotypes standardised per variant. A window whose
  genotypes have rank below 5, such as one of five variants two of which are identical, has
  only that many: no direction its genotypes do not span is added to its basis.
- **Share:** a PC's share in a window is the fraction of its centred variance in the span of
  the window's basis.
- **Per PC** (`PcScan` in provenance): `window`, the window with the largest share, as
  `chrom:start-end`; `window_share`, that share; `rest_share`, the mean share over every window
  that does not overlap it; `ratio`, `rest_share` / `window_share`; `local`, the verdict;
  `window_null_q99`, the threshold it was judged against; and `rest_null_q99`, a diagnostic.
- **The window-side null:** the same PC with its values permuted across samples, 200 times (seed
  405). A permuted vector carries no genotype structure. `window_null_q99` is the permutations'
  99th percentile of the largest window share; `rest_null_q99` their 99th percentile of the rest
  share.

**Provenance** (`adjustment.pc_check`, `opengwasdb.ld.format.PcCheck`): `window_bp` (4,000,000),
`step_bp` (2,000,000), `windows` (the number of windows scanned), `min_share` (0.10), `max_ratio` (0.05), `scan` (one `PcScan` per PC of the file, in order), `dropped` (the local PCs'
names), and `null` (`opengwasdb.ld.format.PcNull`, how chance was made: `kind`, the integer `permutations`
and `seed`, and `quantile`). `adjustment`
records `k_target` (10), `k_min` (5) and `k`, the number used, and `covariates` names them. At
`k` = 0 no PC Check runs: `pc_file`, `pc_check` and `k_min` are null, and `k_target` = k = 0.

**What it found on the HGDP+1kGP PC Files** (the JSON report beside each PC File in
`/data/opengwasdb/work/ogld/pcs/` on the IEU nodes, each naming the commit that made it: `scripts/ld/make_pcs.py` at `bb38fbb`, run whole on all
five panels on 10 November 2026 with the adopted `opengwasdb.ld.build.pc_check`, which selects the MAF ≥
0.05 variants itself. It wrote the five PC Files again byte for byte, and its scans equal the
earlier `--scan-only` reports, `proto/1-build-inputs@8c25ab3:prototypes/build-inputs/results/pcs_v2/<panel>.json` and
`proto/1-build-inputs@8c25ab3:prototypes/build-inputs/results/pc_check_v2_scan.tsv`, to 4×10⁻¹⁶ in every share, null and
ratio, with the same best windows):

| panel | samples (PCA + projected) | windows | local PCs | kept | k used | largest window share among PC1–PC10 (window null), and its ratio | smallest ratio of any PC |
|---|---|---|---|---|---|---|---|
| EUR | 771 (665 + 106) | 1,398 | none | 20 of 20 | 10 | PC1, 16:84030204-88030203: 0.312 (null 0.038), ratio 0.235 | PC2, 0.104 |
| AFR | 801 (608 + 193) | 1,399 | none | 20 of 20 | 10 | PC1, 9:39516-4039515: 0.731 (null 0.041), ratio 0.271 | PC15, 0.173 |
| EAS | 816 (724 + 92) | 1,399 | none | 20 of 20 | 10 | PC1, 14:102242093-106242092: 0.472 (null 0.040), ratio 0.282 | PC7, 0.164 |
| SAS | 782 (672 + 110) | 1,398 | none | 20 of 20 | 10 | PC1, 12:128011565-132011564: 0.466 (null 0.038), ratio 0.344 | PC2, 0.149 |
| AMR | 345 (341 + 4) | 1,398 | none | 20 of 20 | 10 | PC1, 16:6021628-10021627: 0.655 (null 0.092), ratio 0.695 | PC2, 0.167 |

Every panel keeps all 20 PCs and uses PC1–PC10, so t_far's k is 10 everywhere. No PC comes within a factor of two of the 0.05 bar: the smallest ratio is EUR PC2's, 0.104.

**The evidence the owner chose the ratio on** (`proto/1-build-inputs@c1c1639:…/results/rule_power.tsv`).
Karyotype-like scores are the top PC of an inversion region's own genotypes, orthogonalised to
PC1–PC9 and scanned as if each were a PC: they are what a PC that tracks an inversion looks like.

| vector | window share | window null q99 | rest share | rest null q99 | ratio | local |
|---|---|---|---|---|---|---|
| EUR karyotype, 8p23.1 | 0.981 | 0.039 | 0.0070 | 0.0072 | 0.007 | yes |
| EUR karyotype, 17q21.31 | 0.934 | 0.040 | 0.0073 | 0.0071 | 0.008 | yes |
| AMR karyotype, 8p23.1 | 0.706 | 0.087 | 0.0133 | 0.0221 | 0.019 | yes |
| AMR karyotype, 17q21.31 | 0.946 | 0.085 | 0.0138 | 0.0195 | 0.015 | yes |
| EUR PC2, LCT | 0.195 | 0.038 | 0.0204 | 0.0070 | 0.104 | no |
| every real PC of EUR and AMR | | | | | ≥ 0.104 | no |
| AMR PC5–PC20, noise-like | 0.055–0.085 | 0.080–0.098 | | | 0.19–0.27 | no |

- **The four karyotype scores sit at 0.007–0.019, every real PC at ≥ 0.104** (EUR PC2 lowest,
  then AMR PC2 at 0.167), and AMR's noise PCs at 0.19–0.27. A bar of 0.05 drops every karyotype and
  no real PC, with a factor of two to spare on each side.
- **Under the earlier rule EUR's 17q21.31 score was kept**: its rest share (0.0073) beat the rest
  null (0.0071), because H1/H2 frequency carries a trace of ancestry that orthogonalising does not
  remove. The ratio drops it.
- **The window null keeps small panels' noise PCs.** AMR PC5–PC20 reach window shares of
  0.055–0.085 against window nulls of 0.080–0.098; without it, one crossing 0.10 by chance would
  meet the first condition.
- **Two implementations agree** on the shares: the prototype's `proto/1-build-inputs@28b5380:prototypes/build-inputs/m10_local_pc_rule.py`, written
  separately, gives the same windows, window shares and rest shares on EUR as lane 2d's reference
  to every printed digit. ogld#5 adopted that reference as `opengwasdb/ld/build/pc_check.py`, the one
  implementation: the builder, `scripts/ld/make_pcs.py` and the Test Panel's script all run it, and
  it selects the MAF ≥ 0.05 variants itself. `tests/ld/test_pc_check.py` holds it to the three real
  verdicts above (EUR 17q21.31 dropped, EUR PC2 kept, AMR PC7 kept), and to the Test Panel's
  scan as the reference made it before the adoption (its manifest's `pc_check`), to 10⁻¹².

**The rule it replaces.** The first rule failed a PC when one of four named regions explained
≥ 10% of it and ≥ 2× the most any of 47 random 4 Mb windows did. It failed EUR PC2 on LCT, with
gnomAD's PCs (0.217 against 0.047) and with our own region-free ones (0.204 against 0.054)
(`proto/1-build-inputs@96312d8:prototypes/build-inputs/results/pc_check_verdict.tsv`, `proto/1-build-inputs@fb3f26c:prototypes/build-inputs/results/pc_check_verdict_own.tsv`). Those
windows explain up to 5.4% of PC2, far above chance: PC2 is EUR's north–south ancestry axis, and
LCT is differentiated along it. The local-PC rule asks the question that matters, whether the
rest of the genome explains the PC, and keeps it.

## 9. Long-Range LD Regions

[`opengwasdb/ld/build/long-range-ld-regions-grch38.tsv`](../../opengwasdb/ld/build/long-range-ld-regions-grch38.tsv),
data of the package, which owns it, lists the regions in
GRCh38, 1-based inclusive, each with its source and its source coordinates:

- the 24 autosomal regions of Price et al. (2008, _AJHG_ 83:132) and Weale (2010), from the GRCh37
  table of the University of Michigan Genome Wiki, "Regions of high linkage disequilibrium (LD)"
  (read 7 November 2026);
- the 17q21.31 inversion region, chr17:45,761,253–46,765,892, as Soto-Beasley et al. (2020,
  _Parkinsonism Relat Disord_, PMC7686230) give it in GRCh38;
- the MHC (chr6:28,477,797–33,448,354 in GRCh37, the GRC's MHC) and the 8p23.1 inversion
  (chr8:8,055,789–11,980,649 in GRCh37), as Pan-UKB's EUR PCA excluded them
  (atgu/ukbb_pan_ancestry, https://github.com/atgu/ukbb_pan_ancestry/blob/master/pre_process_saige_data.py). Both lie inside a Price region; they are
  listed for their narrower sources.

Each GRCh37 endpoint is lifted alone with UCSC's `hg19ToHg38.over.chain.gz`, so a region is never
broken into fragments. The widely copied GRCh38 table, plinkQC's `high-LD-regions-hg38-GRCh38.txt`
and the Genome Wiki page's own, is such a fragmentary liftover: its only chr17 entry is 33 bp long,
GRCh37's chr2:134.5–138 Mb becomes 119 bp, and chr3:89–97.5, chr5:98–100.5 and chr11:46–57 Mb are
missing (both read 7 November 2026).
No endpoint fell in a chain gap. Price's LCT region, chr2:134.5–138 Mb in GRCh37, is
chr2:133,742,429–137,242,430.

`scripts/ld/make_pcs.py` excludes every region widened by 2 Mb on each side. The margin is measured:
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
  v2 counts are EUR 771, AFR 801, EAS 816, SAS 782, AMR 345 (ADR 0088 lists the five absent).
  These are the PC Files' samples (below).
- **Keep files:** `scripts/make_keeps.py` writes `/data/opengwasdb/work/ogld/keep/<panel>.keep`
  (`$OPENGWASDB_LD_KEEP`), plink2's `#IID` layout: each panel's PC File samples that gnomAD
  v3.1.2's sample metadata does not mark related (`relatedness_inference.related` false, its
  maximal set of unrelated samples), in the PC File's order. A sample the metadata lacks is refused. EUR 665,
  AFR 608, EAS 724, SAS 672, AMR 341. A build reads these (§4); the members files are the PC
  Files' input only.
- **PC Files:** `scripts/make_pcs.py` writes `/data/opengwasdb/work/ogld/pcs/<panel>.eigenvec`
  (`$OPENGWASDB_LD_PCS`) and, beside it, the PC Check's JSON report, which names that file by its
  root variable and its SHA-256; declared source "scripts/ld/make_pcs.py on HGDP+1kGP v2" with its
  commit.
- **Conversion:** §3's BCF command, one fileset per chromosome.

## 11. The Test Panel

The Test Panel, `tests/ld/data/test-panel/`, is v2 chr21 and chr22 for the 771 EUR samples, with the
EUR PC File and independent reference answers. Its README lists what it holds, where each edge case
is, and how `scripts/ld/make_test_panel.py` regenerates it; `tests/ld/testdata.py` is the only way a test
reads it.

- **Reference r** at k = 0 equals `plink2 --r-unphased` to 1.0×10⁻¹⁴ on every pair. At k = 0 and
  k = 10 it is recomputed by a second route, `numpy.linalg.lstsq` residuals on plink2's
  `--export A` dosages of the threshold rows (`dosages.raw`), to 1×10⁻¹².
- **Clump goldens** use `plink2 --clump --clump-unphased`: v2's pgen holds phase, and plink2
  otherwise clumps on phased r², while an LD Store's r is unphased (§2). Greedy clumping over the
  Test Panel's own `reference_r(0)` gives the same leads at r² 0.01 and 0.001 under either reading
  of the threshold (≥ or >).
- **k = 10** uses the first 10 PCs the PC Check keeps on the Test Panel's own common variants:
  PC1–PC10.

## 12. Published LD import inputs

`opengwasdb ld import OUT --work WORK --metadata metadata.json --variants variants.tsv.bgz
--chain hg19ToHg38.over.chain.gz --fasta GRCh38.fa.gz` imports Pan-UKB EUR
(ADR 0101). Its dependencies are `opengwasdb[ld-import]`, not Hail. The fixed source is
`https://pan-ukb-us-east-1.s3.us-east-1.amazonaws.com/ld_release/UKBB.EUR.ldadj.bm`.

- `--metadata` supplies Hail BlockMatrix metadata (`blockSize`, `nRows`, `nCols`,
  `maybeFiltered`, `partFiles`). Native Hail 0.2.64 parts are decoded, never
  interpreting an unallocated block as a measured zero.
- `--variants` is a plain or gzip/BGZF TSV exported once with
  `scripts/ld/export_panukb_variants.py`, outside the package. Required columns
  are `locus` (`chrom:position`, 1-based GRCh37), `alleles` (JSON REF/ALT),
  `AF` (ALT frequency) and `idx` (the existing global matrix row index).
  Chromosomes are contiguous, as in Hail's locus-keyed export, so planning
  holds only one chromosome's row objects at a time; duplicate global indices
  and a chromosome returning after it closed are refused.
- `--chain` maps GRCh37 to GRCh38. REF and ALT are reverse-complemented on
  negative-strand mappings, and REF must match the chromosome in `--fasta`
  (plain or gzip). Unmapped/ambiguous loci, chromosome/strand or allele-span
  failures, allele mismatch/identity, invalid frequency and every member of
  an identity collision are dropped under named loss classes, as in opengwas/ogld#89.
- A non-finite or non-positive covariance diagonal drops its variant, counted
  separately. Surviving covariance is divided by sqrt(C_ii C_jj), then signed
  with `ogld.identity.orient`. The existing writer alone quantises at ×2047
  and emits Near Band, Far Pairs, Partner Rows and the ALID companion.
- The largest complete target radius is measured after liftover per
  chromosome; the store records their minimum, and provenance records all
  radii. A radius no larger than the mandatory Near Band is refused.
- Displaced liftover segments are dropped until every chromosome's radius
  reaches `--far-target-bp` (default 8,000,000; must be above `near_bp` and at most
  `far_bp`'s maximum, 10,000,000, refused at parse time otherwise). The
  GRCh37−GRCh38 offset splits segments where it jumps by more than 100 kb; while
  a target-radius window spans more than the 10 Mb source band, the segment holding
  one extreme with fewer surviving variants is dropped (ties: the one with the lowest
  GRCh37 position). A segment of more than `--displaced-cap` variants (default
  10,000; 0 or more) is never dropped: the chromosome is refused naming its
  GRCh37 and GRCh38 spans and the target. Drops are the `liftover_displaced`
  loss class; provenance records the target, the cap and the per-chromosome
  counts. A chromosome already reaching the target loses nothing. Both options
  are pinned in `WORK`.
- `--ledger` optionally supplies a JSON list of `{name, bytes, sha256}` parts
  (SHA-256 may be absent on first acquisition). Otherwise the public HTTPS
  listing supplies byte counts. `--parts` is an optional **read-only** directory
  of existing parts. New downloads use stdlib HTTPS with retries (8 attempts,
  with backoff), listing byte counts and atomic SHA-256 receipts in
  `WORK/parts/.sha256/`. S3 multipart ETags
  are not SHA-256 hashes: first acquisition pins bytes locally, and later
  reads verify that pin. A supplied SHA-256 is always enforced.
- `--download-workers N` (default 8) fetches a chromosome's missing parts in
  parallel before it is written; each still takes the single-part path (byte
  count, SHA-256 receipt, atomic rename, ownership). The first failed part, once
  in-flight downloads stop, refuses the import by name. Each connection has its
  own short connect timeout (12 s) beside the 120 s read timeout. The bucket
  host's addresses are tried in turn; an address whose TCP connect or TLS
  handshake fails is not tried again in that run, while a timeout, reset or HTTP
  error after the connect leaves it in use and the attempt is retried. If every
  address is dead and none has answered, the dead marks are cleared. TLS connects to the chosen address but verifies the
  bucket's hostname.
- `WORK` is exclusively locked and pins the metadata, TSV, chain and FASTA by
  SHA-256, the radius target and displaced cap, plus the supplied ledger's SHA-256 and the output path. Repeat the same command to resume. A complete
  single-chromosome store in WORK is the checkpoint; an unfinished chromosome
  is rebuilt, never finished over. Retained input parts are re-hashed; deleted
  parts of closed chromosomes are not fetched again.
- `--delete-parts` deletes only owned downloads, after close and after their
  last consuming chromosome (a source block can span a chromosome boundary).
  A receipt grants deletion only when `downloaded` is true and its recorded
  device/inode/ctime still identify the downloaded file. Supplied ledgers
  cannot grant ownership. `--parts` and `WORK/parts` must be distinct and
  disjoint after resolving symlinks: an alias or containment in either
  direction is refused before receipts or downloads are written. Files
  supplied at invocation also remain protected by file identity. Receipts
  without ownership evidence never permit deletion. The import summary reports
  how many parts were kept because the importer did not own them. Without
  `--delete-parts`, parts are retained. `--threads` bounds compression and
  validation workers. Only one chromosome's source working set is live at a time.
- Closed immutable chromosome components are joined by `StoreWriter.join`
  using hard links on the same filesystem. Variant columns are concatenated
  in engine order, with chromosome-local LD coordinates unchanged. A join
  refuses mismatched encodings/distances/thresholds, or chromosome claims
  disagreeing with the supplied panel, adjustment, filters, assembly or source
  kind; matching n and k alone cannot establish one Source Panel. The final
  store is validated and atomically renamed to OUT, which must not exist.
  WORK and OUT must be on the same filesystem. Keep WORK until publication
  succeeds; afterwards its staging stores may be removed without harming OUT.

Provenance records `source.kind = ld_matrix`. Its `source.description` is a
JSON object naming the source URL, licence, producer commit, transformation,
chain, published QC (not reapplied) and per-chromosome radii; `source.sha256`
holds every part and the metadata, TSV, chain and FASTA pins. The panel is
Pan-UKB EUR, n = 420,542, with 15 named covariates (age, sex, age×sex, age²,
age²×sex and PC1–PC10); k is not described as 15 PCs. Losses are in
`variants_dropped`. Thresholds follow `ld-store.md` §3: `t_partner = t_far =
0.001`, far code floor 65. The existing `opengwasdb ld validate` is the sole validator.
