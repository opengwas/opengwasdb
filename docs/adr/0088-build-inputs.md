# Build inputs: a PLINK 2 pgen of complete genotypes, and a PC File ogld does not compute

Formerly ogld ADR 0002.

Issue [#2](https://github.com/opengwas/ogld/issues/2), part of epic [#1](https://github.com/opengwas/ogld/issues/1).
This ADR records the owner's decisions of 7 November 2026 on what an LD Store build reads, with
the evidence that led to them and the options rejected. It **supersedes the PC wording of
[ADR 0087](./0087-near-band-and-thresholded-far-pairs.md), decision 4** ("The PCs come from
LD-pruned common variants, with the Long-Range LD Regions excluded"): the PCs are still computed
that way, but outside ogld, by one recipe, and the builder checks every PC File it is given.
Lane 2c adds the matching clarification to ADR 0087 (#2). The formats and rules are in
[`docs/spec/build-inputs.md`](../spec/build-inputs.md); the terms are in `CONTEXT.md`.

Every number below is from the prototype branch `proto/1-build-inputs`, cited as
`proto/1-build-inputs@<sha>:<path>` so a citation stays fixed if the branch moves. Most are under
`prototypes/build-inputs/`, written `…/` below.

## Context

ADR 0087 decided what an LD Store holds. The builder (ogld#5) needs to know what it reads:
which genotype formats, which PCs, and what a missing genotype means. Three facts forced the
decisions.

**Missing calls break the k = 0 equivalence.** The epic's acceptance criterion is that a build at
k = 0 matches `plink2 --r-unphased` within the quantisation step. The gnomAD v3.1.2 HGDP+1kG VCFs
hold hard calls only (`plink2 --pgen-info`: no dosages, no phased calls), with low-quality
genotypes set missing. On EUR chr22 (`proto/1-build-inputs@fb3f26c:…/results/`):

- 60.1% of the 244,656 sites with MAC ≥ 20 in EUR have at least one missing call, and 62,278
  (25%) are not FILTER=PASS.
- **plink2 uses pairwise-complete samples**: on 2.74 M pairs of chr22:30.0–30.6 Mb its float64 r
  equals numpy's pairwise-complete r to 1.1×10⁻¹⁶ in every call-rate bin
  (`proto/1-build-inputs@96312d8:prototypes/build-inputs/results/missing_window_by_call_rate.tsv`).
- Mean imputation, the only treatment that composes with regressing out PCs, matches plink2 only
  when both variants are fully called (6.4×10⁻¹⁵). One missing call in 771 already moves r by up
  to 0.0235, about 1,500 half-steps of 1/65534. **No call-rate floor below 1.0 makes them agree**
  (`proto/1-build-inputs@96312d8:prototypes/build-inputs/results/missing_window_floors.tsv`):

  | call-rate floor | chr22 MAC ≥ 20 sites kept | largest \|r_plink2 − r_mean\| |
  |---|---|---|
  | 1.0 | 39.9% | 6.4×10⁻¹⁵ |
  | 0.99 | 56.6% | 0.039 |
  | 0.95 | 69.4% | 0.192 |
  | 0.9 | 76.4% | 0.264 |

**gnomAD's subcontinental PC2 tracks LCT in EUR.** gnomAD v3.1.2's
`hgdp_tgp_meta.subcontinental_pca` PCs were computed without excluding any Long-Range LD Region.
The PC Check measured, for PC1–PC10 of all five panels, the share of each PC's variance that a
region's own top 5 genotype PCs explain (R2_local), against 47 random 4 Mb windows
(`proto/1-build-inputs@fb3f26c:…/results/pc_check_verdict.tsv`). The rule first proposed failed a
PC when a region explained ≥ 10% of it and ≥ 2× the most any window did. Of 1,350 tests, two
failed, both EUR PC2:

- **LCT/2q21.3**: R2_local 0.217 against a window median of 0.016 and maximum of 0.047 (4.6×), and
  1,237 LCT variants above PC2's genome-wide maximum single-variant r²;
- the Price et al. chr6:25–35 Mb region holding the MHC: 0.102 (2.2×).

AFR, EAS, SAS and AMR passed every region; no PC was an inversion's axis.

**Our own PCs, without the regions, carry the same axis.** EUR's PCs recomputed by gnomAD's
recipe with the four checked regions excluded ±2 Mb (`proto/1-build-inputs@28b5380:prototypes/build-inputs/m07_own_pcs.py`) reproduce gnomAD's: |corr|
0.996, 0.953, 0.922 and 0.908 for PC1–PC4. Their PC2 still explains 20.4% with LCT's own top 5
PCs, against at most 5.4% for any random window (`proto/1-build-inputs@fb3f26c:prototypes/build-inputs/results/pc_check_verdict_own.tsv`). No LCT variant was
in that PCA, so the association is LCT's real north–south differentiation, which lactase
persistence follows across Europe, not LD weighting the PCA. The first rule cannot tell the two
apart; regressing out an ancestry axis is what PC Adjustment is for.

## Decision

The owner decided (7 November 2026; epic #1 comments and the planning pass on #2):

1. **The builder reads PLINK 2 pgen only.** VCF/BCF, PLINK 1 bed and BGEN are converted with
   plink2 first, by the commands `build-inputs.md` documents. One fileset per chromosome or one
   for the genome; GRCh38 only.
2. **Genotypes must be complete.** The builder refuses an input with any missing genotype and
   names how many and which variant; there is no imputation and no call-rate parameter. Dosages,
   where a pgen carries them, are used as plink2 uses them. So the epic's k = 0 equivalence with
   `plink2 --r-unphased` stands unchanged, pair for pair.
3. **HGDP+1kGP's genotypes are the phased v2 callset** (Koenig et al. 2024, Genome Research):
   `gs://gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/`, one BCF per autosome,
   19.1 GB. It has no missing and no unphased calls; SHAPEIT5 filled in up to 10% missing calls
   per site from haplotypes, unflagged. That is read from the chr22 BCF itself, at the path above
   (7 November 2026): neither the paper (https://pmc.ncbi.nlm.nih.gov/articles/PMC11216312/) nor
   the bucket's README says how missing calls were handled. Before
   phasing it kept, on all 4,091 samples pooled, gnomAD PASS sites with HWE p ≥ 1×10⁻³⁰,
   F_MISSING ≤ 0.1 and 0.5 ≤ ExcHet ≤ 1.5, and after phasing MAC ≥ 2. The panels in it:

   | panel | members | in v2 | absent | t_far at k = 10 |
   |---|---|---|---|---|
   | EUR | 771 | 771 | none | 0.0274211 |
   | AFR | 804 | 801 | HGDP00452, HGDP00472, LP6005441-DNA_G02 | 0.0263797 |
   | EAS | 817 | 816 | HGDP00758 | 0.0258882 |
   | SAS | 783 | 782 | HGDP00175 | 0.0270298 |
   | AMR | 345 | 345 | none | 0.0623952 |

   t_far = max(0.001, 20.84 / (n − k − 1)) on the v2 sample count (ADR 0087, decision 3).
4. **PCs are precomputed and supplied in a PC File; ogld runs no PCA.** For all five HGDP+1kGP
   panels they are **our own**, from the same v2 genotypes, by one recipe script,
   `scripts/make_pcs.py`: gnomAD's subcontinental method (PCA on gnomAD's unrelated samples,
   relateds projected; SNVs with AF 5–95%; LD-pruned at r² < 0.1 in 500 kb), excluding the
   Long-Range LD Regions of `opengwasdb/ld/build/long-range-ld-regions-grch38.tsv` ±2 Mb; 20 PCs.
5. **The PC Check is the local-PC rule, and the builder runs it on every PC File.** A PC is
   local, and dropped from the covariates, when all three hold (the owner's rule, adopted
   7 November 2026 at the review of #2):
   - one 4 Mb window explains ≥ 10% of it;
   - that share is above the window-side null, the 99th percentile of the largest window share
     over 200 permutations of the PC across samples;
   - rest share / window share ≤ 0.05.

   The covariates are the first up to `BuildInputs.k` = 10 PCs it keeps; if fewer survive the
   builder uses those, as long as at least `BuildInputs.k_min` = 5 do, and refuses below 5 (the
   owner's PC minimum, decided the same day). t_far's k is the number of covariates actually used.
   Provenance records `k_target`, `k_min` and `k`, the scan (with each PC's `ratio`,
   `window_null_q99` and, as a diagnostic, `rest_null_q99`), the dropped PCs and the null.
   `k_min` applies only when PCs are used: at k = 0 no PC File is read, nothing is refused for
   want of PCs, and provenance records `k_min` as null (review of #2, 7 November 2026).

   **Why the ratio.** The owner first set the rule as "one window explains ≥ 10% and the rest of
   the genome explains it no better than chance". Measured on karyotype-like scores, it kept EUR's
   17q21.31 axis: 93% of that score lies in one window, but its rest share (0.0073) beat the rest
   null (0.0071), because H1/H2 frequency carries a trace of ancestry
   (`proto/1-build-inputs@c1c1639:…/results/rule_power.tsv`).
   - The four karyotype scores (8p23.1 and 17q21.31, in EUR and AMR) have ratios of 0.007–0.019.
   - Every real PC of EUR and AMR has ≥ 0.104, EUR PC2 lowest.
   - AMR's noise-like PC5–PC20 have 0.19–0.27.

   A bar of 0.05 separates them, so the owner adopted it. The window-side null was added in the
   same review, because at n = 345 a random vector's best window reaches 0.08–0.10 by chance.

   On our v2 PCs it drops no PC in any of the five panels: each keeps 20 of 20 and uses
   PC1–PC10. The smallest ratio of any PC is EUR PC2's, 0.104, which LCT's window explains 0.195
   of (`proto/1-build-inputs@8c25ab3:…/results/pc_check_v2_scan.tsv`).
6. **Superseded by [ADR 0101](./0101-imported-ld.md)** (owner, #88/#93,
   20–21 November 2026): published Pan-UKB covariance may now be imported.
   **Originally: imported LD matrices (Pan-UKB) are a later epic.** This epic's provenance already records
   the source kind and every covariate adjusted for (epic #1 comment of 7 November 2026).
7. **The test data is a committed real slice**, the Test Panel: v2 chr21 and chr22 for the 771
   EUR samples, with the EUR PC File and independent reference answers
   ([`tests/data/test-panel/README.md`](../../tests/data/test-panel/README.md)). The owner approved
   committing this public slice on 7 November 2026. ADR 0087's decision 5, "Genotypes never leave
   the host they are built on", still holds for every non-public Source Panel, such as UK Biobank.
   It does not hold for HGDP+1kGP, whose genotypes are already public under gnomAD's CC0 terms.
8. **An INFO floor needs INFO** (accepted at the review of #2). With `info_min` set, a pgen that
   carries no INFO score (`R2` or `INFO` in the pvar) is refused, so a sequenced panel passes
   `info_min = None` explicitly rather than skip the filter quietly.
9. **Only autosomes are built** (owner, 8 November 2026, epic #1; chrX is a later ticket). A record on X, Y,
   a PAR or MT is refused: a male's X and Y genotypes are hemizygous, the diploid dosage model has
   no rule for them, and the HGDP+1kGP psam files carry no sex to apply one.
10. **pgenlib is an optional `opengwasdb[ld-build]` extra** (owner, 8 November 2026). Reading an LD Store
   needs numpy, zarr and numcodecs only. Building one needs pgenlib, which conda-forge does not
   carry, so `import ogld` works without it and the builder refuses, naming the extra, when it is
   missing.

   _Addition, 10 November 2026 (owner, at the review of #4):_ reading an LD Store also needs
   **fsspec and aiohttp, as required runtime dependencies**, not an extra. The spec makes a read
   over HTTP(S) as much a read as one from disk: §2 keeps the whole store one Zarr hierarchy so
   that a client reads it over HTTP, and §10 reads it so. zarr opens a URL through fsspec's HTTP
   filesystem, which is async and imports aiohttp; fsspec's requests-based alternative is its
   experimental pyodide shim. Both are on conda-forge. A local read does not import aiohttp, so
   an extra was possible, and was rejected: pgenlib is an extra only because conda-forge lacks it,
   and an install that cannot read a published store over HTTP would be a reader that half works.

   _Addition, 10 November 2026 (orchestrator's fix brief for #5, at its review):_ **threadpoolctl
   joins pgenlib in the `opengwasdb[ld-build]` extra.** A build's `threads` must bound it, and each LD
   tile is a matrix product on which OpenBLAS starts a thread per core unless told otherwise:
   the review measured 64 under `build.threads` = 16. threadpoolctl holds BLAS to one thread
   for the build's duration, so the build's own workers are its whole thread count. Only a
   build needs it, so it is an extra, refused by name when missing, as pgenlib is.

   _Amendment, 23 November 2026 (orchestrator's ruling on #455, the move into opengwasdb):_
   **threadpoolctl leaves the extra.** It is a runtime dependency of opengwasdb itself, so a
   missing threadpoolctl cannot happen and is no longer a refusal. The extra, now
   `opengwasdb[ld-build]`, is pgenlib alone. What threadpoolctl does for a build is unchanged.

11. **A Source Panel is its unrelated samples** (owner, 10 November 2026, epic #1, decided at the
   review of #5's whole-panel EUR build). That build's PC diagnostic found 0.0104 of 200,000 seeded
   cross-chromosome pairs at r² ≥ 0.01, against 0.0058 from sampling: the HGDP+1kGP members files
   include 1kGP's trios (EUR has 197 first-degree pairs among 771; AFR 321 among 801), and
   relatives share haplotypes on every chromosome. At k = 0 the rate was 1.76 times what n
   predicts on all 771 EUR, and 1.07 times on the 665 unrelated. Every HGDP+1kGP panel therefore
   uses gnomAD v3.1.2's maximal set of unrelated samples (`relatedness_inference.related` false),
   which leaves no first- or second-degree pair: EUR 665, AFR 608, EAS 724, SAS 672, AMR 341.
   `scripts/make_keeps.py` makes the keep files from gnomAD's sample metadata
   (`build-inputs.md` §10). Unchanged: the PC Files, already computed on the unrelated samples with
   the relatives projected; the Test Panel, which stays at 771 as committed test data whose
   provenance says so; and #5's whole-panel cost evidence, built on 771. UK Biobank (#10) needs
   an unrelated set too, by its own kinship.

## Considered options

- **ogld computes the PCs.** Rejected by the owner: a PCA inside the builder makes every build
  depend on a pruning and PCA implementation ogld would have to own and test. A PC File is one
  input with a hash; any recipe, ours or a cohort's, can make one, and the PC Check judges it.
- **Reading VCF, BCF or BGEN natively.** Rejected: plink2 already converts each, with dosage
  import, REF/ALT handling and ID setting, and pgen is what plink2's own `--r-unphased` and
  `--clump`, the builds' references, read. One input format is one reader to test.
- **gnomAD's subcontinental PCs, under the first PC Check.** Rejected: that rule failed EUR on
  PC2 and LCT, and our own region-free PCs showed it was failing a real ancestry axis. The owner
  chose our own PCs, from the same v2 genotypes the builds read, so a PC File and its genotypes
  never disagree on samples, and every PC File passes the same check.
- **The first PC Check rule** (a region explains ≥ 10% of a PC and ≥ 2× any random window).
  Rejected: it measured whether a region stands out, not whether the rest of the genome explains
  the PC, so it failed EUR PC2, which random windows explain up to 5.4% of, as local.
- **Mean imputation with a call-rate floor.** Rejected: no floor below 1.0 matches plink2 within
  the quantisation step, and the provisional floor of 0.99 drops 43% of chr22's MAC ≥ 20 sites
  while still differing by up to 0.039. Imputation also writes values nobody observed into r
  without saying so.
- **Pairwise-complete samples.** Rejected: it matches plink2 exactly but does not compose with
  regressing out PCs, which needs one complete vector per variant.
- **A synthetic test-data generator.** Rejected by the owner in favour of a committed real slice:
  real genotypes have flipped alleles, split multi-allelic sites and LD structure a generator
  would have to imitate.

## Consequences

- **The epic's k = 0 criterion stands.** With complete genotypes, numpy's Pearson r equals
  `plink2 --r-unphased` (the Test Panel's `reference_r(0)` against its `plink2_r()`: at most 1.0×10⁻¹⁴ on all 146,593 pairs).
- **Fewer variants, so smaller stores.** v2 EUR chr22 has 124,449 sites at MAC ≥ 20, 50.9% of
  the 244,656 that ADR 0087's size estimates counted from the unfiltered VCF, which included
  non-PASS sites and applied no call-rate floor (AMR's count alone was PASS-only). Near Band bytes
  scale roughly with the square of variant density, so EUR's Near Band would be roughly 26% of the
  spec's 308 GB, about 80 GB: a rough figure, from one chromosome; ogld#7 measures it.
- **Some samples are lost.** AFR, EAS and SAS lose three, one and one samples, and t_far moves
  with n (the table above).
- **A missing call is a refusal, not a filter.** A panel with missing calls must be phased or
  imputed upstream, and its provenance says so.
- **The PC Check judges every PC File, ours included**, so a cohort's own PCs get the same test.
  A PC dropped by it costs nothing: the next PC takes its place among the first k.
- **Every input is recorded.** Provenance holds the pgen's and PC File's hashes, the PC File's
  declared source, every filter and its counts, and the PC Check's scan.
- **Clumping is held to plink2's unphased leads.** v2's pgen holds phase, and `plink2 --clump`
  uses phased r² unless told otherwise, so the Test Panel's clump goldens use `--clump-unphased`.
  Greedy clumping over the Test Panel's own unphased reference r gives the same leads.
- **The ratio cannot drop a PC in a small panel.** By chance a window holds about 5 / n of any
  vector's variance, so below about 100 samples no ratio reaches 0.05. Every HGDP+1kGP panel keeps
  at least 341 samples, AMR's unrelated ones (decision 11); a smaller panel would need the rule
  revisited.
