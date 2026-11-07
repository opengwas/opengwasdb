# LD Stores hold a 3 Mb Near Band and thresholded Far Pairs

Issue [#2](https://github.com/opengwas/ogld/issues/2), part of epic [#1](https://github.com/opengwas/ogld/issues/1).
This ADR records the owner's decisions of 7 November 2026 (epic #1, "Decisions already taken"),
and the options they rejected. It also records what the prototypes measured. The format itself is
in `docs/spec/ld-store.md`; the terms are in `CONTEXT.md`.

## Context

Clumping and LD matrices have needed individual-level genotypes on the server: plink `--bfile`
panels for OpenGWAS's 1000 Genomes service, and `pgen` panels for the engine's prototypes. That has
three costs (epic #1):

- **The panels are small.** At r² 0.001, MR's default threshold, sampling alone passes a share
  P(χ²₁ ≥ n × 0.001) of unlinked pairs. That is 37% at n = 817 and 48% at n = 503, the range of
  the 1000 Genomes panels, and 56% at n = 345, HGDP+1kGP AMR
  (`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/chance_ld.tsv`).
- **Large panels cannot be used.** Panels whose genotypes may not be published, such as UK
  Biobank, are out of reach.
- **Every request repeats the genotype reads.**

LD can be published, but individual-level data cannot (owner, 7 November 2026). The Locus Lead
prototype (opengwas/opengwasdb#62, report in opengwas/opengwasdb-stores#447) showed that no
distance or shape rule stands in for clumping. So the engine's `n_loci_*` will come from real LD
clumping at Store build time; its largest Analysis took 48.30 s to clump exactly from genotypes.
ogld provides the LD that clumping reads.

## Decision

The owner decided (epic #1, 7 November 2026):

1. **One LD Store per (ancestry panel, assembly).** The panel is the **Source Panel**, and the
   assembly is the engine's **Reference Assembly**.
2. **Near Band.** Every pair up to **3 Mb** apart is stored, with r signed to the engine's A1 and
   quantised to `int16`. Row i holds r(i, i+1 … the last variant ≤ 3 Mb away): the upper triangle,
   with implicit columns, as magenpy stores LD.
3. **Far Pairs.** Pairs **3–10 Mb** apart are stored only where r² ≥ t_far, with
   t_far = max(0.001, q / (n − k − 1)). Here q = 20.84 is the χ²₁ upper quantile at p = 5×10⁻⁶, n is
   the sample count and k is the number of PCs regressed out:

   | panel | n | t_far (k = 10) |
   |---|---|---|
   | UK Biobank EUR | 49,766 | 0.001 |
   | HGDP+1kGP EUR | 771 | 0.027 |
   | HGDP+1kGP AFR | 804 | 0.026 |
   | HGDP+1kGP EAS | 817 | 0.026 |
   | HGDP+1kGP SAS | 783 | 0.027 |
   | HGDP+1kGP AMR | 345 | 0.062 |

4. **PC Adjustment.** LD is the correlation of genotypes after regressing out the panel's
   principal components. The PCs come from LD-pruned common variants, with the **Long-Range LD
   Regions** excluded. A GWAS adjusted for PCs has null z-score correlations equal to exactly this
   LD, so it is the LD clumping and fine-mapping need.
5. **Publishing.** LD Stores may be published. Genotypes never leave the host they are built on.

A reader gets one of three answers for a pair:

| distance | answer |
|---|---|
| ≤ 3 Mb | the exact r |
| 3–10 Mb | r if r² ≥ t_far; otherwise the **Bound** "r² < t_far", never a stored 0 |
| > 10 Mb | **Not Measured** |

A request the LD Store cannot answer exactly is refused, not approximated: clumping with a window
over 3 Mb at an r² below t_far, and any window over 10 Mb.

This has the same shape as the engine's Hybrid Layout (opengwas/opengwasdb ADR 0026): two
components partition the pairs by a rule, distance here, over one variant table.

## What was measured

The prototypes are on the prototype branch `proto/1-ld-store-layout`, and every number below is
in a result file there. Each is cited as `proto/1-ld-store-layout@65f94ca:<path>`, so a citation
stays fixed if the branch moves. They ran on HGDP+1kGP chr22 at the epic's default MAC ≥ 20,
counted from the unfiltered gnomAD source. At MAC ≥ 50 the same recipe reproduces the shipped
pgens exactly: 189,014 EUR variants. Real r was computed at k = 0 on a 12 Mb slice (20–32 Mb) for
all five panels.

- **The Near Band is large, and it is almost all of an LD Store.** At MAC ≥ 20 it holds
  1.1×10¹¹ (AMR) to 4.8×10¹¹ (AFR) pairs. int16 r compresses only 1.17–1.21× (zstd level 3, at
  the spec's provisional inner chunk), because most pairs are sampling noise around 0. That makes
  an LD Store 183 GB (AMR) to 830 GB (AFR), and 308 GB for EUR. The Far Pairs add 0.1–1.4 GB a
  panel (`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/panel_sizes_inner262144.tsv`).
  AMR's counts are measured genome-wide. Those of EUR, AFR, EAS and SAS are extrapolated from
  chr22's MAC ≥ 20 / MAC ≥ 50 ratio.
- **UK Biobank EUR's size is ogld#10's to measure.** At n = 49,766, MAC ≥ 20 is MAF ≥ 2.0×10⁻⁴.
  The only UK Biobank variant list on the IEU compute nodes, under
  `/data/opengwasdb/reference/ld/ukb/`, is a MAF ≥ 0.005 set
  (`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/ukb_panel_list_maf.json`).
  Its 9.8 M variants would give a 182 GB Near Band, but the MAC ≥ 20 set is larger. Near Band
  bytes grow with variants × partners per row, so roughly with the square of variant density:
  twice the variants is about four times the bytes.
- **Far Pairs are sparse at t_far.** At k = 0, 9.4×10⁻⁵ (EUR) to 2.3×10⁻³ (AMR) of 3–10 Mb pairs
  reach t_far. By construction, sampling alone would put 5×10⁻⁶ there, so this is real LD plus
  population structure. It decays with distance, from 1.4×10⁻⁴ at 3–4 Mb to 6.8×10⁻⁵ at 9–10 Mb in
  EUR. PC Adjustment at k = 10 will lower it, most of all in AMR, the admixed panel
  (`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/r_EUR.json` and
  `proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/r_AMR.json`).
- **Reads are fast, except a whole column.** The prototype timed a real Zarr v3 sharded LD
  Store of the EUR slice. On the spec's provisional layout, from local disk and cold:
  - a 3 Mb LD matrix took 2.0 s;
  - 500 scattered variants took 0.31 s;
  - clumping one region's hits took 26 ms;
  - a single pair took 7.5 ms;
  - a variant's downstream partners took 5.8 ms;
  - its upstream partners took 0.39 s and read 638 MB.

  Over HTTP the times were similar, and so were the bytes
  (`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/access_tables.md`).
- **Zero-variance variants pass MAC ≥ 20.** A site where every called sample is heterozygous
  has no genotype variance, so its r is undefined. The AFR slice has 9 such sites, and the EAS
  and SAS slices 2 each. The builder drops and counts them; it never stores an undefined r
  (`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/r_AFR.json`,
  `proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/r_EAS.json` and
  `proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/r_SAS.json`).

## Considered options

- **Overlapping windows.** Rejected: they store most pairs two or three times and guarantee only
  half the window. Windows of 6 Mb stepped by 3 Mb are the smallest that cover every pair ≤ 3 Mb.
  On EUR chr22 they store 1.96 entries per Near Band pair (8.37×10⁹ against 4.26×10⁹). Even so, a
  pair 3–6 Mb apart is in a window only when its position happens to fall inside one
  (`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/rejected_options.json`).
- **Block-diagonal or eigendecomposed LD**, the engine's LD Reference Panel (opengwas/opengwasdb
  ADR 0031). Rejected: it has no cross-block pairs and is too low-rank for r² 0.001.
  - Of EUR chr22's Near Band pairs, 47.6% cross one of the 20 EUR LD block boundaries. That
    includes 17.4% of the pairs at r² ≥ 0.01 and 2.9% of those at r² ≥ t_far.
  - A sample of n = 771 gives an LD matrix of rank at most 770, and every one of chr22's 20
    blocks holds more variants than that (median 9,284). Truncated eigenvectors therefore
    cannot reproduce pairs at r² 0.001
    (`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/rejected_options.json`).
- **Genotypes on the server.** Rejected: genotypes cannot be published (decision 5), and every
  request would repeat the genotype reads. Exact clumping of the largest Analysis took 48.30 s
  from genotypes (opengwas/opengwasdb-stores#447).

## Consequences

- An LD Store answers every pair up to 3 Mb exactly, so clumping, proxies and LD matrices within
  a 3 Mb window never depend on a threshold.
- Each LD Store is hundreds of GB. Size scales with the square of variant density, so AFR is 2.7×
  EUR. Publishing is a question of hosting, decided with serving in #9.
- **A variant's upstream partners are scattered.** Row i holds only i's downstream partners. Its
  upstream ones are one element in each row of the preceding 3 Mb, so `proxies()`, and anything
  else that needs a whole column, reads the band's span over that 3 Mb, not one row. The
  prototypes measured this and two alternatives. The spec leaves the point open
  (`docs/spec/ld-store.md`, §11).
- t_far is recorded per LD Store, and readers do not recompute it. Every threshold, t_far
  included, is judged on the Stored r, so a reader sees one consistent rule. The cost is that a
  pair whose true r² is exactly t_far lands on whichever side its rounding puts it. It is stored
  on every HGDP+1kGP panel at k = 10. On UK Biobank EUR it reads as the Bound, for a true r² up
  to 0.0010006 (spec §3).
- A request whose threshold is below t_far is refused only when its window exceeds 3 Mb. A small
  panel such as AMR (t_far 0.062) therefore refuses more than a large one. UK Biobank EUR
  (t_far 0.001) refuses, at r2 ≥ 0.001, only windows over 10 Mb.
