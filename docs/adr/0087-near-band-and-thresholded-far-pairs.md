# LD Stores hold a 3 Mb Near Band and thresholded Far Pairs

> **Decision 2's "quantised to `int16`" is superseded by [ADR 0004](./0004-per-store-encoding.md)**
> (owner, 10 November 2026). Each LD Store records its own encoding: a symmetric scale in an
> `int16` container, × 511 for the five HGDP+1kGP stores and × 2047 for UK Biobank EUR, where
> this ADR's format read one scale, × 32767, for all. So the consequence below on where a pair at
> exactly t_far lands is now spec §3's table at each store's own scale. The rest of decision 2
> stands.

> **The merge of the Far Pairs into the Proxy Index, the clarification's note of 8 November 2026,
> is superseded by [ADR 0006](./0006-far-pairs-beside-the-proxy-index.md)** (owner, 12 November
> 2026, on ogld#8's measurements). The Far Pairs are again their own component beside the Proxy
> Index: `matrix` and `region` read them, and `proxies` and `clump` read the index. Decision 3
> stands as written.

> **Clarified on 7 November 2026** (see "Clarification, 7 November 2026" at the end). The
> upstream half is settled as a Proxy Index; a matrix query returns NaN for an unstored pair and a
> number only through the caller's `fill`; and the PC part of decision 4 is superseded by
> [ADR 0002](./0002-build-inputs.md): the PCs are our own, computed outside ogld with the
> Long-Range LD Regions excluded, and the builder drops local PCs. Decision 5 holds for every
> non-public panel; the public Test Panel slice is committed. The decisions below stand as written
> otherwise.

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

## Clarification, 7 November 2026

The owner checked this ADR and the spec at ogld#2 and decided four further points the same day
(epic #1, comment of 7 November 2026, decisions 1, 2, 3 and 6). They clarify the decisions above
and do not reverse them: the answer table and the refusals stand.

**The upstream half is a Proxy Index** (decision 1 of the comment). The consequence above, "A
variant's upstream partners are scattered", is settled. Per variant, the LD Store keeps every
partner within 10 Mb on either side with r̂² ≥ t_far, stored in both directions, so `proxies()`
reads one row (0.045 MB, 9.7 ms on the prototype) instead of the Near Band's column (about 650 MB).
It costs 3.1–8.7% of a store. Because the index holds every Far Pair, in both directions, the spec
recommends it as the Far Pairs' only storage: one dense Near Band and one sparse, symmetric,
thresholded index (`docs/spec/ld-store.md` §5 and §11). That changes no answer and no refusal, and
whether to merge is the owner's decision, put to them at ogld#2. Decision 3 above is unchanged in
meaning: a 3–10 Mb pair is stored exactly when r̂² ≥ t_far.
*8 November 2026:* the owner merged them. The Proxy Index is the Far Pairs' only storage, and a
region matrix wider than 3 Mb reads about 3–9% more bytes for it (spec §11). ogld#8 measures that
cost before the first release.
*12 November 2026:* reversed on ogld#8's measurements, before any store was published: a matrix
of 500 variants on one chromosome read about 890 index chunks for its far pairs, where a Far
Pairs component reads about 11 ([ADR 0006](./0006-far-pairs-beside-the-proxy-index.md)).

**A matrix is one signed-r matrix, NaN where unstored** (decision 2). A matrix query returns the
Stored r of every pair it holds and NaN for every other pair, the Bound and Not Measured alike.
What a NaN means follows from the returned positions, chromosomes and t_far: the Bound when the
pair is 3–10 Mb apart on one chromosome, Not Measured otherwise. A caller that passes `fill`, such
as 0, gets that number in place of NaN. This replaces the epic's acceptance criterion "No query
ever turns an unstored pair into a 0" for a query whose caller asks for a fill, and the spec's
earlier rule that a reader returning a plain numeric matrix must refuse. An LD Store itself still
never holds a 0, or any code, for an unstored pair.

**The queries** (decision 3). `matrix` for a list of ALIDs; `region` for a centre and a radius,
whose width is capped at the store's far distance (radius ≤ 5 Mb); `proxies` for a list of ALIDs,
by default at r² ≥ 0.8 within 10 Mb; and `clump` for ALIDs, a score, r² and a window. Thresholds
compare r̂², values are returned as signed r, and the refusals above stand, compared with the
distances each store records. MR's r² 0.001 with a 10 Mb window is answered only where
t_far = 0.001, on UK Biobank EUR.

**PCs are our own, supplied in a PC File, and the builder drops local ones** (decision 6, and
the owner's later decisions the same day on lane 2d's measurements). ogld's library runs no PCA.
The PCs are computed outside it by a recipe script that follows gnomAD's subcontinental method with
the Long-Range LD Regions ±2 Mb excluded from the PCA, and a build reads them from a PC File. The
builder's PC Check drops a local PC by the owner's rule: one 4 Mb window explains ≥ 10% of it,
that share beats the window-side null, and the rest of the genome explains at most 0.05 of what
that window does. The covariates are the first up to 10 PCs it
keeps; if fewer survive, those that do, as long as at least 5 do, and the build is refused
otherwise. t_far's k is the number used, and provenance records the target, the minimum, the k
used, the scan, the dropped PCs and the null. Decision 4's "The PCs come from
LD-pruned common variants, with the Long-Range LD Regions excluded" is superseded by
[ADR 0002](./0002-build-inputs.md), which has the evidence: gnomAD's PCs, which did not exclude the
regions, and our own, which do, both carry EUR's north–south axis through LCT, and the rule keeps
it as ancestry. What PC Adjustment is, and why, is unchanged.

**Genotypes are complete** (the same later decisions). The builds read the phased HGDP+1kGP v2
callset, which has no missing calls, and the builder refuses an input with any. So decision 2's
Near Band at k = 0 equals `plink2 --r-unphased` exactly, to within the quantisation step, with no
imputation and no call-rate filter.

**Provenance** (the comment "Pan-UKB LD as a source"). Importing Pan-UKB's LD matrices is a later
epic, and the format allows it now. Each store records its source kind (genotypes or an imported
LD matrix), every covariate it adjusted for by name, with t_far's k counting them, and the
distances it measured: 3 Mb near, and at most 10 Mb far (`docs/spec/ld-store.md` §7).

**Decision 5 and the Test Panel** (owner, 7 November 2026). The owner approved committing the
Test Panel, a public slice of HGDP+1kGP v2 chr21 and chr22 for the 771 EUR samples, to this
repository as test data. HGDP+1kGP's genotypes are already public under gnomAD's terms, so this
does not touch decision 5, which still holds for every non-public Source Panel, such as UK
Biobank: its genotypes never leave the host they are on. ADR 0002, decision 7, records the same
from the build inputs' side.

**Panel sizes** (the same later decisions). The phased v2 callset has fewer samples in three
panels: AFR 801, EAS 816 and SAS 782, against decision 3's table's 804, 817 and 783 (EUR 771 and
AMR 345 are unchanged). So their t_far at k = 10 are 0.0263797, 0.0258882 and 0.0270298, and the
spec's §3 table uses these sizes. The consequence above that a pair at exactly t_far "is stored
on every HGDP+1kGP panel at k = 10" no longer holds: on EAS and SAS it reads as the Bound. The
table in decision 3 is left as it was decided.

The format version stays `0.1.0`: no LD Store has been written in it yet.

