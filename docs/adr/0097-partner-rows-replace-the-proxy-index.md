# Partner Rows replace the Proxy Index for clumping and proxies

Part of the Partner Rows epic. The design is `docs/spec/partner-rows.md`. It supersedes, for
`clump` and `proxies`, decision 1 of epic [#45](https://github.com/opengwas/ogld/issues/45)
("keep the upper triangle and Zarr v3; no symmetric duplication and no independent row-frame
format"). The terms are in `CONTEXT.md` once the epic adds them.

**Who decided.** The owner, 18 November 2026, on the measurements of 17–18 November 2026
(`proto/partner-rows`), taking decisions D1–D6 of `docs/spec/partner-rows.md` §1.

## Context

Clumping reads one lead's LD with the hits still open in its window. The Near Band stores that
LD in a row-major upper triangle, in 512 KiB chunks. A lead's upstream partners are a column, one
element in each of thousands of earlier rows. On GCST90476460 (260,821 hits, 3,869 leads, r²
0.01) the reader read 10.7 GB and decoded 22.8 GB, and took 23–26 s cold. Epic #45's codec and
scheduler work reduces the time per decoded byte, but not the 22.8 GB.

Two measurements on the published HGDP+1kGP k = 10 stores changed the trade:

- Only 5.0% of EUR chr22's Near Band pairs have r̂² > 0.01, 4.1–5.0% on AFR, EAS and SAS, and
  16.3% on AMR (n = 341). A variant's two-sided row at that floor has a median of 800 entries,
  against 19,971 dense.
- In both directions and compressed, rows at r̂² > 0.01 project to 75 GB over the five panels,
  against 533.7 GB of stores today. Dense symmetric rows, rejected by #45, project to about 181
  GB for EUR alone.

At r² 0.001 there is no such sparsity. 46.2% of EUR chr22's pairs pass, against 41.9% from
sampling alone, so about 90% of the passing pairs are noise.

## Decision

1. **A new component, the Partner Rows** (`partners/<chrom>/`). For each variant it holds every
   partner within `far_bp`, on both sides, at |code| ≥ the Partner Floor within `near_bp` and ≥
   the far code floor beyond it. Every code is a copy of a Near Band or Far Pairs code. `clump`
   and `proxies` read only this component.
2. **It replaces the Proxy Index** (D1). Since t_partner ≤ t_far, the index is the Partner Rows
   thresholded at the far code floor. `index/` and its validation rules are removed. Net about
   +33 GB (+6%) over the five panels.
3. **Below t_partner within `near_bp`, `clump` and `proxies` refuse** (D2), naming
   `threshold_below_t_partner`. They never read the Near Band. This is a breaking change: r²
   0.001 clumps on the HGDP+1kGP stores, answered today, are refused.
4. **t_partner is set per reference dataset and recorded in provenance** (D3), as t_far is. It
   is a required build input with no default, and never above t_far: 0.01 for HGDP+1kGP and
   0.001 for UK Biobank EUR.
5. **Plain files, not Zarr** (D4). G rows to a zstd frame, with checksum and content size,
   concatenated in `frames.bin`, located by `frames.idx`, described by `manifest.json`. A frame
   is one positional read or HTTP Range request, then one decompress. Any failed check is a
   refusal naming the chromosome and the frame.
6. **Format version 0.2.0** (D5).
7. **The five existing stores are derived, not rebuilt** (D6). One row emitter serves the build
   and `derive-partners`, and both produce the same bytes.
8. **The Near Band, the Far Pairs and their Zarr layout are unchanged** and serve only `matrix`
   and `region`. Measured on EUR chr22, the current 512 KiB chunks suit contiguous regions best:
   a 3 Mb region decodes in 0.23 s single-threaded, about 2× the values it needs. Scattered
   variants cost more but stay acceptable for one-off calls: 1,000 across 10 Mb decode in 0.35 s.

## Consequences

- `clump` and `proxies` do one frame read per lead or query variant, with no speculative reads.
  The clump warmer, the Near Band partner and column reads, and clump's use of the chunk cache,
  single-flight and admission are removed. Epic #45's scheduler and codec work then concerns
  only `matrix` and `region`.
- Clumping's unit of parallel work is the component: hits with no gap of W or more between
  neighbours. Components are independent, so results do not depend on concurrency.
- The store directory is no longer only a Zarr hierarchy. `partners/` has no `zarr.json`, and
  zarr does not see it.
- ALID resolution (`rows_of`, 3.18 s on GCST90476460) becomes most of a clump's time. It is
  #45's mmap ALID work (#47, #471), not this decision.
- The spec's §11 sizes are projections from chr22 with a simpler encoding. The epic measures
  the real bytes, and a result that undermines a decision goes back to the owner.
