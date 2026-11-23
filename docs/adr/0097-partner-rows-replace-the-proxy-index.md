# Partner Rows replace the Proxy Index for clumping and proxies

Formerly ogld ADR 0011.

Part of epic [#64](https://github.com/opengwas/ogld/issues/64). The design is `docs/spec/partner-rows.md`. It supersedes, for
`clump` and `proxies`, decision 1 of epic [#45](https://github.com/opengwas/ogld/issues/45)
("keep the upper triangle and Zarr v3; no symmetric duplication and no independent row-frame
format"). The terms are in `CONTEXT.md`.

> **Decision 5's "plain files, not Zarr" is refined by the clarification of 18 November 2026**
> (see "Clarification, 18 November 2026" at the end). `partners/<chrom>/` is a Zarr group in the
> store's one hierarchy, not a directory of plain files, because the plain files make zarr's
> `members()` warn. Everything else in decision 5 stands: G rows to a zstd frame, one positional
> read or Range request per frame, and a failed check a refusal naming the chromosome and the
> frame. The consequence below that `partners/` has no Zarr metadata document and zarr does not see it is
> superseded, and so is the one that §11's sizes are projections: the clarification records the
> measured ones.

**Status: implemented.** The component, the refusals and the format version landed in opengwas/ogld#67–opengwas/ogld#71,
and opengwas/ogld#72 removed the Proxy Index: `index/`, its writer, its arrays, its six validation rules and
`far.index`, with `clump`'s Near Band reads, `StoreReader.clump_warmer` and `partners`'s row,
column and `among` reads. A 0.2.0 store holds no `index/`; one that still carries a leftover is
read and validated exactly as one without it (ld-store §12). Pending: opengwas/ogld#73, which derives the Partner
Rows for the five published stores and drops each `index/`, and opengwas/ogld#81, the follow-up to the LD
work's timing (below).

**Who decided.** The owner, 18 November 2026, on the measurements of 17–18 November 2026
(`proto/partner-rows`), taking decisions D1–D6 of `docs/spec/partner-rows.md` §1; refined the same
day by the clarification at the end, on opengwas/ogld#65's measurements.

## Context

Clumping reads one lead's LD with the hits still open in its window. The Near Band stores that
LD in a row-major upper triangle, in 512 KiB chunks. A lead's upstream partners are a column, one
element in each of thousands of earlier rows. On GCST90476460 (260,821 hits, 3,869 leads, r²
0.01) the reader read 10.7 GB and decoded 22.8 GB, and took 23–26 s cold. Epic opengwas/ogld#45's codec and
scheduler work reduces the time per decoded byte, but not the 22.8 GB.

Two measurements on the published HGDP+1kGP k = 10 stores changed the trade:

- Only 5.0% of EUR chr22's Near Band pairs have r̂² > 0.01, 4.1–5.0% on AFR, EAS and SAS, and
  16.3% on AMR (n = 341). A variant's two-sided row at that floor has a median of 800 entries,
  against 19,971 dense.
- In both directions and compressed, rows at r̂² > 0.01 project to 75 GB over the five panels,
  against 533.7 GB of stores today. Dense symmetric rows, rejected by opengwas/ogld#45, project to about 181
  GB for EUR alone.

At r² 0.001 there is no such sparsity. 46.2% of EUR chr22's pairs pass, against 41.9% from
sampling alone, so about 90% of the passing pairs are noise.

## Decision

1. **A new component, the Partner Rows** (`partners/<chrom>/`). For each variant it holds every
   partner within `far_bp`, on both sides, at |code| ≥ the Partner Floor within `near_bp` and ≥
   the far code floor beyond it. Every code is a copy of a Near Band or Far Pairs code. `clump`
   and `proxies` read only this component.
2. **It replaces the Proxy Index** (D1). Since t_partner ≤ t_far, the index is the Partner Rows
   thresholded at the far code floor. `index/` and its validation rules are removed. Net an
   estimated **+33 GB (+6%)** over the five panels: that number is this decision's projection from
   chr22, and the clarification below measures +53 GB net at G = 64, EUR's own being +5.38 GB at
   G = 16.
3. **Below t_partner within `near_bp`, `clump` and `proxies` refuse** (D2), naming
   `threshold_below_t_partner`. They never read the Near Band. This is a breaking change: r²
   0.001 clumps on the HGDP+1kGP stores, answered today, are refused.
4. **t_partner is set per reference dataset and recorded in provenance** (D3), as t_far is. It
   is a required build input with no default, and never above t_far: 0.01 for HGDP+1kGP and
   0.001 for UK Biobank EUR.
5. **Plain files, not Zarr** (D4; refined by the clarification at the end: a Zarr group holding
   one chunk). G rows to a zstd frame, with checksum and content size,
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
  single-flight and admission are removed. Epic opengwas/ogld#45's scheduler and codec work then concerns
  only `matrix` and `region`.
- Clumping's unit of parallel work is the component: hits with no gap of W or more between
  neighbours. Components are independent, so results do not depend on concurrency.
- The store directory is no longer only a Zarr hierarchy. `partners/` has no Zarr metadata document, and
  zarr does not see it.
- ALID resolution (`rows_of`, 3.18 s on GCST90476460) becomes most of a clump's time. It is
  opengwas/ogld#45's mmap ALID work (opengwas/ogld#47, #471), not this decision.
- The spec's §11 sizes are projections from chr22 with a simpler encoding. The epic measures
  the real bytes, and a result that undermines a decision goes back to the owner. On 18 November
  2026 they were measured for EUR, and one decision was refined: see below.

## Clarification, 18 November 2026

The owner refined decision 5, and opengwas/ogld#65's measurements fixed the size and the two numbers the
decisions left open (`proto/partner-rows@912a9fa:prototypes/partner-rows/`, recorded in
`docs/spec/partner-rows.md` §11). They refine the decisions above; none is reversed.

**The physical form is a Zarr group, not plain files** (owner, refining D4). Each
`partners/<chrom>/` is a Zarr v3 group in the store's one hierarchy:

- `frames` is a `uint8` array of shape (B,), one chunk of shape (B,), with the `bytes` codec and
  **no compressor**. Its chunk file is the raw concatenated zstd frames, so a lead is still one
  positional read or Range request. A reader resolves that file's path from the array's metadata
  and never assumes one.
- `frames_idx` is a `uint64` array of the F + 1 byte offsets, little-endian, one chunk.
- The manifest's keys are the group's attributes.

The reason is the compatibility check, not a measurement of speed. `zarr.open_group` and
`opengwasdb.ld.open` on a store with a plain-file `partners/` are clean, and `ogld.validate` passes all 27
rules, none of which looks at the new component. But zarr-python 3.4.2's `members()` emits
`ZarrUserWarning: Object at partners is not recognized as a component of a Zarr hierarchy.` A node
in the hierarchy removes the warning. The single-chunk form was checked only by a small synthetic
test so far; opengwas/ogld#67 measures it on a real store. Decision 5's one-read rule survives because one chunk
has no shard index to resolve first.

**The measured size is larger than the projection, and the owner accepted it.** EUR's Partner Rows
are **14.765 GB at G = 16** and **11.821 GB at G = 64**, against decision 5's and §11's 9.0 GB
projection, because the genome keeps **7.27%** of Near Band pairs where chr22 keeps 5.03%: the
projected chromosome was unrepresentative. Net of EUR's 9.382 GB Proxy Index the store grows by
**5.383 GB at G = 16** and 2.439 GB at G = 64, and the owner's same scaling puts the five panels
at roughly **+53 GB net at G = 64**. Decisions 1, 2 and 6 stand; §11's table is the measurements
now, and opengwas/ogld#73 measures the other four panels.

**G = 16, and component workers capped at 4** (measured on opengwas/ogld#65; `docs/spec/partner-rows.md` §1,
§5.2, §9.2). EUR genome bytes are 18.506 GB at G = 4, 16.677 GB at G = 8, 14.765 GB at G = 16 and
11.821 GB at G = 64. With the row-local unshuffle, cold LD on the whole of GCST90476460 is 1.169 s
at G = 4 with two workers and 1.051 / 1.067 s at G = 8 / 16 with four, and 5.753 / 5.109 s with
eight at G = 8 / 16: eight workers regress. G = 16 is the smallest of the configurations that are
equally fast. **The under-1 s LD target moves to opengwas/ogld#71** (owner, 18 November 2026): the best cold time
is 1.051 s, so nothing here claims it.

**The target was not met, and the owner accepted it** (opengwas/ogld#71's owner decision, recorded on opengwas/ogld#72). A
quiet re-time at `cc48a58` on GCST90476460 gave cold LD medians of **2.30 / 1.55 / 2.95 s** at
1 / 2 / 4 workers, and warm medians of **0.98 / 0.97 / 1.35 s** (EUR k = 10). The follow-up is
**opengwas/ogld#81**, decode and gather plus ALID resolution; it owns the worker default too, which this
decision leaves at four. `docs/spec/partner-rows.md` §9 and `docs/spec/ld-store.md` §10 carry the
figures in place of the target.
