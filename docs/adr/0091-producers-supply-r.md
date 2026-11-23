# A producer supplies r; the LD Store makes the rest of itself

Formerly ogld ADR 0005.

> **The Proxy Index is gone** ([ADR 0097](./0097-partner-rows-replace-the-proxy-index.md), #72).
> Where this record speaks of it in the present tense — as a component a store holds, a stream a
> build writes or a read a query makes — read it in the past tense: the Partner Rows hold every
> entry it held, and `clump` and `proxies` read them. Everything else here stands.

> **ADR 0099 supersedes §5's builder unit vectors** ([ADR 0099](./0099-shared-genotype-arithmetic.md), #90).
> The builder computes r through `opengwasdb.ld.genotypes.r(rows, span)` over prepared `Block`s, not from
> an array of unit vectors; §5's memory bound and the rest of the record stand.

Issue [#17](https://github.com/opengwas/ogld/issues/17), part of epic [#1](https://github.com/opengwas/ogld/issues/1).
The format is in `docs/spec/ld-store.md` (§3–§7, §12); the owners are in
`docs/architecture/ownership.md`, "Provenance and what goes in a store"; the terms are in `CONTEXT.md`.

**Who decided.** The owner. On 10 November 2026, from the architecture review of
`orch/1-ld-stores@bd40f52` (candidate B2), they decided that what goes in an LD Store belongs to
`opengwasdb.ld.store`, behind the writer. On 11 November 2026, on ogld#17, they confirmed the signatures
proposed in `orch/1/17-store-assembly@b54ab09`, and ruled on three points:
- `software`, `reference_assembly` and `variants_dropped` are the producer's;
- the builder's unit vectors are windowed too;
- the Proxy Index streams as the Near Band does.

On 11 November 2026 they also ruled that #17's memory criterion is met for what it owns. The
PC Check pass moves to ogld#25.

## Context

Every producer implemented what goes in a store, and the writer only checked it.
- **Three producers each did the store's work.** The builder, `scripts/make_test_store.py` and
  `tests/test_walking_path.py` each quantised, thresholded at the far code floor, mirrored every
  pair into both rows of the Proxy Index, counted the pairs, and computed provenance a store
  derives from itself.
- **The writer kept whole chromosomes.** It held a chromosome's Near Band, and copied it, before
  writing. On chr6 of the whole EUR panel that is 2 × 13.65 GB.
- **The builder kept whole chromosomes too.** It held a chromosome's unit vectors, and laid out
  its Proxy Index whole: 13.9 GB on chr6.

A later producer, an import of a published LD matrix such as Pan-UKB's, would have needed a
fourth copy of each rule. And neither the writer nor the builder survives UK Biobank's n.

## Decision

1. **A producer supplies r, for blocks of rows in order, and what only its source knows.** The
   store does the rest:
   - quantising at the store's encoding (ADR 0090);
   - the Near Band's row lengths;
   - the threshold at the far code floor;
   - both directions of the Proxy Index;
   - the counts;
   - the provenance a store derives from itself;
   - validation.

   `opengwasdb/ld/store/assembly.py` holds all of it, behind `StoreWriter`.
2. **The seam is four calls.**
   - `StoreWriter.create(location, *, variants, encoding, near_bp, far_bp, n, k)` writes the
     Variant Set. It fixes the encoding, the distances, and t_far(n, k) with its floor, before
     any row: the store thresholds as rows arrive.
   - `cut(first, r) -> Cut` is pure and thread-safe, so a producer calls it on its own workers.
     `r` is a dense float rectangle for a block of rows of one chromosome, from the block's
     first row to the last variant within `far_bp` of its last row. Only each row's later
     partners within `far_bp` are read.
   - `write(cut)` takes cuts in global row order.
   - `finish(sourced, *, threads=1) -> Provenance` adds what the store derives. It validates
     the store, and only then writes the provenance.

   The writer exposes its `encoding` and `far_bp` read-only, so a producer gives each once.
3. **Provenance is split in two.** The producer gives `opengwasdb.ld.format.Sourced`: `source`, `panel`,
   `reference_assembly`, `variant_filters`, `variants_dropped`, `adjustment`, `pc_diagnostic`,
   `software` and `build`. The store derives the rest: `format_version`, `encoding`,
   `n_variants`, `near_bp`, `far_bp`, `t_far`, `t_far_q`, `far_code_floor` and `counts`. It
   refuses:
   - a `Sourced` whose `panel.n` or `adjustment.k` is not what it thresholded with;
   - a key it derives itself;
   - a key `Sourced` does not name.
4. **A pair the producer did not measure is never a Bound.** `cut` judges every pair within each
   row's `far_bp`. It refuses NaN, ±inf, or an r past ±1 beyond §3's float allowance, naming
   the pair; it never stores such a value, and never takes it for one below the floor.
5. **Memory is bounded by the block and the `far_bp` window, never the chromosome.**
   - **The Near Band** is created at a chromosome's first block, its length known from the
     positions. It is written a whole shard at a time (the only write zarr 3.4.2 makes without
     reading a shard back), with less than one shard carried.
   - **The Proxy Index** is written row by row. A row is complete when its own block arrives,
     and only the pairs whose later row is still to come are carried.
   - **The builder's unit vectors** are a window, held in one array, so every tile's product is
     the BLAS call it was on the whole chromosome.
   - **Measured** on EUR at × 511, k = 10 and 8 threads, writing peaks at 3.2 GB on chr22 and
     4.8–5.1 GB on chr6, against 6.7 GB and 31.4 GB before. The arrays are byte-identical.

## Considered options

- **Producers supply codes**, as the old writer took them. Rejected: every producer would still
  quantise, and a scale would live in each.
- **Ragged rows**: each row's later partners only, back to back. Rejected: every producer would
  have to slice the band's geometry itself, which is the copy this decision removes. A rectangle
  is what a matrix product, and Pan-UKB's block matrix, give.
- **Pairs as (i, j, r).** Rejected: the store could not tell a pair the producer skipped from one
  below the floor, so a missing pair would read as a Bound.
- **One call that cuts and writes.** Rejected: quantising and thresholding are 40% of a tile's
  time at n = 665 (146 of 369 ms), so doing them on the writer's thread would undo the builder's
  workers.
- **A thread pool inside the writer.** Rejected: it hides a second ordering and memory bound in
  the writer, and the builder needs `r` on its own worker anyway, for the PC Diagnostic.
- **Holding the Proxy Index per chromosome** and mirroring it at the end. Rejected: 13.9 GB on
  chr6, and it grows with the chromosome.
- **The producer passes t_far.** Rejected: t_far is store-derived. The store takes its n and k
  and refuses a `Sourced` that disagrees.
- **The whole `Sourced` at `create`.** Rejected: the build's cost and the PC Diagnostic's change
  in r are known only after the rows are written.
- **The store records `software` itself** (ogld, numpy, zarr). Rejected by the owner: the
  producer knows what computed r, such as pgenlib for a build.
- **A window of separate blocks, multiplied piece by piece.** Rejected: BLAS would see shapes it
  never saw on the whole chromosome, where a small-matrix kernel can round a different ulp. One
  array keeps every product the same call.

## Consequences

- **Every producer writes through these four calls.** That covers the builder, the Test Store's
  script, which is "import a precomputed r" in miniature, and a later Pan-UKB import. An import
  creates the writer with the `far_bp` its lift to GRCh38 leaves whole, its N, and its source's
  count of covariates as k (15 for Pan-UKB). It finishes with `source.kind = "ld_matrix"`.
- **A store written in any blocks, on any number of threads, holds the same arrays.**
  `tests/test_assembly.py` holds every array to a brute force over pairs, written in blocks of 1
  to 100,000 rows on 1 to 4 threads.
- **The PC Check pass was not bounded when this was decided.** It held a chromosome's genotypes
  before tiling: 6.1 GB on chr6 at k = 10, growing with rows × n, and the 10.4 GB peak
  before the tiles also held `require_change_pairs`' list of a chromosome's 100 kb pairs, 9.4 GB.
  ogld#25 bounded both (build-inputs §8): the peak before the tiles is 2.0 GB on chr6. The
  scan now holds `threads` windows' thinned genotypes, about `threads` × 5,000 × n × 25 B: it no
  longer grows with the chromosome's rows, but it grows linearly with n, at about 25 GB on 4
  threads and 50 GB on 8 at UK Biobank's n = 49,766. The figures above this bullet are as measured on
  11 November 2026.
- **`StoreWriter`'s methods changed**, and `write_near` and `write_index` are gone. They are a seam
  inside the package, unpinned since 10 November 2026; no stored byte changes.
