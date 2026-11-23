# The Far Pairs are their own component again, beside the Proxy Index

Formerly ogld ADR 0007.

> **Status: superseded in part by [ADR 0097](./0097-partner-rows-replace-the-proxy-index.md).**
> The Proxy Index this decision puts the Far Pairs beside is gone (#72): the Partner Rows hold
> every entry it held, and `clump` and `proxies` read them. The Far Pairs themselves, their rules
> and their shape stand exactly as decided here; what no longer holds is decision 3's division of
> reads (`partners` reading the index at r2 ≥ t_far), decision 5's `far.index`, which went with
> the index, and every sentence below that speaks of the index in the present tense. This record
> is kept as written, as the measurement of 12 November 2026 that sent the reads to the Far Pairs.

Issue [#31](https://github.com/opengwas/ogld/issues/31), part of epic [#1](https://github.com/opengwas/ogld/issues/1).
It supersedes the merge of the Far Pairs into the Proxy Index that the owner made on
8 November 2026, recorded in [ADR 0087](./0087-near-band-and-thresholded-far-pairs.md)'s
clarification and ld-store §11. The format is in `docs/spec/ld-store.md` (§1, §5–§8, §10–§13); the
owners are in `docs/architecture/ownership.md`; the terms are in `CONTEXT.md`.

**Who decided.** The owner, on 12 November 2026, on ogld#8's measurements (ogld#8, "Owner's
rulings", ruling 2): the Far Pairs come back before the first release, reversing their merge into
the Proxy Index. The ruling of the same day on ogld#31 fixes the shape: a Far Pairs array per
chromosome holding every pair 3–10 Mb apart with r̂² ≥ t_far, once, as (row, partner, code),
sorted for range reads; `proxies` and `clump` keep reading the Proxy Index, and `matrix`'s and
`region`'s pairs beyond 3 Mb read the Far Pairs; the store derives them from the pairs it
already keeps, with no producer change (ADR 0091); validation regains the five rules ld-store §11
named.

## Context

On 8 November 2026 the owner merged the Far Pairs into the Proxy Index. The index holds every
pair within `far_bp` at r̂² ≥ t_far, in both rows, so it held every Far Pair twice, and a separate
component held nothing it did not. ld-store §11 weighed what the merge cost reads and found it small:

- a matrix for a list of ALIDs reads one row per variant either way, and "a row of either
  component sits in one inner chunk, so the requests are the same";
- a region wider than `near_bp` would read 3–9% more bytes.

ogld#8 timed reads of the published HGDP+1kGP EUR store (× 511, k = 10, layout C) from local
disk and over HTTP with 0, 10 and 50 ms added per request (`docs/benchmarks/latency.md`,
12 November 2026). The first estimate held; the second did not:

- **A 10 Mb region** read 6.2–9.3% more bytes for its index rows, as estimated. But it walked
  every entry of every row, 11.4–16.6 M of them, to place the 6,228–11,651 Far Pairs inside the
  region: 99.6% of the entries are copies of Near Band pairs, which it drops. That took 13–30%
  of the read's local time.
- **A matrix of 500 variants on one chromosome** read about 890 index chunks for its pairs
  beyond 3 Mb. An EUR chr22 index row holds about 380 entries, so a 16,384-entry chunk holds
  about 43 rows, and 500 rows of 119,632 rarely share one: they touched 433–448 chunks of each
  of `index/22/partner` and `index/22/r`. chr22's 101,886 Far Pairs fill 7 chunks of each array,
  and the same rows touch 5–6 of them: about 11. The far part was 32% of the read locally and
  58–60% at 10 and 50 ms per request, so the merge more than doubled such a matrix over a
  network.

`proxies` and `clump` were unaffected: at r2 ≥ t_far they read one index row a variant, which
is what they need.

## Decision

1. **The format.** Each chromosome has a `far/<chrom>/` group beside `near/<chrom>/` and
   `index/<chrom>/`: `indptr` (int64, n_c + 1), `partner` (int32) and `r` (int16), shaped and
   filled as the index's arrays are (ld-store §6). Row l holds every m > l with
   `near_bp` < p_m − p_l ≤ `far_bp` and |code| ≥ `far_code_floor`, ascending in m: each Far Pair
   once, in its earlier variant's row, as the Near Band holds its pairs. They are exactly the
   index's entries beyond `near_bp` in their earlier rows, with the same codes: a copy, by
   design, as the index's near half copies the band.
2. **The store writes it** (ADR 0091). A cut already holds the block's pairs at the floor,
   ascending, and which of them lie beyond `near_bp`; those are the block's Far Pairs, and a
   row's are all in its own block. So the assembly writes them row by row with nothing carried
   and no second pass over r, a whole shard at a time as it writes the index. No producer
   changes, and `counts.far_pairs` is their number, as it was.
3. **Who reads what.** Every pair answer beyond `near_bp` reads the Far Pairs: `pair_r` and
   `region_r` scatter the Far Pairs rows of their variants into the matrix, and `r_of_pairs`
   searches the Far Pairs row of a pair's earlier variant. No read of a matrix or a region
   touches an index entry. `partners`, which `proxies` and `clump` call, keeps reading the
   index at r2 ≥ t_far. No public signature changes.
4. **The reader opens the Far Pairs when a request first needs them.** A chromosome's
   positions and its Near Band's and index's `indptr` are read when a request first touches it,
   as before; its Far Pairs' `indptr` is read, and its three arrays' fill values judged, the
   first time a request needs its pairs beyond `near_bp`. So a chromosome that `proxies` or
   `clump` touches costs no more requests or memory than it did. A store without `far/` is
   refused by the first read that needs it, naming the missing array.
5. **Validation regains five rules** (ld-store §12): `far.rows`, `far.codes`, `far.distance` and
   `far.threshold`, the four ld-store §11 named for the Far Pairs before the merge, and
   `far.index`, which holds the Far Pairs equal to the index's entries beyond `near_bp`, pair
   for pair and code for code. With `index.symmetric` that makes them the index's far pairs,
   once each.
6. **The Far Pairs keep the index's array shape:** 16,384 entries an inner chunk, 64 chunks to
   a shard (ld-store §6). They were chosen on the chromosome where they are densest as well as on
   chr22, because they lie very unevenly: EUR chr22 holds 0.85 Far Pairs a variant, chr11 161.5,
   nearly all in rows at 45–55 Mb. For a matrix of 500 variants drawn uniformly on one
   chromosome (seed 8), at 16,384 the Far Pairs touch 5 chunks of each array in 1 run on chr22
   and 51 in 28 on chr11, against the index's 447 in 339 and 529 in 463. A chunk of 4,096 adds
   runs on chr11 (37); one of 65,536 or 262,144 removes few (26, 24) and decodes 2.6 and 8 times
   the bytes. A list clustered where the Far Pairs are densest is the other end: 500 rows at
   chr11's 45–55 Mb touch 440 chunks in 313 runs at 16,384, 198 runs at 65,536 for 3 times the
   bytes, and 48 at 262,144 for 7 times (41, 123 and 294 MiB). The shape is kept for the
   uniform lists, which a larger chunk does not help; ogld#32 fetches a read's runs together,
   which takes their count off the critical path, and a chunk shape can change without a new
   format version (ld-store §13) if served lists turn out clustered. A larger shard removes no run
   and enlarges the shard a writer holds.
7. **`format_version` stays `0.1.0`.** It is an unreleased draft (ld-store §13): no LD Store in it
   has been published, and the owner ruled that the Far Pairs come back before the first
   release.

## Considered options

- **Keep the merge.** Rejected by the owner on ogld#8's numbers: a list matrix over a network
  takes more than twice as long, and a region spends 13–30% of its time discarding index
  entries. Bringing the Far Pairs back costs what §11 listed: 0.03–0.76% more bytes per store, one
  more output of the same scan, and five rules.
- **Read the far half of each index row in place,** with a per-row offset to where its entries
  beyond `near_bp` start. Rejected: it removes the walk over a region's entries, but a list
  matrix still touches one index chunk per listed row, about 890 on EUR chr22. The chunks are
  what cost, and only a component whose rows are a few pairs long packs 500 rows into a few.
- **A Far Pairs component in place of the index's far entries**, the index then holding only
  pairs within `near_bp`. Rejected: `proxies` and `clump` at r2 ≥ t_far would read a variant's
  partners beyond `near_bp` from a column again, one element in each earlier Far Pairs row
  within 10 Mb, which is the scattered read the index exists to avoid (ld-store §11).
- **A smaller inner chunk for the index.** Rejected: 500 rows would still touch about 890
  chunks, each smaller, and a request each over HTTP; and `proxies` would read more chunks per
  row.
- **Rebuilding the five HGDP+1kGP stores** with `scripts/ld/build_hgdp1kgp.py`, about 3.5 hours of
  two 16-thread builds, after deleting them. Not done: their Far Pairs were derived in place
  from each store's own index (below), which writes only new files, so the stores stayed
  readable throughout by the work reading them then (ogld#32, a review of ogld#8). The ticket
  allowed it on one condition, met first on chr22: a derived store byte-identical to a rebuilt
  one.

## Consequences

- **A pair beyond `near_bp` is held three times:** once in the Far Pairs and twice in the index,
  as a pair within it is held once in the band and, at the floor, twice in the index.
  `far.index` and `index.symmetric` hold them to one code per pair; `index.near` holds the band
  and the index to one. A store that breaks any of them is refused before its provenance is
  written (ld-store §12).
- **Validation now has 27 rules.** The five `far.*` rules join the pass that reads the band and the
  index one block of rows at a time, so each Far Pair is decoded once per validation, and
  `far.index` compares a block's Far Pairs with that block's index entries, carrying nothing. ADR
  0096 later added the four ALID-index rules.
- **A reader that has read matrices on every chromosome** keeps a third `indptr`, 8 bytes a
  variant: 69 MB for HGDP+1kGP EUR.
- **The Test Store holds 7 Far Pairs** (4 on chr21, 3 on chr22), 7.7 KB in `far/`. Every other
  array of it is byte-identical to the store written before this decision, and its provenance
  names the commit whose code wrote it, `far/` included.
- **What the reads gained, measured** (`docs/benchmarks/far-pairs.md`), on the published EUR
  store, before and after reading the same bytes. A matrix of 500 variants on one chromosome
  reads 10 Far Pairs chunks on chr22 and 102 on chr11, where it read 894 and 1,058 index chunks,
  and at 50 ms a request takes 53 s and 63 s where it took 129 s and 161 s. chr22 is the best
  case (0.85 Far Pairs a variant); chr11 the densest (161.5). Those are single uniform draws: on
  uniform draws of 500 rows, ogld#8 counted 10–91 times fewer chunks (10–29 times on the eight
  chromosomes with more than 10 Far Pairs a variant, 88–91 on chr21 and chr22); on a list
  clustered where the Far Pairs are densest, about 2–3 times (500 rows at chr11's 45–55 Mb: 880
  against 1,496 chunks, 1.7 times; the spec review's list, 300 of its 500 rows at 44–56 Mb: 438
  against 1,248, 2.8 times). A 10 Mb region's far part fell from 1.7–5.1 s to 12–491 ms where Far Pairs are
  sparse; at chr11's 45–55 Mb its rows' 68.75 M Far Pairs are 203 MB read against the index
  rows' 709 MB, and the region took 12 s locally where it took 30 s.
- **The five HGDP+1kGP stores gained their Far Pairs in place** (12–13 November 2026), with
  `scripts/ld/derive_far_pairs.py`: 628,713,731 in all, 1.46 GB, 0.27% of the 532 GB (EUR 0.32 GB,
  AFR 0.49, EAS 0.19, SAS 0.29, AMR 0.16; raw, at 6 bytes a pair, 0.53–1.15 GB, EUR's 0.80 GB
  about 1% of its store). Each store's count equals the `counts.far_pairs` it recorded when built,
  and none of its other files changed. The derivation writes them from each store's own index
  through the stream a build writes them through. First, on EUR chr22 built twice from the
  published store's inputs, a store derived in place from the tip's build was byte-identical to
  this decision's build in all 136 files but the root document, 8 of them in `far/`, and their
  provenance differed only in `build`. So the five carry what a rebuild would write; their
  `software` names the ogld that built them before the Far Pairs, which wrote everything else.
