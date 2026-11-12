# The Far Pairs are their own component again, beside the Proxy Index

Issue [#31](https://github.com/opengwas/ogld/issues/31), part of epic [#1](https://github.com/opengwas/ogld/issues/1).
It supersedes the merge of the Far Pairs into the Proxy Index that the owner made on
8 November 2026, recorded in [ADR 0001](./0001-near-band-and-thresholded-far-pairs.md)'s
clarification and spec §11. The format is in `docs/spec/ld-store.md` (§1, §5–§8, §10–§13); the
owners are in `docs/architecture/ownership.md`; the terms are in `CONTEXT.md`.

**Who decided.** The owner, on 12 November 2026, on ogld#8's measurements (ogld#8, "Owner's
rulings", ruling 2): the Far Pairs come back before the first release, reversing their merge into
the Proxy Index. The ruling of the same day on ogld#31 fixes the shape: a Far Pairs array per
chromosome holding every pair 3–10 Mb apart with r̂² ≥ t_far, once, as (row, partner, code),
sorted for range reads; `proxies` and `clump` keep reading the Proxy Index, and `matrix`'s and
`region`'s pairs beyond 3 Mb read the Far Pairs; the store derives them from the pairs it
already keeps, with no producer change (ADR 0005); validation regains the five rules spec §11
named.

## Context

On 8 November 2026 the owner merged the Far Pairs into the Proxy Index. The index holds every
pair within `far_bp` at r̂² ≥ t_far, in both rows, so it held every Far Pair twice, and a separate
component held nothing it did not. Spec §11 weighed what the merge cost reads and found it small:

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
   filled as the index's arrays are (spec §6). Row l holds every m > l with
   `near_bp` < p_m − p_l ≤ `far_bp` and |code| ≥ `far_code_floor`, ascending in m: each Far Pair
   once, in its earlier variant's row, as the Near Band holds its pairs. They are exactly the
   index's entries beyond `near_bp` in their earlier rows, with the same codes: a copy, by
   design, as the index's near half copies the band.
2. **The store writes it** (ADR 0005). A cut already holds the block's pairs at the floor,
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
5. **Validation regains five rules** (spec §12): `far.rows`, `far.codes`, `far.distance` and
   `far.threshold`, the four spec §11 named for the Far Pairs before the merge, and
   `far.index`, which holds the Far Pairs equal to the index's entries beyond `near_bp`, pair
   for pair and code for code. With `index.symmetric` that makes them the index's far pairs,
   once each.
6. **`format_version` stays `0.1.0`.** It is an unreleased draft (spec §13): no LD Store in it
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
  within 10 Mb, which is the scattered read the index exists to avoid (spec §11).
- **A smaller inner chunk for the index.** Rejected: 500 rows would still touch about 890
  chunks, each smaller, and a request each over HTTP; and `proxies` would read more chunks per
  row.

## Consequences

- **A pair beyond `near_bp` is held three times:** once in the Far Pairs and twice in the index,
  as a pair within it is held once in the band and, at the floor, twice in the index.
  `far.index` and `index.symmetric` hold them to one code per pair; `index.near` holds the band
  and the index to one. A store that breaks any of them is refused before its provenance is
  written (spec §12).
- **Validation has 23 rules.** The five `far.*` rules join the pass that reads the band and the
  index one block of rows at a time, so each Far Pair is decoded once per validation, and
  `far.index` compares a block's Far Pairs with that block's index entries, carrying nothing.
- **A reader that has read matrices on every chromosome** keeps a third `indptr`, 8 bytes a
  variant: 69 MB for HGDP+1kGP EUR.
- **The Test Store holds 7 Far Pairs** (4 on chr21, 3 on chr22), 7.7 KB in `far/`. Every other
  array of it is byte-identical to the store written before this decision.
