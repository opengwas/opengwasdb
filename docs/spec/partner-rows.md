# Partner Rows: a clump-and-proxy component for the LD Store

Status: **accepted design**, owner's decisions of 18 November 2026 (§1; ADR 0011). It becomes
normative as `docs/spec/ld-store.md` takes the sections it amends, in the epic that builds it.
Format version described: `0.2.0` (decision D5).

This document specifies a new LD Store component, the **Partner Rows**. For each variant it
holds every partner whose Stored r reaches a floor, on both sides, stored as one small compressed
frame that a reader fetches with one positional read and decompresses without Zarr. `clump`
and `proxies` read only this component. The Near Band and Far Pairs stay as they are, and serve
only `matrix` and `region`.

It specifies the format (§3–§5), how the component is built (§6), how it is read and queried
(§7–§8), how both are parallelised (§9), how it is validated (§10), and what it costs (§11). The
evidence is on issues #40, #45 and #50 and in the measurements of 17–18 November 2026 (§11).

Normative language is `ld-store.md`'s: MUST, SHOULD, MAY.

## 0. Why

Clumping reads a small number of values: one lead's LD with the hits still open within its
window. The Near Band stores them in a row-major upper triangle, in fixed 512 KiB chunks
of about 26 rows each. A lead's downstream partners are part of one chunk. Its upstream
partners are a column, one element in each of thousands of earlier rows. On GCST90476460
(260,821 hits, 3,869 leads, r² 0.01) the current reader read 10.7 GB and decoded 22.8 GB to
answer it, and took 23–26 s cold. plink2 needs the genotypes of the hits alone: about 45 MB
(2 bits × 665 samples × 260,821 hits, an estimate).

The problem is how much has to be read and decoded for each value used, not lookup against
compute. Two measured facts make a small, direct structure possible:

- On EUR chr22, **5.0% of Near Band pairs have r̂² > 0.01** (1.9% reach t_far). A lead's
  two-sided row at that floor has a median of 800 entries, against 19,971 in the dense band.
- One row read as one positional read, with no chunk, shard index or Zarr call, took the
  dense chr22 greedy from 0.227 s to 0.056 s on #50's packed-row prototype. The rows there
  were dense; these are 20× smaller.

Epic #45 decided to keep the triangle and reject symmetric rows, because dense symmetric rows
would have cost about 181 GB for the EUR store. Thresholding changes that trade: the Partner Rows
are symmetric rows at about 5% of the size. For `clump` and `proxies` this design supersedes #45's
decision 1. The triangle stays for `matrix` and `region`.

## 1. Decisions (owner, 18 November 2026)

| | decision | consequence |
|---|---|---|
| D1 | **The Partner Rows replace the Proxy Index.** | They hold every index entry (§3), so `proxies` reads them too. Net size about +33 GB (+6%) over the five panels, instead of +75 GB (+14%) beside the index |
| D2 | **`clump` and `proxies` with r2 below t_partner and W ≤ `near_bp` are refused**, `threshold_below_t_partner`. | On HGDP+1kGP at r² 0.001, 90% of the pairs that pass are sampling noise (§11). Today's answered r² 0.001 clumps on those stores become Refusals: a breaking change |
| D3 | **t_partner is set per reference dataset and recorded in the store's metadata**, as t_far is. | A build input with no default: each Source Panel's build names its own (0.01 for HGDP+1kGP, 0.001 = t_far for UK Biobank EUR), and a build without one refuses. It never exceeds t_far (§3) |
| D4 | **Plain files** inside the store directory (§5). | One frames file and one offsets file per chromosome. One read per frame, with no shard index |
| D5 | **Format version 0.2.0.** | A new component, a removed one and a changed refusal table |
| D6 | **Derive** the five existing stores' Partner Rows from their Near Band and Far Pairs (§6.2). | No genotypes needed. Byte-identical to a build |

§12 records the alternatives that were weighed and what they would have cost.

## 2. Terms

To be added to `CONTEXT.md` when accepted.

**Partner Rows**:
For each variant on a chromosome, every partner within `far_bp` on either side whose Stored r
reaches the row's floor, with its code: the **Partner Floor** within `near_bp`, the far code floor
beyond it. A pair appears in both variants' rows with the same code. `clump` and `proxies` read
them, and nothing else does. A pair within reach that is not in a row has a known bound, never a
0 (§4).
_Avoid_: sparse band, clump index, thresholded LD; "Proxy Index", which they replace.

**Partner Threshold** (t_partner):
The r̂² a pair within `near_bp` must reach to be in the Partner Rows. Recorded per store, never
above t_far. Set per reference dataset as a build input and recorded in provenance (D3): 0.01 for
the HGDP+1kGP stores, 0.001 for UK Biobank EUR.
_Avoid_: clumping threshold. A request's r2 is compared against it, and is a different number.

**Partner Floor** (`partner_code_floor`):
The smallest integer c ≥ 0 with (c / s)² ≥ t_partner in float64, at the store's scale s: the
same rule as `far_code_floor` (`ld-store.md` §3). 52 at × 511 for t_partner = 0.01.

**Frame**:
The unit of storage and of reading: the Partner Rows of G consecutive rows of one chromosome,
compressed as one zstd frame (§5).

## 3. What the Partner Rows hold

For chromosome c with n_c rows and positions p:

- Row l holds every m ≠ l with |p_m − p_l| ≤ `far_bp` and
  - |code(l, m)| ≥ `partner_code_floor` when |p_m − p_l| ≤ `near_bp`;
  - |code(l, m)| ≥ `far_code_floor` when |p_m − p_l| > `near_bp`: exactly the Far Pairs, in both
    directions.
- Entries within a row are in strictly ascending m. A row never holds l itself.
- **Both directions.** (l, m, c) is an entry exactly when (m, l, c) is.
- **Copies, by design.** Every code within `near_bp` is the Near Band's code for that pair, and
  every code beyond it is the Far Pairs' code. The component holds no value the store does not
  already hold. It is a second layout of the same codes, chosen for clump and proxy reads.
- **It contains the Proxy Index.** t_partner ≤ t_far, so `partner_code_floor` ≤
  `far_code_floor`. Every Proxy Index entry is therefore an entry here, with the same code: the
  Proxy Index is the Partner Rows thresholded at `far_code_floor`. This is what lets D1 retire
  the index.
- t_partner is a REQUIRED build input, with no default (D3). A build without one refuses.
- t_partner MUST be ≤ t_far. A build given a larger one refuses, naming both values. With a larger
  one, a pair between t_far and t_partner within `near_bp` would be in neither component, and a
  request at r2 ≥ t_far could not be answered.

## 4. Pair answers from the Partner Rows

The Partner Rows answer only threshold questions: does r̂² reach r2. They never answer a matrix
cell. For a pair (l, m) of present variants on one chromosome, at distance d, with the pair not
found in row l:

| d | what the absence means |
|---|---|
| d ≤ `near_bp` | r̂² < t_partner: the Near Band holds the code and it is below the Partner Floor |
| `near_bp` < d ≤ `far_bp` | the Bound, r̂² < t_far, as today |
| d > `far_bp` | Not Measured, as today |

A pair is never answered as 0. A reader MUST NOT turn an absent entry into a Stored r. It uses
the absence only to decide that a threshold at or above the matching bound is not reached.

## 5. Physical layout

### 5.1 Files

```text
<ld store>/
└── partners/<chrom>/
    ├── manifest.json      what the two files hold, and their checksums
    ├── frames.idx         uint64 little-endian, F + 1 byte offsets into frames.bin
    └── frames.bin         F zstd frames, back to back
```

These are plain files, not Zarr arrays (D4). Zarr v3 has no implicit groups, so a directory with
no `zarr.json` is not a node, and zarr ignores it. `ld-store.md` §1's envelope changes from "one
Zarr v3 hierarchy" to "one Zarr v3 hierarchy and its Partner Rows". `partners/` has no
`zarr.json`. The prototype MUST confirm that zarr-python opens and lists such a store without
error or warning.

### 5.2 Frames

- G rows to a frame, G recorded in the manifest. Frame f holds rows [f·G, min((f+1)·G, n_c)), so
  F = ⌈n_c / G⌉ and row l is in frame ⌊l / G⌋. **G defaults to 16**: a median frame is about
  77 KB raw and 15–20 KB compressed on EUR. The prototype measures G ∈ {1, 4, 16, 64} (§13)
  before the default is fixed. Readers MUST take G from the manifest, never assume it.
- Frame f's bytes are `frames.bin[frames.idx[f] : frames.idx[f + 1]]`. `frames.idx[0]` is 0 and
  `frames.idx[F]` is the size of `frames.bin`.
- Each frame is one zstd frame at level 3, with the **content checksum** and the **content size**
  set in its header. Its decompressed payload, all little-endian, is:

  | field | type | count |
  |---|---|---|
  | k, rows in this frame | uint32 | 1 |
  | lengths, entries per row | uint32 | k |
  | partner deltas, byte-shuffled | int32 | E = Σ lengths |
  | codes, byte-shuffled | int16 | E |

  - **Partner deltas.** For row l with partners m₀ < m₁ < …, the first delta is m₀ − l (negative
    when m₀ is upstream) and each later one is mᵢ − mᵢ₋₁ (≥ 1). Decoding is a cumulative sum,
    reset at each row start.
  - **Byte-shuffled** means the array's bytes are stored as planes: every element's byte 0, then
    every element's byte 1, and so on. Deltas are small, so their high planes are almost all 0 or
    0xFF and compress to almost nothing.
  - The payload's length MUST be exactly 4 + 4k + 6E. k MUST be the rows the frame is due.
- An empty row has length 0. A frame whose rows are all empty is still a frame, with E = 0.

### 5.3 Manifest

`partners/<chrom>/manifest.json`:

| key | type | meaning |
|---|---|---|
| `rows` | int | n_c, equal to the chromosome's rows in `variants/` |
| `rows_per_frame` | int | G |
| `frames` | int | F |
| `entries` | int | Σ E over all frames: twice the pairs held |
| `frames_bin_bytes` | int | the size of `frames.bin` |
| `frames_bin_sha256`, `frames_idx_sha256` | string | the two files' SHA-256 |
| `zstd_level` | int | 3 |

The root provenance (`ld-store.md` §7) gains `t_partner` (float), `partner_code_floor` (int),
`counts.partner_entries` (int, every chromosome's `entries`) and `software.zstd` (the library
version, because frame bytes depend on it). It loses `counts.index_entries` (D1).

### 5.4 Why this form

- **One lead, one read, one decode.** The reader holds `frames.idx` (8 bytes per G rows: 4.3 MB
  for EUR's 8.59 M variants at G = 16, against 69 MB for a Near Band `indptr`). It finds frame
  ⌊l / G⌋, reads that byte range and decompresses it. There is no shard index, no codec chain to
  recognise and no event loop. Local reads are `os.pread` or an mmap. HTTP reads are one Range
  request.
- **Corruption is loud.** A missing or short `frames.bin`, a frame whose checksum fails, or a
  payload of the wrong length is refused, naming the chromosome and the frame. It is never read
  as a short row. This is the store's rule that a truncated window is not a short answer.
- **Sizes vary with LD, so chunks cannot be fixed.** Rows hold a median of 800 and up to 4,665
  entries on EUR chr22,
  so a fixed-size chunk always cuts through rows. Grouping a fixed number of rows makes the
  address arithmetic trivial and lets the bytes vary.

## 6. Building

### 6.1 One row emitter, two producers

The Partner Rows are a function of the Near Band and the Far Pairs (§3). One emitter makes them
from codes arriving in row order, block of rows by block of rows, as the Proxy Index writer does
today (`ld-store.md` §6, "How a store is written"):

1. For each block, keep each pair (l, m > l) with |code| at the floor its distance calls for.
2. Each kept pair goes to row l immediately, and to a carry for row m.
3. Row l is complete when its own block has been processed. Its upstream entries are all in the
   carry, from earlier blocks or its own, and its downstream entries are in its own row. When
   rows f·G … f·G + G − 1 are all complete, frame f is assembled, compressed and appended.
4. The carry holds only pairs whose later row is still to come: those within `far_bp` of the
   rows being written. On EUR chr6, where today's index carry at t_far peaks at 33.9 M pairs, a
   carry at 0.01 holds an estimated 2.6× that, about 90 M pairs (under 1 GiB).

Its two producers:

- **The build** (`StoreWriter`): the emitter replaces the Proxy Index writer, fed by the same
  blocks. No new pass over genotypes.
- **Derivation** (§6.2): the same emitter fed by reading the Near Band and the Far Pairs back.

Both MUST produce identical bytes for the same store. The derivation of a store built with Partner
Rows reproduces its `frames.bin` exactly, and a test holds this on the Test Store.

### 6.2 Deriving the five existing stores (D6)

`ogld derive-partners <store> [--threads t] [--chromosomes …]`:

- Reads `near/<c>/r` with the direct decoder (ADR 0009) in blocks of rows, and `far/<c>/` whole
  (101,886 pairs on EUR chr22).
- Writes `partners/<c>/` into a temporary directory beside the store, then renames it into place
  once every frame has been written and checked. The provenance gains its new keys only after
  every chromosome is in place and §10's rules pass. A store with no provenance entry for the
  Partner Rows does not have them, whatever directories exist.
- Drops `index/` afterwards, as its own step and only on request (`--drop-index`), so a derived
  store can be checked against its index before the index is removed.

### 6.3 Parallelising the build

- **Across chromosomes: processes.** Chromosomes share nothing. A pool of P worker processes,
  each taking the largest remaining chromosome first, bounds memory at P chromosome carries plus
  their decode buffers. chr1, chr2 and chr6 dominate. Starting them first keeps the pool's
  makespan near the largest chromosome's time.
- **Within a chromosome: a pipeline of three stages.**
  1. Decoder threads decode the next Near Band blocks ahead. zstd releases the GIL.
  2. One thread thresholds, carries and assembles rows in row order.
  3. A compressor pool compresses frames. zstd releases the GIL. Frames are appended strictly in
     frame order, so the output does not depend on the pool's size or scheduling.
- **Estimated cost.** The measurement scripts decoded, thresholded and symmetrised chr22
  (1.17 × 10⁹ codes) single-threaded in about 1.5 min, with a whole-chromosome sort the
  streaming emitter does not need. The genome is about 70× chr22 by Near Band size: about 1.7 h
  serial, and an estimated 15–20 min per panel with 8 chromosome workers. The prototype measures
  this (§13).

## 7. Reading

`ogld.store.StoreReader` gains one read, replacing the Proxy Index read and the Near Band
partner reads:

```text
partner_row(chrom, l) -> (partners: int32[E_l], codes: int16[E_l])
```

1. On the chromosome's first use, read `manifest.json` and `frames.idx` whole, and check
   `frames.idx[F]` against `frames_bin_bytes`.
2. f = ⌊l / G⌋. If frame f is in the frame cache, use it. Otherwise read its byte range,
   decompress it (the zstd checksum is verified), check k and the payload length (§5.2),
   unshuffle and decode the deltas, and cache the decoded frame.
3. Return row l's slice of the decoded frame.

- **The frame cache** is a byte-bounded LRU of decoded frames, per reader. Neighbouring leads
  often share a frame, and repeated Analyses on one reader hit it. It is separate from the Near
  Band chunk cache, which keeps serving `matrix` and `region`.
- **Local stores** read with `os.pread` on a kept file descriptor, or slice an mmap. Both go
  through the OS page cache. A cold read is one random read of a frame, typically 4–64 KiB.
- **HTTP stores** read one Range request per frame. A request for several frames at once
  (§9.2, option E) coalesces only physically adjacent frames.

## 8. Queries

### 8.1 What each request reads

The refusal table of `ld-store.md` §9 becomes:

| request | answer |
|---|---|
| invalid r2, kb or score | refused: `invalid` (unchanged) |
| W > `far_bp` | refused: `window_over_far` (unchanged) |
| `near_bp` < W ≤ `far_bp` and r2 < t_far | refused: `threshold_below_t_far` (unchanged) |
| `near_bp` < W ≤ `far_bp` and r2 ≥ t_far | answered from the Partner Rows |
| W ≤ `near_bp` and r2 < t_partner | **refused: `threshold_below_t_partner`** (D2) |
| W ≤ `near_bp` and r2 ≥ t_partner | answered from the Partner Rows |

**Why every answered request is exact.** `clump` joins at r̂² > r2 and `proxies` keeps r̂² ≥ r2.
In both cases r2 ≥ t_partner within `near_bp` and r2 ≥ t_far beyond it, so any pair that could
reach r2 has |code| at or above that distance's floor, and is in the row. A pair not in the row
has r̂² below a bound that is ≤ r2, so it cannot reach r2 (§4). On every request both answer, the
answers are those of the current reader, element for element, which §13's acceptance criteria
hold.

`matrix` and `region` are unchanged and never read the Partner Rows.

### 8.2 `clump`

Semantics unchanged (`ld-store.md` §9): greedy, strongest score first, ties by genomic order then
ALID, a hit joins a lead at r̂² > r2 and d < W, results in request order including duplicates.

1. Resolve ALIDs to rows (unchanged, `rows_of`).
2. Judge the request (§8.1). Refuse before reading any frame.
3. Split the present hits into **components**: per chromosome, sort the hits by position and cut
   wherever two consecutive distinct positions are ≥ W apart. No lead can reach a hit in another
   component (the edge needs d < W), so each component's greedy is independent of the others.
   Running them separately, in any order, gives the global greedy's leads and members.
4. For each component, serially:
   - order its hits by score (ties as above) and mark all open;
   - take the strongest open hit as the lead, read `partner_row(c, lead)`, and keep the entries
     whose partner is an open hit of the component, with (code / s)² > r2 and d < W;
   - those hits join the lead, with their signed r. The lead and its members are closed;
   - repeat until no hit is open.
5. Scatter results into request order (unchanged).

"An open hit of the component" is a lookup in an array `hit_slot[row − row_lo]` over the
component's row span, so filtering a row costs O(E_l) vector operations with no hashing.
Duplicate ALIDs share a row and follow today's scatter rule.

**Cost.** One frame read and decode per lead that has a partner row and has not hit the frame
cache. No speculative reads and no reads for eliminated hits. For GCST90476460 that is at most
3,869 frame reads, about 3,869 × 16 KB ≈ 60 MB compressed at EUR's median frame size, against
10.7 GB today. These are estimates. §13 measures them.

### 8.3 `proxies`

For each present query variant, read `partner_row`, keep entries with (code / s)² ≥ r2 and d ≤ W,
and order them as today (r̂² descending, distance ascending, proxy ALID). This is one frame read
per query variant at every answered r2, including those below t_far within `near_bp`, which today
read a Near Band column.

### 8.4 What the proposal deletes

Once D1 and D2 are taken, `clump` and `proxies` never touch `near/` or `index/`. These go:

- the Proxy Index (`index/<c>/`), its writer, and its six validation rules;
- `StoreReader.clump_warmer`, and `partners` reading the Near Band row and column, with `among`;
- clump's use of the Near Band chunk cache, single-flight and admission. The machinery from
  #46/#50/#51 stays only for `matrix` and `region`, which read the Near Band whole-row and
  contiguously.

## 9. Parallelisation

### 9.1 Building

§6.3: chromosomes across processes, decode → emit → compress as a pipeline within each. The
output is deterministic for any P and any thread count.

### 9.2 Querying

The unit of independent work is the **component** (§8.2). Chromosomes are coarser and always
whole sets of components. The options, from simplest:

| | option | what it hides | its cost | use when |
|---|---|---|---|---|
| A | **serial**: components one after another, one thread | nothing | none | default; measure first |
| B | **component tasks on a bounded thread pool** | cold-read latency (local cold, HTTP); zstd decode and `pread` release the GIL | per-lead Python work holds the GIL, so B scales I/O, not CPU | cold or remote stores, large Analyses |
| C | **component tasks on a process pool**; workers open the store once, mmap shared | CPU-bound greedy beyond one core | worker start-up, pickling of hit arrays (small) | only if A and B leave the greedy CPU-bound on large Analyses |
| D | **Analyses in parallel**, each serial, sharing one reader's frame cache | many concurrent users | the frame cache needs a lock | a service handling many requests |
| E | **read-ahead within a component**: also fetch the frames of the next K open hits in score order | latency on HTTP | reads for hits a stronger lead then takes; #41 measured 66% wasted at depth 8 | HTTP stores only, with the waste counted |

**Recommendation.** Build A first, then B with the pool size internal to the reader (no public
signature change), bounded by the reader's thread budget. Reach for C only on evidence.

- B, C and D need no ordering rules: components are independent and results are scattered by
  request index, so the answer is the same at any concurrency. A test holds this by running one
  Analysis at 1, 2 and 8 workers and comparing results byte for byte.
- Two tasks may decode the same frame when two components share one (possible for small W). The
  cost is a duplicate decode, not a wrong answer. No single-flight is needed.
- A refusal is decided before any task starts (§8.2, step 2). A frame that fails its checks fails
  the whole request: outstanding tasks are cancelled or drained, and no partial result is
  returned.

**What will bound latency next.** On GCST90476460 today, ALID resolution (`rows_of`) takes
3.18 s, and store open, score and loci 3.39 s together. Those 3.4 s become most of a sub-second
clump's time. They are out of scope here. They are #50's mmap lookup question, raised as its own
issue.

## 10. Validation

These rules replace the six `index.*` rules and `far.index` in `ld-store.md` §12. Each reads frames in bounded
blocks, never a whole chromosome. Each has a breaker in `tests/contract/test_store.py`.

| rule | checks |
|---|---|
| `partners.files` | every chromosome of the Variant Set has `partners/<c>/` with its three files; the manifest's `rows` is n_c, `frames` is ⌈n_c / G⌉, `frames.idx` has F + 1 ascending entries from 0 to `frames_bin_bytes`, and both SHA-256s match |
| `partners.frames` | every frame decompresses with a valid checksum, its k is the rows due, and its payload is exactly 4 + 4k + 6E bytes |
| `partners.rows` | within a row, partners are strictly ascending, in [0, n_c), and never the row; Σ E equals the manifest's `entries` |
| `partners.codes` | every code is one of the store's encoding (as `near.codes`) |
| `partners.distance` | every entry lies within `far_bp` |
| `partners.threshold` | \|code\| ≥ `partner_code_floor` within `near_bp`, ≥ `far_code_floor` beyond it |
| `partners.symmetric` | (l, m, c) is an entry exactly when (m, l, c) is (the 64-bit hash-sum method of today's `index.symmetric`) |
| `partners.near` | every entry within `near_bp` carries the Near Band's code, and every Near Band code at the Partner Floor is an entry (today's `index.near` method, at the new floor) |
| `partners.far` | the entries beyond `near_bp`, in each pair's earlier row, are exactly the Far Pairs, with the same codes (today's `far.index`, read against the Partner Rows) |
| `provenance.partner_threshold` | t_partner ≤ t_far; `partner_code_floor` recomputes from t_partner at the recorded scale |

`partners.near` and `partners.far` together prove §3's "copies, by design". With
`partners.symmetric`, they prove the component holds every pair it should and no other.

## 11. Size

Measured on chr22 of each published k = 10 HGDP+1kGP store, 18 November 2026, at r̂² > 0.01
within `near_bp`. Each kept pair is stored in both rows, as an int16 partner gap and an int16 code
in two byte-shuffled streams, zstd-3 in blocks of 65,536 entries. The genome is projected from
chr22 by each panel's genome-to-chr22 Near Band size ratio.

| panel | pairs kept | null share at 0.01 | current store | Partner Rows | beside the index | replacing it (D1) |
|---|---|---|---|---|---|---|
| EUR | 5.0% | 1.05% | 79.3 GB | 9.0 GB | +11% | |
| AFR | 4.1% | | 241.3 GB | 26.9 GB | +11% | |
| EAS | 4.1% | | 63.9 GB | 5.9 GB | +9% | |
| SAS | 4.5% | | 84.9 GB | 8.9 GB | +10% | |
| AMR | 16.3% | 7.2% | 64.3 GB | 24.3 GB | +38% | |
| **all** | | | **533.7 GB** | **75.0 GB** | **+14%** | **≈ +33 GB, +6%** |

- "Replacing it" is 75.0 GB, plus about 3 GB for the Far Pairs held in both directions, less the
  Proxy Index's 45.2 GB.
- **These are estimates of the final layout.** The §5 frames store int32 deltas, because
  entries beyond `near_bp` can be more than 32,767 rows apart. They also compress G rows at a
  time, not 65,536 entries. Both cost some compression. The prototype measures the real bytes
  (§13).
- **AMR's sample is small.** At n = 341, 7.2% of unlinked pairs pass r² 0.01 by chance, so about
  44% of AMR's Partner Rows are sampling noise. D3 lets AMR's build set a higher t_partner, up to
  its t_far (0.0632), at the price of refusing AMR clumps below it.
- **Why not lower.** At r² > 0.001, 46.2% of EUR chr22's band passes, against 41.9% expected
  from sampling alone, so about 90% of those pairs are noise. A sparse row there would be larger
  than the dense band (6 bytes × 46% > 2 bytes). D2 refuses such requests. On UK Biobank EUR
  (n = 49,766) the null share at 0.001 is about 10⁻¹², and t_partner = t_far = 0.001 there.

The scripts and per-panel results that produced these numbers: `band_fraction.py`,
`tier_size.py` and `matrix_layout.py`. They are to be committed on the prototype branch
(`proto/partner-rows`) and cited from there, as `ld-store.md` cites `proto/1-ld-store-layout`.

## 12. Alternatives weighed

- **Keep the Proxy Index beside the Partner Rows** (against D1): +75 GB rather than +33 GB, and
  two components holding the same pairs, each with its own writer and validation rules.
- **Keep the Near Band path below t_partner** (against D2): answers r² 0.001 clumps, but keeps
  the clump warmer, the column reads and clump's coupling to the chunk cache. These are what made
  clumping slow, and on HGDP+1kGP they would serve thresholds where 90% of passing pairs are
  noise.
- **One rule for every store, min(0.01, t_far)** (against D3): simpler, but it fixes AMR's cost
  at +38%. A per-dataset t_partner lets AMR's build choose a higher one: at its t_far, the AMR
  row of §11 falls to about its share there, 2.4% of pairs (`ld-store.md` §5).
- **Zarr arrays for the frames** (against D4): `frames.bin` as a `uint8` Zarr array with no inner
  compression keeps one hierarchy, but a cold frame read resolves the shard index first: two
  reads, not one.
- **Fold into 0.1.0** (against D5): no version bump, but a 0.1.0 reader could not tell a store
  with Partner Rows from one without.
- **Dense symmetric rows** (#41, #50; epic #45's rejected arm): fast, but about 181 GB for the
  79.3 GB EUR store. The Partner Rows are the same idea, thresholded: about 9 GB for EUR.
- **Framed triangle rows** (#45's other rejected arm): a lead's upstream partners still fan out
  across thousands of earlier rows. Symmetric rows have no upstream fan-out.

## 13. Plan and acceptance

1. **Prototype (one lane)**, on the published EUR store, chr22 then the genome:
   - derive Partner Rows at G ∈ {1, 4, 16, 64}, and report bytes, derive time and peak memory;
   - clump the 37 chr22 Analyses at r² 0.01 and GCST90476460, cold and warm, under options A
     and B. Report frame reads, bytes, decode CPU and wall time, separated from ALID resolution;
   - check exact fingerprints against the current reader on every Analysis;
   - check that zarr-python opens and lists a store holding `partners/` without complaint.
2. **ADR 0011** records D1–D6 (drafted with this spec). Step 1's measurements may revise G and the
   §11 sizes. A result that undermines a decision goes back to the owner.
3. **Implementation**: the emitter in `StoreWriter`; `derive-partners`; `partner_row` and the frame
   cache; `clump` and `proxies` on Partner Rows; the refusal; the §10 rules and their breakers;
   `ld-store.md` §1, §5, §7, §9, §10, §12 and §13 amended; `CONTEXT.md` terms; a changelog
   fragment (`breaking`, for D2's new refusal).
4. **Rollout**: derive for the five stores; pass the realdata tier on the IEU nodes; then drop
   `index/` (D1).

Acceptance:

- [ ] `clump` and `proxies` answers are identical to the current reader's on every request both
      answer: the contract tests, the Test Store, the 37 chr22 Analyses at r² 0.01, the realdata
      tier and `acceptance-clump` with its eight explained differences.
- [ ] Requests below t_partner within `near_bp` are refused with `threshold_below_t_partner`,
      tested at r2 one step below and at the floor.
- [ ] A derived store's `frames.bin` equals the built one's, byte for byte, on the Test Store.
- [ ] Every §10 rule fails its breaker. A truncated `frames.bin`, a flipped byte in a frame and
      a short payload are each refused, naming the chromosome and the frame.
- [ ] Results are identical at 1, 2 and 8 query workers.
- [ ] GCST90476460 clumps at r² 0.01 in under 1 s of LD work, cold, local disk, excluding ALID
      resolution, which is reported separately.
- [ ] The measured size of the five panels' Partner Rows is recorded and replaces §11's
      projections.
