# Partner Rows: a clump-and-proxy component for the LD Store

Status: **accepted, and normative beside `docs/spec/ld-store.md`**, which takes the sections this
document amends. The decisions are the owner's, of 18 November 2026 (§1; ADR 0097), revised by the
measurements of opengwas/ogld#65 and its fixed G (§11).
Format version described: `0.2.0` (decision D5); the version bump lands with the writer, opengwas/ogld#67.

This document specifies a new LD Store component, the **Partner Rows**. For each variant it
holds every partner whose Stored r reaches a floor, on both sides, stored as one small compressed
frame that a reader fetches with one positional read and decompresses without Zarr. `clump`
and `proxies` read only this component. The Near Band and Far Pairs stay as they are, and serve
only `matrix` and `region`.

It specifies the format (§3–§5), how the component is built (§6), how it is read and queried
(§7–§8), how both are parallelised (§9), how it is validated (§10), and what it costs (§11). The
evidence is on issues opengwas/ogld#40, opengwas/ogld#45, opengwas/ogld#50 and opengwas/ogld#65 and in the measurements of 17–18 November 2026 (§11).

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
  dense chr22 greedy from 0.227 s to 0.056 s on opengwas/ogld#50's packed-row prototype. The rows there
  were dense; these are 20× smaller.

Epic opengwas/ogld#45 decided to keep the triangle and reject symmetric rows, because dense symmetric rows
would have cost about 181 GB for the EUR store. Thresholding changes that trade: the Partner Rows
are symmetric rows at about 5% of the size. For `clump` and `proxies` this design supersedes opengwas/ogld#45's
decision 1. The triangle stays for `matrix` and `region`.

## 1. Decisions (owner, 18 November 2026)

| | decision | consequence |
|---|---|---|
| D1 | **The Partner Rows replace the Proxy Index.** | They hold every index entry (§3), so `proxies` reads them too. Measured on EUR: 14.76 GB at G = 16 (11.82 GB at G = 64), which is +5.38 GB (+2.44 GB) net of its 9.38 GB index; the owner's scaling puts the five panels at about +53 GB net at G = 64 (§11) |
| D2 | **`clump` and `proxies` with r2 below t_partner and W ≤ `near_bp` are refused**, `threshold_below_t_partner`. | On HGDP+1kGP at r² 0.001, 90% of the pairs that pass are sampling noise (§11). Today's answered r² 0.001 clumps on those stores become Refusals: a breaking change |
| D3 | **t_partner is set per reference dataset and recorded in the store's metadata**, as t_far is. | A build input with no default: each Source Panel's build names its own (0.01 for HGDP+1kGP, 0.001 = t_far for UK Biobank EUR), and a build without one refuses. It never exceeds t_far (§3) |
| D4 | **A Zarr group** per chromosome, inside the store's one hierarchy (§5; refined by the owner on 18 November 2026, ADR 0097). | `partners/<chrom>/` holds two arrays, `frames` and `frames_idx`, each one chunk, and the manifest's keys as the group's attributes. One read per frame, with no shard index. Plain files make zarr's `members()` warn; a node in the hierarchy is what removes it, and opengwas/ogld#67 measures that on a real store |
| D5 | **Format version 0.2.0.** | A new component, a removed one and a changed refusal table |
| D6 | **Derive** the five existing stores' Partner Rows from their Near Band and Far Pairs (§6.2). | No genotypes needed. Byte-identical to a build |

**Fixed by the measurements of 18 November 2026:**

- **G = 16** (measured on opengwas/ogld#65; §5.2, §11). It is the smallest G of those equally fast: 14.76 GB of
  EUR genome bytes against G = 8's 16.68 GB, and 1.067 s cold LD against 1.051 s, a difference
  inside the samples' spread.
- **Component workers are capped at 4** for opengwas/ogld#71 (§9.2, option B). At eight, cold LD regresses to
  1.913 / 5.753 / 5.109 s at G = 4 / 8 / 16, against 1.051–1.169 s where each G was fastest.
- **The under-1 s LD target moves to opengwas/ogld#71** (owner, 18 November 2026). The best cold LD is 1.051 s
  (G = 8) and at G = 16 it is 1.067 s cold and 1.058 s warm, so nothing here claims the target.
  opengwas/ogld#71 did not meet it and the owner accepted it (opengwas/ogld#71's owner decision, recorded on opengwas/ogld#72): a quiet
  re-time at `cc48a58` on GCST90476460 gave cold LD medians of **2.30 / 1.55 / 2.95 s** at
  1 / 2 / 4 workers, and warm medians of **0.98 / 0.97 / 1.35 s** (EUR k = 10). The follow-up is
  **opengwas/ogld#81**, decode and gather plus ALID resolution, and it owns the worker default too.

§12 records the alternatives that were weighed and what they would have cost.

## 2. Terms

The **Partner Rows**, the **Partner Threshold**, the **Partner Floor** and the **Frame** are
`CONTEXT.md`'s terms, and this document uses them as it defines them. §4 adds the one thing the
glossary does not say: a pair within reach that is not in a row has a known bound, never a 0.

## 3. Which pairs the Partner Rows hold

`ld-store.md` §5 defines which pairs a row holds; this section gives the reasons behind that rule.

- A pair within `far_bp` is in both variants' rows, with the same code. One row therefore answers a
  variant's partners on both sides, which is the whole point: the Near Band's column read is what
  made `clump` and `proxies` slow.
- Beyond `near_bp` the floor is the far code floor, which is exactly the Far Pairs. So the rows
  beyond `near_bp` are the Far Pairs' codes, in both directions rather than once.
- Entries within a row are in strictly ascending m. A row never holds l itself, so a cumulative
  sum needs no case for it.
- **Copies, by design.** Every code within `near_bp` is the Near Band's code for that pair, and
  every code beyond it is the Far Pairs' code. The component holds no value the store does not
  already hold. It is a second layout of the same codes, chosen for clump and proxy reads.
- **It contains the Proxy Index.** t_partner ≤ t_far, so the Partner Floor is no higher than
  `far_code_floor`. Every Proxy Index entry is therefore an entry here, with the same code: the
  Proxy Index is the Partner Rows thresholded at `far_code_floor`. This is what lets D1 retire
  the index.
- t_partner is a REQUIRED build input, with no default (D3), and MUST be ≤ t_far: a build without
  one refuses naming it, and one given a larger refuses naming both values. With a larger one, a
  pair between t_far and t_partner within `near_bp` would be in neither component, and a request
  at r2 ≥ t_far could not be answered.

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

### 5.1 The group

```text
<ld store>/                   one Zarr v3 hierarchy (`ld-store.md` §1)
└── partners/<chrom>/         a Zarr v3 group; the manifest's keys are its attributes (§5.3)
    ├── frames/               uint8, shape (B,), one chunk of shape (B,), codecs: bytes only
    │   └── c/0               the chunk file: the chromosome's F zstd frames, back to back
    └── frames_idx/           uint64, shape (F + 1,), one chunk: the offsets, little-endian
        └── c/0
```

`partners/` is a node in the store's one Zarr hierarchy, not a directory beside it (D4, refined by
the owner on 18 November 2026; ADR 0097). So `ld-store.md` §1's envelope stays "one Zarr v3
hierarchy", and a reader that lists the store sees the component.

The alternative, plain files in a directory with no group metadata, was measured on the published
EUR store (opengwas/ogld#65, zarr-python 3.4.2): `zarr.open_group`, `opengwasdb.ld.open` and all 27 of `ogld.validate`'s
rules are clean, but `members()` emits `ZarrUserWarning: Object at partners is not recognized as a
component of a Zarr hierarchy.` The single-chunk form below removes that warning. The committed
Test Store is that check since opengwas/ogld#67: `partners/` is a node of its hierarchy, `members()` lists it
with no warning (`tests/ld/test_partner_rows.py`), and the reader and the ten rules resolve each
chunk file from the array's own metadata (`tests/ld/test_partner_row_reads.py`).

Both arrays have **one chunk each**, of the whole array, and no shard index. The `frames` array's
codec chain is `bytes` alone, with **no compressor**, because the bytes it holds are already zstd
frames (§5.2). So its chunk file is the raw concatenated frames, and a frame is one `pread` or
one HTTP Range request into it. A reader MUST resolve that file's path from the `frames` array's
Zarr metadata and its chunk key encoding, never by assuming a name: the default is `c/0` today, and
a store may write another.

### 5.2 Frames

- G rows to a frame, G one of the group's attributes. Frame f holds rows [f·G, min((f+1)·G, n_c)),
  so F = ⌈n_c / G⌉ and row l is in frame ⌊l / G⌋. **G is 16** (§1): the smallest G of those
  equally fast, 14.76 GB of EUR genome bytes against G = 8's 16.68 GB, at 1.067 s against 1.051 s
  cold. On EUR chr22, whose median row holds 800 entries, a median frame is about 77 KB raw and
  about 20 KB compressed: its 7,477 frames are 153.6 MB with the index and the attributes. Readers
  MUST take G from the group's attributes, never assume it.
- Frame f's bytes are the `frames` chunk file's bytes `[offsets[f], offsets[f + 1])`, with
  `offsets` the `frames_idx` chunk. `offsets[0]` is 0 and `offsets[F]` is the file's size.
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

### 5.3 The group's attributes

The `partners/<chrom>/` group's attributes:

| key | type | meaning |
|---|---|---|
| `rows` | int | n_c, equal to the chromosome's rows in `variants/` |
| `rows_per_frame` | int | G |
| `frames` | int | F |
| `entries` | int | Σ E over all frames: twice the pairs held |
| `frames_bytes` | int | the length of the `frames` chunk file |
| `frames_sha256`, `frames_idx_sha256` | string | the two chunk files' SHA-256 |
| `zstd_level` | int | 3 |

The root provenance (`ld-store.md` §7) gains `t_partner` (float), `partner_code_floor` (int),
`counts.partner_entries` (int, every chromosome's `entries`) and `software.zstd` (the library
version, because frame bytes depend on it). `counts.index_entries` went with the Proxy Index in
opengwas/ogld#72 (D1): a build writes no index and records no such count, and one that still carries the key is
not failed for it (ld-store §7, §12).

### 5.4 Why this form

- **One lead, one read, one decode.** The reader holds the `frames_idx` array (8 bytes per G rows:
  4.3 MB for EUR's 8.59 M variants at G = 16, against 69 MB for a Near Band `indptr`). It finds
  frame ⌊l / G⌋, reads its byte range from the `frames` chunk file and decompresses it. There is no
  shard index and no codec chain to recognise. A local store's frames are read with one
  `os.pread(fd, length, offset)` on a descriptor the reader opens once for the chromosome and keeps
  until it is closed, which costs no store call and no event loop: 10.8 µs a 25,600-byte read on a
  512 MiB file, against 1,257 µs through the store's own path on the same file and page cache
  (`scripts/ld/bench/frame_read.py`). An HTTP store's frames are one Range request through the store's
  interface, which is one event-loop hop and one round trip.
- **Corruption is loud.** A missing or short `frames` chunk file, a frame whose checksum fails,
  or a payload of the wrong length is refused, naming the chromosome and the frame. It is never
  read as a short row. This is the store's rule that a truncated window is not a short answer.
- **Sizes vary with LD, so chunks cannot be fixed.** Rows hold a median of 800 and up to 4,665
  entries on EUR chr22,
  so a fixed-size chunk always cuts through rows. Grouping a fixed number of rows makes the
  address arithmetic trivial and lets the bytes vary.

## 6. Building

### 6.1 One row emitter, two producers

The Partner Rows are a function of the Near Band and the Far Pairs (§3). One emitter makes them
from codes arriving in row order, block of rows by block of rows, as the Proxy Index's writer did
before opengwas/ogld#72 removed it (`ld-store.md` §6, "How a store is written"):

1. For each block, keep each pair (l, m > l) with |code| at the floor its distance calls for.
2. Each kept pair goes to row l immediately, and to a carry for row m.
3. Row l is complete when its own block has been processed. Its upstream entries are all in the
   carry, from earlier blocks or its own, and its downstream entries are in its own row. When
   rows f·G … f·G + G − 1 are all complete, frame f is assembled, compressed and appended.
4. The carry holds only pairs whose later row is still to come: those within `far_bp` of the
   rows being written. The prototype's exact row-boundary scan puts the peak at 85,926,574 pairs
   on chr6 and 3,231,727 on chr1: 515.6 MB and 19.4 MB at 6 bytes a pair with the destination
   implicit, or 859.3 MB and 32.3 MB at 10. Either way it is a chromosome's row-boundary peak,
   not a chromosome sort. On the Test Store written in blocks of 16 rows, the emitter's peak carry
   is 5,235 pairs — 52,350 bytes at 10 bytes a pair — and 5,277 pairs in blocks of one row, against
   0 written whole, as the committed store is (`pixi run -e dev python
   scripts/ld/bench/partner_carry.py`).

Its two producers:

- **The build** (`StoreWriter`): the emitter took the Proxy Index writer's place, which opengwas/ogld#72 removed
  it, and is fed by the same blocks. No new pass over genotypes.
- **Derivation** (§6.2): the same emitter fed by reading the Near Band and the Far Pairs back.

Both MUST produce identical bytes for the same store. The derivation of a store built with Partner
Rows reproduces its `frames` chunk file exactly, and a test holds this on the Test Store.

### 6.2 Deriving the existing stores (D6)

`opengwasdb ld derive-partners <store> --t-partner T [--threads t] [--processes p]
[--chromosomes C]… [--drop-index]` (ogld#69):

- `--t-partner` is required and has no default (D3), and it is never above the store's recorded
  `t_far`: a larger one is refused naming both. A store whose provenance already records Partner
  Rows at another threshold is refused too, naming both, because a component at a second floor
  would hold two.
- Reads `near/<c>/r` with the direct decoder (ADR 0095) in blocks of rows, and the Far Pairs as
  their `indptr` whole (n_c + 1 entries) and their `partner` and `r` arrays one block of rows'
  worth at a time (101,886 pairs on EUR chr22). It feeds both to §6.1's emitter, so the `frames`
  and `frames_idx` chunk files are a build's, byte for byte, at any `--threads` and
  `--processes`.
- Writes `partners/<c>/` into a temporary directory inside `partners/`, then renames it into place
  once every frame has been written and checked (§5.1). A run killed part-way therefore leaves
  every chromosome whole or absent, plus at most a `partners/.<c>.tmp-*` a rerun removes; a run
  that derives some chromosomes and not others leaves the others absent (§6.3).
- **The claim is one write, and it is the provenance's.** Only once every chromosome of the Variant
  Set is in place and §10's ten rules pass does a run make `partners/` a node of the store's
  hierarchy and write `format_version` `0.2.0` with the four keys, `counts.partner_entries` being
  every chromosome's own `entries`. That second write is the claim — one replaced metadata
  document — so a `partners/` no provenance names is **unclaimed**: a store that does not have the
  Partner Rows, whatever directories exist. Validation judges nothing in it and a reader refuses
  it, since both key on the recorded `format_version` and `t_partner`, never on a group being there
  (`ld-store.md` §12, §13). A rerun of an unclaimed store settles it: the ten rules judge the
  residue, and one they accept is claimed while one they refuse is removed and derived again —
  never trusted, and never derived again for nothing.
- Writes only inside the store, and never through a symbolic link: before anything is written, a
  run refuses a link at any path it owns — the store's root Zarr metadata document, `partners/`, or any entry of
  `partners/` — even one that stays inside the store, because a claim, a cleanup or a publication
  would then write or delete in the directory the link names. A store's read-only components may
  be links, which is how the scratch stores opengwas/ogld#71 and opengwas/ogld#73 read are laid out. It refuses a
  `zarr.*.partial` the store already holds, in the root or in `partners/`, once before it writes
  and again immediately before the claim: zarr writes a metadata document through that name and
  follows a link there. The check is not a lock — a concurrent writer to the same store could still
  plant one between the check and the write — and this derivation does not support one: a store
  has one writer, as §6.1's carry and §5.1's rename already assume.
- `--chromosomes C`, repeated once per chromosome, names the ones to derive — all the store lacks
  by default — and one the store does not have is refused; one it already has is skipped.
- Drops `index/` afterwards, as its own step and only on request (`--drop-index`), so a derived
  store can be checked against its index before the index is removed. It removes the arrays, then
  `counts.index_entries`, and only from a store whose Partner Rows pass §10's rules, and never in
  the same invocation as a derivation that did not become whole. A removal that fails raises,
  naming the path, and the count stays: the provenance never claims a removal that did not happen.
  An `index/` that is a symbolic link is unlinked and the directory it names is left whole, which
  is what a scratch store over a published one needs. A store this code writes holds no `index/`,
  so the option is what finishes one derived before opengwas/ogld#72; since opengwas/ogld#72 the store validates and reads
  with or without the leftover (ld-store §12).

### 6.3 Parallelising the build and the derivation

- **Across chromosomes: processes.** Chromosomes share nothing. A pool of P worker processes,
  each taking the largest remaining chromosome first, bounds memory at P chromosome carries plus
  their decode buffers. chr1, chr2 and chr6 dominate. Starting them first keeps the pool's
  makespan near the largest chromosome's time. `--processes p` starts that pool with `spawn`, so
  no worker inherits a reader, a chunk cache, a frame source's descriptor or a decoder's pool,
  none of which is fork-safe (§7, §9.2); each worker opens the store itself.
- **Within a chromosome: two stages.** The Near Band's blocks are decoded, thresholded, carried
  and assembled in row order on one thread, and a pool of `--threads` compressor threads compresses
  the frames. zstd releases the GIL, and frames are appended strictly in frame order, so the
  output does not depend on the pool's size or scheduling.
- **The inverse address rule is judged at the floor, not on the whole band.** A block's Near Band
  codes are tested against the Partner Floor first, and only the codes that reach it are given
  their pair by §4's address rule. On EUR chr22 that is 5.03% of 1,165,485,828 codes, and running
  the rule on all of them was 37 s of the chromosome's 41 s of selection, the rest being the read:
  the whole chromosome, frames included, is **29.4 s and 640 MiB peak RSS** at eight compressor
  threads (16 cores' box, `taskset -c 0-7`, load 3.4), against 86.1 s and 790 MiB before. It
  writes 7,477 frames, 153,561,655 bytes and **117,392,902 entries** — which is opengwas/ogld#65's own chr22
  output, to the count and the byte: [ogld's chr22 derivation record](https://github.com/opengwas/ogld/blob/912a9fab30c59b478487258f1c8b6a6766398eb9/prototypes/partner-rows/results/derive_chr22.json) at `912a9fa` records that many entries at
  G = 16, and its two chunk-file SHA-256s are this run's. The 117,189,130 that §11's `sym_entries`
  records for the chromosome is the Near Band's directed count, twice its 58,594,565 kept pairs,
  and the difference is exactly the 101,886 Far Pairs in both directions.
- **The ten rules cost more than the derivation.** They read every frame and, through
  `partners.near` and `partners.far`, the whole Near Band and Far Pairs again, on `--threads`:
  on EUR's genome, at eight threads, 25 min 11 s against the 12 min 49 s the derivation took.
- **Measured cost** (opengwas/ogld#65's prototype, and opengwas/ogld#69's derivation). The prototype took 47.9 s for EUR
  chr22 at all four G, with the selection and the sort shared, and 1,179.3 s of group wall time
  for the other 21 chromosomes at two G on two processes; it sorts a whole chromosome, which the
  streaming emitter does not: its 59.23 GB worker high water is the sort's, and §6.1's
  85,926,574-pair chr6 peak is what a carry-bounded emitter budgets. opengwas/ogld#69 derived the published
  EUR store's other 21 chromosomes in 12 min 49 s on four processes of eight compressor threads
  (load 5–10), leaving them and chr22 as a 14.8 GB component.

## 7. Reading

`opengwasdb.ld.store.StoreReader` gains one read, replacing the Proxy Index read and the Near Band
partner reads:

```text
partner_row(chrom, l) -> (partners: int32[E_l], codes: int16[E_l])
partner_rows(chrom, l…) -> (indptr, partners, codes)
```

1. On the chromosome's first use, read the group's attributes and the `frames_idx` chunk whole,
   and check `frames_idx[F]` against `frames_bytes`.
2. f = ⌊l / G⌋. If frame f is in the frame cache, use it. Otherwise read its byte range from the
   `frames` chunk file, whose path the array's metadata gives (§5.1), decompress it (the zstd
   checksum is verified), check k and the payload length (§5.2), and cache the frame's
   decompressed payload with its rows' lengths.
3. Gather row l's own bytes from the payload's planes — the **row-local unshuffle** — cumulate its
   deltas and return its partners and codes. The frame's other rows are never unshuffled to answer
   one row, so a row costs one row's work: measured on a median EUR frame (16 rows, 800 entries
   each, a 76.9 KiB payload), one row is 21 µs against 272 µs to unshuffle and decode the whole
   frame, of which the decompress is 87 µs either way.

- **The frame cache** is a byte-bounded LRU of frames' decompressed payloads, per reader, which
  drops the least recently used frame first. **Its 256 MiB default is measured** (opengwas/ogld#71): GCST90476460
  and the 37 chr22 Analyses in one reader on a warm page cache made 4,323 frame reads with no cache,
  4,205 at 16 MiB, 4,188 at 64 MiB and 4,132 (191 hits) at 256 MiB, and 1 GiB and 4 GiB made the
  same 4,132. So 256 MiB is the smallest budget that kept every hit the workload offered, and a
  larger one buys nothing; chr22's whole frames file is 117 MB, so its frames fit. Neighbouring
  leads often share a frame, and repeated Analyses on one reader hit it, which is where the 191
  come from. It is separate from the Near Band chunk cache, which keeps serving `matrix` and
  `region`.
- **One seam, two transports.** A frame's bytes are read through the array's own chunk file, whose
  path the metadata gives (§5.1) and whose name is never assumed. A local store's file is held open
  and read with `os.pread` — one syscall a read, no store call, no event loop — and the descriptor
  is opened once per chromosome, on that chromosome's first use, and closed with the reader. A
  store that is not on this disk answers each read as one Range request through its store's
  interface. A cold read is one random read of a frame, typically 4–64 KiB, which the operating
  system's page cache serves warm.
- **Many rows at once** is `partner_rows`, which returns the rows' entries back to back as CSR, in
  the order asked, and coalesces only physically adjacent frames into one read (§9.2): frames with
  bytes between them are read one at a time, because the bytes between them would be read for
  nothing.
- **A store with no Partner Rows is refused, naming the missing component**, and so is a frame
  that fails its declared content checksum or content size, its checksum, its k or its payload
  length, naming the chromosome and the frame. The frame's zstd header MUST declare both the
  content checksum and the content size (§5.2), and a header that does not is refused before
  anything is decoded: a decoder verifies only what a frame's header says is there. A read shorter
  than the frame's bytes — a truncated chunk file, on either transport — is a refusal too. A row is
  never answered from a damaged frame, nor as a short one.

## 8. Queries

### 8.1 What each request reads

`ld-store.md` §9's refusal table carries the rows from 0.2.0; this section does not restate them.
Two rows change: r2 ≥ t_far with a 3–10 Mb window is answered from the Partner Rows rather than the
Proxy Index, and within 3 Mb the table splits at t_partner, refusing `threshold_below_t_partner`
below it (D2). The rest of the table is unchanged.

**Why every answered request is exact** — `ld-store.md` §9's argument, which 0.2.0's rows are the
reason for. On every request both versions answer, the answers are those of the 0.1.0 reader,
element for element, which opengwas/ogld#71's acceptance criteria hold.

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

**Cost** (measured in opengwas/ogld#65 and opengwas/ogld#71). One frame read and decode per lead that has not hit the frame
cache. No speculative reads and no reads for eliminated hits. On the whole of GCST90476460, whose
3,869 leads fall in 95 components, G = 16 with four component workers made 3,854 frame reads and
101,810,367 bytes, with 15 cache hits. Its LD work, cold and warm, in seconds, by the reader's
worker count, median of three alternating fresh subprocesses on 8 CPUs with `taskset -c 0-7`, while
Partner Rows were being derived on the same node (load average 5.6–6.5):

| workers | cold | warm |
|---|---|---|
| 1 | 3.279 (best 2.603) | 1.078 (best 0.975) |
| 2 | 1.904 (best 1.513) | 1.254 (best 1.019) |
| 4 | 1.846 (best 1.832) | 1.552 (best 1.293) |

Under a lighter load (3.5–5) the same configuration measured 1.33–1.87 s cold and 1.33–1.40 s warm
at four workers. `rows_of` is separate and slower than the LD work, 2.3–3.6 s for its 260,821
ALIDs, and the store opens in 0.02–0.05 s. Where the 1.85 s cold goes at four workers: 1.31 s of
`partner_row` CPU — the frames' decompress on a miss, 0.94 s of it in the warm runs — and 0.82 s
of row gather (2.13 s of `partner_row` CPU less that decode), spread over four threads and so
bounded by the GIL, plus the component split, the greedy and the cold reads' latency. Serially the
same parts are 0.87 s of LD CPU: 0.49 s decode, 0.17 s gather, and 0.21 s of the split and the
greedy. **The under-1 s LD target is not met, and the owner accepted it** (opengwas/ogld#71's owner decision,
recorded on opengwas/ogld#72). The figures to hold the reader to are that quiet re-time, at `cc48a58`, on
GCST90476460 at EUR k = 10: cold LD medians of **2.30 / 1.55 / 2.95 s** at 1 / 2 / 4 workers, and
warm medians of **0.98 / 0.97 / 1.35 s**. The follow-up is **opengwas/ogld#81**, decode and gather plus ALID
resolution; the worker default stays at four here and is opengwas/ogld#81's to change. The options the numbers
point at, measured rather than guessed: a reusable
row buffer in `FramePayload.row` (opengwas/ogld#70's module) is 31.7 µs a row against 40.8 µs on a 628-entry
median row, some 35 ms over 3,854 leads; G = 8, which opengwas/ogld#65 measured a little faster than G = 16; and
more than four workers, which regresses here as it did there.

### 8.3 `proxies`

For each present query variant, read `partner_row`, keep entries with (code / s)² ≥ r2 and d ≤ W,
and order them as today (r̂² descending, distance ascending, proxy ALID). This is one frame read
per query variant at every answered r2, including those below t_far within `near_bp`, which today
read a Near Band column.

### 8.4 What the proposal deletes

Once D1 and D2 are taken, `clump` and `proxies` never touch `near/` or `index/`. These go, and
went in opengwas/ogld#72:

- the Proxy Index (`index/<c>/`), its writer, and its six validation rules, with `far.index`;
- `StoreReader.clump_warmer`, and `partners` reading the Near Band row and column, with `among`;
- clump's use of the Near Band chunk cache, single-flight and admission. The machinery from
  opengwas/ogld#46/#50/#51 stays only for `matrix` and `region`, which read the Near Band whole-row and
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
| A | **serial**: components one after another, one thread | nothing | none | built and measured first (opengwas/ogld#71); one worker, when the reader's pool is set to it |
| B | **component tasks on a bounded thread pool**, capped at 4 for opengwas/ogld#71 | cold-read latency (local cold, HTTP); zstd decode and `pread` release the GIL | per-lead Python work holds the GIL, so B scales I/O, not CPU; above four tasks the work regresses | cold or remote stores, large Analyses |
| C | **component tasks on a process pool**; workers open the store once, mmap shared | CPU-bound greedy beyond one core | worker start-up, pickling of hit arrays (small) | only if A and B leave the greedy CPU-bound on large Analyses |
| D | **Analyses in parallel**, each serial, sharing one reader's frame cache | many concurrent users | the frame cache needs a lock | a service handling many requests |
| E | **read-ahead within a component**: also fetch the frames of the next K open hits in score order | latency on HTTP | reads for hits a stronger lead then takes; opengwas/ogld#41 measured 66% wasted at depth 8 | HTTP stores only, with the waste counted |

**Recommendation.** A was built and measured first, then B, whose pool **is the reader's own and
capped at four component workers** for opengwas/ogld#71 (fixed on opengwas/ogld#65's evidence; §1). There is no caller-facing
setting: `opengwasdb.ld.store.reader.COMPONENT_WORKERS_MAX` is both the default and the cap, the reader holds
it, and a test reaches it through a private seam. Four was where G = 8 and G = 16 were fastest,
1.051 and 1.067 s cold; at eight, cold LD regresses to 5.753 s (G = 8), 5.109 s (G = 16) and
1.913 s (G = 4), because the per-lead Python work holds the GIL. opengwas/ogld#71 measured the same answer at 1,
2 and 8 requested workers, one Partner Rows row per lead, and a damaged frame failing the whole
request. Reach for C only on evidence.

**What will bound latency next.** On GCST90476460 today, ALID resolution (`rows_of`) takes
2.0–3.2 s, and store open, score and loci 3.4 s together. Those seconds become most of a
sub-second clump's time. They are out of scope here. They are opengwas/ogld#50's mmap lookup question, raised
as its own issue, and **opengwas/ogld#81** carries the follow-up the owner opened when they accepted opengwas/ogld#71
without the < 1 s target.

## 10. Validation

These rules replaced the six `index.*` rules and `far.index` in `ld-store.md` §12, which opengwas/ogld#68 added
beside them and opengwas/ogld#72 removed, leaving the ten below. Each reads frames in bounded blocks, never a whole chromosome. Each
has a breaker in `tests/ld/contract/test_store.py`.

| rule | checks |
|---|---|
| `partners.files` | every chromosome of the Variant Set has a `partners/<c>/` group with both arrays; its `rows` attribute is n_c, `frames` is ⌈n_c / G⌉, `frames_idx` has F + 1 ascending entries from 0 to `frames_bytes`, and both SHA-256s match |
| `partners.frames` | every frame decompresses with a valid checksum, its k is the rows due, and its payload is exactly 4 + 4k + 6E bytes |
| `partners.rows` | within a row, partners are strictly ascending, in [0, n_c), and never the row; Σ E equals the group's `entries` |
| `partners.codes` | every code is one of the store's encoding (as `near.codes`) |
| `partners.distance` | every entry lies within `far_bp` |
| `partners.threshold` | \|code\| ≥ `partner_code_floor` within `near_bp`, ≥ `far_code_floor` beyond it |
| `partners.symmetric` | (l, m, c) is an entry exactly when (m, l, c) is (the 64-bit hash-sum method of the retired `index.symmetric`) |
| `partners.near` | every entry within `near_bp` carries the Near Band's code, and every Near Band code at the Partner Floor is an entry (the retired `index.near`'s method, at the new floor) |
| `partners.far` | the entries beyond `near_bp`, in each pair's earlier row, are exactly the Far Pairs, with the same codes (the retired `far.index`, read against the Partner Rows) |
| `provenance.partner_threshold` | t_partner ≤ t_far; `partner_code_floor` recomputes from t_partner at the recorded scale |

`partners.near` and `partners.far` together prove §3's "copies, by design". With
`partners.symmetric`, they prove the component holds every pair it should and no other.

## 11. Size

**Measured on the published HGDP+1kGP EUR k = 10 store, 18 November 2026** (opengwas/ogld#65), genome-wide at
the Partner Floor and the Far Pairs in both directions. Bytes are the `frames` and
`frames_idx` arrays with the group's attributes, in decimal units; the net column subtracts EUR's
Proxy Index, 9,381,954,206 bytes.

| G | EUR genome bytes | net of the index | frames | best cold LD on GCST90476460 |
|---|---:|---:|---:|---:|
| 4 | 18,506,168,576 | 9,124,214,370 | 2,146,730 | 1.169 s, 2 workers |
| 8 | 16,676,561,335 | 7,294,607,129 | 1,073,370 | 1.051 s, 4 workers |
| **16** | **14,764,657,158** | **5,382,702,952** | 536,692 | **1.067 s, 4 workers** |
| 64 | 11,820,856,668 | 2,438,902,462 | 134,180 | 1.670 s, 8 workers |

- **G = 16** (§1, §5.2). It is the smallest of the configurations that are equally fast: G = 8's
  12.95% space premium buys 43.8% fewer demanded bytes, 57.2 MB against 101.8 MB, and a cold LD
  time 16 ms lower, inside the 0.9468–1.8849 s spread of its three samples; eight workers regress
  at every G.
- **The genome keeps 7.268% of Near Band pairs at the Partner Floor, against chr22's 5.027%:**
  6,077,753,140 near pairs plus 132,924,736 Far Pairs, 12,421,355,752 directed entries. A
  projection from chr22 alone therefore understated EUR by 39%.
- **The other four panels are measured by opengwas/ogld#73**, when it derives them. The owner's scaling of this
  measurement puts the five panels at roughly **+53 GB net at G = 64**, against the projection's
  +33 GB (+6%); opengwas/ogld#73 returns to the owner before dropping an index whose panel exceeds its
  projection by more than 25%.
- **AMR is the panel to watch.** At n = 341, 7.2% of unlinked pairs pass r² 0.01 by chance, so
  about 44% of AMR's Partner Rows would be sampling noise at that floor, and chr22 keeps 16.3% of
  its band there against EUR's 5.0%. A store records its own t_partner, at most its t_far; the
  five HGDP+1kGP builds use 0.01.
- **Why not lower.** At r² > 0.001, 46.2% of EUR chr22's band passes, against 41.9% expected
  from sampling alone, so about 90% of those pairs are noise. A sparse row there would be larger
  than the dense band (6 bytes × 46% > 2 bytes). D2 refuses such requests. On UK Biobank EUR
  (n = 49,766) the null share at 0.001 is about 10⁻¹², and t_partner = t_far = 0.001 there.

The projections this section first carried — 9.0 GB for EUR, and 75.0 GB over the five panels
beside the index, 45.2 GB of that the index itself — came from chr22 with a simpler encoding. The
measured share is why EUR's came out 39% low. The measurement scripts and their JSON are
`proto/partner-rows@912a9fa:prototypes/partner-rows/derive.py` and
`proto/partner-rows@912a9fa:prototypes/partner-rows/genome_sweep.py`, with their results under
`proto/partner-rows@912a9fa:prototypes/partner-rows/results/`;
`proto/partner-rows@912a9fa:prototypes/partner-rows/reframe.py` (G = 4 and 8, by losslessly
subdividing the G = 16 frames) and
`proto/partner-rows@912a9fa:prototypes/partner-rows/query_bench.py` (the timings).

## 12. Alternatives weighed

- **Keep the Proxy Index beside the Partner Rows** (against D1): +11.8–14.8 GB rather than
  +2.4–5.4 GB net, and two components holding the same pairs, each with its own writer and
  validation rules.
- **Keep the Near Band path below t_partner** (against D2): answers r² 0.001 clumps, but keeps
  the clump warmer, the column reads and clump's coupling to the chunk cache. These are what made
  clumping slow, and on HGDP+1kGP they would serve thresholds where 90% of passing pairs are
  noise.
- **One rule for every store, min(0.01, t_far)** (against D3): simpler, but it fixes AMR's cost
  at the projected +38%. A per-dataset t_partner lets a store's build choose a higher one, up to
  its t_far, for a panel whose chance LD is a larger share of its rows.
- **Plain files for the frames** (against the refined D4): a directory of plain files serves a
  reader, but it makes zarr's `members()` warn that `partners` is not a component of the
  hierarchy, as opengwas/ogld#65 measured (ADR 0097). One chunk per array keeps the frame read one read; a
  Zarr array sharded with a chunk per frame would resolve a shard index first, which is two reads
  rather than one. So the two arrays are Zarr's, with one chunk each.
- **Fold into 0.1.0** (against D5): no version bump, but a 0.1.0 reader could not tell a store
  with Partner Rows from one without.
- **Dense symmetric rows** (opengwas/ogld#41, opengwas/ogld#50; epic opengwas/ogld#45's rejected arm): fast, but about 181 GB for the
  79.3 GB EUR store. The Partner Rows are the same idea, thresholded: 11.8–14.8 GB for EUR,
  measured.
- **Framed triangle rows** (opengwas/ogld#45's other rejected arm): a lead's upstream partners still fan out
  across thousands of earlier rows. Symmetric rows have no upstream fan-out.

## 13. Plan

Epic [#64](https://github.com/opengwas/ogld/issues/64) carries this design's plan and acceptance
criteria: opengwas/ogld#67 writes the rows from `StoreWriter`, opengwas/ogld#69 derives them for the existing stores, opengwas/ogld#70
reads them, opengwas/ogld#71 moves `clump` and `proxies` onto them, opengwas/ogld#68 validates them, opengwas/ogld#72 removes the Proxy
Index (done: the component, its writer, its arrays, its six rules and `far.index` are gone, and
`clump`'s Near Band reads with them), and opengwas/ogld#73 rolls them out to the five HGDP+1kGP stores. This document specifies the format those
tickets build, and does not restate their acceptance.
