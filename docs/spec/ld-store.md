# LD Store Format Specification

Status: draft
Format version described: `0.2.0`, which the code writes (#67); `0.1.0` is the version before the
Partner Rows, which a reader still reads (§13).

This document defines a valid **LD Store**: the arrays, their dtypes and Zarr layout, how r is
quantised, the provenance an LD Store carries, what a reader answers for a pair, what the four
queries return, and when a request is refused. The decisions behind it are in ADR 0001
(`docs/adr/0001-near-band-and-thresholded-far-pairs.md`), including its clarification of
7 November 2026, each store's encoding of r in ADR 0004
(`docs/adr/0004-per-store-encoding.md`), the Far Pairs beside the Proxy Index in ADR 0007
(`docs/adr/0007-far-pairs-beside-the-proxy-index.md`), the ALID search index in ADR 0010
(`docs/adr/0010-alid-search-index.md`), and the Partner Rows in ADR 0011
(`docs/adr/0011-partner-rows-replace-the-proxy-index.md`) with their design in
`docs/spec/partner-rows.md`; the build inputs are in `docs/spec/build-inputs.md` (ADR
0002); the terms are in
`CONTEXT.md`. **ALID**, its **A1** and **Reference Assembly** are the engine's
(opengwas/opengwasdb `CONTEXT.md`, _Variant Identity_), not redefined here. The engine's glossary
says an ALID is formed after "upper-casing, trimming and left alignment"; its code at `1ab2b09`
only upper-cases, requires A, C, G, T and sorts the two alleles, and so does ogld (ADR 0003).

The numbers that chose the layout and the codec come from prototypes run on real HGDP+1kGP
chr22 genotypes. They are on the prototype branch `proto/1-ld-store-layout`, and each is cited as
`proto/1-ld-store-layout@65f94ca:<path>`, so a citation stays fixed if the branch moves.

The semantics of §3, §8 and §9 are implemented in `ogld/format/`, and `tests/test_format.py`
holds that code to this document's tables.

Normative language:

- **MUST** indicates a requirement for a valid LD Store.
- **SHOULD** indicates a recommended default that implementations may override with documented reason.
- **MAY** indicates an optional feature.

![The pair matrix of HGDP+1kGP EUR chr22:20–32 Mb, to scale: the Near Band, the real Far Pairs and the Not Measured region](figures/ld-store-format.svg)

_Figure 1. The pairs an LD Store holds, drawn to scale from the prototype's EUR chr22 slice
(`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/s10_figures.py`). The Near Band
holds each pair once, in the row of its earlier variant: one variant's downstream partners are
its row, and its upstream partners are its column, one element in each earlier row. The Proxy
Index (§5) holds every pair drawn as a Far Pair, and every Near Band pair at r̂² ≥ t_far, in
both variants' rows, so a variant's partners at r̂² ≥ t_far on both sides are one read. The Far
Pairs (§5) hold the pairs drawn as Far Pairs once more, each in its earlier variant's row, so a
matrix reads them without the index._

## 1. Envelope

An LD Store is one directory holding one Zarr v3 hierarchy, with its group metadata at the root.
Its arrays are written with the `sharding_indexed` codec (§6). There MUST be exactly one LD Store
per (**Source Panel**, **Reference Assembly**) (ADR 0001, decision 1).

```text
<ld store>/                  Zarr v3 group; attribute "ld_store": format version, encoding, provenance (§7)
├── variants/                the Variant Set, one row per variant, in genomic order (§2)
│   ├── position             int32
│   ├── alid                 string (UTF-8)
│   ├── a1                   string (UTF-8)
│   ├── a2                   string (UTF-8)
│   ├── a1_frequency         float32
│   └── alid_index/          the ALID search index (§2; ADR 0010)
│       ├── alid             uint8, 32 bytes an entry: the key, NUL right-padded
│       └── row              int32, the Variant Set row of each entry
├── near/<chrom>/            the Near Band of one chromosome (§4)
│   ├── indptr               int64, rows + 1
│   └── r                    int16, every row's codes, row after row
├── index/<chrom>/           the Proxy Index of one chromosome (§5; removed by #72)
│   ├── indptr               int64, rows + 1
│   ├── partner              int32, chromosome-local row of the partner, either side
│   └── r                    int16
├── far/<chrom>/             the Far Pairs of one chromosome (§5)
│   ├── indptr               int64, rows + 1
│   ├── partner              int32, chromosome-local row of the later variant
│   └── r                    int16
└── partners/<chrom>/        the Partner Rows of one chromosome (§5; 0.2.0)
    ├── frames               uint8, one chunk: the chromosome's zstd frames, back to back
    └── frames_idx           uint64, F + 1: the byte offsets of those frames
```

`<chrom>` is the engine's canonical chromosome label (`1`–`22`, `X`, `Y`, `MT`, no `chr` prefix).
The format allows every one of them. For now a builder builds autosomes only and refuses a record
on X, Y, a PAR or MT, because the diploid dosage model has no rule for a male's hemizygous
genotypes (ADR 0002, decision 9). Each chromosome of the Variant Set has its own `near/<chrom>`,
`index/<chrom>`, `far/<chrom>` and, from 0.2.0, `partners/<chrom>` groups. No component crosses a
chromosome, so per-chromosome
arrays lose nothing. They also let a builder
write chromosomes in parallel: a Zarr shard is written whole, and no shard can then span two
chromosomes.

**The Partner Rows** (format 0.2.0; ADR 0011, `docs/spec/partner-rows.md`) are
`partners/<chrom>/`, a Zarr v3 group per chromosome. It holds that chromosome's Partner Rows:
for each variant, every partner within `far_bp` on either side whose Stored r reaches the Partner
Floor within `near_bp` and the far code floor beyond it, in both variants' rows with the same code
(§3 of `docs/spec/partner-rows.md`). The group is a node in this one hierarchy, so a reader that
lists the store sees it, and it has two arrays of one chunk each: `frames`, `uint8` with the
`bytes` codec alone and no compressor, whose chunk file is the chromosome's zstd frames back to
back, and `frames_idx`, `uint64`, their F + 1 byte offsets. The group's attributes are what a
manifest beside them would carry: the rows, the rows to a frame, the frame count, the
entry count, the chunk files' sizes and SHA-256s, and the zstd level. A reader resolves the chunk
file's path from the array's metadata, never by assuming one. `clump` and `proxies` read the
Partner Rows from #71, and the Proxy Index goes with them (#72). A 0.1.0 store has no `partners/`.

An LD Store MUST NOT contain individual-level data (ADR 0001, decision 5): no genotypes, no sample
identifiers, no per-sample values of any kind. A reader ignores any array or group it does not know.

## 2. Variant table (`variants/`)

- Rows MUST be in the engine's variant order: chromosome (`1`–`22`, `X`, `Y`, `MT`), then position
  ascending, then ALID ascending (bytewise) at one position.
- The `variants` group attribute `chromosomes` MUST list, in that order, one
  `{"name": <chrom>, "row_start": int, "row_stop": int}` per chromosome. The ranges are
  contiguous and cover rows `[0, n_variants)`. A row's **chromosome-local row** is its row minus
  its chromosome's `row_start`; `near/`, `index/` and `far/` use local rows.
- `position` is the 1-based position on the Reference Assembly, `int32`.
- `alid` is the engine's ALID, exactly as the engine's identity function forms it. ogld holds a copy
  of that function, tested against the engine's own answers (ADR 0003); this format stores what it forms.
  `a1` and `a2` are the variant's two alleles as the engine's _Variant Identity_ spells them, A1
  first, in full (opengwas/opengwasdb ADR 0017 keeps the full alleles). All three are Zarr v3
  variable-length UTF-8 strings (`string` data type), and none is ever truncated.
- `a1_frequency` is A1's frequency in the Source Panel's samples, after the variant filters,
  `float32`.

**The ALID search index** (`variants/alid_index/`, owner, 15 November 2026; ADR 0010). It holds
the Variant Set's ALIDs again, as fixed-width keys a binary search can run over:
`alid` is one 32-byte key an entry — the ALID's UTF-8 bytes, NUL right-padded — and `row` the
Variant Set row of that entry. The entries are sorted by key, bytewise; two ALIDs of one length
differ only in their own bytes, and padding with NUL sorts an ALID before every ALID it is a
prefix of, which is the order the engine's own index has. It holds every ALID whose UTF-8 form
is at most 32 bytes, which is the engine's rule and its width (`is_indexable_alid`; ADR 0010). An ALID wider than the slot MUST be left out of the
index and counted in `counts.alids_left_out` (§7), never cast into it: the cast truncates, and
two indels at one position that agree over the slot then share one key, so a lookup by either
returns whichever row sorts first — one variant answering to another's name, with nothing to
show it (engine issue #127, 342 such variants on the published finngen-r13 release). A left-out
ALID stays reachable, by the exact scan this section used to describe for every ALID.
`alid_index.rows`, `alid_index.order`, `alid_index.keys` and `alid_index.count` hold the index to
the table it indexes (§12), entry by entry: an entry that names a row whose ALID is wider than the
slot, or whose key is not that row's full ALID, is a store a reader must not trust.

**Why ALID is stored this way.** The variant table keeps ALID as Zarr's variable-length UTF-8
strings, and the index beside it is bytes, because the two reads are different reads. A query
naming variants finds them by ALID, one search over keys whose bytes are the ALIDs; it never
parses a position or decodes a string. A query naming rows returns the table's ALIDs, and a
matrix or a clump returns them for the rows it chose. The index is what the strings are not: an
array of fixed-width keys, sorted, that a binary search — or numpy's — runs over. Reading ALIDs
as strings cost 59 s of the 94 s that finding the variants of #403's 37 Analyses took, through
Zarr's variable-length path, and the keys cost a fraction of that (`docs/benchmarks/alid-index.md`).

Both are stored, not derived from the other columns, so a reader never reimplements identity,
and validation checks the table against (`chromosome`, `position`, `a1`, `a2`) with the identity
function #3 settles, and the index against the table (§12). Keeping both as Zarr arrays leaves
the whole LD Store one Zarr hierarchy that a client reads over HTTP. A reader in another
language needs Zarr's variable-length string extension for the table and nothing but a byte
array for the index, which is why the key is `uint8` and not a fixed-length bytes type: Zarr v3
has no `|S32` data type of its own, and zarr-python 3.4.2 writes one as `null_terminated_bytes`,
an extension it marks unstable and that other implementations do not read (ADR 0010).
zarr-python 3.4.1 writes and reads these strings inside sharded arrays (the prototype's stores,
`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/s05_build_stores.py`).

A variant outside the Variant Set is **Absent** (§8). A string that is not an ALID names no
variant, so it is Absent too.

## 3. Orientation and quantisation

**Orientation.** r(i, j) is the correlation, after **PC Adjustment**, of the A1 dosages of
variants i and j, where A1 is each variant's engine A1. A reader that flips one variant's alleles
flips the sign of every r it is in.

**Quantisation.** r is stored as an `int16` code at the store's scale s, a whole number from 1
to 32767 that the store records as `encoding.r_scale` (§7; ADR 0004):

```text
code = clamp(rint(r × s), −s, +s)      rint rounds half to even (IEEE 754, numpy.rint)
r̂    = code / s                          the Stored r a reader gets back
```

| code | meaning |
|---|---|
| −32768 | reserved: never a correlation. It is every r array's fill value, and a valid LD Store holds none |
| −s … +s | r̂ = code / s; r = ±1 is ±s exactly |
| any other | past the scale: never a correlation, since \|r̂\| would exceed 1. A valid LD Store holds none |

**Each store's scale is its own** (owner, 10 November 2026; ADR 0004). The step is 1/s and the
quantisation error at most half of it, 1/(2s):

| LD Store | encoding | step | max error in r |
|---|---|---|---|
| the five HGDP+1kGP panels | `int16` × 511 | 1.96×10⁻³ | 9.8×10⁻⁴ |
| UK Biobank EUR | `int16` × 2047 | 4.89×10⁻⁴ | 2.4×10⁻⁴ |
| the container's widest | `int16` × 32767 | 3.05×10⁻⁵ | 1.53×10⁻⁵ |

A build takes its scale as an input (`BuildInputs.r_scale`, `docs/spec/build-inputs.md` §1), and
there is no default: a build that was not given one is refused. The issue's measurement on real
HGDP+1kGP EUR chr22 chose the widths: at × 511 the error is 0.03 of r's sampling error at
n = 771, all of EUR's v2 members (0.025 at the 665 a build keeps), and the Near Band costs 1.70×
fewer bits a pair than at × 32767; a whole chr22 build at k = 10 measured 1.68× (§6).

- **Every encoding shares one container**: `int16` codes, −32768 reserved, half-to-even rounding.
  The scale is symmetric, so negating a code to flip an allele never overflows, and r = ±1 is
  exact. At × 32767 it is magenpy's `int16` scale (1 / `iinfo(int16).max`). An `int8` container
  is not built (ADR 0004).
- **The clamp is for float error only.** A computed r that is not finite, or whose magnitude
  exceeds 1 by more than 10⁻⁶, is a builder fault, and quantising refuses it rather than store a
  plausible ±1.
- **The reserved code is the fill value, so an unwritten chunk can never read as r = 0.** In a
  Zarr array, a chunk that was never written reads as the fill value. With a fill value of 0, a
  build that lost a chunk would hand readers a run of plausible zeros, which ADR 0001 forbids.
  With −32768 the same loss reads as a code that validation (§12) and every reader reject, at
  every scale.
- **There is no "undefined" code.** r is undefined only for a variant with no genotype variance
  after PC Adjustment. MAC ≥ 20 does not exclude this: a site where every called sample is
  heterozygous has none. The prototype found 9 such sites in the AFR chr22 slice and 2 each in
  EAS and SAS (`variants_dropped_zero_variance` in
  `proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/r_AFR.json`,
  `proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/r_EAS.json` and
  `proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/r_SAS.json`). A builder
  MUST drop such variants from the Variant Set and count them in provenance (`variants_dropped`,
  §7). Every stored pair then has a defined r, and the answer table (§8) needs no fourth row.
- The scale, dtype, reserved code and rounding are recorded in `encoding` (§7). A reader MUST
  refuse an LD Store whose `encoding` it does not implement, and MUST decode every code at the
  scale its own store records: one process reads stores of different scales side by side
  (`ogld.format.REncoding` is the one place that quantises and decodes).

**Thresholds are judged on the Stored r.** A pair reaches an r² threshold t when r̂² ≥ t,
computed in float64. Every rule below uses this comparison: the Proxy Index's membership (§5),
the Bound (§8), and proxies (§9). Clumping is the one exception: it follows `plink2 --clump`,
which joins a pair at r̂² > r2 and d < W, not ≥ and ≤ (§9; owner, 10 November 2026). So a reader's view is self-consistent. But the
computed r and r̂ differ by up to half the store's step, so a pair whose computed r² lies within
about 2|r| × 1/(2s) of a threshold can fall on either side of it. At × 32767, a computed r² of
0.0099994 is stored as code 3277, which reads back as r̂² = 0.0100018 and reaches 0.01 (the
rounding walkthrough of `proto/1-ld-store-layout@65f94ca:prototypes/answer-semantics.html`). At
× 511 the band is 64 times wider: a computed r² of 0.0101, above 0.01, is stored as code 51, which
reads back as r̂² = 0.0099609 and does not reach it, while 0.0102 is code 52, r̂² = 0.0103554.
ogld#6 lists any pair like this, at the store's own step, when it compares clumping with
`plink2 --clump`.

**The far code floor.** `far_code_floor` is the smallest integer c ≥ 0 with (c / s)² ≥ t_far in
float64, at the store's scale s. A pair is in the Proxy Index exactly when it is within `far_bp`
and |code| ≥ `far_code_floor`, so a pair more than `near_bp` apart is a **Far Pair** exactly then.
It is recorded in provenance, and validation recomputes it at the recorded scale.

**The Partner Floor** is the same rule at the Partner Threshold: `partner_code_floor`, the
smallest integer c ≥ 0 with (c / s)² ≥ t_partner in float64, at the store's scale s, 52 at × 511
for t_partner = 0.01. From 0.2.0 a pair is in the Partner Rows exactly when it is within `far_bp`
and |code| ≥ `partner_code_floor` within `near_bp` or ≥ `far_code_floor` beyond it (§5, §7).

**Where a pair at exactly t_far lands.** Such a pair's code is rint(√t_far × s). It is a Far
Pair when that reaches the floor, and reads as the Bound when it does not. In true r², the cut
between the two is ((`far_code_floor` − ½) / s)². That cut lies within about √t_far / s of
t_far, on either side. So a pair whose true r² is up to that much above t_far can read
"r² < t_far". The table gives each panel at its own scale, on the samples each build keeps: the
unrelated ones (ADR 0002, decision 11), EUR 665, AFR 608, EAS 724, SAS 672 and AMR 341. The Test
Panel keeps all 771 EUR members (owner's ruling), so the Test Store has a row of its own. The
method is the prototype's
(`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/t_far_edge.tsv`, on the
gnomAD v3.1.2 counts, at × 32767); each value was computed in 50-digit decimal arithmetic outside
ogld, and `tests/test_format.py` holds the code to every row.

| LD Store | scale | n | k | t_far | √t_far × s | far code floor | a pair at exactly t_far | true r² cut |
|---|---|---|---|---|---|---|---|---|
| UK Biobank EUR | 2047 | 49,766 | 0 or 10 | 0.001 | 64.73 → 65 | 65 | stored | 0.0009929 |
| HGDP+1kGP EUR | 511 | 665 | 0 | 0.0313855 | 90.53 → 91 | 91 | stored | 0.0313657 |
| HGDP+1kGP EUR | 511 | 665 | 10 | 0.0318654 | 91.22 → 91 | 92 | **the Bound** | 0.0320627 |
| HGDP+1kGP AFR | 511 | 608 | 0 | 0.0343328 | 94.68 → 95 | 95 | stored | 0.0341997 |
| HGDP+1kGP AFR | 511 | 608 | 10 | 0.0349079 | 95.47 → 95 | 96 | **the Bound** | 0.0349273 |
| HGDP+1kGP EAS | 511 | 724 | 0 | 0.0288243 | 86.76 → 87 | 87 | stored | 0.0286543 |
| HGDP+1kGP EAS | 511 | 724 | 10 | 0.0292286 | 87.36 → 87 | 88 | **the Bound** | 0.0293207 |
| HGDP+1kGP SAS | 511 | 672 | 0 | 0.0310581 | 90.06 → 90 | 91 | **the Bound** | 0.0313657 |
| HGDP+1kGP SAS | 511 | 672 | 10 | 0.0315280 | 90.73 → 91 | 91 | stored | 0.0313657 |
| HGDP+1kGP AMR | 511 | 341 | 0 | 0.0612941 | 126.51 → 127 | 127 | stored | 0.0612829 |
| HGDP+1kGP AMR | 511 | 341 | 10 | 0.0631515 | 128.41 → 128 | 129 | **the Bound** | 0.0632360 |
| the Test Store | 511 | 771 | 10 | 0.0274211 | 84.62 → 85 | 85 | stored | 0.0273446 |

At k = 10, the default, a pair at exactly t_far is a Far Pair only on HGDP+1kGP SAS, and so is a
pair a little below it, down to the cut: 0.0313657, against t_far 0.0315280. On EUR, AFR, EAS and
AMR it reads as the Bound: a true r² up to 0.0320627 on EUR, 0.0349273 on AFR, 0.0293207 on EAS
and 0.0632360 on AMR reads "r² < t_far". On UK Biobank EUR a pair at exactly t_far is stored, and
so is one down to 0.0009929. At × 32767 the same rule gives floors of 1037 (UK Biobank EUR) and
5850 (HGDP+1kGP EUR, k = 10, which the × 32767 chr22 build of §6 records), and cuts within
8×10⁻⁶ of t_far; `tests/test_format.py` keeps those floors too. The walkthrough "A pair at
exactly t_far" in `proto/1-ld-store-layout@65f94ca:prototypes/answer-semantics.html` steps
through these cases at × 32767 on the gnomAD v3.1.2 counts.

**t_far's k counts covariates.** t_far = max(0.001, q / (n − k − 1)), where k is the number of
covariates regressed out, named in `adjustment.covariates` (§7); the 1 is the intercept. For a
build that is the number of PCs actually used, and 0 for an unadjusted build. For an imported LD
matrix it is the source's declared count of every covariate it adjusted for: Pan-UKB's 10 PCs
and five age and sex terms make k = 15.

## 4. Near Band (`near/<chrom>/`)

For chromosome c with n_c rows, local rows l = 0 … n_c − 1 and positions p_l:

- The **distance** of a pair is d = |p_m − p_l| in base pairs, on one chromosome. Two variants at
  one position (d = 0, such as the records of a split multi-allelic site) are a pair.
- Each pair is stored once, in the row of its earlier variant: row l holds every m > l with
  d ≤ `near_bp`, in ascending m. `near_bp` is 3,000,000 (ADR 0001, decision 2; §7). Its length is
  len_l = |{m > l : p_m − p_l ≤ near_bp}|. A row stops at its chromosome's last variant.
- `indptr` has n_c + 1 entries: indptr[0] = 0 and indptr[l + 1] = indptr[l] + len_l.
- `r` holds the rows back to back: r(l, m) is at `r[indptr[l] + (m − l − 1)]`.
- Columns are implicit, so there is no column array. `indptr` can be derived from `position`. It
  is stored so a reader needs no position scan, and validation checks it against the positions (§12).
- The diagonal, r(l, l) = 1, is not stored.
- With k = 0, the Near Band is the unadjusted r that `plink2 --r-unphased` computes, to within the
  quantisation step; ogld#5 tests that equivalence.

`indptr` is `int64` because one chromosome's band exceeds 2³¹ codes. AFR chr2 at MAC ≥ 20 holds
about 3.7×10¹⁰ (`largest_chromosome_mac20` in
`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/s08_meta.json`).

## 5. Proxy Index, Far Pairs and Partner Rows (`index/<chrom>/`, `far/<chrom>/`, `partners/<chrom>/`)

The Proxy Index (removed by #72, when the Partner Rows take its place) holds, for each variant,
every partner within `far_bp` on either side whose Stored r reaches t_far (owner, 7 November 2026;
§11):

- Row l holds every m ≠ l on the chromosome with |p_m − p_l| ≤ `far_bp` and |code(l, m)| ≥
  `far_code_floor` (§3), in ascending m. `far_bp` is the distance the store measured: 10,000,000
  for a build, and at most that for any store (§7).
- **Both directions.** A pair is in row l and in row m, with the same code. So one row is a
  variant's partners at r̂² ≥ t_far on both sides, read in one piece.
- `indptr` (`int64`, n_c + 1 entries) delimits row l's entries. `partner` holds their local rows,
  strictly ascending within a row and never l. `r` holds their codes.
- `partner` is `int32`: the largest chromosome's Variant Set at MAC ≥ 20 is about 1.7×10⁶ rows
  (AFR chr2, the same `proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/s08_meta.json`), far inside 2³¹.
- **The Far Pairs are the index's entries more than `near_bp` apart.** A pair with
  `near_bp` < d ≤ `far_bp` is a Far Pair exactly when it is in the index. One that is not has
  r̂² < t_far: its answer is the **Bound** (§8).
- **Its entries within `near_bp` are copies of Near Band codes, by design,** thresholded at t_far
  and transposed for each pair's later variant. They are how the index answers a variant's
  upstream partners without the Near Band's column read, so `proxies` at r2 ≥ t_far reads one
  row. `matrix` reads those pairs from the Near Band and `proxies` from the index, and
  `index.near` holds the two to one code per pair (§12).

**The Far Pairs** (`far/<chrom>/`) hold the index's pairs beyond `near_bp` again, once each, in
the earlier variant's row, as the Near Band holds its pairs (owner, 12 November 2026; ADR 0007):

- Row l holds every m > l with `near_bp` < p_m − p_l ≤ `far_bp` and |code(l, m)| ≥
  `far_code_floor` (§3), in ascending m: exactly row l's index entries above l and beyond
  `near_bp`, with their codes.
- `indptr` (`int64`, n_c + 1 entries) delimits row l's Far Pairs. `partner` holds their local
  rows m, strictly ascending within a row and each above it. `r` holds their codes.
- **They are copies of index entries, by design,** as the index's near half copies the band.
  A Far Pairs row holds the few pairs a matrix needs beyond `near_bp`, where an index row holds
  every partner at r̂² ≥ t_far within `far_bp`: on HGDP+1kGP EUR chr22, 101,886 Far Pairs in 7
  inner chunks of each array, against about 380 index entries a row (§11). So `matrix`,
  `region` and every pair answer read the Far Pairs beyond `near_bp`, and `proxies` and `clump`
  read the index (§10). `far.index` holds the two to one code per pair (§12).

At k = 0, 9.4×10⁻⁵ (EUR) to 2.3×10⁻³ (AMR) of chr22's 3–10 Mb pairs are Far Pairs
(`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/panel_sizes_inner262144.tsv`).
1.0% (AFR) to 2.4% (AMR) of Near Band pairs reach t_far
(`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/upstream_alternatives_sizes_inner262144.tsv`).

**The Partner Rows** (`partners/<chrom>/`; format 0.2.0, written by #67, derived by #69) hold, for
each variant, every partner within `far_bp` on either side whose Stored r reaches the **Partner
Floor** (§3) within `near_bp` and the far code floor beyond it, in both variants' rows with the
same code. They are the Proxy Index's rule at t_partner instead of t_far, so they hold every index
entry: t_partner is never above t_far, and every pair within `near_bp` at the far code floor is at
the Partner Floor too. Rows are grouped 16 to a zstd frame, and the group, its two arrays and the
frame bytes are `docs/spec/partner-rows.md` §5. Every code is a copy of one the store already
holds, a Near Band code within `near_bp` or a Far Pair beyond it, in a second layout chosen so a
lead is one positional read; `partners.near` and `partners.far` (§12) hold the copies to one code
per pair. `clump` and `proxies` read them from #71; until then they read the Proxy Index below,
which #72 removes.

## 6. Physical layout

Every array but the Partner Rows' is Zarr v3 with `sharding_indexed`. Inside a shard, each kind of
array has one codec chain. The shard index is `bytes` then `crc32c`, at the end of the shard
(zarr-python's default).

| array kind | codecs inside the shard |
|---|---|
| numeric: `position`, `a1_frequency`, every `indptr`, `r` and `partner`, and the ALID index's `alid` and `row` | `bytes` (little-endian), then `zstd` level 3 |
| string: `alid`, `a1`, `a2` | `vlen-utf8`, then `zstd` level 0, which zstd reads as its default level, 3 (zarr-python's default for strings, and what the prototype's stores used) |

A shard MUST be a whole multiple of its inner chunk. An array shorter than its shard has one
shard, clipped to the smallest whole number of inner chunks that covers it, as in the engine's
spec §10a. An inner chunk longer than its array is clipped to the array.

| array | dtype | shape | inner chunk | shard | fill value |
|---|---|---|---|---|---|
| `variants/position` | `int32` | (V,) | 16,384 | 1,048,576 | 0 |
| `variants/alid`, `a1`, `a2` | `string` | (V,) | 16,384 | 1,048,576 | `""` |
| `variants/a1_frequency` | `float32` | (V,) | 16,384 | 1,048,576 | NaN |
| `variants/alid_index/alid` | `uint8` | (32N,) | **262,144** | **16,777,216** | 0 |
| `variants/alid_index/row` | `int32` | (N,) | **262,144** | **1,048,576** | −1 |
| `near/<chrom>/indptr` | `int64` | (n_c + 1,) | 16,384 | 1,048,576 | −1 |
| `near/<chrom>/r` | `int16` | (indptr[n_c],) | **262,144** | **67,108,864** | −32768 |
| `index/<chrom>/indptr` | `int64` | (n_c + 1,) | 16,384 | 1,048,576 | −1 |
| `index/<chrom>/partner` | `int32` | (index_c,) | 16,384 | 1,048,576 | −1 |
| `index/<chrom>/r` | `int16` | (index_c,) | 16,384 | 1,048,576 | −32768 |
| `far/<chrom>/indptr` | `int64` | (n_c + 1,) | 16,384 | 1,048,576 | −1 |
| `far/<chrom>/partner` | `int32` | (far_c,) | 16,384 | 1,048,576 | −1 |
| `far/<chrom>/r` | `int16` | (far_c,) | 16,384 | 1,048,576 | −32768 |

V is the Variant Set's size, N the ALID index's entries (V less `counts.alids_left_out`, §7), n_c a
chromosome's rows, index_c its Proxy Index entries and far_c its Far Pairs. An `r`
array is `int16` with fill value −32768 at every scale a store may record (§3): the encoding
changes the codes, never the array's dtype or fill. Every
fill value is one a valid LD Store never holds, so a chunk that was never written cannot read as
data. This table describes how a builder writes an LD Store. Each array's own Zarr metadata
records its chunk, shard and codecs, and readers MUST NOT depend on them: a store re-sharded for
serving (ogld#9) reads the same. A store may record other fill values, but each MUST be one a
valid store never holds in that array: −32768 in an `r` array, a negative number in an `indptr`,
`partner` or `row`, a position ≤ 0, `""` in a string column, and a frequency outside (0, 1) or NaN. A reader
refuses an array whose fill value is not, and `envelope.fill` fails it (§12).

The ALID index's keys are judged 32 bytes at a time, not one byte at a time, and that is why the
array's fill value MUST be 0: a padding byte is 0, so no single byte is one a store never holds,
but a whole key of 0 bytes is one it never holds — no ALID is empty. So a chunk the store lost
reads a whole key of NULs, which the reader refuses, naming the array, rather than a run of empty
ALIDs; and `variants/alid_index/alid`'s fill value being anything else would let a lost chunk read
as keys (`refused_if_lost`, §12's `envelope.fill`).

The Partner Rows' two arrays are the exception to this section: each has one chunk of the whole
array, no shard index, and `bytes` alone for `frames` (§1, `docs/spec/partner-rows.md` §5.1).
Their chunk files' sizes and SHA-256s are the group's attributes, so `envelope.fill` and the rest
of this section's rules apply to them only where §12's Partner Rows rules say so.

**How a store is written** (ogld#17, ADR 0005). Every producer, a build or an import of a
published r, writes through `StoreWriter`, and the store makes its arrays from the r it is given,
block of rows by block of rows, in row order. Every pair within a row's `far_bp` is judged as it
is written. A value that is NaN, infinite, or past ±1 beyond §3's float allowance is refused,
naming the pair. So a pair the producer did not measure is never stored, and never read as a Bound
because it was taken to lie below the floor. A chromosome's Near Band `r` and `indptr` are created at its first
block, since their lengths follow from the positions (§4). Its codes are written a whole shard at
a time, the only write zarr makes without reading a shard back, and less than one shard is carried
to the next block. A row of the Proxy Index is complete when its own block arrives: its partners
below it come from earlier blocks, its partners above from its own. So the index is written row by
row too, carrying only the pairs whose later row is still to come. `partner` and `r` grow a whole
shard at a time, and an index under one shard is written at its clipped shape at the end. The
Far Pairs are written beside the index, row by row (ADR 0007, decision 2): a cut already holds
its block's pairs at the floor, ascending, and marks which lie beyond `near_bp`, and a row's Far
Pairs all lie in its own block. So each block's rows of `far/<chrom>/` are complete when the
block arrives, nothing is carried and r is not read twice; `partner` and `r` grow a whole shard at
a time, as the index's do, and `indptr` is written at the end. While
a build writes its store it holds a shard of each array and the pairs and unit vectors within
`far_bp` of the rows being written, never a chromosome. On HGDP+1kGP EUR chr6, the MHC's, the band
carry is 128 MiB, the index carry at most 33,869,608 pairs (323 MiB), and the writing peaks at
4.8 GB on 8 threads, against 31.4 GB when the writer held the chromosome. The PC Check's pass
before it reads a window's variants when it scans, not a chromosome's genotypes, and the build's
peak before the tiles is 2.0 GB on chr6, against 10.4 GB (build-inputs §8, ogld#25). A
store written in any blocks, on any number of threads, holds the same arrays.

**Why zstd level 3.** int16 r codes are mostly sampling noise around 0, so they compress little,
and the codec was chosen on measured ratio and speed. These are the measurements on EUR chr22's
real Near Band, by inner chunk (elements). Speeds are per core, in MB/s of raw data
(`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/codec_table.md`, rendered
from `proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/codecs_EUR.tsv`):

| codec | ratio, 4,096 | ratio, 32,768 | ratio, 262,144 | encode MB/s, 32,768 | decode MB/s, 32,768 |
|---|---|---|---|---|---|
| zstd 3 | 1.147 | 1.161 | 1.205 | 634 | 807 |
| Blosc zstd 3, byte shuffle | 1.140 | 1.169 | 1.186 | 276 | 654 |
| Blosc zstd 3, bitshuffle (the engine's chain) | 1.213 | 1.076 | 1.007 | 608 | 1,513 |
| Blosc zstd 9, bitshuffle | 1.250 | 1.261 | 1.280 | 8 | 1,137 |
| delta, then Blosc zstd 3, bitshuffle | 1.158 | 1.046 | 1.006 | 428 | 561 |

- The engine's chain, Blosc zstd 3 with bitshuffle, nearly stops compressing at inner chunks of
  32,768 elements and above.
- At 262,144, Blosc zstd 9 with bitshuffle gains 6.2% over zstd 3 (1.280 against 1.205), but
  encodes at 8 MB/s per core against 634.
- A delta filter makes things worse, because neighbouring codes in a row are mostly independent
  noise.
- The other panels agree. zstd 3 at 262,144 gives 1.165 (AFR), 1.199 (EAS), 1.200 (SAS) and
  1.198 (AMR)
  (`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/codecs_AFR_EAS_SAS_AMR.tsv`,
  `proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/codecs_EAS_SAS_AMR_inner262144.tsv`).
- On the sparse arrays, zstd 3 gave 3.47× on `partner`, 1.40× on `r` and 3.23× on `indptr`
  (measured on the Far Pairs' arrays, which have the same shapes and dtypes as the Proxy
  Index's).

**Why these chunk and shard shapes.** The prototype timed three layouts of the Near Band
(§10), as (inner chunk, shard) in codes: A (4,096, 4,194,304), B (32,768, 16,777,216) and C
(262,144, 67,108,864). The Proxy Index takes the variant column read out of `proxies` at
r2 ≥ t_far, and with it C's largest lead: C decoded that column 8.0× faster than B (0.39 s
against 3.13 s). What decides the layout now is the reads that remain.

- **A is ruled out.** A 3 Mb LD matrix took 21 s from local disk, against 3.7 s for B and 2.0 s
  for C.
- **C is faster on reads that span many chunks, most of all once each request costs time.**
  With 20 ms added per request, C took 2.2× less time than B on the 3 Mb matrix (4.21 s against
  9.34 s, one run of the largest region), 3.0× less on 500 scattered variants (0.54 s against
  1.62 s) and 1.2× less clumping one region's hits (69 ms against 85 ms). From local disk it read
  the matrix 1.8× faster. Without added latency the scattered read is close: 0.31 s against
  0.36 s from local disk, and 0.63 s against 0.59 s over HTTP.
- **B wins on single-chunk reads, and on bytes.** From local disk a single pair took 5.2 ms on B
  and 7.5 ms on C. B read 7.0× fewer bytes for it (0.066 MB against 0.46 MB), and 2.5× fewer for
  500 scattered variants (109 MB against 272 MB). On localhost, bandwidth is not a limit. So these
  runs do not show what those bytes cost a client on a slow link.
- **C's Near Band is 3.9% smaller** on the slice (1,583 MB against 1,648 MB) and was written 4.6×
  faster (2.7 s against 12.4 s)
  (`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/stores.json`).
- **The Proxy Index's layout does not depend on this.** A proxies read is one index row: 9.7 ms
  and 0.045 MB from local disk, 12 ms over HTTP and 97 ms with 20 ms added per request, in 4
  requests, on the shape in the table above (§11).

**The layout is C** (owner, 12 November 2026, on ogld#8; ADR 0006). ogld#8 built HGDP+1kGP EUR
chr22 in each layout and timed the reads that differ, locally and over HTTP from a static host
with 0, 10 and 50 ms added to every request (`docs/benchmarks/latency.md`). Clumping below t_far,
the read that costs most, was fastest on C at every latency: 12.3, 44 and 143 s locally, at 10 ms
and at 50 ms for 27 Analyses' chr22 hits, against B's 17.1, 132 and 488 s, because GWAS hits are
dense and a C chunk serves many of a lead's hits in one call. C read regions fastest and is the
smallest, B 18.9% and A 32.4% larger at × 511. B read 3.2–5.6× fewer bytes on small reads, but
those are bound by requests and took the same time on B and C. A change of chunk or shard shape is
not a format version change (§13).

A C shard is 256 inner chunks: 128 MiB raw. As built (ogld#7, k = 10, × 511), a Near Band shard's
median on disk is 51 to 63 MiB by panel, and its largest 73 MiB. That is 1,257 shards for
HGDP+1kGP EUR's Near Band and 3,457 for AFR's, against the 2,800 and 7,200 the earlier estimate
gave at × 32767. The variant table, Proxy Index and their companion arrays that existed when
layout C was chosen use one shape, 16,384 inner and 1,048,576 to a shard. These arrays are read
whole, by row range, or one sparse row at a time, and the prototype's stores used this shape for
them. The ALID index added later is the exception (ADR 0010): its `alid` byte array uses 262,144
bytes to an inner chunk and 16,777,216 to a shard, and its `row` array uses 262,144 and 1,048,576
rows. At EUR's 5.3×10⁹ Proxy Index entries that is about 5,000 shards for each of `partner` and
`r`. Coarsening that shard would change no reader; ogld#8 measured no read that needed it.

**The Far Pairs keep that shape** (ogld#31): 16,384 entries an inner chunk, 64 chunks to a shard.
EUR's 1.33×10⁸ Far Pairs are 140 shards of each array. They are read by row, so what a read pays is
the chunks its rows touch and the runs of neighbouring chunks among them, each run one Zarr call
(§10). The rows lie very unevenly: EUR chr22 holds 0.85 Far Pairs a variant, and chr11 161.5,
68.75 M of its 69.0 M in rows at 45–55 Mb, up to 11,560 in one row. For a matrix of 500 variants
drawn uniformly on one chromosome (ogld#8's draw, seed 8), and one of 500 drawn at chr11's
45–55 Mb, counted from the EUR store's `indptr`:

| chromosome | the index, at 16,384 | Far Pairs at 4,096 | at 16,384 | at 65,536 | at 262,144 |
|---|---|---|---|---|---|
| 22 | 447 chunks in 339 runs | 15 in 1, 0.4 MiB | **5 in 1**, 0.5 MiB | 2 in 1, 0.8 MiB | 1 in 1, 1.5 MiB |
| 11 | 529 in 463 | 98 in 37, 2.3 MiB | **51 in 28**, 4.8 MiB | 33 in 26, 12.4 MiB | 26 in 24, 39.0 MiB |
| 6 | 526 in 476 | 74 in 46, 1.7 MiB | **45 in 21**, 4.2 MiB | 25 in 13, 9.4 MiB | 16 in 10, 24.0 MiB |
| 11, clustered at 45–55 Mb | 748 in 427 | 725 in 348, 17 MiB | **440 in 313**, 41 MiB | 328 in 198, 123 MiB | 196 in 48, 294 MiB |

Chunks and runs are each array's, `partner` and `r` alike; MiB is what the chunks decode to,
both arrays. On the uniform draws a smaller chunk adds runs where rows are long (37 against 28 on
chr11), and a larger one removes few (26 and 24) and decodes 2.6 and 8 times the bytes. A list
clustered where Far Pairs are densest is the other end: there a larger chunk removes runs (198
and 48 against 313) for 3 and 7 times the bytes, and the Far Pairs save only 1.7 times the
index's chunks. The shape is kept for the uniform lists; ogld#32 fetches a read's runs together,
and a chunk shape is not a format version (§13). A larger shard removes no run
here (chr11's list makes 28 runs within shards at 1,048,576 entries, and still 28 at 16,777,216),
and a store's writer holds one shard of each array (§6, ogld#17).

**Bits per pair, measured** (ogld#16). A real build of HGDP+1kGP EUR chr22 at each scale: the
phased v2 callset converted by `docs/spec/build-inputs.md` §3, the 665 unrelated samples of the EUR
keep file, k = 10 from `/data/opengwasdb/work/ogld/pcs/EUR.eigenvec`, 8 threads, at ogld
`88cb8e9`: 119,632 variants and 1,165,485,828 Near Band pairs, on layout C. Bytes
are those of the arrays' files on disk; the Proxy Index's are its `indptr`, `partner` and `r`
together.

| encoding | Near Band | bits per pair | against × 32767 | Proxy Index entries | Proxy Index | bits per entry |
|---|---|---|---|---|---|---|
| `int16` × 511 | 1,002,545,575 B | 6.88 | 1.68× fewer | 45,496,764 | 69,284,904 B | 12.18 |
| `int16` × 2047 | 1,249,975,092 B | 8.58 | 1.34× fewer | 45,664,034 | 78,524,164 B | 13.76 |
| `int16` × 32767 | 1,680,078,624 B | 11.53 | 1.00× | 45,677,130 | 89,977,528 B | 15.76 |

The issue's own measurement, on a k = 0 slice of the MAC ≥ 50 pgen (24–30 Mb, 140 M pairs), gave
7.11, 9.04 and 12.06 bits: 1.70× and 1.33×. The whole chromosome at k = 10 agrees with those ratios
to within 1.5%, though not with their bits. The index holds slightly fewer entries at × 511
because the far code floor's cut moves with the scale (§3): 101,886 Far Pairs against 105,272.

**Sizes, measured** (ogld#7). The five stores as built: the phased v2 callset converted by
`docs/spec/build-inputs.md` §3, each panel's unrelated samples (EUR 665, AFR 608, EAS 724, SAS 672,
AMR 341), MAC ≥ 20, k = 10, × 511, 16 threads, on layout C, at ogld
`0.0.1.dev230+g5d78804cb`. Bytes are those of each array group's files on disk (`near/`, `index/`;
`variants/` adds 90–165 MB per store and is in the total). Bits per pair are the Near Band's bytes
× 8 over its pairs; bits per entry are the Proxy Index's (`indptr`, `partner`, `r`) over its entries.

| LD Store | variants | Near Band pairs | Near Band | bits per pair | Proxy Index entries | Proxy Index | bits per entry | total |
|---|---|---|---|---|---|---|---|---|
| HGDP+1kGP EUR | 8.59 M | 83,623,618,163 | 69.4 GB | 6.64 | 5,287,398,542 | 9.4 GB | 14.20 | 78.9 GB |
| HGDP+1kGP AFR | 14.37 M | 231,225,704,112 | 225.8 GB | 7.81 | 6,682,597,942 | 14.7 GB | 17.62 | 240.7 GB |
| HGDP+1kGP EAS | 7.84 M | 69,995,382,360 | 56.4 GB | 6.45 | 4,328,825,924 | 7.1 GB | 13.12 | 63.6 GB |
| HGDP+1kGP SAS | 8.88 M | 89,530,781,741 | 75.8 GB | 6.78 | 4,879,197,374 | 8.6 GB | 14.09 | 84.5 GB |
| HGDP+1kGP AMR | 7.56 M | 65,192,230,001 | 58.6 GB | 7.19 | 3,337,986,118 | 5.4 GB | 13.05 | 64.1 GB |
| **Total** | 47.2 M | 539,567,716,377 | 486.1 GB | 7.21 | 24,516,005,900 | 45.2 GB | 14.76 | 531.9 GB |

The five total 532 GB, 0.53 TB, before their Far Pairs (ogld#31). Those were derived in place
from each store's own index (ADR 0007) and add 1.46 GB, 0.27%: 18.5 bits a pair on disk, 3.2% of
the index's bytes. Bytes are those of `far/`'s files; raw, a pair is 6 bytes.

| LD Store | Far Pairs | per variant | `far/` | bits per pair | against the index | of the store |
|---|---|---|---|---|---|---|
| HGDP+1kGP EUR | 132,924,736 | 15.5 | 0.32 GB | 19.00 | 3.4% | 0.40% |
| HGDP+1kGP AFR | 192,465,514 | 13.4 | 0.49 GB | 20.47 | 3.3% | 0.20% |
| HGDP+1kGP EAS | 92,650,917 | 11.8 | 0.19 GB | 16.76 | 2.7% | 0.30% |
| HGDP+1kGP SAS | 122,204,575 | 13.8 | 0.29 GB | 18.96 | 3.4% | 0.34% |
| HGDP+1kGP AMR | 88,467,989 | 11.7 | 0.16 GB | 14.83 | 3.0% | 0.26% |
| **Total** | 628,713,731 | 13.3 | 1.46 GB | 18.53 | 3.2% | 0.27% |

**Why the earlier estimate (1.20 TB at × 511) was high.** It was 2.3 times the measurement, for two
reasons. Variants: it took gnomAD v3.1.2's MAC ≥ 20 sites, non-PASS sites included (about 25% of
EUR chr22's), and its sample counts; the builds read the phased v2 callset, whose QC keeps fewer
sites, and apply MAC ≥ 20 to the unrelated samples, so a panel holds 8.6 M variants (EUR) against
12.9 M, 14.4 M (AFR) against 20.9 M, and so on. The Near Band's pair count grows as the square of
that, so it came out 1.7 to 2.3 times fewer than estimated (EUR 8.4×10¹⁰ against 1.86×10¹¹; AFR
2.3×10¹¹ against 4.8×10¹¹; AMR, whose estimate was measured genome-wide, 6.5×10¹⁰ against
1.09×10¹¹). Bits per pair: the estimate carried the prototype's k = 0 sizes (7.9 bits per pair on
EUR) over at a ratio measured on chr22 alone; the builds measure 6.4 to 7.8 (AFR's 7.81 is the one near it). The Proxy
Index came out 16–55% below the estimate's in bytes and 4–47% in entries (AMR's the most).

**UK Biobank EUR's size is ogld#10's measurement.** No UK Biobank genotype was read for this
spec. At n = 49,766, MAC ≥ 20 is MAF ≥ 2.0×10⁻⁴. The only UK Biobank variant list under
`/data/opengwasdb/reference/` on the IEU compute nodes is the engine's UK Biobank EUR LD Reference
Panel's: 9.8 M variants, all with MAF ≥ 0.005
(`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/ukb_panel_list_maf.json`).
That list is a subset of the MAC ≥ 20 set and cannot count it.

- On that list alone, the Near Band would hold 1.10×10¹¹ pairs, or 182 GB at × 32767 and about
  135 GB at UK Biobank EUR's × 2047 (the measured ratio 0.744). That is an upper bound: at
  n = 49,766 the r of unlinked pairs spread √(49,766 / 665) ≈ 8.7 times less than at n = 665, so
  its codes take fewer bits.
- Near Band bytes grow with variants × partners per row, so roughly with the square of variant
  density. Twice the list's variants gives about 729 GB, and three times gives 1.64 TB, at
  × 32767; about 542 GB and 1.22 TB at × 2047
  (`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/panel_sizes_inner262144.tsv`).
- The Far Pairs at t_far = 0.001 cannot be measured at n = 771, the prototype's EUR. There, the
  mean r² at 3–10 Mb (0.0012945) is no higher than sampling alone predicts (1/n = 0.0012970;
  `proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/s08_meta.json`). At an
  assumed 1–5% of 3–10 Mb pairs, the list's Far Pairs would be 6–30 GB, and the Proxy Index,
  which holds each twice beside the Near Band's pairs at r² ≥ 0.001, more than twice that.

## 7. Provenance

The root group attribute `ld_store` is a JSON object. Every key below is REQUIRED, with the type
shown. A value may be null only where the table says so. Other keys MAY appear, and a reader
ignores the ones it does not know. The types are `ogld.format.Provenance`, and
`ogld.format.REQUIRED_KEYS` lists the keys.

**Who records what** (owner, 10 November 2026; ogld#17, ADR 0005). A store's producer gives the keys only
its source knows, `ogld.format.Sourced`: `source`, `panel`, `reference_assembly`,
`variant_filters`, `variants_dropped`, `adjustment`, `pc_diagnostic`, `software` and `build`. The
store derives the rest from itself when it is finished: `format_version`, `encoding`, `n_variants`,
`near_bp`, `far_bp`, `t_far`, `t_far_q`, `far_code_floor` and `counts`. It takes t_far's n and k
when it is created, since it thresholds as rows arrive, and refuses a producer's `panel.n` or
`adjustment.k` that differs. So no producer computes a threshold or a count of its own.

| key | type | meaning |
|---|---|---|
| `format_version` | string | `"0.1.0"` before the Partner Rows, `"0.2.0"` with them (§13) |
| `encoding.r_dtype` | string | `"int16"`, the container of every admitted encoding (§3) |
| `encoding.r_scale` | int | the store's scale s, a whole number from 1 to 32767: r̂ = code / s. 511 for the HGDP+1kGP stores, 2047 for UK Biobank EUR (ADR 0004); a build records the `r_scale` it was given |
| `encoding.r_reserved_code` | int | −32768 |
| `encoding.rounding` | string | `"half_to_even"` |
| `source.kind` | string | `"genotypes"`: built from a Source Panel's genotypes; `"ld_matrix"`: imported from a published LD matrix, such as Pan-UKB's |
| `source.description` | string | what the input was, such as the gnomAD release and files, or the matrix's location and release |
| `source.sha256` | object | SHA-256 per input file, keyed by file name: the genotype files (each fileset's `.pgen`, `.pvar` or `.pvar.zst`, and `.psam`) and the keep file, or the matrix's files |
| `panel.id` | string | the Source Panel's identifier, such as `"hgdp1kgp-v2-EUR"` (`docs/spec/build-inputs.md` §1) |
| `panel.ancestry` | string | the ancestry label, such as `"EUR"` |
| `panel.n` | int | samples the LD was computed from |
| `panel.samples_sha256` | string or null | SHA-256 of the sorted sample identifiers, one per line; null only for an `ld_matrix` source that does not publish them. The identifiers themselves stay with the Source Panel |
| `reference_assembly` | string | the engine's Reference Assembly, such as `"GRCh38"` |
| `variant_filters` | object | the build's variant filters and their values, as `docs/spec/build-inputs.md` §6 applies them: `filter` (array of strings, the FILTER values that pass: `["PASS", "."]`), `mac_min` (int, default 20), `info_min` (float, or null when no INFO floor was applied, as for a sequenced panel) and `autosomes_only` (bool, true: a record off chromosomes 1–22 is refused, §1). A zero-variance variant is dropped with no setting to record, and counted in `variants_dropped`. Biallelic records are the conversion's job, so no filter sees a multi-allelic one. There is no call-rate filter: a build refuses an input with any missing call |
| `variants_dropped` | object | count per reason a variant passing the filters was dropped: at least `zero_variance` and `duplicate_reversed_indel` (int each; the latter is `docs/spec/build-inputs.md` §6's rule, from the engine's ADR 0072) |
| `n_variants` | int | rows of `variants/` |
| `near_bp` | int | the distance up to which the store holds every pair: 3000000 |
| `far_bp` | int | the distance up to which the store measured pairs: 10000000 for a build, and at most that for any store. Every reader and refusal uses this recorded value |
| `adjustment.covariates` | array of strings | every covariate regressed out of the genotypes, by name, in order: the first up to `k_target` PCs the PC Check kept, such as `"PC1"`, `"PC3"` … `"PC11"` when it dropped PC2; empty for unadjusted LD. An import names its source's, such as Pan-UKB's age, sex, age×sex, age², age²×sex and PC1–PC10 |
| `adjustment.k` | int | the number of covariates used: t_far's k (§3). For a build from a PC File, the PCs actually used, between `k_min` and `k_target`; 0 for an unadjusted build; for an import, its source's declared count of covariates |
| `adjustment.k_target` | int or null | the PCs a build aims to use: `BuildInputs.k`, 10 by default, and 0 for an unadjusted build; null for an `ld_matrix` source, which has no target |
| `adjustment.k_min` | int or null | the fewest PCs a build may use: `BuildInputs.k_min`, 5 by default. With fewer surviving the PC Check, the build is refused. It applies only when PCs are used, so it is null when no PC File was read: for an unadjusted build (k = 0, which nothing refuses for want of PCs) and for an `ld_matrix` source |
| `adjustment.pc_file` | object or null | the PC File (`docs/spec/build-inputs.md`): `sha256` and its declared `source`, such as the recipe script and release that made it; null when no PC File was read |
| `adjustment.pc_check` | object or null | the PC Check's result (`docs/spec/build-inputs.md` §8): the rule's `window_bp` (4000000), `step_bp` (2000000), `min_share` (0.1) and `max_ratio` (0.05); `windows`, the number of windows scanned; the `scan`, one entry per PC of the PC File with its best `window`, its `window_share`, its `rest_share` (the mean share of every window that does not overlap that one), `ratio` (rest_share / window_share), `window_null_q99` (the threshold the window share must beat), `rest_null_q99` (a diagnostic) and whether it is `local` (`window_share` ≥ `min_share`, `window_share` > `window_null_q99` and `ratio` ≤ `max_ratio`); the `dropped` PCs; and the `null`, how chance was made: `kind` (string), `permutations` (int), `seed` (int) and `quantile` (float). Null when no PC File was read |
| `t_far` | float | max(0.001, q / (n − k − 1)) in float64, the value every reader uses |
| `t_far_q` | float | 20.84: the χ²₁ upper quantile at p = 5×10⁻⁶ |
| `far_code_floor` | int | §3 |
| `t_partner` | float | the **Partner Threshold**: the r² a pair within `near_bp` must reach to be in the Partner Rows, never above `t_far` (0.2.0; `docs/spec/partner-rows.md` §3) |
| `partner_code_floor` | int | the **Partner Floor**: §3's floor rule at `t_partner` (0.2.0) |
| `pc_diagnostic` | object or null | what PC Adjustment did to chance LD (below); null only for an `ld_matrix` source. A Variant Set with no cross-chromosome pair (one chromosome) records `pairs.n_pairs` 0 and null for the two measured rates, `rate_r2_ge_0_01_unadjusted` and `rate_r2_ge_0_01_adjusted`: each is null only when `n_pairs` is 0, and a number otherwise (10 November 2026). `rate_r2_ge_0_01_sampling` is a formula of n and k, and always a number. `ogld.format.PcDiagnostic` types the two measured rates `float \| None` and the sampling rate `float` |
| `counts.near_pairs` | int | Near Band codes, over all chromosomes |
| `counts.far_candidates` | int | pairs more than `near_bp` and at most `far_bp` apart that were examined |
| `counts.far_pairs` | int | Far Pairs: those of them whose \|code\| reaches `far_code_floor`, each counted once; the entries of every `far/<chrom>/` together |
| `counts.partner_entries` | int | Partner Rows entries, both directions, over all chromosomes: each `partners/<chrom>/` group's `entries` (0.2.0) |
| `counts.index_entries` | int | Proxy Index entries, both directions and within `near_bp` included |
| `counts.alids_left_out` | int | the table's count of ALIDs wider than the ALID index's 32-byte slot, counted from `variants/alid`, whose rows the index has no entry for; the index holds the other `n_variants` − this many (`n_variants − len(variants/alid_index/row)`), and each of these is found by the exact scan at its position (§2; ADR 0010) |
| `software` | object | versions, by package name. A build from genotypes records exactly `ogld`, `numpy`, `zarr`, `zstd` and `pgenlib`, its genotype reader (ogld#5's review); `ogld` is the installed version, which names the commit that built the store when the install is fresh (the realdata tier refuses a stale one) |
| `software.zstd` | string | the zstd library's version, because the Partner Rows' frame bytes depend on it (0.2.0); a build records it, so it is required with them |
| `build.created_at` | string | ISO 8601 UTC |
| `build.wall_seconds` | float | build time, up to the validation `StoreWriter.finish` runs before it writes this provenance: a provenance cannot hold the time taken to check it. On EUR chr22 that validation took another 80 s of a 2 min build |
| `build.peak_memory_bytes` | int | peak memory over the same span: the build's own on Linux, whose high-water mark the build resets when it starts; elsewhere the process's |
| `build.threads` | int | threads used: the build's worker threads, which bound it, since BLAS runs one thread inside a build |

**The Partner Rows' keys (format 0.2.0; written by #67, derived by #69).** The four keys at
`t_partner`, `partner_code_floor`, `counts.partner_entries` and `software.zstd` are in the table
above, and in `ogld.format.REQUIRED_KEYS` with it. Each is required only of a store whose
`format_version` is `0.2.0`: a 0.1.0 store has no Partner Rows, so it has none of their keys.

- `t_partner` (float): the **Partner Threshold**, the r² a pair within `near_bp` must reach to be in
the Partner Rows. It is a required build input with no default (`docs/spec/build-inputs.md` §1),
never above `t_far`: 0.01 for the five HGDP+1kGP stores, 0.001 for UK Biobank EUR.
- `partner_code_floor` (int): the **Partner Floor** (§3) at the recorded scale.
- `counts.partner_entries` (int): the entries of every `partners/<chrom>/` together, twice the
pairs the Partner Rows hold, as each chromosome's `entries` attribute gives it.
- `software.zstd` (string): the zstd library's version, because a frame's bytes depend on it.

`counts.index_entries` stays until the Proxy Index goes, in #72.

`pc_diagnostic` holds `pairs` (the pair set: `source`, opengwas/opengwasdb#405's seeded
cross-chromosome pairs; `seed`, 405; `n_pairs`, 200,000), `rate_r2_ge_0_01_unadjusted` (the share
of those pairs with r² ≥ 0.01 at k = 0), `rate_r2_ge_0_01_adjusted` (the same share after PC
Adjustment), `rate_r2_ge_0_01_sampling` (the share sampling alone predicts,
P(χ²₁ ≥ 0.01 × (n − k − 1))) and `near_100kb_change` (quantiles 0, 0.5, 0.9, 0.99 and 1 of the
change in r from adjustment, for pairs within 100 kb outside the Long-Range LD Regions: ogld#5's "real
LD survives").

- **The pairs** are drawn as #405 drew them
  (`opengwas/opengwasdb@6b97e57:prototypes/locus_leads/panel_chance.py`): at most 3,000 Variant Set variants without replacement by
  `numpy.random.default_rng(405)`, then 200,000 pairs of those on different chromosomes, with
  replacement, by a second generator seeded 405. **A Variant Set on one chromosome has no such
  pair** (a test slice, never a panel's LD Store): `n_pairs` is 0, `source` says so, and the two
  measured rates are null, never a number nothing measured; each is null only when `n_pairs` is
  0. `rate_r2_ge_0_01_sampling` is a formula of n and k and always a number (the owner's
  ruling, orchestrator note 2 on ogld#5).
- **The change in r** is adjusted minus unadjusted r, signed to A1, computed in float64. A whole
  panel has about 10⁹ such pairs, so a build counts them into bins 2⁻²⁰ (about 10⁻⁶) wide: the
  quantiles 0 and 1 are the exact extremes, and each other quantile q is the order statistic at
  rank ⌊q(N − 1)⌋ of the N pairs, to within half a bin. At k = 0 every change is 0.

The provenance holds counts, hashes and settings, never a sample identifier or a per-sample value.

**Why the source, the covariates and the distances are recorded.** Pan-UKB publishes LD matrices
that a later epic may import (epic #1, comment of 7 November 2026). They are in-sample dosage r
over ±10 Mb, adjusted for age and sex terms as well as 10 within-ancestry PCs, on GRCh37
(epic #1, comment "Pan-UKB LD as a source: evidence for a later epic"). An imported store therefore needs to say that it
was not built from genotypes here, to name every covariate t_far's k counts, and to record the
distance it measured, which a lift to GRCh38 can shorten. These keys cost nothing now and would be
a format change later.

**PCs are supplied, and the builder chooses among them** (owner, 7 November 2026; ADR 0002).
ogld's library runs no PCA. The PCs are our own, made outside it by a recipe script that follows
gnomAD's subcontinental method with the Long-Range LD Regions ±2 Mb excluded, and a build reads
them from a PC File. The builder then runs the PC Check, the owner's local-PC rule: a PC is
local, and dropped, when one 4 Mb window explains ≥ 10% of it, that share beats the window-side
null, and the rest of the genome explains at most 0.05 of what that window does
(`docs/spec/build-inputs.md` §8). The covariates are the first up to 10 PCs kept, taken in order; if fewer than 10
survive, the build uses those that do, as long as at least 5 do, and refuses with fewer, naming
how many survived and which were dropped (owner, 7 November 2026). t_far's k is the number used. So the provenance
records the target, the minimum and the k used, the PC File's
hash and declared source and the whole scan, not a PCA method or LD pruning. A PC that tracks a
region the rest of the genome also explains, such as EUR's north–south axis through LCT (ratio
0.104), is ancestry and is kept; an inversion's karyotype axis (ratio 0.007–0.019) is dropped
(ADR 0002 has the evidence). The ratio cannot fall much below 5 / (n × window share), so the rule
needs a few hundred samples: every HGDP+1kGP panel keeps at least 341, AMR's unrelated samples
(ADR 0002, decision 11), well above the about 100 below which no ratio reaches 0.05.

**Genotypes are complete.** The builds read the phased HGDP+1kGP v2 callset, which has no missing
calls, and a build refuses any input with one. So there is no imputation of missing calls and no
call-rate filter, and with k = 0 the Near Band is exactly the r `plink2 --r-unphased` computes, to
within the quantisation step (§4).

## 8. Pair answers

The **Pair Answer** for variants i and j:

| distance | answer |
|---|---|
| ≤ `near_bp` (3 Mb) | the exact r: the Stored r, from the Near Band |
| `near_bp` – `far_bp` (3–10 Mb) | the Stored r if the pair is a Far Pair (r̂² ≥ t_far); otherwise the **Bound** "r² < t_far", never a stored 0 |
| > `far_bp` | **Not Measured** |

The boundaries, exactly:

- d ≤ `near_bp` is the Near Band; `near_bp` < d ≤ `far_bp` is a Far Pair or the Bound; d > `far_bp`
  is Not Measured. The distances are the store's recorded ones (§7).
- Variants on different chromosomes are Not Measured.
- A variant with itself is r = 1, not stored.
- A variant outside the Variant Set is **Absent**, and so is every pair it is in. A reader MUST say
  which variants are Absent, distinct from pairs that are Not Measured.
- The Bound states t_far, the value recorded in provenance. It is judged on the Stored r: it
  means r̂² < t_far. So a pair whose true r² is up to about √t_far / s above t_far, at the store's
  scale s, can read as the Bound. On HGDP+1kGP EUR at k = 10, a true r² up to 0.0320627 reads
  "r² < 0.0318654" (§3).

```mermaid
flowchart TD
  A["variants i and j"] --> B{"both in the Variant Set?"}
  B -->|no| ABS["Absent"]
  B -->|yes| C{"same variant?"}
  C -->|yes| ONE["r = 1, not stored"]
  C -->|no| D{"same chromosome?"}
  D -->|no| NM1["Not Measured"]
  D -->|yes| E{"distance d"}
  E -->|"up to near_bp"| NB["Stored r from near/c/r at indptr of i, plus j - i - 1"]
  E -->|"near_bp to far_bp"| F{"j in the Far Pairs row of i?"}
  F -->|yes| FP["Stored r from far/c/r"]
  F -->|no| BD["Bound: r2 below t_far"]
  E -->|"over far_bp"| NM2["Not Measured"]
```

_Figure 2. How a reader resolves one pair, with i the earlier variant._

**A matrix is one signed-r matrix** (owner, 7 November 2026). A reader that returns the pairs
among a list of variants as a matrix returns the Stored r of each pair, signed to each variant's
A1, with 1 on the diagonal, and NaN for every pair the store does not hold: the Bound and Not
Measured alike.

- **A NaN's kind is recoverable from the result.** The matrix carries its variants' chromosomes
  and positions, t_far and the store's distances. A NaN between two variants on one chromosome
  more than `near_bp` and at most `far_bp` apart is the Bound; any other NaN is Not Measured. No
  NaN can lie within `near_bp`, because the Near Band is complete. So no second matrix of kinds is
  returned. `ogld.format.answer_kinds` reads the kinds back from a matrix made without a fill,
  and refuses one it cannot vouch for: a NaN within `near_bp`, a number where the store holds
  nothing, or a number beyond `near_bp` below t_far.
- **`fill` is the caller's.** A reader MUST NOT put a number in place of an unstored pair unless
  the caller passed `fill`, and then it puts exactly that number. A filled cell cannot be told
  from a stored one, so the matrix records its `fill`, and `LDMatrix.kinds()` refuses a filled
  matrix: the caller asks for the matrix with no fill to read the kinds.
- **The stored data never changes.** An LD Store holds no code for an unstored pair, and no read
  writes one. `fill` acts on a query's result, never on the store.
- Absent variants are listed beside the matrix, in request order, and are not rows of it. A
  present ALID asked for twice is two rows, as asked.

This replaces the earlier rule that a reader returning a plain numeric matrix must refuse when any
pair is not a Stored r, and the epic's acceptance criterion that no query turns an unstored pair
into a 0, for a query whose caller asks for a fill (ADR 0001, clarification of 7 November 2026).

## 9. Queries and refusals

The four queries (owner, 7 November 2026). Each returns signed r, and each judges a threshold on
r̂² (§3).

| query | returns | its pairs | refused when |
|---|---|---|---|
| `matrix(alids, fill=None)` | one signed-r matrix over the present ALIDs, in request order (§8) | every pair of the list | never for distance: a pair more than `far_bp` apart is a NaN that is Not Measured |
| `region(chrom, centre, radius, fill=None)` | one signed-r matrix over every variant with \|p − centre\| ≤ radius, in genomic order (§8) | every pair of the region | the width 2 × radius exceeds `far_bp`: radius > 5 Mb on a 10 Mb store |
| `proxies(alids, r2=0.8, kb=10000)` | per present ALID, every other variant within W with r̂² ≥ r2: query, proxy, signed r, r̂², distance | d ≤ W | the table below |
| `clump(alids, score, r2, kb)` | per present ALID, the lead of the clump it joined and its signed r with that lead | d < W (clump only: plink2's edge) | the table below |

`proxies` considers the pairs with d ≤ W, where W = kb × 1,000 bp, and a pair reaches the threshold
when r̂² ≥ r2 (§3). `clump` is the one exception, following `plink2 --clump` (owner, 10 November
2026): it considers the pairs with d < W, and a pair reaches the threshold when r̂² > r2. The
refusals below are the same for both. The refusals are ADR 0001's, compared with the distances the store
recorded (§7) and its recorded t_far, and from 0.2.0 its recorded t_partner:

| request | format | answer |
|---|---|---|
| r2 outside [0, 1], kb negative or not finite, or a `clump` score that is not one finite, non-negative number per ALID | both | refused: `invalid` |
| W > `far_bp` | both | refused: `window_over_far`. Pairs more than `far_bp` apart are Not Measured |
| `near_bp` < W ≤ `far_bp` and r2 < t_far | both | refused: `threshold_below_t_far`. The store holds only the Bound for the pairs beyond `near_bp` that are not Far Pairs, and that does not decide r̂² ≥ r2 |
| `near_bp` < W ≤ `far_bp` and r2 ≥ t_far | 0.1.0 | answered from the Near Band and the Proxy Index |
| `near_bp` < W ≤ `far_bp` and r2 ≥ t_far | 0.2.0 | answered from the Partner Rows. Every pair beyond `near_bp` left out is not a Far Pair, so r̂² < t_far ≤ r2 |
| W ≤ `near_bp`, any r2 | 0.1.0 | answered from the Near Band alone |
| W ≤ `near_bp` and r2 < t_partner | 0.2.0 | refused: `threshold_below_t_partner`. The Partner Rows hold only the pairs at or above the Partner Floor, and absence there decides nothing below it |
| W ≤ `near_bp` and r2 ≥ t_partner | 0.2.0 | answered from the Partner Rows: every pair that could reach r2 is at or above its distance's floor, so it is in the row |

**Why the answered rows are exact.** `clump` joins at r̂² > r2 and `proxies` keeps r̂² ≥ r2, so a
request's r2 is at least t_partner within `near_bp` and at least t_far beyond it. Any pair that
could reach r2 therefore has |code| at or above its distance's floor and is in the row it reads. A
pair absent from that row has r̂² below a bound no higher than r2 (`docs/spec/partner-rows.md` §4),
so it cannot reach r2. On 0.1.0's rows the Proxy Index and the Near Band give the same argument at
t_far. It is a breaking change on HGDP+1kGP: an r² 0.001 clump with a 3 Mb window, answered today,
is refused from 0.2.0, where about 90% of the pairs such a threshold passes are sampling noise.

- r2 = t_far exactly is answered: the store keeps r̂² ≥ t_far, and the request asks for
  r̂² ≥ t_far. r2 = t_partner exactly is answered from 0.2.0, for the same reason.
- A request is compared with the recorded `t_far`, and from 0.2.0 its recorded `t_partner`, never
  with a rounded display of either.
- A Refusal MUST name its reason. A request MUST NOT be answered approximately instead. A region
  wider than `far_bp` is refused as `region_over_far`, and a negative radius as `invalid`.
- On UK Biobank EUR (t_far = 0.001) only W > 10 Mb is refused at r2 ≥ 0.001, so MR's r² 0.001
  with a 10 Mb window is answered there and nowhere else; its t_partner is 0.001 too, so nothing
  within `near_bp` is refused either. On HGDP+1kGP AMR (t_far 0.0631515 at k = 10, §3), r2 = 0.05
  with a 5 Mb window is refused.
- **`proxies`.** The defaults, r2 = 0.8 and a 10,000 kb window, are answered on every store with
  `far_bp` = 10 Mb, because every t_far is below 0.8. A variant is never its own proxy. Rows are
  grouped by query, in request order, then by r̂² descending, distance ascending and proxy ALID. At
  r2 ≥ t_far a variant's proxies are its Proxy Index row; below t_far, which is answered only
  within `near_bp`, they are its Near Band row and column (§10). From 0.2.0 every answered r2 is
  one Partner Rows frame read (§10, #71), and the Near Band's row and column go (#72).
- **`region`.** Its chromosome passes through the engine's label normalisation
  (`ogld.identity.normalise_chromosome`), so `chr22` reads as `22`. Its variants are those with
  centre − radius ≤ position ≤ centre + radius, in the Variant Set's order.
- **`clump`.** Greedy, strongest score first. The score is |z| or −log10 p, never raw p, which
  underflows to 0 for |z| over 38. Ties go to genomic order, then ALID. Each variant joins the
  strongest lead it reaches. plink2 compares r² with `>` and the window with `<` (checked on plink2 v2.0.0-a.6.9LM and
  recorded beside the equality test, `tests/contract/test_query.py`), so `clump` joins a variant to a lead at r̂² > r2
  and d < W, and `proxies` keeps ≥ and ≤. A stored r̂² equals r2 exactly only at 0 or 1.

`ogld.format.judge_request` and `ogld.format.judge_region` implement the table's refusals, all but
one: a `clump` score that is not one finite, non-negative number per ALID is refused as `invalid`
by `ogld.query.clumping._checked_score`, because it depends on the score, which the judges never
see. `judge_request` runs first.
`proto/1-ld-store-layout@65f94ca:prototypes/answer-semantics.html` drives them and the answer
table interactively, with walkthroughs of the edge cases: exactly 3 Mb, exactly 10 Mb, r²
equal to t_far, rounding onto a threshold, AMR's t_far, a pair at exactly t_far, and a window over
10 Mb.

## 10. Reads

_Informative._ A reader keeps a chromosome's `position` range and its Near Band's `indptr`
once it has opened that chromosome. Its Proxy Index's `indptr` is read only when a request
uses the index, and its Far Pairs' `indptr` when a request needs pairs beyond `near_bp`: 8 bytes a variant each. For HGDP+1kGP EUR's
8.59 M variants that is 69 MB of positions, which a reader holds as `int64`, and 137 MB of
`indptr`, raw, and 69 MB more once matrices have been read on every chromosome. Each query makes one of the reader's reads (`ogld.store.StoreReader`, ogld#14),
and the reads map onto the arrays as follows.

| read | reader | made by | arrays it touches |
|---|---|---|---|
| the rows of a list of ALIDs | `rows_of` | `matrix`, `proxies`, `clump` | for a long list, the ALID index — `variants/alid_index/alid` one shard at a time, then `variants/alid_index/row` at the entries found; for a short one, and for every ALID the index does not hold, the exact scan: `variants/position` and `variants/alid` at each ALID's position. No Proxy Index entry, no `near/` array, and no chromosome loaded for an ALID the index answers |
| a region | `region_r` | `region` | for each of its rows, one contiguous slice of `near/<c>/r`, from the row's first code to its last partner inside the region, never its partners past the region; for its pairs beyond `near_bp`, the Far Pairs rows of its variants, each pair inside the region scattered into the matrix. No Proxy Index entry |
| a submatrix for any variant list | `pair_r` | `matrix` | one element of `near/<c>/r` per pair within `near_bp`; for the pairs beyond it, the Far Pairs rows of the listed variants, each pair whose partner is listed scattered into the matrix. No Proxy Index entry |
| a list of pairs, or one row against many | `r_of_pairs` | `partners` below t_far with `among`; `pair_r`, within `near_bp` only | one element of `near/<c>/r` per pair within `near_bp`; the Far Pairs row of each farther pair's earlier variant, searched. `pair_r`'s pairs beyond `near_bp` are not looked up this way: they are scattered from Far Pairs rows, as its own row says |
| partners at r2 ≥ t_far | `partners` | `proxies`; `clump` | each variant's Proxy Index row: both sides, up to `far_bp`, in one piece. No chunk of `near/<c>/r` |
| partners below t_far, W ≤ `near_bp` | `partners` | `proxies` | each variant's Near Band row up to W, and its column within W: one element in each row of the preceding W (§11) |
| partners below t_far among a list, W ≤ `near_bp` | `partners`, `among` | `clump` | one element of `near/<c>/r` per pair of a variant and a listed variant within W, and nothing else |
| a variant's Partner Rows row | `partner_row` | `proxies`; `clump` from #71 | the frame that holds the row (§10's Partner Rows paragraph): its byte range from the chromosome's `frames_idx` chunk, read out of the `frames` chunk file, decoded, and cached per reader. No `near/` or `index/` array and no shard index. Format 0.2.0, #70 |

So the stored part a query reads depends on its threshold, and the reader chooses it; no query
compares r2 with t_far itself, nor, from 0.2.0, with t_partner:

| query | r2 ≥ t_far | r2 < t_far (answered only for W ≤ `near_bp`, §9) |
|---|---|---|
| `matrix`, `region` | no threshold: the Near Band within `near_bp`, the Far Pairs beyond it | |
| `proxies` | the Proxy Index row of each query variant | the Near Band row and column of each query variant, within W |
| `clump` | the Proxy Index row of each lead, among the hits still unclumped | one Near Band element for each lead and each hit still unclumped within W |

**From 0.2.0** (#70, #71), a variant's partners at every answered threshold are one frame read.
`partner_row` finds frame ⌊l / G⌋, reads its byte range from the chromosome's `frames` chunk file,
verifies the zstd checksum, checks k and the payload length, decodes it and keeps it in a
byte-bounded LRU of decoded frames. That cache is per reader and separate from the Near Band chunk
cache, which keeps serving `matrix` and `region`: neighbouring leads often share a frame, and
repeated Analyses on one reader hit it. Measured on GCST90476460 (#65), whose 3,869 leads fall in
95 components: G = 16 with four component workers read 3,854 frames and 101,810,367 bytes cold,
with 15 cache hits, and did 1.067 s of LD work cold and 1.058 s warm. The Proxy Index row read,
the Near Band's row and column reads and clump's use of the chunk cache, single-flight and
admission go with them (#72); everything else in this section stays for `matrix` and `region`.

`clump` visits one chromosome at a time, strongest first inside it, and scatters results
back into the original request order, including duplicates (ogld#40; ADR 0009). A clump
cannot cross chromosomes, so this changes no lead or member, but avoids evicting chunks to
visit other chromosomes between leads. It reads ALIDs, chromosomes and positions, not the
alleles or allele frequencies it never returns. A sparse chunk-cache read gathers only its
requested elements, rather than concatenating whole cached chunks on every warm lead.

`clump` asks for one lead's partners at a time, among the hits no stronger lead has taken, and
never for a square among a lead and its m candidates, which resolves every pair among them. On the
Test Store's 1,056 hits within 3 Mb, at its × 511 encoding, the leads need 23,228 pairs at r² 0.1.
The squares resolved 1,489,250 of them and took 1,587,155 Stored r from the chunk cache; `partners`
takes 3,122, the 438 leads' Proxy Index rows. At r² 0.01, below t_far, the leads need 7,311 pairs
and `partners` takes exactly those. `tests/test_partner_reads.py` holds both, and `proxies`
reading no string column but the ALIDs it returns. A chromosome's `near/<c>/indptr` is still read when a request first
touches it, as below, at any threshold.

What the reads hold, measured with tracemalloc and held by `tests/test_partner_reads.py`:

- `partners`: 48 bytes per kept entry, its partner, r and distance held once as read and once
  in the result while they are put in order. Beside them, one block as it is read and judged:
  at most `_PAIRS_PER_STEP` entries (about 4 million), or one row's if it holds more, at most
  72 bytes each. The index's entries take 30 bytes each and the Near Band's 53–70.
- `r_of_pairs`: its result, 8 bytes a pair, and one step of pairs, about 60 bytes each, with
  that step's index rows. One row against many is never copied out to a full list.
- `partner_row`: the decoded frame it read, kept by the frame cache, and its row's partners and
  codes as views into it: 6 bytes an entry while the frame is kept, plus the frame's own arrays
  and the `frames_idx` chunk.
- `pair_r` and `region_r`: their n × n float64 result, written in place, and one step or block
  beside it. `matrix` and `region` apply a `fill` in place, so each holds one float64 copy of
  its result.

**The reads on the built stores are measured in `docs/benchmarks/latency.md`** (ogld#8): the
published EUR store from local disk and over HTTP at 0, 10 and 50 ms added per request, and EUR
chr22 built in each layout. Those numbers decided the layout (ADR 0006) and supersede the
prototype's below, which are kept as the record of what chose layout C provisionally.

The prototype timed each read on the EUR chr22 slice, a real Zarr v3 sharded LD Store (1.65 GB,
65,035 variants), under the three layouts of §6. It read from local disk, and over HTTP from a
static server that honours Range requests
(`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/access_tables.md`, rendered
from `proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/access_summary.tsv`).

Medians of 3 runs (10 for a single pair), seconds and MB read. Layouts as (inner chunk, shard) in
codes: A (4,096, 4,194,304), B (32,768, 16,777,216), C (262,144, 67,108,864). Local cold: the
store's pages evicted before each run. HTTP: a static Range-serving server on localhost; +20 ms adds
a 20 ms wait per request. zarr 3.4.1, async concurrency 10. Load average 36.02, 39.58, 42.35 at the
start, 37.32, 37.61, 39.87 at the end.

| read | local cold A | local cold B | local cold C | HTTP cold A | HTTP cold B | HTTP cold C | HTTP +20 ms A | HTTP +20 ms B | HTTP +20 ms C | MB read A / B / C | HTTP requests A / B / C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 Mb LD matrix (row range) ¹ | 21.06 s | 3.70 s | 2.02 s | 21.60 s | 4.19 s | 2.22 s | 32.16 s | 9.34 s | 4.21 s | 432 / 425 / 403 | 62 / 19 / 14 |
| 500 scattered variants | 993 ms | 360 ms | 310 ms | 1.80 s | 594 ms | 631 ms | 2.28 s | 1.62 s | 539 ms | 84 / 109 / 272 | 556 / 402 / 313 |
| clump one region's hits | 49 ms | 38 ms | 26 ms | 66 ms | 47 ms | 35 ms | 94 ms | 85 ms | 69 ms | 9.6 / 9.6 / 9.9 | 5 / 3 / 2 |
| single pair | 5.3 ms | 5.2 ms | 7.5 ms | 8.2 ms | 7.4 ms | 10 ms | 48 ms | 48 ms | 51 ms | 0.024 / 0.066 / 0.460 | 2 / 2 / 2 |
| a variant's downstream Near Band row | 6.5 ms | 5.1 ms | 5.8 ms | 8.5 ms | 7.3 ms | 8.9 ms | 50 ms | 47 ms | 49 ms | 0.045 / 0.066 / 0.391 | 2 / 2 / 2 |
| a variant's upstream Near Band column | 5.79 s | 3.13 s | 392 ms | 7.22 s | 3.70 s | 971 ms | 5.03 s | 4.01 s | 991 ms | 661 / 659 / 638 | 184 / 27 / 15 |
| a variant's upstream Far Pairs | 11 ms | 9.9 ms | 12 ms | 14 ms | 15 ms | 16 ms | 99 ms | 100 ms | 99 ms | 0.147 / 0.147 / 0.147 | 4 / 4 / 4 |

¹ HTTP +20 ms ran the matrix once, on the largest of the three regions (21–24 Mb, 700 MB read); HTTP
cold on that region took A 34.25 s, B 10.92 s, C 3.86 s.

- The 3 Mb matrix reads its rows whole, including partners past the region. So it reads about
  twice the codes the matrix needs: 268–700 MB per region, depending on the region's density.
- Clumping a region's hits is a 10 MB submatrix read, 26–49 ms locally. The hits are 97–173
  variants within 1.5 Mb of a lead, at r² ≥ 0.05 to it.
- With 20 ms added per request, the small reads take about 40 ms longer. zarr-python merges
  neighbouring byte ranges in a shard, so even the scattered reads make tens to hundreds of
  requests, not thousands.
- Opening the slice's store (positions and `indptr`, 0.38 MB) took 26–43 ms locally and
  0.04–0.38 s over HTTP
  (`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/access_open.tsv`).
- The Proxy Index row that replaces the column read is in §11's table.

**How ogld reads** (ogld#4). Opening a store reads its provenance and its chromosome ranges, and
refuses a `format_version` or `encoding` it does not implement (§13). A search for ALIDs reads the
ALID index's two arrays, and only when they pay for themselves (ogld#42, ADR 0010). A list naming
about a two-thousandth of the Variant Set's rows or more is searched in the index: an ALID that
fits the slot finds its row there, without reading `variants/position` or
`variants/alid` at all. A shorter list takes the exact scan for every ALID, which reads a few
string chunks where the index reads the whole of itself. An ALID wider than the slot, and one the
index does not hold, is found by that same scan at its position, which reads the
chromosome's positions and the ALID strings there, as every ALID was found before the index. So a
search can only be faster than it was. A chromosome's `position`
range and its Near Band's `indptr` are read the first time a request touches that chromosome,
through the numeric cache, and kept. Its Proxy Index's `indptr` is read only when used;
its Far Pairs' `indptr` the first time a request needs its pairs
beyond `near_bp`, so `proxies` and `clump`, which never do, read no more than they did before
the Far Pairs (ADR 0007). From 0.2.0 a chromosome's `partners/<chrom>/` adds its group's attributes
and the `frames_idx` chunk, read the first time a request asks for one of its rows, and a frame
cache of its own (§10's Partner Rows paragraph). Every other read the reader makes goes through its cache of decoded inner chunks
(`ogld.store.location.ChunkCache`). It reads the inner chunks that hold the elements a request
needs, whole, which is what Zarr fetches and decodes in any case. So a query that reads a few
pairs at a time fetches each chunk once: clumping the Test Store's 1,056 variants, one read per
lead, took 3.6 s reading through Zarr call by call and 0.7 s with the cache, against 0.6 s for the
walking skeleton's arrays held whole. A matrix resolves its pairs about 4 million at a time,
which bounds its working memory beside the matrix itself. Validation's rules read through Zarr
directly, in bounded windows: one pass over each chromosome, and a second read of the blocks
`index.near` flags in a store it fails (§12).

**Direct numeric decoding** (ogld#40; ADR 0009). For a one-dimensional numeric array whose
metadata declares exactly sharding_indexed with bytes then zstd internally and bytes then
CRC32C in an end-located shard index, `location.py` reads through the store's async interface
using its `decoding.py` backend. A one-byte dtype's `bytes` codec declares no byte order, and the
backend reads it as it lies (ADR 0010); that is how the ALID index's keys are decoded directly. That backend owns shard/frame parsing and one process-shared
seven-worker I/O/decode pool. Both declared
byte orders are honoured. The index's uint64 offset/length pairs are absolute; both all ones
means an empty chunk, which reads the declared fill value, just like a missing shard. CRC32C
mismatches, malformed zstd frames and decoded-length mismatches raise. No private zarr codec/index API is called. Every other chain,
including strings, goes through zarr unchanged. Zarr remains the format and the writer.

The same per-read planner and cache budget apply to both paths. Each directly decoded native
chunk owns its buffer and transfers it to the cache without copying. The planner reserves
one chunk for trimming a padded tail, or half the allowance for non-native byte-order
conversion scratch. Each frame's full decoded length is checked before adoption. At least
one chunk can be read even when its buffers exceed the allowance. No second decoded batch
is queued. Each range submits at most seven decode jobs. A failed piece cancels sibling
fetches and drains running decode jobs before returning. A cancelled shard-index waiter also
drains the shielded shared index fetch without cancelling another reader's request.
Compressed ranges, codec workspace and small index/job bookkeeping are additional to the
decoded-byte bound;
the cache budget does not bound RSS. The pool is reset after fork and shared across readers,
not multiplied per chromosome. Applications budget zarr's I/O helpers separately.

**Clump prefetch** (ogld#40 round 2; ADR 0009). Before a chromosome's greedy pass, the
reader warms the union of chunks its candidate hit pairs can need, only when the buffers
and shard indexes fit the cache budget and at most 65,536 chunk keys are needed. A chromosome
that does not fit uses a sliding positional warm range of W around a lead, halved until it
fits; ordinary reads still resolve partners outside a smaller warm range. Zero cache skips
warming. Planning is O(hits + 65,536), not O(hit pairs). Sparse reads stream bounded chunk
groups, so an oversized warm range never becomes a whole-chromosome decoded pin.

Prefetch uses a separate 256 MiB credit limit for compressed ranges, held until decode
finishes, in addition to the decoded flight allowance. It coalesces at most 512 KiB gaps
into at most 16 MiB ranges and decodes only requested chunks. A single oversized chunk is
the exception to either allowance. Actual store requests remain bounded by async.concurrency.
The async store wrapper bridges exact LocalStore reads through its public get_sync method
on the same pool as decoding; subclasses and HTTP retain async get. No global zarr setting
changes. Known-size zstd frames are checked against RFC 8878 before decoding into a fixed
buffer; unknown-size frames retain allocating decode and output-length validation.

**The cache and its bound** (ogld#4, ogld#32). An opened store's cache keeps decoded chunks and
shard indexes up to `cache_bytes`, dropping the least recently used first. The caller sets it
when opening the store, `ogld.open(location, cache_bytes=...)`: 4 GiB by default, and `0` keeps
nothing, so every read fetches its chunks and shard indexes again. Anything but an int ≥ 0 is an
`InputError`. 4 GiB holds 8,192 of layout C's 512 KiB Near Band chunks. At that size ogld#8
measured repeated reads warm, and at the earlier 256 MiB not (owner, 12 November 2026):

- four Analyses clumped again in 38 s, against 80 s cold;
- 500 chr22 variants read again in 0.13 s, against 1.7–2.9 s.

A single cold pass reads distinct chunks, which no budget saves.

Each kept chunk owns its memory: adopted from a directly decoded single-chunk buffer, or
copied out of a fallback run, never a view retaining uncounted neighbouring chunks. It counts
an upper bound on what it holds. So the budget bounds the memory kept. A numeric chunk counts its buffer, and a
shard index its bytes. A string chunk stays numpy's variable-width strings, as zarr returns them,
which keep each string over 15 UTF-8 bytes in an arena that `nbytes` leaves out (about 2.2 times
`nbytes` for 16-character ALIDs). So it counts `nbytes` + 1.25 × Σ(bytes + 8) + 512, the sum over
those strings: the arena grows by a quarter at a time, a string's size header is at most 8 bytes,
and the allocator takes 328. Measured by tracemalloc on 72 chunks of 1–16,384 strings of 1–20,000
characters, the count was never below what a chunk held.

The count visits no string in Python (ogld#32). numpy counts each string's characters, and one
fixed-width copy of the chunk, at most 256 bytes wide, proves the strings of at most 256
characters ASCII, a byte a character, so their bytes are counted exactly. Every other string
counts 4 bytes a character, UTF-8's most; no ALID or allele the engine forms needs that. On 16,384
ALIDs the count takes 1.6 ms, against 2.4 ms string by string. Fixed-width text would be counted
exactly, but one 2,000-character indel ALID would make its 16,384-row chunk 131 MB.

One lock guards the cache, so threads sharing a reader keep one copy of a chunk, and the count
stays the sum of what is kept. `tests/test_store_reads.py` measures what the cache holds each
time it keeps a chunk or an index, while 16 threads fetch overlapping runs, and holds it to the
budget. The bound is the cache's, and per opened store, so a process with the five HGDP+1kGP
stores open keeps up to 5 × `cache_bytes`. Beside it a reader holds:

- each chromosome's positions and `indptr` arrays it has touched (above);
- what each read in progress holds: the chunks it asked for, one batch of fetches in flight
  (at most 256 MiB, below) with its copy into kept chunks, and its result. The bound is per
  read: reads on one store from several threads each hold a batch of their own, so n reads at
  once hold up to n batches beside the budget.

**The cache keeps shard indexes** (ogld#32). zarr reads a shard's index before the chunks it
needs from that shard, and reads it again on every read. Over HTTP, then, each read of a shard
waited two round trips. The reader opens its store through a wrapper that keeps each index it
fetches in the chunk cache, in the same budget and the same least-recently-used order. §6 puts
the index at the shard's end. zarr reads it as the shard's suffix and reads nothing else that
way, so the wrapper keeps suffix reads and passes every other read through. A read of a shard
read before then pays one round trip; the first read of a shard still pays two, since the index
says where the chunks are. A Near Band shard's index at layout C is 4,100 bytes: 256 chunks × 16,
and a 4-byte checksum.

**A read fetches its runs together** (ogld#32). The chunks a read needs and the cache does not
hold fall into runs of neighbouring chunks. Each run is cut where a shard ends, so each piece
reads one shard. The pieces are fetched in batches, every piece of a batch at once on zarr's event
loop. A batch holds at most zarr's `async.concurrency` pieces (10 by default) and at most 256 MiB
as the cache counts memory, two of layout C's Near Band shards; a larger run is cut to fit. A
read holds one batch at a time: it lets each piece go once its chunks are copied out or their
owned buffers adopted, before it fetches the next batch.

A numeric chunk's decoded size is its buffer, known before it is fetched. A string chunk's is not
known until it is decoded: its strings may be any length. So a string array's batches are planned
at the most an element has held, by the cache's own count, in any chunk of that array the reader
has counted. The first time a reader reads a string array, it fetches one chunk alone to count
it. A string batch then holds at most 256 MiB, unless one of its chunks holds more a string than
any counted before. Then it holds more by that chunk's excess, and later batches are planned at
the larger rate.

A batch waits one round trip for its chunks, and one more before it when a shard's index is not
yet kept, however many runs it holds. Two runs of one shard in a batch fetch its index once: the
second awaits the first's fetch. A piece that fails cancels the others of its batch. Before
ogld#32, each run was its own Zarr call, one after another, and each call waited for the shard
index and then for the chunks.

- **Over HTTP** a store is opened through zarr's fsspec store, whose HTTP filesystem uses aiohttp.
  Both are runtime dependencies. A static host that honours byte ranges is enough: no node is
  found by listing a directory.
- **A lost chunk is refused, in every array.** It reads as its array's fill value. So a reader
  first refuses any array whose fill value a valid store could hold there (§6, `envelope.fill`):
  the variant columns when it opens the store, a chromosome's arrays when a request first
  touches that chromosome, and its Far Pairs' when a request first needs them, each before any
  element of it is decoded. An array the store lacks is refused, naming it. Then a read refuses every value
  no valid store holds, naming the array: the reserved code in an `r` array, a negative `indptr`
  or `partner`, a position ≤ 0, `""` in a string column (which would otherwise make a present ALID
  Absent) and a frequency outside (0, 1). A `partner` filled with 0, zarr's default, would
  otherwise read a lost chunk as partner row 0.
- **Measured on the Test Store, re-sharded** into 512-element inner chunks, 4 to a shard, which
  reads exactly as the committed one (`tests/test_store_reads.py`). On the × 511 Test Store
  (ogld#16), a 3-variant matrix over HTTP fetched 759 of chr22's 97,699 Near Band bytes, in 2
  requests, a shard's index and then its chunk. In all, opening the store and the matrix took 36
  requests and 44,824 bytes, most of it metadata documents, before ogld#32 and after it alike:
  each shard is read once, so no kept index is used. A reader that read whole arrays fetched
  96,582 Near Band bytes. The same read again fetched none.
- **Read latency at scale** on the published EUR store, from local disk and over HTTP with
  latency added, is ogld#8's `docs/benchmarks/latency.md`, and the reader after ogld#32's four
  changes is `docs/benchmarks/fast-reader.md`.

## 11. The upstream half: the Proxy Index

**Settled** by the owner on 7 November 2026 (epic #1, comment of that date): a sparse symmetric
Proxy Index (§5). This section keeps the measurements that decided it; the merge of the Far
Pairs into the index that the owner made on 8 November 2026; and its reversal on 12 November
2026, on ogld#8's measurements, which put the Far Pairs back beside the index (ADR 0007).

**The problem.** The Near Band stores row i as r(i, i+1 … the last variant ≤ `near_bp` away): the
upper triangle. So a variant's upstream partners, the ones before it, are scattered one per row
across every row in the preceding 3 Mb: about 16,000 rows in the EUR slice. Anything that needs a
whole column pays for this, `proxies()` first.

The prototype measured the column read and two alternatives on EUR chr22
(`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/stores.json` for the sizes,
and `proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/access_tables.md` for the times):

- **the outline (upper triangle) alone**;
- **a symmetric band**: row i holds its upstream partners, then its downstream ones, so a column
  is a row. Near Band bytes ×2.0: 3.33 GB against 1.65 GB on the slice;
- **a sparse symmetric proxy index**: per variant, every partner within 10 Mb on either side with
  r̂² ≥ t_far. It answers `proxies(r2 ≥ t_far)` with one row, and adds 5.2% to the slice's bytes
  (85 MB against 1,648 MB). A proxies request at r2 below t_far still reads the column.

These were measured on layout B:

| proxies for one variant, within 3 Mb | local cold | local warm | HTTP cold | HTTP +20 ms | MB read | HTTP requests |
|---|---|---|---|---|---|---|
| outline: the row (downstream half) | 5.1 ms | 2.2 ms | 7.3 ms | 47 ms | 0.066 | 2 |
| outline: the column (upstream half) | 3.13 s | 3.28 s | 3.70 s | 4.01 s | 659 | 27 |
| symmetric band: one row, both halves | 5.6 ms | 3.8 ms | 7.1 ms | 50 ms | 0.122 | 2 |
| proxy index, r² ≥ t_far: one row, both halves, ≤ 10 Mb | 9.7 ms | 4.8 ms | 12 ms | 97 ms | 0.045 | 4 |

The column is the problem. On layout B one variant's upstream half reads 659 MB and takes 3.1 s,
against 0.066 MB and 5 ms for its downstream half. It costs the same warm as cold, so the time
goes in decoding, not disk. On layout C the same column decodes in 0.39 s but still reads 638 MB.
Over a network, every `proxies()` call would move about 640–660 MB. The owner chose the proxy
index: the symmetric band doubles every store, and the index costs 3.1–8.7%
(`proto/1-ld-store-layout@65f94ca:prototypes/ld-store-layout/results/upstream_alternatives_sizes_inner262144.tsv`).

**The index holds every Far Pair.** The index holds every pair within `far_bp` at r̂² ≥ t_far,
in both directions. A Far Pair is exactly such a pair more than `near_bp` apart (§3), so the index
holds every Far Pair, twice. A separate Far Pairs component holds nothing the index does not.
The index already copies the Near Band's pairs at r̂² ≥ t_far, by design, so merging did not make
the store copy-free: it removed the Far Pairs' own copy of each 3–10 Mb pair.
The owner adopted the merged form on 8 November 2026 (epic #1): one dense Near Band and one
sparse, symmetric, thresholded index. Either form gives the same answers and the same
refusals:

- a pair within `near_bp` is still answered from the Near Band;
- a pair beyond it is a Far Pair exactly when its |code| ≥ `far_code_floor`, the rule that put it
  in either component, so it reads the same Stored r or the same Bound (§8);
- the refusals depend only on r2, the window, t_far and the distances (§9), none of which the
  merge touches. `tests/test_format.py` holds the answer and refusal tables, and they do not
  change with the layout.

The trade-off as the owner weighed it on 8 November, on the prototype's numbers at k = 0, with the
five stores' as built at k = 10 (§6) beside them:

| | Near Band + Proxy Index (8–12 November) | Near Band + Far Pairs + Proxy Index (adopted 12 November) |
|---|---|---|
| bytes per panel | the index alone: 11.3–25.7 GB | also the Far Pairs, once more: 0.09 GB (EUR) to 1.39 GB (AMR), 0.03–0.76% of a store, and one more `indptr` per chromosome. As built at k = 10 (§6): 0.16 GB (AMR) to 0.49 GB (AFR) on disk, 0.20–0.40% of a store; raw, at 6 bytes a pair, 0.53–1.15 GB, EUR's 0.80 GB about 1% of its store |
| builder paths | one sparse output, symmetric, from the Near Band rows and the 3–10 Mb scan | two: the upper-triangle Far Pairs and the symmetric index, from the same scan |
| validation rules | `index.*`: six rules (§12) | the six, the four `far.*` rules, and `far.index`, holding the Far Pairs equal to the index's far entries |
| a pair beyond `near_bp` | binary search of one index row: as built at k = 10, about 616 entries on average for EUR (5.29×10⁹ entries over 8.59 M variants, §6); the prototype's k = 0 figure, which the owner weighed, was 485 | binary search of one Far Pairs row: as built, 15.5 entries on average for EUR (§6); the prototype's figure was about 3 |
| a matrix's far pairs | the index rows of the listed variants, about 3.7 KB each raw for EUR as built (2.9 KB on the prototype's figures) | the Far Pairs rows, about 93 bytes each raw for EUR as built |

On 8 November the read costs looked to differ by kilobytes against a Near Band read of hundreds
of megabytes (§10), so reads did not decide it. Both forms keep copies that must agree: the
index's pairs within `near_bp` repeat Near Band codes in either form, and `index.near` holds them
equal. What merging saved was the Far Pairs' own copy of each 3–10 Mb pair, the builder output
that writes it, and the cross-component rules for 3–10 Mb.

**Merged** by the owner on 8 November 2026 (epic #1), after asking what the merge costs in query
time. The reasoning was:
- `proxies`, a single pair, a matrix for a list of ALIDs and `clump` on a list of hits are
  unchanged. Each reads one row per variant, and a row of either component sits in one inner
  chunk, so the requests are the same.
- A region matrix up to `near_bp` wide needs no pair beyond `near_bp`.
- A region matrix wider than `near_bp` reads each region variant's index row instead of its Far
  Pairs row. That is about 3–9% more bytes than the Near Band read it already makes: the index's
  share of a store, about 5% for EUR and 9% for AMR.

**Reversed** by the owner on 12 November 2026 (ogld#8, ruling 2; ADR 0007), before any LD Store
was published, on ogld#8's timings of the published HGDP+1kGP EUR store
(`docs/benchmarks/latency.md`). The bytes held; the requests did not:
- **A region 10 Mb wide** read 6.2–9.3% more bytes for its index rows, as estimated. But it
  walked every entry of every row, 11.4–16.6 M of them, to place 6,228–11,651 Far Pairs, and
  that took 13–30% of its local time.
- **A matrix of 500 variants on one chromosome** is not one row per variant in one chunk. An EUR
  chr22 index row holds about 380 entries, so a 16,384-entry chunk holds about 43 rows, and 500
  rows of chr22's 119,632 touched 433–448 chunks of each of `index/22/partner` and
  `index/22/r`: about 890. chr22's 101,886 Far Pairs fill 7 chunks of each array, and the same
  rows touch 5–6 of them, about 11. The far part was 32% of the read locally and 58–60% at 10
  and 50 ms per request.

So the Far Pairs are their own component again (§1, §5). A matrix and a region read them, and no
index entry; `proxies` and `clump` read the index, as before; `far.index` holds the two to one
code per pair (§12). The cost is the one this section's table gives.

chr22 is the best case: 0.85 Far Pairs a variant, against EUR's 15.5 and chr11's 161.5 (ogld#8's
review; `docs/benchmarks/results/far_pairs.json`). **Measured after** (ogld#31,
`docs/benchmarks/far-pairs.md`), on the published EUR store with its Far Pairs derived, so before
and after read the same bytes:
- **500 variants drawn uniformly on one chromosome** (seed 8) read 10 Far Pairs chunks on
  chr22 where they read 894 index chunks, and 102 on chr11 where they read 1,058. Requests fell
  by 61% and 62%, and the read, cold at 50 ms a request, from 129 s to 53 s on chr22 and from
  161 s to 63 s on chr11. Across ogld#8's uniform draws the Far Pairs touch 10–91 times fewer
  chunks than the index (10–29 on the eight chromosomes with more than 10 Far Pairs a variant);
  a list clustered where they are densest, about 2–3 times fewer (1.7 for 500 rows at chr11's
  45–55 Mb, 2.8 for the spec review's list with 300 of its 500 there).
- **A region 10 Mb wide** spends 12–491 ms on its pairs beyond `near_bp` where Far Pairs are
  sparse, against 1.7–5.1 s; its total is its Near Band read. Where they are densest, chr11's
  45–55 Mb, its rows hold 68.75 M Far Pairs against 196 M index entries: 203 MB of rows read
  against 709 MB, and 12 s locally and 24 s at 50 ms against 30 s and 47 s.

## 12. Validation rules

Each rule runs on write and on demand, by name. A rule reads in bounded windows and never loads a
whole Near Band. A rule fails a store when any target breaks it. `ogld.format.VALIDATION_RULES`
lists the names in this order, and `ogld.validate` returns findings keyed by them.

| rule | checks |
|---|---|
| `envelope.arrays` | the hierarchy is Zarr v3, and every §1 array exists with its §1 dtype and the length §2, §4 and §5 give it, the Far Pairs' included and the Partner Rows' left to their own rules in 0.2.0; `n_variants` is the variant table's length |
| `envelope.fill` | the fill value of every array is one a valid store never holds there (§6): −32768 in every `r` array, at every scale, so an unwritten chunk can never read as r = 0, and so on for the others |
| `provenance.complete` | every §7 key is present with its type, null only where §7 allows; other keys are allowed |
| `provenance.encoding` | `format_version` is one this package reads, and `encoding` is one it implements: exactly the four keys, `int16`, −32768 reserved, `half_to_even`, and a whole `r_scale` from 1 to 32767 (§3). A reader refuses any other before decoding (§13) |
| `provenance.distances` | `near_bp` is 3,000,000, and `near_bp` < `far_bp` ≤ 10,000,000 |
| `provenance.threshold` | `adjustment.k` equals the number of `adjustment.covariates`. When a PC File was read (`pc_file` is not null), `k_min` ≤ k ≤ `k_target`; when none was, `k_min` is null, and a `genotypes` source has k = `k_target` = 0. An `ld_matrix` source has `k_target` null. `t_far` equals max(0.001, q / (n − k − 1)) recomputed from provenance, with n − k − 1 > q (so t_far < 1); and `far_code_floor` equals its value recomputed at the recorded scale |
| `variants.chromosomes` | the `chromosomes` ranges are in chromosome order, contiguous and cover every row |
| `variants.order` | positions ascend within each chromosome, ties broken by ALID ascending, with no repeated ALID |
| `variants.identity` | each `alid` is the identity function's ALID for its chromosome, position, `a1` and `a2` (ogld#3) |
| `variants.frequency` | every `a1_frequency` is finite and in (0, 1) |
| `alid_index.rows` | one entry a variant: `variants/alid_index/row` holds a row of the Variant Set for each entry, none of them twice, and every ALID whose UTF-8 form fits the slot has exactly one entry — so a search that finds no key means the ALID is not in the table (§2, §8) |
| `alid_index.order` | the entries' keys ascend bytewise, strictly: the order every search over them depends on (§2) |
| `alid_index.keys` | each entry's key is the full ALID of the row it names, NUL padded — so that row's ALID fits the slot, no key was truncated into it, and no key is one no variant has (§2; opengwasdb issue #127) |
| `alid_index.count` | `variants/alid_index/alid` holds 32 bytes an entry, and `counts.alids_left_out` is the table's own count of ALIDs wider than the slot (§2, §6, §7) |
| `near.rows` | `indptr` starts at 0 and has n_c + 1 entries; each row's length equals the count of later variants within `near_bp` from `position`. So rows are contiguous and stay within `near_bp` |
| `near.codes` | every code is one of the store's encoding: none is −32768 and none is past the scale, \|code\| ≤ `r_scale`, so \|r̂\| ≤ 1 holds after dequantisation |
| `index.rows` | `indptr` starts at 0, has n_c + 1 entries and ends at the length of `partner` and `r`; within a row partners are strictly ascending, in [0, n_c), and never the row itself |
| `index.codes` | every code is one of the store's encoding, as for `near.codes` |
| `index.distance` | every entry's partner lies within `far_bp` of its row's position |
| `index.threshold` | every entry's \|code\| ≥ `far_code_floor` |
| `index.symmetric` | (l, m) is an entry with code c exactly when (m, l) is an entry with code c |
| `index.near` | every entry within `near_bp` carries the Near Band's code for that pair, and every Near Band code with \|code\| ≥ `far_code_floor` is an entry: the index's near half is the band, thresholded |
| `far.rows` | `indptr` starts at 0, has n_c + 1 entries and ends at the length of `partner` and `r`; within a row partners are strictly ascending, in [0, n_c), and each above the row: each Far Pair is in its earlier variant's row |
| `far.codes` | every code is one of the store's encoding, as for `near.codes` |
| `far.distance` | every Far Pair lies more than `near_bp` and at most `far_bp` apart |
| `far.threshold` | every Far Pair's \|code\| ≥ `far_code_floor` |
| `far.index` | the Far Pairs are exactly the Proxy Index's entries beyond `near_bp` in each pair's earlier row, with the same codes: every such entry is a Far Pair and every Far Pair is such an entry. With `index.symmetric`, they are the index's far pairs, once each |
| `partners.files` | every chromosome of the Variant Set has a `partners/<c>/` group with both of its arrays, each one chunk of its whole length (§5); its `rows` attribute is n_c, `frames` is ⌈n_c / G⌉, `frames_idx` holds F + 1 strictly ascending offsets from 0 to `frames_bytes`, and the two chunk files are the ones their recorded SHA-256s name and physically as long as their metadata says — not the prefix a hash reads; and no entry of `partners/` begins with a dot, which is a temporary group a crashed writer left behind (partner-rows §5.1) |
| `partners.frames` | every frame decompresses with a valid content checksum, its k is the rows it is due — G, or the chromosome's last frame's fewer — and its payload is exactly 4 + 4k + 6E bytes; a frame that fails is a finding naming the chromosome and the frame (partner-rows §5.2, §5.4) |
| `partners.rows` | within a row, partners are strictly ascending, in [0, n_c), and never the row itself; and the frames hold the group's recorded `entries` |
| `partners.codes` | every code is one of the store's encoding, as for `near.codes` |
| `partners.distance` | every entry's partner lies within `far_bp` of its row's position, either side |
| `partners.threshold` | every entry's \|code\| ≥ `partner_code_floor` within `near_bp` and ≥ `far_code_floor` beyond it |
| `partners.symmetric` | (l, m, c) is an entry exactly when (m, l, c) is, by `index.symmetric`'s 64-bit hash sums |
| `partners.near` | every entry within `near_bp` carries the Near Band's code, and every Near Band code at the Partner Floor is an entry: the near half is the band at that floor, both ways. `index.near`'s method and recheck, at the Partner Floor |
| `partners.far` | the entries beyond `near_bp`, in each pair's earlier row, are exactly the Far Pairs, with the same codes: `far.index`'s comparison, read against the Partner Rows |
| `provenance.partner_threshold` | `t_partner` is no larger than `t_far`, and `partner_code_floor` recomputes from `t_partner` at the recorded scale (§3) |

No rule checks the chunk shapes, shards or codecs of §6's arrays; the Partner Rows' two are one
chunk each by construction (§5.1), which a reader's one range request per frame depends on, and
`partners.files` holds them to it. No rule can check that every Far Pair was found: that needs the
genotypes, and ogld#5's tests check it against an independent reference on the Test Panel. Each rule has a store that breaks it, in
`tests/contract/test_store.py` (`BREAKERS`), and the Test Store passes all 37.

**The Partner Rows' rules (format 0.2.0).** The ten rows after `far.index` are the component's,
and `docs/spec/partner-rows.md` §10 is their reasons. They read the frames of
`partners/<chrom>/` in bounded blocks — whole frames, never a whole chromosome — sharing this
section's one pass per chromosome and its thread option. `partners.files` streams a chromosome's
`frames` chunk file in windows to check its hash, and reads its `frames_idx` whole; the other nine
read the blocks. `partners.near` with `partners.far` hold every entry equal to the Near Band's
code within `near_bp` or the Far Pairs' beyond it, so with `partners.symmetric` the component
holds every pair it should and no other.

**What a 0.1.0 store is judged by.** A store has one `format_version` (§13), and 0.1.0 has no
Partner Rows, so a 0.1.0 store with no `partners/` passes these ten with nothing to judge: every
one of them says so and moves on. A 0.2.0 store has the component, so a missing
`partners/<chrom>/` is a finding of `partners.files`, and a rule that needs its frames reports
that it cannot be checked — neither passes it. A 0.1.0 store that carries a `partners/` anyway is
judged by them, as an older store that carries a later key is held to its type (§7).

**On write and on demand** (ogld#4). `ogld.validate(location)` runs the rules on a store on disk
or over HTTP and returns every rule's findings, an empty list for a rule the store passes.
`StoreWriter.finish` runs them on the store it wrote, on the threads the builder gives it, judged
with the provenance it assembled (§7) as JSON will hold it, and writes that provenance only when
all 37 pass. Otherwise it raises
`ValidationFailed`, naming each failing rule and where, and the store keeps no provenance, so a
reader refuses it (§13) rather than read it as a short one.

**What a rule says when it cannot judge.** A rule whose prerequisite is missing or malformed,
such as `near.rows` without a `near_bp` or `index.distance` without a readable `indptr`, reports
a finding that it cannot be checked, naming what it needed. An array that will not decode as its
metadata declares is a finding that it cannot be read. Neither passes the rule, and neither stops
the other rules. So one fault can fail several rules; the breakers' tests assert only that each
fails its own, and that a breaker fails the same rules on a × 511 and a × 2047 store. The rules
that judge a code against the scale, `near.codes`, `index.codes`, `far.codes`, `partners.codes`,
`provenance.threshold`'s far code floor and `provenance.partner_threshold`'s partner code floor,
need the store's encoding: with one this package does not implement they cannot be checked, as
`provenance.encoding` fails.

**How the rules read.** The rules read through Zarr directly. A chromosome's positions and
`indptr` arrays are read whole, as a reader keeps them (§10). Near Band codes and Proxy Index
entries are read in blocks of rows holding at most 8,388,608 codes and 1,048,576 entries, and the
variant columns 1,048,576 rows at a time, so no rule loads a whole Near Band or Proxy Index. An
entry holds an eighth as many as a code because the rules work on each entry through about 100
bytes of arrays, and on each code through about 6: a block's working arrays stay near 150 MiB
however dense the index is. A block holds at most as many Far Pairs as entries, a bound that binds
only in a store that fails `far.index`: a valid row holds fewer Far Pairs than index entries.

The twenty rules that read codes and entries one by one, `near.codes`, the six `index.*` rules,
the five `far.*` rules and the eight `partners.*` rules that read frames, share one pass over each
chromosome: each block of rows is read once and handed to all twenty, so each code, entry, Far Pair
and partner entry of a store that passes is decoded once per validation. `far.index` compares each
block's Far Pairs with that block's index entries beyond `near_bp` in their earlier rows, and
`partners.far` compares the Partner Rows' entries beyond `near_bp` with those same Far Pairs, so
neither carries anything from one block to the next. A caller may give the pass t threads
(`ogld validate --threads t`, or `StoreWriter.finish(sourced, threads=t)`); t blocks are then read
and checked at once, and no more are held. A block the Partner Rows join is not bounded by its
rows: a later row holds an entry for every earlier partner within `far_bp`, so a few thousand
adjacent rows near a chromosome's end carry every pair before them, and the block's own band
codes say nothing about that. Such a block is planned by decoding its frames, and it ends where
its entries reach 524,288; a frame that alone holds more is a block to itself, because a frame is
the component's unit of reading (§5.4). At that budget the ten rules' working arrays measure
**42 MB** on a store of 1,500 variants all within one 3 Mb window with every band code at the
Partner Floor — a store where one unbounded block would hold 2,248,500 entries — so the
component's share of a block stays within the 150 MiB the band's and the index's limits are sized
for. `near.codes`, `index.codes` and `far.codes` read their `r` array in windows of their
own instead when its `indptr` cannot delimit the rows or does not end at `r`'s length, as when it
ends short of `r`: windows of 8,388,608 elements, codes or entries alike, t of them at once.

`alid_index.keys` and `alid_index.rows` read the ALID index's two arrays whole — 275 MB of keys and
34 MB of rows on HGDP+1kGP EUR's 8.59 M — and walk the variant table in blocks beside them, so each
entry is judged against the row it names without a scattered read of the table. `alid_index.order`
reads the keys in blocks, and `alid_index.count` counts the wide ALIDs from the table's own column.

A chunk that will not decode, such as a corrupt zstd frame, is a "cannot be read" finding of each
rule that reads it; a Partner Rows frame that will not decompress, or whose k or payload length is
not what §5.2 says, is `partners.frames`' finding naming the chromosome and the frame, and a
"cannot be read" one of each other rule that needs that block's entries. An `indptr` that ends
past its array is a "cannot be checked" finding of each rule that reads by it. A rule that stops so reports the faults its pass had finished checking,
then that it cannot be read or checked, and fails either way. `index.near` finishes its check of
later-row entries only at a chromosome's end, so a stop inside a chromosome drops that check's
findings there, though never the failure.

- `provenance.complete` checks every key §7 names: those of `ogld.format.REQUIRED_KEYS` and the
  keys inside them, `variant_filters`, `variants_dropped`, `pc_file`, the PC Check's settings and
  its `null`, each `scan` entry's fields, `pc_diagnostic` with its `pairs` and the five quantiles
  of `near_100kb_change`, and `software`'s `ogld`, `numpy` and `zarr`. A null is allowed only
  where §7 allows it: for `panel.samples_sha256` and `pc_diagnostic` only with an `ld_matrix`
  source, and for the two measured rates exactly when `pc_diagnostic.pairs.n_pairs` is 0.
- `provenance.threshold` also holds `t_far_q` to 20.84.
- `index.symmetric` reads each entry once. It adds a 64-bit hash of (earlier row, later row,
  code) to one of two sums kept for the pair's earlier row: one for the entries in that row, one
  for those in the later row. A row whose two sums differ is the earlier row of a pair held one
  way only, or with two codes, and the finding names it. Two different sets of entries give equal
  sums by chance about once in 1.8×10¹⁹; the sums take 16 bytes per row.
- `index.near` compares each block's entries in a pair's earlier row with that Near Band row at
  the far code floor, both ways. Each entry in a pair's later row within `near_bp` must carry the
  band's code: it adds a 64-bit hash of (earlier row, later row, code) to one sum kept for the
  pair's earlier row, and each band pair from that row at the floor adds its hash to the other.
  The pass also keeps, per row, the first and the last row whose entries name it: with the sums,
  32 bytes per row. At a chromosome's end, if any row's sums differ, the blocks holding an entry
  that names such a row are read again, in the pass's windows and on its threads, and each of
  their entries that names such a row is checked against the band's code for its pair, fetched
  by coordinates. So a store that passes reads its band and its index once. That such a
  later-row entry exists at all is `index.symmetric`'s finding.

## 13. Compatibility

`format_version` is `MAJOR.MINOR.PATCH`. A reader MUST refuse a `format_version` or an `encoding`
it does not implement, naming the version or the encoding, rather than decode it; ogld's reader
refuses at open, with an `ogld.InputError`. A change to any array's dtype, encoding or meaning, or
to §8's answers, is a new format version. A change of chunk or shard shape is not, and neither is
a store's scale: any whole scale from 1 to 32767 in the `int16` container is version `0.1.0`, and
each store records its own (§3, §7).

**Version `0.2.0`** (the decision; the bump lands with the writer, #67, and the derivation in #69
writes it too). A store has one `format_version`, so what 0.2.0 requires is everything below, not
a stage of it: a 0.1.0 store is upgraded by deriving its Partner Rows, and a 0.2.0 store without
them is not one. It:

- adds `partners/<chrom>/` (§1, §5): the Partner Rows, whose frames are compressed copies of codes
the store already holds, plus the four provenance keys that describe them and `software.zstd`,
  less `counts.index_entries` (§7);
- refuses `threshold_below_t_partner`, which turns an answered 0.1.0 request below t_partner within
  `near_bp` into a Refusal (§9);
- reads `partner_row` from a frame cache, in place of the Proxy Index row and the Near Band's row
  and column (§10);
- has no Proxy Index: `index/`, its writer, its six rules and `far.index` go (#72), and the Far
  Pairs and the Near Band stay as they are. A reader of a store that still holds both components
  reads the Partner Rows, as #67-#71's does.

A 0.1.0 reader reading a 0.2.0 store sees groups it does not know and ignores them (§1), answering
from the Near Band, the Proxy Index and the Far Pairs if they are there. That is why the version
is recorded: the refusal table and the component set changed, and a reader that took a 0.2.0 store
for a 0.1.0 one would answer a request 0.2.0 refuses.

**Version `0.1.0`** is an unreleased draft: no LD Store in it has been published, so the owner's
decisions of 7 November 2026, the Proxy Index among them, of 10 November 2026, the per-store
encoding (ADR 0004), of 12 November 2026, the Far Pairs beside the index (ADR 0007), and of
15 November 2026, the ALID index beside the variant table (ADR 0010), are part of `0.1.0` rather
than a new version. The Partner Rows are the first change big enough to be its own version (§5,
ADR 0011).
