# An ALID search index in the LD Store, adopting the engine's

Formerly ogld ADR 0010.

**Extended by ADR 0098:** validated local mmap companions may answer fitting misses directly;
the canonical Zarr-index/exact fallback described here remains in use without a companion.

Issue [#42](https://github.com/opengwas/ogld/issues/42), part of epic
[#1](https://github.com/opengwas/ogld/issues/1). ld-store §2, §6, §7, §10, §12 and §13 define the
store; ADR 0089 the identity the keys are made of, and ADR 0095 the decoder that reads them.

**Who decided.** The owner, 15 November 2026, on #403's cold clump timing: finding the variants
of 37 Analyses took 94 s of the 272 s they took at r² 0.01, and 59 s of that was reading ALID
strings through Zarr's variable-length path. The design is the engine's, whose variant table
searches a fixed-width ALID sidecar (`opengwasdb/variants/axis.py` and its ld-store §10a), adapted to
a Zarr array a reader can fetch over HTTP.

## Context

`rows_of` found an ALID by parsing its chromosome and position, binary-searching
`variants/position` within the chromosome's rows, and comparing the full ALID strings at that
position (ld-store §2). That is exact at any length and needs no slot. It also decodes a whole
16,384-string chunk for each position touched, and loads a chromosome's positions and Near Band
`indptr` — 137 MB of `indptr` on EUR — before it can search anything: 31.1 s of the 94 s was
those chromosome loads and 59.1 s the strings. `docs/benchmarks/ld/fast-decode-round2.md` has the
phase table.

[#40](https://github.com/opengwas/ogld/issues/40) had already routed every one-dimensional
numeric array through a direct decoder (ADR 0095), recognising exactly `sharding_indexed` with
`bytes` then `zstd`, and `bytes` then `crc32c` in an end-located index. Strings stayed on Zarr's
path, and the strings were the cost.

## Decision

**The store carries the engine's index, beside the string column.** `variants/alid_index/alid`
holds one 32-byte key an entry — an ALID's UTF-8 bytes, NUL right-padded — sorted bytewise, and
`variants/alid_index/row` the Variant Set row of each entry, in 262 KiB inner chunks and 16 MiB
shards: 8.55 M keys are 273 MB decoded, which is 16,693 chunk decodes in 16 KiB chunks (4.6 s for
a whole read) and 1,044 in these (0.2–0.7 s). `opengwasdb.ld.store.layout` owns the width,
the padding and the order: `alid_keys`, `ordered_index` and `alid_index`. The entries are the
ALIDs whose UTF-8 form is at most 32 bytes, which is the engine's rule and its width
(`is_indexable_alid`), and its ordering: `np.array(..., dtype="|S32")` with a stable argsort,
which pads with NUL and keeps ties in row order. A wider ALID is left out and counted in
`counts.alids_left_out`, never cast into the slot. `tests/ld/data/alid_index/` holds the engine's
own answers for five Variant Sets, made by `make_fixture.py` from the engine's
`write_variant_axis` and the two sidecars it writes, and `tests/ld/test_alid_index.py` holds the
index to them: an ALID at the slot's width, one a byte wider that agrees with it over all 32, two
indels at one position, and a repeated ALID for the stable sort.

**The keys are `uint8`, not a fixed-length bytes type.** Zarr v3 has no `|S32` data type. In
zarr 3.4.2 a `|S32` array is written as `null_terminated_bytes`, an extension zarr marks
unstable and warns may be unreadable by other libraries; a store written that way would be one
zarr-python's own future could change, which ld-store §1 does not allow. A `uint8` array one
dimension long is plain Zarr v3, readable anywhere, and the reader views 32 elements as one key.
It is also what the direct decoder recognises: as a two-dimensional `(n, 32)` array it would not
be, because `NumericDecoder.of` admits one-dimensional arrays only.

**A one-byte `bytes` codec declares no byte order, and the decoder now reads it.** Zarr emits
`{"name": "bytes"}` with no `endian` for a `uint8` element, and `_bytes_chain` refused it, so the
keys would have gone through Zarr while the rest of the store did not. ADR 0095's rule is about
the byte order of a multi-byte element; a one-byte element has none. The chain is accepted when
the element is one byte and no order is declared, and every wider dtype still must declare its
order. The shard index's `bytes` codec declares `little` as before.

**A reader searches the index and still scans what it misses.** `StoreWriter.finish` derives the
index from the store's own ALID column (`opengwasdb.ld.store.assembly.derive_alid_index`), so no producer
supplies or orders one and a store written before the index gets the same one (ADR 0091).

- An ALID shorter than the slot is looked for in the index — but only in a list long enough to
  pay for it: a list naming about a two-thousandth of the Variant Set's rows or more
  (`_INDEX_SHARE`, 4,000 of EUR's 8.59 M ALIDs). Such a list searches the index with numpy's own binary search, read one
  shard at a time in key order — the unit the store itself reads, from the array's own shapes, 16 MiB
  and 524,288 keys on EUR, each shard let go before the next is fetched — so nothing a lookup holds
  grows with the store. A short list takes the exact scan instead, because a search's
  chunks cost more than the string chunks a few loci touch: measured, one ALID's scan is 0.07 s
  against the index's 0.9 s, and 500 of them 0.11 against 0.33, while at 4,000 the index wins and
  keeps winning (`docs/benchmarks/ld/alid-index.md`). A binary search reading only the chunks one
  key's path touches was measured too, and rejected: with chunks big enough to read the whole index
  in a second, a few hundred keys touch nearly all of them.
- An ALID wider than the slot has no key (`layout.alid_fits`, the engine's own rule) and is
  scanned. An ALID at the slot's width is indexed like every other one: validation holds the index
  to the table entry by entry — `alid_index.keys` refuses a key that is not its row's full ALID, so
  a truncated key, a key of a wider row's bytes, or an invented key fails it — and the reader
  trusts a store that passes, as it trusts every other array. The first draft kept a slot-width
  reader guard instead, which would have spent 98 chromosome loads on #403's lists to defend
  against a store the rules now refuse.
- An ALID the index does not hold — absent, too wide, or one a broken build left out — takes the
  exact scan at its own position, which is what found it before. So a miss, an unsorted index and
  an index with an entry dropped all answer as before; only a hit can be faster.
- Every answer is therefore byte-identical, and `tests/ld/test_alid_index.py` holds that against an
  index rebuilt out of order, whose misses the scan answers; a truncated index is `alid_index.keys`'
  breaker, because it is a store validation refuses rather than one a reader second-guesses.

**Nothing else changes.** No existing array's bytes change, no public signature changes, and
`format_version` stays `0.1.0`: adding an array is not one of §13's three triggers — a change to
an existing array's dtype, encoding or meaning, or to §8's answers — and ADR 0093 added the Far
Pairs the same way. §13 now lists the owner's 15 November 2026 decision among those in `0.1.0`.

## Alternatives and costs

- **No index, and the strings read faster.** Rejected by the owner's measurement: 59 s of 94 s is
  the variable-length path, and no read of it avoids decoding a whole string chunk per position.
- **A `|S32` array.** Rejected: an unstable Zarr extension other implementations do not read.
- **A `(n, 32)` `uint8` array.** Rejected: not directly decodable, so the keys would read through
  Zarr's slower path, which is what the index exists to avoid.
- **A key with a length byte or a per-entry width.** Rejected: it buys nothing the NUL padding
  does not, and `alid_index.keys` holds every entry to its row's full ALID instead.
- **Search every ALID through the index, a miss meaning absent.** Rejected: a build that left an
  entry out, or wrote its keys unsorted, would then report a present variant absent. The scan is
  what makes the index an optimisation rather than a second source of truth.
- **Read the whole index for one ALID.** Rejected: 309 MB on EUR for one name, where the chunks on
  one search's path are about 0.4 MB.
- **Rebuild the five stores** rather than derive the index into them. Rejected as in ADR 0093:
  deriving adds files and changes no other byte.

## What it did

On #403's 37 Analyses at r² 0.01, cold, 8 threads, `rows_of` fell from 94.0 s to 50.4 s and the
37 clumps from 272.5 s to 229.4 s, with 23,445 leads and all 37 result fingerprints unchanged.
opengwas/ogld#8's step-1 reads are unchanged: every list they time is short enough to take the exact scan, and
they read the same chunks and make the same requests as before. `docs/benchmarks/ld/alid-index.md` has
the phase table, the chunk-size measurements and the list-length crossover.

## The stores

The five HGDP+1kGP stores are derived in place by `scripts/ld/derive_alid_index.py`, which writes
`variants/alid_index-partial/` from each store's own ALID column, renames it into place, and only
then records the count in the provenance — a new document renamed over the old one. So a run that
fails leaves a store that validation refuses and a second run repairs, and a store already derived
is finished rather than refused. On HGDP+1kGP EUR chr22 the derived index
is byte-identical to a rebuild's, and the store's provenance differs only in `build` and
`software`, as opengwas/ogld#31 showed for the Far Pairs.
