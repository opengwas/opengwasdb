# A validated local mmap companion of the canonical ALID index

Formerly ogld ADR 0012.

Ported for [#81](https://github.com/opengwas/ogld/issues/81) from #47 at `9b6fdb6`
(PR #59 on epic #45, not previously on main). This extends ADR 0096; it changes neither
ADR 0097's Partner Rows nor the Store format version.

**Decision authority:** the owner's #81 brief requests this isolated port of #47's
representation, publication and local lookup behavior onto main.

> **The bundle's owner moved, and kept its behaviour** (see "Clarification, 21 November 2026" at
> the end). It is `_MappedAlids`, a module-level class in `opengwasdb/ld/store/variants.py`, owned and
> released by the `VariantTable` that composes it (#94); `reader._MappedAlids` below names it as
> this ADR first wrote it.

## Context

After Partner Rows, ALID resolution is a larger cost than clump's LD work: #81 records
1.6–2.4 s on GCST90476460's 260,821 hits. The canonical Zarr index already owns the key
width, padding, order and rows. Local lookup can search those bytes directly without
another identity algorithm, a second sort, or loading the index through compressed shards.
HTTP and older Stores must retain the canonical index and exact scan.

## Decision

Store `variants/alid_mmap/<generation>/{keys.bin,rows.i32le,manifest.json}` locally.
Keys copy the canonical uint8 bytes viewed as `|S32`; rows copy signed little-endian int32
in exactly canonical order. The deterministic manifest uses sorted keys, compact separators,
UTF-8 and no timestamps. Its exact fields are `schema` (integer 2), `count`, `key_dtype`
(`|S32`), `row_dtype` (`<i4`), `sources`, and `payloads`. Sources pin SHA-256 of normalized
root metadata with only the `alid_mmap` attribute removed, and every file of both canonical
index arrays: metadata and every chunk/shard file. Payloads pin both binary files' SHA-256.
Generation is the manifest bytes' SHA-256. Layout owns the digest/pointer and one payload
descriptor table for assembly and validation; row widths follow the recorded dtype.

**Schema 2 (decided in #81's review, 19 November 2026):** schema 1 pinned only canonical
metadata, so a companion copied from a Store with identical metadata but different keys could
change answers. Before any mapped byte is trusted, schema 2 verifies canonical content pins
and payload hashes. Schema 1 refuses naming the Store and manifest, with an instruction to
re-derive using `scripts/derive_alid_mmap.py`. Remove the old pointer/generation only on an
isolated private Store before re-deriving; never repair a live reader's generation in place.

The optional root attribute is exactly `{"generation":"<64 lowercase hex digits>"}`.
Its replacement is the publication point. A cooperating derivation holds a file lock,
replaces an abandoned `.partial`, writes and fsyncs payloads, manifest and partial directory,
renames to an immutable generation and fsyncs its parent, then atomically replaces and
fsyncs root metadata. Retrying reuses identical unpublished generations, validates published
ones, and refuses conflicting or corrupt ones. Old generations remain for existing mappings.
Derive only on isolated/quiescent local Stores; do not mutate canonical files during reading.

`StoreWriter.finish` validates the canonical Store, records provenance, then calls
`assembly.derive_alid_mmap`; `scripts/derive_alid_mmap.py` calls that same helper. A failed
companion creation leaves either a valid Store without a pointer or a complete publication.
The helper first applies the canonical index's existing four validation rules. Before writing
the root pointer, both publication paths then call `validation.require_alid_mmap` on the new
generation's payloads. It uses the same bounded exact byte-comparison as `alid_index.count`,
including entry count and full payload lengths; a failed comparison leaves the prior root
unchanged and the generation unpublished.

Canonical file pins bind the companion to the actual Store, not merely equal metadata.
Missing, added or changed canonical files refuse, naming the Store, generation and concrete
source file. Inventory follows shard-directory symlinks rather than silently omitting their
files; cycles, repeated directories, walk errors and nonregular sources refuse. Files remain
immutable while a reader uses its checked mappings. Exact validation
also compares companion payloads to canonical arrays; payload hashes alone do not prove
correspondence.

**Trusted publication (decided in #81's review, 19 November 2026):** the supported input contract is a generation
published only by ogld, after exact canonical correspondence. At use, the reader verifies
integrity and content binding (every canonical index file's pin), not correspondence and not
producer authenticity. A hand-forged generation whose payloads, manifest and pointer were all
rewritten consistently is outside what the reader detects, just as re-hashed forged Partner
Rows frames are. `opengwasdb ld validate` detects the noncorrespondence. A canonical file changed after
a reader has attached is likewise outside the immutable-files assumption. No per-lookup
correspondence pass or mandatory prevalidation is added for trusted ogld publications. The existing `alid_index.count`
rule includes integrity and bounded exact comparison; there are still 30 validation rules.
A self-consistent row permutation is a breaker: range, uniqueness and hashes pass, but
canonical correspondence fails. Artifact bytes contain no filesystem paths or inode identity,
so an unchanged Store copy derives identical bytes. Payload descriptors retain device, inode,
size, mtime and ctime only for the opened-file comparison before mapping. The already-digested
manifest needs no runtime stat identity, and the bundle has no unused sharing identity.

**Verify on first use (decided in #81's review, 19 November 2026):** opening does not verify or map the companion. The
first fitting lookup checks the manifest, all canonical source pins and payload hashes, then
maps read-only keys and rows, once under a reader-local lock. A failure is sticky: that lookup
and every later lookup refuse with the same diagnostic, even if files are subsequently repaired.
Close shares the attachment lock, so no late attachment can outlive it; closing before lookup
acquires no mappings. Wide-only scans need no companion. This preserves integrity while
charging only lookups that need it, not row-based matrices/regions or unused opens. Fitting
UTF-8 queries use vectorized binary search and equality, preserving order, misses and duplicates
without parsing loci or reading string/position columns. Embedded NUL and unencodable names
are Absent. Over-width names take the exact scan. Since construction/validation establish
completeness, fitting misses need no scan. This extends ADR 0096's always-scan-misses rule
only for the validated companion. Missing pointers and HTTP retain the canonical fallback;
a present broken publication refuses rather than silently falling back.

`reader._MappedAlids` — `_MappedAlids` in `opengwasdb/ld/store/variants.py` since #94, and owned by the
`VariantTable` that composes it — owns its mappings and their diagnostic path. Lookup returns
independent int64 rows. Release serializes with lookup, discards views, then closes mappings, and
is idempotent. `StoreReader.close` releases the bundle with its Partner Rows resources, by closing
its variant table. Failed
attachment immediately releases already acquired mappings, even with a retained traceback.
Normal object collection also releases mappings; no process-global registry is introduced.

## Differences from #47

Main already has the chromosome-loci cache introduced with Partner Rows: the exact scan uses
it and never initializes Near Band or Far Pairs metadata. The transient `_scan_positions`
helper from #47 is unnecessary. Main's explicit reader close also releases the mapping bundle.
The Proxy Index remains retired. No public signatures change.

#46's benchmark harness and #50's shared generation/request leases are not on main. This ADR
does not claim either, or cite their different ADR 0097 as main's contract. Cross-reader
sharing and process admission remain possible later epic #45 work, not guarantees of this port.
#47's historical lookup numbers are not this port's timing evidence; #81 measures end-to-end
clumping separately. The committed Test Store remains companion-less, exercising compatibility;
regenerating it now produces a companion, checked byte-for-byte against canonical arrays in tests.

## Consequences and rejected alternatives

The raw companion adds 36 bytes per indexed variant, about 309 MB on EUR, outside decoded
cache budgets. First use hashes those bytes and about 64 MB of EUR's canonical index files,
which can cost cold I/O; this cost is reported separately from the search. Zarr remains canonical and
interoperable; HTTP never fetches companion files.

A second producer sort is rejected because it can drift from ADR 0096. Mutable fixed payload
paths are rejected because existing mmaps could see partial replacement. Process-global mapping
caches are rejected here because sharing/admission is separate work. Hashing canonical arrays
by decoding on every lookup is rejected: construction and exact validation establish
correspondence, while canonical-file and companion hashes bind the checked generation to the
Store under the quiescent immutable-files requirement. Metadata-only pins were insufficient.

Breakers cover truncation, missing/wrong payloads, malformed manifests and schema types,
stale metadata, self-consistent wrong keys/rows and row permutations. Interrupted publication
is restartable without exposing a partial pointer. Lifetime tests use real mappings, retained
tracebacks and event-gated concurrent lookup/release rather than timing guesses.

## Clarification, 21 November 2026

Epic [#18](https://github.com/opengwas/ogld/issues/18)'s owner decision of 21 November 2026 took the
variant table and the ALID lookup out of `StoreReader` into `opengwasdb.ld.store.VariantTable`
(`opengwasdb/ld/store/variants.py`, #94) as a pure move, so a second reader composes the same lookup. The
companion's owner is unchanged in what it does and named by where it now lives: `_MappedAlids` is a
module-level class in `opengwasdb/ld/store/variants.py`, and the `VariantTable` owns and releases it, which
`StoreReader.close` does by closing that table. First-use verification, the sticky refusal, the
lookup/release lock, the cost of the first use and the absence of any process-global registry are
as written above.
