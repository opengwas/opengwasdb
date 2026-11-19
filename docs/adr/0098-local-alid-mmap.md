# A validated local mmap companion of the canonical ALID index

Ported for [#81](https://github.com/opengwas/ogld/issues/81) from #47 at `9b6fdb6`
(PR #59 on epic #45, not previously on main). This extends ADR 0010; it changes neither
ADR 0011's Partner Rows nor the Store format version.

**Decision authority:** the owner's #81 brief requests this isolated port of #47's
representation, publication and local lookup behavior onto main.

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
UTF-8 and no timestamps. Its exact fields are `schema` (integer 1), `count`, `key_dtype`
(`|S32`), `row_dtype` (`<i4`), `sources`, and `payloads`. Sources pin SHA-256 of normalized
root metadata with only the `alid_mmap` attribute removed, and raw metadata of both canonical
arrays. Payloads pin both binary files' SHA-256. Generation is the manifest bytes' SHA-256.

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
The helper first applies the canonical index's existing four validation rules.

Root/index metadata pins cheaply detect generation mismatch; they do not detect unannounced
in-place canonical shard mutation. Exact validation compares companion payloads to canonical
arrays. Payload hashes alone do not prove correspondence. The existing `alid_index.count`
rule includes integrity and bounded exact comparison; there are still 30 validation rules.
A self-consistent row permutation is a breaker: range, uniqueness and hashes pass, but
canonical correspondence fails. Artifact bytes contain no filesystem paths or inode identity,
so an unchanged Store copy derives identical bytes. Runtime identity additionally includes
manifest/payload device, inode, size, mtime and ctime, detecting replacement before mapping.

Local reader attachment checks the descriptor and payload integrity and maps read-only keys
and rows. Attachment, including hashing, belongs to opening, not opened lookup. Fitting
UTF-8 queries use vectorized binary search and equality, preserving order, misses and duplicates
without parsing loci or reading string/position columns. Embedded NUL and unencodable names
are Absent. Over-width names take the exact scan. Since construction/validation establish
completeness, fitting misses need no scan. This extends ADR 0010's always-scan-misses rule
only for the validated companion. Missing pointers and HTTP retain the canonical fallback;
a present broken publication refuses rather than silently falling back.

`reader._MappedAlids` owns mappings and representation identity. Lookup returns independent
int64 rows. Release serializes with lookup, discards views, then closes mappings, and is
idempotent. `StoreReader.close` releases the bundle with its Partner Rows resources. Failed
attachment immediately releases already acquired mappings, even with a retained traceback.
Normal object collection also releases mappings; no process-global registry is introduced.

## Differences from #47

Main already has the chromosome-loci cache introduced with Partner Rows: the exact scan uses
it and never initializes Near Band or Far Pairs metadata. The transient `_scan_positions`
helper from #47 is unnecessary. Main's explicit reader close also releases the mapping bundle.
The Proxy Index remains retired. No public signatures change.

#46's benchmark harness and #50's shared generation/request leases are not on main. This ADR
does not claim either, or cite their different ADR 0011 as main's contract. Cross-reader
sharing and process admission remain possible later epic #45 work, not guarantees of this port.
#47's historical lookup numbers are not this port's timing evidence; #81 measures end-to-end
clumping separately. The committed Test Store remains companion-less, exercising compatibility;
regenerating it now produces a companion, checked byte-for-byte against canonical arrays in tests.

## Consequences and rejected alternatives

The raw companion adds 36 bytes per indexed variant, about 309 MB on EUR, outside decoded
cache budgets. Opening hashes those bytes and may cost cold I/O. Zarr remains canonical and
interoperable; HTTP never fetches companion files.

A second producer sort is rejected because it can drift from ADR 0010. Mutable fixed payload
paths are rejected because existing mmaps could see partial replacement. Process-global mapping
caches are rejected here because sharing/admission is separate work. Hashing canonical arrays
on every open is rejected: build/validation establish correspondence, and opening checks
metadata pins and companion integrity under the quiescent immutable-files requirement.

Breakers cover truncation, missing/wrong payloads, malformed manifests and schema types,
stale metadata, self-consistent wrong keys/rows and row permutations. Interrupted publication
is restartable without exposing a partial pointer. Lifetime tests use real mappings, retained
tracebacks and event-gated concurrent lookup/release rather than timing guesses.
