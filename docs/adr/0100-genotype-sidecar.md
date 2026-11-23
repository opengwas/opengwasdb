# A local sidecar over existing pgens

Formerly ogld ADR 0014.

Issue opengwas/ogld#91, epic opengwas/ogld#18. Authority: the owner's 20–21 November 2026 rulings on opengwas/ogld#18
and the opengwas/ogld#91 resume ruling on relative paths. This records those rulings and the
implementation choices they leave to the reader ticket. ADR 0099 remains the
single arithmetic owner; ADR 0098 remains the ALID companion contract.

## Context

A public HGDP+1kGP panel's pgens occupy far less space than its LD Store. The
owner adopted on-host computation with exactly the LD Store's answers and
refusals, not a new wider measurement contract. Copying calls into Zarr or
retaining decoded calls would waste disk or multiply worker memory.

## Decision

A Genotype Store is the existing pgen filesets and a local Zarr v3 sidecar whose
root provenance key is `genotype_store`, format `0.1.0`. It has no `ld_store`
claim. The variant columns, ALID Index and optional local companion use the LD
writer/assembly and VariantTable unchanged in meaning. The writer exposes the
variant-table half and companion derivation accepts an explicit provenance key;
existing callers retain their defaults.

Each pgen, psam and pvar is recorded relative to the sidecar root using
`os.path.relpath`, POSIX separators and allowing `..`. The reader resolves these
paths against its own location, checks existence, size and nanosecond mtime on
open, and refuses naming the changed file. Validation additionally checks hashes.
There is no override argument. Moving the sidecar and pgens together preserves
resolution; moving the sidecar alone does not.

Small adjustment arrays use uncompressed NumPy `.npy` files opened read-only
with mmap, so processes share their pages. `opengwasdb.ld.genotypes` owns these paths,
dtypes and descriptors. The arrays hold pgen row, fileset index, projections,
residual norms, A1 signs and the common PC basis. A request reads dosages through
Genotypes and calls `adjust.r`; it never caches decoded calls. BLAS runs one
thread per request. Matrix quantisation/assembly operates on vectorised row
blocks, bounding temporary storage independently of the result square.

The builder reuses sample planning, Variant Set selection, PC Check and FLAT
drops. It records the same panel/adjustment/threshold/encoding facts as an LD
Store. No pair counts are claimed for components it does not materialise. Its
PC Diagnostic is null: scanning the Near Band solely for that diagnostic would
undo the sidecar's build-cost saving; PC selection and its complete Check remain
recorded. This distinction is explicit in the sidecar spec.

The two rules a pair's answer turns on — whether a pair at a distance is Stored, and
which floor a Partner Rows row keeps it at — are `REncoding.stored_mask` and
`REncoding.partner_mask`, beside the code floors. The builder's cut, the Partner Rows
emitter and this reader all call them, so the Bound and Not Measured cannot drift
between a materialised store and a computed one.

The reader composes rather than subclasses: it holds its own `OpenedStore` and composes
`VariantTable` exactly as `StoreReader` does, and both satisfy `opengwasdb.ld.query.reader.Reader`
structurally. So a genotype reader has no LD reader's members that mean nothing here
(`r_of_pairs`, `frame_cache_bytes`, `has_partner_rows`), and `opengwasdb.ld.open` and `ogld.validate`
read the root once and hand the opened metadata to the kind that owns it.

The clump seam is a request-local context manager yielding a row callable.
For LD Stores it yields the existing partner_row, a no-op with respect to I/O
and answers. For genotype reads it holds only a component's hit calls during
that component, then releases them. This working set is not a cache: no calls
survive a query. Candidate selection never decodes non-hits, and correlation
rows are computed only for surviving leads. A component's lead window is
selected with sorted positions before calling the shared arithmetic.

Pgen handles are kept for the reader's lifetime, leased one to a decoding thread at a time and
closed by `close()`. On the whole-genome EUR sidecar a clump builds 25 handles (one per fileset
plus the parallel slices) in 0.65 s of 4.19 s; building each decode's handle afresh builds 101 in
1.59 s of 6.07 s. The idle set is bounded at eight handles: a handle keeps the pgen file pages
its decode touched, and keeping all 22 at once takes a clump's peak RSS from 1.35 GB to 2.18 GB,
1.17 GB above the LD Store reader's, past the ticket's bound. At eight the peak is 1.70 GB
(+0.69 GB), and a repeated whole-genome clump still reuses what its last chromosomes left. This
is a handle cache, never a decoded-call cache.

Measured (opengwas/ogld#91): assembling one proxy window's dosage block whole costs 432 ms,
with 214 ms inside read/reorder, and the 3 Mb region 4.05 s. Decode writes
disjoint output slices on at most four reader threads, with no concatenation or
retained calls. Rows at tied loci that differ between physical pgen and ALID
order are reordered in place rather than copying the whole block. The shared
float64 dosage/r path is preserved; no integer-only arithmetic shortcut is
introduced.

Region tiles compute only the upper triangle and mirror its codes. Quantising
16-row stripes keeps temporary arrays cache-sized; 512-row stripes measure
5.01 s. The region measures 1.79–1.88 s.
BLAS limit/restore operations run on one process-local control thread, with an
active-request count, because threadpoolctl's OpenBLAS controller owns a thread
and an early request exit must not restore limits under another active request.
A fresh-process concurrent-request test starts at two threads, observes one
through both requests, then verifies restoration to two. The reader's `request()`
seam enters the guard once per public query — `matrix`, `region`, `proxies`,
`clump` — and not per row of a partner response; a plain per-request
`threadpool_limits` would restore the caller's limit while another
request is still running, and one entered in a worker thread would restore it
from the wrong thread, so the process-wide count stays.

## Consequences and rejected alternatives

Absolute paths were rejected by the owner's ruling: a deployment moves its input
tree. Path overrides were rejected to avoid silently substituting another panel.
A second call container and a decoded-call cache are rejected on disk and
per-worker memory cost. Compressed adjustment arrays would force each worker to
decode its own copy; mmap instead uses the shared page cache. Publishing
individual-level data or remote genotype reading is not authorised by this ADR.
LD Stores remain the route for restricted and large panels.
