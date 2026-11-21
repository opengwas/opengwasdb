# A local sidecar over existing pgens

Issue #91, epic #18. Authority: the owner's 20–21 November 2026 rulings on #18
and the #91 resume ruling on relative paths. This records those rulings and the
implementation choices they leave to the reader ticket. ADR 0013 remains the
single arithmetic owner; ADR 0012 remains the ALID companion contract.

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
with mmap, so processes share their pages. `ogld.genotypes` owns these paths,
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

The clump seam is a request-local context manager yielding a row callable.
For LD Stores it yields the existing partner_row, a no-op with respect to I/O
and answers. For genotype reads it holds only a component's hit calls during
that component, then releases them. This working set is not a cache: no calls
survive a query. Candidate selection never decodes non-hits, and correlation
rows are computed only for surviving leads. A component's lead window is
selected with sorted positions before calling the shared arithmetic.

## Consequences and rejected alternatives

Absolute paths were rejected by the owner's ruling: a deployment moves its input
tree. Path overrides were rejected to avoid silently substituting another panel.
A second call container and a decoded-call cache are rejected on disk and
per-worker memory cost. Compressed adjustment arrays would force each worker to
decode its own copy; mmap instead uses the shared page cache. Publishing
individual-level data or remote genotype reading is not authorised by this ADR.
LD Stores remain the route for restricted and large panels.
