# Genotype Store sidecar

Status: implementation of #91; authority: #18's owner rulings of 20–21 November
2026 and #91's relative-path ruling. ADR 0100 records the storage choices.

## Container and shared data

A local Zarr v3 root has `genotype_store` provenance, format `0.1.0`, and no
`ld_store` attribute. Opening it as an LD Store is refused; opening an LD Store
as a Genotype Store is refused. `ogld.open` dispatches by the root key.

`variants/`, its chromosome ranges, ALID Index and companion are exactly
ld-store.md §2/§14, written by the same writer and assembly and read by
VariantTable. No Near Band, Far Pairs or Partner Rows are materialised.

## Adjustment data

`genotypes/` is a Zarr group containing local uncompressed NumPy files. Each
`.npy` descriptor records its shape and SHA-256 under provenance `arrays`.
Readers use `np.load(mmap_mode="r", allow_pickle=False)` and refuse missing,
unreadable or differently shaped/typed arrays. V is the Variant Set size,
n the kept sample count and k the covariate count.

| file | dtype | shape |
|---|---|---|
| `pgen_row.npy` | little-endian int64 | V |
| `fileset.npy` | little-endian int32 | V |
| `projection.npy` | little-endian float64 | V × (k + 1) |
| `norm.npy`, `sign.npy` | little-endian float64 | V |
| `basis.npy` | little-endian float64 | n × (k + 1) |

Calls are not stored or cached. `adjust.prepare` writes the projection, norm
and sign; `adjust.r` alone computes correlation. Sign is +1 for A1=ALT, −1
otherwise. Sample selections and optional reorder arrays are persisted for
each fileset, together with its original sample count and panel name.

## Provenance and input paths

Provenance records source hashes, panel n, assembly, filters, dropped counts,
PC File hash/source, complete PC Check, k/covariate names, encoding, near/far
distances, t_far and both code floors, t_partner, software and build cost.
`counts` records only `alids_left_out`; it does not claim stored pair counts.
`pc_diagnostic` is null because the sidecar computes no exhaustive pair pass.

`filesets` contains pgen/psam/pvar pins plus sample selection. Each pin contains
`path`, `size`, `mtime_ns` and `sha256`. Paths are written using
`os.path.relpath(input, sidecar_root)`, POSIX separators, with `..` allowed.
They resolve against the opened sidecar root, not the working directory.
Every resolved fileset member must exist and match size/mtime on open. A
refusal names the file and changed field. There is no path-override API.
`inputs` additionally pins the keep file and, at k > 0, PC File. Validation
checks all input hashes and every adjustment array hash.

## Validation and publication

The shared variant/chromosome/ALID and applicable provenance rules judge the
canonical shared data. `genotypes.provenance` checks the genotype envelope,
`genotypes.inputs` checks every external pin including SHA-256, and
`genotypes.arrays` checks all array descriptors/hashes, dimensions against n/k/V,
finite projections, positive finite norms, A1 signs, basis orthonormality and
nonnegative row/in-range fileset mappings. A missing prerequisite is a finding,
not a passed rule. Input/array content is immutable during reading; hashes are
integrity pins, not producer authentication.

The builder runs these same checks before writing `genotype_store`. A failed
candidate remains unreadable. Once the claim is written, the shared companion
assembly validates correspondence and publishes its optional pointer atomically
as in ADR 0098. Its failure can leave a complete sidecar without a companion,
never a half-published mapping. `ogld validate` and public `ogld.validate`
dispatch by kind; the LD Store validator's rules and behaviour are unchanged.

## Pair answers

`matrix` computes in bounded row blocks using one BLAS thread, quantises only
through REncoding, and applies ld-store.md §8 unchanged. Pairs on different
chromosomes or beyond far_bp are NaN; beyond near_bp, codes below the recorded
far floor are NaN (the Bound). Diagonal entries are 1. Fills and Absent remain
the existing query module's responsibilities. `region` uses the same vectorised
assembly and existing region refusal. Regions compute only the upper triangle,
in 128-row arithmetic tiles and 16-row quantisation stripes, then mirror the
codes. This follows the LD Store's earlier-variant orientation and avoids both
recomputing correlations and a square-sized temporary allocation.

## Threshold queries and clump working set

`partner_row(s)` computes only within far_bp and keeps codes at the recorded
Partner Floor within near_bp and far floor beyond it. Existing proxies/clump
modules retain all threshold, edge, ordering and refusal semantics of
ld-store.md §9, including threshold_below_t_partner despite the lack of physical
Partner Rows.

The explicit reader seam `clump_rows(chrom, local_hit_rows, window)` is a
context manager yielding a lead-row callable. An LD Store returns its existing
partner_row with no reads or precomputation. A Genotype Store decodes only that
component's hits once as request-local working memory, computes each surviving
lead against hits within the window through `adjust.r`, and releases the block
when the component ends. It never sets reader attributes to restrict a request,
never computes an eliminated lead's row, and retains no calls between queries.

Large pgen reads decode disjoint 8,192-row output slices on at most four native
reader threads, each leasing its pgen handle. Handles are kept for the reader's lifetime, one
thread at a time, at most eight idle, and released by `close()`. `Genotypes.rows_into` writes into
the final float64 buffer; only physically reordered rows are permuted in place
when possible. No hard-call substitution is made for dosage-bearing inputs.
The builder checks completeness, so query reads rely on the pinned, immutable-input
contract instead of scanning every dosage.

BLAS is bounded to one thread while requests are active. A process-wide active
count and a single control thread keep threadpoolctl's controlling-thread
ownership consistent: the last request restores the caller's original limit.
No decoded calls or array mappings are held by this control mechanism. Reader
processes use spawn, not inherited readers or executors after fork.
