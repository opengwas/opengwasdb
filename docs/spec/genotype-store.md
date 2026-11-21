# Genotype Store sidecar

Status: implementation of #91; authority: #18's owner rulings of 20–21 November
2026 and #91's relative-path ruling. ADR 0015 records the storage choices.

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

## Pair answers

`matrix` computes in bounded row blocks using one BLAS thread, quantises only
through REncoding, and applies ld-store.md §8 unchanged. Pairs on different
chromosomes or beyond far_bp are NaN; beyond near_bp, codes below the recorded
far floor are NaN (the Bound). Diagonal entries are 1. Fills and Absent remain
the existing query module's responsibilities. `region` uses the same vectorised
assembly and existing region refusal.
