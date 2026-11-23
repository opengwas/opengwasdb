# Import published covariance as adjusted-genotype correlation

Formerly ogld ADR 0015.

Issue #93, epic #88. Authority: the owner's 20 November 2026 rulings on #88,
#89's diagonal-normalisation ruling, and #93's kickoff ruling (21 November).

## Context

Pan-UKB publishes UK Biobank EUR residual covariance, not Pearson r. The
producer at `ca9a04cd4c95ebcd6eb92b2096281508b8becf86` writes Z Zᵀ/n after
covariate projection without re-standardisation. Restricted genotypes need
not leave their host to publish this summary data. The whole source is about
14 TB; downloading it whole before import is not viable.

## Decision

An LD Store may be imported from a published LD matrix. Its contract remains
the correlation of adjusted genotypes, signed to engine A1. Covariance is
normalised as r_ij = C_ij / sqrt(C_ii C_jj); non-finite and non-positive
diagonals drop variants under separate counted loss classes. Identity,
quantisation, thresholds and assembly remain their existing owners'.

The importer lives under `opengwasdb.ld.build.importing`. Hail is not a dependency:
Hail 0.2.64 BlockMatrix parts are read natively. The variant table is exported
once with Hail by a script outside the package and supplied as TSV. `lz4` and
`pyliftover` are optional `opengwasdb[ld-import]` dependencies, loaded only on the import
path with a refusal naming the extra. All Pixi test environments install them.

GRCh37 coordinates are lifted to GRCh38 and REF alleles checked against FASTA,
reverse-complementing negative-strand alleles. Unmapped/ambiguous positions,
span/strand/chromosome failures, allele mismatch, invalid identity/frequency
and identity collisions have named loss counts. The complete target radius
is measured per chromosome after liftover; the store uses their minimum.

Liftover can place a displaced block of variants inside another region's Near
Band, which would join pairs outside the published 10 Mb source band. Rather than
refuse the chromosome, the import drops the smaller of the two segments holding a
target-radius window's GRCh37 extremes (segments split where the GRCh37−GRCh38
offset jumps by more than 100 kb; ties drop the one with the lowest GRCh37
position), repeating until the radius reaches the target, `--far-target-bp`
(default 8 Mb, at most the format's 10 Mb `far_bp`; #99, owner ruling). A
segment over `--displaced-cap` (default 10,000 variants) is never dropped: the chromosome
is refused by name. Dropped variants are counted as `liftover_displaced`.

One chromosome is the unit of streaming, restart and optional part deletion.
HTTPS downloads use the public bucket, retries, listing byte counts and a
SHA-256 ledger. A restart verifies retained inputs and skips closed
chromosomes. Parts are deletable only after their chromosome is closed;
shared parts must remain until their last consuming chromosome closes.

The existing writer requires every variant up front. Rather than change a
genotype build's output, refusals or poisoned-writer contract, import closes
one ordinary single-chromosome store at a time in its exclusively locked work
directory. These are restart checkpoints, each validated by the existing
writer. The additive `StoreWriter.join` hard-links their immutable LD
components, concatenates the variant half in bounded slices, derives the
combined index/companion and validates before publishing. Joined row ranges
come from `opengwasdb.ld.format.geometry.chromosome_ranges_from_counts`: closed
chromosome counts avoid allocating a genome-wide chromosome-label array.
Work and output
must share a filesystem. A failed final join is rebuilt from closed inputs;
an incomplete chromosome is rebuilt from verified parts. This avoids a
whole-genome diagonal-parts prepass (which would defeat chromosome staging),
at the cost of validating each chromosome and then the combined store.
The work directory pins metadata, TSV, chain and FASTA hashes and output
identity and the supplied hash ledger's SHA-256. Supplied local parts remain
read-only: their directory and the download cache must be distinct and
disjoint after resolving symlinks, so receipts and downloads never enter the
supplied directory. Initial downloads establish local SHA-256 receipts
because S3 listings publish sizes, not SHA-256;
subsequent reads and supplied hash ledgers enforce the pinned bytes.

A download receipt records ownership (`downloaded`) and the downloaded
file's device/inode/ctime. Only that recorded file may be deleted; a supplied
ledger cannot grant ownership, and old receipts without ownership retain
their parts. Supplied files are protected by file identity at invocation,
not spelling of the directory: symlink/path aliases of the download cache
must not turn supplied inputs into owned downloads.

Provenance records source kind `ld_matrix`, source URL and producer commit,
SHA-256 of every part, matrix metadata, TSV, chain and FASTA, all named
covariates, n = 420,542, k = 15, published QC (not reapplied), transformation,
loss classes and per-chromosome complete radii. Pan-UKB EUR uses ×2047 and
`t_partner = t_far = 0.001` (far code floor 65). It is not comparable to a
genotype-derived k = 0 or k = 10 panel.

## Consequences and rejected alternatives

Hail stays an export-time tool rather than a runtime requirement. A native
reader must refuse corrupt parts by name and never interpret an unallocated
block as zero. A tiny worked-literal fixture pins Hail's binary format without
committing genotypes. Using covariance unchanged was rejected because it is
not correlation. Assuming a 10 Mb target radius was rejected because liftover
can join unmeasured pairs. Whole-genome staging was rejected on disk cost.
The existing `opengwasdb ld validate` remains the sole validator. The matched 511/0.01
analysis view, benchmarking and the whole-genome run belong elsewhere.

## Concurrent downloads and dead addresses (#97)

S3's DNS for the bucket rotates over addresses, about a third of which never
answer from the IEU host, and a single stream is 13–20 MB/s: the whole-genome
import would spend days stalled. A chromosome's missing parts are therefore
fetched by `--download-workers` threads (default 8) through the existing single-part path
(`Parts.ensure`), so byte counts, receipts and ownership are unchanged and peak disk is still
one chromosome's parts. Connections use a 12 s connect timeout apart from the read timeout, and try
the resolved addresses in turn; an address whose connect or TLS handshake fails is remembered as dead for the run (a later timeout or reset is an ordinary failed attempt). TLS
verifies the real bucket hostname (SNI and certificate) against the chosen address. If every
address is marked dead, the marks are cleared: an outage may have passed. A part gets 8
attempts; the final failure refuses by part name.
