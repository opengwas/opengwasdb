# Native BlockMatrix worked literal (#93)

`part-0` is a 3×3 column-major Breeze DenseMatrix in Hail 0.2.64's
RichDenseMatrixDouble encoding: little-endian int32 rows and columns, byte
row-major = 0, then float64 values. The literal matrix is
`[[4, 1, -3], [1, 1, 0.75], [-3, 0.75, 9]]`.
The payload is one LZ4 block without an embedded size; its stream header is
little-endian int32 (compressed length + 4), int32 raw length.

This worked example follows `InputBuffers.scala` and
`RichDenseMatrixDouble.scala` at hail-is/hail tag 0.2.64. It is not genotype
data. It pins the native decoder without Hail in CI and costs under 100 B.

`part-row-major` and `part-column-major` hold the nonsymmetric 2×3 literal
`[[1, 2, 3], [4, 5, 6]]`, with order bytes 1 and 0 respectively. Symmetric
covariance alone would not catch a decoder transposing storage order.

The end-to-end fixture uses a different worked covariance for chromosome 2:
`part-1` is `[[16, 2, 2], [2, 4, -1], [2, -1, 1]]`. Its correlations before A1
orientation are 1/4, 1/2 and −1/2; flipping the middle variant's A1 gives
−1/4, 1/2 and 1/2, hence ×2047 codes
`[[2047, -512, 1024], [-512, 2047, 1024], [1024, 1024, 2047]]`.
`metadata.json` allocates the two diagonal blocks in a 6×6 matrix, and
`ledger.json` pins their different bytes. `variants.tsv`
places three variants per chromosome at positions 1, 2 and 4,000,001, so both
the Near Band and Far Pairs are nonempty. `identity.chain` preserves their
coordinates. The gzip FASTA has A everywhere except G at position 2, allowing
the middle variant's ALT to be A1 and exercising sign flips. The uncompressed
FASTA is repetitive; its committed compressed size is under 8 KB.
`tiny.chain` and `tiny.fa` independently exercise chain gaps, negative strands
and allele mismatch.
