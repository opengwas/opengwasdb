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
