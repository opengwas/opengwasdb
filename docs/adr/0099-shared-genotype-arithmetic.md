# One owner for pgen reading and PC-adjusted r

Issue #90, epic #18. The owner's ruling of 21 November 2026 on #18 adopts this formula
and module layer. This records that ruling, not a new Store format or provenance contract.
ADR 0002 decision 10 still applies: pgenlib remains the optional `ogld[build]` extra,
imported only when opening a pgen, with the same refusal naming the extra.

## Context

The LD Store builder and on-demand genotype reads need the same PC Adjustment. Neither
`build` nor `query` may import the other. A copied formula violates ownership's **call,
never copy** rule and can silently give different Stored r for the same Source Panel.

## Decision

`ogld.genotypes` owns pgen reading, the orthonormal covariate basis Q of [intercept,
kept PCs], residuals and their FLAT test, projections p = xQ, residual norms
N = ‖x − pQᵀ‖, and the A1 sign s (+1 where A1 is ALT, −1 otherwise).
Its single `r(rows, span)` computes, in float64 BLAS:

```
r_ij = s_i s_j (x_i·x_j − p_i·p_j) / (N_i N_j)
```

A row-against-window call is the same function on a one-row block. Quantisation remains
`ogld.format.REncoding`'s, including half-to-even rounding and its floating-error refusal.
At k = 0 Q contains the intercept alone. The PC Diagnostic's adjusted-minus-unadjusted
change must call the same function on each basis. Residual norms retain the explicit
residual calculation and original FLAT refusal text (build-inputs §6); subtracting squared
norms would suffer cancellation near that check.

For complete hard calls x ∈ {0, 1, 2}, every partial sum of x_i·x_j is a nonnegative
whole number ≤ 4n. Float64 represents all of them exactly while 4n ≤ 2⁵³, independent
of summation order, BLAS threads or blocking. Thus only the short (k + 1)-term correction,
projections, norms and normalisation round, rather than a sample-length sum of residual
products. This argument does not claim exact real-number PC arithmetic or prove code
equality at rounding boundaries: tests against the frozen builder and independent Test
Store, and a base-against-head chr22 rebuild, must establish that. Dosage-bearing pgens
retain float64 dosage reads and this same formula, without the integer exactness claim.
An int8 hard-call read path does not replace dosage reads in the builder.

The PC Diagnostic's float summaries are reproducible only to float rounding across software
versions, while Stored r codes remain exactly equal in the acceptance comparisons. The
orchestrator accepted the chr22 diagnostic-extrema drift of at most 3.1×10⁻¹⁵ in #90's
first review round; the diagnostic is provenance about chance LD, not a Pair Answer.

The import layer is `identity, format → store → genotypes → build, query → cli`.
`genotypes` may import only `store`, `format`, `identity` (and ubiquitous `version`).
Build threading policy (`bounded`) stays in `build`. Build input parsing and Variant Set
selection remain in `build`; the pgen reader accepts a structural input description,
not an upward import of the builder's Fileset type.

## Rejected alternatives and consequences

- Two formula owners, or retaining residual-unit GEMM for tiles and the correction formula
  for reads: rejected because the same pair could acquire two codes.
- Integer dot products in the reader and a different builder kernel: rejected in favour
  of one float64 BLAS path, including single rows.
- Making pgenlib a required dependency: rejected; an LD Store reader needs no pgenlib.
- Store-format, encoding or provenance changes: unnecessary and outside this ticket.

Blocks retain ALT dosages plus small per-variant adjustment data rather than unit vectors.
The builder must still hold only its bounded window. Exact code comparisons and build-time
measurements, not the algebra alone, determine whether switching the builder is acceptable.
