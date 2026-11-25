- **The Store Reader contract compares `pval` and `log10_p` to four ulps across machines
  (issue #474).** Both come from the libm (`erfc`, `erf`, `log`, `log1p`), which differs in
  the last ulp between the bundle's generating host and GitHub's runner. So the runner failed
  every `pval` in the bundle's 9 releases on an exact comparison, and the resolved
  `log10_p` of the lookup and range PheWAS in 6 of them. `query_digests` now keeps each
  answer's `pval` as numbers beside the digests rather than in them, as it already kept
  `log10_p`. `differences(..., rtol=CROSS_MACHINE_RTOL)` and `readable_differences` compare
  both by one rule, to within `4 * 2**-52` relative, NaN equal to NaN. The rewrite's proof,
  which reads both sides in one build, still compares them exactly.
