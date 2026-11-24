- **The Store Reader contract compares `pval` to four ulps across machines (issue #474).**
  `pval` is libm's `erfc`, which differs in the last ulp between the bundle's generating
  host and GitHub's runner, so every `pval` in the bundle's 9 releases failed its exact
  comparison there while `log10_p` and every other value matched. `query_digests` now keeps
  each answer's `pval` as numbers beside the digests rather than in them, and
  `differences(..., pval_rtol=CROSS_MACHINE_PVAL_RTOL)` and `readable_differences` compare
  them to within `4 * 2**-52` relative, NaN equal to NaN. The rewrite's proof, which reads
  both sides in one build, still compares them exactly.
