- **Every builder resolves a duplicate (variant, Analysis) the same way (issue
  #357, D5, ADR 0072).** Rows that agree on z and SE are one association: they
  collapse to the first, frequency and all (`duplicate_collapsed`, counted).
  Rows that disagree are all dropped, since nothing says which is right, and are
  logged by ALID (`duplicate_conflicting`). Before, `build-dense-vcf` and
  `build-hybrid` stored the last row in silence, including the last of
  GCST90295971's seven conflicting pairs, one of them of opposite sign.
  `build-dense` refused the build, and `build-ragged-ssf` dropped conflicting
  rows with a line on stdout. The rule is Variant Routing's
  `collapse_duplicates`; `Placed` reports what it collapsed and dropped. The
  Admission record's rule set is `admission/5`.
