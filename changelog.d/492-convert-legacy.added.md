- **`ogdb rewrite-release --convert-legacy --lifted|--not-lifted` converts a 0.2.0, 0.2.1
  or 0.2.2 Dense release to 0.3.0 (issue #492).** It fixes what can be fixed exactly: every
  SE above 65,504 is masked (grid, Indexed Variant Subsets, Top-Hit tiers) and counted in
  `n_missing_se_out_of_range`, and an unlifted release's rows name their own ALID as
  `source_alid`. What only a rebuild fixes, a source's sorted indels and a lifted release's
  missing `source_alid`, is recorded by validator rule name in
  `provenance.converted_from.deviations`, and `validate` reports exactly those rules as
  warnings naming the conversion. The proof compares every deliberate change cell for cell.
  The rewrite's range question now spans at most 10,000 variants, so a large release can be
  proved; every fixture's range is unchanged.
