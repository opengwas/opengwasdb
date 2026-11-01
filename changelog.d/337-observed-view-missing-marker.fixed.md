- **`check-observed-view` reads Z/SE missingness through each plane's missing
  marker, so a cell holding a finite Z and an infinite SE is a value rather
  than a missing one (issue #337, spec §15, ADR 0038 §6).** Completion that
  reproduced its source's eight `+inf` SE cells exactly was refused on 31 Oct
  2026 with `overflow.z_se_missingness_mismatch=2`, because the check tested
  `isfinite` where the spec says missing marker; such a cell now passes when it
  is identical in both Stores and fails as a `se_mismatch` when it is not. The
  check reports a new informational `infinite_cells` counter in each part's
  totals — Dense, Overflow, crossovers and Ragged — and `validate` warns, without
  failing the release, when a Store holds an infinite SE, naming the cell count
  and the first Analyses holding one.
