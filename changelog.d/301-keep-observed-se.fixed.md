- **`complete-ragged` keeps every observed SE exactly when SE is stored
  `int8_residual` (issue #301, ADR 0078).** Completion used to refit the
  residual SE model over the completed rows, so every observed SE was
  re-quantised. On three Chen 2023 Analyses all 135,596 observed SEs moved
  (median 0.09%, max 0.14%), and `check-observed-view` failed the pair with
  `se_mismatch`. Completion now codes every cell against the source's
  `se_coefficients` and range, as Dense and Hybrid completion already did.
  Each observed SE decodes to its source value, and an imputed SE outside the
  residual range is stored as an exact exception. Repeated from the same
  sources, no observed SE moves and the check passes. Each Analysis's
  exception share is recorded in `manifest.json` under
  `provenance.completion.se_exceptions`. Any Analysis above the 2% budget is
  named in a warning. Completion never refits the model or falls back to
  `float16` to stay under the budget, and `validate` does not enforce it.
  `format_version` does not move.
