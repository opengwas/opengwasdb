- **Every resolver manifest now declares `p_value_test` for each Analysis, with no
  default (issue #371, ADR 0085).** It names the test the source's `p_value` comes
  from: `wald_1df`, `joint_2df` or `mr_mega`. `resolve-analyses` refuses a
  manifest without the column, or a row whose value is blank or unknown.
  - `mr_mega` is refused before the source is read. MR-MEGA's beta/se is a t on
    (cohorts − 2) df, not a z.
  - `joint_2df` resolves with the effect-consistency check skipped
    (`skip_reason = joint_test_p_value`).
  - `wald_1df` is checked as before.

  The declaration is bound into the fingerprint, so every existing Resolution
  Record is recomputed by `--resume`. `AnalysisRequest.p_value_test` is a
  required keyword argument.
