- **`associations.se_infinite`'s warning no longer says `beta = z * se` reads as
  infinity (issue #473).** Since ADR 0104 a query reads `beta` and `pval` as NaN
  where a release before format 0.2.3 stores `+inf` SE beside a finite z; the
  warning now says so, and that the stored values are that z and `+inf`.
