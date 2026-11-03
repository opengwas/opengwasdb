- **A row with an infinite z is dropped and logged, not refused (`z_nonfinite`;
  issue #357, D4, ADR 0072).** Before, `build-dense` raised on one. A tabular
  source's reached the encoder and failed the build. The GWAS-VCF reader
  skipped one in silence. One association's statistic no longer refuses its
  Analysis: Association Admission drops the row and logs it by source key, on
  every build path. The Ragged builders now log the rows they used to skip for
  a missing effect (`effect_missing`) or an infinite one. Spec §6a says so.
