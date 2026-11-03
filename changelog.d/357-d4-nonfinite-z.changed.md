- **A row with an infinite z is dropped and logged, not refused (`z_nonfinite`;
  issue #357, D4, ADR 0072).** Before, `build-dense` raised on one. A tabular
  source's reached the encoder and failed the build. The GWAS-VCF reader
  skipped an infinite ES in silence, and replaced an infinite EZ by ES/SE.
  FinnGen and GWAS-SSF read an infinite beta as no effect. Every reader now
  keeps an effect it reports as infinite, and one association's statistic no
  longer refuses its Analysis: Association Admission drops the row and logs it
  by source key, on every build path. The resolver's metrics still read an
  infinite beta as absent. The Ragged builders now log the rows they used to
  skip for a missing effect (`effect_missing`) or an infinite one. Spec §6a
  says so.
