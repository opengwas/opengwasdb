- **A row with an unusable reported SE is dropped and logged on every build
  path (`se_unusable`; issue #357, D3, ADR 0072).** An SE that is missing, not
  positive or not finite meant something different on every path:
  - `build-dense-vcf` and `build-hybrid` stored a GWAS-VCF NaN or infinite SE;
  - the tabular readers and the GWAS-VCF reader skipped a missing or zero SE
    silently;
  - `build-dense` raised, or stored an SE of 0 beside a given z.

  Now each such row reaches Association Admission, which drops it and logs it by
  source coordinate. A reader still recovers an SE it can (from a CI or a p)
  first. `ReaderAssociation` no longer refuses a negative SE: the gate judges
  it.
