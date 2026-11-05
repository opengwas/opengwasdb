- **A row with a missing chromosome or an allele outside A, C, G and T is
  dropped and logged, the same on every reader (issue #365, ADR 0072).** The
  GWAS-VCF build streams validated no allele, and every reader kept `NA` and `.`
  as chromosome names, so such a row became an ALID such as `NA:2000:A:G`. The
  tabular readers dropped an allele outside the alphabet unseen, and so did
  GWAS-SSF and FinnGen with a row too short to reach its alleles, a missing
  allele; `build-dense` refused the whole source. Now every Source Reader
  leaves the row out of its associations and its variants, and the build's
  Admission record counts and logs it under `chromosome_missing` or
  `allele_invalid`, ahead of every other rule. An alt, random or unplaced contig such as
  `11_KI270927v1_alt` is still a chromosome. No published Store holds such an
  ALID (a census of every release), so no published release changes.
