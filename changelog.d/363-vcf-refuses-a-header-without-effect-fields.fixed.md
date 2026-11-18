- **A GWAS-VCF whose header declares neither `EZ` nor both `ES` and `SE` is
  refused instead of read as a source of no association (issue #363).** The
  reader derived every record's z from one of those two spellings, so a file
  with neither yielded nothing and its Analysis was stored with 0 associations
  and no error. The refusal names the file and the FORMAT tags it did find. A
  declared tag whose record holds `.` is unchanged: that association is
  missing, not the file broken. Both `bcftools` failures the reader reports --
  the header probe and the record query -- now name the file they were
  reading. All 2,514 GWAS-VCF sources under `/data/opengwasdb/raw/gwas-vcf/`
  declare `EZ` with `ES` and `SE`, so no published Store's source is refused.
