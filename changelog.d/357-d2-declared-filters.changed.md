- **A declared INFO or MAF threshold filters every build, not only a Hybrid one
  (issue #357, D2, ADR 0072).** `build-dense-vcf` and `build-ragged-ssf` read an
  Analysis's declared imputation score and frequency, and drop and count the
  rows the threshold excludes, exactly as `build-hybrid` does. Before, they
  ignored a declared threshold. `build-ragged-besd` judges a declared MAF
  threshold on the `.esi` frequency. Without `--store-esi-eaf` it has none, so
  it refuses such a threshold, and a BESD source has no imputation score, so an
  INFO threshold there is refused when the overlay is read. `build-dense` reads
  a `maf_threshold` column in its records, one threshold per Analysis, and
  judges it on the records' own frequency; its records carry no imputation
  score, so it refuses an `info_score_threshold` other than `NaN`. No published
  Dense or Ragged release declares a threshold. Every manifest-driven builder
  reads the policies one way (`opengwasdb.admission.declared.declared_rules`).
