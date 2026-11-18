- **Every builder now standardises standard errors by the Analysis's
  `original_sd` in one place, and `build-ragged-besd` and `build-ragged-ssf`
  now do it at all (issue #366, §7, ADR 0084).** Before this, `build-dense`,
  `build-dense-vcf` and `build-hybrid` each did the divide themselves, and the
  two Ragged builders read the manifest's `original_sd_method` but stored the
  source's SEs unscaled. OGS-00006, OGS-00024, OGS-00025 and OGS-00026 record
  `estimated_from_source_maf` for 4,443 Analyses between them, leave
  `original_sd` blank, and hold the source's own SEs.

  The divide now happens in `storable`, the one gate every writer passes. Its
  `phenotype_sd` argument has no default, and a completion passes "already
  applied", so each SE is divided exactly once. `z` is scale-free and is stored
  as given. `build-ragged-ssf` now records `original_sd` in `analyses.tsv`.
  Dense, Dense-from-VCF and Hybrid Stores hold the same values as before.

- **`validate` fails a Release that declares a rescaling
  `original_sd_method` without the SD it was divided by.** The new rule,
  `analyses.original_sd_recorded`, fails such an Analysis when `original_sd` is
  blank, non-finite or not positive, and names it. It applies at every
  `format_version`, so it fails the four Releases above until they are
  repaired.

- **Every builder refuses a non-finite `original_sd`.** A missing, non-finite
  or non-positive magnitude now refuses the build and names the Analysis,
  instead of being divided by. `build-dense`, `build-dense-vcf` and
  `build-hybrid` already refused a non-positive `original_sd`, but they
  accepted `inf`, which stores every standard error as 0.
