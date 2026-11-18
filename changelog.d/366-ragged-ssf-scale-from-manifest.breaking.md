- **`build-ragged-ssf` has no `--stored-effect-scale`, and `build_ragged_from_ssf`
  has no `stored_effect_scale` parameter (issue #366, ADR 0084).** The option
  defaulted to `sd` for every Analysis, which labelled a row with no
  `original_sd_method` as SD Units while storing its source's SEs undivided.
  The scale is now each manifest row's `stored_effect_scale` column. A row
  that declares none, or one the vocabulary does not hold, is refused, naming
  the Analysis. A row declaring `sd` must also declare a rescaling
  `original_sd_method` (with its `original_sd`) or `declared_standardised`;
  `log_or` and `log_hazard` rows need no SD. Add the column to the manifest
  and drop the flag. Every published Ragged Release but OGS-00001 and OGS-00002 declares
  both columns in its `analyses.tsv`; those two are blank in both.
