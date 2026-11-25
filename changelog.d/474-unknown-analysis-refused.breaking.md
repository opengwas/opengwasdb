- **An unknown `analysis_id` is refused on every query that names one (issue #474).**
  `analysis()` (with or without an Indexed Variant Subset), `lookup()`, `rho()`,
  `rho_row()` and `rho_matrix()` answered an Analysis the release does not hold with an
  empty result, which reads exactly as "no association". On every layout they now raise
  `UnknownAnalysisError`, as `top_hits()` already did (#483); a `lookup()` naming one
  unknown Analysis among known ones raises too. `opengwasdb query-analysis` and
  `query-lookup` print the error and exit 1. The error carries `analysis_id`;
  `UnstoredThresholdError` carries `threshold` and `stored`, and a chromosome no ALID can
  spell raises `UnsupportedChromosome` (a `VariantNormalisationError`) carrying
  `chromosome`, so a caller reads the values without parsing the message.
