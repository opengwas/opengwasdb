- **Completion keeps every observed EAF exactly, on every layout, and Dense and
  Hybrid completion keep every observed SE exactly (issue #390, extends #301).**
  `encode_eaf` takes the `exact` mask `encode_se` has, threaded from the one
  `carried` flag: an observed frequency whose residual code would decode to any
  other `float32` is held as an exact exception, judged by decoding through the
  one EAF decoder, and a cell already on a code's value keeps the code. Ragged
  completion's EAF writer, Dense `_encode_band` and the Hybrid crossover fold
  now pass `carried`; Dense and Hybrid had re-quantised an observed exception
  that sat off the code grid (a half step: EAF 0.15048 became 0.15073, SE
  0.10933 became 0.10946 on the fixture). Dense and Hybrid completion also record
  each Analysis's SE exception share in `provenance.completion.se_exceptions`,
  as Ragged has since #301; no `analyses.tsv` column is added and `format_version`
  does not move.
