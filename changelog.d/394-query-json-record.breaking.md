- **`query --format json` emits the association record whole, as RFC 8259 JSON
  (issue #394, spec §19, ADR 0070).** A row carries `eaf` and
  `association_status` beside `variant_index`, `analysis_index`, `z` and `se`,
  plus `n` where the release has it. A consumer of this output must:

  - widen its row schema for the two new keys;
  - read a missing value as `null`, where the bare `NaN` and `Infinity` tokens
    used to be. `z` and `se` are `null` on a `missing` association, and `eaf`
    is `null` wherever the cell has no frequency — which is not the same thing,
    so `association_status` is what says an association is missing, and an
    imputed cell of a release with no association-frequency plane still carries
    the LD Reference Panel's frequency (spec §9, ADR 0036);
  - accept `z`, `se` and `eaf` as a number, `null`, or the string `"Infinity"`
    or `"-Infinity"`, which is how a ±infinite value is written now that the
    output parses as JSON. An infinite SE is a value (spec §15), so it may not
    be read as a missing one.

  For `opengwasdb-stores`: the query walkthrough's index-keyed form, its
  example rows and its rendered page change.
