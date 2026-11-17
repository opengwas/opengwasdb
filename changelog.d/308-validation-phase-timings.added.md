- **`validate` logs every rule it runs as a timed phase (issue #308, ADR 0074).**
  Each rule that runs logs its start and its elapsed time, each pass of a shared scan
  logs a progress line with an ETA per block, and the run closes with a report
  attributing the wall time to the rules and scan passes that spent it. A phase is
  keyed by the rule and the target it judged, so a Hybrid release's Overflow and Dense
  Component are told apart rather than summed. A rule a Store does not call for, or one
  a failed prerequisite stops, gets no phase and no row, so the report never claims a
  rule was judged that was not; one that looks and finds nothing to judge logs that and
  is not timed. The report `validate` prints is unchanged, in text and in JSON: the
  timings go to the log. On the FinnGen R13 pilot (OGS-00003) the report gives 77.1 s
  to the Store Variant Table read and 29.1 s to the association scan — 71.3% and 26.9%
  of the 108.1 s it accounted for, in a run of 110.69 s — and the table's 324 blocks
  and the scan's 85 bands report progress, where the first 77 s of a validation used
  to be silent. Counting a pass can never change its verdict: a corrupt sidecar the
  count reads fails without stopping the rules that would have judged the release.
