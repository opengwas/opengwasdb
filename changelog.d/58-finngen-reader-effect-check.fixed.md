- **A FinnGen r13 Analysis resolved through the FinnGen reader is checked
  against its own p-value (ADR 0069).** `resolve-analyses` reads a source
  declared `opengwasdb.finngen-r13` since issue #56, but that reader's rows
  carried no p-value, so every such Analysis recorded
  `skip_reason: no_source_p_value` for a file that reports `pval` on every
  row. The reader now reads `pval`, as the built-in `finngen-r13` Column Map
  does. It reads it for the check only: a row whose `sebeta` is unusable still
  gets no standard error from it, and Association Admission drops that row.
