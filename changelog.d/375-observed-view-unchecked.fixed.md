- **A part `check-observed-view` did not compare reports its tallies as `null`, not 0
  (issue #375).** When the axes do not map, the Analyses differ, or a Hybrid part is
  skipped, no cell is inspected there — but the verdict wrote that part's counters as 0,
  so `se_mismatch: 0` read as a clean census of cells nobody looked at. Every counter, each
  `max_abs_diff`, the stratum tallies and `all_analyses` are now `null` for an uncompared
  part, and `null` in `crossovers.totals` too when the crossover comparison is skipped. The
  keys are unchanged, and a verdict whose parts were all compared is byte-identical apart
  from `seconds`, so `opengwasdb-stores` renders an unchecked counter as `unchecked` rather
  than `None` or 0. `bands_checked` and `analyses_checked` stay 0: they count what was read,
  and nothing was. A `null` tally neither fails a verdict nor passes one; the structural
  failure still names itself in `failures`.
