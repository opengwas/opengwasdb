- **A frequency outside (0, 1) is absent in every build, and counted (issue
  #357, D7, ADR 0072).** Every Source Reader already read a frequency of exactly
  0 or 1, one beyond the interval or one that is not a finite number as absent
  (ADR 0036); `build-dense` stored a record's frequency as given, and now reads
  it the same way. Each Analysis's entry in the Admission record counts these
  (`eaf_made_absent`), over the rows its source yielded; no row is dropped for
  one. `build-ragged-besd` logs its `.esi`'s count once, since that frequency is
  every probe's. `SourceReader.stream_associations` takes an optional
  `StreamTally`, which counts them.
