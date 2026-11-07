- **The CLI's logging layout is exported as `LOG_FORMAT` and `LOG_DATE_FORMAT`
  (issue #311).** The layout a step's stderr is written in was spelled inside
  `logging.basicConfig`; a benchmark that parses those lines now imports the two
  constants instead of copying them, so a change to the layout reaches the
  parser and its round-trip test. Nothing a user sees changes: the same format
  and the same date format are used.
