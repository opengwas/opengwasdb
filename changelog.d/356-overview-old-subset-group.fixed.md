- **`overview.html` renders an Indexed Variant Subset written before the
  Overflow-only record, instead of `regenerate-overview` failing on the store
  (issue #356, ADR 0065).** The page formatted an absent `overflow_only_count`
  as a digit-grouped number, which raised `ValueError: Cannot specify ',' with
  's'` on OGS-00012 and left its `overview.html` unregenerable. The count reads
  `0`, as ADR 0065 defines it — the builder that wrote such a group could only
  refuse an Overflow-only ALID, so it indexed all it could rather than leaving a
  count untaken — and an attribute no decision gives a meaning reads blank. The
  tab also shows the Overflow-only column the store format already calls for.
