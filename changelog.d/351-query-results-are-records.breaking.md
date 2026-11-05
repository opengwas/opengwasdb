- **Query methods return an `opengwasdb.associations.Associations` record, not
  a `dict` (issue #351, ADR 0070).** It reads as the same mapping, with the
  same keys and dtypes, so `result["z"]`, `items()` and `"n" in result` work
  as before. It cannot be mutated and is not a `dict` instance: a caller that
  adds a column to a result, or tests `isinstance(result, dict)`, takes
  `dict(result)` first.
