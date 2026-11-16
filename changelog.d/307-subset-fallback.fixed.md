- `validate` now warns when an Indexed Variant Subset cannot use the association
  pass's digests because the subset identity changed. It compares all units cell
  by cell against the primary planes, rather than silently taking the slower path
  (issues #314, #315).
