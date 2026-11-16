- Reference Completion prepares and owns an LD block's fills as arrays,
  avoiding a Python object and repeated ALID encoding for each imputed cell
  (#422). Region caps, filled values, quality counts and format-3 checkpoint
  resume semantics are unchanged.
