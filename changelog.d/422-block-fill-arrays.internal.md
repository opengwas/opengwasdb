- Reference Completion prepares and owns an LD block's fills as arrays,
  avoiding a Python object and repeated ALID encoding for each imputed cell
  (#422). Region caps, filled values, quality counts and format-3 checkpoint
  resume semantics are unchanged.
  On the profiled OGS-00009 block (7,024 variants, 2,514 Analyses), two
  interleaved measurements reduce mean block time from
  189.3 to 137.5 seconds (27.4% less), with byte-identical
  checkpoints. The completed OGS-00003 pilot retains identical array values
  and metadata.
