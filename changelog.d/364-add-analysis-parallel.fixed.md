- **`RaggedCSRWriter.add_analysis` refuses arrays that are not parallel, naming
  each one's length (issue #364).** A length-1 SE handed three variants was
  broadcast to all three, so three associations carried that one SE's value and
  the write succeeded; a short `eaf` or `variant_index` raised `IndexError` from
  a boolean mask instead, after the source rows had been read. The writer now
  applies the Association Codec's one rule for "parallel" — `require_parallel` —
  at its entry, before any array is cast or stored, naming `variant_index`, z,
  SE, `eaf` and `n` with their lengths. No build reaches it: every builder
  reads an Analysis's arrays from one row set.
