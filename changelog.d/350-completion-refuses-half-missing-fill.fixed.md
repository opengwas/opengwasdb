- **`complete-dense` refuses to store a fill with one of z and SE missing
  (issue #350, ADR 0077).** Its z and SE band passes each filled a cell where
  their own plane was missing, so a fill record with a z and no SE was
  stored as `z 5.0, SE NaN, imputed 1`, a cell spec §15 forbids. One pass
  now fills both, and judges each completed band whole before any plane
  holds it: such a cell raises `NotAdmitted` with Association Admission's
  rule, naming the rows. No fill a completion makes is one, since each
  candidate passes `admissible` first; the check is that the write cannot
  store one. The merged pass reads each fill shard and the source SE plane
  once, where the two passes read both twice. On 1.14M completed rows of
  OGS-00016's 1,691 Analyses it took 1,611 s against 2,147 s. It holds one
  more float32 band (1.35 GB there; the traced peak rose 1.27 GB), and it
  writes the same bytes.
