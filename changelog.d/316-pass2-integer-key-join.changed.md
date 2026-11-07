- **Pass 2 matches each association to its row by an integer key join
  (issue #316).** A Routing Key is held as one `uint64` -- its chromosome, its
  position, and its alleles, exactly for a one-base SNV and as a hash of the
  whole key for anything else -- in place of a binary search over the keys'
  bytes. A hash only narrows the search: a hashed key reaches a row only when
  its whole Routing Key equals the row's, so two keys that share a hash never
  trade rows. Dense and Hybrid builds route through the same table. Spills,
  Stores and Admission records are unchanged. Rebuilding the FinnGen R13 pilot
  (OGS-00003, 21,327,043 keys, ten workers), Pass 2 took 216.7 s an Analysis
  against 275.3 s (means of three interleaved rebuilds), each Pass 2 worker
  held 0.67 GiB of private memory against 1.64 GiB, and the Store, its
  `analyses.tsv`, `variants.tsv.gz` and Admission record were identical.
