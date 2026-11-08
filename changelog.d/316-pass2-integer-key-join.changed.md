- **Pass 2 matches each association to its row by an integer key join
  (issue #316).** A Routing Key is held as one `uint64` -- its chromosome, its
  position, and its alleles, exactly for a one-base SNV and as a hash of the
  whole key for anything else -- in place of a binary search over the keys'
  bytes. A hash only narrows the search: a hashed key reaches a row only when
  its whole Routing Key equals the row's, so two keys that share a hash never
  trade rows. Dense and Hybrid builds route through the same table. Spills,
  Stores and Admission records are unchanged. Rebuilding the FinnGen R13 pilot
  (OGS-00003, 21,327,043 keys, ten workers), Pass 2 took 21% less time an
  Analysis: 216.7 s against 275.3 s, means of three interleaved rebuilds, or
  1.27x the throughput (9%, 29% and 24% less in the three pairs). Each Pass 2
  worker's private memory fell from 1.66 and 1.61 GiB to 0.68 and 0.65 GiB;
  its VmHWM barely moved (9.06 to 8.80 GiB), being mostly pages it shares with
  the parent. The Store, its `analyses.tsv`, `variants.tsv.gz` and Admission
  record were identical.
