- **Builds read FinnGen R13 sources as columns, and Pass 2 takes about a fifth
  of the time (issue #318, ADR 0076).** A Source Reader may now offer its
  associations as columns (`stream_association_batches`); Dense and Hybrid
  Pass 2 route them where it does, and read every other reader one association
  at a time, as before. The FinnGen R13 reader parses its file 32 MiB at a
  time with pandas and yields exactly the associations, drops and frequency
  counts of its per-row reader, which stays the reference: a block or row the
  columns cannot show they read as the per-row rules do is read by those
  rules. On a
  21,326,102-row R13 file the read takes 40.1 s against 220.7 s (median of six
  interleaved rounds; parse floor 24.7 s). Rebuilding the FinnGen pilot
  OGS-00003 (ten Analyses, ten workers), Pass 2 took 49.3 and 51.4 s an
  Analysis against 263.0 and 278.7 s, in two interleaved pairs: 81% less time,
  5.4x the throughput. Each Pass 2 worker holds about 0.27 GiB more private
  memory (0.89 and 1.00 GiB, mean, against 0.62 and 0.74). The Store, its
  `analyses.tsv` and `variants.tsv.gz`, and the Admission record were
  identical. A source the columns cannot read -- a quote, a carriage return,
  a NUL or a non-ASCII byte -- is read row by row with the same result, and
  the build logs a warning naming it, the byte and the reason. A Hybrid
  build's off-reference rows are encoded from their fields rather than from a
  key string per row. `ReaderAssociation.reference` and `ReaderBatch.reference`,
  which nothing read after #316, are removed.
