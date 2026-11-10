- **GWAS-SSF sources, and Column Map sources through them, stream to Pass 2 as
  columns (issue #319).** `GwasSsfReader` and `ColumnMapReader` offer
  `stream_association_batches`, so a Dense or Hybrid build's Pass 2 reads a file
  a block at a time instead of one association at a time. The batches are the
  per-row reader's associations, bit for bit: the same effect source (beta,
  odds ratio, signed z, or recovered from the row's columns), the same SE
  recovery, the same rows dropped and faults named, and the same indels placed
  by `hm_code` (ADR 0079). A block the columns cannot show they read as the
  per-row rules do is read by those rules, and logged once per source with its
  byte offset and why; a source with a declared imputation score is read row by
  row throughout. Stores and Admission records are unchanged. FinnGen's
  columnar engine now lives in `opengwasdb.readers.columnar`, shared by both.
