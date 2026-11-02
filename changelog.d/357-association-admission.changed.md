- **Association Admission decides which of an Analysis's source rows a Store
  admits, in one module (`opengwasdb.admission`, issue #357).** The resolver and
  the Hybrid builder judge every batch through one per-Analysis `Gate`, which
  drops a row for exactly one reason (INFO first, then MAF) and counts it under
  that reason. `opengwasdb.build.row_admission` is gone. The resolver's record
  and a Hybrid Store's `info_score` and `maf` provenance are unchanged. A
  `build-hybrid --checkpoint` record now has format 3, and a format-2 record is
  refused on resume, because its Pass 2 counts predate the reasons.
