- **Association Admission decides which of an Analysis's source rows a Store
  admits, in one module (`opengwasdb.admission`, issue #357).** The resolver and
  every builder judge each batch through one per-Analysis `Gate`, which drops a
  row for exactly one reason and counts it under that reason. The rules that
  decide whether a row is an association come first (an unusable reported SE,
  then no effect), then the declared INFO and MAF filters.
  `opengwasdb.build.row_admission` is gone. The resolver's record and a Hybrid
  Store's `info_score` and `maf` provenance are unchanged.
- **A Source Reader now yields a row with no effect, with a NaN z,** where it
  used to skip it. Association Admission drops it (`effect_missing`) and logs it
  by source coordinate. A GWAS-SSF file that names no effect column still
  yields no rows.
- **An interrupted `build-hybrid --checkpoint` from an older package must be
  restarted:** the checkpoint format is now 3, and a resume refuses a format-2
  checkpoint. Its Pass 2 counts now live in the build's Admission record.
