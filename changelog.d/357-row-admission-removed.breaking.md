- **`opengwasdb.build.row_admission` is removed (issue #357, ADR 0072).** Its
  rule, `admit_rows`, is Association Admission's declared-filter stage
  (`opengwasdb.admission`), which the resolver and every builder judge through.
