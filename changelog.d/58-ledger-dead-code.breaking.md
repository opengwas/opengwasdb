- **Four dead names are removed from the package's interface: `opengwasdb.build`
  no longer exports `build_dense_observed_from_sources`, and `stats` no longer
  has `beta_from_z_se` or `finite_pair` nor `index` its `count_rows`
  re-export (issue #58).** No caller exists in this repository's package code or
  in `opengwasdb-stores`. The one caller of `beta_from_z_se` was a test,
  `tests/test_indexed_subset_query.py`, which now multiplies z by se itself —
  the arithmetic the derived-statistic contract says a caller has to do anyway.
  `opengwasdb/query/results.py` is deleted as well: ADR 0020 removed
  `AssociationResult` and left a docstring-only module nothing imported. The
  builder itself is unchanged and is still reached as
  `opengwasdb.build.observed.build_dense_observed_from_sources`, which is what
  the `build-dense` command, ten test modules and `conftest.py` call.
  `variant_by_identifier` and `_parse_canonical_alid`, the ledger's other two
  `index.sqlite` rows, had already gone — with the `variants` table in #128
  (`a8dcea1`) and in the cleanup that followed it (`7eefe7d`).
