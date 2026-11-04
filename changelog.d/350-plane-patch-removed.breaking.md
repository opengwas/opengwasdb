- **`DenseZPlane.patch`, `DenseEafPlane.patch` and `DenseSePlane.patch` are
  removed, and so are `opengwasdb.encoding`'s `write_se_dense`,
  `write_se_csr` and `write_eaf_csr` (issue #350, ADR 0077).** Each wrote one
  plane of a cell, so a caller could store a z without its SE, or code an SE
  against a frequency its cell no longer held. A cell is now written whole:
  `AssociationCodec.of_component(group, plan).encode(CellValues(z, se, eaf),
  Placement.pairs(rows, cols, n_analyses), where=...)`, then
  `write_dense_cells(group, plan, rows, cols, encoded)`; a whole component is
  written with `write_encoded_planes`. `opengwasdb-stores`'
  `tests/workflow/test_workflow.py` perturbs an observed z with
  `DenseZPlane.patch` twice, and has to move to these.
