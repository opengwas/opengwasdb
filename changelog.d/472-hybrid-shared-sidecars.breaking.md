- **A format 0.3.0 Hybrid's Dense Component no longer carries a copy of the shared
  variants or Analyses (issue #472, ADR 0102 §4, ADR 0105).** Not released until
  #468–#472 land on epic #450's branch. From 0.3.0, `dense/` holds `manifest.json`,
  `data.zarr/`, `dense_to_shared.npy` and, Reference-Completed, `index.sqlite`. It no
  longer carries `variants.tsv.gz`, `variants.tsv.gz.tbi`, the ALID sidecar
  (`variant_alid_utf8/offsets/keys/order.npy`), the position sidecar
  (`variant_position*.npy`), the rsID index (`variant_rsid_bytes/rows.npy`),
  `analyses.tsv` or `overview.html`. Each is the shared one's rows again, and on a
  completed release the copy had lost rsIDs and every `source_alid` (#481). On
  OGS-00004 (Hybrid Observed-Only, 13,718,605 panel rows) the 0.2 copies come to
  1,098,940,214 B; a 0.3.0 component that kept its copy would carry 1,265,761,783 B.
  The closed envelope fails a 0.3.0 Dense Component carrying any of them.
- **The Dense Component's variants are the shared axis's, through `dense_to_shared`.**
  Opened as a release, its `variant_axis()` is the shared axis in its panel rows
  (`opengwasdb.store_reader.panel_axis.PanelVariantAxis`), and its `analyses_path` is
  the Hybrid's. `VariantAxis(<hybrid>/dense)` on a 0.3.0 release raises
  `DenseComponentSharesAxis` instead of opening files that are not there. A Hybrid
  query resolves an identifier and a range once, against the shared sidecars.
- **The Dense Component's own `n_missing_se_out_of_range` is no longer recorded.** The
  Hybrid's `analyses.tsv` counts both components, as it did. A completion of a 0.3.0
  Hybrid counts what a completion of its 0.2.3 form counts; the
  `analyses.se_out_of_range_partition` rule has nothing to compare on 0.3.0.
- **`ogdb rewrite-release` drops the copies** in a new step, `dense_shared_axis`, and
  refuses, before anything is written, a source whose `dense_to_shared.npy` does not
  name, for each Dense row, the shared row holding that row's ALID, in range and
  strictly ascending. It does not compare the copies' rsID or `source_alid`.
- **Validation.** `hybrid.dense_to_shared` reports a row out of range, a row named
  twice and a descending pair each by itself; `hybrid.coverage` fails a map that does
  not name one shared row for each row of a 0.3.0 Dense Component's grid. The
  `variant_table.*` rules do not judge a 0.3.0 Dense Component, and the `analyses.*`
  table rules judge a 0.3.0 Hybrid's own `analyses.tsv`.
- **Reading a missing Variant Table raises `FileNotFoundError`** instead of crashing the
  process inside pysam.
