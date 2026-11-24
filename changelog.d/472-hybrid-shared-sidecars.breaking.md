- **A format 0.3.0 Hybrid's Dense Component no longer carries a copy of the shared
  variants or Analyses (issue #472, ADR 0102 §4, ADR 0105).** From 0.3.0, `dense/` holds `manifest.json`,
  `data.zarr/`, `dense_to_shared.npy` and, Reference-Completed, `index.sqlite`. It no
  longer carries `variants.tsv.gz`, `variants.tsv.gz.tbi`, the ALID sidecar
  (`variant_alid_utf8/offsets/keys/order.npy`), the position sidecar
  (`variant_position*.npy`), the rsID index (`variant_rsid_bytes/rows.npy`),
  `analyses.tsv` or `overview.html`. Each is the shared one's rows again, and on a
  completed release the copy had lost rsIDs and every `source_alid` (#481). On
  OGS-00004 (Hybrid Observed-Only, 13,718,605 panel rows) the 0.2 copies come to
  1,098,940,214 B, and on OGS-00011 as completed (14,267,257 panel rows) to
  1,175,803,588 B; a 0.3.0 component that kept its copy would carry 1,265,761,783 B and
  1,347,866,437 B. Both were rewritten read only with the version refusal bypassed
  (neither is 0.2.3). OGS-00004, with its `hm3` Indexed Variant Subset, was rewritten
  with this package: the proof matched on every question, the subset's 16 included, and
  the whole validator passed the copy but for the `n_missing_se_out_of_range` column a
  0.2.2 source lacks. OGS-00011's comparison, both sides read by this package, left out
  the one broad `range_phewas`, too large for it, and changed no answer on the rest. The
  closed envelope fails a 0.3.0 Dense Component carrying any of the dropped files.
- **The Dense Component's variants are the shared axis's, through `dense_to_shared`.**
  Opened as a release, its `variant_axis()` is the shared axis in its panel rows
  (`opengwasdb.store_reader.panel_axis.PanelVariantAxis`), and its `analyses_path` is
  the Hybrid's. `VariantAxis(<hybrid>/dense)` on a 0.3.0 release raises
  `DenseComponentSharesAxis` instead of opening files that are not there, and a
  `dense/` under a parent `manifest.json` that cannot be read raises
  `UnreadableHybridManifest`. A Hybrid query resolves an identifier and a range once,
  against the shared sidecars.
- **0.2 Hybrids are read through the shared axis too.** On every format, a Hybrid query's
  Dense Component is now read through the shared Variant Axis and the shared Analyses,
  never its own copy, because a completed 0.2 Hybrid's copy lost rsIDs and every
  `source_alid` the shared table keeps (#481). `range_phewas` was the last query still
  reading the copy, by position; `phewas` and `lookup` had read the shared axis since
  #465. What shows the old read path gives the same answers: the golden digests of the
  two Hybrid fixtures, pinned by the reader before #472, are unchanged; and on OGS-00004
  the package before #472 reading the 0.2.2 source and this one reading the rewritten
  0.3.0 copy agreed on all 142 questions, both `range_phewas` included, and every probe.
- **The Dense Component's own `n_missing_se_out_of_range` is gone with
  `dense/analyses.tsv`.** The Hybrid's `analyses.tsv` counts both components, as it did;
  the Dense Component's own count is no longer recorded anywhere. What its completion left
  out is still recorded in its manifest (`provenance.completion.se_out_of_range`), and
  `check-observed-view` accounts that against what it finds
  (`observed_view._component_accounting`). A completion of a 0.3.0
  Hybrid counts what a completion of its 0.2.3 form counts; the
  `analyses.se_out_of_range_partition` rule has nothing to compare on 0.3.0.
- **`ogdb rewrite-release` drops the copies** in a new step, `dense_shared_axis`, and
  refuses, before anything is written, a source whose `dense_to_shared.npy` does not
  name, for each Dense row, the shared row holding that row's ALID, in range and
  strictly ascending. It does not compare the copies' rsID or `source_alid`.
- **`ogdb rewrite-release` carries an Indexed Variant Subset to the new release.** A new
  step, `indexed_subsets`, records the new `release_id` and `format_version` in each
  subset, which holds the same cells of the new release (ADR 0061); before, every
  rewritten release with a subset failed `subsets.indexed`. It refuses a source whose
  subset already names another release. The proof asks each subset the spread Analyses.
- **Validation.** `hybrid.dense_to_shared` reports a row out of range, a row named
  twice and a descending pair each by itself; `hybrid.coverage` fails a map that does
  not name one shared row for each row of a 0.3.0 Dense Component's grid. The
  `variant_table.*` rules do not judge a 0.3.0 Dense Component, and the `analyses.*`
  table rules judge a 0.3.0 Hybrid's own `analyses.tsv`.
- **Reading a missing Variant Table raises `FileNotFoundError`** instead of crashing the
  process inside pysam.
