- **A completion records the LD panel it read, not a constant (issue #327).**
  `provenance.completion.ld_panel_id` was `eur-hg38-gpm` on every Dense, Ragged
  and Hybrid completion whatever panel the run named, and Hybrid never passed
  one. It is now the caller's `--ld-panel-id` (new on `complete-dense`,
  `complete-ragged`, `complete-hybrid`; an empty one is an error) or, with
  none, `sha256:` and the first 16 hex digits of the panel's digest. Four
  fields are added beside it: `ld_panel_id_source`, `ld_panel_path` (as given),
  `ld_panel_digest`, `ld_panel_ancestry_dir` (spec §12.1). The digest is the
  sha256 of every file the loader reads, whole, in 5 s on the HGDP+1kGP EUR
  panel (37 GB) and 42 s on the UKB EUR panel (337 GB). A checkpoint holds the
  panel's digest, and `complete-dense-resume` and `resume_ragged_completion`
  refuse a panel whose digest differs, and a checkpoint with completed blocks
  and no digest (one written before this change). `completion.ld_panel` fails a
  release that records some of the five fields, an id its digest does not
  derive, or a Hybrid and Dense Component naming two panels; a completion that
  records none of the added fields (one before this change) gets the warning
  `completion.ld_panel_unrecorded`. A source's own `provenance.completion` is
  now merged into the completed release's, not replaced. `_LD_PANEL_ID` is
  removed.
  **Breaking, for Python callers:** `complete_dense_store`, `complete_ragged_store`
  and `complete_hybrid_store` take `(source_path, dest_path, panel, options)`:
  one `PanelSource(path, ancestry="EUR", name=None)` in place of `ld_dir`,
  `ancestry` and `ld_panel_id`, and one `CompletionOptions(min_cor, thresh,
  release_id, n_workers, overwrite)` in place of those five keywords, each with
  the default it had (both in `opengwasdb.completion.ld_panel`). The CLI flags
  are unchanged apart from `--ld-panel-id`.
