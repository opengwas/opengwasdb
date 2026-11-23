- **A format 0.3.0 release no longer carries what nothing read, or what only the
  validator read and can recompute (issue #471, ADR 0102).** Not released until #468–#472
  land on epic #450's branch. From 0.3.0:
  - `index.sqlite` holds only `completion_quality`, so it ships only in a
    Reference-Completed release (a Hybrid's in its Dense Component). An Observed-Only
    release, and a Hybrid's outer release, has none, and the closed envelope fails one
    that does (`envelope.closed`); one holding anything else fails
    `sqlite.provenance_only`.
  - The `index.sqlite` `metadata` table (the `dense` layout blob, `schema_version`,
    `n_variants`, `n_analyses`) and `variant_aliases` are gone. `manifest.json` is the
    one record of the Dense layout.
  - The Top-Hit tiers store `z` but not `abs_z` or `p_value` (575 MB on OGS-00009). The
    validator recomputes both from `z`; a new rule, `top_hits.threshold`, fails a tier
    row whose |z| does not clear its threshold, and `top_hits.abs_z` holds a 0.2
    release's stored pair to what its `z` implies.
- **On 0.3.0 the Store Reader never opens `index.sqlite`.** An rsID that the rsID index
  lacks is a miss, and SQLite is not loaded. A 0.2 Dense or Hybrid release still falls
  back to `variant_aliases`.
- **`ogdb rewrite-release` drops them** in two steps, `provenance_index` and
  `tier_stored_p`, which leave them out of the copy. It refuses a source in three
  cases: its `variant_aliases` names an rsID the rsID index lacks; its `dense` blob
  disagrees with its manifest; or a tier's stored `abs_z` or `p_value` is not what its
  `z` implies. Its proof also asks about rsIDs no release names.
- **`ogdb build-dense-top-hits` and `build-ragged-top-hits` keep a release's format.**
  A 0.2 release's rebuilt tiers still store `abs_z` and `p_value`, recomputed from `z`;
  a 0.3.0 release's store neither.
