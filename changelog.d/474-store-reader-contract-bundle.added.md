- **The Store Reader contract bundle (issue #474).** `tests/store_reader_contract/` holds
  nine small format 0.3.0 Store Releases -- Dense, Ragged and Hybrid, each Observed-Only
  and Reference-Completed -- and what each answers to every question the rewrite's proof
  asks, as digests and as readable rows. ogquery, opengwaspy, opengwasr and any other port
  of the Store Reader test against it; its README names what each release exercises.
  `query_digests` asks every question once (`questions`) and renders each answer as a
  digest and as readable rows (`release_answers`); it now also asks the refusals the query
  contract defines, so `ogdb rewrite-release` compares them between the source and the
  rewritten release. `pixi run -e dev contract-check` regenerates the bundle and fails on
  any difference. `list_indexed_subsets`, `indexed_subset_namespace_path` and
  `is_valid_indexed_subset_name` are the Store Reader's (`opengwasdb.store_reader.indexed`).
