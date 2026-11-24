- **The Store Reader contract bundle (issue #474).** `tests/store_reader_contract/` holds
  nine small format 0.3.0 Store Releases -- Dense, Ragged and Hybrid, each Observed-Only
  and Reference-Completed -- and what each answers to every question the rewrite's proof
  asks, as digests and as readable rows. ogquery, opengwaspy, opengwasr and any other port
  of the Store Reader test against it; its README names what each release exercises.
  `query_digests.release_digests` now also asks the refusals the query contract defines
  (`top_hits` of an unknown Analysis, a range on a chromosome no ALID can spell), so
  `ogdb rewrite-release` compares them between the source and the rewritten release.
  `list_indexed_subsets` is the Store Reader's (`opengwasdb.store_reader.indexed`), and
  `opengwasdb.layouts.dense.indexed_subsets` re-exports it.
