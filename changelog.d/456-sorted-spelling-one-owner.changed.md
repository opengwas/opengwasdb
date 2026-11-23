- **`opengwasdb.variants` owns both ALID spellings, and an LD Store lookup
  refuses an indel ALID spelt reference first (issue #456).** The LD
  package's own copy of the engine's identity rules, `opengwasdb.ld.identity`,
  is gone. LD Stores still spell every pair sorted, an indel's too (ADR 0079
  §5; their format does not move until #462), and now form that spelling with
  the engine's own code: `opengwasdb.variants.sorted_spelling`
  (`sorted_identity`, `SortedIdentity`, `sorted_alid_locus`,
  `resolve_sorted_reversed_indels`), judged by `orient_to_canonical` and
  spelt by `sorted_alid_alleles`. `opengwasdb.variants.normalise` gains
  `AUTOSOMES` and `chromosome_rank`, which raises `VariantNormalisationError`
  for a label off `1`–`22`, `X`, `Y`, `MT` where the LD copy raised
  `KeyError`. Refusals are the engine's `VariantNormalisationError` family in
  place of `IdentityError`; two records that are not one site's split raise
  `DuplicateRecord` (reason `duplicate_record`). An LD Store reader given a
  deletion's reference-first ALID, such as `1:827587:GGGGCC:G`, used to answer
  Absent, as if the panel lacked the variant; it now raises
  `ReferenceFirstIndel`, naming the sorted spelling and #462, and `ogld` exits
  2. Every other lookup answers as before, and the Test Store's Variant Set is
  unchanged row for row.
