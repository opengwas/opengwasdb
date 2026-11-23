- **`opengwasdb.variants` owns both ALID spellings, and an LD Store lookup
  refuses an indel ALID spelt reference first (issue #456).** The LD
  package's own copy of the engine's identity rules, `opengwasdb.ld.identity`,
  is gone. LD Stores still spell every pair sorted, an indel's too (ADR 0079
  §5; their format does not move until #462), and now form that spelling with
  the engine's own code: `opengwasdb.variants.sorted_spelling`
  (`sorted_identity`, `SortedIdentity`, `sorted_alid_locus`,
  `resolve_sorted_reversed_indels`), judged by `orient_to_canonical` and
  spelt by `sorted_alid_alleles`. `opengwasdb.variants.normalise` gains
  `AUTOSOMES`, `CANONICAL_CHROMOSOMES` and `chromosome_rank`, which raises
  `VariantNormalisationError` for a label off `1`–`22`, `X`, `Y`, `MT` where
  the LD copy raised `KeyError`. Refusals are the engine's
  `VariantNormalisationError` family in place of `IdentityError`:
  `chromosome_missing` and `allele_invalid` keep their typed classes and
  reasons, and two records that are not one site's split raise
  `DuplicateRecord` (reason `duplicate_record`). The rest (an unsupported
  chromosome number, a bad position, identical alleles) had reason `""`. They
  now arrive as a bare `VariantNormalisationError`, which has no `.reason`
  attribute. An LD Store reader given a deletion's reference-first ALID, such
  as `1:827587:GGGGCC:G`, used to answer Absent, as if the panel lacked the
  variant. It now raises `ReferenceFirstIndel`, and `ogld` exits 2. The
  message names the sorted ALID the Store would hold it under
  (`1:827587:G:GGGGCC`), says that row is the deletion unless the panel also
  has the insertion there, and that r there is signed against the other
  allele (#462). One case stays silent: an insertion's reference-first ALID is
  also a sorted ALID, so at a tandem repeat where the Store holds only the
  deletion of the same two alleles, it finds the deletion's row. The Store
  records no REF to tell them apart (recorded on #462). Every other lookup
  answers as before, and the Test Store's Variant Set is unchanged row for
  row.
