- **`extract-variant-reference` records the lift's re-orientation instead of
  refusing a reverse-strand lift, and the artifact format changes both ways
  (issue #354, ADR 0071).** This supersedes the refusal 0.5.0 introduced. A
  reverse-strand chain block complements a variant's alleles; each row's ALID
  is now formed from the projected alleles, and a new `reoriented_keys` column
  lists the row's source keys whose statistics the lift re-orients. A build
  given `--variant-reference` re-signs exactly those, so it stores what the
  two-pass build stores (spec §4). The routing column is renamed from
  `source_keys` to `routed_source_keys`.
  - An artifact whose header names `source_keys`, as every earlier
    `extract-variant-reference` wrote, is refused with `UnmarkedArtifactError`
    naming the file, by the single-pass builds, `build-ragged-besd`'s lift check
    and `resolve-analyses --variant-reference`. Extract it again. A plain ALID
    list and a Store Variant Table are read as before.
  - An engine at 0.5.0 or earlier refuses to route by a new artifact (`missing
    the column(s) source_keys`) rather than route it without its
    re-orientation; its BESD lift check and resolver read only the artifact's
    ALIDs, which are the projected ones. Extract and build with one release.
  - `ReverseStrandLiftError` is removed from `opengwasdb.variants.routing`.
    Extraction no longer raises it and nothing replaces it, so drop the import
    and its `except` clause. It was a `ValueError`, so an `except ValueError`
    that caught it needs no change.
  - `UnmarkedArtifactError`, a `ValueError`, is new in `opengwasdb.variants`, and
    `refuse_unmarked_artifact`, the header check every reader calls, in
    `opengwasdb.variants.reference`.
  - `write_variant_reference` takes a required `reoriented=` keyword, and
    `VariantReference` a required `reoriented` field: an artifact always says
    which keys its lift re-oriented, and an empty set is a statement, not a
    default.
  - A key list with an empty element (`;`, `key;;`) or a key named twice is
    refused, naming the row; an empty cell lists no keys.
  - On OGS-00008's manifest (10 UKB-B hg19 Analyses), the artifact routes all
    9,847,706 source keys to the two-pass ALID with the two-pass sign, 11,286
    of them re-oriented.
