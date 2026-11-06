- **A Variant Reference artifact records the lift's re-orientation, and one
  without that record is refused (issue #354, ADR 0071).** A reverse-strand
  chain block complements a variant's alleles. `extract-variant-reference`
  named such a key by its pre-lift alleles at the lifted position, and a
  single-pass build read the artifact as re-orienting nothing, so it stored the
  wrong ALID, or, for a palindromic pair, the right ALID with the wrong sign.
  Each row's ALID is now formed from the projected alleles, and a new
  `reoriented_keys` column lists the row's source keys whose statistics the lift
  re-orients. A build given `--variant-reference` re-signs exactly those, so it
  stores what the two-pass build stores (spec §4).
  - An artifact without `reoriented_keys`, which every earlier
    `extract-variant-reference` wrote, is refused with `UnmarkedArtifactError`
    naming the file. Extract it again. A plain ALID list and a Store Variant
    Table are read as before.
  - `write_variant_reference` takes a required `reoriented=` keyword, and
    `VariantReference` a required `reoriented` field: an artifact always says
    which keys its lift re-oriented, and an empty set is a statement, not a
    default.
  - On OGS-00008's manifest (10 UKB-B hg19 Analyses), the artifact routes all
    9,847,706 source keys to the two-pass ALID with the two-pass sign, 11,286
    of them re-oriented.
