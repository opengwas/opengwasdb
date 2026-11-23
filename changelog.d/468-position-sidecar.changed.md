- **A 0.3.0 release answers `range_phewas` from a position sidecar, without tabix or
  pysam (issue #468, ADR 0105 §6).** Not released until #469–#472 land on epic #450's
  branch. Every build and completion writes three more files beside the Variant Table:
  `variant_position.npy` (each row's position, `<u4`), `variant_position_chromosomes.npy`
  and `variant_position_chromosome_offsets.npy` (each chromosome's rows). A range is two
  binary searches in its chromosome's slice. It answers exactly as a 0.2 release answers
  by tabix, which is unchanged. A Hybrid's nested Dense Component carries its own sidecar.
  A 0.3.0 release missing it is refused by name. The writer refuses rows out of genomic
  order. The reader refuses a sidecar whose chromosomes cannot slice its rows: offsets
  that do not number the chromosomes plus one, a chromosome named twice, or a label that
  is not UTF-8.
- **`ogdb rewrite-release` gains a `position_sidecar` step**, and its proof asks
  `range_phewas` at the first and last variant, either side of a chromosome boundary, over
  empty ranges, and on a position several variants share. A release whose rows cannot
  pose some of these (one chromosome, say) leaves them out. The log line names them, and
  the manifest's `provenance.format_rewrite.range_questions_not_posed` records them.
- **New validation rule `variant_table.position_sidecar`.** It checks the sidecar against
  the Variant Table and its own order. `envelope.required` and `envelope.closed` name its
  files from 0.3.0 only. `variant_table.representatives` still checks the tabix index,
  which 0.3.0 ships as a convenience that no query reads.
- Both repositories' docs change: opengwasdb-stores' release manifests and walkthrough
  name the envelope's files.
