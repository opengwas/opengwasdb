- **Format 0.3.0 carries each row's `source_alid` in a sidecar (issue #470, ADR 0105 §8).**
  `variant_source_alid_state.npy`
  gives each row a byte: 0, no `source_alid`; 1, its own ALID; 2, another ALID, which
  `variant_source_alid_utf8.npy` and `variant_source_alid_offsets.npy` hold for those
  rows alone, whole however long (an indel past 100 bytes). A release built on its own
  assembly so costs one byte a row: 21.3 MB on OGS-00003's 21.3 M rows, 5.8% of its
  table, where storing every row's text would have cost 489 MB, 1.34 times the table and
  the ALID sidecar's text again. `write_variant_axis` writes them, so every build and completion does,
  and a Reference-Completed release keeps each observed row's value and gives a panel row
  none. They are in the 0.3.0 envelope: a 0.3.0 release without them fails
  `envelope.required` and is refused at open (`MissingSourceAlidSidecar`), and a 0.2
  release carrying them fails `envelope.closed`. The writer refuses a `source_alid` the
  table could not hold as the sidecar does: `.`, or one holding a tab, newline or NUL
  (`UnencodableSourceAlidError`).
- **`variants_table()`, `variants()`, `variants_in_range()` and resolved rows with
  variant info gain a `source_alid` column** (the three reads are one projection, the
  axis's `rows_by_indices`),
  on every format: `None` in `variants_table()` and `.` in a resolved row where a row has
  none. A 0.3.0 release reads it from the sidecar, a 0.2 release from its table. The
  `opengwasdb query-* --variant-info` TSV output is unchanged.
- **On 0.3.0 the record methods read no Variant Table.** `VariantAxis.by_index`,
  `by_indices`, `all`, `records`, `range` and `by_identifier` name each row from the
  sidecars, so neither completion of a 0.3.0 source nor `validate --source`'s join
  (`indices_by_source_alid`) reads `variants.tsv.gz`. The join returns None, as for a
  table that predates the column, when no row has a `source_alid`, so `validate
  --source` lifts its samples itself. The validator reads the table through the new
  `VariantAxis.written_table_records`. A 0.2 release is read as before.
- **`ogdb build-dense` writes each row's own ALID as its `source_alid`.** It lifts
  nothing, and wrote none before, so its releases now say each row's source named it
  as stored, as every other build's do for an unlifted row.
- **New validation rule `variant_table.source_alid_sidecar`** (0.3.0): each row's state is
  the one the table's `source_alid` and ALID give, each lifted row's text is the table's,
  every state is 0-2, text is held for lifted rows alone, and it holds one row per row.
  `variant_table.source_alid` counts sourced rows from the sidecar on 0.3.0.
- **`ogdb rewrite-release` gains a `source_alid_sidecar` step**: it writes the sidecar
  from the table's `source_alid` column and ALIDs (state 0 on every row of a seven-column
  table) and refuses unless the sidecar read back gives every row the table's state and
  text. The proof
  compares `source_alid` through `variants_table()` and the resolved rows.
- opengwasdb-stores' release manifests and walkthrough name the envelope's files, so they
  change when opengwasdb-stores moves its pin to this version.
- **A 0.3.0 Hybrid carries the sidecar at its root only**: its Dense Component, which
  carries no Variant Axis (#472), answers each row's `source_alid` as its shared row's
  through `PanelVariantAxis`, and the rewrite writes the Dense Component none.
