- **Format 0.3.0 carries each row's `source_alid` in a sidecar (issue #470, ADR 0105 §8).**
  Not released until #472 lands on epic #450's branch. `variant_source_alid_utf8.npy` and
  `variant_source_alid_offsets.npy` hold each row's pre-liftover ALID, UTF-8 with row
  offsets, in row order; an empty value is none, and a long one (an indel past 100 bytes)
  is stored whole. `write_variant_axis` writes them, so every build and completion does,
  and a Reference-Completed release keeps each observed row's value and gives a panel row
  none. They are in the 0.3.0 envelope: a 0.3.0 release without them fails
  `envelope.required` and is refused at open (`MissingSourceAlidSidecar`), and a 0.2
  release carrying them fails `envelope.closed`. The writer refuses a `source_alid` the
  table could not hold as the sidecar does: `.`, or one holding a tab, newline or NUL
  (`UnencodableSourceAlidError`).
- **`variants_table()` and resolved rows with variant info gain a `source_alid` column**,
  on every format: `None` in `variants_table()` and `.` in a resolved row where a row has
  none. A 0.3.0 release reads it from the sidecar, a 0.2 release from its table. The
  `opengwasdb query-* --variant-info` TSV output is unchanged.
- **On 0.3.0 the record methods read no Variant Table.** `VariantAxis.by_index`,
  `by_indices`, `all`, `records`, `range` and `by_identifier` name each row from the
  sidecars, so neither completion of a 0.3.0 source nor `validate --source`'s join
  (`indices_by_source_alid`, which scans the sidecar) reads `variants.tsv.gz`. The
  validator reads the table through the new `VariantAxis.table_records`. A 0.2 release
  is read as before.
- **New validation rule `variant_table.source_alid_sidecar`** (0.3.0): the sidecar names
  each row's `source_alid` as the table does, and holds one row per row.
  `variant_table.source_alid` counts sourced rows from the sidecar on 0.3.0.
- **`ogdb rewrite-release` gains a `source_alid_sidecar` step**: it writes the sidecar
  from the table's `source_alid` column (empty on every row of a seven-column table) and
  refuses unless the sidecar read back names every row's as the table does. The proof
  compares `source_alid` through `variants_table()` and the resolved rows.
- opengwasdb-stores' release manifests and walkthrough name the envelope's files, so they
  change when opengwasdb-stores moves its pin to this version.
