- **A `--convert-legacy --lifted` conversion records `variant_table.lifted_contig` as a
  deviation when the release has rows on alt, random or unplaced contigs with no
  `source_alid` (issue #504).** `variant_table.lifted_contig` cannot judge such rows, and
  warned "N not judged (no source ALID)" with nothing recording why, so publishing the
  converted OGS-00009 (1,271 such rows) was refused. The conversion now records the rule
  and the count in `provenance.converted_from` (`lifted_contig_unjudged`).
  `manifest.converted_from` accepts it only where `lifted` is true,
  `variant_table.source_alid` is listed and the count is positive. `validate` then gives
  that warning the conversion's note. A recorded count that differs from the table's fails
  the rule. A row judged lifted onto an alt contig is never lowered, and every other
  release keeps the plain warning.
