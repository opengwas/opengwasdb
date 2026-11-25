- **The query facade answers variant-only reads (issue #484).** `variant_count()`,
  `variants(identifiers)` and `variants_in_range(chromosome, start, end)` return the
  variant count, and the `variants_table()` rows for rsIDs or ALIDs and for a range, on
  every layout and every readable format version, through the Variant Axis `lookup`
  and `range_phewas` use. They need no `variants_table()` over every variant, and on 0.3.0
  read no `variants.tsv.gz` and load neither pysam nor sqlite3. The rewrite proof digests
  them (`variant_reads`).
