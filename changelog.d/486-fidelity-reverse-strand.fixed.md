- **`fidelity.source` passes a correctly lifted Store on reverse-strand rows
  (issue #486).** A reverse-strand chain block complements a variant's
  alleles, which makes the other allele A1, so the build stores the source's z
  negated (issue #284). The check compared the source's z with the stored one
  without that projection, and failed every reverse-strand row of a lifted
  Dense Store whose values were right. It now lifts each sample through the
  build's own lift and `statistics_swap`, always for a cross-assembly source,
  and signs the source z as the Store does. A sample the Store holds by
  `source_alid` but the given chain places elsewhere or nowhere fails the
  check, naming the chain: its strand is unknown, and it is neither compared
  under a guessed one nor dropped. A cross-assembly check of a Store that
  carries `source_alid` now lifts its sample too, so it needs the chain the
  build used (`chain_file`), or downloads the default one as the fallback
  always did. Stored data is unchanged.
