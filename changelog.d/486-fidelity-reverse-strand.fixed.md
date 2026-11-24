- **`fidelity.source` passes a correctly lifted Store on reverse-strand rows,
  and lifts only with the chain the build used (issue #486).** A
  reverse-strand chain block complements a variant's alleles, which makes the
  other allele A1, so the build stores the source's z negated (issue #284).
  The check compared the source's z with the stored one without that
  projection, and failed every reverse-strand row of a lifted Dense Store whose
  values were right. It now lifts each sample of a cross-assembly source
  through the build's own lift and `statistics_swap`, whatever the
  `source_alid` join finds, and signs the source z as the Store does. The
  chain is the one the release records in `provenance.chain_file` (and
  `chain_sha256`, where recorded): a `chain_file` with another label or other
  bytes is refused, and with none given the recorded chain is used if it is
  pyliftover's built-in one or is on disk at the recorded path; otherwise the
  check is refused, naming the recorded chain. A sample the Store holds by
  `source_alid` that the chain places nowhere, or on another row, fails the
  check, naming the chain. Stored data is unchanged.
