- **A row no source observed records no `source_alid` (issue #261).** A
  single-pass build given a plain ALID list or a Store Variant Table as
  `--variant-reference`, with GRCh38 sources, took the reference as the source of
  every ALID it listed and wrote each unobserved row's own ALID as its
  `source_alid`, as if a source had supplied it; `validate` passed it. Such a row
  is now `.`, as the two-pass `--reference-panel` build writes it, and the two
  builds write the same variant table. An observed row's `source_alid` comes from
  the source sites the routing placed on it. An extraction artifact's recorded
  source keys are kept. `build-dense-vcf` and `build-hybrid` refuse to write a
  `source_alid` no source supplied, checked against the coordinates recorded from
  the sites the build read rather than against its own record of what was
  observed (Hybrid checks the map after the fold); a plain reference whose routing
  records no observation is refused. To know which rows were observed,
  the build now reads its sources (Pass 1) even when the reference names its own
  rsids. OGS-00005's 8 Oct 2026 rebuild holds 600,977 Dense Component rows whose
  `source_alid` is their own ALID for this reason; its next rebuild writes `.`.
