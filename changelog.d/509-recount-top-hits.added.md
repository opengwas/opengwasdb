- **`opengwasdb recount-top-hits STORE` sets a release's Top-Hit Counts from its stored
  tiers, in place (issue #509, ADR 0107).** It recounts a 0.2 Hybrid's Dense Component's
  `analyses.tsv` first, then the root's, and refuses, writing nothing, a release it cannot
  recount. It matches each row by `analysis_index`, as the rule does, and keeps the table's
  row order. It writes only a table whose counts change, as `clump` does, then regenerates
  any `overview.html` that no longer matches its table, so a re-run completes a killed one.
  It prints what changed as JSON. Every writer of `overview.html` now writes it atomically. On a copy of OGS-00011 it changed 3,047 of 3,245
  Analyses in 1.14 s. Every other column, `n_loci_5e8` and `n_unclumped_5e8` included,
  is left byte-identical. The owner approved it for published Stores on 26 Nov 2026.
