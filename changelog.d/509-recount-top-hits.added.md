- **`opengwasdb recount-top-hits STORE` sets a release's Top-Hit Counts from its stored
  tiers, in place (issue #509, ADR 0107).** It recounts a 0.2 Hybrid's Dense Component's
  `analyses.tsv` first, then the root's, and refuses, writing nothing, a release it cannot
  recount. It writes only a table whose counts change, as `clump` does, then regenerates its
  overview, and prints what changed as JSON. On a copy of OGS-00011 it changed 3,047 of 3,245
  Analyses in 1.8 s. Running it on a published Store awaits the owner's sign-off.
