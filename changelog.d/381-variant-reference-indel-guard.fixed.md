- **Variant References now refuse unmarked indels before routing or filtering
  a build (issue #381, ADR 0079).** Regenerated references declare reference-first
  semantics with `#opengwasdb-variant-reference<TAB>1` as their first line; both
  artifact writers emit it. Dense/Hybrid builds, Hybrid's legacy panel loader,
  the resolver and BESD's streamed lift check share the guard. Unmarked SNV-only
  references remain usable, including the resolver's existing header aliases,
  capitalization and whitespace-delimited tables. Detached Store Variant Tables
  containing indels
  require a regenerated declared reference too. No allele is guessed or swapped.
