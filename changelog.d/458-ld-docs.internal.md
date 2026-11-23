- **The LD engine's ADRs, specs and glossary now speak as opengwasdb's
  (issue #458, epic #453).** ADRs 0087–0101 each say which ogld ADR they were; every
  `ADR 00NN`, `spec §N` and `build inputs §N` in `opengwasdb/ld/`, `tests/ld/`,
  `scripts/ld/` and the LD docs names the new ADR number or spec file; and
  `CONTEXT.md` gains an *LD Stores* section. `Variant Identity` now says what the
  code does (no left alignment; indels reference-first in GWAS Stores, sorted in LD
  Stores until #462), `ALID Search Index` is a defined term, and
  `CONTRIBUTING.md` documents the LD test tiers. Prose that named the `ogld`
  package, CLI or extras names `opengwasdb.ld`, `opengwasdb ld` and `opengwasdb[ld-build]`.
