- **`opengwasdb clump STORE --ld-store ANCESTRY=PATH …` writes the Clumps of a 0.3.0
  Store Release's 5e-8 tier, and `n_loci_5e8` and `n_unclumped_5e8` beside `n_hits_5e8`
  (ADR 0106, spec §18.1, issue #418).** Each Analysis's observed rows of the tier are
  clumped against the LD Store its `assigned_ancestry` maps to, at r² = that store's t_far
  and kb 10,000, by |z| with ties by Store Variant Index; a reference-first indel is looked
  up by its sorted spelling, its r re-signed to the Store's A1. The step writes, in place,
  `top_hits/p_5e_08/clumps/default/` (`lead`, `r`) in every component holding the tier,
  the two `analyses.tsv` columns, `overview.html`, and the manifest's `clumps` entry last.
  An absent, imputed, ambiguous or unmapped row is marked, never dropped; an Analysis with
  no mapped LD Store gets blank counts, never 0. It refuses, before writing anything, a
  Store of any format but 0.3.0, one with no 5e-8 tier, a Reference-Completed one whose
  tier has no `imputed`, and an LD Store of another assembly or one that refuses the
  configuration. The engine holds no default LD Store: the mapping is the caller's. The
  `format_version` does not move. `validate` gains the seven `clumps.*` rules (spec
  §20.0), and `--ld-store ANCESTRY=PATH`, which lets `clumps.reclump` re-clump a seeded
  sample; without it that rule is recorded as not run. `analyses.tsv` gains two columns
  wherever the step runs: a change for opengwas/opengwasdb-stores too.
