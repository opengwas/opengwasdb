- **`complete-hybrid` recounts `n_hits_5e8`, `n_hits_5e6` and `n_hits_5e4` from the Dense
  Component's final tiers after its panel-crossover fold (issue #509).** It counted the tiers
  it built before the fold, then rebuilt them without recounting, so a completed Hybrid's
  counts disagreed with its tiers: OGS-00011 had 1,028 of 3,245 Analyses wrong at 5e-8, which
  failed Clumps' `n_loci_5e8 + n_unclumped_5e8 ≤ n_hits_5e8` on 26 of them. A new rule,
  `analyses.top_hit_counts`, holds every `analyses.tsv`'s counts to its tiers (a Hybrid
  root's to both components'): an error from format 0.2.2, a warning before. Of the 21
  published Stores, only OGS-00011 fails it.
