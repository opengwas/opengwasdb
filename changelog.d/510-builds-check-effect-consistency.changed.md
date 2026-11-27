- **Every build now checks each stored z against its source's own p-value, and
  refuses the build naming every Analysis that contradicts it (issue #510, ADR
  0069).** OGS-00032 and OGS-00033 published eleven Analyses whose z was wrong by a
  scale: ten odds ratios stored as log-odds (median |Δ| 24.5–34 orders of
  magnitude of p) and one effect off by 10⁵ (1.2 × 10¹²). The check existed, but only
  in `resolve-analyses`, and their Resolution Records came from opengwasdb 0.3.3,
  which predates it. `build-hybrid`, `build-dense-vcf` and `build-ragged-ssf`
  now read each source's first 1,000,000 rows through the resolver's own
  projection and admission, at about 5 s per GWAS-SSF source. The error says
  `effect_consistency=refused` and the rule that fired, so the registry can
  exclude by reason. A build manifest may declare `p_value_test=joint_2df` to
  skip an Analysis whose p is a joint test. Without that declaration the
  Analysis is checked.
- **`analyses.tsv` gains `effect_consistency`, `effect_consistency_reason`,
  `effect_consistency_median_abs_delta`, `effect_consistency_median_delta` and
  `effect_consistency_n` (spec §7a, §9.2).** Every build stamps them: `passed`
  with the measurement, or `skipped` with the reason. GWAS-VCF, BESD and
  association-TSV sources carry no p-value, so they record
  `skipped: no_source_p_value`, never a blank. `effect_consistency.evidence`
  fails a recorded `refused` and evidence that contradicts itself.
  `effect_consistency.unchecked` warns on a blank outcome, which every release
  built before this change has. `format_version` does not move.
- **New validation rule `associations.z_plausible` audits a genome-wide Store's
  stored z.** It flags an Analysis with more than 30% of its observed
  associations at 5e-8, summed over a Hybrid's two components, on releases
  declaring `association_coverage` `gwas`.
  It is an error from `format_version` 0.2.2 and a warning before that. The
  twelve mis-scaled Analyses are at 81% or more, sound ones at 9.7% or less.
  Across every published Store it flags only GCST002047 in OGS-00005, ADR
  0069's own example, so `validate` fails that release until it is rebuilt
  without the Analysis.
