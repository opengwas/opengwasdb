- **Every build now checks each stored z against its source's own p-value, first,
  and refuses the build naming every Analysis that contradicts it (issue #510, ADR
  0069).** OGS-00032 and OGS-00033 published eleven Analyses whose z was wrong by a
  scale: ten odds ratios stored as log-odds (median |Δ| 24.5–34 orders of
  magnitude of p) and one effect off by 10⁵ (1.2 × 10¹²). The check existed, but only
  in `resolve-analyses`, and their Resolution Records came from opengwasdb 0.3.3,
  which predates it. `build-hybrid`, `build-dense-vcf` and `build-ragged-ssf`
  now read each source's first 1,000,000 rows through the resolver's own
  projection and admission, at about 5 s per GWAS-SSF source. This is each
  build's first read of its sources, before it lifts or routes anything, so a
  refusal comes before hours of routing. `build-ragged-ssf` gains `--n-workers`
  for the check. The error says `effect_consistency=refused` and the rule that
  fired, so the registry can exclude by reason. A build manifest may declare
  `p_value_test` (spec §7a, ADR 0085): `joint_2df` skips the check, `mr_mega`
  refuses the Analysis unread. Without a declaration the Analysis is checked.
- **A GWAS-VCF's `LP` is now read as its p-value (`p = 10^-LP`), so GWAS-VCF
  Analyses are checked, in every build and in `resolve-analyses`.** Ten ukb-b
  VCFs pass with median |Δ| ≤ 0.0021. Only the metric blocks query `LP`, so the
  association path is unchanged. A source whose format carries no p-value (BESD,
  association TSV, a GWAS-VCF without `LP`, a Column Map without `p_value`) is
  recorded `skipped: no_source_p_value` without being read. A Resolution Record
  written before keeps its `no_source_p_value` until it is recomputed.
- **`analyses.tsv` gains `effect_consistency`, `effect_consistency_reason`,
  `effect_consistency_median_abs_delta`, `effect_consistency_median_delta` and
  `effect_consistency_n` (spec §7a, §9.2).** Every build writes all five on every
  row: `passed` with the measurement, or `skipped` with the reason.
  `effect_consistency.evidence` fails a release with some of the columns, a blank
  outcome under them, a recorded `refused`, and evidence that contradicts itself.
  `effect_consistency.unchecked` warns on a release with none of them, which is
  every release built before this change. A completion or `--convert-legacy`
  rewrite of such a release keeps the columns absent. `format_version` does not
  move.
- **New validation rule `associations.z_plausible` audits a genome-wide Store's
  stored z.** It flags an Analysis with more than 30% of its observed
  associations at 5e-8, summed over a Hybrid's two components, on releases
  declaring `association_coverage` `gwas`.
  It is an error from `format_version` 0.2.2 and a warning before that. The
  twelve mis-scaled Analyses are at 81% or more, sound ones at 9.7% or less.
  Across every published Store it flags only GCST002047 in OGS-00005, ADR
  0069's own example, so `validate` fails that release until it is rebuilt
  without the Analysis.
