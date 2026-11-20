# Declare the test a source's p-value comes from

Issue [#371](https://github.com/opengwas/opengwasdb/issues/371), under epic #347
("Builds never store a statistic that contradicts its source"). It extends
[ADR 0069](0069-refuse-an-effect-that-contradicts-its-sources-p-value.md),
the resolver's check of beta/se against the source's own p-value. The owner
ruled on 20 Nov 2026, in the #347 orchestration session:
- a declared column, not a threshold ("fine");
- the MR-MEGA Analyses are dropped ("drop");
- the Analyses whose low-frequency rows go wrong get a MAF threshold ("maf
  threshold").

No format change: the declaration is a resolver input and is not stored.

## Context

A Store keeps z (= beta/se) and SE. A query answers with the two-sided normal
p of that z (spec §6). The source's p is read once, by ADR 0069's check, and
the check judges the median Δ over a 20,000-row sample. Wherever a source's p
is not the Wald p of its beta/se, the Store answers with a p the source never
published. [statistic-disagreement.md](../statistic-disagreement.md) lists every
such circumstance.

#371 asked for an Analysis-level refusal on the share of stored hits
(|z| ≥ 5.45) whose source p is above 1e-4. It measured that share on all 4,783
sources of OGS-00011, read whole: 49.8 billion rows. There is no gap to put a
threshold in.
- The eight MR-MEGA Analyses span 13.5–99.9%.
- Four Analyses that are not MR-MEGA sit inside that range, at 78.2%, 31.7%,
  24.0% and 15.3%. Their false hits lie in particular rows (low frequency, or an
  SE printed to two decimals), not throughout the Analysis.
- The 33 joint 2-df Analyses have no false hits at all, but up to 4,477 lost
  hits each. Their stored interaction coefficient is valid. Their p tests it
  together with a main effect.
- Median |Δ| does not separate the classes either: MR-MEGA sits at
  0.232–0.248, joint tests at 0.165–0.189, and GCST90292553, with no false
  hits, sits above both at 0.343.

What test a p-value comes from is a fact about how the Analysis was run. No
statistic of the file recovers it.

## Decision

**Every row of a resolver manifest declares `p_value_test`, with no default.**
The Source Collection fills it. A manifest without the column, or a row whose
value is blank or unknown, is refused when it is read, naming the Analysis.

| `p_value_test` | The source's p is | The resolver |
|---|---|---|
| `wald_1df` | the Wald p of beta/se | runs ADR 0069's check, as before |
| `joint_2df` | a joint test of the stored coefficient and a main effect | reads the source and skips the check: `skip_reason = joint_test_p_value`. The Analysis resolves |
| `mr_mega` | MR-MEGA's 2-df chi-square, beside an SE scaled by the residual mean square, so beta/se is a t on (cohorts − 2) df | refuses the Analysis before reading it. The record's `status` is `controlled_failure`, `error` begins `p_value_test is mr_mega`, and `diagnostics.effect_consistency` is `null` |

The declaration is bound into the fingerprint (`resolution_config.p_value_test`),
so a record resolved under one declaration is never resumed under another.

The rows whose low-frequency associations go wrong are a matter of admission
for each Analysis, not of this check. The Source Collection raises each such
Analysis's `maf_threshold` (ADR 0054). The values come from #371's measurement
and live in `opengwasdb-stores`.

## Consequences

- **Breaking for every resolver manifest.** A manifest without `p_value_test`
  no longer resolves. `opengwasdb-stores`' generators write it, seeded from each
  Analysis's label and `analysis_software`, with a reviewed list for the
  Analyses neither names.
- **Every existing Resolution Record has a different fingerprint digest.**
  `--resume` recomputes it, as it did for ADR 0069.
- **The declaration is trusted, and nothing checks it.** An MR-MEGA or joint-test
  source declared `wald_1df` resolves and is stored as it was before #371
  (`test_a_source_declared_a_wald_test_when_it_is_not_is_stored`). A source
  declared `joint_2df` is never compared, so a defect the check would have
  refused, such as an odds ratio in `beta`, passes too. #371 measured a
  cross-check (median |Δ| above 0.16 with a signed median below −0.03, over the
  first million rows). It caught all 41, but also two sources it could not
  explain (GCST90292553 and GCST006979). It is not adopted. A misdeclaration is
  the Source Collection's to review.
- **A joint-test Analysis's query p is still the 1-df p of its interaction z.**
  The declaration is not stored, so a Store does not tell a reader which of its
  Analyses those are. Loci significant only through their main effect are lost
  hits in it.
- The two Analyses whose p comes from another model of the same data
  (GCST90002238 and GCST90002227) put their false hits at low frequency too,
  so they take a MAF threshold as well. Nothing here addresses an SE printed
  with too few digits (GCST90244669). It remains stored, as the catalogue
  records.
