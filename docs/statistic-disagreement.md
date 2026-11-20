# When a source's p-value and its beta/se disagree

A Store keeps two statistics per association: z (= beta / se) and SE
([spec §6](spec/store-format.md#6-stored-statistics)). It never keeps the
source's p-value. **The p a query returns is the two-sided normal p of the stored
z** (`opengwasdb/query/resolve.py`, `log10_p_two_sided`). That is exact for a
Wald test, which is what most sources report. Wherever a source's p is *not* the
Wald p of its own beta/se, the Store answers with a p the source never published.

The source's p is read once, at resolve time, as a witness against beta/se:
the effect-consistency check ([its contract](spec/resolver-effect-consistency.md),
ADR 0069). That check compares the two over a 20,000-row sample and refuses an
Analysis on either of two medians:
- the median |Δ| is above 1.0 (#345);
- the median signed Δ is above +0.1 (#370).

Here Δ = log10 p implied by z − log10 p reported. A median sees an Analysis that
is wrong throughout. It cannot see:
- a fault confined to some rows;
- a p that is legitimately from a different test, but close.

This page lists every circumstance measured so far, on the 4,783 sources of
OGS-00011 read whole (#357, #371; evidence in the issue threads). For each it
gives which number is right and what the engine does today. Each row names the
test that pins that behaviour:
- `catalogue: <name>` is a case in `tests/test_statistic_disagreement_catalogue.py`, built
  from the circumstance's mechanism;
- other test names are in `tests/test_resolve_effect_consistency.py`.

A test that pins "stored" is deliberate. It records a gap, and it changes when
the gap is closed.

**Terms.**
- A *false hit* has stored |z| ≥ 5.45 (p 5e-8), where the source's p is above 1e-4.
- A *lost hit* has stored |z| < 3.72, where the source's p is at most 5e-8.
- The shares and medians below are whole-file unless they are marked otherwise.

## The source is wrong, and the check refuses it

| Circumstance | Which is right | Measured | Today | Pinned by |
|---|---|---|---|---|
| The effect column is on the wrong scale: odds ratios in `beta`, or log-odds in `odds_ratio` | source p | GCST002047 median \|Δ\| 1,710; GCST004461 230; GCST90503316 878; GCST90165257 11.9 | refused, `median_abs_delta` | `test_a_beta_column_holding_odds_ratios_is_refused`, `test_an_odds_ratio_column_holding_log_odds_is_refused` |
| beta/se carries no signal: SPACox's beta 9 beside SE 1000; an SE column holding exp(SE) | source p | GCST90275159–66 and GCST90572740/1: signed median +0.279 to +0.337 | refused, `median_delta` | `test_a_deflated_z_is_refused_by_the_signed_rule_though_its_abs_median_is_under_one` |
| The p column holds the upper-tail one-sided p of the source's z: p/2 where z > 0, 1 − p/2 where z < 0 | beta/se | GCST90446466: signed median +0.252 (+0.239 on the resolver’s sample) | refused, `median_delta`. The message says the stored z is deflated; it is the p that is wrong. Read through a Column Map that omits `p_value`, it resolves with the check skipped (`no_source_p_value`) | `catalogue: p_is_one_sided`, `test_a_source_whose_map_omits_p_value_is_skipped_not_refused` |

## The source's p is not a Wald p of the stored coefficient, and the check passes it

| Circumstance | Which is right | Measured | Today | Pinned by |
|---|---|---|---|---|
| **MR-MEGA.** The SE is scaled by the residual mean square RSS/NDF, so beta/se is a t on (cohorts − 2) df. p is a separate 2-df chi-square | neither: the stored z is not a z | 8 Analyses (GCST90833021, GCST90444480, GCST90014299–304): median \|Δ\| 0.232–0.248; 13.5–99.9% of hits false. GCST90833021 returns z −1054 at source p 0.81 | **stored**, with an inflated z (#371) | `catalogue: mr_mega` |
| **Joint 2-df test** ("(2df test)", "(2df)"): the stored coefficient is the interaction's, and p tests interaction and main effect together | both, but they are different tests | 33 Analyses: median \|Δ\| 0.165–0.189, 0 false hits, up to 4,477 lost hits each | **stored**. The query returns the interaction's 1-df p. Loci significant through their main effect are lost | `catalogue: joint_two_df`, `test_a_store_returns_the_p_of_its_stored_z_never_the_sources` |
| **p from another model of the same data**: beta/se from the raw trait, p from the inverse-normal-transformed trait | both | GCST90002238: median \|Δ\| 0.113, 31.7% of hits false, all at MAF 0.5–1% (z 27.6 at p 0.19). GCST90002227: 12.8% | **stored** | `catalogue: p_from_another_model` (the median only; the rare-variant rows are the next row's mechanism) |
| **Low-frequency rows in a small case-control study**: the Wald SE is too small beside the score-test p | source p, for those rows | GCST90572747 (N 6,355): 78.2% of hits false, 343 of 359 at MAF 1–5%. GCST90243957 (SNPTEST, N 10,771): 24.0%. GCST90243956: 7.5%. All have median \|Δ\| ≤ 0.013 | **stored**: a fault in rows, invisible to a median. A per-Analysis MAF threshold (ADR 0054) removes those rows | `catalogue: low_frequency_rows_inflated` |
| **SE printed with too few digits** (`0.01`) | source p | GCST90244669 (BOLT-LMM): 15.3% of hits false, all just above \|z\| 5.45 | **stored** | `catalogue: se_printed_to_two_decimals` |
| **Sample-size-weighted meta-analysis**: METAL's N-weighted p beside inverse-variance beta/se | both | GCST004732: median \|Δ\| 0.158, 0 false hits | stored. The difference is within what either weighting allows | `catalogue: sample_size_weighted_z` |
| **p at a floor**: written no smaller than 1e-300, or a double's limit | z | many sources, per row | stored, correctly. The check works in log space | `catalogue: p_at_a_floor`, `test_implied_p_survives_a_z_where_the_naive_form_underflows` |

## No witness

| Circumstance | Today | Pinned by |
|---|---|---|
| The source has no p-value: GWAS-VCF, BESD, or a Column Map without `p_value` | skipped, `no_source_p_value`; nothing is compared | `test_a_source_with_no_p_value_column_is_skipped_with_a_reason` |
| Every SE was derived from the p-value (ADR 0055) | skipped, `all_se_from_p_value`. Such rows are never compared, because they would agree by construction | `test_a_source_whose_every_se_comes_from_its_p_value_is_skipped` |

## Not yet explained

- **GCST90292553** (the CAMP method, Ojavee et al. 2023): median |Δ| 0.343, signed −0.343, and 0 false hits among 30,571 hits.
- **GCST006979:** carries a second p column; median |Δ| 0.189, and 106 false hits among 131,534.

Both are stored.

## What closes the gaps

- **#371** has the measurement behind the second table.
  - No threshold on the share of false hits separates MR-MEGA from the rest. MR-MEGA spans 13.5–99.9%, and four Analyses that are not MR-MEGA sit inside that range.
  - The measurement proposes a declared manifest column, with no default, naming the test a source's p comes from. The resolver would then refuse `mr_mega` and skip the check for a joint test.
- **Rows that go wrong at low frequency** are a matter of admission per Analysis (a MAF threshold), not of this check.
