# The Store Reader derives beta and pval, once

## Status

Accepted. Issue [#466](https://github.com/opengwas/opengwasdb/issues/466), part of epic
[#450](https://github.com/opengwas/opengwasdb/issues/450) (the Store Reader, `ogquery`). It revises
[ADR 0020](0020-sparse-array-query-contract.md)'s "callers derive", and it adds to
[ADR 0068](0068-an-se-beyond-float16-is-missing.md) what a query derives from an infinite SE. ADRs
0020, 0068 and [0070](0070-one-presence-rule-for-every-reader.md) otherwise stand.

**Who decided.** The issue names the three questions below. The answers are the #450
orchestration's rulings for #466 (23 Nov 2026), given under the owner's plan of record for the epic
([comment](https://github.com/opengwas/opengwasdb/issues/450#issuecomment-7103286577)). The owner
has not ruled on them separately.

## Context

A Store Release stores z and SE. Its beta is `z * se`, and its two-sided p is
`erfc(|z| / sqrt(2))` (spec §6). ADR 0020 left both derivations to callers: "Callers who need derived
fields (`beta`, `p_value`) compute them from `z` and `se`". Each caller then did it its own way:

- the CLI printed p as `10 ** log10_p`;
- five benchmarks computed beta as `z * se` from lookup results;
- two benchmarks computed -log10 p with scipy's `log_ndtr`;
- ogquery's clients (opengwaspy, opengwasr) would each have written a fourth version.

Two edge cases give different answers in different hands:

- **Underflow.** p goes subnormal at |z| = 37.5 and is 0.0 in float64 from |z| = 38.5. The
  FADS1/FADS2 window reaches |z| = 47.8 (issue #104).
- **An infinite SE.** A release before format 0.2.3 can hold `+inf` beside a finite z (ADR 0068).
  `z * inf` is `±inf`, or NaN when z is 0. Read from z alone, p looks like an ordinary value.

The Store Reader is now the one caller every consumer goes through (#464), so the derivation can
live in it.

## Decisions

1. **Where: on every association record.** `opengwasdb.store_reader.stats.association_statistics(z,
   se)` is the one derivation. It returns `beta`, `pval` and `log10_p` from a result's z and SE
   arrays. The `Associations` record that every query method returns (Dense, Ragged and Hybrid;
   `analysis`, `phewas`, `range_phewas`, `range_by_analysis`, `lookup` and `top_hits`) exposes
   `beta` and `pval` as mapping keys. They are derived from its own `z` and `se` the first time
   either is read. `resolve_rows` carries `beta`, `pval` and `log10_p` on every resolved row, from
   the same function.
2. **Underflow: `pval` is the float64 value, and `log10_p` is the exact one.** Where p is below
   float64's range, `pval` is subnormal (|z| 37.5 to 38.5) or 0.0 (beyond). `log10_p` stays finite
   and exact: at z = 47.8 (float32) it is -497.92. Rank and display by `log10_p`.
   - The CLI's TSV `p` column prints `pval` to three significant figures, and prints `<1e-300`
     where `log10_p` is below -300.
   - `query --format json` prints `log10_p` beside `pval`, so a JSON consumer can recover the
     exact value where `pval` is 0.0.
3. **A missing association, or an infinite SE, is NaN.** Where z or SE is NaN, `beta`, `pval` and
   `log10_p` are NaN. Where SE is infinite they are also NaN, never infinite. ADR 0068 rules that an
   SE above `float16`'s range is an error, which a build now treats as missing. A release before
   0.2.3 still holds such a cell, and its z and SE are still read as values (ADR 0068 decision 6,
   ADR 0070). What is derived from them is not a value: `z * inf` would read as an infinitely
   large effect, and a p from z alone would read as an ordinary result. An infinite z beside a
   finite SE keeps its infinities: beta `±inf`, `pval` 0.0, `log10_p` `-inf`.
4. **Type: float64**, whatever the planes' stored dtype (`float16`, `float32` or a decoded
   fixed-point code). `pval` is computed as `erfc(|z| / sqrt(2))` in float64, the same expression
   as the scalar `p_value_from_z`, bit for bit. `log10_p` is computed exactly as
   `log10_p_two_sided` computes it. Each distinct |z| is evaluated once.

Nothing else in `opengwasdb/` derives beta or p from a stored z and SE.
`tests/test_store_reader_beta_pval.py` scans the package and fails if it finds a second
derivation. Three source-side readers compute `z * se` from a source's own columns before
anything is stored, and the scan allows those by name. The build-time derivations
(`top_hits.py`'s stored `p_value` and `build/effect_consistency.py`) are a different thing: each
computes from the values a build is writing, not from a query result.

## Consequences

- **Every result carries two more arrays**, `beta` and `pval`, both float64. Every array a result
  held before is unchanged. The query digests on the six fixture releases
  (`tests/golden/store_reader_digests.json`) are identical to the pins written at `5d600a1d` once
  `beta` and `pval` are dropped. The `log10_p` values agree with those pins to 1e-12, the tolerance
  the pins already used.
- **A large result costs one sort of its |z| more**, the first time `beta` or `pval` is read
  (below). A caller that reads only z and SE pays nothing, because the record
  derives the two arrays lazily. Measured on 2M `float32` z with 11,045 distinct |z|:
  `association_statistics` takes 0.17 s, and `log10_p_two_sided` alone 0.13 s.
- **CLI output gains columns.**
  - TSV: `beta`, after `se` and before `p`.
  - JSON: `beta`, `pval` and `log10_p`.
  - A positional reader of the TSV sees `p`, `eaf` and `association_status` move one column right.
    opengwasdb-stores documents these columns, so the change is a documentation change there too.
- **On a release before 0.2.3 that holds `+inf` SE cells**, the TSV `p` column prints `NA` for
  them, where it used to print the p of their z. OGS-00011 holds 8 such cells (ADR 0068).
- **Benchmarks read the Store Reader's values.**
  - The per-instrument `beta_exp`/`beta_out` come from the record.
  - `_hybrid_report.neglog10_p` and `validate_finngen_source_fidelity.py`'s store-side -log10 p
    come from `log10_p`.
  - The beta values are the same float64 products as before. The -log10 p values agree with the
    scipy `log_ndtr` form to 1e-12.
  - Each benchmark's committed output still comes from its last run.
  - The IVW p-values are statistics of an estimator, not of a stored association, so they are
    unchanged.

## Rejected

- **Only on resolved rows.** The index-keyed record is what opengwaspy and opengwasr read through
  ogquery, and what the benchmarks read. A beta only on resolved rows would leave them all
  deriving their own.
- **`pval` as `10 ** log10_p`.** It is one derivation fewer, but it loses precision
  near p = 1e-300: measured, `10 ** log10_p` is up to 2.6e-13 relative from
  `math.erfc` for 36 <= |z| <= 37.4. It would also put a second spelling of p in the code.
- **`pval` as the clamped smallest subnormal, or a string, where p underflows.** A clamped value
  is a wrong answer that looks right. A string makes the column unusable as numbers. `log10_p` is
  already the exact value, so `pval` only has to be honest that it underflowed, and 0.0 is.
- **beta `±inf` where SE is infinite**, as `z * se` gives. ADR 0068 rules such an SE an error, so
  an infinite effect would be a fabricated value, not a measured one.
- **`log10_p` on the record as a third mapping key.** The record's keys are the query contract
  (spec §19), and the issue asks for `beta` and `pval`. `log10_p` is already on resolved rows and
  in the JSON output, and it is one call away (`association_statistics`) for a caller that holds a
  record.
