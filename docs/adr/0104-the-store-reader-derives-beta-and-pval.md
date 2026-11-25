# The Store Reader derives beta and pval, once

## Status

Accepted. Issue [#466](https://github.com/opengwas/opengwasdb/issues/466), part of epic
[#450](https://github.com/opengwas/opengwasdb/issues/450) (the Store Reader, `ogquery`).

What this ADR changes in other documents:

- [ADR 0020](0020-sparse-array-query-contract.md): it revises "callers derive", and ADR 0020 carries
  a dated banner saying so.
- [ADR 0068](0068-an-se-beyond-float16-is-missing.md): it adds what a query derives from an
  infinite SE, and decision 6 there carries a dated note.
- Spec [§19](../spec/store-format.md#19-query-result-contract) (the query result contract): rewritten
  to list `beta` and `pval` among the record's keys, give the derivation table, and describe the
  JSON and TSV output.
- Spec §20: the `se_infinite` warning line now says a query reads `beta` and `pval` as NaN there,
  instead of "`beta = z × se` reads as infinity".

ADRs 0020, 0068 and [0070](0070-one-presence-rule-for-every-reader.md) otherwise stand.

**Who decided.** The issue names the three questions below. The answers are the #450
orchestration's rulings for #466 (23 Nov 2026), given under the owner's plan of record for the epic
([comment](https://github.com/opengwas/opengwasdb/issues/450#issuecomment-7103286577)). The ruling on
an infinite SE follows the owner's own ruling of 1 Nov 2026 (ADR 0068). Beyond that, the owner has
not ruled on these answers separately.

## Context

A Store Release stores z and SE. Its beta is `z * se`, and its two-sided p is
`erfc(|z| / sqrt(2))` (spec §6). ADR 0020 left both derivations to callers: "Callers who need
derived fields (`beta`, `p_value`) compute them from `z` and `se`". Each caller then did it its own
way:

- the CLI printed p as `10 ** log10_p`;
- six benchmarks computed beta as `z * se` from lookup results;
- two benchmarks computed -log10 p with scipy's `log_ndtr`;
- ogquery's clients (opengwaspy, opengwasr) would each have written yet another version.

Two edge cases give different answers in different hands:

- **Underflow.** p goes subnormal at |z| = 37.5 and is 0.0 in float64 from |z| = 38.5. The
  FADS1/FADS2 window reaches |z| = 47.8 (issue #104).
- **An infinite SE.** A release before format 0.2.3 can hold `+inf` beside a finite z (ADR 0068).
  `z * inf` is `±inf`, or NaN when z is 0. Read from z alone, p looks like an ordinary value.

The Store Reader is now the one caller every consumer goes through (#464), so the derivation can
live in it.

## Decision

1. **Where: on every association record.**
   - `opengwasdb.store_reader.stats.association_statistics(z, se)` is the one derivation. It
     returns `beta`, `pval` and `log10_p` from a result's z and SE arrays.
   - The `Associations` record exposes `beta` and `pval` as mapping keys. Every query method returns
     that record: Dense, Ragged and Hybrid; `analysis`, `phewas`, `range_phewas`,
     `range_by_analysis`, `lookup` and `top_hits`.
   - The record derives all three the first time `beta` or `pval` is read, and keeps them
     (`Associations.statistics`). Asking whether a key is present derives nothing.
   - `resolve_rows` and the CLI's JSON output read the record's values through
     `associations.statistics_of`. They do not derive them again.
2. **Underflow: `pval` is the float64 value, and `log10_p` is the exact one.**
   - Where p is below float64's range, `pval` is subnormal (|z| 37.5 to 38.5) or 0.0 (beyond).
   - `log10_p` stays finite and exact: at z = 47.8 (float32) it is -497.92.
   - Rank and display by `log10_p`.
   - The CLI's TSV `p` column prints `pval` to three significant figures, or `<1e-300` where
     `log10_p` is below -300.
   - `query --format json` prints `log10_p` beside `pval`, so a JSON consumer can recover the exact
     value where `pval` is 0.0.
3. **A missing association, or an infinite SE, is NaN.**
   - Where z or SE is NaN, `beta`, `pval` and `log10_p` are NaN.
   - Where SE is infinite they are also NaN, never infinite. This follows the owner's ruling of
     1 Nov 2026 (ADR 0068): an SE that `float16` cannot hold is almost certainly an error, so the
     association is treated as missing. A build since 0.2.3 stores no such SE.
   - A release before 0.2.3 still holds such cells, and their z and SE are still read as values
     (ADR 0068 decision 6, ADR 0070). What would be derived from them is not a value: `z * inf`
     would read as an infinitely large effect, and a p from z alone would read as an ordinary
     result.
   - An infinite z beside a finite SE keeps its infinities: beta `±inf`, `pval` 0.0, `log10_p`
     `-inf`.
   - `log10_p_two_sided(z)`, which `opengwasdb.query` still exports, is a function of z alone. It
     stays finite where SE is infinite. A resolved row's `log10_p` is `association_statistics`',
     and is NaN there.
4. **Type: float64**, whatever the planes' stored dtype (`float16`, `float32` or a decoded
   fixed-point code).
   - `pval` is `p_value_from_z`'s value, bit for bit.
   - `log10_p` is `log10_p_two_sided`'s, which `association_statistics` calls.
   - Each is evaluated once per distinct |z|.

### The guard

`tests/test_derivation_guard.py` parses every module of `opengwasdb/` and `benchmarks/`. It fails
on any of these:

- a `*` chain with a z-like and an se-like factor: a name, attribute or string subscript such as
  `z`, `rec.z`, `result["se"]` or `d["se_exp"]`;
- `np.multiply` of the same;
- a call of `erfc`, `p_value_from_z`, `norm.sf`, `logsf` or `log_ndtr`;
- `10 **` or `np.power(10, …)` of a `log10_p`.

It exempts `store_reader/stats.py`, where the derivation lives. Every other site is allowed
explicitly, by module and enclosing function, with its reason, and an entry that matches nothing
fails the guard. The allowed sites are:

- the source-side readers (`effect_source`, `gwas_vcf`, `tabular`) and
  `phenotype_sd_pipeline._gather_evidence`, which read a source's columns before anything is
  stored;
- the build-time harvest of the Top-Hit Index's stored `p_value` (`top_hits._write_tier`);
- `effect_consistency`'s check of the values a build is writing;
- an LD sampling tail (`ld/build/diagnostic.sampling_rate`);
- the benchmarks' IVW p-values, which are statistics of an estimator, not of a stored association.

A planted source checks that the guard sees each form it claims to. That source includes
`result["z"] * result["se"]`, `rec.z * rec.se`, `np.multiply(z, se)` and a product split across
lines.

## Consequences

- **Every result carries two more arrays**, `beta` and `pval`, both float64. Every array a result
  held before is unchanged. On the six fixture releases (`tests/golden/store_reader_digests.json`
  as it was then), once `beta` and `pval` are dropped, every query digest was identical to the
  pins written at `5d600a1d`, and the `log10_p` values agreed with those pins to 1e-12, the
  tolerance the pins used.

  *24 Nov 2026 (#474):* those pins are gone. The Store Reader contract bundle
  (`tests/store_reader_contract/`) replaced them: its expected answers carry `beta` and `pval`
  for every shape, and `tests/test_store_reader_digests.py` still checks both against
  `association_statistics` on the six releases.
- **The first read of `beta` or `pval` costs one evaluation per distinct |z|**, with the same
  method as `log10_p_two_sided`, twice: once for `pval` and once for `log10_p`. A caller that reads
  only z and SE pays nothing, because the record derives lazily.
- **CLI output gains columns.**
  - TSV: `beta`, after `se` and before `p`. A positional reader sees `p`, `eaf`, `n` and
    `association_status` move one column right. This is a breaking change.
  - JSON: `beta`, `pval` and `log10_p`.
  - opengwasdb-stores reads these columns, so the change is a documentation change there too.
- **On a release before 0.2.3 that holds `+inf` SE cells**, the TSV `p` column prints `NA` for
  them, where it used to print the p of their z. OGS-00011 holds 8 such cells (ADR 0068).
- **Benchmarks read the Store Reader's values.** The per-instrument `beta_exp`/`beta_out` come from
  the record. The store-side -log10 p of the Hybrid locus and regional reports, and of the FinnGen
  fidelity check, come from `log10_p`.

## Alternatives rejected

- **Only on resolved rows.** The index-keyed record is what opengwaspy and opengwasr read through
  ogquery, and what the benchmarks read. A beta only on resolved rows would leave them all
  deriving their own.
- **`pval` as `10 ** log10_p`.** It is one derivation fewer. But `10 ** x` rounds through the
  exponent, so it is not `p_value_from_z`'s value bit for bit. It would also put a second spelling
  of p in the code.
- **`pval` as the clamped smallest subnormal, or a string, where p underflows.** A clamped value is
  a wrong answer that looks right. A string makes the column unusable as numbers. `log10_p` is
  already the exact value, so `pval` only has to be honest that it underflowed, and 0.0 is.
- **beta `±inf` where SE is infinite**, as `z * se` gives. ADR 0068 rules such an SE an error, so
  an infinite effect would be a fabricated value, not a measured one.
- **`log10_p` on the record as a third mapping key.** The record's keys are the query contract
  (spec §19), and the issue asks for `beta` and `pval`. `log10_p` is on resolved rows, in the JSON
  output, and on `Associations.statistics` for a caller that holds a record.
