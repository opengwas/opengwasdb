# Derive an effect from a signed z-score, EAF and per-variant N

ADRs 0049 and 0050 made an Analysis's effect column a resolved fact and
enumerated its accepted spellings. This ADR adds a signed `z_score` as a third
effect source, and records the one place in this seam where the reader
*derives* an effect by approximation rather than reading or converting one.

## Context

GWAS-SSF permits an Analysis to report its result as a z-score instead of an
effect size. A reader that only reads `beta`/`odds_ratio` yields nothing for
such a file: every row's beta is absent and the association stream is empty —
the silent failure #213 removed for `odds_ratio`.

For a signed z, effect-allele frequency `f` and sample size `N`:

```
se   = 1 / sqrt(2 * f * (1 - f) * (N + z^2))
beta = z * se
```

Unlike `beta = log(odds_ratio)`, this is **not a change of units**. It assumes
the phenotype is standardised (`var(Y) = 1`), so the derived beta is in
phenotype-SD units by construction. `f` and `N` are per-row in qualifying files
— `N` genuinely varies per variant — and the row's own N must be read, never a
study-level scalar.

Real GWAS-Catalog EUR hybrid pool: of 207 files with a z-score column, **77
carry no `beta`/`odds_ratio` column at all**, and exactly **36** of those are
usable under this rule (usable per-row EAF and N, signed z). The remaining 41
are 38 with `effect_allele_frequency` entirely `NA`, 2 case-control, and 1 with
no N. Those must stay unreadable through this path, not be rescued with a
substituted frequency.

## Decision

**A signed z is a derived, standardised effect source, guarded on three sides.**

1. `EffectSourceKind.Z_SCORE` accepts the enumerated spellings
   `("z_score", "Zscore", "ZScore", "z")`; precedence is
   `("beta", "BETA") > "odds_ratio" > z-score`, so a file that carries a beta
   column is never read through its z column.
2. `derive_z_score_effect(z, f, N)` returns `(beta, se)` by the formula above,
   or `None` when any input is missing, `f` is outside `(0, 1)`, or `N` is
   non-positive. The caller drops the row; it never substitutes a frequency or a
   study-level N. The per-row sample size is resolved from the enumerated
   spellings `("n", "N")` under the same duplicate/ambiguity rule as an effect
   column.
3. `EffectSource` reports `is_derived=True` **and** `assumes_standardised=True`,
   so a caller cannot obtain a z-derived effect without obtaining the
   assumption it rests on.
4. `CaseControlZScoreError(ValueError)` is raised when the Analysis's
   `stored_effect_scale` is `log_or` or `log_hazard`: a standardised beta is not
   a log-OR, and deriving one would silently relabel an effect it is not.
5. `UnsignedZScoreError(ValueError)` is raised when the z column carries no
   negative value. A `|z|` or chi-square statistic has the same magnitude and
   no sign; reading it as signed would give every derived effect the wrong
   direction. The column is read until a negative value proves it signed, before
   any row is yielded — a prefix cannot prove signedness, and a partially
   consumed stream must not look plausible.
6. Both the row-wise (`stream_projected_metrics`) and blocked
   (`stream_projected_metric_chunks`) projections derive identically, asserted
   bit-for-bit.

The scale policy (4) lives with the reader, because only the reader knows
`stored_effect_scale`; the sign guard (5) lives with the tabular projection
mechanics, because it reads the source file.

## Consequences

- **36 real Analyses become readable** that previously produced an empty
  association stream. Verified on `GCST90129599` (`Zscore`, uppercase `N`) and
  `GCST90559206` (`z_score`): the derived `beta` and `se` equal the formula
  bit-for-bit for 20/20 rows each, and the case-control refusal reproduces on
  real data.
- **A z-only file with no `n`/`N` column is unusable, loudly silent per row**:
  every row yields `beta=se=None` and drops from the association stream. That is
  the documented contract for a missing input, not an error, matching how an
  unusable `beta` behaves.
- **The sign guard costs one extra read of the z column**, bounded by the first
  negative value for a signed column (and a full read for an unsigned one). It
  is paid only by z-derived sources.
- **A file carrying a `beta` column is never read through its z column**, even
  when that beta is entirely `NA`. This is the explicit precedence rule; the 36
  qualifying files are exactly those without a beta/odds_ratio column.
- **`extract_at_sites` derives `se` from z but is not sign-guarded**, because
  `se` depends on `z^2` and not its sign; ancestry assignment (which reads only
  AF) therefore keeps working on a z-only file.

## Rejected

- **Reading a study-level N or the analysis's `sample_size` for every row.** `N`
  varies per variant in these files, and a study-level scalar would give a
  precise-looking `se` that is wrong for most rows.
- **Deriving on the `log_or`/`log_hazard` scale.** The formula yields a
  standardised beta; relabelling it a log-OR is a wrong answer that looks right.
- **Rescuing an `NA` EAF from a reference panel or a default 0.5.** Absence and
  a substituted value are different; the row drops.
- **Treating an all-non-negative z column as signed.** It is the one shape where
  every derived sign would be wrong, and it is cheap to refuse.
- **Folding the case-control refusal into `extract_at_sites`.** That path
  produces AF and SE, not a beta, and refusing there would break ancestry
  assignment for case-control z-only files over a derivation they never read.
