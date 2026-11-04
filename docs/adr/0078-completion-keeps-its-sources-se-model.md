# Reference Completion keeps its source's SE model

Issue [#301](https://github.com/opengwas/opengwasdb/issues/301). The owner
chose option 1 on 4 Nov 2026. This ADR amends ADR 0037 §3, which let Ragged
completion refit `se_coefficients`, and closes the consequence that ADR 0077
left open for issue #338.

## Context

A completed release holds its parent's associations unchanged. Spec §12 says
so, the Observed-View Check (#289, #296) checks it, and opengwasdb-stores ADR
0037 depends on it when it publishes a completed Store under its parent's id.

Dense and Hybrid completion patch their source's SE plane and read
`se_coefficients` from the source. Ragged completion rewrites its whole CSR
plane from decoded values. It refitted the residual model over the completed
rows (observed and imputed) and coded every cell against the new model. Each
observed code then decoded to a slightly different SE:

- On three Chen 2023 Analyses completed against the HGDP+1kGP EUR panel
  (opengwasdb-stores #232, 22 Oct 2026), all 135,596 observed SEs moved: by a
  median of 0.09%, and at most 0.14%. Z and EAF were identical.
- On the test fixtures it moved them even when nothing was imputed, because the
  refit was taken over decoded values rather than over the source's own.

The strict Ragged check refused the pair with `se_mismatch`. That blocked
completing OGS-00006 under its own id (opengwasdb-stores #224).

## Decision

**A completed release codes every SE against its source's `se_coefficients`
and its source's range. Completion never refits the model and never narrows
the plane to `float16`.**

- **Observed cells are carried exactly.** Ragged completion passes its
  observed cells to `AssociationCodec.encode` as `carried`. A carried cell
  whose residual code would decode to anything other than its value is held
  as an exact exception, and every other cell keeps its source code. Two
  cases need this, both rare: a source exception whose `float32` value rounds
  back inside the range, and an observed cell whose EAF decodes differently
  after completion. Without the rule each would be re-quantised by up to half
  a step, silently. With it every observed SE decodes to its source value,
  and the source's exceptions carry over with their values.
- **Imputed cells are coded against the same model.** An imputed cell whose
  residual falls outside the range becomes an exact exception, as any
  out-of-range residual does (spec §6a).
- **The 2% exception budget is reported, not enforced.** A build uses the
  budget to choose its range, and completion does not choose again. Imputed
  cells may take an Analysis past 2%. A Ragged completion records each
  Analysis's share in `provenance.completion.se_exceptions` and logs a
  warning naming every Analysis over the budget. The owner decided not to
  measure the budget before this change: completions from now on show
  whether imputed cells breach it. `validate` does not check the budget,
  which bounds side-table bytes and not accuracy (ADR 0037 §3), so a breach
  never makes a completed store invalid.

## Considered options

- **Let the check tolerate re-quantisation** (option 2): compare SE to within
  the codec's error and reword the contract to "within the codec's error".
  Rejected. "Identical" would no longer mean identical, and a real defect
  smaller than half a step would pass the check.
- **Store SE as `float16` in every Ragged Store that may be completed**
  (option 3). Rejected. Whether a Store will be completed is not known when it
  is built, and residual coding saves about half the SE bytes.
- **Refit, then store as exceptions the observed cells the refit moves.**
  Rejected. Nearly every observed cell moves (1,194 of 1,196 on the fixture),
  so every observed SE would become an exception.
- **Fall back to `float16`, or refit, when imputed cells breach the budget.**
  Rejected. Either changes observed values that the check protects, and the
  budget bounds bytes, not accuracy.
- **Carry the source's codes directly**, without encoding them again.
  Rejected. Completion reorders the CSR and remaps the variant axis, so the
  codes and the exception table would have to be re-keyed outside the codec.
  ADR 0077 put the codec in one module to stop that. Encoding the decoded
  values again with the `carried` rule gives the same codes and is checked
  against the decode itself.

## Consequences

- A completed Ragged release decodes every observed SE exactly as its source
  does, and the strict Observed-View Check passes it when SE is
  `int8_residual`. The Chen 2023 completion was repeated from the same sources
  on 4 Nov 2026. With the refit, all 135,596 observed SEs moved (median 0.091%,
  max 0.138%) and the check failed with `se_mismatch` = 135,596. With this
  change none moved, the coefficients are the source's bit for bit, and the
  check passes. Of the 28,297 imputed SEs, none is an exception: each
  Analysis's share is 0% (8,383, 11,269 and 8,645 imputed SEs). The source has
  no exceptions either.
- The open consequence in ADR 0077 is gone. An Analysis that issue #338's
  refusal leaves with fewer than two SEs no longer has a fit to fail, because
  completion fits nothing.
- A completed release's SE exception table may grow past 2% of an Analysis's
  SEs. The record and the warning make that visible. If breaches become
  common, a later decision can weigh their bytes against the alternatives.
- Dense and Hybrid completion behaved this way already, but they record no
  per-Analysis exception share. Only a Ragged completion writes
  `se_exceptions`.
- `format_version` does not move. The arrays a reader decodes keep their
  meaning, and `se_exceptions` is a new key in a provenance block whose keys
  vary by layout already.
