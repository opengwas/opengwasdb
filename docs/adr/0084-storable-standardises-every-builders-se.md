# `storable` standardises every builder's SE by its Analysis's phenotype SD

Issue [#366](https://github.com/opengwas/opengwasdb/issues/366), under epic #347
("Builds never store a statistic that contradicts its source"). It rests on
[ADR 0029](0029-standardise-continuous-effects-via-phenotype-sd.md)
(`stored_se = original_se / sd` for `original_sd_method` methods 2-5) and
amends [ADR 0077](0077-one-module-owns-how-an-association-is-encoded.md)
(`storable` is the one judgement a writer makes before any plane holds a cell).
No format change: the contract always said SD Units (spec §7).

## Context

ADR 0029 says what to store; nothing said where. Each builder divided on its
own: `build-dense` (`_standardised`), `build-dense-vcf` (before `storable`),
and `build-hybrid` three times, for the Dense axis, the Overflow and the
off-reference spill. All of them called `apply_se_divisor`, but each caller
had to remember to. The Ragged builders never did. `build-ragged-besd` and
`build-ragged-ssf` read `original_sd_method` and stored their sources' SEs.
`build-ragged-ssf` also dropped `original_sd` from `analyses.tsv`.

Four published Ragged SSF Store Releases came out of this: OGS-00006, -00024,
-00025 and -00026. Their 4,443 Analyses all declare `estimated_from_source_maf`,
with SDs from 0.64 to 1.04, and hold SEs that are the source's. Nothing failed.
z is scale-free, so the Observed-View Check (#345) cannot see it, and an SE
that is off by a factor of 0.98 still looks like an SE. A first fix for #366
added the divide to the two Ragged builders, which made seven call sites. The
owner rejected that shape on 18 Nov 2026.

## Decision

**`opengwasdb.encoding.phenotype_sd.PhenotypeSd` holds the divide, and
`storable` is the only caller that applies it.**

- `storable(cells, analyses, n_analyses, *, where, phenotype_sd)` has no
  default for `phenotype_sd`, and neither has `RaggedCSRWriter.add_analysis`,
  which hands it on. A writer that does not say which scale its SEs are in
  does not run.
- A builder hands its source's SEs with `PhenotypeSd.dividing_by(sd, ...)`,
  one SD per Analysis, indexed as `analyses` is. `storable` returns the cells
  with each SE divided by its Analysis's SD. Every rule after that judges the
  divided SE, the one the Store holds: Admission's check and the #338
  out-of-range rule. The canonical-marker check judges the SE as it was handed
  over, in the dtype a writer gave it.
- `PhenotypeSd.APPLIED` divides by nothing and returns the cells handed over.
  Three kinds of writer pass it:
  - a completion (Dense, Ragged, Hybrid crossovers and Overflow), whose cells
    are its source Store's and its fills;
  - `AssociationCodec.encode`, whose cells are what a Store holds (a builder's
    grid already through `storable`, or a completion's cells), so encoding
    never standardises;
  - the Hybrid Overflow assembly, which writes the Pass 2 spills that
    `_stored_parts` already took through `storable`.
- The SD itself is still parsed from the manifest, once, by
  `layouts.build_manifest` (`_parse_original_sd`, and `original_sd_divisor`
  for Ragged's optional columns). It refuses an SD that is missing, non-finite
  or not positive, naming the Analysis. `PhenotypeSd.dividing_by` refuses one
  too, for a caller that skipped the parser. The rescaling tiers are
  `model.enums.SD_RESCALING_METHODS`.
- The divide is `float32` by `float32`: `se / np.float32(sd)`, the arithmetic
  `apply_se_divisor` used. It returns `se` itself, not a copy, when every SD
  is 1.0. `apply_se_divisor` is deleted.
- `tests/test_phenotype_sd_standardisation.py` scans the package and fails on any division by
  an SD-named value outside `phenotype_sd.py`. Run against the code before
  this ADR, it finds `apply_se_divisor`.
  `tests/test_sd_standardisation_seam.py` builds one source declaring
  `original_sd = 2.0` through all seven paths: Dense, Dense-from-VCF, the
  Hybrid Dense axis, Overflow and off-reference spill, Ragged BESD and Ragged
  SSF. Each stores SE 0.25 from the source's 0.5. On the code before #366 the
  BESD and SSF cases fail.

### Divided once

A source SE reaches a Store through exactly one `storable` call that carries the
Analysis's SD. Every later call that sees the same value is one of the
`APPLIED` writers above, and each of them says so at its call site. Completion
reads a Store, never a source, so its SEs are always in SD Units. Even if a
completion wanted the SD, it has no way to learn it: `original_sd` lives in
`analyses.tsv`, and no completion reads that column.

## Considered options

- **Standardise in the Source Reader**, so every reader yields SD Units.
  Rejected for two reasons. Admission, duplicate resolution and the Admission
  Record judge and log source values, and the source-fidelity check reads
  sources through the same readers. More to the point, a writer would still
  accept any SE it was handed. The defect was a writer that never received
  the SD, and only a writer-side argument without a default makes that
  impossible to write.
- **Keep the divide per builder and add a seam test.** That was the first fix.
  It is seven call sites, and an eighth builder can omit the divide just as
  BESD and SSF did. The test would catch an existing builder, but not a new
  one it does not list.

## Consequences

- A Dense, Dense-from-VCF or Hybrid Store holds the same values as before
  this ADR; the commit message for #366 gives the evidence. A Ragged Store
  whose manifest declares a rescaling method now holds standardised SEs, and
  that change is the fix.
- `storable` is no longer idempotent for a builder's SD: calling it twice with
  the same `dividing_by` divides twice. A writer that re-judges cells passes
  `APPLIED`. ADR 0077's "pure and idempotent" now holds for `APPLIED` only.
- Repairing the four Releases is an opengwasdb-stores ticket.
