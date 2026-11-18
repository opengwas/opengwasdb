# `storable` standardises every builder's SE by its Analysis's phenotype SD

Issue [#366](https://github.com/opengwas/opengwasdb/issues/366), under epic #347
("Builds never store a statistic that contradicts its source"). It rests on
[ADR 0029](0029-standardise-continuous-effects-via-phenotype-sd.md)
(`stored_se = original_se / sd` for `original_sd_method` methods 2-5) and
amends [ADR 0077](0077-one-module-owns-how-an-association-is-encoded.md)
(`storable` is the one judgement a writer makes before any plane holds a cell).
No format change: the contract always said SD Units (spec §7). The #338 out-of-range
rule ([ADR 0068](0068-an-se-beyond-float16-is-missing.md)) judges the SE this ADR's
divide produces.

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

**`opengwasdb.encoding.sd_units.PhenotypeSd` holds the divide, and
`storable` applies it to every cell a writer stores.** One other caller uses
it: `validate`'s source-fidelity check calls `PhenotypeSd.standardised` to
compute the SE a build stored from a source row. It stores nothing.

- `storable(cells, analyses, n_analyses, *, where, phenotype_sd)` has no
  default for `phenotype_sd`, and neither has `RaggedCSRWriter.add_analysis`,
  which hands it on. A writer that does not say which scale its SEs are in
  does not run.
- A builder hands its source's SEs with `PhenotypeSd.dividing_by(sd, ...)`,
  one SD per Analysis, indexed as `analyses` is. `dividing_by()` with no SD is
  refused, and so is a count that is not the batch's `n_analyses`, naming both
  counts. "No rescaling" is the 1.0 that the manifest parser supplies for a method that
  carries no SD (see the 18 Nov 2026 addendum for a row that names none): no divisor parameter or field in the package has a
  default (issue #366). `storable` returns the cells
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
  - the Hybrid Overflow assembly (`layouts/hybrid/overflow_spills.py`), which
    writes the Pass 2 spills that `_stored_parts` already took through
    `storable`. It reads no source row, and it lives outside `build.py` so
    that `build.py`, which does, stays off the allowlist below.
- The SD itself is still parsed from the manifest, once, by
  `layouts.build_manifest` (`_parse_original_sd`, and `original_sd_divisor`
  for Ragged's optional columns). It refuses an SD that is missing, non-finite
  or not positive, naming the Analysis. `PhenotypeSd.dividing_by` refuses one
  too, for a caller that skipped the parser. The rescaling tiers are
  `model.enums.SD_RESCALING_METHODS`, shared by the parser and `validate`.
- The divide is `float32` by `float32`: `se / np.float32(sd)`, the arithmetic
  `apply_se_divisor` used. It returns `se` itself, not a copy, when every SD
  is 1.0. `apply_se_divisor` is deleted.
- `tests/test_phenotype_sd_standardisation.py` scans the package and fails on any division by
  an SD-named value outside `sd_units.py`. Run against the code before
  this ADR, it finds `apply_se_divisor`.
  `tests/test_sd_standardisation_seam.py` builds one source declaring
  `original_sd = 2.0` through all seven paths: Dense, Dense-from-VCF, the
  Hybrid Dense axis, Overflow and off-reference spill, Ragged BESD and Ragged
  SSF. Each stores SE 0.25 from the source's 0.5. On the code before #366 the
  BESD and SSF cases fail. A negative test makes Ragged SSF pass `APPLIED` and
  watches the seam assertion fail.
- **What is enforced, and how.** `APPLIED` is a public value, and no type stops
  a writer from passing it with a source's SEs. Instead, the same test file
  fails when:
  - any module in `opengwasdb/` names `APPLIED` (found by AST, whatever its
    filename) unless it is on an explicit allowlist of stored-cell writers,
    `_MAY_SAY_APPLIED`. Each entry carries a one-line reason. The list is:
    the owner, `association_codec` (`encode`), `completion/fill_shards`,
    Hybrid and Ragged completion, and `hybrid/overflow_spills`. A new module
    that needs `APPLIED` is added to the list on purpose. An entry whose
    module no longer exists, or no longer names `APPLIED`, also fails, so the
    list cannot keep a stale exception. A package-wide scan replaced a filename glob
    over the source builders: a builder copied to
    `layouts/ragged/source_ssf.py`, or a helper in `source_sd.py`, passed the
    glob but fails this scan;
  - anything but `sd_units.py` constructs a `PhenotypeSd`;
  - any divisor parameter or field has a default.

  An empty Ragged Analysis passes its own `dividing_by`, like any other.

### Divided once

Each source builder judges each source association in one `storable` call that
carries the Analysis's SD:
- the grid, for Dense;
- the Pass 2 column, for Dense-from-VCF and Hybrid;
- `add_analysis`, for Ragged.

Every later writer of the same values is one of the `APPLIED` writers above.
None of them is a source builder, and the allowlist scan checks that nothing
else names `APPLIED`. Completion reads a
Store, never a source, and reads no `original_sd`. The seam test pins the
outcome, one division per route, with literal values.

Nothing in the types prevents a second `dividing_by` call on cells that are
already standardised. A builder that made one would fail the seam test for
its route, not an import or a type check.

### `validate` checks what the builders now record

`analyses.original_sd_recorded` fails a Release whose `analyses.tsv` gives a
rescaling `original_sd_method` with an `original_sd` that is blank, non-finite or
not positive, naming the Analysis (spec §7a, §20.0). That is the record each of
the four Releases left. Every Release fails the rule, whatever its
`format_version`, and no older version is let off with a warning, unlike #337's
infinite SE. That SE was a value its Store faithfully held. These Releases
hold SEs that are wrong by their own metadata. No format version marks the
fix, because none changed. A warning would also let a Release that is wrong
validate clean.

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
- Repairing the four Releases is an opengwasdb-stores ticket. Until they are
  repaired, `validate` fails them.

## Addendum, 18 Nov 2026: a blank method is "not standardised, scale unknown"

The text above said "no rescaling" was never an omission. For the Ragged
builders it was: a row with no `original_sd_method` was read as divisor 1.0, and
`build-ragged-ssf` still labelled the Analysis `stored_effect_scale=sd` from its
own default, so an undivided Analysis could be called SD Units. The code now
reads it this way:

- A blank `original_sd_method` with a blank `stored_effect_scale` means *not
  standardised, scale unknown*. It claims nothing, and is what OGS-00001 and
  OGS-00002 hold. Only the Ragged BESD builder can build it.
- A row labelled `stored_effect_scale=sd` must name a rescaling method
  (`SD_RESCALING_METHODS`) or `declared_standardised`. Any other method, or
  none, is refused naming the Analysis, in `require_sd_scale_standardised`,
  which Dense, Dense-from-VCF, Hybrid, `build-ragged-besd` and
  `build-ragged-ssf` all call (issue #366).
- `build-ragged-ssf` has no `--stored-effect-scale`: the scale is each manifest
  row's, and a row that declares none is refused.

No published Release is affected (census of `opengwasdb-stores/stores/*/analyses.tsv`:
every Ragged Release declares a method, except OGS-00001 and OGS-00002, which are
blank in both columns).

The module is `opengwasdb/encoding/sd_units.py`, named for CONTEXT.md's SD Units; it was
`encoding/phenotype_sd.py`, which `build/phenotype_sd.py` (the estimate of the SD) shared a
name with. The class keeps its name, `PhenotypeSd`: it is the phenotype SD a builder divides by.
