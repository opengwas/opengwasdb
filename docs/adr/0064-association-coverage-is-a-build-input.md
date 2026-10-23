# Association Coverage is a build input, not a layout constant

Narrows [ADR 0009](0009-separate-layout-coverage-completion.md)'s two-valued
vocabulary and completes what that ADR separated out. Issue
[#298](https://github.com/opengwas/opengwasdb/issues/298), a prerequisite for
the Analysis Catalog in `opengwasdb-stores` (stores #313, #341).

## Context

ADR 0009 made Association Coverage a first-class axis of a Store Release,
independent of Primary Storage Layout and of Completion State. The manifest got
the field, `docs/spec/store-format.md` §2 got the contract, and then the
builders quietly turned it back into a property of the layout:

- `layouts/dense/build.py`, `layouts/dense/build_vcf.py`,
  `layouts/hybrid/build.py` (both manifests) and `layouts/hybrid/complete.py`
  wrote `full` unconditionally;
- `layouts/ragged/build_ssf.py` and `layouts/ragged/build_besd.py` wrote
  `cis_and_signals` unconditionally.

No build took the value as an input, so no manifest could disagree with its
layout's constant. Two published Stores show what that cost:

- **OGS-00006 (Chen 2023, plasma metabolites)** is filtered to significant
  trans signals and suggestive leads (`sig-trans-1mb_suggestive-leads`). A
  metabolite has no cis region, yet the release says `cis_and_signals`.
- **eQTLGen's cis-eQTL Stores** hold cis associations only, with no trans
  signals at all, and also say `cis_and_signals`.

Both are wrong answers that look like right answers, in a field a reader is
expected to trust: the first over-states what the release holds, the second is
wrong in both directions depending on how it is read. Nothing failed, because
nothing could.

There is a second problem with `full` itself: it does not say what it is full
*of*. A Dense store's `full` means "every association the source reports",
which is a meaningful guarantee; the name made it easy to read as "the complete
matrix", which it is not.

## Decision

### 1. Four values, and `full` is renamed `gwas`

| value | what the release retains |
|---|---|
| `gwas` | every association the source reports, genome-wide |
| `cis` | a cis window around each Analysis's target, and nothing else |
| `cis_and_signals` | a cis window plus significant signals elsewhere |
| `signals_only` | significant signals and their flanks, with no cis window |

`full` is **read** as `gwas` and never written again. `AssociationCoverage`
implements this in `_missing_`, the one place a manifest string becomes a
value, so every reader — `StoreManifest.load`, `info`, validation — gets the
mapping without repeating it. A published release needs no rewrite: it opens,
validates and reports `gwas`. The alias is **transitional** (§5a): it exists
only while a supported pin still writes `full`.

A writer that edits `manifest.json` as JSON — the converter, the restamp, the
Provenance Amendment, the variant-index build — never constructs a
`StoreManifest`, so `_missing_` never runs for it. Each of them normalises the
value through `normalise_coverage` instead, which is why "never written again"
holds on those paths too rather than only on the enum's.

### 2. Every build states its coverage; only the Ragged layouts may state any of them

`build-ragged-ssf` and `build-ragged-besd` take `--association-coverage` with
**no default**: the same command builds a cis-eQTL release, a
significant-signals metabolite release and a full sum-statistics release of the
same layout, and a default would be the layout constant this ADR removes. The
library functions take it as a required keyword for the same reason — a caller
that does not know the coverage of what it is building does not know what it is
publishing.

`build-dense`, `build-dense-vcf`, `build-hybrid` and
`build-hybrid-from-catalogue` take the option, default to `gwas`, and **refuse
any other value**.

The refusal is a representation constraint, not a re-conflation of ADR 0009's
axes. A Dense or Hybrid release holds one cell per (variant, Analysis). An
association a narrower coverage excluded is therefore not absent from the
matrix: it is a cell, and a cell left missing means "the source did not report
this" — which is exactly what Reference Completion fills in from the reference
panel, and what a query reports as an imputed or unobserved association. Two
different facts with one representation is the failure class this project
exists to avoid, so a Dense or Hybrid build cannot faithfully hold `cis`,
`cis_and_signals` or `signals_only` and says so instead of writing a manifest
that claims otherwise. A Ragged sequence is the list of associations kept, so
absence states itself, and all four are representable there.

`COVERAGES_BY_LAYOUT` in `model/enums.py` is the single statement of which
layout carries which values, and both the builders and `validate` read it — a
build cannot write a release validation would reject, and validation cannot
invent a rule no build is held to.

### 3. Completion keeps its source's coverage and takes no option

Reference Completion writes imputed cells into its source's arrays. It does not
retain more or fewer of the source's associations, so the coverage is carried
across rather than re-decided, exactly as `format_version`, `store_id`,
`reference_assembly` and the encoding are (ADR 0038 §4). `complete-dense` still
requires a `gwas` source, because Dense completion is defined over a matrix in
which every missing cell is an imputation target.

### 4. The value is printed where it is consumed

`info` showed it and still does. #294 makes `opengwasdb-stores`' `register`
read `association_coverage` from the stdout JSON of its `build` or `complete`
step, and record `null` for a key nothing prints, so every builder and
completion command now prints it. The CLI reads it back from the release it
just wrote rather than echoing an argument: the recorded value is then the one
on disk, including for a resumed or a completed build where the argument does
not exist.

### 5. This does not move `format_version`, under the pre-release rule

`format_version` is `0.2.0`, below `1.0.0`, and **spec §21.0 decides this case**:
before `1.0.0`, a change confined to a manifest vocabulary or to required
metadata is applied by editing the existing Stores' metadata in place, as one
coherent step across this package, `opengwasdb-stores` and the published Stores;
it does not move the format version, it is recorded in `CHANGELOG.md`, and an
older package may refuse the new values.

That last consequence is real here, so it is stated plainly rather than argued
away. A package built before this change meets `association_coverage: gwas` and
refuses the release, because the vocabulary is closed and an unknown value is
refused by name instead of guessed at (spec §21.2). The owner's decision of
23 Oct 2026 accepts that: one coherent vocabulary across the three repositories
and the published Stores is worth more than an old package reading a release
written after it, and the alternative — a series bump plus a rebuild or
converter for every Store — buys only the opposite direction of compatibility,
for a project that has not released `1.0` and will change the format again.

From `1.0.0` the rule returns to ADR 0038 §1 / ADR 0041 §2: a vocabulary change
is incompatible and moves the series, because a released format is a contract
with readers outside this project. ADRs 0038 and 0041 carry a banner naming
this decision, and spec §21.1/§21.2 point at §21.0 for it.

What the rename does **not** need is a conversion. Every `0.1.0` and `0.2.0`
release stays readable, because `full` is read as `gwas` (§1); the in-place
metadata edit that brings a published Store's own manifest onto the new
vocabulary is a stores-side sweep, run once, and every edited Store is
re-validated. Completion is untouched: it preserves its source's coverage either
way, so no release becomes un-completable (which the alternative — moving
`CURRENT_FORMAT_VERSION` — would have caused, since completion refuses a source
it cannot write: ADR 0038 §4).

### 5a. The `full` read is transitional

`AssociationCoverage._missing_` maps `full` to `gwas` for a limited time, and
for two reasons: Stores built at the pin in use keep writing `full` until
`opengwasdb-stores` re-pins past this change, and the in-place metadata sweep has
not run yet. Both are transitional.

It is removed — with the `full` case in `_missing_`, and nothing else — once no
published Store says `full` and no supported pin writes it. So that this is not
forgotten, `CHANGELOG.md` records it as transitional, and removing it is a
`no-changelog`-exempt-able change of one method plus a test.

## Rejected options

- **Keep `full` as a written value and add `gwas` as an alias for it.**
  Rejected: two spellings of one fact is the ambiguity the rename removes, and
  the next reader has to know which builds write which. `full` is read, never
  written.
- **Give the Ragged builders a `cis_and_signals` default so existing callers
  keep working.** Rejected: that is precisely the defect. The default is the
  layout constant, and the two published Stores above are what it produces.
- **Let a Dense build write any coverage and rely on validation.** Rejected:
  the release would be written, published and only then refused, and a
  distributor that never runs `validate` would ship it.
- **Drop `association_coverage` from Dense and Hybrid manifests instead of
  constraining it.** Rejected: the field is required by spec §2 and read by
  consumers that do not know a release's layout; `gwas` states the guarantee
  explicitly and keeps one schema.
- **Move `format_version` to `0.3.0`.** Rejected under spec §21.0: it is not
  what a below-1.0 vocabulary change does, and it would force a converter on
  every published Store — completion preserves its source's version and refuses
  one it cannot write (ADR 0038 §4), so nothing could be completed until it was
  converted, to record a rename of one manifest string.
- **Keep `full` forever, and let both spellings mean `gwas`.** Rejected: the
  read alias is transitional (§5a) because two spellings of one fact is the
  ambiguity the rename removes; a distinct value in a closed vocabulary is read
  by everyone forever.

## Consequences

- A manifest's `association_coverage` now describes the Store, and two Ragged
  releases of one Source Collection can honestly differ.
- `opengwasdb-stores` must pass the coverage for every Ragged build. A Ragged
  `build.yaml` that does not state it fails at the CLI rather than building a
  release whose manifest is a guess.
- Every pre-`#298` release reports `gwas` where it reported `full`, in `info`,
  in validation and in the Analysis Catalog. Nothing about its data changed.
- `validate` gains one rule per layout: a Dense or Hybrid release declaring a
  coverage its layout cannot carry is invalid, and so is a value outside the
  four.
- The Analysis Catalog can group and filter Stores by retained coverage, which
  is what stores #341 needs.
