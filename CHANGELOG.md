# Changelog

All notable changes to the `opengwasdb` package.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is described in [`CONTRIBUTING.md`](CONTRIBUTING.md#versioning) —
in particular, **the package version and a Store Release's `format_version`
are different things** and move independently. See the compatibility table at
the end of this file.

## [Unreleased]

Work lands on `dev` and appears here under *Unreleased* until `dev` merges to
`main`, at which point it is cut into a version.

### Fixed

- **A long indel could answer another variant's lookup** (#127). The ALID
  search index is a fixed-width array — that is what makes `np.searchsorted`
  work over it as an mmap — but it was built with
  `np.array(..., dtype="|S64")`, which truncates silently. Two indels at one
  position whose alleles agreed over 64 bytes collapsed to one key, and a
  lookup by either full ALID returned whichever row sorted first. On the
  published `finngen-r13/r13-pilot-20` release: 6,216 truncated ALIDs, 248
  keys shared by more than one variant, **342 variants answering to another's
  name**, through both `by_alid` and the vectorised `indices_by_identifiers`
  that dense `lookup()` uses. An over-wide ALID is now left out of the index
  and counted, the rule `_write_rsid_index` has followed since #109; it stays
  reachable by exact scan over its position. `validate_store` rejects an index
  holding a key shared by two variants, so an affected store says so.

### Changed

- **The Store Variant Axis no longer keeps a relational copy of itself**
  (#128). `index.sqlite`'s `variants` table duplicated every column of
  `variants.tsv.gz` — which carries `source_alid` besides — was written by one
  builder, read by one self-described "legacy" function that nothing called,
  and had a `rsid` column left `NULL` in all 21,230,615 FinnGen rows while the
  TSV had rsids for 96.9% of them. Dropped, along with its range index and
  `variant_by_identifier()`. Its duplicate-ALID validation moves onto the ALID
  search index — the structure queries actually read, and the one that also
  catches #127's collisions. Measured on the #117 rebuilds, the variant axis is
  69–88% of a pilot store against the statistics' 11–31%: **803 MB of FinnGen's
  4.86 GB** was this table.
- **The ALID index slot narrows from 64 to 32 bytes**, now that an over-wide
  ALID is excluded rather than truncated and the width is a size/latency trade
  rather than a correctness one. Measured across the FinnGen, GWAS Catalog and
  metabolome pilots, ALID length is mean 14.9 and p99 21; at 32 bytes the index
  halves (**1,359 MB → 679 MB on FinnGen**) and 0.21% of variants resolve by
  exact scan instead of binary search.

### Added

- **Getting-started documentation now covers the first local Store Release**
  (#110). A fresh checkout can follow `docs/getting-started.md` to install with
  Pixi, build and validate the in-repository tiny Dense fixture, run PheWAS,
  exact-lookup and top-hit queries, and distinguish absent EAF from a fabricated
  value.

- **Top-hit indexes carry decoded effect allele frequency** (#131–#134,
  ADR 0040). Dense, Ragged, and both Hybrid components now answer `top_hits()`
  from one compact derived structure without reopening or gathering from the
  source EAF plane. Rebuilding an existing index adds `float32 eaf`; readers
  retain a plane-backed fallback for older indexes. Dense/Hybrid VCF builds
  populate it inline with a variant-row-ordered pass. On the rebuilt ukb-b
  index, per-Analysis top hits recovered from the 86.6 ms regression to
  1.301 ms (pre-EAF: 1.17 ms), while global top hits recovered from 7,129 ms
  to 473.945 ms (pre-EAF: 488 ms).

- **`eaf` is stored as a per-variant baseline plus a per-cell `int8` logit
  residual, and `format_version` moves to `2.0`** (#116, ADR 0037 §2/§4).
  ADR 0036 shipped EAF as a `float32` plane parallel to `z`/`se`, which nearly
  doubled a store's statistic bytes for a column that is annotation rather than
  the finding. The semantics are unchanged — EAF is still per (variant,
  Analysis), still oriented to the stored effect allele, still declared per
  Analysis by `eaf_scope`. Only the physical encoding changes.
  - **The transform is the logit**, `log(f / (1 − f))`, not `log(f)`: a residual
    error of `d` moves `f` by a relative `(1 − f)·d` and `1 − f` by a relative
    `f·d`, so neither the effect allele frequency nor the minor allele frequency
    users filter on can be wrong by more than `d`. `log(f)` is blind to the
    minor side, which for a frequency near 1 is the only side anyone reads.
    Measured on FinnGen chr1 (8 endpoints, 429,961 variants, 3.44M EAF-bearing
    cells) the two transforms leave residuals of the same width — sd **0.0170
    against 0.0169**, max 0.632 for both — so the bound costs nothing.
  - **Two of the 256 codes are reserved**, leaving 254 levels: `-128` is "this
    Analysis reports no EAF here" (decodes to NaN, and is *not* a residual of
    zero), `-127` is "exception — exact `float32` in the plane's table". A cell
    that is neither round-trips to within half a step. On real pilot
    frequencies: **p99 0.184% / max 0.262%** at ±0.5 (FinnGen) and **p99
    0.374% / max 0.394%** at ±1.0 (GWAS Catalog), against half-step bounds of
    0.197% and 0.394%; on a synthetic sweep spanning all three ranges, max
    0.219% / 0.419% / 0.812%. All figures are the worse of the relative error
    on EAF and on 1 − EAF, because the minor allele frequency is what users
    filter on. Every exception cell round-trips exactly.
  - **Frequencies of 0 and 1, residuals outside the range, and cells at a
    variant with no usable baseline are held exactly** in an
    `eaf_exception_index` / `eaf_exception_value` table beside the plane — the
    same structure, keying and validation as `z`'s overflow table, sharing one
    implementation so the two cannot drift. Resolving one costs **109 ns/cell**
    against a 1,000-entry table, 219 ns against 100,000 and 684 ns against
    1,000,000, paid only for the cells that are exceptions.
  - **The range is chosen from measured data and measured bytes**, never
    inferred from the layout: the smallest of ±0.5 / ±1.0 / ±2.0 whose exception
    fraction is within 2%, then compared against ADR 0036's `float32` plane, and
    `float32` is written when the residual coding would not be smaller. A sparse
    store with roughly one EAF-bearing cell per variant pays more for the
    per-variant baseline than the `int8` cell saves, and stays in `float32`.
    Measured by running the shipped encoder over chr1 of two pilots' sources as
    a Dense grid, under the store's own zstd-3 + bitshuffle codec: FinnGen
    (429,961 variants x 8 endpoints, 8.00 EAF cells/variant) picks ±0.5 and
    costs **0.718 B/cell against `float32`'s 1.373, −48%**; GWAS Catalog EUR
    (419,192 x 9 studies, 4.86 cells/variant) picks ±1.0 and costs **2.023
    against 5.072, −60%**. The per-variant baseline measures 0.424 B/cell
    across 8 Analyses, which is ADR 0037's own prediction of 0.42. The *plane*
    figures do not reproduce ADR 0037 §2's 0.14 / 0.52 B/cell — the same
    encoding measures 0.294 / 1.245 here, 2.1x and 2.4x — so the issue's
    "within 20% of the table" criterion is **not met by these numbers**. It
    cannot be settled from source files anyway: it is a measurement on rebuilt
    Store Releases, which is #117, and whichever way that lands one of the two
    tables is wrong. The decision is unaffected — the residual coding is
    48–60% smaller than the plane it replaces on both pilots.
  - **Reference-panel EAF for imputed cells** (#113, superseded in approach). An
    imputed cell's EAF *is* the panel's and is identical for every Analysis
    imputed at that variant, so it is a per-variant `eaf_reference` array rather
    than per-cell data, applied on read through the imputed mask that Association
    Status already records. It amortises to ~0 B/cell on wide Dense grids, but
    #117 measured **3.206 B/cell** on completed `eqtlgen`, where a sparse Ragged
    store has only 2.4 cells per variant. Read straight from the panel, it never
    travels through the resumable completion checkpoint — which was #113's
    stated blocker.
  - **An observed cell whose source reported no EAF stays NaN.** It does not
    fall back to the panel: FinnGen's frequencies differ from the EUR panel by
    up to **3000×**, so substituting one would hand a user a plausible number
    that is wrong by three orders of magnitude, precisely for the rare variants
    they filter on. `decode_eaf` requires the imputed mask and the reference
    array together whenever a release declares reference EAF, so the
    substitution cannot be half-applied.
  - **`eaf_reference` is declared per component.** A Hybrid release's Dense
    Component has imputed cells where its Ragged Overflow does not, and each
    carries its own manifest; it is the one field on which the two components'
    plans may differ, and validation compares them with it normalised away.
  - **A release whose Analyses reported no frequency at all still carries panel
    EAF on the cells it imputes** — the case #113 was actually raised about. It
    carries `eaf_reference` and no `eaf` array: NaN on every observed cell, the
    panel's frequency on every imputed one, and `eaf_scope=association` on the
    Analyses that gained them. Both completion pipelines and both read planes
    do this; declaring `reference` beside an `absent` plane is legal, and
    validation no longer reads that pairing as a contradiction.
  - **An LD Reference Panel that declares no EAF completes rather than
    failing.** The panel is supplied for imputation and its frequencies are
    optional (`completion.ld_panel` has always read a missing one as NaN), so
    asking it for reference EAF must not turn a supported panel into a failed
    completion that takes the store's own frequencies with it. The same holds
    for a panel with no ancestry directory or no block tables: it is asked, not
    depended on.
  - **A Hybrid release's `eaf_scope` cross-check reads both components.**
    `eaf_reference` is per component, and a Hybrid release's Ragged Overflow is
    observed-only, so its top-level plan declares none while its Dense
    Component does. Judging the release from the top-level plan alone called a
    Hybrid store whose only frequencies were the panel's a contradiction of its
    own `analyses.tsv`.
  - **A pair-resolving CSR read reports the panel's frequencies too.**
    `eaf_pairs` short-circuited on "this component stores no `eaf` array",
    which is true of a release carrying only reference EAF — and is not the
    same question as "this component has a frequency to report". Top-hit reads
    go through it.
  - **A completed Hybrid release records the panel at the top level**, not only
    inside its Dense Component: `provenance.completion` now names the
    `ld_panel_id`, `ancestry` and `method`, read back from the component that
    did the imputation so the two manifests cannot name different panels.
    Manifest surface, so `opengwasdb-stores` needs it too.
  - **An `encoding` block of this version must declare its `eaf` plan.** A
    missing `eaf` key names ADR 0036's optional plane only below plan-schema
    version 2; at or above it the release is malformed and is rejected rather
    than read as an older one. A block with no `version` at all is read as
    version 1, since `version` was added with the plan itself.
  - **Validation gains the plan-versus-arrays rules for `eaf`**: a plane whose
    dtype contradicts the manifest, a residual plane missing its baseline or its
    exception table, an exception cell with no entry, a table describing a cell
    that is not an exception, an `eaf_reference` array the plan does not declare,
    and — the disagreement that got through review on #106 — an `eaf_scope` that
    contradicts the release's declared plan.
  - **`ogdb info` reports the `eaf` encoding** alongside `z` and `se`. Manifest
    and CLI surface, so `opengwasdb-stores` needs the same change.
  - **An ancestry-excluded Analysis no longer declares completion it did not
    get.** Dense completion imputes every LD block for every Analysis and
    applies the ancestry-match filter afterwards (ADR 0028); the filter was
    applied to the fills at the write but not to the `completion_quality` rows
    they came with, so a nonmatching Analysis carried through observed-only
    declared a nonzero `completion_n_imputed_total` — and, on a source with no
    frequencies of its own completed against a panel with some, an `eaf_scope`
    of `association` derived from that count, on an Analysis whose every cell
    reads NaN. The filter is now applied once, where checkpoint output becomes
    the release's, so the table and the arrays cannot disagree; the writer
    checks the shards honour it rather than silently re-filtering. Ragged was
    never affected — it excludes nonmatching Analyses at block assignment.
  - **Validation compares each Analysis's completion metadata with its own
    cells**, not only the store's plan with the store's arrays: an Analysis
    that declares imputed cells must hold at least one, and one that holds them
    must account for them. Categorical rather than a count comparison, since
    the rollup and the arrays count different things. Alongside it, a
    layout-independent rule read from `analyses.tsv` alone — a blank
    `completed_against` and a nonzero `completion_n_imputed_total` are a
    contradiction.
  - **A completed Hybrid release fails rather than name no panel.** Copying the
    panel identity up from the Dense Component caught every exception and wrote
    nulls, so a manifest could reach `reference_completed` with no
    `ld_panel_id` at all — the field #116 made load-bearing — and only a log
    line to say so.
  - **A release whose only frequencies are the panel's needs no orientation
    evidence.** Found by rebuilding `eqtlgen-cis-pilot` on the merged code, not
    by a test: it is built from BESD, which carries no EAF at all, so its build
    runs no orientation check and leaves `eaf_orientation` blank. Completion
    then stamps `eaf_scope=association` for the panel's reference EAF, and
    §9.1's evidence rule rejected the release — demanding a check on a cohort
    frequency column that does not exist and that no build could supply. Every
    Reference-Completed release built from an EAF-less source failed
    `validate_store`. The rule now applies only where some component declares
    an `eaf` plane; reference EAF is oriented by construction, through the same
    reader the check itself uses. Both no-frequency fixtures missed it because
    their builders record `unverified`, which warns rather than fails.
  - **A panel with no `EAF` imputes nothing, and the spec now says so.** An
    imputed `se` is scaled by the panel's heterozygosity, so such a panel
    produces no imputed cells rather than an `se` derived from a substituted
    frequency. The completion succeeds and the release's own frequencies
    survive it, which is what the fix above is for; "imputed cells read NaN" is
    vacuous here, and §6a no longer implies otherwise.

- **`z` is stored as `int16` fixed point, and `format_version` moves to `1.0`**
  (#114, ADR 0037 §1). `float16`'s step doubles with magnitude, so it was least
  precise exactly where p-values are steepest: worst-case p error was 26.4% at
  |z| = 30 and the actual error at the FADS1/FADS2 hit (|z| = 47.8) was a factor
  of 1.82. Stored `z` is now `round(z x 1024)` with uniform precision — 1.6%
  worst-case p error at the range edge, 0.24% at z = 5 — and smaller: **1.400
  against 1.591 B/cell, −12%**, measured by re-encoding a 50.3M-cell band
  (28.5M present) of the `ukb-b` pilot under the store's own zstd-3 + bitshuffle
  codec. Bitshuffle groups the near-constant high bytes of a bounded quantity,
  where `float16`'s exponent bits churn for the small values that dominate.
  (ADR 0037 reports −20% from a different sample; the `int16` side agrees at
  ~1.4–1.47 B/cell and the `float16` baseline is what varies with the data.)
  - **Two codes are reserved**, because an integer plane cannot hold NaN and the
    missing-cell contract depended on it: `-32768` is missing (decodes to NaN,
    paired `se` must also be missing), `-32767` is out of range. Spec §15 now
    defines missingness per plane's declared codec rather than as NaN.
  - **Out-of-range values are held exactly, not clipped and not rejected.** The
    `ukb-b` survey found max |z| = 137.5 (HERC2/OCA2) with 6,346 cells above 32
    (6,312 in the copy re-surveyed here, at `/data/opengwasdb/wip/ukb-b`),
    so the representable range is backed by a sparse `z_overflow_index` /
    `z_overflow_value` table beside each plane — 74 KB against a multi-gigabyte
    store. The 500 strongest cells of that pilot, |z| up to 137.5, round-trip
    through the table exactly (max |Δz| = 0.000000, p ratio 1.000000; the
    largest is log10 p = −4107.68). A build now fails only on a non-finite or
    malformed statistic.
  - **`se` stays `float16`,** deliberately: it spans 3.2 decades and needs
    relative precision, which a float exponent already provides. The right
    encoding follows from the shape of the quantity; this is not a preference
    for integers.
  - **`StoreEncoding` and `StoreCodec`** (`opengwasdb.encoding`, issue #119) —
    the plan is decided once per build in `StoreEncoding.decide()`, recorded in
    `manifest.json`, and read back by every builder, completion pass, query
    adapter and validation rule. No read path re-derives it, and a reader
    meeting an encoding kind it does not implement rejects the release.
    `StoreEncoding.legacy()` covers pre-#114 stores, whose planes are `float16`.
  - **Validation cross-checks the plan against the arrays**: a plane whose dtype
    contradicts the manifest, an out-of-range cell with no overflow entry, a
    table describing a cell that is not out of range, or a Hybrid release whose
    two components declare different plans all fail the store.
  - **A release at `format_version` 1.0 or above must declare its encoding.** A
    missing block is refused rather than falling back to the legacy plan, which
    would decode an `int16` plane as `float16` and return z-scores a thousand
    times too large — a plausible number, and so the worst possible outcome.
    The codec refuses the same disagreement reached directly.
  - **Manifest change**: the per-layout `provenance.*.dtype` field is now
    `se_dtype`. It only ever described the float planes, and leaving it named
    `dtype` next to an `int16` `z` would have made the manifest quietly wrong.
    A reader of the old key gets nothing rather than a wrong answer.
  - `format_version` becomes `1.0`. `0.1` releases stay readable — they decode
    under the legacy plan — and are never written again; **completing a `0.1`
    store is now refused** (ADR 0038 §4) rather than stamping its version onto
    newly encoded arrays. Rebuild instead.
- **Allele-flipped EAF is rejected at build time** (#115, ADR 0037 §6). Each
  Analysis's A1-oriented `eaf` is correlated against a reference over a
  deterministic sample of variants before any statistic array is written;
  `r < 0` fails the build, naming the Analysis and the observed `r`. This
  catches `GCST003566` in the `gwas-catalog-eur-hybrid` pilot, which reports
  `effect_allele_frequency` against the *other* allele (r = -0.9992 against the
  EUR panel, where every other study in the release reads +0.999). A
  correlation rather than a difference threshold, so a bottlenecked cohort
  whose frequencies differ from the panel by 3000x still passes.
  - New `--eaf-reference` / `--eaf-reference-ancestry` /
    `--allow-unverified-eaf` options on `build-dense-vcf`, `build-hybrid` and
    `build-ragged-ssf`, reading either an LD panel directory or a table with an
    `eaf` column.
  - With no reference and three or more Analyses, the consensus of the other
    Analyses is used instead; a build whose Analyses contradict each other
    fails rather than guessing which is right.
  - Outcomes are persisted: `eaf_orientation`, `eaf_orientation_r` and
    `eaf_orientation_n` in `analyses.tsv`, plus the reference's identity and
    checksum in `manifest.json`'s `provenance.eaf_orientation`.
  - **The same check at ancestry assignment**, where the frequencies were
    already being compared to a reference: a mis-oriented Analysis is left
    Unassigned with `gate_reason=eaf_orientation`, and the evidence is written
    into the Analysis Catalogue (`eaf_orientation`, `eaf_orientation_r`). The
    NNLS residual gate already rejected such an Analysis (0.5788 against a
    threshold of 0.06 for `GCST003566`) but could not say why, and an inverted
    Analysis does not merely fail to fit — it fits as another super-population,
    AFR at 0.696, above the τ = 0.50 gate. Recalibrating τ/δ cannot re-admit it.
  - `audit-eaf-orientation` re-runs the correlation against a supplied panel
    over a built store's own arrays, for stores built before the check existed.
- **A store-format versioning and migration policy** (#112, ADR 0038).
  `format_version` is `MAJOR.MINOR`, with the split defined by what a reader
  that does not know about a change would do: major if it would misinterpret
  the store, minor if it would still read correctly everything it knew about.
  Spec §21 now answers what bumps which, what a reader owes a store it did not
  write, and "I have an old store — now what?".
  - `SUPPORTED_FORMAT_VERSIONS` becomes a major → highest-known-minor mapping.
    An unknown major is rejected; a newer minor within a known major is read
    with a warning. It was previously `frozenset({"0.1"})`, exact-set
    membership over opaque strings, against which §21's "reject unsupported
    *major* versions, MAY support older *minor* versions" was not implementable
    — and which had no test coverage at all.
  - A `format_version` that is not `MAJOR.MINOR` is rejected
    (`MalformedFormatVersion`, a subclass of `UnsupportedFormatVersion`: to a
    caller deciding whether it can read a release, unparseable and
    from-the-future are the same answer).
  - Migration expectations are documented against the **Provenance Amendment**
    exception CONTEXT.md already defines. Noted rather than smoothed over:
    `scripts/migrate_store_to_analyses_tsv.py` rewrites `analyses.tsv` in place,
    which is outside that exception. It predates the policy and its targets are
    stores that should be rebuilt instead; bringing it into line is a change of
    its own.
- CI (`.github/workflows/ci.yml`): tests, tooling baselines, and a changelog
  gate on every pull request and push to `dev`/`main`. The repository had no CI
  at all before this.
- A pull-request template carrying the correctness, test and documentation
  checklists from `CONTRIBUTING.md`.
- `CLAUDE.md` — a short orientation file for AI coding sessions, pointing at the
  same standards rather than restating them.

### Changed

- **Ancestry assignment reads any Source Format, not only GWAS-VCF** (#115).
  `assign_from_source` resolves the reader through
  `opengwasdb.readers.registry`; a source manifest may now carry an optional
  `source_reader_capability` column, defaulting to `opengwasdb.gwas-vcf`. The
  old restriction is why `GCST003566` was never examined: the
  `gwas-catalog-eur-hybrid` family is harmonised GWAS-SSF, so every Analysis in
  it carries `ancestry_assignment_method=source_trusted_no_af`. The Catalogue
  carries the capability through, so it can now drive a build of those sources.
- **`validate` now rejects a store that carries EAF it never checked** (#115).
  Every Analysis with `eaf_scope=association` must record EAF orientation
  evidence; `unverified` is reported as a warning. Deliberate: a frequency
  column nobody has checked is indistinguishable from one reported against the
  other allele. It invalidates any Store Release built between ADR 0036 (which
  began retaining EAF) and this change, until it is rebuilt or audited with
  `audit-eaf-orientation` — but no such release exists today: every pilot on
  disk predates ADR 0036 and carries no `eaf_scope` column at all, so the rule
  does not apply to them.
- The Analysis Catalogue gains `eaf_orientation`, `eaf_orientation_r`,
  `source_reader_capability` and `gate_orientation_flip_r` columns (#115), and
  `assign-ancestry` gains `--orientation-flip-r`. All are annotation columns:
  the build-manifest superset invariant (`BUILD_COLUMNS` first, in order) is
  unchanged. Recalibrating an older Catalogue appends the new gate column
  rather than failing on it.
- `analyses.tsv` gains three columns (`eaf_orientation`, `eaf_orientation_r`,
  `eaf_orientation_n`, #115). Store-only: a release manifest neither carries
  nor needs them, and `opengwasdb-stores` accepts them through the schema's
  existing superset property (nothing there enumerates the column list). Its
  build generators, however, invoke the CLI: they should start passing
  `--eaf-reference` so pilot rebuilds are verified rather than `unverified`.
  `GCST003566` itself is already excluded from the EUR hybrid pilot there, with
  the evidence recorded in `inclusion_reason` (`opengwasdb-stores` eefcc81).
- `CONTRIBUTING.md` gains a Documentation section: what goes stale, what
  regenerates it, that benchmark numbers are re-run rather than edited, that
  `opengwasdb-stores` holds docs depending on this package's CLI surface, and
  a pre-merge checklist.

### Fixed

- **Reference Completion refuses a panel no Analysis matches, instead of
  completing to nothing** (#98, ADR 0039 §2). `derive_impute_analysis_ids`
  compared the `--ancestry` flag to each Analysis's `assigned_ancestry` by
  string equality, and the two carry different vocabularies — panel
  directories are named `EUR`, the registry may record `European`. A store
  recording the word therefore matched **zero** Analyses, and completion ran to
  a successful finish, produced a release stamped `reference_completed`, passed
  `validate_store()`, and reported `0 imputed` in a line that reads as "nothing
  was imputable". The LD work had all succeeded: `completion_quality`, written
  before the filter applies, held correlations up to 0.97 and 51M+ imputable
  cells across 1,357 blocks. Two changes:
  - spellings of one ancestry are reconciled through an explicit alias table,
    matched exactly on a normalised label, in either direction. Not through
    `ancestry.routing.reported_to_superpop`, whose ordered substring matching
    answers `AFR` for `"North African"` — fine for guessing at a cohort's
    free-text description, not for deciding which Analyses a panel may impute.
    A panel named outside the vocabulary still matches on exact equality;
  - a genuinely empty match raises `AncestryFilterError`, naming the panel, the
    values the store holds, and whether the panel's own name was understood.
- **Ragged Reference Completion no longer does nothing for a Store Family with
  no gene target** (#102, ADR 0039 §1, spec §17). Regions were identified only
  from a cis window around `trait_chr`/`trait_bp`, so an Analysis without one —
  small-molecule metabolomics, by design — was silently passed through: 0
  blocks enumerated, 0 imputed, a "completed" store byte-identical to its
  source. All four `metabolome-plasma-2023` full releases (4,443 Analyses)
  completed this way. Such an Analysis's regions are now the LD blocks it
  already holds enough observations in.
  - "Enough" is `impute.min_observed_points()` (4), the number the imputation
    gate actually enforces: `poly_rescale` returns NaN below it and
    `impute_z_block` then rejects the block, so a lower threshold enumerates
    blocks that add panel variants as missing rows and impute none of them.
    The constant is exported from `impute` rather than restated.
  - The count is over the Analysis's observations **at the block's own panel
    variants**, which is what `run_block` fits on — not over its variants
    falling inside the block's base-pair extent, which an off-panel variant
    inflates.
  - The panel is read once per completion, one chromosome at a time: holding a
    genome-wide panel's blocks resident costs roughly a gigabyte, and these are
    the families that span the genome.
  - `RaggedCSRReader.variant_indices()` exposes an Analysis's variant footprint
    without decoding its statistics.
- **Reference Completion could stamp a `format_version` onto arrays it had not
  encoded that way** (#112). All three completion paths copied the source's
  `format_version` into the completed release, which is the right rule —
  completion writes into the source's arrays and therefore its encoding — but
  did it incidentally. The moment `CURRENT_FORMAT_VERSION` moves ahead of a
  store on disk (which #114 does), that produces a release that lies about its
  own encoding. Completion now refuses a source it can read but cannot write,
  and does so *before* the imputation rather than at manifest-write time.
  Nothing can reach it today; that is why it was worth adding now.
- `benchmarks/README.md` documented seven commands as `uv run`, which stopped
  working when the project moved to Pixi.
- `README.md` described a two-month-old codebase as "newly scaffolded".

## [0.2.0] — 2026-08-22

The first tagged version. `0.1.0` was the placeholder the project carried from
its first commit and was never released, so this entry backfills the changes a
user would notice across the whole history to date.

### Added

- **Effect allele frequency is stored** (ADR 0036, #106). Every source format
  reported it and every reader parsed it; no layout kept it. Now a per
  (variant, Analysis) plane in Dense, Ragged and Hybrid, oriented to the stored
  effect allele, declared per Analysis by a new `eaf_scope` column, and exposed
  through the query facade and `--variant-info`.
- **rsid lookups work** (#109). An rsid search index is written by
  `write_variant_axis`, so every layout gets one. Collision policy is explicit:
  an rsid names every row it appears on.
- **Human-readable query output by default** (#104), with `--variant-info` for
  rsid and eaf, and `--format json` for the raw index-keyed result.
- **`overview.html`** — a generated, store-wide browsable rendering of
  `analyses.tsv` with persisted Top-Hit Counts (ADR 0032).
- **Hybrid layout** — a Dense Component over a reference panel plus a Ragged
  Overflow Component for off-panel observations (ADR 0026).
- **Reference Completion** for Dense, Ragged and Hybrid, with ancestry-matched
  imputation (ADR 0028) and per-block quality records.
- **Ancestry assignment from allele frequencies** (ADR 0029) and the Analysis
  Catalogue ingestion hub (ADR 0027).
- **Rho matrix** for Dense stores (ADR 0025).
- Validation: a closed-envelope rule (#80), rsid index coverage, and EAF range
  and shape checks.

### Changed

- **`analyses.tsv` is the single Analytical Metadata contract for every layout**
  (ADR 0030, ADR 0034). `phenotype_id`/`phenotype_label`/`trait_id` retired in
  favour of `analysis_label`/`trait_ontology_id`/`trait_ontology_label`;
  `gene_id`/`gene_name` retired too (ADR 0035). Ragged's divergent SQLite
  `analyses` table is gone — a leftover one is now a validation failure.
- **Manifest metadata reaches the store.** Dense and Hybrid in #86, Ragged in
  #83 — fifteen shared-core columns that manifests supplied and builders
  silently dropped, now carried through one shared `PassthroughMetadata`.
- The `SourceReader` interface carries variant identifiers (`SourceVariant`)
  and per-association EAF, both of which sources provided and the interface
  previously had nowhere to put.
- Query results are a documented adapter contract across layouts (ADR 0033).

### Fixed

- **Hybrid `analyses_table()` undercounted Top-Hit Counts** by delegating to the
  Dense Component (#107). Hid 4,476 hits on the gwas-catalog pilot; one Analysis
  reported 10,727 instead of 14,706.
- **Ragged and Hybrid Reference Completion silently dropped data** on rebuild —
  rsids (#109) and observed EAF (found in review of #106). The eqtlgen pilot had
  49,967 rsids observed and none in its completed sibling.
- **Dense and Hybrid never captured rsids at all** (#109) — 0 in 50,000 rows of
  both the finngen and gwas-catalog pilots.
- Disjoint-partition violation on Hybrid LD-panel extension (#99).
- Duplicate canonical-variant rows within a Ragged Analysis (#101); the
  `trait_id` requirement dropped from the Ragged SSF builder (#100).
- Reference completion now warns when no Analysis carries `assigned_ancestry`
  and it would otherwise impute everything against one panel (#108).

### Known limitations

- Compressed sizes and the overflow table's real cost are measured in ADR 0037
  on pilot data, not yet on a rebuilt genome-scale store (#117).
- `eaf` is still `float32` and `se` is still `float16` per stored cell; the
  residual encodings that shrink them are #116 and #118, and the plan carries
  only `z` and `se` until then.
- Reference-Completed releases carry no EAF on imputed cells (#113, #116).
- Existing Store Releases predate every build-time fix above and must be
  rebuilt to gain them (#117).

## Store format compatibility

The package version and the store `format_version` are independent. A store
records the `format_version` it was written against; the package records which
it can read.

| package | writes `format_version` | reads |
|---|---|---|
| 0.2.0 | 0.1 | 0.1 |
| unreleased (`dev`) | 2.0 | 0.x, 1.x, 2.0 |

[Unreleased]: https://github.com/opengwas/opengwasdb/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/opengwas/opengwasdb/releases/tag/v0.2.0
