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

### Added

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
| unreleased (`dev`) | 1.0 | 0.x, 1.0 |

[Unreleased]: https://github.com/opengwas/opengwasdb/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/opengwas/opengwasdb/releases/tag/v0.2.0
