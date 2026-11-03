# Analytical Metadata travels with the Store Release; `analyses.tsv` replaces the SQLite `analyses` table

A Store Release is defined (CONTEXT.md) to remain interpretable without a
catalogue service — but the store's own schema has never carried more than
`stored_effect_scale`. Everything else needed to interpret an association
(Assigned Ancestry, sample-size structure, the effect-scale provenance and
dispersion diagnostic from ADR-0029, a Reference Completion Quality summary) has
only ever been recorded upstream, in an ingest manifest or the `opengwasdb-stores`
registry — so a downloaded store has never actually met its own self-containment
contract for these fields.

## Decision

Every Store Release carries its own **Analytical Metadata** — metadata that
affects the interpretation of association statistics, matching the term as
already defined in `opengwasdb-stores`' `CONTEXT.md` (see also `Store Release`
below). Placement is decided by size and access pattern, not by ranking which
fields matter most:

- **Small, per-Analysis facts** — Assigned Ancestry, Ancestry Composition
  (as wide `ancestry_prop_*` columns, small enough at ~21 reference populations
  to stay flat rather than a separate table), sample-size kind/scope/counts,
  `original_effect_scale`/`original_sd`/`original_sd_method`/dispersion, and a
  Reference Completion Quality rollup — live in a new `analyses.tsv` at the store
  root. This file is the **sole source of truth** for Analysis metadata:
  `index.sqlite`'s existing `analyses` table is removed, not extended. Every
  current use of it is either a full-table scan materialised into a Python dict
  or a single indexed point lookup by `analysis_id` (`opengwasdb/index/sqlite.py`,
  `opengwasdb/query/facade.py`); both translate directly to reading `analyses.tsv`
  into a dict once at store-open, at equal or lower cost, and no SQL `JOIN`
  anywhere in the codebase touches `analyses`.
- **Large, fine-grained, tooling-only data** — `completion_quality` at
  LD-block-by-Analysis granularity (millions of rows for a genome-wide store) —
  stays SQLite-only. `analyses.tsv` carries only its per-Analysis rollup.
- A generated, store-wide `overview.html` (a searchable/sortable table over
  `analyses.tsv`) is added as a human-browsable artifact.

`analyses.tsv` and `overview.html` are both build-time-derived outputs, written
once by the builder and never independently authored or edited — there is exactly
one place Analysis metadata is written, so there is nothing to keep in sync.

## Considered options

- **Keep `analyses.tsv` as an export of an unchanged SQLite `analyses` table.**
  Rejected: two persisted copies of the same small data inside one release, with
  no consumer that needs the SQL copy — pure duplication for no functional gain.
- **Extend the SQLite `analyses` table only; no flat-file export.** Rejected:
  fails the self-contained/human-accessible goal — a downloader would need a
  SQLite client and the schema knowledge to interpret their own store.
- **A forced boundary between "interpretation-critical" and "audit-trail-only"
  metadata, with only the former in-store.** Rejected: no consumer is well served
  by the store withholding evidence (e.g. NNLS residuals, dispersion inputs) that
  it has no cost reason to omit; the size/access-pattern split above achieves the
  same schema simplicity without deciding whose judgement of "important" wins.

## Consequences

- `docs/spec/store-format.md` needs updating: the release envelope (§1), the
  `analyses` table definition, and the validation rules (§20) all currently
  predate this decision.
- Ragged/BESD stores (`opengwasdb/layouts/ragged/build_besd.py`) had their own,
  divergent `analyses` schema with no `stored_effect_scale` at all. They were
  brought onto this contract by [ADR 0034](0034-unify-analysestsv-across-layouts-retire-phenotype-id.md)
  (issues #69–#73): every Ragged release carries `analyses.tsv` and no
  `analyses` table.
- `completion_quality.analysis_index` becomes a positional reference into
  `analyses.tsv` rather than a SQL foreign key, since SQLite cannot enforce a
  constraint across a file boundary; a validator rule must check this instead of
  the database engine.
