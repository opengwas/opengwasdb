# Registry-orchestration seam, canonical manifest column aliases, and per-release option precedence

Records the boundary between `opengwasdb` and the `opengwasdb-stores` registry
(epic #177), establishing shared canonical column alias resolution (#172) and
per-release builder option precedence (#174).

## Context

`opengwasdb-stores` is an orchestration repository whose only responsibility is
issuing `opengwasdb` command lines. It should never read, mutate, or synthesize
association data or manifest rows. Prior to epic #177, several seams forced the
registry into manifest synthesis, column projection, and redundant parsing:

1. **Manifest column vocabulary (#172)**: Dense and Hybrid builders accepted
   canonical `analyses.tsv` column names (`analysis_id`, `source_file`,
   `analysis_label`, `sample_size`) after issue #170, but Ragged SSF still
   required legacy names (`n`, `filtered_file`), and alias resolution was
   duplicated across builder modules.
2. **Per-release constants as row columns (#174)**: Per-release configuration
   such as `source_reader_capability` and `source_assembly` were only readable
   as per-row manifest columns. For homogeneous collections (e.g. FinnGen R13 or
   UK Biobank), the registry had to materialize a derived manifest with
   identical columns on every row purely to reach builder defaults.

## Decision

### 1. Shared manifest column alias resolver (`opengwasdb.model.manifest_columns`)

All builders and pipelines resolve manifest column names through a unified
model:

- **Canonical precedence**: When both canonical and legacy column names are
  present in a manifest header, the canonical name always wins.
- **Multiple legacy aliases**: Canonical names may map to multiple legacy
  spellings in fixed precedence order (for example, `source_file` resolves
  from `source_file`, then `file_path`, then `filtered_file`).
- **Path resolution**: In Ragged SSF builds, absolute `source_file` paths are
  used directly; relative paths are joined to `--filtered-dir`.
- **Label semantics**: Present-but-blank `analysis_label` values are preserved
  as empty strings in Dense `analyses.tsv` (ADR 0034), while Ancestry pipeline
  manifest reading retains its fallback to `analysis_id`.

### 2. Per-release CLI option precedence

`build-dense-vcf` and `build-hybrid` accept optional `--source-reader-capability`
and `--source-assembly` flags. Precedence is strictly:

$$\text{per-row manifest column} > \text{CLI option} > \text{hardcoded default}$$

- **Per-row override**: If a manifest row carries an explicit column value, it
  takes precedence over the CLI option. Disagreement between a row and the CLI
  default is valid and not an error.
- **Parse-time validation**: Invalid CLI values fail at argument parse time with
  informative errors. Reader capabilities are validated against
  `opengwasdb.readers.known_capabilities()`, and genome builds are validated
  against `opengwasdb.build.liftover.normalise_build()`.
- **Defaults**: When both row column and CLI option are omitted, builders retain
  their established defaults: `opengwasdb.gwas-vcf` for capability and `hg19`
  for source assembly.

## Consequences

- The registry no longer needs to synthesize derived manifests with constant
  columns for homogeneous collections.
- All builders consume canonical `analyses.tsv` files directly.
- Existing manifests with legacy column spellings or per-row overrides continue
  to work without modification.
