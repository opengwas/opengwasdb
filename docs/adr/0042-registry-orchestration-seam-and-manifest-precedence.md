# Registry-orchestration seam, canonical manifest column aliases, per-release option precedence, machine-readable evidence, and phenotype-SD computation

Records the boundary between `opengwasdb` and the `opengwasdb-stores` registry
(epic #177), establishing shared canonical column alias resolution (#172),
BESD metadata overlay (#173), per-release builder option precedence (#174),
machine-readable validation evidence (#175), and phenotype-SD estimation from a
canonical manifest (#176).

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
2. **Missing BESD manifest input (#173)**: `build-ragged-besd` derived all
   analysis metadata from `.epi` probe records alone, with no mechanism to
   supply registry-side Analytical and Attribution Metadata (e.g. for eQTLGen).
3. **Per-release constants as row columns (#174)**: Per-release configuration
   such as `source_reader_capability` and `source_assembly` were only readable
   as per-row manifest columns. For homogeneous collections (e.g. FinnGen R13 or
   UK Biobank), the registry had to materialize a derived manifest with
   identical columns on every row purely to reach builder defaults.
4. **Evidence scraping from prose (#175)**: `validate` and `info` emitted only
   human-readable text, forcing downstream tools to parse English prose off
   stdout/stderr to populate registry `validation.yaml` records.
5. **Re-implemented source reading for phenotype SD (#176)**: the registry's
   `phenotype_sd_estimate.py` shim took JSON `se`/`af`/`beta` arrays on the
   command line, so every caller had to open the source file and align alleles
   itself -- re-implementing what `SourceReader.extract_at_sites` already does,
   once per Store Family, alongside a second allele-harmonisation
   implementation in R.

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

### 2. BESD Analysis metadata overlay (`build-ragged-besd --analyses`)

`build-ragged-besd` accepts an optional `--analyses <analyses.tsv>` manifest:

- Joined to EPI probe analyses by `analysis_id`.
- Mismatches in either direction (extra manifest IDs not in BESD, or BESD probes
  missing from the manifest) fail loudly naming all offending IDs.
- BESD `.epi`-derived genomic coordinates (`trait_chr`, `trait_bp`) remain
  authoritative over manifest values.
- Shared-core Analytical Metadata (`PassthroughMetadata`, `assigned_ancestry`,
  `sample_size`, `stored_effect_scale`) and Attribution Metadata are overlaid
  onto the generated `analyses.tsv`.
- When omitted, output is byte-identical to prior builds.

### 3. Per-release CLI option precedence

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

### 4. Machine-readable validation and inspection evidence

`validate` and `info` support `--format json`, defaulting to human-readable text:

- `validate --format json` outputs a single structured JSON object
  `{"ok": boolean, "errors": list[str], "warnings": list[str]}` on stdout with
  empty stderr on success. If validation fails (`ok=false`), the JSON object is
  emitted on stdout and the process exits with a non-zero exit code.
- `info --format json` outputs a single structured JSON object containing core
  manifest metadata and a decomposed, structured `encoding` dictionary
  (`z`, `se`, `eaf`), avoiding composed human strings.

### 5. Phenotype-SD estimation (`estimate-phenotype-sd`) and the computation/acceptance split

`estimate-phenotype-sd <analyses.tsv> <out.tsv>` is the one seam that joins
`opengwasdb.build.phenotype_sd.estimate_phenotype_sd`'s pure computation
(ADR 0029) to real sources:

- It reads the canonical `analyses.tsv` through the shared alias resolver,
  resolves a `SourceReader` per row through `opengwasdb.readers.registry`
  (every registered capability; an unknown one fails naming the known set),
  and reads each source once for the `se`/`af`/`beta` evidence the row's
  caller-chosen `original_sd_method` needs -- no arrays on the command line and
  no caller-side allele alignment.
- `--af-source source` uses the reader's own A1-oriented frequency;
  `--af-source reference` substitutes an A1-oriented reference frequency
  (through the same `iter_eaf_reference` machinery the build-time orientation
  check uses).
- The output TSV carries the shared-core `analyses.tsv` spellings
  (`analysis_id`, `original_sd`, `original_sd_method`, `original_sd_dispersion`,
  `notes`) so the caller merges a column rather than translating a table. A
  missing or unusable sample size writes `original_sd_method=unavailable` with
  a blank `original_sd`, never a fabricated value. Results are order-preserving
  and independent of `--n-workers`.
- **The computation/acceptance split**: the command computes the number. It does
  not choose the method tier, the tolerance, or whether a disagreement blocks a
  release. The manifest's `original_sd_method` is read, never inferred, and a
  tier this command does not compute fails loudly rather than being guessed at;
  those remain registry acceptance policy.

## Consequences

- The registry no longer needs post-build patching to attach Analytical and
  Attribution Metadata to BESD-built stores.
- The registry no longer needs to synthesize derived manifests with constant
  columns for homogeneous collections.
- All builders consume canonical `analyses.tsv` files directly.
- Registry `validation.yaml` can be populated directly from machine-readable JSON
  output without prose scraping.
- Effect-scale estimation is reachable from a canonical `analyses.tsv` with no
  caller-side source reading or allele alignment, and its output drops into the
  same shared-core columns the builders write.
- Existing human-readable text output remains byte-for-byte identical.

