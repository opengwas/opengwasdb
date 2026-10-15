# Resolving BESD Analyses

`opengwasdb resolve-analyses` resolves each probe of a BESD file set as an
Analysis (issue #282, ADR 0062). A BESD file set is three files beside one
prefix: `<prefix>.esi` (one row per variant), `<prefix>.epi` (one row per
probe) and `<prefix>.besd` (each probe's betas and SEs).

## Declaring a probe

One manifest row per probe:

| column | value |
|---|---|
| `source_reader_capability` | `opengwasdb.besd` — required; a prefix has no extension to default from |
| `source_file` | the file-set prefix, without `.esi`/`.epi`/`.besd` |
| `source_probe_id` | the `.epi` probe ID; required for BESD, refused for any other format |
| `sample_size` | the Analysis's N; required when its SD tier estimates one |
| `source_genome_build` | as for every format; GRCh37 needs `--chain-file` |
| `stored_effect_scale`, `original_sd_method` | as for every format |

Several rows may name the same file set; each is its own Analysis and record.
A declared INFO filter is refused for BESD, as for every non-GWAS-SSF format.

## What is measured, and where

**Ancestry and orientation: once per file set.** The `.esi`'s seventh column is
the frequency of `a1`, the effect allele. Before any worker starts, the parent
reads each file set's `.esi` once, aligns every row to the reference build
(lifting GRCh37 rows by the GWAS-VCF projection: one unique mapping per variant,
alleles complemented on a reverse-strand mapping, indels and ambiguous mappings
refused), orients each frequency to canonical A1, and fits the Ancestry
Reference Panel. A row contributes a site when its frequency is usable
(`stats.parse_af`: inside (0, 1)), its alleles are not a palindromic pair, and
its ALID is on the panel; a later row wins a repeated ALID. Unlike a tabular
row, no SE is required: the frequency belongs to the variant, not to an
association.

Every probe's record carries that one fit unchanged — `assigned_ancestry`, the
compositions, the gate outcome and `eaf_orientation`/`eaf_orientation_r` —
with:

- `diagnostics.ancestry_scope = "source_file"` (every other format records
  `"analysis"`);
- `diagnostics.ancestry_measurement_id`, a digest of the member files' sizes
  and mtimes, the builds and chain, the panel and groups, the extraction panel,
  the gates, the scan bound and the code version — equal on every probe that
  shares the measurement;
- `ancestry_sites`, `ancestry_rows_read`, `ancestry_stop_reason` and
  `ancestry_reference_rows_matched` describing the `.esi`.

`index.json` lists each measurement once under `source_measurements`, with the
Analyses that used it in this invocation.

**Phenotype SD: per probe.** Each probe's slice of the `.besd` is read by
seeking, joined to the `.esi` identity and frequency arrays, and passed through
the ordinary resolver: the same evidence sample, method tiers and estimator.
`rows_read` and every build-eligible count are the probe's own. A beta must be
finite and an SE finite and positive, or the value is missing. A probe row whose
`.esi` variant did not align is left out, as every reader leaves out a row that
names no canonical variant.

## Refusals

Each is a `controlled_failure` record with `phenotype_sd` null. A probe refused
for its own reasons still carries its file set's measurement — the ancestry fit,
`ancestry_scope = source_file`, the shared `ancestry_measurement_id` and the
`.esi` counts — because every measurement is recorded whether the Analysis
passes or not (stores ADR 0036). When the file set's measurement itself failed,
`ancestry` is null and the diagnostics name the attempted measurement under
`source_file` scope with zero counts.

- **No frequencies.** A `.esi` with no usable frequency on any row — no seventh
  column, or `NA` throughout — fails every probe in its file set. A file whose
  frequencies are partly `NA` is measured from the rows that have one.
- **No sample size.** A probe whose `original_sd_method` estimates an SD
  (`estimated_from_*`, on a quantitative scale) and whose row has no usable
  `sample_size`. BESD records no per-variant N, so nothing can stand in for it.
  Only tiers that estimate an SD are refused: a `declared_standardised`,
  `source_provided` or case-control probe needs no N.
- **A file set that cannot be read or aligned**: a missing member, a malformed
  `.esi`/`.epi` row (`BESDMetadataError`), a repeated probe ID, an unsupported
  build pair, or an indel under a lift. Every probe in it fails with the reason.
- **A probe the `.epi` does not list**, corrupt column offsets for it (a
  negative start, or an SE block before the beta block), or a `.besd` slice
  past the end of the file or naming a row past the end of the `.esi`. That
  probe fails; the file's other probes resolve.
- **A missing or blank `source_probe_id` on a BESD row**, or one set on any
  other format's row, is a manifest error: the command fails before resolving.

## Resume

`source_file` is a prefix, so a BESD record's fingerprint also binds each member
file's size and mtime (`source_member_files`) and `resolution_config.
source_probe_id`. A rewritten `.esi` invalidates every probe of its file set.
