# Getting started

This page takes a fresh checkout to a local Store Release, validates it, and runs the query commands used by the rest of the documentation. It deliberately uses an in-repository fixture so the first run needs no production data or machine-specific paths.

## Install the development environment

Install [Pixi](https://pixi.sh), then run commands through the Pixi environments declared in `pyproject.toml`:

```bash
pixi run -e dev test        # pytest
pixi run -e dev lint        # ruff check .
pixi run -e dev typecheck   # mypy opengwasdb
```

The `dev` environment includes Python dependencies plus native tooling such as `bcftools`. The separate `report` environment is only needed for rendering benchmark Quarto documents:

```bash
pixi run -e report quarto render docs/benchmark-output/opengwasdb_vs_besdq_comparison.qmd
```

For ad hoc Python commands, use the managed interpreter rather than a bare system `python`:

```bash
pixi run -e dev python -c "import opengwasdb; print(opengwasdb.__version__)"
```

## Build a tiny Dense Store Release

The tiny source file is `docs/examples/tiny-associations.tsv`. It has four observed associations across two Analyses and three variants. It is intentionally small; production Store Releases are normally built from GWAS-VCF, GWAS-SSF, BESD, or the `opengwasdb-stores` generators.

```bash
rm -rf /tmp/opengwasdb-tiny.opengwasdb
pixi run -e dev opengwasdb build-dense \
  docs/examples/tiny-associations.tsv \
  /tmp/opengwasdb-tiny.opengwasdb \
  --store-id tiny \
  --release-id observed-v1
```

Expected shape:

```json
{"n_analyses": 2, "n_variants": 3, "output_path": "/tmp/opengwasdb-tiny.opengwasdb"}
```

## Validate and inspect the Store Release

Always validate a Store Release before treating query results as evidence:

```bash
pixi run -e dev opengwasdb validate /tmp/opengwasdb-tiny.opengwasdb
```

Expected output:

```text
valid
```

Inspect the release envelope and declared encodings:

```bash
pixi run -e dev opengwasdb info /tmp/opengwasdb-tiny.opengwasdb
```

You should see `format_version: 0.1.0`, `primary_layout: dense`, and `completion_state: observed_only`.

## Query by variant

`query-phewas` extracts one variant across all Analyses. Identifiers may be rsids or canonical ALIDs.

```bash
pixi run -e dev opengwasdb query-phewas \
  /tmp/opengwasdb-tiny.opengwasdb \
  rs111 \
  --variant-info
```

Expected rows:

```text
analysis_id	analysis_label	rsid	chromosome	position	alid	effect_allele	other_allele	z	se	p	eaf	association_status
height_eur	Height EUR pilot	rs111	1	100	1:100:A:G	A	G	2	0.0999756	0.0455	.	observed
ldl_eur	LDL EUR pilot	rs111	1	100	1:100:A:G	A	G	-6	0.199951	1.97e-09	.	observed
```

The `.` in `eaf` is intentional for this fixture: it carries no Effect Allele Frequency plane, so the query returns absence rather than fabricating a default.

## Query by exact variant and Analysis

`query-lookup` takes comma-separated variants and comma-separated Analysis IDs. The second query uses the canonical ALID after allele normalisation; the source row at position 200 is stored as `1:200:C:T` and its signed `z` is negated because the source effect allele was not canonical A1.

```bash
pixi run -e dev opengwasdb query-lookup \
  /tmp/opengwasdb-tiny.opengwasdb \
  1:200:C:T \
  height_eur \
  --variant-info
```

Expected row:

```text
analysis_id	analysis_label	rsid	chromosome	position	alid	effect_allele	other_allele	z	se	p	eaf	association_status
height_eur	Height EUR pilot	rs222	1	200	1:200:C:T	C	T	-6	0.119995	1.97e-09	.	observed
```

## Query top hits

The tiny Dense builder writes the top-hit index as part of the build. The default threshold is genome-wide significance, `p <= 5e-8`.

```bash
pixi run -e dev opengwasdb query-top-hits \
  /tmp/opengwasdb-tiny.opengwasdb \
  --limit 3 \
  --variant-info
```

Expected rows:

```text
analysis_id	analysis_label	rsid	chromosome	position	alid	effect_allele	other_allele	z	se	p	eaf	association_status
height_eur	Height EUR pilot	rs222	1	200	1:200:C:T	C	T	-6	0.119995	1.97e-09	.	observed
ldl_eur	LDL EUR pilot	rs111	1	100	1:100:A:G	A	G	-6	0.199951	1.97e-09	.	observed
```

## Indexed Variant Subsets (fast repeated extraction)

An **Indexed Variant Subset** is an optional, rebuildable Dense query index over
a caller-supplied list of canonical ALIDs (ADR 0053). It covers every Analysis in
the Store Release and holds the full available statistics (Z, SE and EAF, with
the release's Association Status for a Reference-Completed store), laid out
Analysis-major. It is derived data: adding or deleting it changes no association
and no Analytical Metadata.

Build one from a list of canonical ALIDs (one per nonblank line) and the list's
Reference Assembly. The assembly must match the release's, because cross-Store
Variant Identity is assembly plus ALID:

```bash
pixi run -e dev opengwasdb build-indexed-subset \
  /tmp/opengwasdb-tiny.opengwasdb hm3 \
  --variant-list /tmp/hm3.alid.txt \
  --reference-assembly GRCh38
```

Then read one Analysis through it — the result is identical to filtering the
ordinary `query-analysis` result to that subset:

```bash
pixi run -e dev opengwasdb query-analysis \
  /tmp/opengwasdb-tiny.opengwasdb height_eur --indexed-subset hm3
```

`--overwrite` atomically replaces an existing subset of the same name.
`--band-cells` bounds peak build memory (the complete write is staged, validated
against the primary planes, and published by rename). `opengwasdb
regenerate-overview STORE` lists every published subset — name,
requested/resolved/absent counts, input checksum, Reference Assembly and build
provenance read from the group's own metadata, with the physical size computed
from the group's files at render time. An unreadable group, or one whose
recorded name disagrees with its directory, is surfaced as an invalid entry
rather than silently omitted.

Failure behaviour is deliberate and loud:

- a malformed, repeated or non-canonical ALID fails the build, naming the line;
- a Reference Assembly mismatch, a subset name outside
  `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`, or a list matching no Store variant is
  refused before anything is written;
- ALIDs absent from the Store Variant Table are **counted and recorded, never
  treated as present**;
- `--indexed-subset` on an unknown, incomplete, stale, corrupt or
  non-Dense subset stops with an error naming the Store and subset. It never
  falls back to the ordinary query path, because a silent fallback would return
the slow answer while appearing to be indexed;
- a query without `--indexed-subset` is unchanged and consults no subset state.

## Where to go next

- Store contract: [`docs/spec/store-format.md`](spec/store-format.md)
- Domain glossary: [`CONTEXT.md`](../CONTEXT.md)
- Design decisions: [`docs/adr/`](adr/)
- Benchmark commands: [`benchmarks/README.md`](../benchmarks/README.md)

The broader query walkthrough and production Store Release catalogue live in the sibling `opengwasdb-stores` repository. If a change alters CLI output, manifest fields, or `analyses.tsv` columns, update that documentation too.

## Troubleshooting

### `pixi` is not found

Some machines install Pixi under `~/.pixi/bin` without adding it to `PATH`:

```bash
export PATH="$HOME/.pixi/bin:$PATH"
```

### A command works with `pixi run -e dev` but not with `python` or `opengwasdb`

Use the Pixi-managed commands unless you have installed the package and native tools yourself. `bcftools` is supplied by the Pixi environment, not by `pip install -e .`.

### A variant query returns no rows

No rows can mean several different things:

- the variant identifier is not present in this Store Release;
- the Analysis did not retain that association, especially in a `cis_and_signals` Store Release;
- the Store Release is Reference-Completed and the requested observed-only mode excludes imputed associations;
- the identifier is an ALID in a different allele order or Reference Assembly.

Use `--variant-info` while debugging so the returned chromosome, position, alleles, rsid, and ALID are visible.

### Effect Allele Frequency is `.`

`.` means absent. It is different from `0`, `0.5`, or a panel frequency. Observed associations whose source reported no EAF remain absent; Reference EAF is only reported for imputed associations when the completed release carries it.

### Reference completion imputes zero Analyses, or every Analysis

Completion is ancestry-matched when `assigned_ancestry` is populated. A mismatch between the `--ancestry` value and the Analysis metadata can exclude every Analysis; a Store Release with no assigned ancestry cannot prove the chosen LD Reference Panel matches its Analyses. Current code surfaces these cases, but the safe response is to inspect `analyses.tsv` and the completion summary before using the completed release.

### Benchmark numbers changed

Benchmark artifacts are measurements. Re-run the relevant `benchmarks/*.py` script and render the corresponding Quarto document with the `report` environment; do not hand-edit benchmark numbers.
