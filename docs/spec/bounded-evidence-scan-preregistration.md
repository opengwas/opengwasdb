# Preregistration: Bounded-Prefix and Compiled-Parser Acceleration for Resolver Evidence Scans

**Document status:** Locked before result inspection (issue #209). Results are
recorded separately in [`bounded-evidence-scan-report.md`](./bounded-evidence-scan-report.md),
generated from the committed JSON artifacts once the study has run.
**Issue:** [opengwasdb#209](https://github.com/opengwas/opengwasdb/issues/209)
**Depends on:** opengwasdb#207 / ADR 0044 (`resolve_analysis`), opengwasdb#208
(`resolve-analyses`), and the full-reference policy study in
[opengwasdb-stores#152](https://github.com/opengwas/opengwasdb-stores/issues/152).

---

## 1. Context

The Phase B resolver (`opengwasdb.build.resolve.resolve_analysis`, ADR 0044)
reads each compressed GWAS-SSF source once and accumulates both the ancestry
fit's frequencies and the phenotype-SD evidence in that pass. It still
decompresses the whole source and normalises every row in Python.

The full-reference policy study (opengwasdb-stores#152) rejected the fixed
10,000-site extraction panel: genome-wide assignment concordance was 95.10%,
below the locked 98% threshold. That decision does **not** establish that every
row of a sorted GWAS file must be read. This study asks whether a deterministic
prefix of the source -- fixed raw rows, or a fixed number of usable
ancestry-reference matches -- preserves the full-scan resolution, and separately
whether a faster compiled parser exists that preserves GWAS-SSF semantics.

Two effects are measured separately: **early termination** and **parser
throughput**.

## 2. Frozen inputs

| Resource | Role | Location | SHA-256 |
|---|---|---|---|
| Evaluation manifest | 106 Analyses, stratified | `docs/benchmark-output/opengwasdb_resolver_evidence_scan_manifest.tsv` | `962618a5f72e0fa6027173e7b0b102cba472c9b4c9ff460f7bada04f17c2d557` |
| Ancestry Reference | 5,808,902-site panel | `/data/opengwasdb/reference/ancestry-mixture/ref_freqs.hg38.tsv.gz` | `067914e0c21fe2f0d464f8489e7d91bd4eeafdbf71bc0b93e895a2117ddc0925` |
| Group map | fine → super-population | `/data/opengwasdb/reference/ancestry-mixture/ancestry_groups.tsv` | `005093f9a8f74cc792e6ee4e828f7f4fa1ca7508e92c2d1dd463f320bac3834d` |
| Full-scan comparator | per-Analysis `full_*` records | `/data/opengwasdb/work/gwas-catalog-eur-hybrid/concordance/concordance_results.json` | `609fff779ad148a37af4aa397af801505ea54578aa295b11f9c434a48dcafb23` |

The manifest is the frozen 106-Analysis stratified sample from opengwasdb-stores#152
(50 quantitative decile rows, 50 case-control decile rows, 6 explicit edge
cases). It is copied here verbatim so the evaluation frame is committed with the
study; its `data_file` paths are the acquisition mirror and are read-only.

Gates are fixed at the ADR-0028 / `config-full.yaml` values: `tau=0.50`,
`delta=0.20`, `n_min=5000`, `residual_max=0.06`, `orientation_flip_r=-0.5`,
`maf_floor=0.01`, `evidence_sample=20_000`. Method tiers are the release's:
quantitative → `stored_effect_scale=sd`, `original_sd_method=estimated_from_source_maf`;
case-control → `stored_effect_scale=log_or`, `original_sd_method=binary_trait`.
No gate, tier or reference is retuned at any point in this study.

## 3. Stopping rules tested

Every rule is applied with the same resolver, gates and references. The full
scan (`scan_limit=None`, `stop_reason=eof`) is the comparator.

**Fixed raw-row prefixes** (`ScanLimit.max_rows`): 25,000; 50,000; 100,000;
250,000; 500,000; 1,000,000 source rows.

**Evidence-driven prefixes** (`ScanLimit.max_ancestry_sites`): stop once the
ancestry fit has accumulated 5,000; 10,000; 20,000; or 50,000 **distinct
reference sites with a usable frequency**. The counted quantity is the evidence
the NNLS fit consumes, not raw matches, so a repeated ALID cannot inflate it.

Every rule is reported, including ones that fail. No threshold is added,
removed or moved after results are seen.

## 4. Metrics recorded per (Analysis, rule)

Per Analysis and rule the harness records: `stop_reason`; source rows consumed;
usable ancestry sites; distinct chromosomes and maximum chromosome reached;
compressed source bytes (file size) and uncompressed bytes consumed
(instrumented faithful counter); wall/user/system seconds; peak RSS; the whole
`AncestryAssignment` (assigned ancestry, dominant super-population, proportions,
margin, overlap, residual, gate reason, EAF orientation outcome and `r`,
compositions); the whole phenotype-SD resolution (status, reason, SD,
dispersion, evidence considered, estimate inputs, `evidence_sampled`); and the
difference from the full-scan record.

The exact input manifest, commands, tool versions and cache condition are
recorded with the artifact.

## 5. Locked acceptance criteria

A stopping rule is acceptable only if **all** criteria hold across every
stratum (source layout, study design, size decile, AF availability, chromosome
ordering, known failure mode):

**Ancestry** (at least as strict as opengwasdb-stores#152):

1. **Assignment/gate concordance ≥ 98%** — the fraction of Analyses whose
   `assigned_ancestry` *and* `gate_reason` both match the full scan.
2. **Orientation sensitivity 100%** — every Analysis whose full scan fails the
   `eaf_orientation` gate is also reported as `eaf_orientation` by the rule.
3. **Zero false-positive EUR assignments** — no Analysis that the full scan
   leaves Unassigned (or gates out) is assigned `EUR` by the rule.
4. **Zero execution errors** — no rule introduces an error the full scan does
   not have.

**Phenotype SD** (quantitative Analyses with a full-scan estimate):

5. **Status agreement 100%** — `estimated`/`skipped`/`unavailable` and the
   reason agree exactly.
6. **Estimate difference ≤ 2%** relative difference in implied SD.
7. **Dispersion difference ≤ 2%** relative difference in dispersion.

Quantitative Analyses lacking usable source AF are reported as **controlled
exclusions** and are kept in the reported denominator rather than dropped from
it (criterion 5 still applies to them; criteria 6–7 apply where a full-scan
estimate exists).

## 6. Decision rule

Among the rules that satisfy every criterion, adopt the **smallest**: fewest
`max_rows` first, then fewest `max_ancestry_sites`, then the rule that consumes
fewest rows on the evaluation set. Ties are broken by the rule with the lower
mean wall time.

- **If a rule passes:** propose it as an explicit, non-default resolver option
  (`ScanLimit`), carrying a stopping-rule version, thresholds, parser
  implementation/version, rows and usable matches consumed, and whether EOF or
  an early-stop condition ended the scan in the resume fingerprint. Full scanning
  stays the default.
- **If no rule passes:** retain the full scan; report the negative result and
  adopt only a semantics-preserving compiled-parser improvement if the parser
  benchmark supports one.

## 7. Parser/decompressor prototypes

Measured on the same warm-cache condition, with decompression and parsing
separated:

1. `stream_projected_metrics` — current Python `gzip` + bytes projection.
2. External `gzip -dc` (argv-safe, filenames never shell-interpolated) feeding
   the same projection.
3. External `pigz -dc` feeding the same projection.
4. pandas C-engine chunked read with `usecols`, explicit types and NA handling
   (pandas is already a project dependency).
5. PyArrow `open_csv` streaming, only if an isolated import is available; never
   added as a production dependency for the experiment.
6. R `data.table::fread(cmd=...)` as a reference upper bound where R is
   available; never a production dependency.

Decompression-only throughput is measured separately for Python `gzip`,
`gzip -dc` and `pigz -dc`. A prototype that stops before EOF must reap its
decompressor and handle the expected broken pipe.

A candidate parser is admissible only if its projected rows are field-for-field
identical to `stream_projected_metrics` over parity fixtures covering:
ordinary and `hm_*` GWAS-SSF projections; reordered and extra columns;
ragged/short and quoted rows; missing, invalid and non-finite numerics;
chromosome and allele normalization; effect-allele flipping and EAF orientation;
palindromic/invalid identities; and duplicate/file-order determinism.

## 8. Non-goals

- Reconsidering the rejected 10k extraction panel without new evidence.
- Adding R or PyArrow as OpenGWASDB runtime dependencies.
- Trading false-positive EUR assignments or orientation sensitivity for speed.
- Introducing a second physical source scan for phenotype-SD evidence.
- Changing Phase B candidate acceptance or invoking Phase A Store construction.
- Mutating or re-downloading the source mirror.

## 9. Reproduction

```bash
# Ancestry prefix study (writes the JSON artifact)
pixi run -e dev python benchmarks/benchmark_resolver_evidence_scan.py \
  --manifest docs/benchmark-output/opengwasdb_resolver_evidence_scan_manifest.tsv \
  --ancestry-reference /data/opengwasdb/reference/ancestry-mixture/ref_freqs.hg38.tsv.gz \
  --ancestry-groups /data/opengwasdb/reference/ancestry-mixture/ancestry_groups.tsv \
  --cores 64 --parser-analyses 0 \
  --output docs/benchmark-output/opengwasdb_resolver_evidence_scan.json

# Parser/decompressor prototype benchmark
pixi run -e dev python benchmarks/benchmark_resolver_evidence_scan.py \
  --manifest docs/benchmark-output/opengwasdb_resolver_evidence_scan_manifest.tsv \
  --ancestry-reference /data/opengwasdb/reference/ancestry-mixture/ref_freqs.hg38.tsv.gz \
  --ancestry-groups /data/opengwasdb/reference/ancestry-mixture/ancestry_groups.tsv \
  --parsers-only --parser-analyses 12 --parser-repetitions 1 \
  --output docs/benchmark-output/opengwasdb_resolver_evidence_scan_parsers.json

# Render the results report from the two artifacts (never hand-edited numbers)
pixi run -e dev python benchmarks/benchmark_resolver_evidence_scan.py \
  --manifest docs/benchmark-output/opengwasdb_resolver_evidence_scan_manifest.tsv \
  --ancestry-reference /data/opengwasdb/reference/ancestry-mixture/ref_freqs.hg38.tsv.gz \
  --ancestry-groups /data/opengwasdb/reference/ancestry-mixture/ancestry_groups.tsv \
  --report-only docs/benchmark-output/opengwasdb_resolver_evidence_scan.json \
  --merge docs/benchmark-output/opengwasdb_resolver_evidence_scan_parsers.json \
  --report docs/spec/bounded-evidence-scan-report.md
```
