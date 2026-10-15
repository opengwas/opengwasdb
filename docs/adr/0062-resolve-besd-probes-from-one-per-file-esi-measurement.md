# Resolve BESD probes from one per-file `.esi` measurement

Every Analysis is resolved; there is no declared-only path (stores ADR 0036).
A BESD file set is the one Source Format with many Analyses in one source:
eQTLGen's cis-eQTL release is one `.esi` of 8,932,843 variants, one `.epi` of
19,250 probes and one 2 GB `.besd`, and each probe is an Analysis. Until issue
#282 `resolve-analyses` could not read it, so eQTLGen waited.

The resolver's other formats hold one Analysis per file, and it reads each file
once per Analysis (ADR 0044). BESD splits the evidence differently:

- **Per variant, shared by every probe:** the `.esi`. Its alleles fix each row's
  canonical identity and orientation; its seventh column is the frequency of
  `a1`, the effect allele. `build_besd.py` said "BESD carries no frequencies"
  — that was true of what the builder stores, not of the format, and eQTLGen
  reports one on every row.
- **Per association:** the `.besd`. A probe's beta and SE, and nothing else:
  no frequency and no sample size.

## Decision

**A manifest row declares one probe.** `source_reader_capability =
opengwasdb.besd`, `source_file` is the file-set prefix (what
`build-ragged-besd` takes), and the new `source_probe_id` names the `.epi`
probe. The probe ID is required on a BESD row and refused on any other, and is
never inferred from `analysis_id`: the builder's `probe::tissue` IDs show the
two need not match. `sample_size` is the existing per-Analysis column.

**Ancestry and allele orientation are measured once per file.** The parent
process reads the `.esi` once — streaming, into arrays — aligns it to the
reference build, fits the Ancestry Reference Panel against its A1-oriented
frequencies, and holds the result in a module global the forked workers
inherit, exactly as the panel itself is shared (ADR 0045). Every probe's record
carries that one `AncestryAssignment` — its composition, gates and
`eaf_orientation` — under `diagnostics.ancestry_scope = source_file` and a
shared `ancestry_measurement_id`, a digest of everything the measurement
depends on. `index.json` lists each measurement once with the Analyses that
share it. The site rule is the tabular one (usable frequency, no palindromic
pair, a panel site, later rows win a repeated ALID) less its SE requirement:
a per-variant frequency belongs to no single association's SE.

**Phenotype SD is per probe.** Each probe's `.besd` slice is read by seeking —
the header, two column offsets and the slice — and joined onto the `.esi`
arrays to form an ordinary metrics block. From there it is the existing
resolver (ADR 0044): the same evidence sample, tiers and estimator, with the
manifest's `sample_size`. The record has the same shape as a GWAS-SSF record.

**An hg19 file set lifts through the GWAS-VCF projection.** `--chain-file`,
one unique mapping per SNP, alleles complemented on a reverse-strand mapping,
and an indel or ambiguous mapping refused (ADR 0044's assembly rules). The
assembly plan admits BESD beside GWAS-VCF.

**Two refusals, each a controlled failure of the Analysis, never a value:**

1. **A `.esi` with no usable frequency** — no seventh column, or `NA` on every
   row — fails every probe in the file. Ancestry and orientation have nothing to
   be measured from, and no reference frequency is substituted.
2. **A probe whose SD tier needs a sample size its row does not supply** fails.
   For a tabular source a missing `N` is recorded as `phenotype_sd.status =
   unavailable`; for BESD it is refused outright, because nothing in the format
   can ever supply one, so its absence is always a manifest omission rather than
   a fact about the source. The refusal is scoped to tiers that estimate an
   SD: a probe whose tier needs no estimate (`declared_standardised`,
   `source_provided`, `binary_trait`, a log-OR or log-hazard scale) needs no
   `N` and is not refused.

**A refused probe still records its file's measurement.** "Every Analysis's
measurements are recorded whether it passes or not" (stores ADR 0036): a probe
refused for its missing sample size, or for its own damaged slice, carries the
file set's ancestry fit, `ancestry_scope = source_file`, the shared
`ancestry_measurement_id` and the `.esi` counts, and only `phenotype_sd` is
null. A file set whose measurement itself failed (no frequencies, unreadable,
unalignable) has no fit to carry; its probes record the attempted
`ancestry_measurement_id` under `source_file` scope with zero counts, never an
Analysis-scope record of zero sites.

**Resume binds the file set and the probe.** `source_file` is a prefix, so a
BESD fingerprint adds each member file's size and mtime and the probe ID. Both
keys appear only on BESD fingerprints; every other format's digest is
unchanged. The member files are stat'ed once per row, and the measurement id
reuses those stats, so a fingerprint and the id it travels with cannot disagree.

**"Once" is counted.** `besd_measurement_count()` reports how many file sets
the process has measured. A run measures each file set it resolves exactly once
in the parent — a per-probe measurement would count one per probe, and one made
in a forked worker would not be counted at all — and a fully resumed run
measures none. A probe whose file set was not measured is a defect in the run
and raises, rather than being recorded as a failure of the data. The parent
drops the measurements, and their arrays, when the run's probes are done.

## Consequences

- **eQTLGen resolves.** On the 10-probe eQTLGen pilot (`/data/besd/eqtlgen-sparse`,
  GRCh37 lifted with the UCSC hg19→hg38 chain, `sample_size` 31,684, the CLI's
  default 50,000-site ancestry bound) all 10 records succeeded and share one
  measurement: EUR, dominant proportion 0.909, `eaf_orientation = passed` with
  r = 0.982, the bound reached after 102,734 `.esi` rows. Nine probes' SDs lie
  between 1.007 and 1.106 — near 1, as they must be, since eQTLGen's `.besd`
  betas were computed from z-scores on an SD scale and the estimate is
  `sqrt(N_manifest / (N_variant + z²))`. TNF (`ENSG00000232810`) gave 2.039,
  which that formula reads as a per-variant N near 7,600: the manifest's
  study-wide N is wrong for that probe, and the record shows it rather than
  hiding it. The seeking probe reader returned arrays identical to
  `BESDReader`'s for 497 probes (3,201,819 associations).
- **The measurement is a parent-process cost.** eQTLGen's `.esi` is read,
  lifted and fitted once in the parent before any worker starts, serially per
  file set. Its arrays hold about 9M canonical ALID strings, shared copy-on-write
  with the workers; the pilot run peaked at 8.2 GB resident with the full
  Ancestry Reference Panel loaded, and used 157 s of CPU (each probe 0.2–0.6 s).
  A collection of many BESD file sets measures them one after another.
- **Ancestry diagnostics change meaning by scope.** For `source_file`,
  `ancestry_sites`, `ancestry_rows_read`, `ancestry_stop_reason` and
  `ancestry_reference_rows_matched` describe the `.esi`; `rows_read` and the
  build-eligible counts stay the probe's own. `--max-ancestry-sites` bounds the
  `.esi` fit; `--max-rows` bounds both the `.esi` rows the fit considers and
  each probe's rows.
- **The builder still stores no EAF.** `build-ragged-besd` declares `eaf:
  absent`; storing the `.esi` frequency per association is separate work.
  The builder's own hg19 lift (`build_liftover_lookup`) does not complement
  alleles on a reverse-strand mapping, which this resolver does; the two agree
  on every forward-strand SNP.

## Rejected

- **Fit ancestry per probe from the probe's own rows.** It reads the same
  frequencies 19,250 times, each time over a cis window of a few thousand sites
  instead of the file's millions — most probes would fail the 5,000-site
  overlap gate — and two probes of one cohort could be labelled differently.
- **Measure lazily in each worker, cached per process.** Simpler wiring, but
  `--n-workers 32` measures one file 32 times, and "once per file" would be a
  property of the scheduler rather than of the design.
- **Load the whole `.besd` once and share it.** `BESDReader` does, for the
  build, which visits every probe. The resolver visits one probe per Analysis;
  2 GB in the parent buys nothing that a seek does not.
- **Default a missing sample size, or downgrade it to `unavailable`.** A
  default is the plausible wrong answer this project exists to refuse; and
  `unavailable` would record as a property of the source what is in fact a gap
  in the manifest.
- **Dispatch BESD through the reader registry.** The registry maps a
  capability to a one-Analysis-per-file reader factory, `(path, scale)`. A
  BESD probe needs a parent-made measurement and a probe ID, so the registry
  would have to grow a second factory shape for one format; the manifest
  resolver checks the capability where it branches instead.
- **Satisfy the tabular ancestry rule with a placeholder SE.** It would pass
  every row the rule exists to filter and hide that the rule had been bypassed.
