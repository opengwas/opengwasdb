# Resolve BESD probes from one per-file `.esi` measurement

> **Amended by [ADR 0067](0067-per-association-sample-size-and-besd-allele-frequency.md)**
> (issue #302). The consequence "the builder still stores no EAF" no longer
> holds unconditionally: `build-ragged-besd --store-esi-eaf` stores the `.esi`
> frequency per association, checked once against a reference as this ADR's
> resolver measures it once, and `--sample-size-table` stores each pair's
> sample size. Without either flag the builder stores what it always did. The
> resolution decisions below are unchanged.

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
one unique mapping per variant, alleles complemented on a reverse-strand mapping,
and an indel or ambiguous mapping refused (ADR 0044's assembly rules). The
assembly plan admits BESD beside GWAS-VCF.

**A post-lift collision is ambiguous, and all of it is dropped.** A chain set
does not have to be injective: two hg19 loci can both map into one hg38
position, and when their projected alleles agree they become one canonical
variant claimed by two ESI entries. Nothing in the file set says which entry a
probe's association belongs to, so every entry in the colliding group is
dropped, with its associations, exactly as a liftover failure is — never a
winner chosen by position, by allele order or by row order.

**That drop is this builder's own policy, not a Dense rule it inherits.** The
Dense VCF builder meets the same collapse and answers differently, because a
dense column can hold both source variants' associations against one row:
`_source_alids_by_alid` blanks the ambiguous source-coordinate provenance
(`hg38_to_source[hg38_alid] = None  # collision → ambiguous`) and the column
keeps the last stream occurrence for that row (`tests/test_dense_vcf_build.py::
test_last_wins_dedup_on_collision`). Its only association-dropping collision is
a different one — `_drop_cross_assembly_collisions` drops raw tuples declared
under *both* assemblies, which is one string naming two physical loci rather
than two loci resolving to one variant. What BESD borrows from Dense is the
refusal to guess: Dense will not invent which source coordinate a collapsed
row came from, and BESD will not invent which entry a collapsed variant's
association came from. BESD has to go further because a Ragged CSR cannot
absorb the collapse at all — holding both would put one `variant_index` twice
inside one Analysis's segment, and `(variant, Analysis)` resolution cannot tell
the two rows apart.

The build says how many entries went, on stdout and as
`provenance.lift.n_collisions` / `n_collision_groups` beside `n_lifted`. The
same refusal covers a source that lists one variant twice with no lift
involved, since it is the store's invariant rather than the chain's; such a
build has no lift record to hold the counts, so it records them at the
provenance's root (issue #304).

On eQTLGen's full file set (8,932,823 `.esi` rows, 19,250 probes) 5,736 rows
fail the lift and 26 of the remaining 8,927,087 landings claim 13 canonical
variants between them — so the axis holds 8,927,061 variants, 13 fewer than the
8,927,074 the colliding build stored, and 341 associations go with them
(127,112,772 against 127,113,113). The 13 collapsed variants are the difference
that build's log had been reporting without a reason. None of them is a
duplicated source row — no two `.esi` rows share a `(chrom, bp, allele pair)` —
so every one is two hg19 loci landing on one hg38 coordinate. The whole build
takes 12m01s, against 15m25s for the same file set with the collision left in,
and validates `ok`.

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
  *(Since ADR 0067 it does, with `--store-esi-eaf`.)*
- **The builder and this resolver now share one lift (issue #284).**
  `build_liftover_lookup` projects a lifted variant's alleles onto the chain
  block's strand through the same helper `oriented_variant` uses
  (`readers.gwas_vcf.project_lifted_alleles`), so the two cannot disagree about a
  reverse-strand SNP. Until #284 they did: the builder kept the source's alleles,
  so every reverse-strand row was stored under an hg38 position carrying
  hg19-strand alleles. Over eQTLGen's 8,932,843 `.esi` rows that was 8,200
  variants whose ALID changes and 9,314 reverse-strand rows; 1,654 of the
  variants were contradicted by the EUR alid panel, and the projected lift
  contradicts it nowhere. The builder now also *checks* the lift:
  `--variant-reference` must not spell a reverse-strand variant on the other
  strand, which fails the build naming the variant, and `provenance.lift`
  records the reference used or that none was supplied. It keeps one thing this
  resolver does not: where a position maps under more than one chain block it
  takes the highest-scoring mapping, because a batch lift of nine million rows
  cannot fail the way one Analysis can (ADR 0044).
- **The Ragged segment invariant tightened to catch it (issue #304).** A
  repeated `variant_index` inside one Analysis's segment is a duplicate
  association, not an ordered one, so §11/§20 now require each segment to be
  *strictly* increasing and validation reports a repeat separately from a
  decrease, naming the Analysis. Before this the collision was visible only
  through a p-value tier's top-hit index: on OGS-00027's partial release
  `validate` failed with `top-hit index p_5e_08 has incorrect or non-genomic
  analysis slices` (likewise `p_5e_06`, `p_5e_04`) while the CSR rule passed
  the same 103 duplicated `variant_index` values in 74 of 19,250 Analyses. A
  10-probe pilot hits no collision, so the defect only appeared at full scale.
- **Dense and Hybrid re-orient, but do not check.** Both ingest the projected
  alleles' orientation from the same lift -- a swapped canonical order negates z
  and inverts EAF (`statistics_swap`) -- because the rename #284 introduced
  would otherwise leave them storing source-signed statistics under the
  projected allele pair. Neither records a `provenance.lift` or runs the
  reference check; adding that to them is a separate decision, not a
  consequence of the shared lift. One Hybrid path still refuses rather than
  guesses: a single-pass build's off-reference rows are lifted in the fold,
  which carries no orientation, so such a build fails loudly
  (`UnorientedOffReferenceLiftError`) when that lift lands on the reverse
  strand. A two-pass build lifts in Pass 1 and is unaffected.

## Rejected

- **Keep one entry of a colliding group and drop the rest.** Which of two hg19
  loci "wins" a shared hg38 coordinate would be decided by row order, which is
  not evidence about the data. Dense can keep last-wins for such a pair because
  its column is one row and no `(variant, Analysis)` key changes meaning; a
  Ragged segment cannot, and the ambiguity is exactly about which entry the
  association belongs to, so keeping either would attribute one locus's
  association to the other.
- **Fail the build on a collision.** Nine million `.esi` rows are not one
  Analysis: a batch lift cannot refuse the way ADR 0044's per-Analysis
  resolution can, and 13 ambiguous variants in 8.9M is not a reason to have no
  eQTLGen store. It is a reason to drop them loudly and record the count.
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
