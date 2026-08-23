# Ancestry assignment from allele frequencies; ancestry-matched completion

Reference completion imputes against a single **ancestry-specific LD Reference
Panel**, so an Analysis can only be completed against a panel of its own ancestry
— imputing a European GWAS against an East-Asian LD structure is wrong. The
`ieu-a` / `ieu-b` consortium collections are a mixture of ancestries, and their
source-declared `population` metadata is coarse, often literally `"Mixed"`, and
sometimes wrong. We need to decide each Analysis's ancestry *ourselves*, from its
summary statistics, and route it accordingly.

## Decision

**Recover each Analysis's ancestry from its summary-statistic allele frequencies**
(GWAS-VCF `FORMAT/AF`, present throughout this harmonised collection), and let that
govern routing. The method follows Privé (2022): model an Analysis's allele
frequencies as a **non-negative, sum-to-one mixture** of reference-population
frequencies over a common variant set, solved by NNLS, yielding **Ancestry
Composition**. We **fit against a fine reference** — Privé (2022)'s UK Biobank
"global reference of worldwide populations" (allele frequencies for ~5.8M variants
across 21 ancestry groups, the `snp_ancestry_summary` reference; the **Ancestry
Reference Panel**) — for accuracy on admixed and edge populations, then
**aggregate the proportions up to super-populations** for the routing label,
because our LD panels are keyed to super-populations. Fit-fine, label-coarse.

An Analysis's **Assigned Ancestry** is the dominant super-population, admitted only
if it clears a **multi-gate rule**: proportion ≥ τ, margin over the runner-up ≥ δ,
reference-SNP overlap ≥ N_min, EAF orientation not inverted, and NNLS fit residual
below a gate. Failing any gate leaves the Analysis **Unassigned** rather than
force-labelled. τ/δ are **calibrated**
against the **Reported Population** across the collection — which is used only to
choose the operating point and to audit disagreements, **never to route**.

**Ancestry-Matched Completion:** a store records **Assigned Ancestry per Analysis**,
and reference completion imputes an Analysis only when its Assigned Ancestry matches
the panel being applied; other Analyses are left observed-only (`completed_against =
null`). Store-level Completion State stays a coarse release flag ("a completion pass
ran"); per-cell Association Status remains the ground truth for what was imputed.
This makes ancestry-homogeneity **optional at the store contract level** — but for
the first `ieu-a`/`ieu-b` build we deliberately build a **homogeneous EUR store**
from a EUR subset of the Catalogue, because mixing ancestries in one store inflates
missingness (non-overlapping off-panel tails) and only EUR has an LD panel today.

## Considered options

- **Trust the source-declared `population`.** Rejected: coarse, frequently `"Mixed"`,
  sometimes wrong — it is the thing AF-matching exists to improve on. Kept only as a
  calibration/audit signal.
- **PC-projection ancestry (individual-level).** Not applicable: we have summary
  statistics, not genotypes. The AF-mixture is the summary-statistic analogue and is
  what Privé's `snp_ancestry_summary` implements.
- **Fit directly against 5 super-population means (coarse-only).** Rejected:
  super-population mean frequencies model admixed samples poorly (AMR is itself
  admixed), so the EUR-vs-rest gate is miscalibrated. Fitting fine and aggregating
  fixes this at the cost of a larger static reference table.
- **A single dominant-proportion threshold.** Rejected in favour of the multi-gate
  rule: overlap and residual gates catch corrupt / mis-oriented AF that a bare
  proportion threshold would force-label; the margin gate catches near-ties.
- **Leaving mis-orientation to the residual gate** (the original decision;
  amended below). Rejected on evidence: it detects the condition but cannot name
  it, and it carries more weight alone than it should.
- **Store-level ancestry as a homogeneity requirement.** Rejected: a per-Analysis
  attribute + Ancestry-Matched Completion is strictly more general (supports future
  multi-panel completion) and costs nothing here, where we still choose a homogeneous
  build.

### Amendment: EAF orientation is its own gate (issue #115)

The residual gate does catch a mis-oriented AF column — measured on the real
`GCST003566`, chr22 against the Ancestry Reference Panel:

| | NNLS residual | correlation vs reference consensus | gate |
|---|---|---|---|
| `GCST005076` as published | 0.0090 | +0.98 | ok → EUR (0.951) |
| `GCST005076` deliberately inverted | 0.5853 | −0.98 | residual |
| `GCST003566` as published | 0.5788 | −0.9992 | residual |
| `GCST003566` with AF un-inverted | 0.0146 | +0.9992 | ok → EUR (1.000) |

Two things that measurement shows the residual gate cannot do on its own:

- **It cannot name the cause.** A flipped Analysis, a corrupt AF column and a
  cohort the reference cannot represent all report `gate_reason="residual"`.
  Only the first is something a curator can act on — by excluding the source or
  reporting it upstream — and only if they are told which it is.
- **It is carrying the ancestry label alone.** An inverted Analysis does not
  merely fail to fit; it fits as a *different* super-population. `GCST003566`
  comes back as **AFR at 0.696**, above the τ = 0.50 proportion gate. The
  residual gate is the only thing between an inverted frequency column and a
  confidently wrong Assigned Ancestry, which is too much for one
  general-purpose gate.

So the correlation is computed explicitly — it is free, over arrays the fit
already holds — and gets its own gate, checked *before* the residual because it
is the more specific reading of the same evidence. Its threshold
(`orientation_flip_r`, default −0.5) is deliberately not the build-time rule:
`opengwasdb.build.eaf_orientation` fails on the *sign*, because any negative
correlation makes a stored column untrustworthy, whereas this gate is claiming
a *cause* and needs the evidence to support it. An inverted column reads about
−1; an otherwise-corrupt one sits near zero and could fall either side. A study
between the two is still refused — by the residual gate, which is the honest
answer when the cause is not established.

Ancestry assignment also stops being GWAS-VCF-only, and resolves its reader
through `opengwasdb.readers.registry`. That restriction is why this defect went
unexamined: the `gwas-catalog-eur-hybrid` family is harmonised GWAS-SSF, so
every Analysis in the pilot carries
`ancestry_assignment_method=source_trusted_no_af` — the source's declared
population, taken on trust, with its frequencies never compared to anything.

This does not replace the build-time check. A build can be driven by a manifest
that never went through the Catalogue, the build-time check reads back what was
actually stored (after orientation, liftover and dedup) rather than what the
source file said, and ADR 0030 requires the store to carry its own evidence.
The two are the same rule at different points, and the earlier one is the one
that saves an hour of building.

## Consequences

- Palindromic (A/T, C/G) variants are excluded from the fit (strand-ambiguous,
  unalignable); the fit uses common (reference MAF ≥ ~1%) variants in the
  reference ∩ study-AF intersection, without heavy LD-pruning (NNLS tolerates
  correlated SNPs).
- The **Ancestry Reference Panel** (Privé 2022's UK Biobank reference frequencies
  for 21 groups — bigsnpr `ref_freqs.csv.gz`, on GRCh37, lifted to hg38 and re-keyed
  to canonical ALIDs — plus a fine→super-population map) is a static artifact to
  source and normalise. The liftover is a one-off on a fixed file, cheaper than
  standing up a raw callset.
- Ancestry assignment is an **annotator that writes into the Analysis Catalogue**
  (ADR 0027): Assigned Ancestry, Ancestry Composition, gate results, EAF
  orientation evidence (issue #115), and the Reported-Population comparison. Stores inherit Assigned Ancestry via the subset
  they are built from.
- Unassigned and non-target-ancestry Analyses are annotated and **parked in the
  Catalogue**, not dropped — re-routable when their panel exists, without
  re-extracting allele frequencies.
- Completion becomes ancestry-aware: it reads per-Analysis Assigned Ancestry and
  imputes only matching Analyses, recording `completed_against` per Analysis.
