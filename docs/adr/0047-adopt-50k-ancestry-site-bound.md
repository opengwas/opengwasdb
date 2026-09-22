# Adopt a 50,000-site ancestry bound for the full-catalog release

> **Superseded in part by [ADR 0048](./0048-decouple-ancestry-site-bound-from-quantitative-phenotype-sd.md).**
> ADR 0048 decouples the 50,000-site ancestry bound from quantitative phenotype-SD
> evidence gathering so that quantitative Studies stream to EOF for exact SD
> estimation while case-control Studies retain early termination. The 13.4x whole-frame
> speedup noted below was measured under v1 physical-stop semantics across all studies;
> under v2 semantics, speedup applies to the case-control subset.

ADR 0046 retained the full source scan after the #209 evaluation rejected every
preregistered early-stop rule. This ADR supersedes that policy decision for the
full GWAS Catalog release, on the maintainer's explicit judgement that the
computational saving across ~6,000 GWAS outweighs the measured risk. It does not
change the evaluation's numbers; it changes what the release does with them.

## Context

The #209 study measured the `max_ancestry_sites` rule on the frozen 106-Analysis
frame. At 50,000 usable ancestry-reference sites:

- **Speed:** 25/106 sources reach EOF; the rest stop at the bound. Aggregate
  wall time falls from 9,587s to 1,255s, a **13.4x** mean speedup (per-study
  median ~98x; the files that dominate total time stop early at 93-148x). This
  is before the block-wise projection (ADR 0046's vectorised follow-on), so the
  two compound.
- **Ancestry:** 105/106 Analyses agree exactly on Assigned Ancestry and gate
  reason. The one exception is a **false-positive EUR**: `GCST90859377` is
  Unassigned by the full scan (`residual` 0.110 > `residual_max` 0.06) and is
  assigned EUR at 0.998 (residual 0.006) by the bound.
- **Phenotype SD:** status agreement 100%; the implied-SD estimate differs by at
  most 8.6% (most under 2%); and applying the release's own acceptance rule
  (`effect_scale_validation.R`: `dispersion > 0.5` -> warning) flips **0 of 54**
  quantitative outcomes.

## Decision

**The `resolve-analyses` manifest CLI bounds every scan at 50,000 usable
ancestry-reference sites by default.** `--max-ancestry-sites 0` restores the
full scan; `--max-rows` bounds by source rows instead. The per-Analysis
`resolve_analysis` seam keeps a full scan as its default, so a direct caller
opts in explicitly.

**The bound is part of the resolution's provenance.** Every record's
`fingerprints.resolution_config.scan_limit` carries the rule version and
thresholds, `diagnostics.stop_reason` says whether EOF or the bound ended the
scan, and a changed bound invalidates resume. A bounded resolution is a
statement about a prefix and can never be read back as a full-source one.

## Consequences

- **One known false-positive EUR is accepted.** `GCST90859377` will be labelled
  EUR in a release built with this default. It is recorded here rather than
  hidden: the release's validation must not present the bounded assignment as
  the full-genome one, and the Analysis is a candidate for a full-scan
  re-resolution or exclusion.
- **The saving is real but uneven.** Sources with sparse reference coverage --
  and the ~24% that reach EOF -- save little or nothing; the large genome-wide
  sources, which dominate the batch, save 90x+. A 6,000-GWAS batch is dominated
  by those, so the aggregate saving holds.
- **The bound is a release policy, not a scientific claim.** It is a
  compute/accuracy trade the maintainer owns; the #209 report remains the
  evidence for what it costs.
- **A future change to the rule is a new ADR.** Vectorising the sampling hash
  (the remaining ~4x) changes which rows the SD sample selects and must not be
  folded into this bound.

## Rejected

- **Full scan for the full-catalog release.** Correct but the reason #209 was
  opened: ~6,000 genome-wide sources at full scan is not affordable.
- **A random fixed reference panel (10k-250k sites).** The non-EUR pilot shows
  it preserves the assignment and fails safe to Unassigned, but it does **not**
  save time -- every source row is still projected -- so it does not serve the
  compute goal.
- **Waiting for a rule that clears the locked zero-false-positive criterion.**
  No rule in the tested space does; the maintainer accepts the measured
  exception instead.
