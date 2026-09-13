"""Phenotype-SD estimation at ingest (issue #15; ADR-0029).

Given per-variant standard errors and matched allele frequencies (source or
reference) plus a sample size, estimates a per-study phenotype SD so
continuous-trait effects can be rescaled to SD units at build time
(`stored_se = original_se / sd`). Pure computation over arrays -- no file
I/O, no bcftools, no store awareness -- so it is testable against synthetic
data with a known true SD.

This module owns the shared `median(se * sqrt(2*af*(1-af)))` primitive ADR-0029
repurposes from `opengwasdb.completion.impute.scalar_n_se`'s pre-existing
per-study SE-scale constant (used there for a different purpose: imputing SE
at reference-panel positions during Reference Completion, not phenotype-SD
estimation). `scalar_n_se` calls `se_scale_constant` below rather than
recomputing the same median independently, so there is one implementation of
the formula, not two.

Only estimates the three ADR-0029 method tiers that are genuinely
computation (`estimated_from_source_maf`, `estimated_from_reference_maf`,
`estimated_from_beta_distribution`); `declared_standardised`, `source_provided`,
and `binary_trait` require no estimation and are the caller's decision before
this module is ever invoked. Method selection is caller-supplied, not
inferred: the caller decides which tier applies to the inputs it has and
passes that tier in; this module reports back the method it actually applied
-- which is the requested one on success, or `unavailable` when the sample
size is missing or unusable, regardless of which method was requested. A
missing/unusable sample size is refused rather than silently estimated,
matching this codebase's existing convention of an explicit unresolved
outcome over a defaulted value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opengwasdb.model.enums import OriginalSdMethod

_ESTIMATION_METHODS = frozenset(
    {
        OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF,
        OriginalSdMethod.ESTIMATED_FROM_REFERENCE_MAF,
        OriginalSdMethod.ESTIMATED_FROM_BETA_DISTRIBUTION,
    }
)

#: Public name for the tiers this module computes, so callers that must
#: distinguish "a computable tier" from "a caller decision" (the
#: `estimate-phenotype-sd` CLI, issue #176) read one definition rather than
#: re-listing the tiers.
ESTIMATION_METHODS = _ESTIMATION_METHODS


@dataclass(frozen=True)
class PhenotypeSdEstimate:
    """A per-study phenotype SD estimate, the method tier actually applied,
    and a dispersion diagnostic over the estimate's supporting evidence."""

    sd: float
    method: OriginalSdMethod
    dispersion: float
    notes: str | None = None


def se_scale_samples(se: np.ndarray, af: np.ndarray) -> np.ndarray:
    """Per-variant `se * sqrt(2*af*(1-af))` values behind the ADR-0029
    scalar estimator, filtered to finite `se` and `af` strictly within (0, 1).
    Empty when nothing is valid.

    This exact filter matches `opengwasdb.completion.impute.scalar_n_se`'s
    original validity mask (finite se, finite af, 0 < af < 1, no `se > 0`
    requirement) so refactoring `scalar_n_se` to call `se_scale_constant`
    changes nothing about its output.
    """
    valid = np.isfinite(se) & np.isfinite(af) & (af > 0) & (af < 1)
    samples: np.ndarray = se[valid] * np.sqrt(2.0 * af[valid] * (1.0 - af[valid]))
    return samples


def se_scale_constant(se: np.ndarray, af: np.ndarray) -> float:
    """`median(se * sqrt(2*af*(1-af)))` over valid entries. NaN when none."""
    samples = se_scale_samples(se, af)
    if samples.size == 0:
        return float("nan")
    return float(np.median(samples))


def _mad(values: np.ndarray) -> float:
    """Median absolute deviation -- ADR-0029's dispersion diagnostic."""
    if values.size == 0:
        return float("nan")
    return float(np.median(np.abs(values - np.median(values))))


def _unavailable(notes: str) -> PhenotypeSdEstimate:
    return PhenotypeSdEstimate(
        sd=float("nan"), method=OriginalSdMethod.UNAVAILABLE, dispersion=float("nan"), notes=notes
    )


def estimate_phenotype_sd(
    method: OriginalSdMethod,
    sample_size: float | None,
    *,
    se: np.ndarray | None = None,
    af: np.ndarray | None = None,
    beta: np.ndarray | None = None,
) -> PhenotypeSdEstimate:
    """Estimate a per-study phenotype SD for the given ADR-0029 method tier.

    `method` must be one of `estimated_from_source_maf`,
    `estimated_from_reference_maf`, or `estimated_from_beta_distribution` --
    the tiers this module actually computes. `se`/`af` (for the two
    AF-based tiers) or `beta` (for the beta-distribution tier) must be
    supplied accordingly; which inputs are available is the caller's
    decision, not something this function infers from what happens to be
    non-None.

    Returns a refused (`unavailable`) estimate, regardless of the requested
    method, when `sample_size` is missing, non-finite, or non-positive --
    the ADR-0029 formula only ever yields `sd_scale / sqrt(N)` as a combined
    quantity without `N`, so there is nothing to report without it.
    """
    if method not in _ESTIMATION_METHODS:
        raise ValueError(
            f"estimate_phenotype_sd only computes {sorted(m.value for m in _ESTIMATION_METHODS)}, "
            f"got {method.value!r} -- that tier is a caller decision, not an estimate"
        )
    if sample_size is None or not np.isfinite(sample_size) or sample_size <= 0:
        return _unavailable("sample size is missing, non-finite, or non-positive")

    if method is OriginalSdMethod.ESTIMATED_FROM_BETA_DISTRIBUTION:
        if beta is None:
            raise ValueError("estimated_from_beta_distribution requires beta")
        return _estimate_from_beta_distribution(beta, method)

    if se is None or af is None:
        raise ValueError(f"{method.value} requires se and af")
    return _estimate_from_allele_frequency(se, af, sample_size, method)


def _estimate_from_allele_frequency(
    se: np.ndarray,
    af: np.ndarray,
    sample_size: float,
    method: OriginalSdMethod,
) -> PhenotypeSdEstimate:
    samples = se_scale_samples(se, af)
    if samples.size == 0:
        return _unavailable("no variant had both a finite se and an af strictly within (0, 1)")
    sqrt_n = float(np.sqrt(sample_size))
    implied_sd = samples * sqrt_n
    return PhenotypeSdEstimate(
        sd=float(np.median(implied_sd)), method=method, dispersion=_mad(implied_sd)
    )


_MAD_TO_SD = 1.4826  # consistency constant for MAD as a normal-SD estimator


def _estimate_from_beta_distribution(
    beta: np.ndarray, method: OriginalSdMethod
) -> PhenotypeSdEstimate:
    """`sd` is a robust (MAD-based), not plain-`std`, spread estimate: ADR-0029's
    own rationale for preferring a median-based scalar over per-variant
    reconstruction elsewhere -- avoiding distortion from large effects at
    genome-wide-significant loci -- applies just as much to this fallback
    tier's "random common-variant" betas, which are only expected to be
    near-null, not guaranteed to be. ADR-0029 does not specify a distinct
    dispersion formula for this tier (unlike the two AF-based tiers' shared
    per-variant implied-sd dispersion, `_estimate_from_allele_frequency`):
    `dispersion` here is proportional to `sd` itself rather than an
    independent diagnostic, which is an honest limitation, not a hidden
    inconsistency.
    """
    valid = beta[np.isfinite(beta)]
    if valid.size < 2:
        return _unavailable("fewer than 2 finite beta values to estimate a spread from")
    mad = _mad(valid)
    return PhenotypeSdEstimate(sd=_MAD_TO_SD * mad, method=method, dispersion=mad)
