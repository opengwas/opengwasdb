"""NNLS ancestry mixture and the multi-gate admission rule (ADR 0028).

Models a study's A1-oriented allele frequencies as a non-negative, sum-to-one
mixture of the fine reference groups (solved by NNLS with a soft sum-to-one
penalty row), aggregates the fitted proportions to super-populations, and admits
a single **Assigned Ancestry** only when the dominant super-population clears the
proportion (τ), margin (δ), overlap (N_min), EAF-orientation, and NNLS-residual
gates. Any failure leaves the Analysis **Unassigned** rather than force-labelled.

The orientation gate is issue #115's check, applied here as well as at build
time. It costs nothing extra: the fit already holds the study's A1-oriented
frequencies and the reference's, so the correlation is over arrays in hand.
Two things it buys, neither of which the residual gate delivers on its own:

* **A specific diagnosis.** A mis-oriented study fails the residual gate --
  measured at 0.58 against a residual_max of 0.06, where a correctly oriented
  study reads 0.009 -- but `gate_reason="residual"` reads the same for a
  flipped study, a corrupt AF column, and a genuinely unusual cohort. The
  correlation names the flip, so a curator can act on it (excluding the source,
  or reporting it upstream) instead of investigating three possibilities.
* **A guard on the ancestry label itself.** A flipped European study does not
  merely fail to fit: measured on `GCST003566`, it fits as 0.696 **AFR** --
  above the τ = 0.50 proportion gate. The NNLS residual is the only thing
  standing between an inverted frequency column and a confidently wrong
  ancestry, which is more weight than one general-purpose gate should carry
  alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.optimize import nnls  # type: ignore[import-untyped]

from opengwasdb.ancestry.reference import AncestryReference
from opengwasdb.build.eaf_orientation import correlate_frequencies
from opengwasdb.model.enums import EafOrientationOutcome, StoredEffectScale
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY, GwasVcfReader
from opengwasdb.readers.interface import af_only
from opengwasdb.readers.registry import resolve_reader


@dataclass(frozen=True)
class Gates:
    """Multi-gate admission thresholds (ADR 0028)."""

    tau: float = 0.50  # min dominant super-population proportion
    delta: float = 0.20  # min margin of dominant over runner-up
    n_min: int = 5_000  # min overlapping reference sites
    # Corrupt AF, or a cohort the reference cannot represent. Mis-orientation
    # used to land here too; it now has its own gate, which names it (issue #115).
    residual_max: float = 0.06  # max RMS NNLS residual
    # At or below this correlation against the reference consensus, the study's
    # frequencies are reported as mis-oriented rather than as a generic residual
    # failure. Unlike the build-time check (`opengwasdb.build.eaf_orientation`),
    # which fails on the *sign* because any negative correlation makes the column
    # untrustworthy, this one is a diagnosis and needs the evidence to support
    # it: an inverted column reads about -1 (measured -0.9992 on GCST003566),
    # while a half-corrupt one sits near zero and could fall either side of it.
    # A study between the two is still rejected -- by the residual gate, which
    # is the honest answer when the cause is not established.
    orientation_flip_r: float = -0.5
    sum_to_one_penalty: float = 10.0  # weight of the soft Σα = 1 constraint row


@dataclass(frozen=True)
class AncestryAssignment:
    """Result of fitting one Analysis against the Ancestry Reference Panel."""

    assigned_ancestry: str | None  # super-population, or None if Unassigned
    dominant_superpop: str | None
    dominant_proportion: float
    runner_up_margin: float
    af_overlap: int
    residual: float
    # "ok" | "overlap" | "eaf_orientation" | "residual" | "proportion" | "margin"
    gate_reason: str
    # Issue #115: the sign of the correlation between this study's A1-oriented
    # frequencies and the reference's. `passed`/`failed`/`unverified`, recorded
    # for every Analysis rather than implied by the assignment having succeeded.
    eaf_orientation: str = EafOrientationOutcome.UNVERIFIED.value
    eaf_orientation_r: float = float("nan")
    superpop_composition: dict[str, float] = field(default_factory=dict)
    fine_composition: dict[str, float] = field(default_factory=dict)


def assign_ancestry(
    study_af: dict[str, float],
    reference: AncestryReference,
    gates: Gates | None = None,
) -> AncestryAssignment:
    """Fit ``study_af`` against the reference and apply the multi-gate rule."""
    gates = gates or Gates()

    rows = [reference.index[a] for a in study_af if a in reference.index]
    overlap = len(rows)
    if overlap == 0:
        return AncestryAssignment(
            assigned_ancestry=None,
            dominant_superpop=None,
            dominant_proportion=0.0,
            runner_up_margin=0.0,
            af_overlap=0,
            residual=float("nan"),
            gate_reason="overlap",
        )

    row_idx = np.asarray(rows, dtype=np.int64)
    ref_alids = reference.alids[row_idx]
    b = np.asarray([study_af[a] for a in ref_alids.tolist()], dtype=np.float64)
    A = reference.freqs[row_idx]  # (overlap, G)

    # Drop any reference NaNs (uninformative for this fit).
    valid = ~np.isnan(A).any(axis=1) & ~np.isnan(b)
    A, b = A[valid], b[valid]
    overlap = int(A.shape[0])
    if overlap == 0:
        return AncestryAssignment(
            assigned_ancestry=None,
            dominant_superpop=None,
            dominant_proportion=0.0,
            runner_up_margin=0.0,
            af_overlap=0,
            residual=float("nan"),
            gate_reason="overlap",
        )

    # EAF orientation (issue #115), against the unweighted mean of the fine
    # groups. Direction is what is being read, and direction is the one thing a
    # population bottleneck does not change, so the consensus of every group
    # serves regardless of which one this study turns out to belong to -- there
    # is no circularity with the assignment this same call is computing.
    # `gates.n_min`, not the build-time default, so this module has one overlap
    # threshold rather than two that could contradict each other: a fit admitted
    # on N sites is one whose direction is readable on N sites.
    orientation = correlate_frequencies(b, A.mean(axis=1), min_overlap=gates.n_min)

    alpha = _fit_mixture(A, b, gates.sum_to_one_penalty)
    residual = float(np.sqrt(np.mean((A @ alpha - b) ** 2)))

    fine_composition = dict(zip(reference.groups, alpha.tolist(), strict=True))
    sp_props = reference.aggregate(alpha)
    superpop_composition = dict(zip(reference.superpops, sp_props.tolist(), strict=True))

    order = np.argsort(sp_props)[::-1]
    dominant_superpop = reference.superpops[int(order[0])]
    dominant_proportion = float(sp_props[order[0]])
    runner_up = float(sp_props[order[1]]) if sp_props.size > 1 else 0.0
    margin = dominant_proportion - runner_up

    gate_reason = apply_gates(
        gates,
        overlap=overlap,
        residual=residual,
        dominant_proportion=dominant_proportion,
        margin=margin,
        eaf_orientation_r=orientation.r,
    )
    assigned = dominant_superpop if gate_reason == "ok" else None

    return AncestryAssignment(
        assigned_ancestry=assigned,
        dominant_superpop=dominant_superpop,
        dominant_proportion=dominant_proportion,
        runner_up_margin=margin,
        af_overlap=overlap,
        residual=residual,
        gate_reason=gate_reason,
        eaf_orientation=orientation.outcome.value,
        eaf_orientation_r=orientation.r,
        superpop_composition=superpop_composition,
        fine_composition=fine_composition,
    )


def assign_from_source(
    path: str | Path,
    reference: AncestryReference,
    gates: Gates | None = None,
    *,
    capability: str = GWAS_VCF_CAPABILITY,
    regions_file: str | Path | None = None,
    region: str | None = None,
    liftover: object | None = None,
) -> AncestryAssignment:
    """Extract AF at reference sites from any Source Format, then assign ancestry.

    Resolves ``capability`` through `opengwasdb.readers.registry` rather than
    assuming GWAS-VCF (issue #115). The assumption had a cost: the
    `gwas-catalog-eur-hybrid` family is harmonised GWAS-SSF, so it could not be
    assigned from its own allele frequencies at all, and every Analysis in that
    pilot carries `ancestry_assignment_method=source_trusted_no_af` -- the
    source's declared population, taken on trust. `GCST003566`'s inverted
    frequency column sat behind that trust, unexaminable, until a store was
    built from it.

    ``regions_file``/``region``/``liftover`` are GWAS-VCF's bcftools
    optimisations and are applied only to that reader (a tabular reader scans
    its file once regardless, so they would mean nothing there). ``liftover``
    (a pyliftover ``LiftOver``) orients a study on a different assembly than
    the reference (e.g. GRCh37 GWAS-VCF → GRCh38 reference); see
    ``GwasVcfReader`` for why it disables the ``regions_file`` optimisation
    when set. ``region`` (bcftools ``-r``) restricts the read to one
    chromosome, in either mode.

    Two behaviours narrow slightly versus the pre-#21 AF-only extraction this
    replaces, both accepted as consequences of no longer maintaining two
    extraction implementations: called with no ``regions_file``/``region``/
    ``liftover`` (the bare default), this now always builds a ``-R`` regions
    file internally -- the VCF must be bgzip+tabix-indexed, where the old
    default was an unindexed full scan. And a site with a usable AF but an
    unparseable/missing SE is now dropped from the overlap entirely, where AF
    alone used to be enough -- the overlap gate is the intended safeguard
    against a shrunken site set, not a silent behaviour change to correctness.
    """
    # `stored_effect_scale` is required by the reader interface but unread by
    # `extract_at_sites`, which returns frequencies and standard errors only.
    reader = resolve_reader(capability, path, StoredEffectScale.SD)
    if isinstance(reader, GwasVcfReader):
        reader = replace(
            reader, liftover=liftover, regions_file=regions_file, region=region
        )
    sites = reader.extract_at_sites(reference.index.keys())
    study_af = af_only(sites)
    return assign_ancestry(study_af, reference, gates)


def assign_from_vcf(
    vcf_path: str | Path,
    reference: AncestryReference,
    gates: Gates | None = None,
    *,
    regions_file: str | Path | None = None,
    region: str | None = None,
    liftover: object | None = None,
) -> AncestryAssignment:
    """`assign_from_source` for a GWAS-VCF. Retained as the name callers use."""
    return assign_from_source(
        vcf_path,
        reference,
        gates,
        capability=GWAS_VCF_CAPABILITY,
        regions_file=regions_file,
        region=region,
        liftover=liftover,
    )


def _fit_mixture(A: np.ndarray, b: np.ndarray, penalty: float) -> np.ndarray:
    """NNLS with a soft Σα = 1 row, then renormalise to sum exactly to one."""
    n_groups = A.shape[1]
    if penalty > 0:
        A_aug = np.vstack([A, np.full((1, n_groups), penalty)])
        b_aug = np.concatenate([b, [penalty]])
    else:
        A_aug, b_aug = A, b
    alpha_raw, _ = nnls(A_aug, b_aug)
    alpha = np.asarray(alpha_raw, dtype=np.float64)
    total = alpha.sum()
    if total > 0:
        alpha = alpha / total
    return alpha


def apply_gates(
    gates: Gates,
    *,
    overlap: int,
    residual: float,
    dominant_proportion: float,
    margin: float,
    eaf_orientation_r: float = float("nan"),
) -> str:
    """Return the first failing gate, or ``"ok"``. Order matters (ADR 0028).

    Pure function of the summary statistics, so a calibrated τ/δ pick can relabel
    a Catalogue from its stored numbers without re-extracting allele frequencies.

    Orientation is checked before the residual because it is the more specific
    diagnosis of the same evidence: a mis-oriented study fails both, and being
    told which one it is decides what a curator does next (issue #115).
    Measured against the Ancestry Reference Panel, a correctly oriented study
    correlates around +0.98 and an inverted one around -0.98, while an AF column
    corrupted in some other way sits near zero -- so `orientation_flip_r`
    separates "inverted" from "unusable for a reason not established here",
    which the residual gate then catches without claiming a cause.

    A non-finite `eaf_orientation_r` skips the gate rather than failing it, so a
    Catalogue written before this column existed relabels exactly as before.
    """
    if overlap < gates.n_min:
        return "overlap"
    if np.isfinite(eaf_orientation_r) and eaf_orientation_r <= gates.orientation_flip_r:
        return "eaf_orientation"
    if not np.isfinite(residual) or residual > gates.residual_max:
        return "residual"
    if dominant_proportion < gates.tau:
        return "proportion"
    if margin < gates.delta:
        return "margin"
    return "ok"
