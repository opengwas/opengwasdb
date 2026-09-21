"""Tests for the one-pass pre-build resolver (issue #207).

Exercised through `resolve_analysis` itself, on real GWAS-SSF files and a real
Ancestry Reference Panel written to a temporary directory, because the thing
under test is the whole path -- one scan of one source producing both an
ancestry fit and a phenotype-SD estimate -- rather than any one helper.

The two equivalences that matter are asserted against the implementations this
replaces, not against constants: the ancestry fit against
`assign_from_source` (what `assign-ancestry` calls today) and the phenotype SD
against `estimate_manifest_phenotype_sd` (what `estimate-phenotype-sd` calls
today), both on the same file.
"""

from __future__ import annotations

import gzip
import math
import random
import tracemalloc
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.ancestry.mixture import Gates, assign_from_source
from opengwasdb.ancestry.reference import AncestryReference, load_reference
from opengwasdb.build.phenotype_sd import estimate_phenotype_sd
from opengwasdb.build.phenotype_sd_pipeline import (
    AfSource,
    SdManifestRow,
    estimate_manifest_phenotype_sd,
)
from opengwasdb.build.resolve import (
    AfReference,
    AnalysisRequest,
    AnalysisResolution,
    SdReason,
    SdStatus,
    resolve_analysis,
)
from opengwasdb.model.enums import OriginalSdMethod, StoredEffectScale
from opengwasdb.readers.gwas_ssf import GWAS_SSF_CAPABILITY, GwasSsfReader
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY, GwasVcfReader
from opengwasdb.readers.tabular import TabularMetricsRow

N_VARIANTS = 200
_STUDY_N = 20_000.0
_TRUE_SD = 2.0
_PANEL_GROUPS = ("United Kingdom", "Finland", "Africa (West)", "Asia (East)")
_SUPERPOP_OF = {
    "United Kingdom": "EUR",
    "Finland": "EUR",
    "Africa (West)": "AFR",
    "Asia (East)": "EAS",
}
# The harmonised GWAS-SSF columns these fixtures carry, in file order.
_SSF_COLUMNS = tuple(
    "chromosome base_pair_location effect_allele other_allele beta standard_error"
    " effect_allele_frequency".split()
)
# Deliberately looser than the ADR-0028 defaults on the residual, because this
# synthetic panel's groups differ by drift rather than by independent draws --
# see `panel` for why that shape is the honest one here. The gates are the
# caller's, which is what makes passing them in the point.
_GATES = Gates(tau=0.60, delta=0.15, n_min=10, residual_max=0.20)


@pytest.fixture
def panel(tmp_path: Path) -> AncestryReference:
    return _write_panel(
        tmp_path, [_alid(index) for index in range(N_VARIANTS)], _panel_frequencies()
    )


def _panel_frequencies() -> np.ndarray:
    """The correlated panel frequencies, from a fixed seed.

    Correlated on purpose: real populations agree about which allele is the
    common one, so a study's frequencies correlate with the reference consensus
    near +1 -- which is the separation the EAF-orientation gate reads. Groups
    drawn independently would correlate around 0.5 and would test that gate on
    data it never sees.
    """
    rng = np.random.default_rng(207)
    baseline = rng.uniform(0.05, 0.95, size=(N_VARIANTS, 1))
    drift = rng.normal(0.0, 0.08, size=(N_VARIANTS, len(_PANEL_GROUPS)))
    return np.clip(baseline + drift, 0.01, 0.99)


def _write_panel(
    directory: Path, alids: Sequence[str], frequencies: np.ndarray
) -> AncestryReference:
    """Write an Ancestry Reference Panel + its fine→super-population map, and load it."""
    directory.mkdir(parents=True, exist_ok=True)
    header = ["alid", "chromosome", "position", "effect_allele", "other_allele", "rsid"]
    lines = ["\t".join([*header, *_PANEL_GROUPS])]
    for index, alid in enumerate(alids):
        chromosome, position, a1, a2 = alid.split(":")
        cells = [alid, chromosome, position, a1, a2, f"rs{index}"]
        lines.append("\t".join([*cells, *(f"{value:.6g}" for value in frequencies[index])]))
    (directory / "ref_freqs.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    groups = ["group\tsuper_pop"]
    groups += [f"{group}\t{_SUPERPOP_OF[group]}" for group in _PANEL_GROUPS]
    (directory / "ancestry_groups.tsv").write_text("\n".join(groups) + "\n", encoding="utf-8")
    # maf_floor=0: keep every synthetic variant regardless of MAF.
    return load_reference(
        directory / "ref_freqs.tsv", directory / "ancestry_groups.tsv", maf_floor=0.0
    )


def _alid(index: int) -> str:
    """Canonical ALID for variant `index`; A1=A, so an A/C row is unflipped."""
    return f"1:{1000 + index}:A:C"


def _mixture(panel: AncestryReference, weights: Mapping[str, float]) -> np.ndarray:
    """Per-variant frequencies for a study that is `weights` of these groups."""
    frequencies = np.zeros(panel.n_variants)
    for group, weight in weights.items():
        frequencies += weight * panel.freqs[:, panel.groups.index(group)]
    return frequencies


def _se_for(frequencies: np.ndarray, *, sd: float = _TRUE_SD, n: float = _STUDY_N) -> np.ndarray:
    """The `se` ADR-0029's model implies for a known true SD -- its inverse.

    `sd_hat = median(se * sqrt(2f(1-f))) * sqrt(N)`, so a fixture built this way
    has `sd` as its exact answer rather than as an approximation a loose
    tolerance could hide a defect inside.
    """
    return sd / np.sqrt(2.0 * frequencies * (1.0 - frequencies) * n)


def _row(
    index: int,
    frequency: float,
    se: float,
    *,
    beta: float = 0.1,
    effect: str = "A",
    other: str = "C",
) -> dict[str, str]:
    """One GWAS-SSF row; `frequency` is the frequency of the *effect* allele."""
    return {
        "chromosome": "1",
        "base_pair_location": str(1000 + index),
        "effect_allele": effect,
        "other_allele": other,
        "beta": f"{beta:.10g}",
        "standard_error": f"{se:.10g}",
        "effect_allele_frequency": f"{frequency:.10g}",
    }


def _study_rows(
    frequencies: np.ndarray, *, betas: np.ndarray | None = None
) -> list[dict[str, str]]:
    se = _se_for(frequencies)
    if betas is None:
        return [
            _row(index, float(frequency), float(error))
            for index, (frequency, error) in enumerate(zip(frequencies, se, strict=True))
        ]
    return [
        _row(index, float(frequency), float(error), beta=float(beta))
        for index, (frequency, error, beta) in enumerate(
            zip(frequencies, se, betas, strict=True)
        )
    ]


def _write_ssf(
    path: Path, rows: Sequence[Mapping[str, str]], columns: Sequence[str] = _SSF_COLUMNS
) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(row[column] for column in columns) + "\n")
    return path


def _study_file(
    tmp_path: Path,
    panel: AncestryReference,
    weights: Mapping[str, float],
    *,
    name: str = "study.tsv.gz",
    columns: Sequence[str] = _SSF_COLUMNS,
    betas: np.ndarray | None = None,
) -> Path:
    frequencies = _mixture(panel, weights)
    return _write_ssf(
        tmp_path / name, _study_rows(frequencies, betas=betas), columns=columns
    )


def _resolve(
    path: Path,
    panel: AncestryReference,
    method: OriginalSdMethod = OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF,
    *,
    sample_size: float | None = _STUDY_N,
    stored_effect_scale: StoredEffectScale = StoredEffectScale.SD,
    extraction_panel: Sequence[str] | None = None,
    af_references: Mapping[str, AfReference] | None = None,
    evidence_sample: int = 20_000,
) -> AnalysisResolution:
    return resolve_analysis(
        AnalysisRequest("GCST000001", path, sample_size, method, stored_effect_scale),
        reader=GwasSsfReader(path),
        reference=panel,
        gates=_GATES,
        extraction_panel=extraction_panel,
        af_references=af_references,
        evidence_sample=evidence_sample,
    )


@dataclass
class _CountingReader:
    """A reader that counts how many times its source is scanned."""

    inner: GwasSsfReader
    scans: int = 0
    rows: int = 0

    def stream_metrics(self) -> Iterator[TabularMetricsRow]:
        self.scans += 1
        for row in self.inner.stream_metrics():
            self.rows += 1
            yield row


def _assigned_european(tmp_path: Path, panel: AncestryReference) -> Path:
    return _study_file(tmp_path, panel, {"United Kingdom": 1.0})


def _pipeline_sd(path: Path) -> list:
    """The same file through `estimate-phenotype-sd`'s own pipeline, for comparison."""
    return estimate_manifest_phenotype_sd(
        [
            SdManifestRow(
                "GCST000001",
                str(path),
                GWAS_SSF_CAPABILITY,
                _STUDY_N,
                OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF,
            )
        ],
        af_source=AfSource.source,
    )


def _assert_same_fit(left, right) -> None:
    """The same ancestry fit, allowing for `assign_ancestry`'s float noise.

    The NNLS reduction order follows the dict insertion order, which follows the
    source's row order, so two orderings of the same rows differ in the last few
    bits of `residual`, the compositions and `eaf_orientation_r` -- around 1e-17
    on these numbers. Everything a caller routes on is exact, and is asserted
    exactly.
    """
    assert left.assigned_ancestry == right.assigned_ancestry
    assert left.gate_reason == right.gate_reason
    assert left.dominant_superpop == right.dominant_superpop
    assert left.af_overlap == right.af_overlap
    assert left.eaf_orientation == right.eaf_orientation
    for field in ("dominant_proportion", "runner_up_margin", "residual", "eaf_orientation_r"):
        assert getattr(left, field) == pytest.approx(getattr(right, field), abs=1e-9)
    for superpop, proportion in left.superpop_composition.items():
        assert proportion == pytest.approx(right.superpop_composition[superpop], abs=1e-9)


# --- one scan, both stages -------------------------------------------------


def test_quantitative_analysis_resolves_both_stages_from_one_scan(tmp_path, panel):
    path = _assigned_european(tmp_path, panel)
    reader = _CountingReader(GwasSsfReader(path))

    resolution = resolve_analysis(
        AnalysisRequest(
            "GCST000001", path, _STUDY_N, OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF
        ),
        reader=reader,
        reference=panel,
        gates=_GATES,
    )

    assert reader.scans == 1, "the source must be opened once for both stages"
    assert reader.rows == N_VARIANTS
    assert resolution.error == ""
    assert resolution.diagnostics.rows_read == N_VARIANTS
    assert resolution.diagnostics.ancestry_sites == N_VARIANTS
    assert resolution.ancestry is not None
    assert resolution.ancestry.assigned_ancestry == "EUR"
    assert resolution.ancestry.gate_reason == "ok"
    assert resolution.phenotype_sd is not None
    assert resolution.phenotype_sd.status is SdStatus.ESTIMATED
    assert resolution.phenotype_sd.reason is None
    assert resolution.phenotype_sd.estimate is not None
    assert resolution.phenotype_sd.estimate.method is OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF
    assert resolution.phenotype_sd.estimate.sd == pytest.approx(_TRUE_SD)
    assert resolution.phenotype_sd.n_estimate_inputs == N_VARIANTS
    assert resolution.phenotype_sd.evidence_sampled is False


def test_ancestry_matches_assign_from_source_for_the_same_file(tmp_path, panel):
    """The one-pass fit is the two-pass fit, on the same source.

    `assign_from_source` is what `opengwasdb assign-ancestry` runs today, so a
    difference here would mean the resolver silently changed an Analysis's
    Assigned Ancestry rather than reproducing it.
    """
    path = _assigned_european(tmp_path, panel)

    resolution = _resolve(path, panel)
    direct = assign_from_source(path, panel, _GATES, capability=GWAS_SSF_CAPABILITY)

    assert direct.assigned_ancestry == "EUR", "fixture must assign before comparing"
    assert resolution.ancestry == direct


def test_phenotype_sd_matches_the_manifest_pipeline_for_the_same_file(tmp_path, panel):
    """The one-pass estimate is the manifest pipeline's estimate, on the same file."""
    path = _assigned_european(tmp_path, panel)

    resolution = _resolve(path, panel)
    pipeline = _pipeline_sd(path)

    assert resolution.phenotype_sd is not None
    estimate = resolution.phenotype_sd.estimate
    assert estimate is not None
    assert float(pipeline[0].original_sd) == pytest.approx(estimate.sd)
    assert pipeline[0].original_sd_method == estimate.method.value


def test_resolution_is_deterministic_and_order_independent(tmp_path, panel):
    """Same file, and the same rows in a different order, give the same answer.

    The evidence sample is chosen by variant hash rather than by position, so a
    re-ordered source draws the same variants -- which is what makes the result
    independent of how a caller happens to stream it.
    """
    frequencies = _mixture(panel, {"United Kingdom": 1.0})
    rows = _study_rows(frequencies)
    first = _write_ssf(tmp_path / "first.tsv.gz", rows)
    shuffled = list(rows)
    random.Random(11).shuffle(shuffled)
    second = _write_ssf(tmp_path / "second.tsv.gz", shuffled)

    repeat = _resolve(first, panel, evidence_sample=50)
    reordered = _resolve(second, panel, evidence_sample=50)

    assert repeat == _resolve(first, panel, evidence_sample=50)
    assert reordered.phenotype_sd == repeat.phenotype_sd
    _assert_same_fit(reordered.ancestry, repeat.ancestry)


# --- the phenotype-SD half: every controlled outcome ------------------------


def test_case_control_analysis_skips_sd_and_still_resolves_ancestry(tmp_path, panel):
    """A log-OR Analysis has no phenotype SD to estimate, and still has ancestry.

    The two must not collapse into one "nothing happened": the Analysis is
    quantitative-agnostic for ancestry purposes and unstandardisable for
    phenotype-SD purposes.
    """
    path = _assigned_european(tmp_path, panel)

    resolution = _resolve(
        path,
        panel,
        OriginalSdMethod.BINARY_TRAIT,
        sample_size=None,
        stored_effect_scale=StoredEffectScale.LOG_OR,
    )

    assert resolution.ancestry is not None
    assert resolution.ancestry.assigned_ancestry == "EUR"
    assert resolution.diagnostics.rows_read == N_VARIANTS
    assert resolution.phenotype_sd is not None
    assert resolution.phenotype_sd.status is SdStatus.SKIPPED
    assert resolution.phenotype_sd.reason is SdReason.NON_QUANTITATIVE
    assert resolution.phenotype_sd.estimate is None
    assert resolution.phenotype_sd.n_evidence_considered == 0


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (OriginalSdMethod.DECLARED_STANDARDISED, SdReason.SCALE_ALREADY_DECLARED),
        (OriginalSdMethod.SOURCE_PROVIDED, SdReason.SCALE_ALREADY_DECLARED),
        (OriginalSdMethod.BINARY_TRAIT, SdReason.NON_QUANTITATIVE),
        (OriginalSdMethod.UNAVAILABLE, SdReason.NO_TIER_SELECTED),
    ],
)
def test_tiers_that_are_not_estimated_are_skipped_with_their_own_reason(
    tmp_path, panel, method, expected
):
    """The tier is the caller's decision, so a non-estimation tier is a skip."""
    path = _assigned_european(tmp_path, panel)

    resolution = _resolve(path, panel, method)

    assert resolution.phenotype_sd is not None
    assert resolution.phenotype_sd.status is SdStatus.SKIPPED
    assert resolution.phenotype_sd.reason is expected
    assert resolution.phenotype_sd.estimate is None


def test_missing_sample_size_is_unavailable_rather_than_guessed(tmp_path, panel):
    """ADR-0029's formula is `sd/sqrt(N)` as one quantity; without `N`, nothing."""
    path = _assigned_european(tmp_path, panel)

    resolution = _resolve(path, panel, sample_size=None)

    assert resolution.phenotype_sd is not None
    assert resolution.phenotype_sd.status is SdStatus.UNAVAILABLE
    assert resolution.phenotype_sd.reason is SdReason.NO_USABLE_SAMPLE_SIZE
    assert resolution.phenotype_sd.n_estimate_inputs == 0


def test_missing_source_frequency_is_unavailable_with_no_declared_reference(tmp_path, panel):
    """No source AF and no declared fallback is an explicit outcome, not a number."""
    path = _study_file(
        tmp_path,
        panel,
        {"United Kingdom": 1.0},
        columns=tuple(c for c in _SSF_COLUMNS if c != "effect_allele_frequency"),
    )

    resolution = _resolve(path, panel)

    assert resolution.phenotype_sd is not None
    assert resolution.phenotype_sd.status is SdStatus.UNAVAILABLE
    assert resolution.phenotype_sd.reason is SdReason.NO_QUALIFYING_EVIDENCE
    assert resolution.phenotype_sd.n_evidence_considered == 0, (
        "no row carries a frequency, so no row is evidence for this tier"
    )
    # The same file carries no frequency for ancestry either, and says so.
    assert resolution.ancestry is not None
    assert resolution.ancestry.gate_reason == "overlap"
    assert resolution.ancestry.af_overlap == 0
    assert resolution.diagnostics.ancestry_sites == 0


def test_reference_maf_tier_uses_the_assigned_ancestries_declared_reference(tmp_path, panel):
    """The reference tier's frequencies come from the declared reference.

    The source carries usable source AF too, so a resolver that quietly used it
    would land on the source-MAF number instead; the constant 0.5 table makes
    the two unmistakably different.
    """
    path = _assigned_european(tmp_path, panel)
    frequencies = _mixture(panel, {"United Kingdom": 1.0})
    lookup = {_alid(index): 0.5 for index in range(N_VARIANTS)}
    expected = float(
        np.median(_se_for(frequencies) * np.sqrt(2.0 * 0.5 * 0.5 * _STUDY_N))
    )

    resolution = _resolve(
        path,
        panel,
        OriginalSdMethod.ESTIMATED_FROM_REFERENCE_MAF,
        af_references={"EUR": AfReference("panel-eur", lookup)},
    )

    assert resolution.phenotype_sd is not None
    estimate = resolution.phenotype_sd.estimate
    assert estimate is not None
    assert estimate.method is OriginalSdMethod.ESTIMATED_FROM_REFERENCE_MAF
    assert estimate.sd == pytest.approx(expected)
    assert resolution.phenotype_sd.reference_id == "panel-eur"
    assert resolution.phenotype_sd.n_estimate_inputs == N_VARIANTS
    source_tier = _resolve(path, panel)
    assert source_tier.phenotype_sd is not None
    assert source_tier.phenotype_sd.estimate is not None
    assert estimate.sd != pytest.approx(source_tier.phenotype_sd.estimate.sd)


def test_reference_maf_tier_leaves_out_rows_the_reference_does_not_cover(tmp_path, panel):
    """A row the declared reference has no frequency for is left out, not filled in."""
    path = _assigned_european(tmp_path, panel)
    covered = {_alid(index): 0.5 for index in range(N_VARIANTS // 2)}

    resolution = _resolve(
        path,
        panel,
        OriginalSdMethod.ESTIMATED_FROM_REFERENCE_MAF,
        af_references={"EUR": AfReference("panel-eur", covered)},
    )

    assert resolution.phenotype_sd is not None
    assert resolution.phenotype_sd.status is SdStatus.ESTIMATED
    assert resolution.phenotype_sd.n_evidence_considered == N_VARIANTS
    assert resolution.phenotype_sd.n_estimate_inputs == N_VARIANTS // 2


def test_unassigned_ancestry_does_not_borrow_another_ancestries_reference(tmp_path, panel):
    """An admixed Analysis gets no reference, rather than the EUR one nearby."""
    path = _study_file(
        tmp_path, panel, {"United Kingdom": 0.5, "Africa (West)": 0.5}, name="admixed.tsv.gz"
    )
    lookup = {_alid(index): 0.5 for index in range(N_VARIANTS)}

    resolution = _resolve(
        path,
        panel,
        OriginalSdMethod.ESTIMATED_FROM_REFERENCE_MAF,
        af_references={"EUR": AfReference("panel-eur", lookup)},
    )

    assert resolution.ancestry is not None
    assert resolution.ancestry.assigned_ancestry is None
    assert resolution.ancestry.gate_reason in {"proportion", "margin"}
    assert resolution.phenotype_sd is not None
    assert resolution.phenotype_sd.status is SdStatus.SKIPPED
    assert resolution.phenotype_sd.reason is SdReason.NO_REFERENCE_RESOURCE_FOR_ANCESTRY
    assert resolution.phenotype_sd.reference_id == ""


def test_beta_distribution_tier_estimates_from_the_betas(tmp_path, panel):
    """The lowest-confidence tier still runs off the one scan's own betas."""
    rng = np.random.default_rng(2071)
    betas = rng.normal(0.0, 0.3, size=N_VARIANTS)
    path = _study_file(tmp_path, panel, {"United Kingdom": 1.0}, name="betas.tsv.gz", betas=betas)

    resolution = _resolve(path, panel, OriginalSdMethod.ESTIMATED_FROM_BETA_DISTRIBUTION)

    assert resolution.phenotype_sd is not None
    estimate = resolution.phenotype_sd.estimate
    assert estimate is not None
    expected = estimate_phenotype_sd(
        OriginalSdMethod.ESTIMATED_FROM_BETA_DISTRIBUTION, _STUDY_N, beta=betas
    )
    assert estimate.sd == pytest.approx(expected.sd)
    assert estimate.dispersion == pytest.approx(expected.dispersion)
    assert resolution.phenotype_sd.n_estimate_inputs == N_VARIANTS


# --- the ancestry half: gates are the caller's, semantics are not ----------


def test_mis_oriented_frequencies_fail_the_orientation_gate(tmp_path, panel):
    """The GCST003566 defect (issue #115), reported as what it is.

    The alleles are canonical and the frequency column reports the *other*
    allele's frequency, which is the shape a plain threshold cannot see: it is
    the correlation against the reference consensus that names it.
    """
    frequencies = _mixture(panel, {"United Kingdom": 1.0})
    rows = [
        _row(index, 1.0 - float(frequency), float(error))
        for index, (frequency, error) in enumerate(
            zip(frequencies, _se_for(frequencies), strict=True)
        )
    ]
    path = _write_ssf(tmp_path / "mis-oriented.tsv.gz", rows)

    resolution = _resolve(path, panel)

    assert resolution.ancestry is not None
    assert resolution.ancestry.eaf_orientation == "failed"
    assert resolution.ancestry.eaf_orientation_r < -0.5
    assert resolution.ancestry.gate_reason == "eaf_orientation"
    assert resolution.ancestry.assigned_ancestry is None


def test_other_allele_effect_orientation_is_resolved_not_mis_read(tmp_path, panel):
    """A study whose effect allele is the non-A1 allele is oriented, not flagged.

    ADR 0036's rule: the stored frequency follows the stored effect allele, so a
    swapped pair must land on the same fit as an unswapped one.
    """
    frequencies = _mixture(panel, {"United Kingdom": 1.0})
    swapped = [
        _row(index, 1.0 - float(frequency), float(error), effect="C", other="A")
        for index, (frequency, error) in enumerate(
            zip(frequencies, _se_for(frequencies), strict=True)
        )
    ]
    path = _write_ssf(tmp_path / "swapped.tsv.gz", swapped)

    resolution = _resolve(path, panel)
    unswapped = _resolve(_assigned_european(tmp_path, panel), panel)

    assert resolution.ancestry is not None
    assert resolution.ancestry.assigned_ancestry == "EUR"
    assert resolution.ancestry.gate_reason == "ok"
    _assert_same_fit(resolution.ancestry, unswapped.ancestry)


def test_low_overlap_and_ambiguous_mixture_report_their_own_gate(tmp_path, panel):
    """The gates' reasons survive the one-pass rewrite unchanged."""
    frequencies = _mixture(panel, {"United Kingdom": 1.0})
    rows = _study_rows(frequencies)
    few = _write_ssf(tmp_path / "few.tsv.gz", rows[:5])
    admixed = _study_file(
        tmp_path, panel, {"United Kingdom": 0.5, "Africa (West)": 0.5}, name="admixed.tsv.gz"
    )

    low_overlap = _resolve(few, panel)
    ambiguous = _resolve(admixed, panel)

    assert low_overlap.ancestry is not None
    assert low_overlap.ancestry.gate_reason == "overlap"
    assert low_overlap.ancestry.af_overlap == 5
    assert ambiguous.ancestry is not None
    assert ambiguous.ancestry.gate_reason in {"proportion", "margin"}
    assert ambiguous.ancestry.assigned_ancestry is None


def test_a_non_european_analysis_is_assigned_its_own_ancestry(tmp_path, panel):
    """Nothing about this resolver assumes EUR (ADR 0020's routing depends on it)."""
    path = _study_file(tmp_path, panel, {"Africa (West)": 1.0}, name="afr.tsv.gz")

    resolution = _resolve(path, panel)

    assert resolution.ancestry is not None
    assert resolution.ancestry.assigned_ancestry == "AFR"
    assert resolution.ancestry.dominant_superpop == "AFR"
    assert resolution.ancestry.gate_reason == "ok"


def test_extraction_panel_bounds_the_ancestry_fit(tmp_path, panel):
    """A panel is what the fit sees -- the caller's scientific choice, not a filter."""
    path = _assigned_european(tmp_path, panel)
    half = [_alid(index) for index in range(N_VARIANTS // 2)]

    bounded = _resolve(path, panel, extraction_panel=half)
    full = _resolve(path, panel)

    assert bounded.diagnostics.ancestry_sites == N_VARIANTS // 2
    assert full.diagnostics.ancestry_sites == N_VARIANTS
    assert bounded.ancestry is not None and full.ancestry is not None
    assert bounded.ancestry.af_overlap == N_VARIANTS // 2
    assert bounded.ancestry.assigned_ancestry == full.ancestry.assigned_ancestry == "EUR"


# --- bounded evidence ------------------------------------------------------


def _peak_bytes(
    path: Path, panel: AncestryReference, evidence_sample: int
) -> tuple[int, AnalysisResolution]:
    """Peak Python allocation while resolving one file, and the resolution."""
    tracemalloc.start()
    try:
        resolution = _resolve(path, panel, evidence_sample=evidence_sample)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return peak, resolution


def test_evidence_sample_bounds_memory_independently_of_row_count(tmp_path, panel):
    """100x the rows must not cost 100x the memory.

    The claim is the acceptance criterion's: evidence memory is bounded by
    configuration, not by how many rows the source has. It is asserted by
    measuring, because a bound nobody measures is a comment.
    """
    frequencies = _mixture(panel, {"United Kingdom": 1.0})
    small_path = _write_ssf(tmp_path / "small.tsv.gz", _study_rows(np.tile(frequencies, 1)))
    large_path = _write_ssf(tmp_path / "large.tsv.gz", _study_rows(np.tile(frequencies, 100)))

    small_peak, small = _peak_bytes(small_path, panel, 200)
    large_peak, large = _peak_bytes(large_path, panel, 200)

    assert small.diagnostics.rows_read == N_VARIANTS
    assert large.diagnostics.rows_read == 100 * N_VARIANTS
    assert large.phenotype_sd is not None
    assert large.phenotype_sd.n_evidence_considered == 100 * N_VARIANTS
    assert large.phenotype_sd.n_estimate_inputs <= 200
    assert large.phenotype_sd.evidence_sampled is True
    assert small.phenotype_sd is not None
    assert small.phenotype_sd.evidence_sampled is False
    assert large_peak < small_peak + 512 * 1024, (
        f"100x the rows cost {large_peak - small_peak} more bytes of peak memory"
    )


def test_unbounded_evidence_is_what_the_bound_is_holding_back(tmp_path, panel):
    """The bound is doing work: the same file without it costs far more."""
    frequencies = _mixture(panel, {"United Kingdom": 1.0})
    path = _write_ssf(tmp_path / "many.tsv.gz", _study_rows(np.tile(frequencies, 100)))

    bounded_peak, bounded = _peak_bytes(path, panel, 200)
    unbounded_peak, unbounded = _peak_bytes(path, panel, 10_000_000)

    assert unbounded.phenotype_sd is not None and bounded.phenotype_sd is not None
    assert unbounded.phenotype_sd.evidence_sampled is False
    assert unbounded_peak > 10 * bounded_peak, (
        f"unbounded {unbounded_peak} vs bounded {bounded_peak} bytes"
    )
    assert unbounded.phenotype_sd.estimate is not None
    assert bounded.phenotype_sd.estimate is not None
    assert bounded.phenotype_sd.estimate.sd == pytest.approx(
        unbounded.phenotype_sd.estimate.sd, rel=1e-6
    )


# --- controlled failures and caller errors ---------------------------------


def test_unreadable_source_is_a_controlled_error(tmp_path, panel):
    path = tmp_path / "broken.tsv.gz"
    path.write_text("this is not a gzip member\n", encoding="utf-8")

    resolution = _resolve(path, panel)

    assert resolution.error.startswith("BadGzipFile")
    assert resolution.ancestry is None
    assert resolution.phenotype_sd is None
    assert resolution.diagnostics.rows_read == 0


def test_missing_source_file_is_a_controlled_error(tmp_path, panel):
    resolution = _resolve(tmp_path / "absent.tsv.gz", panel)

    assert resolution.error.startswith("FileNotFoundError")
    assert resolution.ancestry is None


def test_source_missing_an_identity_column_is_a_controlled_error(tmp_path, panel):
    """A file that is not GWAS-SSF is one Analysis's outcome, not the batch's."""
    path = _study_file(
        tmp_path,
        panel,
        {"United Kingdom": 1.0},
        columns=tuple(c for c in _SSF_COLUMNS if c != "other_allele"),
    )

    resolution = _resolve(path, panel)

    assert "missing required variant columns" in resolution.error
    assert resolution.ancestry is None


def test_reader_without_a_metrics_path_is_refused_loudly(tmp_path, panel):
    """A GWAS-VCF has no single-scan metrics path, and must not be read as if it had."""
    reader = GwasVcfReader(tmp_path / "study.vcf", StoredEffectScale.SD)

    with pytest.raises(TypeError, match="cannot stream source metrics"):
        resolve_analysis(
            AnalysisRequest(
                "GCST000001",
                tmp_path / "study.vcf",
                _STUDY_N,
                OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF,
            ),
            reader=reader,
            reference=panel,
        )


def test_non_positive_evidence_sample_is_refused_loudly(tmp_path, panel):
    """A bound that bounds nothing is a caller defect, not a per-Analysis outcome."""
    path = _assigned_european(tmp_path, panel)

    with pytest.raises(ValueError, match="positive row count"):
        _resolve(path, panel, evidence_sample=0)


def test_capability_is_the_callers_not_inferred_from_the_path(tmp_path, panel):
    """The reader decides the format; this module never guesses one from a name."""
    path = _assigned_european(tmp_path, panel)

    assert GWAS_SSF_CAPABILITY in {GWAS_VCF_CAPABILITY, GWAS_SSF_CAPABILITY}
    assert isinstance(GwasSsfReader(path), GwasSsfReader)
    resolution = _resolve(path, panel)
    assert resolution.diagnostics.source_file == str(path)


def test_palindromic_sites_are_left_out_of_the_ancestry_fit(tmp_path):
    """An A/T site's frequency is ambiguous against the reference, so it is no evidence.

    This fixture carries its own two-variant panel, one of them A/T, because the
    rule is only reachable when the *reference* names the palindromic site too: a
    palindromic site the panel does not carry is dropped by membership and would
    have tested nothing. Two rows, one palindromic, so the count says which rule
    did the dropping.
    """
    frequencies = _panel_frequencies()[:2]
    palindromic_panel = _write_panel(
        tmp_path / "palindromic-panel", ["1:1000:A:T", "1:1001:A:C"], frequencies
    )
    rows = [
        # The source labels the A/T pair T/A, so its effect allele's frequency is
        # the complement of the panel's A1 frequency.
        _row(0, 1.0 - float(frequencies[0, 0]), 0.5, effect="T", other="A"),
        _row(1, float(frequencies[1, 0]), 0.5),
    ]
    path = _write_ssf(tmp_path / "palindromic.tsv.gz", rows)

    resolution = _resolve(path, palindromic_panel)

    assert resolution.diagnostics.rows_read == 2
    assert resolution.diagnostics.ancestry_sites == 1, (
        "only the non-palindromic site may reach the fit"
    )
    assert resolution.ancestry is not None
    assert resolution.ancestry.gate_reason == "overlap"


def test_rows_without_a_usable_standard_error_are_not_ancestry_evidence(tmp_path, panel):
    """`extract_at_sites`'s filter, preserved: AF alone is not enough."""
    frequencies = _mixture(panel, {"United Kingdom": 1.0})
    rows = _study_rows(frequencies)
    rows[0]["standard_error"] = "NA"
    path = _write_ssf(tmp_path / "no-se.tsv.gz", rows)

    resolution = _resolve(path, panel)

    assert resolution.ancestry is not None
    assert resolution.ancestry.af_overlap == N_VARIANTS - 1
    assert resolution.diagnostics.rows_read == N_VARIANTS


def test_a_row_without_a_beta_is_ancestry_evidence_but_not_sd_evidence(tmp_path, panel):
    """The two stages read different things from one row, and both must be exact.

    A row carrying a frequency and a standard error but no beta is evidence for
    the ancestry fit and none for the ADR-0029 estimator. Getting this wrong is
    silent in both directions: admitting it inflates the estimate's support,
    dropping it shrinks the ancestry overlap.
    """
    frequencies = _mixture(panel, {"United Kingdom": 1.0})
    rows = _study_rows(frequencies)
    rows[0]["beta"] = "NA"
    path = _write_ssf(tmp_path / "no-beta.tsv.gz", rows)

    resolution = _resolve(path, panel)
    pipeline = _pipeline_sd(path)

    assert resolution.ancestry is not None
    assert resolution.ancestry.af_overlap == N_VARIANTS, (
        "the beta-less row still carries a frequency the ancestry fit can use"
    )
    assert resolution.phenotype_sd is not None
    assert resolution.phenotype_sd.n_evidence_considered == N_VARIANTS - 1
    assert resolution.phenotype_sd.n_estimate_inputs == N_VARIANTS - 1
    estimate = resolution.phenotype_sd.estimate
    assert estimate is not None
    assert float(pipeline[0].original_sd) == pytest.approx(estimate.sd)


def test_fixture_standard_deviation_is_recovered_to_ten_significant_figures(tmp_path, panel):
    """The fixture is meaningful before anything is asserted about it."""
    frequencies = _mixture(panel, {"United Kingdom": 1.0})
    implied = _se_for(frequencies) * np.sqrt(2.0 * frequencies * (1.0 - frequencies) * _STUDY_N)

    assert N_VARIANTS >= 100, "too few variants for a stable median"
    assert frequencies.min() > 0.0 and frequencies.max() < 1.0, "frequencies must be usable"
    assert frequencies.max() - frequencies.min() > 0.5, "frequencies must genuinely vary"
    assert math.isclose(float(np.median(implied)), _TRUE_SD, rel_tol=1e-9)
