"""The EAF survey and orientation verification log each part as a timed phase (#310).

Between Pass 2 and the encoding plan a Dense build ran for hours with no log
line. Each of its parts -- the axis's site hashes, the per-Analysis survey, the
parent's accumulation of the samples, the EAF Reference load and the orientation
verification -- must now say how long it took, and the survey must report
progress, without changing the orientation report.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
from eaf_fixtures import dense_manifest, frequencies, panel_alids, reference_table, write_ssf

from opengwasdb.build.eaf_orientation import (
    EafOrientationError,
    check_eaf_orientation,
    enforce_eaf_orientation,
    load_eaf_reference,
    verify_eaf_orientation,
)
from opengwasdb.layouts.dense import build_vcf
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.model.enums import EafOrientationOutcome

_PROGRESS_STEP = 2
_N_ANALYSES = 5
_TIMED_PARTS = (
    "EAF site hashes",
    "EAF survey",
    "EAF reference load",
    "EAF orientation verification",
    "EAF survey and orientation verification",
)
_LOGGERS = (
    "opengwasdb.layouts.dense.build_vcf",
    "opengwasdb.layouts.hybrid.build",
    "opengwasdb.build.eaf_orientation",
)


@pytest.fixture
def freqs() -> dict[str, float]:
    return frequencies(seed=3)


def _sources(tmp_path: Path, freqs: dict[str, float]) -> dict[str, Path]:
    return {
        f"A{i}": write_ssf(tmp_path / f"a{i}.tsv.gz", freqs, flipped=False, seed=i + 1)
        for i in range(_N_ANALYSES)
    }


def _assert_phase_logged(
    messages: list[str], survey: str = "EAF survey", n_analyses: int = _N_ANALYSES
) -> None:
    for part in _TIMED_PARTS:
        assert any(m.startswith(f"{part}: start") for m in messages), f"{part}: no start line"
        assert any(m.startswith(f"{part}: done in") for m in messages), f"{part}: no end line"
    progress = [m for m in messages if m.startswith(f"{survey}:") and "ETA" in m]
    assert any(f"{n_analyses}/{n_analyses}" in m for m in progress), f"{survey}: no final progress"
    assert len(progress) > 1, f"{survey}: progress did not report between start and end"
    # One summed line, after the loop: it says how long the parent's folding took
    # over all the Analyses, not a start and an end around the whole survey.
    summed = [m for m in messages if m.startswith(f"{survey} accumulation:")]
    assert len(summed) == 1, summed
    assert re.fullmatch(
        rf"{re.escape(survey)} accumulation: done in \S+ across {n_analyses} Analyses", summed[0]
    )


def _run(caplog, monkeypatch, build, tmp_path, freqs, **kwargs) -> list[str]:
    monkeypatch.setattr(build_vcf, "SURVEY_PROGRESS_EVERY", _PROGRESS_STEP)
    sources = _sources(tmp_path, freqs)
    assert len(sources) > _PROGRESS_STEP
    for name in _LOGGERS:
        caplog.set_level(logging.INFO, logger=name)
    build(
        dense_manifest(tmp_path / "manifest.tsv", sources),
        tmp_path / "store.opengwasdb",
        store_id="s",
        release_id="r",
        eaf_reference=reference_table(tmp_path / "reference.tsv", freqs),
        **kwargs,
    )
    return [record.getMessage() for record in caplog.records]


@pytest.mark.parametrize("n_workers", [1, 2])
def test_dense_build_logs_each_part_of_the_eaf_phase(
    tmp_path, freqs, monkeypatch, caplog, n_workers
):
    messages = _run(
        caplog, monkeypatch, build_dense_from_vcf_manifest, tmp_path, freqs, n_workers=n_workers
    )

    _assert_phase_logged(messages)


def test_hybrid_build_logs_the_same_parts_for_each_component(
    tmp_path, freqs, monkeypatch, caplog
):
    panel = panel_alids(tmp_path / "panel.txt", dict(list(freqs.items())[: len(freqs) // 2]))

    messages = _run(
        caplog,
        monkeypatch,
        build_hybrid_from_vcf_manifest,
        tmp_path,
        freqs,
        reference_panel=panel,
    )

    _assert_phase_logged(messages)
    _assert_phase_logged(messages, survey="EAF survey (ovf)")


def _observations(freqs: dict[str, float]) -> dict[str, dict[str, float]]:
    """One passing, one flipped and one too-small-to-verify Analysis."""
    alids = list(freqs)
    return {
        "ok": {a: freqs[a] for a in alids},
        "flipped": {a: 1.0 - freqs[a] for a in alids},
        "small": {a: freqs[a] for a in alids[:50]},
    }


def test_verification_returns_the_report_the_check_computes(tmp_path, freqs):
    """Logging wraps the check; it must not change what the check reports."""
    reference = reference_table(tmp_path / "reference.tsv", freqs)
    observations = _observations(freqs)
    sites = sorted({a for observed in observations.values() for a in observed})
    expected = check_eaf_orientation(
        observations, reference=load_eaf_reference(reference, sites), n_sites=len(sites)
    )
    outcomes = {e.analysis_id: e.outcome for e in expected.evidence}
    # The oracle is only worth anything if it holds a flipped and a flagged Analysis.
    assert outcomes["flipped"] is EafOrientationOutcome.FAILED
    assert outcomes["small"] is EafOrientationOutcome.UNVERIFIED
    assert outcomes["ok"] is EafOrientationOutcome.PASSED

    with pytest.raises(EafOrientationError) as raised:
        verify_eaf_orientation(observations, eaf_reference=reference, allow_unverified=True)
    with pytest.raises(EafOrientationError) as direct:
        enforce_eaf_orientation(expected, allow_unverified=True)
    assert str(raised.value) == str(direct.value)

    without_flipped = {k: v for k, v in observations.items() if k != "flipped"}
    sites = sorted({a for observed in without_flipped.values() for a in observed})
    expected = check_eaf_orientation(
        without_flipped, reference=load_eaf_reference(reference, sites), n_sites=len(sites)
    )
    report = verify_eaf_orientation(
        without_flipped, eaf_reference=reference, allow_unverified=True
    )
    # repr, not ==: an unverified Analysis's correlation is NaN, which is never equal to itself.
    assert repr(report) == repr(expected)
