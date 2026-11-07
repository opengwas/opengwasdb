"""The resolver checks a source's effect against its own p-value (#345, ADR 0069).

A source whose `beta` holds odds ratios, or whose `odds_ratio` holds log-odds,
is read without complaint -- both are finite numbers -- and the z-score a build
would store from it is wrong by a scale. The source's own p-value is the
independent witness. These tests drive `resolve_analysis` and
`resolve_analyses_manifest` on small GWAS-SSF files built here, so the verdict
each is expected to reach comes from how the file was made, not from running the
check on it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from test_resolve import _GATES, _alid, _panel_frequencies
from test_resolve import _write_panel as _write_resolve_panel
from test_resolve_manifest import _write_panel as _write_manifest_panel

from opengwasdb.ancestry.reference import AncestryReference
from opengwasdb.build.effect_consistency import (
    ConsistencySample,
    ConsistencySkip,
    EffectConsistencyPolicy,
)
from opengwasdb.build.resolve import AnalysisRequest, AnalysisResolution, resolve_analysis
from opengwasdb.build.resolve_manifest import resolve_analyses_manifest
from opengwasdb.model.enums import OriginalSdMethod, StoredEffectScale
from opengwasdb.readers.column_map import COLUMN_MAP_CAPABILITY, as_column_map
from opengwasdb.readers.finngen import FINNGEN_R13_CAPABILITY
from opengwasdb.readers.gwas_ssf import _METRICS_COLUMNS, GwasSsfReader
from opengwasdb.readers.registry import resolve_reader
from opengwasdb.readers.tabular import MetricsChunk, stream_projected_metric_chunks

N_ROWS = 120
_SE = 0.05
#: |z| from 0.5 to 12: source p-values from ~0.6 down to ~1e-33.
_Z = np.concatenate([np.linspace(0.5, 12.0, N_ROWS // 2), -np.linspace(0.6, 11.0, N_ROWS // 2)])


@pytest.fixture
def panel(tmp_path: Path) -> AncestryReference:
    return _write_resolve_panel(
        tmp_path, [_alid(index) for index in range(200)], _panel_frequencies()
    )


def _two_sided_p(z: np.ndarray) -> np.ndarray:
    return 2.0 * stats.norm.sf(np.abs(z))


def _write(path: Path, columns: Sequence[str], cells: dict[str, Sequence[object]]) -> Path:
    """One GWAS-SSF file; `cells` maps a column name to its per-row values."""
    rows = len(next(iter(cells.values())))
    lines = ["\t".join(columns)]
    for index in range(rows):
        fixed = {
            "chromosome": "1",
            "base_pair_location": str(1000 + index),
            "effect_allele": "A",
            "other_allele": "C",
            "effect_allele_frequency": "0.3",
        }
        values = {**fixed, **{name: str(column[index]) for name, column in cells.items()}}
        lines.append("\t".join(values[column] for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


_BASE = ["chromosome", "base_pair_location", "effect_allele", "other_allele"]


def _beta_file(tmp_path: Path, *, beta_holds: str = "beta", p_scale: float = 1.0) -> Path:
    """A source built from known z: beta = z * se, p = the two-sided p of z.

    `beta_holds="odds_ratio"` writes exp(beta) into the `beta` column -- the
    defect of GCST002047.
    """
    beta = _Z * _SE
    shown = np.exp(beta) if beta_holds == "odds_ratio" else beta
    return _write(
        tmp_path / "beta.tsv",
        [*_BASE, "beta", "standard_error", "p_value", "effect_allele_frequency"],
        {
            "beta": [f"{v:.10g}" for v in shown],
            "standard_error": [f"{_SE:.10g}"] * N_ROWS,
            "p_value": [f"{v:.6g}" for v in _two_sided_p(_Z) * p_scale],
        },
    )


def _log_odds_in_odds_ratio_file(tmp_path: Path) -> Path:
    """The defect of GCST004461: log-odds written into `odds_ratio`.

    Every effect is positive so every row stays usable and the row-wise reading
    `log(odds_ratio)` is finite on all of them.
    """
    z = np.abs(_Z)
    return _write(
        tmp_path / "logodds.tsv",
        [*_BASE, "odds_ratio", "standard_error", "p_value"],
        {
            "odds_ratio": [f"{v:.10g}" for v in z * _SE],
            "standard_error": [f"{_SE:.10g}"] * N_ROWS,
            "p_value": [f"{v:.6g}" for v in _two_sided_p(z)],
        },
    )


def _chunk_of_wrong_scale_rows(n: int) -> MetricsChunk:
    """`n` rows whose stored z (20) contradicts their source p (0.5) wildly."""
    return MetricsChunk(
        alid=np.array([f"1:{1000 + i}:A:C" for i in range(n)], dtype=object),
        flipped=np.zeros(n, dtype=bool),
        palindromic=np.zeros(n, dtype=bool),
        af_alt=np.full(n, 0.3),
        beta=np.full(n, 1.0),
        se=np.full(n, 0.05),
        p_value=np.full(n, 0.5),
    )


def _resolve(
    path: Path,
    panel: AncestryReference,
    *,
    policy: EffectConsistencyPolicy | None = None,
    evidence_sample: int = 20_000,
) -> AnalysisResolution:
    request = AnalysisRequest(
        "GCST000001", path, 20_000.0, OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF,
        StoredEffectScale.SD,
        **({} if policy is None else {"effect_consistency_policy": policy}),
    )
    return resolve_analysis(
        request,
        reader=GwasSsfReader(path),
        reference=panel,
        gates=_GATES,
        evidence_sample=evidence_sample,
    )


def _projected(path: Path):
    chunks = list(stream_projected_metric_chunks(path, _METRICS_COLUMNS))
    return {
        name: np.concatenate([getattr(chunk, name) for chunk in chunks])
        for name in ("beta", "se", "p_value", "se_from_p_value")
    }


def _assert_meaningful(path: Path, *, admitted: int = N_ROWS) -> dict[str, np.ndarray]:
    """The fixture reaches the check: rows admitted, p-values present and wide-ranging."""
    columns = _projected(path)
    assert len(columns["beta"]) == admitted
    assert np.isfinite(columns["p_value"]).all()
    log_p = np.log10(columns["p_value"])
    assert log_p.max() - log_p.min() > 20, "p-values must span orders of magnitude"
    return columns


# --- refusal -----------------------------------------------------------------


def test_a_beta_column_holding_odds_ratios_is_refused(tmp_path: Path, panel):
    path = _beta_file(tmp_path, beta_holds="odds_ratio")
    columns = _assert_meaningful(path)
    # The effects really are on the wrong scale: every value is exp(true beta),
    # near 1, where a beta of this study would be within +-0.6.
    assert np.allclose(np.log(columns["beta"]), _Z * _SE, atol=1e-8)
    assert columns["beta"].min() > 0.5

    result = _resolve(path, panel)

    assert result.error, "an effect on the wrong scale must fail the resolution"
    assert "p-value" in result.error and "threshold 1" in result.error
    assert result.phenotype_sd is None
    measured = result.diagnostics.effect_consistency
    assert measured is not None and measured.refused
    assert measured.n_rows_compared == N_ROWS
    assert measured.median_abs_delta is not None and measured.median_abs_delta > 5.0
    assert f"{measured.median_abs_delta:.3g}" in result.error


def test_an_odds_ratio_column_holding_log_odds_is_refused(tmp_path: Path, panel):
    path = _log_odds_in_odds_ratio_file(tmp_path)
    columns = _assert_meaningful(path)
    # What the build would store is log(log-odds): negative for every row here,
    # which no study of effects this size produces by chance.
    assert (columns["beta"] < 0).all()

    result = _resolve(path, panel)

    assert result.error
    assert result.diagnostics.effect_consistency is not None
    assert result.diagnostics.effect_consistency.refused


# --- resolves, measurement recorded ------------------------------------------


def test_a_clean_source_resolves_with_the_measurement_recorded(tmp_path: Path, panel):
    path = _beta_file(tmp_path)
    _assert_meaningful(path)

    result = _resolve(path, panel)

    assert result.error == ""
    measured = result.diagnostics.effect_consistency
    assert measured is not None
    assert measured.skip_reason is None and not measured.refused
    assert measured.n_rows_considered == N_ROWS
    assert measured.n_rows_compared == N_ROWS and not measured.sampled
    # p is written to 6 significant figures: |Δ| is at most 0.5e-5 / ln(10) / 1.
    assert measured.median_abs_delta is not None and measured.median_abs_delta < 1e-5


def test_the_measurement_is_a_hand_checkable_literal_on_a_tiny_source(tmp_path: Path, panel):
    # Three rows of beta 0.2, se 0.1: z = 2, whose two-sided p is 0.0455003.
    # Each reports p = 0.00455, a tenth of that, so Δ = log10(0.0455003 / 0.00455)
    # = +1.0000025 on every row.
    path = _write(
        tmp_path / "tiny.tsv",
        [*_BASE, "beta", "standard_error", "p_value"],
        {"beta": ["0.2"] * 3, "standard_error": ["0.1"] * 3, "p_value": ["0.00455"] * 3},
    )
    policy = EffectConsistencyPolicy(max_median_abs_delta=5.0, min_compared_rows=1)

    measured = _resolve(path, panel, policy=policy).diagnostics.effect_consistency

    assert measured is not None
    assert measured.n_rows_compared == 3
    assert measured.median_abs_delta == pytest.approx(1.0, abs=1e-4)
    assert measured.median_delta == pytest.approx(1.0, abs=1e-4)
    assert not measured.refused


def test_the_sample_is_bounded_and_says_so(tmp_path: Path, panel):
    path = _beta_file(tmp_path)
    _assert_meaningful(path)
    policy = EffectConsistencyPolicy(min_compared_rows=10)

    measured = _resolve(
        path, panel, policy=policy, evidence_sample=40
    ).diagnostics.effect_consistency

    assert measured is not None
    assert measured.n_rows_considered == N_ROWS
    assert measured.n_rows_compared == 40 and measured.sampled


def test_a_verdict_needs_the_minimum_rows_actually_compared_not_merely_offered():
    # 120 comparable rows offered, one retained by a bound of 1: a median of one
    # row must not refuse an Analysis under a minimum of 50.
    sample = ConsistencySample(1)
    sample.admit(_chunk_of_wrong_scale_rows(N_ROWS))

    measured = sample.measure(EffectConsistencyPolicy())

    assert measured.n_rows_considered == N_ROWS
    assert measured.skip_reason is ConsistencySkip.TOO_FEW_ROWS
    assert not measured.refused and measured.median_abs_delta is None


def test_a_sample_bound_below_the_minimum_is_refused_loudly(tmp_path: Path, panel):
    path = _beta_file(tmp_path)
    with pytest.raises(ValueError, match=r"evidence_sample.*\b1\b.*min_compared_rows.*\b50\b"):
        _resolve(path, panel, evidence_sample=1)


def test_the_manifest_run_refuses_a_sample_bound_below_the_minimum(tmp_path: Path):
    reference, groups, _ = _write_manifest_panel(tmp_path / "reference")
    manifest = _manifest(tmp_path, {"X": _beta_file(tmp_path)})
    with pytest.raises(ValueError, match=r"evidence_sample.*min_compared_rows"):
        resolve_analyses_manifest(
            manifest, tmp_path / "records", ancestry_reference=reference,
            ancestry_groups=groups, reference_build="GRCh38", n_workers=1, evidence_sample=10,
        )


def test_implied_p_survives_a_z_where_the_naive_form_underflows(tmp_path: Path, panel):
    # |z| = 137: 2 * sf(137) is 0.0 in float64, so a naive implementation gives
    # log10(0) = -inf. The reported p, 1e-3000 in the real world, is written at
    # the smallest value a float holds so the row stays comparable.
    assert 2.0 * stats.norm.sf(137.0) == 0.0
    path = _write(
        tmp_path / "huge.tsv",
        [*_BASE, "beta", "standard_error", "p_value"],
        {"beta": ["13.7"] * 3, "standard_error": ["0.1"] * 3, "p_value": ["5e-324"] * 3},
    )
    policy = EffectConsistencyPolicy(max_median_abs_delta=1e6, min_compared_rows=1)

    measured = _resolve(path, panel, policy=policy).diagnostics.effect_consistency

    assert measured is not None and measured.median_abs_delta is not None
    assert math.isfinite(measured.median_abs_delta)
    # By hand: ln p = ln 2 - z²/2 - ln(z√(2π)) = 0.693 - 9384.5 - 5.839 = -9389.65,
    # so log10 p = -4077.87; the reported 5e-324 is log10 = -323.31. Δ = -3754.56.
    assert measured.median_delta == pytest.approx(-3754.56, abs=0.1)


# --- a FinnGen source, through its map and through its reader -----------------


def _finngen_file(tmp_path: Path, *, beta_holds: str = "beta") -> Path:
    """A FinnGen R13 endpoint file: `pval` beside `beta` and `sebeta`."""
    beta = _Z * _SE
    shown = np.exp(beta) if beta_holds == "odds_ratio" else beta
    lines = ["#chrom\tpos\tref\talt\trsids\tnearest_genes\tpval\tmlogp\tbeta\tsebeta\taf_alt"]
    for index in range(N_ROWS):
        p = _two_sided_p(_Z[index : index + 1])[0]
        lines.append(
            f"1\t{1000 + index}\tC\tA\t\tG\t{p:.6g}\t0\t{shown[index]:.10g}\t{_SE}\t0.3"
        )
    path = tmp_path / f"finngen_{beta_holds}.tsv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


#: The two ways a FinnGen R13 Analysis is read: the built-in Column Map, and the
#: FinnGen reader's own capability. Each must carry `pval` to the check.
_FINNGEN_READERS = ("finngen-r13 map", "FinnGen reader")


def _resolve_finngen(path: Path, panel: AncestryReference, read_by: str) -> AnalysisResolution:
    request = AnalysisRequest(
        "FG1", path, 20_000.0, OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF, StoredEffectScale.SD
    )
    if read_by == "FinnGen reader":
        reader = resolve_reader(FINNGEN_R13_CAPABILITY, path, StoredEffectScale.SD)
    else:
        reader = resolve_reader(
            COLUMN_MAP_CAPABILITY, path, StoredEffectScale.SD,
            column_map=as_column_map("finngen-r13"),
        )
    return resolve_analysis(request, reader=reader, reference=panel, gates=_GATES)


@pytest.mark.parametrize("read_by", _FINNGEN_READERS)
def test_a_finngen_source_with_a_mislabelled_effect_is_refused(
    tmp_path: Path, panel, read_by: str
):
    path = _finngen_file(tmp_path, beta_holds="odds_ratio")
    result = _resolve_finngen(path, panel, read_by)

    measured = result.diagnostics.effect_consistency
    assert measured is not None
    assert measured.skip_reason is None, f"the {read_by} must carry `pval` to the check"
    assert measured.refused and result.error


@pytest.mark.parametrize("read_by", _FINNGEN_READERS)
def test_a_clean_finngen_source_is_compared_and_passes(tmp_path: Path, panel, read_by: str):
    result = _resolve_finngen(_finngen_file(tmp_path), panel, read_by)

    assert result.error == ""
    measured = result.diagnostics.effect_consistency
    assert measured is not None and measured.skip_reason is None
    assert measured.n_rows_compared == N_ROWS and not measured.refused


# --- skips are recorded ------------------------------------------------------


def test_a_source_with_no_p_value_column_is_skipped_with_a_reason(tmp_path: Path, panel):
    path = _write(
        tmp_path / "nop.tsv",
        [*_BASE, "beta", "standard_error"],
        {"beta": [f"{v:.8g}" for v in _Z * _SE], "standard_error": [str(_SE)] * N_ROWS},
    )
    assert np.isfinite(_projected(path)["beta"]).all()

    result = _resolve(path, panel)

    assert result.error == ""
    measured = result.diagnostics.effect_consistency
    assert measured is not None and not measured.refused
    assert measured.skip_reason is ConsistencySkip.NO_SOURCE_P_VALUE
    assert measured.median_abs_delta is None


def test_a_source_whose_every_se_comes_from_its_p_value_is_skipped(tmp_path: Path, panel):
    path = _write(
        tmp_path / "pse.tsv",
        [*_BASE, "beta", "p_value"],
        {
            "beta": [f"{v:.8g}" for v in _Z * _SE],
            "p_value": [f"{v:.6g}" for v in _two_sided_p(_Z)],
        },
    )
    columns = _assert_meaningful(path)
    assert columns["se_from_p_value"].all(), "every standard error must be derived"

    result = _resolve(path, panel)

    assert result.error == ""
    measured = result.diagnostics.effect_consistency
    assert measured is not None and not measured.refused
    assert measured.skip_reason is ConsistencySkip.ALL_SE_FROM_P_VALUE


def test_too_few_comparable_rows_is_skipped_not_judged(tmp_path: Path, panel):
    # Ten rows of odds ratios in `beta`: wrong, but fewer rows than the minimum.
    full = _beta_file(tmp_path, beta_holds="odds_ratio")
    lines = full.read_text().splitlines()
    small = tmp_path / "few.tsv"
    small.write_text("\n".join(lines[:11]) + "\n")
    assert len(_projected(small)["beta"]) == 10

    result = _resolve(small, panel, policy=EffectConsistencyPolicy(min_compared_rows=50))

    assert result.error == ""
    measured = result.diagnostics.effect_consistency
    assert measured is not None and not measured.refused
    assert measured.skip_reason is ConsistencySkip.TOO_FEW_ROWS
    assert measured.n_rows_considered == 10


# --- the Resolution Record ---------------------------------------------------


def _manifest(tmp_path: Path, sources: dict[str, Path]) -> Path:
    columns = [
        "analysis_id", "source_file", "source_reader_capability", "source_genome_build",
        "stored_effect_scale", "original_sd_method", "sample_size",
    ]
    lines = ["\t".join(columns)]
    for analysis_id, source in sources.items():
        lines.append(
            "\t".join(
                [analysis_id, str(source), "opengwasdb.gwas-ssf", "GRCh38",
                 "sd", "source_provided", "1000"]
            )
        )
    path = tmp_path / "manifest.tsv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _records(tmp_path: Path, sources: dict[str, Path], **kwargs) -> dict[str, dict]:
    reference, groups, _ = _write_manifest_panel(tmp_path / "reference")
    summary = resolve_analyses_manifest(
        _manifest(tmp_path, sources),
        tmp_path / "records",
        ancestry_reference=reference,
        ancestry_groups=groups,
        reference_build="GRCh38",
        n_workers=1,
        **kwargs,
    )
    return {
        analysis_id: json.loads((summary.records_dir / f"{analysis_id}.json").read_text())
        for analysis_id in sources
    }


def test_the_record_carries_the_measurement_the_skip_and_the_refusal(tmp_path: Path):
    clean = _beta_file(tmp_path)
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad = _beta_file(bad_dir, beta_holds="odds_ratio")
    no_p = _write(
        tmp_path / "nop.tsv",
        [*_BASE, "beta", "standard_error"],
        {"beta": [f"{v:.8g}" for v in _Z * _SE], "standard_error": [str(_SE)] * N_ROWS},
    )

    records = _records(tmp_path, {"CLEAN": clean, "BAD": bad, "NOP": no_p})

    measured = records["CLEAN"]["diagnostics"]["effect_consistency"]
    assert records["CLEAN"]["status"] == "success"
    assert measured["skip_reason"] is None and measured["refused"] is False
    assert measured["n_rows_compared"] == N_ROWS and measured["sampled"] is False
    assert measured["median_abs_delta"] < 1e-5
    assert set(measured) == {
        "n_rows_considered", "n_rows_compared", "sampled", "median_abs_delta",
        "median_delta", "skip_reason", "refused",
    }

    refused = records["BAD"]
    assert refused["status"] == "controlled_failure"
    assert "p-value" in refused["error"]
    assert refused["diagnostics"]["effect_consistency"]["refused"] is True
    assert refused["diagnostics"]["effect_consistency"]["median_abs_delta"] > 5.0

    skipped = records["NOP"]["diagnostics"]["effect_consistency"]
    assert records["NOP"]["status"] == "success"
    assert skipped["skip_reason"] == "no_source_p_value"
    assert skipped["median_abs_delta"] is None


def test_the_threshold_is_bound_into_the_fingerprint(tmp_path: Path):
    path = _beta_file(tmp_path)
    default = _records(tmp_path / "a", {"X": path})["X"]["fingerprints"]
    looser = _records(
        tmp_path / "b", {"X": path},
        effect_consistency_policy=EffectConsistencyPolicy(max_median_abs_delta=3.0),
    )["X"]["fingerprints"]

    assert default["resolution_config"]["effect_consistency"] == {
        "max_median_abs_delta": 1.0, "min_compared_rows": 50,
    }
    assert looser["resolution_config"]["effect_consistency"]["max_median_abs_delta"] == 3.0
    assert default["fingerprint_digest"] != looser["fingerprint_digest"]


def test_a_record_with_the_measurement_resumes_under_its_policy_and_not_another(tmp_path: Path):
    path = _beta_file(tmp_path)
    reference, groups, _ = _write_manifest_panel(tmp_path / "reference")
    manifest = _manifest(tmp_path, {"X": path})

    def run(policy: EffectConsistencyPolicy):
        return resolve_analyses_manifest(
            manifest, tmp_path / "records", ancestry_reference=reference,
            ancestry_groups=groups, reference_build="GRCh38", n_workers=1,
            resume=True, effect_consistency_policy=policy,
        )

    first = run(EffectConsistencyPolicy())
    stored = json.loads((first.records_dir / "X.json").read_text())
    assert first.n_resumed == 0

    again = run(EffectConsistencyPolicy())
    assert again.n_resumed == 1
    assert json.loads((again.records_dir / "X.json").read_text()) == stored

    changed = run(EffectConsistencyPolicy(max_median_abs_delta=2.0))
    assert changed.n_resumed == 0


def test_an_unscanned_source_has_no_measurement_not_a_skip(tmp_path: Path, panel):
    """A source never read is not a measured absence of p-values (final review)."""
    missing = tmp_path / "nowhere.tsv"
    request = AnalysisRequest(
        "GCST000001", missing, 20_000.0, OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF,
        StoredEffectScale.SD,
    )
    result = resolve_analysis(
        request, reader=GwasSsfReader(missing), reference=panel, gates=_GATES,
    )

    assert result.error.startswith("FileNotFoundError")
    assert result.diagnostics.rows_read == 0
    assert result.diagnostics.effect_consistency is None
