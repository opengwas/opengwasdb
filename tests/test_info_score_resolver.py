"""Public resolve-analyses INFO filtering and canonical-row accounting (#175)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_resolve_manifest import _write_panel

from opengwasdb.build.resolve_manifest import read_resolve_manifest, resolve_analyses_manifest
from opengwasdb.model.info_score_policy import InfoScorePolicy, InfoScoreState
from opengwasdb.readers.interface import ImputationScoreDeclaration, ImputationScoreKind

DECLARATION = ImputationScoreDeclaration(
    "quality_metric", ImputationScoreKind.IMPUTATION_INFO, "Provider Table 2 INFO"
)


@pytest.fixture
def sources(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    ref, groups, _ = _write_panel(tmp_path / "reference")
    axis = tmp_path / "axis.tsv"
    axis.write_text("alid\n" + "\n".join(f"1:{1000 + i}:A:C" for i in range(50)) + "\n")
    source = tmp_path / "scores.tsv"
    source.write_text(
        "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\t"
        "standard_error\teffect_allele_frequency\tquality_metric\tINFO\n"
        + "\n".join(
            f"1\t{pos}\tA\tC\t{beta}\t0.1\t0.3\t{score}\t0.99"
            for pos, beta, score in (
                (1000, "0.2", "0.69"), (1001, "0.2", "0.7"),
                (2000, "0.2", "0.8"), (1002, "0.2", "NA"),
                (1003, "0.2", "oops"), (1004, "0.2", "inf"),
                (1005, "0.2", "1.1"), (1006, "NA", "0.9"),
            )
        ) + "\n"
    )
    return ref, groups, axis, source


def _manifest(
    path: Path, source: Path, policy: str, analysis_id: str = "GCST_INFO",
    method: str = "source_provided",
) -> Path:
    names = ["analysis_id", "source_file", "source_reader_capability", "stored_effect_scale",
             "original_sd_method", "sample_size"]
    values = [analysis_id, str(source), "opengwasdb.gwas-ssf", "sd", method, "1000"]
    if policy != "legacy":
        names.append("info_score_threshold")
        values.append(policy)
    if policy not in ("legacy", "NaN"):
        names += ["imputation_score_column", "imputation_score_kind",
                  "imputation_score_provenance"]
        values += ["quality_metric", "imputation_info", "Provider Table 2 INFO"]
    path.write_text("\t".join(names) + "\n" + "\t".join(values) + "\n")
    return path


def _resolve(tmp_path: Path, sources: tuple[Path, Path, Path, Path], policy: str,
             *, source: Path | None = None, max_rows: int | None = None,
             method: str = "source_provided", resume: bool = False) -> dict:
    ref, groups, axis, original = sources
    summary = resolve_analyses_manifest(
        _manifest(tmp_path / "manifest.tsv", source or original, policy, method=method),
        tmp_path / "records",
        ancestry_reference=ref, ancestry_groups=groups, variant_reference=axis,
        max_rows=max_rows, n_workers=1, resume=resume,
    )
    assert summary.n_total == 1
    return json.loads((summary.records_dir / "GCST_INFO.json").read_text())


def test_positive_info_filters_before_reference_and_evidence(tmp_path: Path, sources):
    result = _resolve(tmp_path, sources, "0.7")
    assert result["status"] == "success"
    d = result["diagnostics"]
    assert d["canonical_rows_observed"] == d["rows_read"] == 8
    assert d["canonical_rows_retained"] == 3  # equality, off-axis, invalid beta
    assert d["info_rows_below_threshold"] == 1
    assert [d[f"info_rows_{name}"] for name in (
        "missing", "malformed", "nonfinite", "out_of_range", "usable"
    )] == [1, 1, 1, 1, 4]
    assert d["info_score_state"] == "filtered"
    assert d["variant_reference_rows_matched"] == 7  # legacy pre-INFO count
    assert d["ancestry_reference_rows_matched"] == 2
    assert d["build_eligible_rows"] == 2
    assert d["build_eligible_rows_on_variant_reference"] == 1
    assert d["build_eligible_rows_off_variant_reference"] == 1


@pytest.mark.parametrize("policy,state", [
    ("0", "disabled"), ("NaN", "unavailable"), ("legacy", "legacy_absent")
])
def test_no_filter_states_keep_every_canonical_row(tmp_path: Path, sources, policy, state):
    result = _resolve(tmp_path, sources, policy)
    assert result["status"] == "success"
    d = result["diagnostics"]
    assert d["info_score_state"] == state
    assert d["canonical_rows_observed"] == d["canonical_rows_retained"] == 8
    assert d["info_rows_below_threshold"] == 0
    assert d["build_eligible_rows_on_variant_reference"] == 6
    assert d["build_eligible_rows_off_variant_reference"] == 1
    assert d["info_rows_usable"] == (0 if policy in ("NaN", "legacy") else 4)
    parsed = read_resolve_manifest(tmp_path / "manifest.tsv")[0]
    assert parsed.info_score_policy.state == InfoScoreState(state)


def test_declared_column_missing_or_no_usable_scores_is_failure(tmp_path: Path, sources):
    no_column = tmp_path / "no_column.tsv"
    no_column.write_text("\n".join(
        "\t".join(cell for index, cell in enumerate(line.split("\t")) if index != 7)
        for line in sources[3].read_text().splitlines()
    ) + "\n")
    missing = _resolve(tmp_path, sources, "0.7", source=no_column)
    assert missing["status"] == "controlled_failure"
    assert "GCST_INFO" in missing["error"]
    assert "quality_metric" in missing["error"]
    all_missing = tmp_path / "all_missing.tsv"
    lines = sources[3].read_text().splitlines()
    all_missing.write_text(lines[0] + "\n" + lines[1].replace("0.69", "NA") + "\n")
    no_scores = _resolve(tmp_path, sources, "0.7", source=all_missing)
    assert no_scores["status"] == "controlled_failure"
    assert "GCST_INFO" in no_scores["error"]
    assert "no usable scores" in no_scores["error"]


def test_info_policy_filters_sd_evidence_and_changes_resume_key(tmp_path: Path, sources):
    filtered = _resolve(tmp_path, sources, "0.7", method="estimated_from_source_maf")
    assert filtered["phenotype_sd"]["n_evidence_considered"] == 2
    assert filtered["fingerprints"]["resolution_config"]["info_score_state"] == "filtered"
    legacy = _resolve(tmp_path, sources, "legacy", resume=True)
    unavailable = _resolve(tmp_path, sources, "NaN", resume=True)
    assert legacy["diagnostics"]["info_score_state"] == "legacy_absent"
    assert unavailable["diagnostics"]["info_score_state"] == "unavailable"
    assert legacy["fingerprints"]["fingerprint_digest"] != (
        unavailable["fingerprints"]["fingerprint_digest"]
    )


def test_boundary_one_retains_only_exact_score(tmp_path: Path, sources):
    source = tmp_path / "boundary.tsv"
    source.write_text(sources[3].read_text().replace("0.69\t0.99", "1\t0.99"))
    d = _resolve(tmp_path, sources, "1", source=source)["diagnostics"]
    assert d["canonical_rows_observed"] == 8
    assert d["canonical_rows_retained"] == 1
    assert d["info_rows_below_threshold"] == 3
    assert d["build_eligible_rows_on_variant_reference"] == 1


def test_ancestry_site_limit_uses_retained_physical_position(tmp_path: Path, sources):
    ref, groups, axis, source = sources
    manifest = _manifest(tmp_path / "manifest.tsv", source, "0.7")
    summary = resolve_analyses_manifest(
        manifest, tmp_path / "records", ancestry_reference=ref, ancestry_groups=groups,
        variant_reference=axis, max_ancestry_sites=1, n_workers=1,
    )
    d = json.loads((summary.records_dir / "GCST_INFO.json").read_text())["diagnostics"]
    assert d["stop_reason"] == d["ancestry_stop_reason"] == "ancestry_site_limit"
    assert d["rows_read"] == d["ancestry_rows_read"] == 2
    assert d["canonical_rows_observed"] == 2
    assert d["canonical_rows_retained"] == 1
    assert d["info_rows_below_threshold"] == 1
    assert d["build_eligible_rows_on_variant_reference"] == 1


def test_bounded_chunk_preserves_scores_and_physical_limit(tmp_path: Path, sources):
    result = _resolve(tmp_path, sources, "0.7", max_rows=5)
    d = result["diagnostics"]
    assert d["stop_reason"] == "row_limit"
    assert d["rows_read"] == d["canonical_rows_observed"] == 5
    assert d["canonical_rows_retained"] == 2
    assert d["build_eligible_rows_on_variant_reference"] == 1
    assert d["build_eligible_rows_off_variant_reference"] == 1


def test_policy_state_cannot_contradict_threshold_and_declaration():
    assert InfoScorePolicy().state is InfoScoreState.LEGACY_ABSENT
    assert InfoScorePolicy(state=InfoScoreState.UNAVAILABLE).info_score_threshold is None
    assert InfoScorePolicy(0.0, DECLARATION, InfoScoreState.DISABLED).state is (
        InfoScoreState.DISABLED
    )
    assert InfoScorePolicy(0.7, DECLARATION, InfoScoreState.FILTERED).state is (
        InfoScoreState.FILTERED
    )
    for kwargs in (
        {"info_score_threshold": 0.7},
        {"imputation_score_declaration": DECLARATION},
        {"info_score_threshold": 0.7, "imputation_score_declaration": DECLARATION},
        {"info_score_threshold": 0.7, "imputation_score_declaration": DECLARATION,
         "state": InfoScoreState.DISABLED},
        {"info_score_threshold": 0.0, "imputation_score_declaration": DECLARATION,
         "state": InfoScoreState.FILTERED},
        {"state": InfoScoreState.FILTERED},
        {"info_score_threshold": -0.1, "imputation_score_declaration": DECLARATION,
         "state": InfoScoreState.FILTERED},
        {"info_score_threshold": float("nan"), "imputation_score_declaration": DECLARATION,
         "state": InfoScoreState.FILTERED},
    ):
        with pytest.raises(ValueError):
            InfoScorePolicy(**kwargs)
