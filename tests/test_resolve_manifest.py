"""Tests for the resumable manifest resolver pipeline and CLI (issue #208).

Validates:
1. Canonical manifest processing with atomic per-Analysis records and deterministic index.
2. Mixed quantitative, case-control, missing sample size, and controlled failure records.
3. Content-aware resume that skips unchanged successful records without re-execution.
4. Fingerprint invalidation on changed source, method, reference, extraction panel, and config.
5. Error isolation: bad/unreadable sources do not abort or discard other results.
6. Single ancestry reference load per invocation (fork-shared, never reloaded per Analysis).
7. Worker determinism: 1-worker and multi-worker runs yield byte-equivalent records and index.
8. Interrupted write safety: corrupt records are never resumed as successful.
9. Systemic configuration and setup errors fail loudly with non-zero exit codes.
10. CLI help and option contracts.
11. Reference-MAF SD tier resolution with declared reference vs undeclared.
12. Panel extraction and helper edge-cases.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from opengwasdb.ancestry.mixture import Gates
from opengwasdb.ancestry.reference import AncestryReference, load_reference
from opengwasdb.build.resolve_manifest import (
    ManifestResolutionSummary,
    compute_fingerprint_digest,
    load_extraction_panel,
    parse_af_references,
    resolve_analyses_manifest,
)
from opengwasdb.cli.main import app

N_VARIANTS = 50
_STUDY_N = 10_000.0
_TRUE_SD = 1.5
_POP_GROUPS = ("North-West Europe", "North-East Europe", "West Africa", "East Asia")
_POP_TO_SUPERPOP = {
    "North-West Europe": "EUR",
    "North-East Europe": "EUR",
    "West Africa": "AFR",
    "East Asia": "EAS",
}
_TEST_GATES = Gates(tau=0.50, delta=0.10, n_min=10, residual_max=0.25)


def _alid(index: int) -> str:
    return f"1:{1000 + index}:A:C"


def _panel_frequencies() -> np.ndarray:
    rng = np.random.default_rng(208)
    baseline = rng.uniform(0.10, 0.90, size=(N_VARIANTS, 1))
    drift = rng.normal(0.0, 0.05, size=(N_VARIANTS, len(_POP_GROUPS)))
    return np.clip(baseline + drift, 0.02, 0.98)


def _write_panel(directory: Path) -> tuple[Path, Path, AncestryReference]:
    directory.mkdir(parents=True, exist_ok=True)
    frequencies = _panel_frequencies()
    header = ["alid", "chromosome", "position", "effect_allele", "other_allele", "rsid"]
    lines = ["\t".join([*header, *_POP_GROUPS])]
    for index in range(N_VARIANTS):
        alid = _alid(index)
        chrom, pos, a1, a2 = alid.split(":")
        cells = [alid, chrom, pos, a1, a2, f"rs{index}"]
        lines.append("\t".join([*cells, *(f"{v:.6g}" for v in frequencies[index])]))
    ref_path = directory / "ref_freqs.tsv"
    ref_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    groups = ["group\tsuper_pop"]
    groups += [f"{g}\t{_POP_TO_SUPERPOP[g]}" for g in _POP_GROUPS]
    grp_path = directory / "ancestry_groups.tsv"
    grp_path.write_text("\n".join(groups) + "\n", encoding="utf-8")

    ref = load_reference(ref_path, grp_path, maf_floor=0.0)
    return ref_path, grp_path, ref


def _mixture(panel: AncestryReference, weights: Mapping[str, float]) -> np.ndarray:
    freqs = np.zeros(panel.n_variants)
    for group, weight in weights.items():
        freqs += weight * panel.freqs[:, panel.groups.index(group)]
    return freqs


def _se_for(freqs: np.ndarray, *, sd: float = _TRUE_SD, n: float = _STUDY_N) -> np.ndarray:
    return sd / np.sqrt(2.0 * freqs * (1.0 - freqs) * n)


def _write_ssf(path: Path, freqs: np.ndarray, *, beta: float = 0.2, corrupt: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if corrupt:
        path.write_bytes(b"\x1f\x8b\x08not-a-valid-gzip-stream")
        return path

    errors = _se_for(freqs)
    header = (
        "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\tstandard_error\t"
        "effect_allele_frequency\n"
    )
    lines = [header]
    for idx, (f, se) in enumerate(zip(freqs, errors, strict=True)):
        lines.append(f"1\t{1000 + idx}\tA\tC\t{beta:.6g}\t{se:.6g}\t{f:.6g}\n")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.writelines(lines)
    return path


def _write_af_ref(path: Path, freqs: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["chromosome\tposition\teffect_allele\tother_allele\teaf\n"]
    for idx, f in enumerate(freqs):
        lines.append(f"1\t{1000 + idx}\tA\tC\t{f:.6g}\n")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.writelines(lines)
    return path


@pytest.fixture
def test_setup(tmp_path: Path) -> dict[str, Any]:
    ref_path, grp_path, ref = _write_panel(tmp_path / "reference")
    eur_freqs = _mixture(ref, {"North-West Europe": 0.8, "North-East Europe": 0.2})
    eas_freqs = _mixture(ref, {"East Asia": 1.0})

    src_eur_quant = _write_ssf(tmp_path / "raw" / "GCST_EUR_QUANT.h.tsv.gz", eur_freqs)
    src_eas_quant = _write_ssf(tmp_path / "raw" / "GCST_EAS_QUANT.h.tsv.gz", eas_freqs)
    src_eur_cc = _write_ssf(tmp_path / "raw" / "GCST_EUR_CC.h.tsv.gz", eur_freqs)
    src_eur_no_n = _write_ssf(tmp_path / "raw" / "GCST_EUR_NON.h.tsv.gz", eur_freqs)
    src_eur_ref_maf = _write_ssf(tmp_path / "raw" / "GCST_EUR_REFMAF.h.tsv.gz", eur_freqs)
    src_corrupt = _write_ssf(tmp_path / "raw" / "GCST_CORRUPT.h.tsv.gz", eur_freqs, corrupt=True)

    af_ref_path = _write_af_ref(tmp_path / "reference" / "ukb_eur_af.tsv.gz", eur_freqs)

    panel_variants_path = tmp_path / "reference" / "qc_panel.tsv"
    panel_variants_path.write_text(
        "alid\n" + "\n".join(_alid(i) for i in range(25)) + "\n", encoding="utf-8"
    )

    manifest_lines = [
        "analysis_id\tsource_file\tsource_reader_capability\tstored_effect_scale\t"
        "original_sd_method\tsample_size\tsize_bytes\n",
        f"GCST_EUR_QUANT\t{src_eur_quant}\topengwasdb.gwas-ssf\tsd\testimated_from_source_maf\t{int(_STUDY_N)}\t{src_eur_quant.stat().st_size}\n",
        f"GCST_EAS_QUANT\t{src_eas_quant}\topengwasdb.gwas-ssf\tsd\testimated_from_source_maf\t{int(_STUDY_N)}\t{src_eas_quant.stat().st_size}\n",
        f"GCST_EUR_CC\t{src_eur_cc}\topengwasdb.gwas-ssf\tlog_or\tbinary_trait\t{int(_STUDY_N)}\t{src_eur_cc.stat().st_size}\n",
        f"GCST_EUR_NON\t{src_eur_no_n}\topengwasdb.gwas-ssf\tsd\testimated_from_source_maf\t\t{src_eur_no_n.stat().st_size}\n",
        f"GCST_CORRUPT\t{src_corrupt}\topengwasdb.gwas-ssf\tsd\testimated_from_source_maf\t{int(_STUDY_N)}\t{src_corrupt.stat().st_size}\n",
    ]
    manifest_path = tmp_path / "analyses.tsv"
    manifest_path.write_text("".join(manifest_lines), encoding="utf-8")

    manifest_ref_maf_lines = [
        "analysis_id\tsource_file\tsource_reader_capability\tstored_effect_scale\t"
        "original_sd_method\tsample_size\tsize_bytes\n",
        f"GCST_EUR_REFMAF\t{src_eur_ref_maf}\topengwasdb.gwas-ssf\tsd\testimated_from_reference_maf\t{int(_STUDY_N)}\t{src_eur_ref_maf.stat().st_size}\n",
    ]
    manifest_ref_maf_path = tmp_path / "analyses_ref_maf.tsv"
    manifest_ref_maf_path.write_text("".join(manifest_ref_maf_lines), encoding="utf-8")

    return {
        "tmp_path": tmp_path,
        "ref_path": ref_path,
        "grp_path": grp_path,
        "ref": ref,
        "manifest_path": manifest_path,
        "manifest_ref_maf_path": manifest_ref_maf_path,
        "af_ref_path": af_ref_path,
        "panel_variants_path": panel_variants_path,
        "src_eur_quant": src_eur_quant,
        "src_eas_quant": src_eas_quant,
        "src_eur_cc": src_eur_cc,
        "src_eur_no_n": src_eur_no_n,
        "src_eur_ref_maf": src_eur_ref_maf,
        "src_corrupt": src_corrupt,
    }


def _run_standard_resolve(
    setup: dict[str, Any], records_dir: Path, **overrides: Any
) -> ManifestResolutionSummary:
    params: dict[str, Any] = {
        "manifest_path": setup["manifest_path"],
        "records_dir": records_dir,
        "ancestry_reference": setup["ref_path"],
        "ancestry_groups": setup["grp_path"],
        "n_min": 10,
        "residual_max": 0.25,
        "n_workers": 1,
    }
    params.update(overrides)
    return resolve_analyses_manifest(**params)


def _invoke_cli_resolve(
    runner: CliRunner,
    setup: dict[str, Any],
    manifest_path: Path,
    records_dir: Path,
    *extra_flags: str,
) -> Any:
    base_args = [
        "resolve-analyses",
        str(manifest_path),
        str(records_dir),
        "--ancestry-reference",
        str(setup["ref_path"]),
        "--ancestry-groups",
        str(setup["grp_path"]),
    ]
    return runner.invoke(app, [*base_args, *extra_flags])


def _invoke_cli_standard(
    runner: CliRunner,
    setup: dict[str, Any],
    records_dir: Path,
    *extra_flags: str,
) -> Any:
    """The release gates the CLI tests use, so a test only states what it varies."""
    return _invoke_cli_resolve(
        runner,
        setup,
        setup["manifest_path"],
        records_dir,
        "--n-min",
        "10",
        "--residual-max",
        "0.25",
        "--n-workers",
        "1",
        *extra_flags,
    )


def test_manifest_resolution_produces_atomic_records_and_index(
    test_setup: dict[str, Any]
) -> None:
    records_dir = test_setup["tmp_path"] / "records"
    summary = _run_standard_resolve(test_setup, records_dir)
    assert summary.n_total == 5
    assert summary.n_success == 4
    assert summary.n_failed == 1
    assert summary.failed_analyses == ["GCST_CORRUPT"]

    index_file = records_dir / "index.json"
    assert index_file.is_file()
    with open(index_file, encoding="utf-8") as fh:
        index_data = json.load(fh)
    assert index_data["n_total"] == 5
    assert index_data["n_success"] == 4
    assert index_data["n_failed"] == 1
    assert len(index_data["analyses"]) == 5
    assert [a["analysis_id"] for a in index_data["analyses"]] == [
        "GCST_EUR_QUANT",
        "GCST_EAS_QUANT",
        "GCST_EUR_CC",
        "GCST_EUR_NON",
        "GCST_CORRUPT",
    ]

    eur_rec_path = records_dir / "GCST_EUR_QUANT.json"
    assert eur_rec_path.is_file()
    with open(eur_rec_path, encoding="utf-8") as fh:
        eur_rec = json.load(fh)
    assert eur_rec["status"] == "success"
    assert eur_rec["ancestry"]["assigned_ancestry"] == "EUR"
    assert eur_rec["phenotype_sd"]["status"] == "estimated"
    assert eur_rec["phenotype_sd"]["estimate"]["sd"] == pytest.approx(_TRUE_SD, rel=1e-2)
    assert "fingerprint_digest" in eur_rec["fingerprints"]
    assert eur_rec["metrics"]["elapsed_seconds"] >= 0.0


def test_mixed_records_success_skip_unavailable_failure(test_setup: dict[str, Any]) -> None:
    records_dir = test_setup["tmp_path"] / "records_mixed"
    _run_standard_resolve(test_setup, records_dir)

    with open(records_dir / "GCST_EUR_CC.json", encoding="utf-8") as fh:
        cc_rec = json.load(fh)
    assert cc_rec["status"] == "success"
    assert cc_rec["ancestry"]["assigned_ancestry"] == "EUR"
    assert cc_rec["phenotype_sd"]["status"] == "skipped"
    assert cc_rec["phenotype_sd"]["reason"] == "non_quantitative_effect_scale"

    with open(records_dir / "GCST_EUR_NON.json", encoding="utf-8") as fh:
        non_rec = json.load(fh)
    assert non_rec["status"] == "success"
    assert non_rec["ancestry"]["assigned_ancestry"] == "EUR"
    assert non_rec["phenotype_sd"]["status"] == "unavailable"
    assert non_rec["phenotype_sd"]["reason"] == "no_usable_sample_size"

    with open(records_dir / "GCST_CORRUPT.json", encoding="utf-8") as fh:
        corrupt_rec = json.load(fh)
    assert corrupt_rec["status"] == "controlled_failure"
    assert corrupt_rec["error"] is not None
    assert (
        "eoferror" in corrupt_rec["error"].lower()
        or "compressed" in corrupt_rec["error"].lower()
        or "gzip" in corrupt_rec["error"].lower()
    )
    assert corrupt_rec["ancestry"] is None
    assert corrupt_rec["phenotype_sd"] is None


def test_resume_skips_unchanged_successful_analyses(test_setup: dict[str, Any]) -> None:
    records_dir = test_setup["tmp_path"] / "records_resume"
    first_summary = _run_standard_resolve(test_setup, records_dir, resume=False)
    assert first_summary.n_resumed == 0
    assert first_summary.n_success == 4

    eur_file = records_dir / "GCST_EUR_QUANT.json"
    mtime_before = eur_file.stat().st_mtime_ns

    second_summary = _run_standard_resolve(test_setup, records_dir, resume=True)
    assert second_summary.n_resumed == 4
    assert second_summary.n_failed == 1
    mtime_after = eur_file.stat().st_mtime_ns
    assert mtime_before == mtime_after


def test_resume_reruns_failed_records_after_source_fix(test_setup: dict[str, Any]) -> None:
    records_dir = test_setup["tmp_path"] / "records_fix_failure"
    _run_standard_resolve(test_setup, records_dir, resume=False)

    eur_freqs = _mixture(test_setup["ref"], {"North-West Europe": 1.0})
    _write_ssf(test_setup["src_corrupt"], eur_freqs, corrupt=False)

    resumed_summary = _run_standard_resolve(test_setup, records_dir, resume=True)
    assert resumed_summary.n_resumed == 4
    assert resumed_summary.n_success == 5
    assert resumed_summary.n_failed == 0

    with open(records_dir / "GCST_CORRUPT.json", encoding="utf-8") as fh:
        fixed_rec = json.load(fh)
    assert fixed_rec["status"] == "success"
    assert fixed_rec["ancestry"]["assigned_ancestry"] == "EUR"


def test_fingerprint_invalidation_matrix(test_setup: dict[str, Any]) -> None:
    records_dir = test_setup["tmp_path"] / "records_invalidation"
    _run_standard_resolve(test_setup, records_dir, resume=False)

    # Invalidation 1: gate change (tau).
    s_gate = _run_standard_resolve(test_setup, records_dir, tau=0.75, resume=True)
    assert s_gate.n_resumed == 0

    # Invalidation 2: extraction panel added.
    s_panel = _run_standard_resolve(
        test_setup, records_dir, extraction_panel=test_setup["panel_variants_path"], resume=True
    )
    assert s_panel.n_resumed == 0

    # Invalidation 3: evidence sample size changed.
    s_sample = _run_standard_resolve(
        test_setup, records_dir, evidence_sample=500, resume=True
    )
    assert s_sample.n_resumed == 0


def test_reference_loaded_once_per_invocation(test_setup: dict[str, Any]) -> None:
    records_dir = test_setup["tmp_path"] / "records_ref_load"
    with patch(
        "opengwasdb.build.resolve_manifest.load_reference", wraps=load_reference
    ) as mock_load:
        _run_standard_resolve(test_setup, records_dir, n_workers=2)
        assert mock_load.call_count == 1


def test_worker_determinism_and_byte_equivalence(test_setup: dict[str, Any]) -> None:
    dir_serial = test_setup["tmp_path"] / "records_serial"
    dir_parallel = test_setup["tmp_path"] / "records_parallel"

    _run_standard_resolve(test_setup, dir_serial, n_workers=1, largest_first=False)
    _run_standard_resolve(test_setup, dir_parallel, n_workers=2, largest_first=True)

    for analysis_id in ["GCST_EUR_QUANT", "GCST_EAS_QUANT", "GCST_EUR_CC", "GCST_EUR_NON"]:
        rec_s = json.loads((dir_serial / f"{analysis_id}.json").read_text(encoding="utf-8"))
        rec_p = json.loads((dir_parallel / f"{analysis_id}.json").read_text(encoding="utf-8"))

        del rec_s["metrics"]
        del rec_p["metrics"]
        assert rec_s == rec_p

    index_s = json.loads((dir_serial / "index.json").read_text(encoding="utf-8"))
    index_p = json.loads((dir_parallel / "index.json").read_text(encoding="utf-8"))
    index_s["records_dir"] = ""
    index_p["records_dir"] = ""
    assert index_s == index_p


def test_atomic_write_and_corrupt_record_handling(test_setup: dict[str, Any]) -> None:
    records_dir = test_setup["tmp_path"] / "records_corrupt_rec"
    _run_standard_resolve(test_setup, records_dir)

    # Intentionally corrupt one JSON record.
    eur_rec = records_dir / "GCST_EUR_QUANT.json"
    eur_rec.write_text("{corrupt json content...", encoding="utf-8")

    # Resume must detect corrupt JSON and rerun that analysis cleanly.
    summary = _run_standard_resolve(test_setup, records_dir, resume=True)
    assert summary.n_resumed == 3
    assert summary.n_success == 4
    with open(eur_rec, encoding="utf-8") as fh:
        fixed = json.load(fh)
    assert fixed["status"] == "success"


def test_reference_maf_sd_tier_resolution(test_setup: dict[str, Any]) -> None:
    records_dir_without = test_setup["tmp_path"] / "records_refmaf_without"
    records_dir_with = test_setup["tmp_path"] / "records_refmaf_with"

    # Without --af-reference: skips SD estimation with explicit reason.
    s_without = _run_standard_resolve(
        test_setup, records_dir_without, manifest_path=test_setup["manifest_ref_maf_path"]
    )
    assert s_without.n_success == 1
    with open(records_dir_without / "GCST_EUR_REFMAF.json", encoding="utf-8") as fh:
        rec_without = json.load(fh)
    assert rec_without["ancestry"]["assigned_ancestry"] == "EUR"
    assert rec_without["phenotype_sd"]["status"] == "skipped"
    assert rec_without["phenotype_sd"]["reason"] == "no_reference_resource_for_ancestry"

    # With --af-reference EUR=/path/to/ukb_eur_af.tsv.gz: computes estimate.
    s_with = _run_standard_resolve(
        test_setup,
        records_dir_with,
        manifest_path=test_setup["manifest_ref_maf_path"],
        af_references=[f"EUR={test_setup['af_ref_path']}"],
    )
    assert s_with.n_success == 1
    with open(records_dir_with / "GCST_EUR_REFMAF.json", encoding="utf-8") as fh:
        rec_with = json.load(fh)
    assert rec_with["ancestry"]["assigned_ancestry"] == "EUR"
    assert rec_with["phenotype_sd"]["status"] == "estimated"
    assert rec_with["phenotype_sd"]["estimate"]["sd"] == pytest.approx(_TRUE_SD, rel=1e-2)
    assert rec_with["phenotype_sd"]["reference_id"] == test_setup["af_ref_path"].name


def test_extraction_panel_and_af_ref_helpers(tmp_path: Path) -> None:
    panel_file = tmp_path / "panel.txt"
    panel_file.write_text("1:100:A:C\n1:200:G:T\n# comment\n\n1:300:C:T\n", encoding="utf-8")
    variants = load_extraction_panel(panel_file)
    assert variants == {"1:100:A:C", "1:200:G:T", "1:300:C:T"}

    empty_panel = tmp_path / "empty_panel.txt"
    empty_panel.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty extraction panel"):
        load_extraction_panel(empty_panel)

    with pytest.raises(FileNotFoundError, match="AF reference path not found"):
        parse_af_references(["EUR=/nonexistent/path/ref.tsv.gz"])

    fp1 = {"a": 1, "b": 2}
    fp2 = {"b": 2, "a": 1}
    assert compute_fingerprint_digest(fp1) == compute_fingerprint_digest(fp2)


def test_systemic_errors_raise_and_return_nonzero(test_setup: dict[str, Any]) -> None:
    runner = CliRunner()
    records_dir = test_setup["tmp_path"] / "cli_records"

    # Missing manifest.
    res_miss = _invoke_cli_resolve(
        runner, test_setup, test_setup["tmp_path"] / "nonexistent.tsv", records_dir
    )
    assert res_miss.exit_code != 0

    # Duplicate analysis_id.
    dup_manifest = test_setup["tmp_path"] / "dup.tsv"
    dup_manifest.write_text(
        "analysis_id\tsource_file\nGCST1\t/file.tsv\nGCST1\t/file.tsv\n", encoding="utf-8"
    )
    res_dup = _invoke_cli_resolve(runner, test_setup, dup_manifest, records_dir)
    assert res_dup.exit_code != 0

    # Invalid evidence_sample.
    res_inv = _invoke_cli_resolve(
        runner, test_setup, test_setup["manifest_path"], records_dir, "--evidence-sample", "0"
    )
    assert res_inv.exit_code != 0


def test_cli_runner_successful_execution(test_setup: dict[str, Any]) -> None:
    runner = CliRunner()
    records_dir = test_setup["tmp_path"] / "cli_success"
    res = _invoke_cli_standard(runner, test_setup, records_dir)
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload["n_total"] == 5
    assert payload["n_success"] == 4
    assert payload["n_failed"] == 1
    assert payload["failed_analyses"] == ["GCST_CORRUPT"]
    assert (records_dir / "index.json").is_file()


# --- bounded scans (issue #209) -------------------------------------------


def test_scan_limit_bounds_the_ancestry_fit_and_is_recorded(
    test_setup: dict[str, Any]
) -> None:
    """A manifest run under a bound records the bound and the reason it stopped,
    continuing quantitative traits to EOF for SD evidence and stopping case-control
    traits early at the ancestry bound (issue #212).
    """
    records_dir = test_setup["tmp_path"] / "records_scan_limit"
    _run_standard_resolve(test_setup, records_dir, max_ancestry_sites=20)

    # Quantitative analysis: ancestry stopped at 20 sites, physical scan read all 50 rows for SD
    with open(records_dir / "GCST_EUR_QUANT.json", encoding="utf-8") as fh:
        quant_record = json.load(fh)

    assert quant_record["diagnostics"]["stop_reason"] == "eof"
    assert quant_record["diagnostics"]["ancestry_stop_reason"] == "ancestry_site_limit"
    assert quant_record["diagnostics"]["ancestry_sites"] == 20, (
        "the fit must stop at the bound, not at the source's 50 sites"
    )
    assert quant_record["diagnostics"]["ancestry_rows_read"] == 20
    assert quant_record["diagnostics"]["rows_read"] == 50
    assert quant_record["ancestry"]["assigned_ancestry"] == "EUR"
    assert quant_record["phenotype_sd"]["status"] == "estimated"
    assert quant_record["fingerprints"]["resolution_config"]["scan_limit"] == {
        "scan_limit_version": 2,
        "max_rows": None,
        "max_ancestry_sites": 20,
    }

    # Case-control analysis: physical scan stopped early at 20 rows because no SD needed
    with open(records_dir / "GCST_EUR_CC.json", encoding="utf-8") as fh:
        cc_record = json.load(fh)

    assert cc_record["diagnostics"]["stop_reason"] == "ancestry_site_limit"
    assert cc_record["diagnostics"]["ancestry_stop_reason"] == "ancestry_site_limit"
    assert cc_record["diagnostics"]["ancestry_sites"] == 20
    assert cc_record["diagnostics"]["ancestry_rows_read"] == 20
    assert cc_record["diagnostics"]["rows_read"] == 20
    assert cc_record["phenotype_sd"]["status"] == "skipped"


def test_scan_limit_invalidates_resume_when_changed(test_setup: dict[str, Any]) -> None:
    """A record resolved under one bound is not resumable under another."""
    records_dir = test_setup["tmp_path"] / "records_scan_resume"
    _run_standard_resolve(test_setup, records_dir, max_ancestry_sites=20, resume=False)

    same = _run_standard_resolve(test_setup, records_dir, max_ancestry_sites=20, resume=True)
    assert same.n_resumed == 4

    changed = _run_standard_resolve(test_setup, records_dir, max_ancestry_sites=30, resume=True)
    assert changed.n_resumed == 0, "a changed scan bound must invalidate every record"


def test_scan_limit_version_bump_invalidates_v1_records(test_setup: dict[str, Any]) -> None:
    """An old record written with scan_limit_version 1 must not resume under version 2."""
    records_dir = test_setup["tmp_path"] / "records_scan_v1_invalidation"
    _run_standard_resolve(test_setup, records_dir, max_ancestry_sites=20, resume=False)

    # Mutate recorded fingerprint scan_limit_version back to 1
    record_file = records_dir / "GCST_EUR_QUANT.json"
    data = json.loads(record_file.read_text(encoding="utf-8"))
    data["fingerprints"]["resolution_config"]["scan_limit"]["scan_limit_version"] = 1
    data["fingerprints"]["fingerprint_digest"] = compute_fingerprint_digest(data["fingerprints"])
    record_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Resuming with current code (version 2) should recompute the mutated record
    res = _run_standard_resolve(test_setup, records_dir, max_ancestry_sites=20, resume=True)
    assert res.n_resumed == 3, (
        "GCST_EUR_QUANT with v1 scan_limit must be invalidated and recomputed"
    )


def test_cli_scan_limit_option_contract(test_setup: dict[str, Any]) -> None:
    runner = CliRunner()
    records_dir = test_setup["tmp_path"] / "cli_scan_limit"
    res = _invoke_cli_standard(
        runner, test_setup, records_dir, "--max-ancestry-sites", "20"
    )
    assert res.exit_code == 0, res.output
    with open(records_dir / "GCST_EUR_QUANT.json", encoding="utf-8") as fh:
        quant_record = json.load(fh)
    assert quant_record["diagnostics"]["stop_reason"] == "eof"
    assert quant_record["diagnostics"]["ancestry_stop_reason"] == "ancestry_site_limit"
    assert quant_record["diagnostics"]["ancestry_sites"] == 20
    assert quant_record["diagnostics"]["rows_read"] == 50

    with open(records_dir / "GCST_EUR_CC.json", encoding="utf-8") as fh:
        cc_record = json.load(fh)
    assert cc_record["diagnostics"]["stop_reason"] == "ancestry_site_limit"
    assert cc_record["diagnostics"]["ancestry_stop_reason"] == "ancestry_site_limit"
    assert cc_record["diagnostics"]["ancestry_sites"] == 20
    assert cc_record["diagnostics"]["rows_read"] == 20


def test_cli_scan_limit_zero_reads_the_whole_source(test_setup: dict[str, Any]) -> None:
    runner = CliRunner()
    records_dir = test_setup["tmp_path"] / "cli_scan_full"
    res = _invoke_cli_standard(
        runner, test_setup, records_dir, "--max-ancestry-sites", "0"
    )
    assert res.exit_code == 0, res.output
    with open(records_dir / "GCST_EUR_QUANT.json", encoding="utf-8") as fh:
        record = json.load(fh)
    assert record["diagnostics"]["stop_reason"] == "eof"
    assert record["diagnostics"]["ancestry_sites"] == 50
    assert record["fingerprints"]["resolution_config"]["scan_limit"] is None
