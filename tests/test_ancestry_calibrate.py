"""Tests for ancestry threshold calibration + relabelling (issue 064)."""

from __future__ import annotations

import csv
from pathlib import Path

from typer.testing import CliRunner

from opengwasdb.ancestry import calibrate
from opengwasdb.ancestry.mixture import Gates
from opengwasdb.cli.main import app

# A synthetic Catalogue: the columns calibration reads from. Stored stats let us
# relabel under new gates without any allele-frequency re-extraction.
_HEADER = [
    "trait_id",
    "file_path",
    "trait_name",
    "n",
    "assigned_ancestry",
    "reported_population",
    "af_overlap",
    "nnls_residual",
    "dominant_superpop",
    "dominant_proportion",
    "runner_up_margin",
    "gate_reason",
    "catalogue_version",
    "ancestry_reference_version",
    "gate_tau",
    "gate_delta",
    "gate_n_min",
    "gate_residual_max",
]


def _row(trait_id, reported, assigned, dom_sp, dom_prop, margin, overlap=50000, residual=0.02):
    return {
        "trait_id": trait_id,
        "file_path": f"/data/{trait_id}.vcf.gz",
        "trait_name": trait_id,
        "n": "1000",
        "assigned_ancestry": assigned,
        "reported_population": reported,
        "af_overlap": str(overlap),
        "nnls_residual": f"{residual:.6g}",
        "dominant_superpop": dom_sp,
        "dominant_proportion": f"{dom_prop:.6g}",
        "runner_up_margin": f"{margin:.6g}",
        "gate_reason": "ok" if assigned != "Unassigned" else "proportion",
        "catalogue_version": "cat-v1",
        "ancestry_reference_version": "prive2022-hg38",
        "gate_tau": "0.9",
        "gate_delta": "0.2",
        "gate_n_min": "20000",
        "gate_residual_max": "0.06",
    }


def _rows():
    return [
        _row("eur1", "European", "EUR", "EUR", 0.97, 0.9),
        _row("eur2", "White European", "Unassigned", "EUR", 0.88, 0.7),  # below τ=0.9
        _row("afr1", "African", "AFR", "AFR", 0.95, 0.85),
        _row("mix1", "Mixed", "Unassigned", "EUR", 0.55, 0.1),
        _row("eas1", "East Asian", "EAS", "EAS", 0.99, 0.95),
    ]


def _write(tmp_path: Path, rows) -> Path:
    path = tmp_path / "catalogue.tsv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_normalise_reported():
    assert calibrate.normalise_reported("European") == "EUR"
    assert calibrate.normalise_reported("East Asian") == "EAS"
    assert calibrate.normalise_reported("Mixed") == "Mixed"
    assert calibrate.normalise_reported("African American") == "Mixed"  # admixed
    assert calibrate.normalise_reported("") == "Unknown"


def test_crosstab_and_operating_point():
    rows = _rows()
    counts = calibrate.crosstab(rows)
    assert counts[("EUR", "EUR")] == 1
    assert counts[("EUR", "Unassigned")] == 1  # eur2 was force-set Unassigned here
    assert counts[("Mixed", "Unassigned")] == 1
    op = calibrate.operating_point(rows)
    assert op["reported_eur"] == 2
    assert op["reported_mixed"] == 1
    assert op["reported_mixed_unassigned"] == 1


def test_disagreements_ignores_mixed_and_flags_conflicts():
    rows = _rows()
    conflicts = calibrate.disagreements(rows)
    ids = {d.trait_id for d in conflicts}
    assert ids == {"eur2"}  # reported EUR but Unassigned; mix1 (Mixed) not audited


def test_relabel_from_stored_stats_no_reextraction():
    rows = _rows()
    # Loosen τ to 0.85: eur2 (dom 0.88, margin 0.7) now clears the gate → EUR.
    relabelled = calibrate.relabel(rows, Gates(tau=0.85, delta=0.2, n_min=20000, residual_max=0.06))
    by_id = {r["trait_id"]: r for r in relabelled}
    assert by_id["eur2"]["assigned_ancestry"] == "EUR"
    assert by_id["eur2"]["gate_reason"] == "ok"
    assert by_id["eur2"]["gate_tau"] == "0.85"
    # mix1 (dom 0.55) still fails and stays Unassigned.
    assert by_id["mix1"]["assigned_ancestry"] == "Unassigned"
    # Tightening τ to 0.98 drops eur1 (0.97) back to Unassigned.
    tight = calibrate.relabel(rows, Gates(tau=0.98, delta=0.2, n_min=20000, residual_max=0.06))
    assert {r["trait_id"]: r["assigned_ancestry"] for r in tight}["eur1"] == "Unassigned"


def test_calibrate_cli_prints_crosstab_and_relabels(tmp_path):
    catalogue = _write(tmp_path, _rows())
    out = tmp_path / "relabelled.tsv"
    report = tmp_path / "disagreements.tsv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "calibrate-ancestry",
            str(catalogue),
            "--tau",
            "0.85",
            "--delta",
            "0.2",
            "--out",
            str(out),
            "--report",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "reported\\assigned" in result.output
    assert "disagreements" in result.output
    assert report.exists() and out.exists()

    with open(out, newline="", encoding="utf-8") as fh:
        relabelled = {r["trait_id"]: r for r in csv.DictReader(fh, delimiter="\t")}
    assert relabelled["eur2"]["assigned_ancestry"] == "EUR"  # admitted at τ=0.85
    assert relabelled["eur2"]["gate_tau"] == "0.85"


def test_relabel_preserves_catalogue_as_manifest_columns(tmp_path):
    """Relabelling preserves the Catalogue's trait_id/file_path/trait_name/n
    columns (the build manifest's own columns) in order, though a relabelled
    Catalogue is still not on its own a complete build manifest as of issue
    #17: stored_effect_scale is a genuinely separate build input ancestry
    assignment never needs."""
    catalogue = _write(tmp_path, _rows())
    runner = CliRunner()
    out = tmp_path / "relabelled.tsv"
    args = ["calibrate-ancestry", str(catalogue)]
    args += ["--tau", "0.85", "--delta", "0.2", "--out", str(out)]
    result = runner.invoke(app, args)
    # Asserted before anything downstream: without it, a CLI that failed
    # outright reads as an empty output file, not as a broken command.
    assert result.exit_code == 0, result.output
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert [r["trait_id"] for r in rows] == ["eur1", "eur2", "afr1", "mix1", "eas1"]
