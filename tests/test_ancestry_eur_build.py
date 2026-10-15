"""Build a EUR Hybrid store from an Analysis Catalogue subset (issue 065).

The Catalogue is a superset of the build manifest, so selecting EUR is a pure row
filter. The unchanged build-hybrid consumes it; per-Analysis Assigned Ancestry
rides straight into the store's ``analyses.tsv`` as part of the build (issue #22),
and release-level Catalogue provenance is folded into ``manifest.json``.
Non-EUR/Unassigned Analyses are absent from the store (still parked in the
Catalogue).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import opengwasdb.ancestry.subset as subset_module
from opengwasdb.ancestry.subset import build_hybrid_from_catalogue, subset_catalogue
from opengwasdb.cli.main import app
from opengwasdb.model.analyses import read_analyses
from opengwasdb.validation import validate_store


def _store_analysis_ids(store: Path) -> set[str]:
    """Read the store's analysis ids from its analyses.tsv."""
    return {row["analysis_id"] for row in read_analyses(store / "analyses.tsv").rows}


def _ancestry_map(store: Path) -> dict[str, str]:
    """Return {analysis_id: assigned_ancestry} from the store's analyses.tsv."""
    return {row["analysis_id"]: row["assigned_ancestry"] for row in read_analyses(
        store / "analyses.tsv"
    ).rows}


def _ancestry_provenance(store: Path) -> dict:
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    return manifest["provenance"]["ancestry"]

# hg19 coordinates that liftover to the hybrid fixture's hg38 ALIDs.
HG19_POS_1 = 100_000
HG19_POS_2 = 1_000_000
HG19_POS_3 = 1_500_000
HG38_ALID_1 = "1:100000:A:G"  # on-panel → Dense Component
HG38_ALID_3 = "1:1564620:A:G"  # on-panel → Dense Component


def _vcf(tmp_path: Path, name: str) -> Path:
    header = (
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size">\n'
        '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">\n'
        '##FORMAT=<ID=EZ,Number=A,Type=Float,Description="Z-score">\n'
        "##SAMPLE=<ID=STUDY1,StudyType=Continuous>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSTUDY1\n"
    )
    rows = (
        f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"
        f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n"
        f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n"
    )
    path = tmp_path / f"{name}.vcf"
    path.write_text(header + rows, encoding="utf-8")
    return path


def _panel(tmp_path: Path) -> Path:
    panel = tmp_path / "panel.txt"
    panel.write_text(f"{HG38_ALID_1}\n{HG38_ALID_3}\n", encoding="utf-8")
    return panel


def _catalogue(tmp_path: Path) -> Path:
    """A Catalogue with two EUR, one AFR, and one Unassigned Analysis."""
    vcfs = {name: _vcf(tmp_path, name) for name in ("eur_a", "eur_b", "afr_c", "unassigned_d")}
    header = [
        "trait_id",
        "file_path",
        "trait_name",
        "n",
        "assigned_ancestry",
        "reported_population",
        "ancestry_prop_EUR",
        "ancestry_prop_AFR",
        "catalogue_version",
        "ancestry_reference_version",
    ]
    ver = ["cat-v1", "prive2022-hg38"]
    rows = [
        ["eur_a", str(vcfs["eur_a"]), "EUR A", "1000", "EUR", "European", "0.95", "0.02", *ver],
        ["eur_b", str(vcfs["eur_b"]), "EUR B", "1200", "EUR", "European", "0.91", "0.04", *ver],
        ["afr_c", str(vcfs["afr_c"]), "AFR C", "900", "AFR", "African", "0.03", "0.92", *ver],
        [
            "und_d", str(vcfs["unassigned_d"]), "Und", "800", "Unassigned", "Mixed",
            "0.4", "0.4", *ver,
        ],
    ]
    path = tmp_path / "catalogue.tsv"
    path.write_text(
        "\t".join(header) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def test_subset_catalogue_is_pure_row_filter(tmp_path):
    catalogue = _catalogue(tmp_path)
    manifest = tmp_path / "eur_manifest.tsv"
    result = subset_catalogue(
        catalogue,
        manifest,
        ancestry="EUR",
        stored_effect_scale="sd",
        original_sd_method="declared_standardised",
    )
    assert result.n_total == 4 and result.n_kept == 2
    assert result.subset_filter == "assigned_ancestry == EUR"
    assert result.catalogue_version == "cat-v1"
    # The manifest is still a Catalogue superset: same header, EUR rows only.
    header = manifest.read_text().splitlines()[0].split("\t")
    assert header[0] == "trait_id" and "assigned_ancestry" in header
    assert "ancestry_assignment_method" in header
    from opengwasdb.layouts.dense.build_vcf import _read_manifest

    loaded = _read_manifest(manifest)
    assert [m.trait_id for m in loaded] == ["eur_a", "eur_b"]
    assert {m.metadata.ancestry_assignment_method for m in loaded} == {"af_assigned"}


def test_legacy_unassigned_catalogue_subset_records_failed_assignment(tmp_path):
    """A pre-#86 Catalogue has no method column, but still records that its
    Unassigned rows came from an attempted AF assignment whose gates failed."""
    manifest = tmp_path / "unassigned_manifest.tsv"
    subset_catalogue(
        _catalogue(tmp_path),
        manifest,
        ancestry="Unassigned",
        stored_effect_scale="sd",
        original_sd_method="declared_standardised",
    )

    from opengwasdb.layouts.dense.build_vcf import _read_manifest

    loaded = _read_manifest(manifest)
    assert [row.trait_id for row in loaded] == ["und_d"]
    assert loaded[0].metadata.ancestry_assignment_method == "unassigned"


def test_build_eur_hybrid_records_provenance_and_validates(tmp_path):
    catalogue = _catalogue(tmp_path)
    store = tmp_path / "eur.opengwasdb"
    subset = build_hybrid_from_catalogue(
        catalogue,
        store,
        reference_panel=_panel(tmp_path),
        store_id="eur-store",
        release_id="v1",
        stored_effect_scale="sd",
        original_sd_method="declared_standardised",
        ancestry="EUR",
    )
    assert subset.n_kept == 2

    # The store validates.
    result = validate_store(store)
    assert result.ok, result.errors

    # Only EUR-assigned Analyses are present in the store.
    assert _store_analysis_ids(store) == {"eur_a", "eur_b"}

    # Per-Analysis Assigned Ancestry, in analyses.tsv.
    ancestry_map = _ancestry_map(store)
    assert ancestry_map == {"eur_a": "EUR", "eur_b": "EUR"}

    # The Catalogue's ancestry_prop_* composition columns ride through the
    # manifest verbatim and land in analyses.tsv too (issue #22) -- not just
    # assigned_ancestry itself.
    rows = {r["analysis_id"]: r for r in read_analyses(store / "analyses.tsv").rows}
    assert {row["ancestry_assignment_method"] for row in rows.values()} == {"af_assigned"}
    assert rows["eur_a"]["ancestry_prop_EUR"] == "0.95"
    assert rows["eur_a"]["ancestry_prop_AFR"] == "0.02"
    assert rows["eur_b"]["ancestry_prop_EUR"] == "0.91"

    # Store-level provenance links back to the Catalogue.
    prov = _ancestry_provenance(store)
    assert prov["catalogue_version"] == "cat-v1"
    assert prov["subset_filter"] == "assigned_ancestry == EUR"
    assert prov["ancestry_reference_version"] == "prive2022-hg38"
    assert prov["n_analyses"] == 2


def test_build_hybrid_from_catalogue_forwards_a_column_map(
    tmp_path: Path, monkeypatch
) -> None:
    received: dict[str, object] = {}

    def capture_build(*args: object, **kwargs: object) -> None:
        received.update(kwargs)

    monkeypatch.setattr(
        "opengwasdb.layouts.hybrid.build.build_hybrid_from_vcf_manifest", capture_build
    )
    monkeypatch.setattr(subset_module, "record_catalogue_provenance", lambda *_args: None)

    build_hybrid_from_catalogue(
        _catalogue(tmp_path),
        tmp_path / "mapped.opengwasdb",
        reference_panel=_panel(tmp_path),
        store_id="mapped",
        release_id="v1",
        stored_effect_scale="sd",
        original_sd_method="declared_standardised",
        column_map="gwas-ssf",
    )

    assert received["column_map"] == "gwas-ssf"
    result = CliRunner().invoke(app, ["build-hybrid-from-catalogue", "--help"])
    assert result.exit_code == 0, result.output
    assert "--column-map" in result.output


def test_build_hybrid_from_catalogue_cli(tmp_path):
    catalogue = _catalogue(tmp_path)
    store = tmp_path / "eur_cli.opengwasdb"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build-hybrid-from-catalogue",
            str(catalogue),
            str(store),
            "--reference-panel",
            str(_panel(tmp_path)),
            "--store-id",
            "eur-store",
            "--release-id",
            "v1",
            "--stored-effect-scale",
            "sd",
            "--original-sd-method",
            "declared_standardised",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output.strip().splitlines()[-1])
    assert summary["ancestry"] == "EUR"
    assert summary["n_kept"] == 2 and summary["n_total"] == 4
    assert _ancestry_map(store) == {"eur_a": "EUR", "eur_b": "EUR"}
