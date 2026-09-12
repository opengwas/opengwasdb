"""Tests for the assign-ancestry pipeline + CLI (issue 063).

Builds a tiny reference panel and a handful of synthetic (bgzipped, indexed)
GWAS-VCFs, then drives the whole Catalogue-annotation path — asserting parking of
non-EUR/Unassigned Analyses, version stamps, and that the output is independent of
worker count and reproducible.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from opengwasdb.ancestry import (
    Gates,
    annotate_catalogue,
    load_reference,
    read_source_manifest,
)
from opengwasdb.cli.main import app

GROUPS = ["United Kingdom", "Finland", "Africa (West)", "Asia (East)"]
GROUP_TO_SUPERPOP = {
    "United Kingdom": "EUR",
    "Finland": "EUR",
    "Africa (West)": "AFR",
    "Asia (East)": "EAS",
}
N_VARIANTS = 40


def _write_reference(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    rng = np.random.default_rng(20260711)
    freqs = rng.uniform(0.05, 0.95, size=(N_VARIANTS, len(GROUPS)))
    alids = [f"1:{1000 + i}:A:C" for i in range(N_VARIANTS)]

    freqs_path = tmp_path / "ref_freqs.tsv"
    header = ["alid", "chromosome", "position", "effect_allele", "other_allele", "rsid", *GROUPS]
    lines = ["\t".join(header)]
    for i, alid in enumerate(alids):
        chrom, pos, a1, a2 = alid.split(":")
        lines.append(
            "\t".join([alid, chrom, pos, a1, a2, f"rs{i}", *[f"{f:.6g}" for f in freqs[i]]])
        )
    freqs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    groups_path = tmp_path / "ancestry_groups.tsv"
    groups_path.write_text(
        "group\tsuper_pop\n" + "".join(f"{g}\t{GROUP_TO_SUPERPOP[g]}\n" for g in GROUPS),
        encoding="utf-8",
    )
    return freqs_path, groups_path, freqs


def _study_vcf(tmp_path: Path, name: str, freqs: np.ndarray, weights: dict[str, int]) -> Path:
    """A bgzipped+indexed GWAS-VCF whose AF is a group mixture of the reference."""
    b = np.zeros(N_VARIANTS)
    for group, w in weights.items():
        b += w * freqs[:, GROUPS.index(group)]
    b /= sum(weights.values())

    # AF and SE FORMAT fields: extract_at_sites (issue #21) is a combined AF+SE
    # lookup and drops a site missing either, so every row must carry both.
    header = (
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n'
        '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">\n'
        "##SAMPLE=<ID=S1,StudyType=Continuous>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
    )
    rows = "".join(
        f"1\t{1000 + i}\t.\tC\tA\t.\tPASS\t.\tAF:SE\t{b[i]:.6g}:0.1\n" for i in range(N_VARIANTS)
    )
    plain = tmp_path / f"{name}.vcf"
    plain.write_text(header + rows, encoding="utf-8")
    out = tmp_path / f"{name}.vcf.gz"
    subprocess.run(
        ["bcftools", "view", str(plain), "-Oz", "-o", str(out), "--write-index=tbi"],
        check=True,
        capture_output=True,
    )
    return out


def _gates() -> Gates:
    return Gates(tau=0.90, delta=0.20, n_min=10, residual_max=0.05)


@pytest.fixture
def scenario(tmp_path):
    freqs_path, groups_path, freqs = _write_reference(tmp_path)
    reference = load_reference(freqs_path, groups_path, maf_floor=0.0)
    studies = {
        "eur_clean": _study_vcf(tmp_path, "eur", freqs, {"United Kingdom": 1}),
        "afr_clean": _study_vcf(tmp_path, "afr", freqs, {"Africa (West)": 1}),
        "admixed": _study_vcf(tmp_path, "mix", freqs, {"United Kingdom": 1, "Asia (East)": 1}),
    }
    manifest = tmp_path / "source_manifest.tsv"
    lines = ["trait_id\tfile_path\ttrait_name\tn\treported_population"]
    lines.append(f"eur1\t{studies['eur_clean']}\tEUR study\t1000\tEuropean")
    lines.append(f"afr1\t{studies['afr_clean']}\tAFR study\t2000\tAfrican")
    lines.append(f"mix1\t{studies['admixed']}\tMixed study\t3000\tMixed")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path, reference, manifest, freqs_path, groups_path


def test_annotate_catalogue_parks_non_eur_and_unassigned(scenario):
    tmp_path, reference, manifest, _fp, _gp = scenario
    rows = annotate_catalogue(
        read_source_manifest(manifest),
        reference,
        _gates(),
        tmp_path / "catalogue.tsv",
        catalogue_version="cat-v1",
        ancestry_reference_version="prive2022-hg38",
        n_workers=1,
    )
    by_id = {r.trait_id: r.assignment for r in rows}
    assert len(rows) == 3  # every Analysis retained
    assert by_id["eur1"].assigned_ancestry == "EUR"
    assert by_id["afr1"].assigned_ancestry == "AFR"  # non-EUR but present (parked)
    assert by_id["mix1"].assigned_ancestry is None  # Unassigned, still present


def test_catalogue_independent_of_worker_count(scenario):
    tmp_path, reference, manifest, _fp, _gp = scenario
    src = read_source_manifest(manifest)
    serial = tmp_path / "serial.tsv"
    parallel = tmp_path / "parallel.tsv"
    for out, workers in [(serial, 1), (parallel, 3)]:
        annotate_catalogue(
            src,
            reference,
            _gates(),
            out,
            catalogue_version="cat-v1",
            ancestry_reference_version="prive2022-hg38",
            n_workers=workers,
        )
    assert serial.read_text() == parallel.read_text()


def _assign_ancestry_args(
    manifest: Path, catalogue: Path, freqs_path: Path, groups_path: Path, *extra: str
) -> list[str]:
    """The scenario's CLI arguments, plus any per-test *extra* flags."""
    return [
        "assign-ancestry",
        str(manifest),
        str(catalogue),
        "--ancestry-reference",
        str(freqs_path),
        "--ancestry-groups",
        str(groups_path),
        "--maf-floor",
        "0.0",
        "--tau",
        "0.90",
        "--delta",
        "0.20",
        "--n-min",
        "10",
        "--residual-max",
        "0.05",
        *extra,
    ]


def test_assign_ancestry_cli(scenario):
    tmp_path, _reference, manifest, freqs_path, groups_path = scenario
    catalogue = tmp_path / "catalogue.tsv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        _assign_ancestry_args(
            manifest, catalogue, freqs_path, groups_path, "--catalogue-version", "cat-v1"
        ),
    )
    assert result.exit_code == 0, result.output
    import json

    summary = json.loads(result.output.strip().splitlines()[-1])
    assert summary["n_analyses"] == 3
    assert summary["n_assigned"] == 2  # EUR + AFR assigned; admixed parked
    assert summary["n_parked"] == 1
    assert summary["superpops"] == ["AFR", "EAS", "EUR"]

    # The Catalogue carries trait_id/file_path/trait_name/n (the build
    # manifest's own columns) plus its ancestry annotations, in order. It is
    # not, on its own, a complete build manifest as of issue #17:
    # stored_effect_scale is a genuinely separate build input ancestry
    # assignment never needs (opengwasdb.ancestry.subset stamps it on when
    # bridging a Catalogue subset into an actual build).
    header = catalogue.read_text().splitlines()[0].split("\t")
    assert header[:4] == ["trait_id", "file_path", "trait_name", "n"]
    assert "catalogue_version" in header and "ancestry_reference_version" in header

    rows = catalogue.read_text().splitlines()[1:]
    assert [r.split("\t")[0] for r in rows] == ["eur1", "afr1", "mix1"]


def test_read_source_manifest_accepts_canonical_analyses_tsv_columns(tmp_path):
    """Issue #170: the ancestry manifest reader takes the canonical
    ``analyses.tsv`` names (with the legacy spellings still readable)."""
    canonical = tmp_path / "canonical_manifest.tsv"
    canonical.write_text(
        "analysis_id\tsource_file\tanalysis_label\tsample_size\treported_population\n"
        "eur1\t/build/eur.vcf.gz\tEUR study\t1000\tEuropean\n",
        encoding="utf-8",
    )

    rows = read_source_manifest(canonical)

    assert len(rows) == 1
    assert rows[0].trait_id == "eur1"
    assert rows[0].file_path == "/build/eur.vcf.gz"
    assert rows[0].trait_name == "EUR study"
    assert rows[0].n == 1000
    assert rows[0].reported_population == "European"


@pytest.mark.parametrize("worker_flag", ["--n-workers", "--workers"])
def test_assign_ancestry_cli_accepts_worker_flag_spellings(scenario, worker_flag):
    """``--n-workers`` is the primary spelling; ``--workers`` stays accepted
    as an alias so existing callers keep working (issue #170 closeout)."""
    tmp_path, _reference, manifest, freqs_path, groups_path = scenario
    catalogue = tmp_path / f"catalogue-{worker_flag.lstrip('-')}.tsv"
    result = CliRunner().invoke(
        app,
        _assign_ancestry_args(manifest, catalogue, freqs_path, groups_path, worker_flag, "2"),
    )
    assert result.exit_code == 0, result.output
    assert catalogue.exists()
