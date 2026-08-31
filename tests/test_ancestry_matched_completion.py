"""Ancestry-matched completion (issue 066).

On a mixed store, only Analyses whose Assigned Ancestry matches the applied panel
are imputed; the rest stay observed-only. On a homogeneous store every Analysis
matches, so completion is unchanged (no regression).

Each trait observes six panel SNPs; the LD block adds a panel-only SNP
(1:1264620:C:T, in no VCF) as a genuine imputation target — enough observed points
to survive the imputation quality fit's outlier removal. Only the
panel-ancestry-matching trait may fill that target.

hg19 → hg38 anchors (all A/G): 100000→100000, 700000→764620, 900000→964620,
1100000→1164620, 1300000→1364620, 1500000→1564620.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np

from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.hybrid.complete import complete_hybrid_store
from opengwasdb.model.analyses import read_analyses
from opengwasdb.query import query_store
from opengwasdb.validation import validate_store


def _analyses_by_id(store: Path) -> dict[str, dict[str, str]]:
    return {row["analysis_id"]: row for row in read_analyses(store / "analyses.tsv").rows}

PANEL_ALIDS = [
    "1:100000:A:G",
    "1:764620:A:G",
    "1:964620:A:G",
    "1:1164620:A:G",
    "1:1364620:A:G",
    "1:1564620:A:G",
]


def _vcf(tmp_path: Path, name: str) -> Path:
    header = (
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size">\n'
        '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">\n'
        '##FORMAT=<ID=EZ,Number=A,Type=Float,Description="Z-score">\n'
        "##SAMPLE=<ID=S,StudyType=Continuous>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n"
    )
    # Six observed panel SNPs (enough points to survive outlier removal in the
    # imputation quality fit). The target 1:1264620 is panel-only (in no VCF).
    rows = [
        "1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
        "1\t700000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.4\n",
        "1\t900000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.5:0.3\n",
        "1\t1100000\t.\tA\tG\t.\tPASS\t.\tES:SE\t0.6:0.2\n",
        "1\t1300000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.2:0.3\n",
        "1\t1500000\t.\tA\tG\t.\tPASS\t.\tES:SE\t0.9:0.25\n",
    ]
    p = tmp_path / f"{name}.vcf"
    p.write_text(header + "".join(rows), encoding="utf-8")
    return p


def _write_ld_block(block_dir: Path, block_name: str, snps, rho: float = 0.7) -> None:
    """Write a flat LD block with a strong equicorrelation matrix (rho off-diagonal).

    Equicorrelation gives a dominant top eigenvector, so imputation of the held-out
    panel SNP reliably fills — unlike a random matrix, whose weak structure the
    elastic-net kernel collapses to a rejected (NaN) fit.
    """
    block_dir.mkdir(parents=True, exist_ok=True)
    n = len(snps)
    lines = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
    for alid, eaf, bp in snps:
        chrom, _pos, a1, a2 = alid.split(":")
        lines.append(f"{chrom}\t{alid}\t{a2}\t{a1}\t{eaf}\t{bp}")
    (block_dir / f"{block_name}.tsv").write_text("\n".join(lines) + "\n")
    ld = np.full((n, n), rho)
    np.fill_diagonal(ld, 1.0)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for row in ld:
            gz.write(("\t".join(f"{v:.6f}" for v in row) + "\n").encode())
    (block_dir / f"{block_name}.unphased.vcor1.gz").write_bytes(buf.getvalue())


def _make_ld_panel(tmp_path: Path) -> Path:
    root = tmp_path / "ld_panel"
    _write_ld_block(
        root / "EUR" / "1", "100000-1600000",
        [
            ("1:100000:A:G", 0.35, 100_000),
            ("1:764620:A:G", 0.30, 764_620),
            ("1:964620:A:G", 0.40, 964_620),
            ("1:1164620:A:G", 0.42, 1_164_620),
            ("1:1264620:C:T", 0.40, 1_264_620),  # panel-only → imputation target
            ("1:1364620:A:G", 0.44, 1_364_620),
            ("1:1564620:A:G", 0.45, 1_564_620),
        ],
    )
    return root


def _build_source(tmp_path: Path, ancestry: dict[str, str]) -> Path:
    vcfs = {name: _vcf(tmp_path, name) for name in ancestry}
    manifest = tmp_path / "manifest.tsv"
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\tassigned_ancestry"
    ]
    for name, anc in ancestry.items():
        lines.append(f"{name}\t{vcfs[name]}\t{name}\t1000\tsd\tdeclared_standardised\t{anc}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    panel = tmp_path / "panel.txt"
    panel.write_text("\n".join(PANEL_ALIDS) + "\n", encoding="utf-8")

    src = tmp_path / "src.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, src, reference_panel=panel, store_id="hyb", release_id="v1"
    )
    return src


def test_mixed_store_imputes_only_matching_ancestry(tmp_path):
    src = _build_source(tmp_path, {"trait_eur": "EUR", "trait_eas": "EAS"})
    ld = _make_ld_panel(tmp_path)
    dst = tmp_path / "dst.opengwasdb"

    complete_hybrid_store(src, dst, ld, ancestry="EUR", min_cor=0.0, thresh=0.9)
    assert validate_store(dst).ok, validate_store(dst).errors

    q = query_store(dst)
    # EUR-assigned analysis gains an imputed cell at the panel target.
    eur = q.analysis("trait_eur")
    assert "imputed" in set(eur["association_status"].tolist())
    # EAS-assigned analysis is untouched: observed-only, no imputed cells.
    eas = q.analysis("trait_eas")
    assert set(eas["association_status"].tolist()) == {"observed"}

    # completed_against recorded per Analysis (panel ancestry, or empty), in analyses.tsv.
    analyses = _analyses_by_id(dst)
    assert analyses["trait_eur"]["completed_against"] == "EUR"
    assert analyses["trait_eas"]["completed_against"] == ""
    assert analyses["trait_eas"]["assigned_ancestry"] == "EAS"  # still recorded


def test_an_excluded_analysis_declares_none_of_the_completion_it_did_not_get(tmp_path):
    """The metadata of an ancestry-excluded Analysis must describe its own cells.

    The blocks are imputed for every Analysis and the ancestry-match filter is
    applied afterwards, so a nonmatching Analysis arrives at the writer with
    both fills and completion-quality rows. Dropping only the fills left
    `completion_n_imputed_total` counting cells the release does not hold --
    and, because these VCFs carry no frequency while the panel does,
    `eaf_scope` was derived from that count and stamped `association` on an
    Analysis whose every cell reads NaN (ADR 0028, ADR 0037 §4).
    """
    src = _build_source(tmp_path, {"trait_eur": "EUR", "trait_eas": "EAS"})
    dst = tmp_path / "dst.opengwasdb"
    complete_hybrid_store(
        src, dst, _make_ld_panel(tmp_path), ancestry="EUR", min_cor=0.0, thresh=0.9
    )

    analyses = _analyses_by_id(dst)
    eur, eas = analyses["trait_eur"], analyses["trait_eas"]
    # Fixture meaningfulness: without a completed Analysis to compare against,
    # "declares nothing" would hold for a run that imputed nothing at all.
    assert int(eur["completion_n_imputed_total"]) > 0, "nothing was imputed; nothing is under test"
    assert eur["eaf_scope"] == "association", "the panel's frequencies never reached the release"

    assert int(eas["completion_n_imputed_total"] or 0) == 0
    assert eas["completion_median_pearson_r"] == ""
    assert eas["eaf_scope"] != "association", (
        "an Analysis excluded from completion holds no imputed cell, so it holds no "
        "frequency for one"
    )


def test_homogeneous_eur_store_no_regression(tmp_path):
    # Same store, both EUR: every analysis matches → both targets imputed.
    src = _build_source(tmp_path, {"trait_eur": "EUR", "trait_eur2": "EUR"})
    ld = _make_ld_panel(tmp_path)
    dst = tmp_path / "dst.opengwasdb"

    complete_hybrid_store(src, dst, ld, ancestry="EUR", min_cor=0.0, thresh=0.9)
    assert validate_store(dst).ok, validate_store(dst).errors

    q = query_store(dst)
    for tid in ("trait_eur", "trait_eur2"):
        assert "imputed" in set(q.analysis(tid)["association_status"].tolist())

    analyses = _analyses_by_id(dst)
    assert analyses["trait_eur"]["completed_against"] == "EUR"
    assert analyses["trait_eur2"]["completed_against"] == "EUR"


def test_mixed_store_imputes_fewer_cells_than_homogeneous(tmp_path):
    # The ancestry filter must reduce the imputed-cell count vs imputing all.
    (tmp_path / "mixed").mkdir()
    (tmp_path / "homo").mkdir()
    mixed_src = _build_source(tmp_path / "mixed", {"trait_eur": "EUR", "trait_eas": "EAS"})
    homo_src = _build_source(tmp_path / "homo", {"trait_eur": "EUR", "trait_eas": "EUR"})
    ld = _make_ld_panel(tmp_path)

    mixed = complete_hybrid_store(
        mixed_src, tmp_path / "mixed_dst.opengwasdb", ld, ancestry="EUR", min_cor=0.0, thresh=0.9
    )
    homo = complete_hybrid_store(
        homo_src, tmp_path / "homo_dst.opengwasdb", ld, ancestry="EUR", min_cor=0.0, thresh=0.9
    )
    assert mixed.n_imputed < homo.n_imputed
    assert mixed.n_imputed >= 1  # the matching EUR analysis still imputes
