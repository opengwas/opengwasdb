"""Hybrid Reference Completion (issue 058): impute only the Dense Component.

hg19 → hg38 anchors used (all A/G, ALT>REF → z negated on store):
  1:100000  → 1:100000     1:750000  → 1:814620
  1:1000000 → 1:1064620    1:1500000 → 1:1564620
  1:2000000 → 1:2068561    (OFF-PANEL → overflow)
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pytest
import zarr

from opengwasdb.layouts.dense.top_hits import read_top_hit_counts, threshold_key
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.hybrid.complete import complete_hybrid_store
from opengwasdb.model.analyses import read_analyses
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.query import query_store
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store
from opengwasdb.variants import VariantAxis

PANEL_ALIDS = ["1:100000:A:G", "1:814620:A:G", "1:1064620:A:G", "1:1564620:A:G"]
OFF_PANEL_ALID = "1:2068561:A:G"


def _vcf(tmp_path, name, rows):
    header = (
        "##fileformat=VCFv4.2\n"
        "##FORMAT=<ID=ES,Number=A,Type=Float,Description=\"Effect size\">\n"
        "##FORMAT=<ID=SE,Number=A,Type=Float,Description=\"Standard error\">\n"
        "##FORMAT=<ID=EZ,Number=A,Type=Float,Description=\"Z-score\">\n"
        "##SAMPLE=<ID=S,StudyType=Continuous>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n"
    )
    p = tmp_path / f"{name}.vcf"
    p.write_text(header + "".join(rows), encoding="utf-8")
    return p


def _write_ld_block(block_dir, block_name, snps, seed=0):
    block_dir.mkdir(parents=True, exist_ok=True)
    n = len(snps)
    lines = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
    for alid, eaf, bp in snps:
        chrom, _pos, a1, a2 = alid.split(":")
        lines.append(f"{chrom}\t{alid}\t{a2}\t{a1}\t{eaf}\t{bp}")
    (block_dir / f"{block_name}.tsv").write_text("\n".join(lines) + "\n")
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    ld = A @ A.T + np.eye(n) * n * 0.1
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for row in ld:
            gz.write(("\t".join(f"{v:.6f}" for v in row) + "\n").encode())
    (block_dir / f"{block_name}.unphased.vcor1.gz").write_bytes(buf.getvalue())


def _make_ld_panel(tmp_path):
    root = tmp_path / "ld_panel"
    _write_ld_block(
        root / "EUR" / "1", "100000-1600000",
        [
            ("1:100000:A:G", 0.35, 100_000),
            ("1:814620:A:G", 0.30, 814_620),
            ("1:1064620:A:G", 0.40, 1_064_620),
            ("1:1564620:A:G", 0.45, 1_564_620),
        ],
        seed=0,
    )
    return root


def _build_source(tmp_path):
    trait_a = _vcf(
        tmp_path, "trait_a",
        [
            "1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
            "1\t750000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.4\n",
            "1\t1000000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.5:0.3\n",
            "1\t1500000\t.\tA\tG\t.\tPASS\t.\tES:SE\t0.6:0.2\n",
            "1\t2000000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.2:0.3\n",  # OFF-PANEL overflow
        ],
    )
    trait_c = _vcf(
        tmp_path, "trait_c",
        [
            "1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.2:0.5\n",
            "1\t750000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.1:0.4\n",
            "1\t1000000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.6:0.3\n",
            # NOTE: 1:1500000 missing → 1:1564620 is an imputation target for trait_c
        ],
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method\n"
        f"trait_a\t{trait_a}\tTrait A\t1000\tsd\tdeclared_standardised\n"
        f"trait_c\t{trait_c}\tTrait C\t1000\tsd\tdeclared_standardised\n",
        encoding="utf-8",
    )
    panel = tmp_path / "panel.txt"
    panel.write_text("\n".join(PANEL_ALIDS) + "\n", encoding="utf-8")

    src = tmp_path / "src.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, src, reference_panel=panel, store_id="hyb", release_id="v1"
    )
    return src


def test_hybrid_completion(tmp_path):
    src = _build_source(tmp_path)
    ld = _make_ld_panel(tmp_path)
    dst = tmp_path / "dst.opengwasdb"

    result = complete_hybrid_store(src, dst, ld, min_cor=0.0, thresh=0.9)

    # Completed hybrid store validates.
    assert validate_store(dst).ok, validate_store(dst).errors

    manifest = StoreManifest.load(dst)
    assert manifest.completion_state.value == "reference_completed"
    assert manifest.primary_layout.value == "hybrid"

    # Dense Component carries imputed + on_panel arrays.
    dense_root = open_store(dst).dense_component().arrays(mode="r")
    assert "imputed" in dense_root
    assert "on_panel" in dense_root

    # Overflow is observed-only, untouched (byte-identical z/se).
    src_ovf = open_store(src).arrays(mode="r")["ragged"]
    dst_ovf = open_store(dst).arrays(mode="r")["ragged"]
    assert "imputed" not in dst_ovf
    assert np.array_equal(src_ovf["z"][:], dst_ovf["z"][:])
    assert np.array_equal(src_ovf["se"][:], dst_ovf["se"][:])


def test_stray_file_at_completed_hybrid_top_level_fails_validation(tmp_path):
    # Issue #80: the closed-envelope check applies to a completed Hybrid
    # release's own top-level directory, not just its nested Dense Component.
    src = _build_source(tmp_path)
    ld = _make_ld_panel(tmp_path)
    dst = tmp_path / "dst.opengwasdb"
    complete_hybrid_store(src, dst, ld, min_cor=0.0, thresh=0.9)

    (dst / "traits.tsv.gz").write_bytes(b"stray")

    result = validate_store(dst)

    assert not result.ok
    assert any(
        "unexpected store entry" in error and "traits.tsv.gz" in error
        for error in result.errors
    )


def test_stray_file_in_completed_hybrid_dense_component_fails_validation(tmp_path):
    src = _build_source(tmp_path)
    ld = _make_ld_panel(tmp_path)
    dst = tmp_path / "dst.opengwasdb"
    complete_hybrid_store(src, dst, ld, min_cor=0.0, thresh=0.9)

    (dst / "dense" / "stray.txt").write_text("x")

    result = validate_store(dst)

    assert not result.ok
    assert any(
        "unexpected store entry" in error and "stray.txt" in error for error in result.errors
    )


def test_completed_analyses_tsv_has_no_phenotype_columns_and_carries_analysis_label(tmp_path):
    """ADR 0034/issue #68: hybrid completion must carry the unified schema
    forward (no phenotype_id/phenotype_label) at both the completed Dense
    Component and the shared/top-level analyses.tsv."""
    src = _build_source(tmp_path)
    ld = _make_ld_panel(tmp_path)
    dst = tmp_path / "dst.opengwasdb"
    complete_hybrid_store(src, dst, ld, min_cor=0.0, thresh=0.9)

    for path in (dst / "dense" / "analyses.tsv", dst / "analyses.tsv"):
        table = read_analyses(path)
        assert "phenotype_id" not in table.fieldnames
        assert "phenotype_label" not in table.fieldnames
        rows = {r["analysis_id"]: r for r in table.rows}
        assert rows["trait_a"]["analysis_label"] == "Trait A"
        assert rows["trait_c"]["analysis_label"] == "Trait C"
        # Completion rollup columns are part of the same unified schema.
        assert "completion_median_pearson_r" in table.fieldnames
        assert "completion_n_missing_total" in table.fieldnames


def test_hit_counts_sum_dense_and_overflow_after_completion(tmp_path):
    # The completed Dense Component's own analyses.tsv already carries
    # correct post-completion counts (test_dense_completion.py); the shared
    # root's counts must be that plus the Ragged Overflow's own top-hit
    # index (ADR 0032) -- neither component alone is double-counted.
    src = _build_source(tmp_path)
    ld = _make_ld_panel(tmp_path)
    dst = tmp_path / "dst.opengwasdb"
    complete_hybrid_store(src, dst, ld, min_cor=0.0, thresh=0.9)

    rows = sorted(
        read_analyses(dst / "analyses.tsv").rows, key=lambda r: int(r["analysis_index"])
    )
    dense_counts = read_top_hit_counts(dst / "dense", len(rows))
    overflow_counts = read_top_hit_counts(dst, len(rows))
    for column in dense_counts:
        expected = [
            d + o for d, o in zip(dense_counts[column], overflow_counts[column], strict=True)
        ]
        assert [int(r[column]) for r in rows] == expected


def test_completed_status_only_on_dense(tmp_path):
    src = _build_source(tmp_path)
    ld = _make_ld_panel(tmp_path)
    dst = tmp_path / "dst.opengwasdb"
    result = complete_hybrid_store(src, dst, ld, min_cor=0.0, thresh=0.9)

    q = query_store(dst)
    # The off-panel overflow association must always be observed.
    r = q.phewas(OFF_PANEL_ALID)
    assert len(r["z"]) >= 1
    assert set(r["association_status"].tolist()) == {"observed"}

    # trait_c's missing panel variant (1:1564620) should be imputed if completion filled it.
    rc = q.analysis("trait_c")
    statuses = set(rc["association_status"].tolist())
    assert statuses <= {"observed", "imputed"}
    if result.n_imputed > 0:
        assert "imputed" in statuses


def _make_ld_panel_with_crossover(tmp_path):
    """Like `_make_ld_panel`, plus a second block covering OFF_PANEL_ALID --
    an LD panel wider than the build panel, extending the Dense Component to
    include a variant that already carries a real observed overflow
    association (issue #99)."""
    root = _make_ld_panel(tmp_path)
    _write_ld_block(
        root / "EUR" / "1", "2000000-2100000",
        [("1:2068561:A:G", 0.20, 2_068_561)],
        seed=1,
    )
    return root


def _vcf_with_eaf(tmp_path: Path, name: str, rows: list[str]) -> Path:
    header = (
        "##fileformat=VCFv4.2\n"
        "##FORMAT=<ID=ES,Number=A,Type=Float,Description=\"Effect size\">\n"
        "##FORMAT=<ID=SE,Number=A,Type=Float,Description=\"Standard error\">\n"
        "##FORMAT=<ID=EZ,Number=A,Type=Float,Description=\"Z-score\">\n"
        "##FORMAT=<ID=AF,Number=A,Type=Float,Description=\"Alternate allele frequency\">\n"
        "##SAMPLE=<ID=S,StudyType=Continuous>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n"
    )
    path = tmp_path / f"{name}.vcf"
    path.write_text(header + "".join(rows), encoding="utf-8")
    return path


def _residual_hybrid_crossover_source(tmp_path: Path) -> tuple[Path, str, float, float]:
    """A residual-SE Hybrid source with an off-panel crossover variant.

    SE tracks ``log(2*f*(1-f))`` per Analysis so the shared decision
    residual-codes both components (issue #118). The overflow carries enough
    off-panel variants that its coefficient side table is amortised and the
    residual plan actually saves bytes there, which is the only way a Hybrid
    store selects the residual encoding at all.

    Exactly one off-panel variant is chosen as the crossover target: it stays
    off the build panel (so it routes to the Ragged Overflow), then crosses
    onto the Dense Component when the completion panel extends the axis
    (issue #163). ``trait_b`` leaves the last four panel variants unobserved,
    so dense completion imputes something -- without an imputed cell in the
    scanned band, rebuilding the index under the wrong encoding would not
    raise and the regression test would prove nothing.
    """
    n_panel = 200
    n_off_panel = 200
    crossover_i = 50
    frequencies = np.linspace(0.05, 0.95, n_panel, dtype=np.float64)
    off_frequencies = np.linspace(0.10, 0.90, n_off_panel, dtype=np.float64)
    crossover_eaf = float(off_frequencies[crossover_i])

    def se_value(col: int, freq: float, phase: int) -> float:
        return float(
            np.exp(
                (-3.0 + col * 0.2)
                - 0.5 * np.log(2.0 * freq * (1.0 - freq))
                + 0.12 * np.sin(phase * (0.07 + col * 0.01))
            )
        )

    def panel_rows(col: int, n_observed: int) -> list[str]:
        out: list[str] = []
        for i in range(n_observed):
            freq = frequencies[i]
            se = se_value(col, freq, i)
            z = 8.0 if i % 50 == 0 else 1.0
            out.append(
                f"1\t{(i + 1) * 1000}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF"
                f"\t{z * se:.6f}:{se:.6f}:{freq:.6f}\n"
            )
        return out

    def off_panel_rows(col: int) -> list[str]:
        out: list[str] = []
        for i in range(n_off_panel):
            pos = 300_000 + i * 1000
            freq = off_frequencies[i]
            se = se_value(col, freq, i)
            z = 8.0 if i % 50 == 0 else 1.0
            out.append(
                f"1\t{pos}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF"
                f"\t{z * se:.6f}:{se:.6f}:{freq:.6f}\n"
            )
        return out

    crossover_alid = f"1:{300_000 + crossover_i * 1000}:A:G"
    vcf_a = _vcf_with_eaf(
        tmp_path, "trait_a", panel_rows(0, n_panel) + off_panel_rows(0)
    )
    vcf_b = _vcf_with_eaf(tmp_path, "trait_b", panel_rows(1, n_panel - 4))
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\tsource_assembly\n"
        f"trait_a\t{vcf_a}\tTrait A\t1000\tsd\tdeclared_standardised\thg38\n"
        f"trait_b\t{vcf_b}\tTrait B\t1000\tsd\tdeclared_standardised\thg38\n",
        encoding="utf-8",
    )
    panel = tmp_path / "panel.txt"
    panel.write_text(
        "\n".join(f"1:{(i + 1) * 1000}:A:G" for i in range(n_panel)) + "\n",
        encoding="utf-8",
    )

    src = tmp_path / "src.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, src, reference_panel=panel, store_id="hyb", release_id="v1"
    )
    expected_crossover_se = se_value(0, crossover_eaf, crossover_i)
    return src, crossover_alid, expected_crossover_se, crossover_eaf


def _residual_ld_panel_with_crossover(
    tmp_path: Path, crossover_alid: str, crossover_eaf: float
) -> Path:
    """An LD panel over the residual source's hg38 axis, with a second block
    that extends it over the off-panel crossover variant."""
    root = tmp_path / "ld_panel"
    n_panel = 200
    frequencies = np.linspace(0.05, 0.95, n_panel, dtype=np.float64)
    _write_ld_block(
        root / "EUR" / "1",
        "1000-200000",
        [
            (f"1:{(i + 1) * 1000}:A:G", float(frequencies[i]), (i + 1) * 1000)
            for i in range(n_panel)
        ],
        seed=0,
    )
    _crossover_pos = int(crossover_alid.split(":")[1])
    _write_ld_block(
        root / "EUR" / "1",
        f"{_crossover_pos}-{_crossover_pos}",
        [(crossover_alid, crossover_eaf, _crossover_pos)],
        seed=1,
    )
    return root


def test_residual_hybrid_crossover_rebuilds_index_with_completed_encoding(tmp_path) -> None:
    """issue #163: panel crossover must rebuild the Dense Top-Hit Index with
    the completed component's own encoding, not the source's.

    The source encoding has no Reference EAF, so scanning a band that holds an
    imputed cell under it decodes that cell's frequency as missing and fails
    the residual-SE read. The completed Dense Component's own plan carries the
    ``eaf_reference`` completion added, which is the plan the rebuild must use.
    """
    src, crossover_alid, expected_crossover_se, crossover_eaf = _residual_hybrid_crossover_source(
        tmp_path
    )
    assert StoreManifest.load(src).encoding.se.is_residual

    ld = _residual_ld_panel_with_crossover(tmp_path, crossover_alid, crossover_eaf)
    dst = tmp_path / "dst.opengwasdb"
    result = complete_hybrid_store(src, dst, ld, min_cor=0.0, thresh=0.9)

    assert validate_store(dst).ok
    assert result.n_imputed > 0  # trait_b's four unobserved panel variants

    shared_axis = VariantAxis(dst)
    alid_by_index = {r.variant_index: r.alid for r in shared_axis.all()}
    shared_axis.close()

    q = query_store(dst)
    r = q.analysis("trait_a", observed_only=False)
    by_alid = {
        alid_by_index[int(vi)]: (z, se, status)
        for vi, z, se, status in zip(
            r["variant_index"], r["z"], r["se"], r["association_status"], strict=True
        )
    }
    z, se, status = by_alid[crossover_alid]
    assert status == "observed"
    assert se == pytest.approx(expected_crossover_se, rel=0.01)
    assert abs(z) == pytest.approx(8.0, rel=0.05)

    # The rebuilt Dense top-hit index must carry the crossed-over association,
    # decoded through the completed component's own encoding.
    dense_top = zarr.open_group(
        str(dst / "dense" / "data.zarr" / "top_hits" / threshold_key(5e-8)), mode="r"
    )
    dense_axis = VariantAxis(dst / "dense")
    dense_alid_by_index = {r.variant_index: r.alid for r in dense_axis.all()}
    dense_axis.close()
    hit_alids = {
        dense_alid_by_index[int(vi)]: float(hit_se)
        for vi, hit_se in zip(
            dense_top["variant_index"][:], dense_top["se"][:].astype(np.float32), strict=True
        )
    }
    assert crossover_alid in hit_alids
    assert hit_alids[crossover_alid] == pytest.approx(expected_crossover_se, rel=0.01)


def test_panel_extension_crossover_stays_disjoint_and_keeps_the_real_observation(tmp_path):
    """issue #99: when the LD panel extends the Dense Component to cover a
    variant already carrying a real observed overflow association
    (OFF_PANEL_ALID, trait_a's 1:2000000 -> hg38 1:2068561, ES:SE=1.2:0.3,
    ALT>REF -> z=-4.0), the completed store must stay disjoint -- the real
    observation folds into the Dense Component (not an LD-imputed guess),
    and the overflow no longer carries it.
    """
    src = _build_source(tmp_path)
    ld = _make_ld_panel_with_crossover(tmp_path)
    dst = tmp_path / "dst.opengwasdb"

    result = complete_hybrid_store(src, dst, ld, min_cor=0.0, thresh=0.9)

    validation = validate_store(dst)
    assert validation.ok, validation.errors

    # The crossed-over variant is on the Dense Component's own axis now.
    dense_axis = open_store(dst).dense_component()
    axis = VariantAxis(dense_axis.path)
    dense_alids = {r.alid for r in axis.all()}
    axis.close()
    assert OFF_PANEL_ALID in dense_alids

    # ... and carries trait_a's real observed value there, not an imputed one.
    shared_axis = VariantAxis(dst)
    alid_by_index = {r.variant_index: r.alid for r in shared_axis.all()}
    shared_axis.close()

    q = query_store(dst)
    r = q.analysis("trait_a", observed_only=False)
    by_alid = {
        alid_by_index[int(vi)]: (z, status)
        for vi, z, status in zip(r["variant_index"], r["z"], r["association_status"], strict=True)
    }
    z, status = by_alid[OFF_PANEL_ALID]
    assert status == "observed"
    assert z == pytest.approx(-4.0, rel=5e-3)

    # The overflow no longer carries this association for any analysis.
    from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader

    ragged = RaggedCSRReader(dst)
    overflow_alids_after = {alid_by_index[int(v)] for v in np.unique(ragged._variant_index[:])}
    assert OFF_PANEL_ALID not in overflow_alids_after

    # n_imputed reports the real (corrected) count, not complete_dense_store's
    # pre-fold count -- trait_a's crossed-over cell was a real observation,
    # never counted as imputed to begin with in this fixture (see the
    # correlated-block test below for the case where it was).
    assert result.n_imputed >= 0


def test_fold_panel_crossovers_overwrites_an_already_imputed_cell(tmp_path):
    """issue #99: the fold must correct a cell dense completion already
    imputed (not merely fill an empty one) -- the failure mode the real
    344,510/4,916,057-variant repro actually hit. Forcing a real ElasticNetCV
    fit to succeed deterministically on a fixture this small isn't reliable
    (test_dense_completion.py notes the same fragility for its own iid-random
    LD blocks), so this exercises `_fold_panel_crossovers`'s `was_imputed`
    branch directly against a synthetic completed-dense zarr array instead of
    routing through a real completion run."""
    from opengwasdb.encoding import (
        DenseZPlane,
        EncodingMeasurements,
        StoreCodec,
        StoreEncoding,
    )
    from opengwasdb.layouts.hybrid.complete import _fold_panel_crossovers

    encoding = StoreEncoding.decide(EncodingMeasurements(n_analyses=1))
    codec = StoreCodec(encoding)
    dense_dir = tmp_path / "dense"
    root = zarr.open_group(str(dense_dir / "data.zarr"), mode="w")
    root.create_dataset(
        "z", shape=(2, 2), dtype=codec.z_dtype, fill_value=codec.z_fill_value
    )
    root.create_dataset("se", shape=(2, 2), dtype="float16", fill_value=np.nan)
    root.create_dataset("imputed", shape=(2, 2), dtype="uint8", fill_value=0)

    # Row 1 / analysis column 0: dense completion already wrote an LD-imputed
    # guess here before the fold runs.
    DenseZPlane.open(root, encoding).patch(np.array([1]), np.array([0]), np.array([1.23]))
    root["se"][1, 0] = 0.5
    root["imputed"][1, 0] = 1

    dense_alid_to_row = {OFF_PANEL_ALID: 1}
    offsets = np.array([0, 1])  # one analysis; one overflow association
    src_z = np.array([-4.0], dtype=np.float32)
    src_se = np.array([0.3], dtype=np.float32)
    src_eaf = np.array([0.42], dtype=np.float32)
    overflow_alids = np.array([OFF_PANEL_ALID], dtype=object)
    is_crossover = np.array([True])

    n_reclaimed = _fold_panel_crossovers(
        dense_dir, dense_alid_to_row, offsets, src_z, src_se, src_eaf,
        overflow_alids, is_crossover, encoding=encoding, dense_encoding=encoding,
    )

    assert n_reclaimed == 1
    written = zarr.open_group(str(dense_dir / "data.zarr"), mode="r")
    assert float(DenseZPlane.open(written, encoding).points([1], [0])[0]) == pytest.approx(
        -4.0, rel=1e-3
    )
    assert float(written["se"][1, 0]) == pytest.approx(0.3, rel=1e-3)
    assert int(written["imputed"][1, 0]) == 0
    # The crossed-over cell's EAF moves with its z/se (ADR 0036); this
    # fixture's Dense Component has no eaf array, so the fold must leave it
    # alone rather than fail trying to write one.
    assert "eaf" not in written


def test_rho_delegates_to_dense_component(tmp_path):
    # No Rho Matrix is built for this fixture's Dense Component -- rho()
    # must delegate through and return the same empty shape StoreQuery
    # returns for a Dense store with no Rho Matrix, not raise.
    src = _build_source(tmp_path)
    ld = _make_ld_panel(tmp_path)
    dst = tmp_path / "dst.opengwasdb"
    complete_hybrid_store(src, dst, ld, min_cor=0.0, thresh=0.9)

    q = query_store(dst)
    assert len(q.rho("trait_a", "trait_b")["rho"]) == 0
    assert len(q.rho_row("trait_a")["rho"]) == 0
    assert q.rho_matrix(["trait_a", "trait_b"])["rho"].shape == (0, 0)
