"""Every writer judges an Analysis's associations whole, and logs an SE above 65,504.

An SE above `float16`'s largest value is stored as `+inf` in a `float16` plane
and stays an association (spec §15, issue #337). Until issue #338 records the
count in each Store, a build logs it per Analysis, once per writer that stores
the cells (#350).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import test_hybrid_build as hb
import test_hybrid_completion as thc
import test_ragged_completion as trc
from test_dense_vcf_build import HG19_POS_1, HG19_POS_3, _make_manifest, _make_vcf

from opengwasdb.admission import NotAdmitted
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRWriter

_TALLY = "SE value(s) above 65,504, more than a float16 plane holds"


def _tally_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and _TALLY in record.getMessage()
    ]


def test_the_ragged_writer_refuses_an_association_admission_would_not_admit() -> None:
    writer = RaggedCSRWriter(4)
    with pytest.raises(NotAdmitted, match=r"build-x, Analysis a: .*one of z and SE missing: 1"):
        writer.add_analysis(
            np.array([0, 2]),
            np.array([1.0, 2.0], np.float32),
            np.array([0.1, np.nan], np.float32),
            where="build-x, Analysis a",
        )


def test_the_ragged_writer_logs_each_analysis_s_out_of_range_ses(caplog) -> None:
    writer = RaggedCSRWriter(4)
    with caplog.at_level(logging.WARNING):
        for se in ([0.1, 0.2], [70_000.0, 65_504.0]):
            writer.add_analysis(
                np.array([0, 2]), np.array([1.0, 2.0], np.float32), np.array(se, np.float32)
            )
    assert _tally_lines(caplog) == [
        f"the Ragged writer, analysis_index 1: 1 {_TALLY} (spec §15)"
    ]


def test_a_dense_vcf_build_logs_the_analysis_whose_se_is_out_of_range(tmp_path, caplog) -> None:
    vcf_a = _make_vcf(
        tmp_path,
        "trait_a",
        [
            f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:100000\n",
            f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n",
        ],
    )
    vcf_b = _make_vcf(
        tmp_path, "trait_b", [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t6.0:0.5\n"]
    )
    manifest = _make_manifest(tmp_path, [("trait_a", vcf_a, "A"), ("trait_b", vcf_b, "B")])
    with caplog.at_level(logging.WARNING):
        build_dense_from_vcf_manifest(
            manifest, tmp_path / "dense.opengwasdb", store_id="s", release_id="r"
        )
    assert _tally_lines(caplog) == [f"build-dense-vcf, Analysis trait_a: 1 {_TALLY} (spec §15)"]


def test_a_hybrid_build_logs_each_component_s_out_of_range_ses(tmp_path: Path, caplog) -> None:
    """trait_a's SE above the range sits on the panel (Dense) and off it (Overflow)."""
    vcf_a = hb._make_vcf(
        tmp_path,
        "trait_a",
        [
            f"1\t{hb.HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t2.0:100000:0.2\n",
            f"1\t{hb.HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE:AF\t1.5:90000:0.3\n",
            f"1\t{hb.HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE:AF\t0.6:0.2:0.4\n",
        ],
    )
    vcf_b = hb._make_vcf(
        tmp_path, "trait_b", [f"1\t{hb.HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t6.0:0.5:0.25\n"]
    )
    manifest = hb._make_manifest(tmp_path, [("trait_a", vcf_a, "A"), ("trait_b", vcf_b, "B")])
    with caplog.at_level(logging.WARNING):
        build_hybrid_from_vcf_manifest(
            manifest,
            tmp_path / "hybrid.opengwasdb",
            reference_panel=hb._panel(tmp_path),
            store_id="s",
            release_id="r",
            n_workers=1,
        )
    assert sorted(_tally_lines(caplog)) == [
        f"build-hybrid trait_a Dense: 1 {_TALLY} (spec §15)",
        f"the Ragged writer, analysis_index 0: 1 {_TALLY} (spec §15)",
    ]


def _completion_lines(caplog: pytest.LogCaptureFixture, writer: str) -> list[str]:
    return [line for line in _tally_lines(caplog) if line.startswith(writer)]


def test_ragged_completion_logs_each_analysis_s_out_of_range_ses_once(tmp_path, caplog) -> None:
    """Counted once, by the write: the SE fit's round trip and the encode judge the
    same rows again, and log nothing."""
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    trc._write_esi(
        fixture / "test.esi",
        [
            {"chr": "1", "snp_id": "rs1001", "bp": 1_000_000, "a1": "A", "a2": "G"},
            {"chr": "1", "snp_id": "rs1002", "bp": 1_100_000, "a1": "C", "a2": "T"},
            {"chr": "1", "snp_id": "rs1003", "bp": 1_200_000, "a1": "A", "a2": "C"},
        ],
    )
    trc._write_epi(
        fixture / "test.epi",
        [
            {"chr": "1", "probe_id": "ENSG00000000001", "bp": 1_050_000, "gene": "GENE1"},
            {"chr": "1", "probe_id": "ENSG00000000002", "bp": 1_150_000, "gene": "GENE2"},
        ],
    )
    trc._write_besd_sparse_3f(
        fixture / "test.besd",
        2,
        [[(0, 0.1, 0.02), (1, -0.2, 70_000.0)], [(1, 0.5, 0.05), (2, -0.15, 0.025)]],
    )
    observed = tmp_path / "obs.opengwasdb"
    trc.build_ragged_from_besd(
        fixture / "test", observed, store_id="t", release_id="obs", tissue="Blood",
        association_coverage="cis",
    )
    with caplog.at_level(logging.WARNING):
        trc.complete_ragged_store(
            observed, tmp_path / "comp.opengwasdb",
            trc._make_ld_panel(tmp_path, "1", 900_000, 1_300_000),
            ancestry="EUR", cis_window_bp=500_000, min_cor=0.0, release_id="comp",
        )
    assert _completion_lines(caplog, "complete-ragged") == [
        f"complete-ragged: 1 {_TALLY} (spec §15): ENSG00000000001::Blood: 1"
    ]


def test_the_crossover_fold_logs_its_out_of_range_se(tmp_path, caplog) -> None:
    """trait_a's off-panel association at 1:2068561 becomes a crossover once the LD
    panel extends the Dense axis. Only the fold holds it: Dense completion ran
    before it was folded in, and the rebuilt Overflow leaves it out."""
    src = thc.build_two_trait_source(
        tmp_path,
        {
            "trait_a": [(100_000, "2.0:0.5"), (1_000_000, "1.5:0.3"), (2_000_000, "1.2:70000")],
            "trait_c": [(100_000, "2.2:0.5"), (1_000_000, "1.6:0.3")],
        },
    )
    with caplog.at_level(logging.WARNING):
        thc.complete_hybrid_store(
            src, tmp_path / "dst.opengwasdb", thc._make_ld_panel_with_crossover(tmp_path),
            min_cor=0.0, thresh=0.9,
        )
    assert _completion_lines(caplog, "complete-hybrid") == [
        f"complete-hybrid, the crossover fold: 1 {_TALLY} (spec §15): trait_a: 1"
    ]
