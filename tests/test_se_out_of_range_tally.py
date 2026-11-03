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
