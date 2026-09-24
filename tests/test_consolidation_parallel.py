"""The EAF spill survey and the Ragged Overflow CSR assembly across --n-workers.

Both phases walk independent columns on one core in the serial path; #219 runs
them through `ordered_map` while combining results in Analysis order. These
tests build fixtures whose columns finish out of order -- the first column is
orders of magnitude larger than the rest, so it is the last to complete -- and
assert the parallel path is byte-for-byte the serial one, including the order
of the concatenated EAF sample. A combination in completion order reorders that
sample (and, for the CSR, its offsets) and fails here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_hybrid_build import _make_manifest, _make_vcf, _panel

import opengwasdb.layouts.dense.build_vcf as dense_build
from opengwasdb.build.eaf_orientation import site_hashes
from opengwasdb.layouts.dense.build_vcf import survey_eaf_spills
from opengwasdb.layouts.hybrid.build import (
    _assemble_overflow_csr,
    build_hybrid_from_vcf_manifest,
)
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRWriter

# The first column is far larger than the rest, so under any pool it is the
# last to finish: a map combined in completion order puts it last, not first.
_BIG = 200_000
_SMALL = 4
_ALIDS = [f"1:{i + 1}:A:G" for i in range(_BIG + 2 * _SMALL)]
_ID_BY_COL = {col: f"trait_{col}" for col in range(6)}
_k = 32


def _write_survey_spills(
    root: Path,
    *,
    suffix: str = "",
    index_key: str = "rows",
    row_map: np.ndarray | None = None,
) -> None:
    rng = np.random.default_rng(219)
    for col, size in enumerate((_BIG, _SMALL, _SMALL, _SMALL, 0, _SMALL)):
        if size == 0:
            continue  # column 4 has no spill at all
        if row_map is None:
            rows = np.arange(col * _SMALL, col * _SMALL + size, dtype=np.int64)
        else:
            # Spill row indices live on a smaller dense axis; row_map carries
            # them onto the shared axis the hashes describe.
            rows = np.arange(size, dtype=np.int64) % len(row_map)
        eaf = rng.random(size)
        eaf[::7] = np.nan
        np.savez(root / f"{col}{suffix}.npz", **{index_key: rows, "eaf": eaf})


def _survey(root: Path, n_workers: int, *, row_map: np.ndarray | None = None) -> object:
    return survey_eaf_spills(
        root,
        _ID_BY_COL,
        _ALIDS,
        site_hashes(_ALIDS),
        k=_k,
        suffix=".ovf" if row_map is not None else "",
        index_key="variant_index" if row_map is not None else "rows",
        row_map=row_map,
        n_workers=n_workers,
    )


def _assert_surveys_match(left, right) -> None:
    assert left.observations == right.observations
    assert left.n_spill_cells == right.n_spill_cells
    assert left.n_eaf_cells == right.n_eaf_cells
    np.testing.assert_array_equal(left.sample_rows, right.sample_rows)
    np.testing.assert_array_equal(left.sample_values, right.sample_values)


def test_survey_parallel_matches_serial(tmp_path: Path) -> None:
    _write_survey_spills(tmp_path)
    serial = _survey(tmp_path, n_workers=1)
    parallel = _survey(tmp_path, n_workers=4)

    # Fixture is meaningful: the survey read real sample rows and frequencies.
    assert serial.sample_rows.size > 0
    assert serial.n_spill_cells == _BIG + 4 * _SMALL
    assert serial.n_eaf_cells > 0
    assert serial.observations["trait_4"] == {}

    _assert_surveys_match(serial, parallel)


def test_survey_parallel_matches_serial_with_row_map(tmp_path: Path) -> None:
    row_map = np.arange(len(_ALIDS), dtype=np.int64)[::-1]
    _write_survey_spills(tmp_path, suffix=".ovf", index_key="variant_index", row_map=row_map)
    serial = _survey(tmp_path, n_workers=1, row_map=row_map)
    parallel = _survey(tmp_path, n_workers=4, row_map=row_map)

    assert serial.sample_rows.size > 0
    _assert_surveys_match(serial, parallel)


def _write_overflow_spills(root: Path, n_variants: int) -> None:
    rng = np.random.default_rng(219)
    for col, size in enumerate((100_000, _SMALL, _SMALL, _SMALL, 0, _SMALL)):
        if size == 0:
            continue  # column 4 has no spill
        variant_index = np.sort(rng.integers(0, n_variants, size=size)).astype(np.int32)
        z = rng.standard_normal(size).astype(np.float32)
        se = np.abs(rng.standard_normal(size)).astype(np.float32)
        eaf = rng.random(size).astype(np.float32)
        if col == 1:
            eaf = np.full(size, np.nan, dtype=np.float32)  # no frequency at all
        np.savez(
            root / f"{col}.ovf.npz", variant_index=variant_index, z=z, se=se, eaf=eaf
        )


def _csr_contents(writer: RaggedCSRWriter) -> tuple[list, list, list, list, list]:
    return (
        list(writer._offsets),
        [np.array(a) for a in writer._variant_indices],
        [np.array(a) for a in writer._zscores],
        [np.array(a) for a in writer._ses],
        [np.array(a) for a in writer._eafs],
    )


def test_overflow_csr_parallel_matches_serial(tmp_path: Path) -> None:
    n_variants = 1_000_000
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial_dir.mkdir()
    parallel_dir.mkdir()
    _write_overflow_spills(serial_dir, n_variants)
    _write_overflow_spills(parallel_dir, n_variants)

    serial_csr, serial_has_eaf = _assemble_overflow_csr(
        serial_dir, len(_ID_BY_COL), n_variants, n_workers=1
    )
    parallel_csr, parallel_has_eaf = _assemble_overflow_csr(
        parallel_dir, len(_ID_BY_COL), n_variants, n_workers=4
    )

    # Fixture is meaningful: the first column dominates the CSR and the
    # no-frequency column really carries none.
    assert serial_csr.n_analyses == len(_ID_BY_COL)
    assert serial_csr.n_associations == 100_000 + 4 * _SMALL
    assert not serial_has_eaf[1]

    np.testing.assert_array_equal(serial_has_eaf, parallel_has_eaf)
    left_offsets, left_vi, left_z, left_se, left_eaf = _csr_contents(serial_csr)
    right_offsets, right_vi, right_z, right_se, right_eaf = _csr_contents(parallel_csr)
    assert left_offsets == right_offsets
    for left, right in zip(
        (left_vi, left_z, left_se, left_eaf),
        (right_vi, right_z, right_se, right_eaf),
        strict=True,
    ):
        for part_left, part_right in zip(left, right, strict=True):
            np.testing.assert_array_equal(part_left, part_right)

    # Both phases consumed their spills.
    assert not list(serial_dir.glob("*.ovf.npz"))
    assert not list(parallel_dir.glob("*.ovf.npz"))


def test_overflow_spill_removed_only_after_use(tmp_path: Path, monkeypatch) -> None:
    """A spill is unlinked after its Analysis reaches the CSR, never before: a
    failure adding one column leaves that column and the later ones on disk."""
    _write_overflow_spills(tmp_path, n_variants=1_000_000)
    original = RaggedCSRWriter.add_analysis
    calls = 0

    def flaky(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("assembly failed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RaggedCSRWriter, "add_analysis", flaky)
    with pytest.raises(RuntimeError, match="assembly failed"):
        _assemble_overflow_csr(tmp_path, len(_ID_BY_COL), 1_000_000, n_workers=1)

    assert not (tmp_path / "0.ovf.npz").exists()  # used, then deleted
    assert (tmp_path / "1.ovf.npz").exists()  # the failing column, still unused
    assert (tmp_path / "2.ovf.npz").exists()  # never reached


def test_hybrid_survey_failure_leaves_no_spill_dir(tmp_path: Path, monkeypatch) -> None:
    """A worker failure in the parallel survey propagates and the build's
    spill-dir lifetime still removes every spill."""
    vcf1 = _make_vcf(
        tmp_path,
        "trait_a",
        [
            "1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t2.0:0.5:0.2\n",
            "1\t1000000\t.\tC\tT\t.\tPASS\t.\tES:SE:AF\t1.5:0.3:0.3\n",
        ],
    )
    vcf2 = _make_vcf(
        tmp_path,
        "trait_b",
        ["1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t6.0:0.5:0.25\n"],
    )
    manifest = _make_manifest(
        tmp_path, [("trait_a", vcf1, "Trait A"), ("trait_b", vcf2, "Trait B")]
    )

    def boom(rows, eaf, hashes, *, k: int) -> tuple[np.ndarray, np.ndarray]:
        raise RuntimeError("survey failed")

    monkeypatch.setattr(dense_build, "sample_column_rows", boom)
    with pytest.raises(RuntimeError, match="survey failed"):
        build_hybrid_from_vcf_manifest(
            manifest,
            tmp_path / "failed.opengwasdb",
            reference_panel=_panel(tmp_path),
            store_id="s",
            release_id="r",
            n_workers=2,
        )

    assert not list(tmp_path.glob(".*hybridspill.*"))
    assert not list(tmp_path.glob(".*pass2spill.*"))
