"""The Dense SE decision comes from a bounded, deterministic sample (issue #146).

The measurement pass used to visit every row chunk and compress every cell up
to four times, so a genome-scale store paid a full survey for a decision that
only needs representative statistics. The survey now visits at most
`MAX_MEASURED_CHUNKS` whole chunks, spread evenly, and scales byte totals by
the cells it saw. These tests pin the two things that must not move:

- the decision must be the same one an exhaustive survey would make (a sample
  that picks the wrong range silently changes what a store contains), and
- an Analysis whose fit is bad enough to revert the plane must still be caught
  when a bounded sample would only see a fraction of it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from opengwasdb.encoding.measure import (
    SeMeasurementRecord,
    sample_chunks,
)
from opengwasdb.encoding.plan import EafEncoding, SeEncoding, StoreEncoding, ZEncoding
from opengwasdb.encoding.se import optimise_dense_se_joint


def _preliminary() -> StoreEncoding:
    return StoreEncoding(
        z=ZEncoding("float16"),
        se=SeEncoding("float16"),
        eaf=EafEncoding("float32"),
    )


def _grid(tmp_path, *, well_fitted: bool, n_clean: int = 1) -> tuple[object, np.ndarray]:
    """A Dense scratch grid with i.i.d. rows and optional ragged columns."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(tmp_path / "dense.zarr"), mode="w")
    rng = np.random.default_rng(7)
    n_rows = 40_000
    eaf = rng.uniform(0.05, 0.95, (n_rows, n_clean)).astype(np.float32)
    predictor = np.log(2 * eaf * (1 - eaf))
    if well_fitted:
        se = np.exp(-3.0 - 0.5 * predictor + 0.05 * rng.standard_normal(eaf.shape)).astype(
            np.float32
        )
    else:
        # Unrelated to frequency: residuals span many decades, so the plane
        # must revert to float16 no matter how little of it a sample sees.
        se = np.exp(rng.normal(-3.0, 2.0, eaf.shape)).astype(np.float32)
    group.create_dataset("eaf", data=eaf, chunks=(1000, 1), dtype="float32")
    group.create_dataset("se", data=se, chunks=(1000, 1), dtype="float32")
    group.create_dataset("z", data=np.ones_like(eaf), chunks=(1000, 1), dtype="float16")
    return group, se


def test_sample_chunks_is_bounded_deterministic_and_full_when_small() -> None:
    # A plane with fewer chunks than the cap is measured exhaustively.
    full = sample_chunks(40_000, 1000)
    assert full.is_full
    assert full.sampled_chunks == full.total_chunks == 40

    big = sample_chunks(40_000 * 100, 1000)  # 4,000 chunks
    assert big.total_chunks == 4000
    assert big.sampled_chunks == 64
    assert big.starts == sample_chunks(40_000 * 100, 1000).starts  # deterministic
    assert big.starts[0] == 0
    # Evenly spread across the axis: consecutive sampled starts keep an even gap.
    gaps = {b - a for a, b in zip(big.starts, big.starts[1:], strict=False)}
    assert len(gaps) <= 2


def test_sampled_decision_matches_exhaustive_when_the_coding_wins(tmp_path) -> None:
    exhaustive_group, _ = _grid(tmp_path / "exhaustive", well_fitted=True)
    sampled_group, _ = _grid(tmp_path / "sampled", well_fitted=True)

    exhaustive, _ = optimise_dense_se_joint(
        exhaustive_group, _preliminary(), measure_max_chunks=10**6
    )
    sampled, _ = optimise_dense_se_joint(sampled_group, _preliminary(), measure_max_chunks=4)

    assert exhaustive.se.is_residual
    assert sampled.se == exhaustive.se


def test_sampled_decision_matches_exhaustive_when_a_bad_column_reverts(
    tmp_path,
) -> None:
    """The GCST007320 case survives sampling (issue #146 AC6).

    One badly fitting Analysis among nineteen clean ones must revert the whole
    plane. Every sampled row chunk carries a cell for every Analysis, so a
    bounded sample still sees the bad column's exception share.
    """
    exhaustive_group, _ = _grid(tmp_path / "exhaustive", well_fitted=True, n_clean=19)
    sampled_group, _ = _grid(tmp_path / "sampled", well_fitted=True, n_clean=19)

    def add_bad_column(group) -> None:
        source = group["se"]
        rng = np.random.default_rng(0)
        ragged = np.exp(rng.normal(-3.0, 2.0, (source.shape[0], 1))).astype(np.float32)
        eaf = group["eaf"][:].astype(np.float32)
        full = np.concatenate([eaf, eaf[:, :1]], axis=1)
        se = np.concatenate([source[:], ragged], axis=1).astype(np.float32)
        group.create_dataset("eaf", data=full, chunks=(1000, 1), dtype="float32", overwrite=True)
        group.create_dataset("se", data=se, chunks=(1000, 1), dtype="float32", overwrite=True)
        group.create_dataset(
            "z", data=np.ones_like(se), chunks=(1000, 1), dtype="float16", overwrite=True
        )

    add_bad_column(exhaustive_group)
    add_bad_column(sampled_group)

    exhaustive, _ = optimise_dense_se_joint(
        exhaustive_group, _preliminary(), measure_max_chunks=10**6
    )
    sampled, _ = optimise_dense_se_joint(sampled_group, _preliminary(), measure_max_chunks=2)

    assert exhaustive.se == SeEncoding("float16")
    assert sampled.se == exhaustive.se


def test_the_record_says_what_the_measurement_saw(tmp_path) -> None:
    group, _ = _grid(tmp_path, well_fitted=True)
    record = SeMeasurementRecord()

    optimise_dense_se_joint(group, _preliminary(), record=record, measure_max_chunks=4)

    assert record.dense is not None
    assert record.dense.sampled_chunks == 4
    assert record.dense.total_chunks == 40
    assert record.to_manifest() == {
        "dense": {"sampled_chunks": 4, "total_chunks": 40},
        "overflow": "none",
    }
