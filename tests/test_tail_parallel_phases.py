"""The post-Pass-2 tail's row-chunk loops parallelise without changing a byte (#221).

The SE fit, measurement, count and rewrite, and the top-hit gather, all walk
independent zarr row chunks. Run across ``--n-workers`` they must produce the
serial path's encoding, coefficients, rewritten plane, exception table and
Top-Hit Indexes; ``n_workers <= 1`` must stay in-process. Each phase must also
log its start, end and elapsed time, with progress through the loop.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import zarr

from opengwasdb.build import ordered_pool
from opengwasdb.encoding import se as se_module
from opengwasdb.encoding.plan import EafEncoding, SeEncoding, StoreEncoding, ZEncoding
from opengwasdb.encoding.se import OverflowCells, optimise_dense_se_joint
from opengwasdb.encoding.timing import PhaseTimer
from opengwasdb.layouts.dense.top_hits import (
    _gather_in_row_chunks,
    build_top_hit_indexes,
    write_top_hit_indexes_for_store,
)

_PLAIN = StoreEncoding(
    z=ZEncoding("float16"), se=SeEncoding("float16"), eaf=EafEncoding("float32")
)
_PRELIMINARY = StoreEncoding(
    z=ZEncoding("float16"), se=SeEncoding("float16"), eaf=EafEncoding("float32")
)
_ROW_CHUNK = 1000


def _se_group(tmp_path: Path, name: str):
    """A well-fitted six-chunk SE plane with exceptions in two of its chunks.

    Rows 10 and 2500 sit in row chunks 0 and 2; a reduction that consumed the
    chunks out of order would place their side-table rows the other way round.
    """
    group = zarr.open_group(str(tmp_path / name), mode="w")
    n_rows = 6000
    eaf = np.linspace(0.05, 0.95, n_rows, dtype=np.float32)[:, None]
    predictor = np.log(2 * eaf * (1 - eaf))
    se = np.exp(-3.0 - 0.5 * predictor).astype(np.float32)
    for row in (10, 2500):
        se[row, 0] = np.float32(np.exp(-3.0 - 0.5 * predictor[row, 0] + 8.0))
    group.create_dataset("eaf", data=eaf, chunks=(_ROW_CHUNK, 1), dtype="float32")
    group.create_dataset("se", data=se, chunks=(_ROW_CHUNK, 1), dtype="float32")
    group.create_dataset("z", data=np.ones_like(eaf), chunks=(_ROW_CHUNK, 1), dtype="float16")
    return group


def _se_snapshot(group) -> dict[str, np.ndarray]:
    return {
        "se": np.asarray(group["se"][:]),
        "coefficients": np.asarray(group["se_coefficients"][:]),
        "exceptions": np.asarray(group["se_exception_index"][:]),
        "exception_values": np.asarray(group["se_exception_value"][:]),
    }


def _small_store(tmp_path: Path, name: str) -> Path:
    """A two-chunk-per-axis dense store with a handful of strong candidates."""
    store = tmp_path / name
    root = zarr.open_group(str(store / "data.zarr"), mode="w")
    rng = np.random.default_rng(20250924)
    n_rows, n_analyses = 100, 3
    z = rng.normal(0, 1, (n_rows, n_analyses)).astype(np.float16)
    for row, col in ((5, 0), (6, 1), (80, 2), (99, 0)):
        z[row, col] = np.float16(6.0 if col != 1 else -7.0)
    se = np.abs(rng.normal(0.1, 0.01, (n_rows, n_analyses))).astype(np.float16)
    eaf = rng.uniform(0.05, 0.95, (n_rows, n_analyses)).astype(np.float32)
    root.create_dataset("z", data=z, chunks=(10, n_analyses), dtype="float16")
    root.create_dataset("se", data=se, chunks=(10, n_analyses), dtype="float16")
    root.create_dataset("eaf", data=eaf, chunks=(10, n_analyses), dtype="float32")
    return store


def _tier_arrays(store: Path) -> dict[str, np.ndarray]:
    root = zarr.open_group(str(store / "data.zarr"), mode="r")
    out: dict[str, np.ndarray] = {}
    for key in root["top_hits"]:
        for name in root[f"top_hits/{key}"]:
            out[f"{key}/{name}"] = np.asarray(root[f"top_hits/{key}/{name}"][:])
    return out


# ── SE: parallel equals serial, and the reduction respects chunk order ──────


def test_se_parallel_matches_serial_across_several_chunks(tmp_path) -> None:
    serial_group = _se_group(tmp_path, "serial.zarr")
    parallel_group = _se_group(tmp_path, "parallel.zarr")

    serial_choice, serial_coefficients = optimise_dense_se_joint(
        serial_group, _PRELIMINARY, n_workers=1
    )
    parallel_choice, parallel_coefficients = optimise_dense_se_joint(
        parallel_group, _PRELIMINARY, n_workers=3
    )

    # The fixture only means anything if the coding was chosen and there are
    # exceptions in more than one row chunk to order.
    assert serial_choice.se.is_residual, "fixture is meaningful only if coding is chosen"
    assert serial_coefficients is not None
    serial = _se_snapshot(serial_group)
    assert len(serial["exceptions"]) >= 2
    assert len(np.unique(serial["exceptions"] // _ROW_CHUNK)) > 1

    assert parallel_choice == serial_choice
    assert parallel_coefficients is not None
    np.testing.assert_array_equal(parallel_coefficients, serial_coefficients)
    parallel = _se_snapshot(parallel_group)
    for name, values in serial.items():
        np.testing.assert_array_equal(parallel[name], values, err_msg=name)


def test_the_se_reduction_is_sensitive_to_chunk_order(tmp_path, monkeypatch) -> None:
    """If the reduction ignored order, this fixture would be indistinguishable.

    The companion parallel-equals-serial test therefore fails for any
    reduction that consumes chunks as they complete.
    """
    ordered = _se_group(tmp_path, "ordered.zarr")
    optimise_dense_se_joint(ordered, _PRELIMINARY, n_workers=1)
    expected = _se_snapshot(ordered)

    real_map = ordered_pool.ordered_map

    def reversed_map(fn, items, n_workers, max_in_flight=None):
        return iter(reversed(list(real_map(fn, items, n_workers, max_in_flight))))

    monkeypatch.setattr(se_module, "ordered_map", reversed_map)
    reversed_group = _se_group(tmp_path, "reversed.zarr")
    optimise_dense_se_joint(reversed_group, _PRELIMINARY, n_workers=1)
    got = _se_snapshot(reversed_group)

    assert not np.array_equal(got["exceptions"], expected["exceptions"])


def test_joint_overflow_measurement_matches_across_workers(tmp_path, caplog) -> None:
    """The Hybrid overflow's flat chunks parallelise with the same decision."""
    dense_chunks = (1000, 2)
    serial_group = _two_analysis_group(tmp_path, "dense-serial.zarr", dense_chunks)
    parallel_group = _two_analysis_group(tmp_path, "dense-parallel.zarr", dense_chunks)
    overflow = _overflow_cells()

    with caplog.at_level(logging.INFO, logger="opengwasdb.encoding.se"):
        serial_choice, serial_coefficients = optimise_dense_se_joint(
            serial_group, _PRELIMINARY, overflow=overflow, overflow_chunk=97, n_workers=1
        )
        parallel_choice, parallel_coefficients = optimise_dense_se_joint(
            parallel_group, _PRELIMINARY, overflow=overflow, overflow_chunk=97, n_workers=3
        )

    assert serial_choice.se.is_residual
    assert parallel_choice == serial_choice
    assert parallel_coefficients is not None and serial_coefficients is not None
    np.testing.assert_array_equal(parallel_coefficients, serial_coefficients)
    serial = _se_snapshot(serial_group)
    parallel = _se_snapshot(parallel_group)
    for name, values in serial.items():
        np.testing.assert_array_equal(parallel[name], values, err_msg=name)

    # The dense progress is not the whole phase: the overflow fold is its own
    # logged step, so 100% of the dense loop is not reported as "SE fit done".
    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("SE fit (dense):") for message in messages)
    assert any(message.startswith("SE fit (overflow): start") for message in messages)
    assert any(message.startswith("SE fit (overflow): done in") for message in messages)


def _two_analysis_group(tmp_path: Path, name: str, chunks: tuple[int, int]):
    """A well-fitted two-Analysis plane, so a shared model is chosen."""
    group = zarr.open_group(str(tmp_path / name), mode="w")
    n_rows = 6000
    eaf = np.linspace(0.05, 0.95, n_rows, dtype=np.float32)[:, None]
    eaf = np.repeat(eaf, 2, axis=1)
    predictor = np.log(2 * eaf * (1 - eaf))
    se = np.exp(-3.0 - 0.5 * predictor).astype(np.float32)
    se[10, 0] = np.float32(np.exp(-3.0 - 0.5 * predictor[10, 0] + 8.0))
    se[2500, 1] = np.float32(np.exp(-3.0 - 0.5 * predictor[2500, 1] + 8.0))
    group.create_dataset("eaf", data=eaf, chunks=chunks, dtype="float32")
    group.create_dataset("se", data=se, chunks=chunks, dtype="float32")
    group.create_dataset("z", data=np.ones_like(eaf), chunks=chunks, dtype="float16")
    return group


def _overflow_cells() -> OverflowCells:
    """Several hundred off-axis cells spread over 2 Analyses, 97 per chunk."""
    rng = np.random.default_rng(7)
    n = 700
    eaf = rng.uniform(0.05, 0.95, n).astype(np.float32)
    se = np.exp(-3.0 - 0.5 * np.log(2 * eaf * (1 - eaf))).astype(np.float32)
    return OverflowCells(
        se_values=se,
        eaf_values=eaf,
        analysis_indices=(np.arange(n) % 2).astype(np.int64),
        n_analyses=2,
    )


def test_n_workers_one_keeps_the_serial_path(tmp_path, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("no pool may be created at n_workers=1")

    monkeypatch.setattr(ordered_pool, "ProcessPoolExecutor", boom)
    group = _se_group(tmp_path, "in-process.zarr")
    selected, _ = optimise_dense_se_joint(group, _PRELIMINARY, n_workers=1)
    assert selected.se.is_residual


def test_se_phases_log_start_end_and_progress(tmp_path, caplog) -> None:
    group = _se_group(tmp_path, "logged.zarr")
    with caplog.at_level(logging.INFO, logger="opengwasdb.encoding.se"):
        optimise_dense_se_joint(group, _PRELIMINARY, n_workers=2)

    messages = [record.getMessage() for record in caplog.records]
    for label in ("SE fit", "SE measurement", "SE rewrite", "SE rewrite count"):
        assert any(message.startswith(f"{label}: start") for message in messages), label
        assert any(message.startswith(f"{label}: done in") for message in messages), label
    # The dense loop's progress says so; it is not labelled as the whole fit.
    assert any(message.startswith("SE fit (dense):") for message in messages)
    assert any("elapsed" in message and "ETA" in message for message in messages)
    assert any("SE phase timings" in message for message in messages)


def _bad_fit_group(tmp_path: Path, name: str, chunks: tuple[int, int] = (200, 1)):
    """A Dense scratch plane whose SE defies the MAF model, so the coding loses."""
    group = zarr.open_group(str(tmp_path / name), mode="w")
    n_rows = 600
    eaf = np.linspace(0.05, 0.95, n_rows, dtype=np.float32)[:, None]
    predictor = np.log(2 * eaf * (1 - eaf))
    se = np.exp(-3.0 - 0.5 * predictor + 6.0 * np.sin(np.arange(n_rows)[:, None])).astype(
        np.float32
    )
    group.create_dataset("eaf", data=eaf, chunks=chunks, dtype="float32")
    group.create_dataset("se", data=se, chunks=chunks, dtype="float32")
    group.create_dataset("z", data=np.ones_like(eaf), chunks=chunks, dtype="float16")
    return group, se


def test_the_float16_fallback_is_logged_and_timed(tmp_path, caplog) -> None:
    """The path a real build selected must log its start, end and progress."""
    group, source_se = _bad_fit_group(tmp_path, "fallback.zarr")
    timer = PhaseTimer()
    with caplog.at_level(logging.INFO, logger="opengwasdb.encoding.se"):
        selected, coefficients = optimise_dense_se_joint(group, _PRELIMINARY, timer=timer)

    assert not selected.se.is_residual, "fixture is meaningful only if the coding is declined"
    assert coefficients is None
    assert group["se"].dtype == np.dtype("float16")
    np.testing.assert_array_equal(
        np.asarray(group["se"][:], dtype=np.float32),
        np.asarray(source_se, dtype=np.float16).astype(np.float32),
    )
    assert "rewrite.narrow" in timer.seconds
    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("SE float16 narrowing: start") for message in messages)
    assert any(message.startswith("SE float16 narrowing: done in") for message in messages)
    assert any(
        message.startswith("SE float16 narrowing:") and "elapsed" in message
        for message in messages
    )
    assert any("SE phase timings" in message for message in messages)


# ── Top-hit index: gather and index identical across workers ────────────────


def test_gather_in_row_chunks_matches_across_workers(tmp_path) -> None:
    store = _small_store(tmp_path, "gather")
    root = zarr.open_group(str(store / "data.zarr"), mode="r")
    rows = np.array([3, 5, 12, 47, 48, 80, 99, 0], dtype=np.int64)
    cols = np.array([1, 0, 2, 1, 2, 0, 1, 2], dtype=np.int64)

    def band(r0: int, r1: int) -> np.ndarray:
        return np.arange(r0, r1, dtype=np.float32)[:, None] * 100 + np.arange(3)[None, :]

    assert len(np.unique(rows // 10)) > 1, "fixture is meaningful only if rows span chunks"
    serial = _gather_in_row_chunks(root, rows, cols, band, n_workers=1)
    parallel = _gather_in_row_chunks(root, rows, cols, band, n_workers=3)
    np.testing.assert_array_equal(parallel, serial)


def test_rebuilt_top_hit_index_matches_across_workers(tmp_path) -> None:
    serial_store = _small_store(tmp_path, "serial-hits")
    parallel_store = _small_store(tmp_path, "parallel-hits")
    thresholds = (1e-4,)

    build_top_hit_indexes(serial_store, thresholds=thresholds, encoding=_PLAIN, n_workers=1)
    build_top_hit_indexes(parallel_store, thresholds=thresholds, encoding=_PLAIN, n_workers=3)

    serial = _tier_arrays(serial_store)
    parallel = _tier_arrays(parallel_store)
    assert serial, "fixture is meaningful only if the index has arrays"
    assert set(parallel) == set(serial)
    for name, values in serial.items():
        np.testing.assert_array_equal(parallel[name], values, err_msg=name)


def test_inline_top_hit_write_matches_across_workers(tmp_path) -> None:
    serial_store = _small_store(tmp_path, "serial-inline")
    parallel_store = _small_store(tmp_path, "parallel-inline")
    rows = np.array([0, 5, 6, 47, 48, 80, 99], dtype=np.int64)
    cols = np.array([2, 0, 1, 1, 2, 2, 0], dtype=np.int64)
    z = np.linspace(4.0, 7.0, len(rows), dtype=np.float32)
    se = np.full(len(rows), 0.1, dtype=np.float32)

    write_top_hit_indexes_for_store(
        serial_store, rows, cols, z, se, _PLAIN, n_workers=1
    )
    write_top_hit_indexes_for_store(
        parallel_store, rows, cols, z, se, _PLAIN, n_workers=3
    )

    serial = _tier_arrays(serial_store)
    parallel = _tier_arrays(parallel_store)
    assert serial, "fixture is meaningful only if the index has arrays"
    for name, values in serial.items():
        np.testing.assert_array_equal(parallel[name], values, err_msg=name)


def test_top_hit_phases_log_start_end_and_progress(tmp_path, caplog) -> None:
    store = _small_store(tmp_path, "logged-hits")
    with caplog.at_level(logging.INFO, logger="opengwasdb.layouts.dense.top_hits"):
        build_top_hit_indexes(store, thresholds=(1e-4,), encoding=_PLAIN, n_workers=2)

    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("Top-hit scan: start") for message in messages)
    assert any(message.startswith("Top-hit scan: done in") for message in messages)
    assert any("elapsed" in message and "ETA" in message for message in messages)
    assert any("Top-hit phase timings" in message for message in messages)
