"""The Dense SE rewrite sizes its exception table itself (issue #145).

The rewrite pre-allocated its `se_exception_index`/`se_exception_value` arrays
at exactly the size the *exhaustive* measurement counted, which tied the table
to a pass that #146 will stop making exhaustive. The rewrite now derives its
own count from a codes-only pass (no compression), and still fails loudly if
the streamed encode disagrees with the allocation.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from opengwasdb.encoding import se as se_module
from opengwasdb.encoding.plan import EafEncoding, SeEncoding, StoreEncoding, ZEncoding
from opengwasdb.encoding.planes import DenseSePlane
from opengwasdb.encoding.se import optimise_dense_se_joint


def _preliminary() -> StoreEncoding:
    return StoreEncoding(
        z=ZEncoding("float16"),
        se=SeEncoding("float16"),
        eaf=EafEncoding("float32"),
    )


def _dense_group_with_exceptions(tmp_path):
    """A well-fitted plane plus ten cells whose residual exceeds every range.

    The ten inflated cells are exceptions at any candidate range; every other
    cell has a near-zero residual, so the fixture can only pass by storing
    exactly those ten in the side table. 6000 rows keep the per-Analysis
    coefficient cost amortised so the coding clears the byte gate against
    `float16`.
    """
    group = zarr.open_group(str(tmp_path / "dense.zarr"), mode="w", zarr_format=2)
    eaf = np.linspace(0.05, 0.95, 6000, dtype=np.float32)[:, None]
    predictor = np.log(2 * eaf * (1 - eaf))
    se = np.exp(-3.0 - 0.5 * predictor).astype(np.float32)
    se[2500:2510, 0] = np.exp(-3.0 - 0.5 * predictor[2500:2510, 0] + np.float64(8.0)).astype(
        np.float32
    )
    group.create_array("eaf", data=np.asarray(eaf, dtype="float32"), chunks=(1000, 1))
    group.create_array("se", data=np.asarray(se, dtype="float32"), chunks=(1000, 1))
    group.create_array("z", data=np.asarray(np.ones_like(eaf), dtype="float16"), chunks=(1000, 1))
    return group, se


_EXCEPTION_SLICE = slice(2500, 2510)


def test_rewrite_sizes_its_table_from_its_own_count(tmp_path) -> None:
    group, source_se = _dense_group_with_exceptions(tmp_path)

    selected, _ = optimise_dense_se_joint(group, _preliminary())

    assert selected.se.is_residual, "fixture is meaningful only if the coding is chosen"
    assert group["se_exception_index"].shape[0] == 10
    decoded = DenseSePlane.open(group, selected).band(0, int(group["se"].shape[0]))
    np.testing.assert_array_equal(decoded[_EXCEPTION_SLICE, 0], source_se[_EXCEPTION_SLICE, 0])


def test_rewrite_ignores_a_measurement_that_saw_no_exceptions(tmp_path, monkeypatch) -> None:
    """AC1: the table no longer rides on the measurement.

    A measurement that under-counted exceptions -- the exact failure a sampled
    measurement could introduce later -- must not leave the rewrite with a
    table too small to hold them. Simulate it by zeroing the measured counts.
    """
    group, _ = _dense_group_with_exceptions(tmp_path)
    real_measure = se_module._measure_dense

    def blind_measure(source, eaf_plane, coefficients, timer, n_workers):
        cost = real_measure(source, eaf_plane, coefficients, timer, n_workers)
        return cost._replace(
            exception_counts={c: np.zeros_like(v) for c, v in cost.exception_counts.items()}
        )

    monkeypatch.setattr(se_module, "_measure_dense", blind_measure)

    selected, _ = optimise_dense_se_joint(group, _preliminary())

    assert selected.se.is_residual
    assert group["se_exception_index"].shape[0] == 10


def test_the_rewrites_exception_tables_refuse_arrays_that_do_not_pair_up() -> None:
    """Three indices with one value wrote that value into all three exception
    cells: a silent broadcast into a stored side table (issue #392)."""
    index = np.zeros(3, dtype=np.int64)
    value = np.zeros(3, dtype=np.float32)
    sink = se_module._RewriteSink(
        pending=None,
        exception_index=index,
        exception_value=value,
        exception_count=3,
        row_chunk=1,
        n_rows=3,
        n_chunks=1,
        chunk_timer=None,
    )
    with pytest.raises(
        ValueError,
        match=r"the SE rewrite's exception tables: se_exception_index of length 3 "
        r"and se_exception_value of length 1 must be parallel",
    ):
        se_module._write_exception_tables(
            sink, [np.arange(3, dtype=np.int64)], [np.zeros(1, dtype=np.float32)], 3
        )
    assert index.tolist() == [0, 0, 0]
    assert value.tolist() == [0.0, 0.0, 0.0]
