"""The shared genotype arithmetic against the frozen builder and independent Test Store
(opengwas/ogld#90)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import build_fixtures as made
import genotypes_base as base
import numpy as np
import pytest
import store_copies
import testdata

from opengwasdb.ld.format import REncoding


@pytest.mark.parametrize("k", [0, 10])
@pytest.mark.parametrize("scale", [511, 32767])
def test_codes_equal_the_base_builder_for_every_near_band_pair(k: int, scale: int) -> None:
    from opengwasdb.ld.genotypes import covariates, prepare, r

    v = testdata.variants()
    rows = np.flatnonzero(v.passes_filters)
    calls = made.hard_calls()[rows].astype(np.float64)
    is_alt = made.a1_signs()[rows] > 0
    names = list(testdata.pcs().names[:k])
    values = testdata.pcs().values[:, :k]
    old = base.unit_vectors(calls, base.covariates(names, values, calls.shape[1]), is_alt)
    expected = old @ old.T
    block = prepare(calls, covariates(names, values, calls.shape[1]), is_alt)
    actual = r(block, block)
    both = np.triu(np.ones(actual.shape, dtype=bool), 1)
    np.testing.assert_allclose(actual[both], expected[both], rtol=0, atol=base.R_TOLERANCE)
    near = (v.chrom[rows, None] == v.chrom[rows][None, :]) & (
        np.abs(v.pos[rows, None] - v.pos[rows][None, :]) <= 3_000_000
    )
    near &= np.triu(np.ones(near.shape, dtype=bool), 1)
    assert near.sum() > 100_000, "compare all Near Band pairs, not an empty slice"
    enc = REncoding(scale)
    np.testing.assert_array_equal(enc.quantise(actual[near]), enc.quantise(expected[near]))


def test_codes_equal_the_independently_built_test_store_for_every_stored_pair() -> None:
    from opengwasdb.ld.genotypes import covariates, prepare, r
    from opengwasdb.ld.store import StoreReader

    rows = np.flatnonzero(testdata.variants().passes_filters)
    calls = made.hard_calls()[rows].astype(np.float64)
    cov = covariates(list(testdata.pcs().names[:10]), testdata.pcs().values[:, :10], calls.shape[1])
    block = prepare(calls, cov, made.a1_signs()[rows] > 0)
    reader = StoreReader.open(store_copies.TEST_STORE)
    try:
        wanted = reader.rows_of(made.alids(rows))
        assert (wanted >= 0).all()
        stored = reader.pair_r(wanted)
        held = np.isfinite(stored) & np.triu(np.ones(stored.shape, dtype=bool), 1)
        assert held.sum() > 100_000
        enc = REncoding(store_copies.R_SCALE)
        np.testing.assert_array_equal(
            enc.quantise(r(block, block)[held]), enc.quantise(stored[held])
        )
    finally:
        reader.close()


@pytest.mark.parametrize("k", [0, 10])
@pytest.mark.parametrize("scale", [511, 32767])
def test_codes_are_identical_in_two_block_sizes_and_one_row_at_a_time(k: int, scale: int) -> None:
    """One BLAS thread: the whole matrix in two block sizes, and a row at a time, agree.

    Multi-threaded BLAS is held in the realdata tier
    (`tests/ld/realdata/test_genotype_threads.py`): a small CI runner runs many xdist workers at
    once, and eight or even four OpenBLAS threads in each oversubscribe its CPUs and spin until
    the test times out.
    """
    from genotype_codes import assert_codes_identical

    assert_codes_identical((1,), k, scale)


def test_pgen_int8_and_float64_paths_read_the_same_test_panel_calls() -> None:
    from opengwasdb.ld.build.passes import plan
    from opengwasdb.ld.genotypes import Genotypes

    p = testdata.panel()
    fileset = plan((p.pgen,), p.keep, None, made.PANEL_ID).filesets[0]
    fileset = replace(fileset, order=np.arange(p.n_samples - 1, -1, -1, dtype=np.int64))
    rows = np.array([31, 0, 12, 31], dtype=np.int64)
    with Genotypes(fileset) as reader:
        calls = reader.range_int8(0, reader.variant_count)
        assert calls.dtype == np.int8
        np.testing.assert_array_equal(calls, reader.range(0, reader.variant_count))
        np.testing.assert_array_equal(calls, made.hard_calls()[:, ::-1])
        np.testing.assert_array_equal(reader.rows_int8(rows), reader.rows(rows))
        np.testing.assert_array_equal(reader.rows_int8(rows), calls[rows])
        assert reader.rows_int8(np.array([], dtype=np.int64)).shape == (0, p.n_samples)
        assert reader.range_int8(0, 0).shape == (0, p.n_samples)


def test_build_and_query_have_no_pgen_calls_or_pc_adjustment_implementation() -> None:
    import ast
    import re

    root = Path(__file__).resolve().parents[2] / "opengwasdb" / "ld"
    forbidden = re.compile(
        r"\b(?:import pgenlib|PgenReader|read_dosages_\w+|linalg\.qr|"
        r"def residuals|unit_vectors|residual_fit)\b"
    )
    found = []
    for package in ("build", "query"):
        for path in (root / package).rglob("*.py"):
            source = path.read_text()
            found.extend(
                (str(path.relative_to(root)), number, line)
                for number, line in enumerate(source.splitlines(), 1)
                if forbidden.search(line)
            )
            for node in ast.walk(ast.parse(source)):
                if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult)):
                    continue
                names = [part.id for part in ast.walk(node) if isinstance(part, ast.Name)]
                attributes = [
                    part.attr for part in ast.walk(node) if isinstance(part, ast.Attribute)
                ]
                if any(re.search(r"unit|resid", name) for name in names + attributes):
                    found.append((str(path.relative_to(root)), node.lineno, ast.unparse(node)))
    assert not found, f"genotypes owns pgen reads and PC-adjusted r, not build/query: {found}"


def test_a_missing_call_is_not_hidden_by_either_read_path(tmp_path: Path) -> None:
    import opengwasdb.ld
    from opengwasdb.ld.build.passes import plan
    from opengwasdb.ld.format import InputError
    from opengwasdb.ld.genotypes import Genotypes

    calls = made.hard_calls()[:2].copy()
    calls[1, 3] = -9
    pgen = made.write_fileset(tmp_path / "missing", np.arange(2), genotypes=calls)
    p = testdata.panel()
    fileset = plan((pgen,), p.keep, None, made.PANEL_ID).filesets[0]
    with Genotypes(fileset) as reader:
        assert reader.range_int8(0, 2)[1, 3] == -9
        assert reader.rows_int8(np.array([1], dtype=np.int64))[0, 3] == -9
        assert reader.range(0, 2)[1, 3] == -9
    with pytest.raises(
        InputError, match=rf"panel {made.PANEL_ID}.*1 missing.*{testdata.variants().id[1]}"
    ):
        opengwasdb.ld.build(made.inputs(pgen), tmp_path / "store")


def test_rows_into_refuses_a_buffer_of_another_size_or_layout() -> None:
    """The caller's buffer must be one row per variant and as many kept samples, contiguous."""
    from opengwasdb.ld.build.passes import plan
    from opengwasdb.ld.genotypes import Genotypes

    p = testdata.panel()
    fileset = plan((p.pgen,), p.keep, None, made.PANEL_ID).filesets[0]
    with Genotypes(fileset) as reader:
        rows = np.array([0, 1], dtype=np.int64)
        with pytest.raises(ValueError, match="contiguous"):
            reader.rows_into(rows, np.empty((1, len(fileset.subset)), dtype=np.float64))
        strided = np.empty((2, 2 * len(fileset.subset)), dtype=np.float64)[:, ::2]
        with pytest.raises(ValueError, match="contiguous"):
            reader.rows_into(rows, strided)
