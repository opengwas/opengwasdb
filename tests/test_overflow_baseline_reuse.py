"""The Ragged Overflow's per-variant EAF baseline, derived once (issue #230).

A Hybrid build asks `RaggedCSRWriter` for the same baseline three times over
the same cells -- the joint SE fit's Analysis-aligned batches, that fit's
chunk-aligned measurement batches, and the CSR flush -- and a standalone Ragged
build asks a fourth time, for its own SE measurement. Each ask used to be a
fresh pass over the plane with the two derived-then-copied per-variant buffers
`eaf_baseline_from_sorted_runs` allocates. Holding the answer has to change the
number of passes and nothing else, so every test here compares against the
re-deriving path itself: the baseline, the arrays coded against it and the
chosen plan must all be the ones the per-ask path produces.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from types import MethodType

import numpy as np
from test_overflow_cell_streaming import _model_writer, _residual_se_encoding, _stored

from opengwasdb.encoding import StoreEncoding
from opengwasdb.layouts.ragged import zarr_csr

#: The plane `test_overflow_cell_streaming`'s fixtures use: enough cells per
#: Analysis that a residual EAF plane is worth choosing (a plane with far more
#: variants than cells is correctly `float32`), and no single-cell Analysis, so
#: the SE plane stays residual and the streamed SE write is exercised too.
_SIZES = [1500, 1800, 900, 1900]


def _recompute_each_time(self, encoding: StoreEncoding) -> np.ndarray | None:
    """The writer as it was before #230: derive on every ask, hold nothing."""
    return None if not encoding.eaf.is_residual else self._derive_eaf_baseline()


def _traced_growth(work: Callable[[], object]) -> int:
    """Bytes the process still holds after `work`, by `tracemalloc`."""
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        work()
        return tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()


def test_the_baseline_is_derived_once_and_reused(tmp_path, monkeypatch) -> None:
    """One derivation across every phase that needs it, and the arrays and plan
    of a writer that re-derived on each ask.

    The count is what fails if a phase derives its own: `_eaf_baseline` is the
    single accessor the fit, the measurement, the standalone SE measurement and
    the flush all reach the baseline through, so a second pass over the plane
    shows up here as a second call.
    """
    derivations: list[int] = []
    derive = zarr_csr.eaf_baseline_from_sorted_runs

    def counted(index_runs, value_runs, n_variants):
        derivations.append(1)
        return derive(index_runs, value_runs, n_variants)

    monkeypatch.setattr(zarr_csr, "eaf_baseline_from_sorted_runs", counted)

    writer = _model_writer(_SIZES)
    encoding = _residual_se_encoding(writer, len(_SIZES))
    # Fixture is meaningful: every ask below really does ask for a baseline, and
    # the plane really is coded against one.
    assert encoding.eaf.is_residual, "fixture must select a residual EAF plane"

    held = writer._eaf_baseline(encoding)
    writer.se_fit_inputs(encoding)
    # The fit and the measurement read the frequencies back from the written
    # `eaf` plane, so the write comes first (issue #232).
    writer.write_eaf_plane(tmp_path / "reused", encoding, region_cells=512)
    list(writer.se_fit_batches(cell_budget=1024))
    list(writer.se_fit_chunk_batches(64, cell_budget=64 * 8))
    writer.flush_se(tmp_path / "reused", encoding, region_cells=512)

    assert len(derivations) == 1, f"baseline derived {len(derivations)} times over the same cells"
    assert writer._eaf_baseline(encoding) is held, "every ask must get the held array back"

    # The re-deriving writer is the pre-#230 behaviour on identical input, so
    # agreement with it is what shows reuse changed only the number of passes.
    rederiving = _model_writer(_SIZES)
    monkeypatch.setattr(rederiving, "_eaf_baseline", MethodType(_recompute_each_time, rederiving))
    rederiving_encoding = _residual_se_encoding(rederiving, len(_SIZES))
    rederiving.flush(tmp_path / "rederived", encoding, region_cells=512)

    assert len(derivations) > 1, "the comparison writer must really re-derive"
    assert rederiving_encoding == encoding, "reuse must not move the chosen plan"

    reused, each_time = _stored(tmp_path / "reused"), _stored(tmp_path / "rederived")
    assert sorted(reused) == sorted(each_time)
    for name in reused:
        np.testing.assert_array_equal(reused[name], each_time[name], err_msg=name)
    # The baseline itself is one of those arrays, so the equations every residual
    # in the plane is coded against are compared bit-for-bit above.
    assert "eaf_baseline" in reused


def test_an_analysis_added_after_a_derivation_drops_the_baseline() -> None:
    """A held baseline is a median over the cells that produced it, so a later
    `add_analysis` must not be coded against the older one -- a cell can be the
    only witness at its variant, and a variant with no witness has no baseline
    at all."""
    writer = _model_writer(_SIZES)
    encoding = _residual_se_encoding(writer, len(_SIZES))
    before = writer._eaf_baseline(encoding)
    assert before is writer._eaf_baseline(encoding)

    # A variant the held baseline has no value for, or only one Analysis's
    # value for: either way a second cell at it has to move the median.
    witnessed = np.bincount(
        np.concatenate(writer._variant_indices), minlength=writer._n_variants
    )
    target = int(np.flatnonzero(np.isnan(before) | (witnessed == 1))[0])
    extra = (
        np.array([target], dtype=np.int32),
        np.zeros(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
        np.array([0.99], dtype=np.float32),
    )
    writer.add_analysis(*extra)
    after = writer._eaf_baseline(encoding)

    assert after is not before, "a new cell must invalidate the held baseline"
    assert not (np.isnan(before[target]) and np.isnan(after[target]))
    assert np.isnan(before[target]) or after[target] != before[target]

    # And what it recomputes is what a writer never asked before derives.
    scratch = _model_writer(_SIZES)
    scratch.add_analysis(*extra)
    np.testing.assert_array_equal(scratch._eaf_baseline(encoding), after)


def test_the_held_baseline_is_a_per_variant_array_not_a_per_cell_cache() -> None:
    """What reuse retains is one `float32` per variant, whatever the cell count:
    four times the cells must not retain four times the bytes, which is what a
    cache of anything derived per cell would do."""
    thin, fat = _model_writer([2000] * 2), _model_writer([2000] * 8)
    assert fat.n_associations == 4 * thin.n_associations
    assert thin._n_variants == fat._n_variants
    # One plan for both, from a third writer: neither plane under test may have
    # been asked for a baseline before its own growth is measured.
    encoding = _residual_se_encoding(_model_writer(_SIZES), len(_SIZES))
    assert encoding.eaf.is_residual

    thin_bytes = _traced_growth(lambda: thin._eaf_baseline(encoding))
    fat_bytes = _traced_growth(lambda: fat._eaf_baseline(encoding))

    # What reuse retains is the `n_variants` float32s of the baseline (8 KB for
    # these 2,000 variants) plus a little of the derivation's own scratch,
    # whatever the cell count. A cache of anything derived per cell would
    # retain 4 bytes a cell -- 64 KB on the fat plane, four times this bound.
    bound = 2 * 4 * thin._n_variants
    assert 0 < thin_bytes <= bound, f"held {thin_bytes} bytes"
    assert fat_bytes <= bound, f"held {fat_bytes} bytes over 4x the cells"
