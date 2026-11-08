"""`RaggedCSRWriter` writes a completed association sequence (#424).

Ragged completion wrote its CSR with its own writer, over flat arrays of every
completed cell, so a genome-wide completion had to hold the whole Store in
memory. It now writes through the builders' writer, region by region, which
needed four things only completion has: the imputed plane, the panel's
frequency for imputed cells (`eaf_reference`, ADR 0037 §4), observed cells
carried exactly under the source's SE model (#301), and each Analysis's share
of SEs held as exact exceptions (spec §6a). A writer made with a
`ReferenceCompletion` does all of them; one made without, as a build makes it,
none.

What completion stored before is what `AssociationCodec.encode` makes of the
whole batch at once (`write_encoded_planes`); the writer's regions must store
the same arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from opengwasdb.admission import NotAdmitted
from opengwasdb.encoding import (
    SE_EXCEPTION,
    SE_MISSING,
    EafEncoding,
    SeEncoding,
    StoreEncoding,
    ZEncoding,
    write_eaf_reference,
)
from opengwasdb.encoding.association_codec import AssociationCodec, CellValues, Placement
from opengwasdb.encoding.planes import write_encoded_planes
from opengwasdb.layouts.ragged.zarr_csr import (
    RAGGED_ZARR_PATH,
    RaggedCSRWriter,
    ReferenceCompletion,
)
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store.arrays import ArrayRole

PLAN = StoreEncoding(
    z=ZEncoding(kind="int16_fixed", scale=1024),
    se=SeEncoding(kind="int8_residual", residual_range=0.5),
    eaf=EafEncoding(kind="int8_residual", residual_range=0.5),
).with_eaf_reference(True)
N_VARIANTS = 40
COEFFICIENTS = np.array([[-3.0, -0.5], [-2.5, -0.4], [-2.0, -0.6]], dtype=np.float32)


def _cells() -> tuple[np.ndarray, list[np.ndarray], CellValues]:
    """Three Analyses of observed, imputed and missing cells, as completion holds them.

    Observed SEs sit on the source's code grid, as a source decodes them, except
    one half a step off it, which only the carry rule stores exactly. Imputed
    cells have no frequency of their own; one of Analysis 1's sits far below the
    model, an exact exception. Analysis 2 holds nothing.
    """
    rng = np.random.default_rng(424)
    counts = [14, 9, 0]
    vi = np.concatenate([np.sort(rng.choice(N_VARIANTS, n, replace=False)) for n in counts])
    ai = np.repeat(np.arange(3), counts)
    imputed = np.zeros(len(vi), dtype=np.uint8)
    imputed[[3, 4, 5, 15, 16]] = 1
    missing = np.zeros(len(vi), dtype=bool)
    missing[[6, 17]] = True
    z = rng.normal(0, 3, len(vi)).astype(np.float32)
    eaf = rng.uniform(0.1, 0.9, len(vi)).astype(np.float32)
    on_model = AssociationCodec(
        PLAN, eaf_baseline=_baseline(), eaf_reference=_reference(),
        se_coefficients=COEFFICIENTS,
    ).round_trip(
        CellValues(z, np.full(len(vi), 0.05, np.float32), eaf, imputed=np.zeros_like(imputed)),
        Placement.sequence(0, vi, ai, 3),
        where="the fixture",
    )
    se = np.asarray(on_model.se, dtype=np.float32).copy()
    se[0] *= np.float32(np.exp(0.5 * PLAN.se.step))  # observed, off the grid
    se[15] = np.float32(1e-4)  # imputed, beyond the residual range
    eaf[imputed == 1] = np.nan
    z[missing], se[missing], eaf[missing] = np.nan, np.nan, np.nan
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return offsets, [vi, ai], CellValues(z, se, eaf, imputed=imputed)


def _baseline() -> np.ndarray:
    return np.linspace(0.2, 0.8, N_VARIANTS, dtype=np.float32)


def _reference() -> np.ndarray:
    return np.linspace(0.3, 0.7, N_VARIANTS, dtype=np.float32)


def _write_whole(store, offsets, vi, ai, cells) -> np.ndarray:
    """What completion wrote before #424: the batch encoded at once, then stored."""
    root = store_arrays.open_group_for_write(store / RAGGED_ZARR_PATH, "w")
    for name, data, dtype in (
        ("offsets", offsets, np.int64),
        ("variant_index", vi.astype(np.int32), np.int32),
        ("imputed", cells.imputed, np.uint8),
    ):
        role = (
            ArrayRole.ASSOCIATION_OFFSETS if name == "offsets" else ArrayRole.ASSOCIATION_SEQUENCE
        )
        store_arrays.create_array(root, name, role, data=data, dtype=dtype,
                                  compressor=store_arrays.compressor())
    encoded = AssociationCodec(
        PLAN, eaf_baseline=_baseline(), eaf_reference=_reference(),
        se_coefficients=COEFFICIENTS,
    ).encode(cells, Placement.sequence(0, vi, ai, 3), where="whole", carried=cells.imputed == 0)
    write_encoded_planes(
        root, PLAN, encoded, role=ArrayRole.ASSOCIATION_SEQUENCE,
        compressor=store_arrays.compressor(), eaf_baseline=_baseline(),
        se_coefficients=COEFFICIENTS,
    )
    write_eaf_reference(root, _reference(), compressor=store_arrays.compressor(),
                        role=ArrayRole.RAGGED_PER_VARIANT)
    return np.asarray(encoded.se)


def _write_streamed(store, offsets, vi, cells):
    writer = RaggedCSRWriter(
        N_VARIANTS, ReferenceCompletion(("a", "b", "c"), _reference(), where="the test")
    )
    for a in range(3):
        at = slice(int(offsets[a]), int(offsets[a + 1]))
        writer.add_analysis(
            vi[at], cells.z[at], cells.se[at], eaf=cells.eaf[at], imputed=cells.imputed[at]
        )
    return writer.flush(store, PLAN, eaf_baseline=_baseline(), se_coefficients=COEFFICIENTS)


def _arrays(store) -> dict[str, tuple]:
    root = store_arrays.open_group(store / RAGGED_ZARR_PATH)
    return {
        name: (str(a.dtype), a.shape, a.chunks, a.shards, np.asarray(a[...]).tobytes())
        for name, a in root.arrays()
    }


def test_a_completed_sequence_is_stored_as_the_whole_batch_encodes_it(tmp_path):
    offsets, (vi, ai), cells = _cells()
    whole_se = _write_whole(tmp_path / "whole", offsets, vi, ai, cells)
    # Fixture first: each case it is about is in the codes the whole encode made.
    assert whole_se[0] == SE_EXCEPTION, "fixture: the off-grid observed SE is carried exactly"
    assert whole_se[15] == SE_EXCEPTION, "fixture: an imputed SE beyond the range"
    assert (whole_se[1:3] != SE_EXCEPTION).all(), "fixture: on-grid observed SEs keep codes"
    assert whole_se[6] == SE_MISSING

    _write_streamed(tmp_path / "streamed", offsets, vi, cells)

    whole, streamed = _arrays(tmp_path / "whole"), _arrays(tmp_path / "streamed")
    assert {"imputed", "eaf_reference", "se_exception_index", "se_coefficients"} <= whole.keys()
    assert streamed.keys() == whole.keys()
    for name in whole:
        assert streamed[name] == whole[name], name
    attrs = store_arrays.open_group(tmp_path / "streamed" / RAGGED_ZARR_PATH).attrs
    assert attrs["completion_state"] == "reference_completed"


def test_the_writer_counts_each_analysis_s_exact_exceptions(tmp_path):
    """Per Analysis: SEs held, SEs held as exact exceptions, and both among
    imputed cells -- what a completion records against the 2% budget."""
    offsets, (vi, ai), cells = _cells()
    codes = _write_whole(tmp_path / "whole", offsets, vi, ai, cells)
    imputed = cells.imputed == 1

    counts = _write_streamed(tmp_path / "streamed", offsets, vi, cells)

    def per_analysis(mask: np.ndarray) -> list[int]:
        return np.bincount(ai[mask], minlength=3).tolist()

    held, exact = codes != SE_MISSING, codes == SE_EXCEPTION
    assert per_analysis(exact & ~imputed)[0] > 0, "fixture: an observed exception"
    assert per_analysis(exact & imputed)[1] > 0, "fixture: an imputed exception"
    assert counts.n_se.tolist() == per_analysis(held)
    assert counts.n_exceptions.tolist() == per_analysis(exact)
    assert counts.n_imputed_se.tolist() == per_analysis(held & imputed)
    assert counts.n_imputed_exceptions.tolist() == per_analysis(exact & imputed)


def test_an_imputed_flag_on_a_missing_cell_is_refused():
    """Judged with its cells (`storable`): a missing cell is not an imputed one."""
    writer = RaggedCSRWriter(N_VARIANTS)
    with pytest.raises(NotAdmitted, match="imputed"):
        writer.add_analysis(
            np.array([1, 2], dtype=np.int32),
            np.array([1.0, np.nan], dtype=np.float32),
            np.array([0.1, np.nan], dtype=np.float32),
            imputed=np.array([0, 1], dtype=np.uint8),
        )


def test_a_build_given_no_imputed_masks_writes_neither_plane(tmp_path):
    """Builders pass none, and their component is what it was: no `imputed`,
    no `eaf_reference`, and an observed-only completion state."""
    writer = RaggedCSRWriter(N_VARIANTS)
    writer.add_analysis(
        np.array([1, 2], dtype=np.int32),
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([0.1, 0.2], dtype=np.float32),
    )
    writer.flush(tmp_path, StoreEncoding(z=PLAN.z, se=SeEncoding(kind="float16")))

    root = store_arrays.open_group(tmp_path / RAGGED_ZARR_PATH)
    assert "imputed" not in root and "eaf_reference" not in root
    assert root.attrs["completion_state"] == "observed_only"


def test_a_completion_naming_other_analyses_than_it_writes_is_refused(tmp_path):
    """Its out-of-range SEs are logged by Analysis id, so ids that do not name
    the Analyses written would put one Analysis's count under another's name."""
    writer = RaggedCSRWriter(N_VARIANTS, ReferenceCompletion(("a",), None, where="the test"))
    for _ in range(2):
        writer.add_analysis(
            np.array([1], dtype=np.int32), np.array([1.0], np.float32), np.array([0.1], np.float32)
        )
    with pytest.raises(ValueError, match="1 Analysis ids name a sequence of 2 Analyses"):
        writer.flush(tmp_path, StoreEncoding(z=PLAN.z, se=SeEncoding(kind="float16")))
