"""Completion never re-quantises an observed EAF (issue #390, extending #301's SE rule).

A residual EAF plane holds a frequency as a code unless its residual is out of
range. A source cell held as an exact exception whose float32 value, coded against
the completed release's baseline, lands inside the range would, re-encoded
without the carry, become a code that decodes half a step away. Completion
decodes every observed cell from its source, so it must stay an exception.
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from shutil import copytree

import numpy as np
import zarr
from stored_rows import stored_rows
from test_hybrid_completion import (
    _residual_hybrid_crossover_source,
    _residual_ld_panel_with_crossover,
)
from test_se_residual_encoding import _residual_source_and_panel

from opengwasdb import StoreManifest, open_store
from opengwasdb.completion.ld_panel import PanelSource
from opengwasdb.completion.options import CompletionOptions
from opengwasdb.encoding import EAF_EXCEPTION, SE_EXCEPTION, SE_MISSING
from opengwasdb.encoding.codec import (
    EafExceptionBuilder,
    StoreCodec,
    expit,
    logit,
    se_from_log,
    se_log_prediction,
)
from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.layouts.hybrid.complete import complete_hybrid_store
from opengwasdb.layouts.ragged.complete import complete_ragged_store
from opengwasdb.query import query_store
from opengwasdb.store.arrays import ArrayRole, compressor, create_array
from opengwasdb.store_reader.encoding.plan import (
    EafEncoding,
    SeEncoding,
    StoreEncoding,
    ZEncoding,
)
from opengwasdb.store_reader.encoding.planes import DenseEafPlane, DenseSePlane
from opengwasdb.variants.axis import VariantAxis

_RESIDUAL = StoreEncoding(
    z=ZEncoding("int16_fixed", scale=1024),
    se=SeEncoding("int8_residual", residual_range=0.5),
    eaf=EafEncoding("int8_residual", residual_range=0.5),
)
_OFF_GRID_STEPS = 20.5  # half a code step: the most a code cannot express
_PINNED = 20  # Analysis 0's row: observed, with a code, on the panel


def _put(group, name: str, data: np.ndarray) -> None:
    """Replace one exception-table array of a store the test is patching."""
    create_array(
        group, name, ArrayRole.EXCEPTION_TABLE, data=data, compressor=compressor(), overwrite=True
    )


def _alid_of(store: Path, variant_index: int) -> str:
    with closing(VariantAxis(store)) as axis:
        return next(v.alid for v in axis.all() if v.variant_index == variant_index)


def _index_of(store: Path, alid: str) -> int:
    with closing(VariantAxis(store)) as axis:
        return next(v.variant_index for v in axis.all() if v.alid == alid)


def _off_grid(codec: StoreCodec, baseline: float) -> np.float32:
    step = codec.encoding.eaf.step
    return np.float32(expit(np.float64(logit(np.float64(baseline))) + _OFF_GRID_STEPS * step))


def test_encode_eaf_exact_holds_a_value_no_code_reaches() -> None:
    """Unit seam: a carried cell off the code grid is an exception; the same
    cell uncarried, and a carried cell on the grid, keep their codes."""
    codec = StoreCodec(_RESIDUAL)
    base = np.array([0.3, 0.3], np.float32)
    on_grid = codec.decode_eaf(np.array([4], np.int8), baseline=base[:1])[0]  # a code's own value
    off = _off_grid(codec, 0.3)
    values = np.array([off, on_grid], np.float32)
    exact = np.array([True, True])
    pos = np.arange(2)
    assert codec.encode_eaf(values, baseline=base, positions=pos, exceptions=EafExceptionBuilder())[
        0
    ] != (EAF_EXCEPTION), "fixture: in range, so only the carry can hold this cell"
    builder = EafExceptionBuilder()
    codes = codec.encode_eaf(values, baseline=base, positions=pos, exceptions=builder, exact=exact)
    assert codes[0] == EAF_EXCEPTION
    assert codes[1] == 4, "a carried cell already on a code keeps it"
    assert len(builder.table().index) == 1


def test_ragged_completion_keeps_an_observed_eaf_exception(
    residual_ragged_scenario, tmp_path: Path
) -> None:
    scenario = residual_ragged_scenario
    source = Path(copytree(scenario.source, tmp_path / "source.opengwasdb"))
    group = open_store(source).arrays(mode="a")["ragged"]
    encoding = StoreManifest.load(source).encoding
    codec = StoreCodec(encoding)
    offsets = group["offsets"][:]
    at = int(offsets[0]) + _PINNED
    variant = int(group["variant_index"][at])
    baseline = float(group["eaf_baseline"][variant])
    value = _off_grid(codec, baseline)
    assert group["eaf"][at] != EAF_EXCEPTION, "fixture: the cell starts coded"
    assert len(group["eaf_exception_index"][:]) == 0

    group["eaf"][at] = EAF_EXCEPTION
    for name, data in (
        ("eaf_exception_index", np.array([at], dtype=group["eaf_exception_index"].dtype)),
        ("eaf_exception_value", np.array([value], np.float32)),
    ):
        _put(group, name, data)
    alid = _alid_of(source, variant)
    before = stored_rows(source, 0)
    k = int(np.flatnonzero(before.variant_index == variant)[0])
    assert before.eaf[k] == value, "fixture: the source decodes to the pinned value"

    out = tmp_path / "comp.opengwasdb"
    complete_ragged_store(
        source,
        out,
        PanelSource(scenario.panel, ancestry="EUR"),
        CompletionOptions(min_cor=0.0, release_id="comp"),
        cis_window_bp=1_000_000,
    )
    after = stored_rows(out, 0)
    j = int(np.flatnonzero(after.variant_index == _index_of(out, alid))[0])
    assert after.eaf[j] == value, f"completion moved an observed EAF: {after.eaf[j]!r} != {value!r}"


def test_an_ordinary_cell_keeps_its_code(residual_ragged_scenario) -> None:
    """No needless exceptions: the unpatched source and its completion hold the
    same number of EAF exceptions (no imputed cell here is out of range)."""
    scenario = residual_ragged_scenario
    src = open_store(scenario.source).arrays(mode="r")["ragged"]
    done = open_store(scenario.completed).arrays(mode="r")["ragged"]
    assert len(src["eaf_exception_index"][:]) == 0, "fixture: the source holds no EAF exception"
    assert len(done["eaf_exception_index"][:]) == 0


def _pin_dense_cell(source: Path, row: int) -> tuple[np.float32, np.float32]:
    """Re-code Dense `(row, 0)` as an EAF exception and an SE exception, each a
    half code step off the grid and inside the range; returns the two values."""
    root = zarr.open_group(str(source / "data.zarr"), mode="r+", zarr_format=3)
    encoding = StoreManifest.load(source).encoding
    width = int(root["z"].shape[1])
    flat = row * width
    eaf = _off_grid(StoreCodec(encoding), float(root["eaf_baseline"][row]))
    prediction = se_log_prediction(
        np.array([eaf]), np.array([0], np.int64), root["se_coefficients"][:]
    )
    se = np.float32(se_from_log(prediction + _OFF_GRID_STEPS * encoding.se.step)[0])
    for plane, value, index, values in (
        ("eaf", eaf, "eaf_exception_index", "eaf_exception_value"),
        ("se", se, "se_exception_index", "se_exception_value"),
    ):
        root[plane][row, 0] = EAF_EXCEPTION if plane == "eaf" else SE_EXCEPTION
        for name, data in (
            (index, np.array([flat], dtype=root[index].dtype)),
            (values, np.array([value], np.float32)),
        ):
            _put(root, name, data)
    return eaf, se


def _dense_cell(store: Path, row: int) -> tuple[np.float32, np.float32]:
    root = zarr.open_group(str(store / "data.zarr"), mode="r", zarr_format=3)
    encoding = StoreManifest.load(store).encoding
    eaf = DenseEafPlane.open(root, encoding).band(row, row + 1)[0, 0]
    se = DenseSePlane.open(root, encoding).band(row, row + 1)[0, 0]
    return np.float32(eaf), np.float32(se)


def test_dense_completion_keeps_an_observed_eaf_and_se_exception(tmp_path: Path) -> None:
    """The Dense counterpart: both planes' observed exceptions survive `_encode_band`."""
    source, panel, _ = _residual_source_and_panel(tmp_path)
    eaf, se = _pin_dense_cell(source, 20)
    assert _dense_cell(source, 20) == (eaf, se), "fixture: the source decodes to the pins"
    out = tmp_path / "comp.opengwasdb"
    complete_dense_store(
        source, out, PanelSource(panel, ancestry="EUR"),
        CompletionOptions(min_cor=0.0, release_id="comp"),
    )
    got = _dense_cell(out, 20)
    moved = [
        f"{name} {a!r} != {b!r}"
        for name, a, b in (("EAF", got[0], eaf), ("SE", got[1], se))
        if a != b
    ]
    assert not moved, f"completion moved an observed value: {moved}"


def _pin_crossover_se(src: Path, alid: str, eaf: float) -> np.float32:
    """Re-code the Ragged component's crossover SE for trait_a (Analysis 0) as an
    exact exception half a step off the *Dense Component's* model, in range."""
    store = open_store(src)
    dense = store.dense_component().arrays(mode="r")
    encoding = StoreManifest.load(src / "dense").encoding
    prediction = se_log_prediction(
        np.array([eaf], np.float32), np.array([0], np.int64), dense["se_coefficients"][:]
    )
    value = np.float32(se_from_log(prediction + _OFF_GRID_STEPS * encoding.se.step)[0])
    group = store.arrays(mode="a")["ragged"]
    with closing(VariantAxis(src)) as axis:
        index = next(r.variant_index for r in axis.all() if r.alid == alid)
    offsets, variants = group["offsets"][:], group["variant_index"][:]
    at = int(offsets[0]) + int(np.flatnonzero(variants[offsets[0] : offsets[1]] == index)[0])
    group["se"][at] = SE_EXCEPTION
    table = np.append(group["se_exception_index"][:], at)
    values = np.append(group["se_exception_value"][:], value)
    order = np.argsort(table, kind="stable")
    for name, data in (("se_exception_index", table[order]), ("se_exception_value", values[order])):
        _put(group, name, data)
    return value


def test_hybrid_crossover_fold_keeps_the_observed_se(tmp_path: Path) -> None:
    """The fold re-codes an observation against the Dense Component's baseline and
    model, which are not the Ragged source's; carried, it reads back as the source did.

    SE only: this fixture's EAF plan is `float32`, which cannot move, so the fold's EAF
    `carried` is not exercised by any Hybrid fixture (a residual-EAF Hybrid is not
    buildable from the test builders)."""

    src, alid, _se, crossover_eaf = _residual_hybrid_crossover_source(tmp_path)

    def cell(store: Path) -> tuple[np.float32, np.float32]:
        with closing(VariantAxis(store)) as axis:
            by_index = {r.variant_index: r.alid for r in axis.all()}
        with query_store(store) as q:
            r = q.analysis("trait_a", observed_only=True)
        k = next(i for i, v in enumerate(r["variant_index"]) if by_index[int(v)] == alid)
        return np.float32(r["eaf"][k]), np.float32(r["se"][k])

    pinned = _pin_crossover_se(src, alid, float(crossover_eaf))
    before = cell(src)
    assert before[1] == pinned, "fixture: the source decodes to the pinned SE"
    ld = _residual_ld_panel_with_crossover(tmp_path, alid, crossover_eaf)
    dst = tmp_path / "dst.opengwasdb"
    complete_hybrid_store(src, dst, PanelSource(ld), CompletionOptions(min_cor=0.0, thresh=0.9))
    after = cell(dst)
    moved = [
        f"{n} {a!r} != {b!r}"
        for n, a, b in (("EAF", after[0], before[0]), ("SE", after[1], before[1]))
        if a != b
    ]
    assert not moved, f"the crossover fold moved an observed value: {moved}"


def test_hybrid_se_exceptions_record_counts_the_final_dense_plane(tmp_path: Path) -> None:
    """The Dense Component's `se_exceptions` is the release's: the Dense pass's counts
    less the cells the crossover fold overwrote, plus the fold's own. Compared with a
    count taken from the final planes."""
    src, alid, _se, crossover_eaf = _residual_hybrid_crossover_source(tmp_path)
    _pin_crossover_se(src, alid, float(crossover_eaf))
    ld = _residual_ld_panel_with_crossover(tmp_path, alid, crossover_eaf)
    dst = tmp_path / "dst.opengwasdb"
    complete_hybrid_store(src, dst, PanelSource(ld), CompletionOptions(min_cor=0.0, thresh=0.9))
    component = open_store(dst).dense_component()
    record = component.manifest.provenance["completion"]["se_exceptions"]
    root = component.arrays(mode="r")
    codes, imputed = root["se"][:], root["imputed"][:]
    held = codes != SE_MISSING
    expected = {
        "n_se": held.sum(axis=0),
        "n_exceptions": (codes == SE_EXCEPTION).sum(axis=0),
        "n_imputed_se": (held & (imputed == 1)).sum(axis=0),
        "n_imputed_exceptions": ((codes == SE_EXCEPTION) & (imputed == 1)).sum(axis=0),
    }
    for key, counted in expected.items():
        assert [e[key] for e in record["analyses"]] == counted.tolist(), key
    assert expected["n_exceptions"][0] >= 1, "fixture: the fold holds an exception for trait_a"


def test_dense_completion_records_its_se_exceptions_per_analysis(tmp_path: Path) -> None:
    """A Dense completion that carries exact SE exceptions says how many, per
    Analysis, in `provenance.completion.se_exceptions`, as Ragged has since #301;
    the counts are the exception table's, which they must not disagree with."""
    source, panel, _ = _residual_source_and_panel(tmp_path)
    _pin_dense_cell(source, 20)
    out = tmp_path / "comp.opengwasdb"
    complete_dense_store(
        source, out, PanelSource(panel, ancestry="EUR"),
        CompletionOptions(min_cor=0.0, release_id="comp"),
    )
    record = StoreManifest.load(out).provenance["completion"]["se_exceptions"]
    root = zarr.open_group(str(out / "data.zarr"), mode="r", zarr_format=3)
    width = int(root["se"].shape[1])
    held = root["se_exception_index"][:] % width
    by_analysis = [e["n_exceptions"] for e in record["analyses"]]
    assert by_analysis == [int((held == a).sum()) for a in range(width)]
    assert by_analysis[0] >= 1, "fixture: the pinned exception is counted"
    assert [e["analysis_id"] for e in record["analyses"]] == ["a", "b"]
    assert record["analyses"][1]["n_imputed_se"] > 0, "fixture: `b` has imputed SEs"
