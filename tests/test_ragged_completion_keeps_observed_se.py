"""`complete-ragged` keeps a residual SE plane's observed values exactly (issue #301).

Completion used to refit `se_coefficients` over the completed rows, so every
observed SE of a residual-coded Store was re-quantised against a new model. On
three Chen 2023 Analyses all 135,596 observed SEs moved (median 0.09%, max
0.14%), and the strict Ragged Observed-View Check (#296) refused the pair with
`se_mismatch`. The owner's decision (option 1, 4 Nov 2026): completion keeps the
source's coefficients, every observed cell decodes to the source's value and
every observed exception is carried, imputed cells are coded against the
source's model, and one outside the residual range is an exact exception. The
2% per-Analysis exception budget is recorded and reported by name, never met by
refitting or by falling back to `float16`.

One residual-coded source (`test_ragged_residual_completion`'s), completed three
ways:

* `in_range` -- that module's panel: the imputed cells sit near the source's SE
  model, so they code as ordinary residuals. That module's session fixture
  builds this one, since its source, panel and completion arguments are the
  same.
* `nothing_imputed` -- no Analysis admitted. The refit moved observed SE even
  here, with no imputed cell, because it fitted decoded values rather than the
  source's own.
* `out_of_range` -- a panel whose frequencies at the observed positions are a
  tenth of the Store's, so the scalar-N SE it imputes sits about 1.1 log-units
  below the source's model, beyond the residual range. Only `b` is admitted, so
  `b` breaches the 2% budget and `a`, carried observed-only, does not.

A fourth fixture copies the source and re-codes one observed cell as an exact
exception whose value is half a code step off the grid: the drift the carry rule
exists for, which a completion that forgot to carry its observed cells would
re-quantise.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from shutil import copytree

import numpy as np
import pytest
from logged_warnings import logged_warnings
from region_tables import NO_SIGNIFICANT_REGIONS, write_region_table
from stored_rows import stored_rows
from test_ragged_residual_completion import _FREQS, RaggedResidualScenario, _source_bp

from opengwasdb import StoreManifest, open_store
from opengwasdb.completion.impute import scalar_n_se
from opengwasdb.encoding import SE_EXCEPTION, SE_EXCEPTION_BUDGET, SE_MISSING
from opengwasdb.encoding.codec import StoreCodec, se_residual_codes
from opengwasdb.layouts.ragged.complete import complete_ragged_store
from opengwasdb.model.analyses import read_analysis_records
from opengwasdb.store.arrays import ArrayRole, compressor, create_array
from opengwasdb.validation import validate_store
from opengwasdb.validation.observed_view import check_observed_view
from opengwasdb.variants.axis import VariantAxis

COMPLETIONS = ("in_range", "nothing_imputed", "out_of_range")
# Panel-only positions of the out-of-range panel: half-way between source rows
# 0..11, inside the block and the cis window.
_N_NEW = 12
_NEW_FREQ = 0.3
# Analysis `b` observes source rows 0..39 except these, which it gains imputed.
_B_UNOBSERVED = range(36, 40)
_OVER_BUDGET = {"in_range": [], "nothing_imputed": [], "out_of_range": ["b"]}


@dataclass(frozen=True)
class _Cell:
    """One association as a Store holds it: its decoded SE, its SE code, its flag."""

    se: np.float32
    code: int
    imputed: bool
    exact: np.float32 | None  # the exception table's value, for a `-127` cell


@dataclass(frozen=True)
class _Completion:
    path: Path
    panel: Path
    n_imputed: int
    warnings: list[str]


@dataclass(frozen=True)
class _Completions:
    source: Path
    runs: dict[str, _Completion]


@dataclass(frozen=True)
class _Pinned:
    """A copy of the fixture source holding one exact SE exception whose value is
    half a code step from the nearest code, and that copy's completion."""

    source: Path
    completed: _Completion
    key: tuple[str, str]  # (Analysis id, ALID) of the patched cell
    value: np.float32  # the value the source now holds for it


#: How far from the nearest code the patched cell's residual sits, in steps. Half
#: a step is the largest distance a code cannot express; well inside +-127.
_OFF_GRID_STEPS = 20.5
#: Analysis `a`'s source row to patch: observed, on the panel, not an exception.
_PINNED_ROW = 20


def _panel_eaf_text(value: float) -> str:
    return f"{value:.7g}"


def _write_panel(block_dir: Path, rows: list[tuple[str, float, int]], seed: int) -> None:
    """One chr-1 LD block: its variant table and a precomputed eigendecomposition.

    Each row's LD and `EAF` are coded on its ALID's A1, which it names as its
    `EA`; a panel row whose `EA` is neither allele is not placed (ADR 0079).
    """
    block_dir.mkdir(parents=True)
    lines = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
    lines += [
        f"1\t{alid}\t{alid.split(':')[3]}\t{alid.split(':')[2]}\t{_panel_eaf_text(eaf)}\t{bp}"
        for alid, eaf, bp in rows
    ]
    (block_dir / "skewed.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    n = len(rows)
    noise = np.random.default_rng(seed).normal(size=(n, n))
    ld = noise @ noise.T + np.eye(n) * n * 0.1
    values, vectors = np.linalg.eigh((ld + ld.T) / 2)
    order = np.argsort(values)[::-1]
    np.savez(block_dir / "skewed.ldeig.npz", values=values[order], vectors=vectors[:, order])


def _skewed_panel_rows() -> list[tuple[str, float, int]]:
    """Source rows 0..39 at a tenth of the Store's frequency, then the panel-only rows."""
    rows = [(f"1:{_source_bp(k)}:A:G", float(_FREQS[k]) / 10, _source_bp(k)) for k in range(40)]
    rows += [
        (f"1:{_source_bp(k) + 500}:A:G", _NEW_FREQ, _source_bp(k) + 500) for k in range(_N_NEW)
    ]
    return rows


def _complete(
    source: Path, root: Path, panel: Path, impute_analysis_ids: set[str] | None = None
) -> _Completion:
    root.mkdir(parents=True, exist_ok=True)
    with logged_warnings() as warnings:
        result = complete_ragged_store(
            source, root / "comp.opengwasdb", panel, ancestry="EUR", cis_window_bp=1_000_000,
            min_cor=0.0, release_id="comp", impute_analysis_ids=impute_analysis_ids,
            regions=write_region_table(root / "regions.tsv", NO_SIGNIFICANT_REGIONS),
        )
    return _Completion(root / "comp.opengwasdb", panel, result.n_imputed, warnings)


@pytest.fixture(scope="module")
def completions(
    tmp_path_factory: pytest.TempPathFactory, residual_ragged_scenario: RaggedResidualScenario
) -> _Completions:
    """That module's completed release reused as `in_range`, whose source, panel and
    completion arguments are the same (session-scoped, so it is built once), and
    two more completions of the same source this module needs its own of.

    `nothing_imputed` is the same completion with no Analysis admitted, which the
    scenario's release cannot answer; `out_of_range` has its own panel.
    """
    root = tmp_path_factory.mktemp("keeps_observed_se_301")
    source, panel = residual_ragged_scenario.source, residual_ragged_scenario.panel
    skewed = root / "out_of_range" / "panel"
    _write_panel(skewed / "EUR" / "1", _skewed_panel_rows(), seed=301)
    return _Completions(
        source,
        {
            "in_range": _Completion(
                residual_ragged_scenario.completed,
                panel,
                residual_ragged_scenario.result.n_imputed,
                residual_ragged_scenario.warnings,
            ),
            "nothing_imputed": _complete(
                source, root / "nothing_imputed", panel, impute_analysis_ids=set()
            ),
            "out_of_range": _complete(
                source, root / "out_of_range", skewed, impute_analysis_ids={"b"}
            ),
        },
    )


def _ragged(store: Path):
    return open_store(store).arrays(mode="r")["ragged"]


def _alid_of(store: Path, variant_index: int) -> str:
    with closing(VariantAxis(store)) as axis:
        return next(v.alid for v in axis.all() if v.variant_index == variant_index)


@pytest.fixture(scope="module")
def off_grid(completions: _Completions, tmp_path_factory: pytest.TempPathFactory) -> _Pinned:
    """The fixture source copied, with one observed cell re-coded as an exact
    exception whose value is half a code step off the grid and in range -- the
    drift ADR 0078's first case describes -- then completed.

    A copy, so the module's own source keeps the two exceptions its tests assert.
    The value comes from the codec's own prediction over the frequency the cell
    decodes to, never from a hand-written model (issue #301).
    """
    root = tmp_path_factory.mktemp("off_grid_301")
    source = Path(copytree(completions.source, root / "source.opengwasdb"))
    group = open_store(source).arrays(mode="a")["ragged"]
    encoding = StoreManifest.load(source).encoding
    codec = StoreCodec(encoding)
    ai, k = 0, _PINNED_ROW
    offsets, variants = group["offsets"][:], group["variant_index"][:]
    row = stored_rows(source, ai)
    at = int(offsets[ai]) + k
    assert int(variants[at]) == int(row.variant_index[k]), "fixture: row order is plane order"
    prediction = codec._se_prediction(
        np.asarray(row.eaf[k : k + 1], dtype=np.float32),
        np.array([ai], np.int64),
        group["se_coefficients"][:],
    )
    value = codec._se_from_codes(prediction, np.array([_OFF_GRID_STEPS]), encoding.se.step)[0]
    residual = np.log(np.float64(value)) - prediction
    steps = float(residual[0]) / encoding.se.step
    assert abs(steps - _OFF_GRID_STEPS) < 0.1, f"fixture: {steps} is not half a step off grid"
    assert abs(steps) < 127, f"fixture: {steps} must be inside the residual range"
    _, exceptional = se_residual_codes(np.array([value]), residual, encoding.se.step)
    assert not exceptional[0], "fixture: in range, so only the carry can hold this cell"

    # Re-key the plane and its table the way the store keeps them: sorted by index.
    group["se"][at] = SE_EXCEPTION
    index = np.append(group["se_exception_index"][:], at)
    values = np.append(group["se_exception_value"][:], value)
    order = np.argsort(index, kind="stable")
    for name, data in (("se_exception_index", index), ("se_exception_value", values)):
        create_array(
            group,
            name,
            ArrayRole.EXCEPTION_TABLE,
            data=data[order],
            compressor=compressor(),
            overwrite=True,
        )
    key = ("a", _alid_of(source, int(row.variant_index[k])))
    patched = _cells(source)[key]
    assert patched.code == SE_EXCEPTION and patched.exact is not None, "fixture: held exactly"
    completed = _complete(source, root, completions.runs["in_range"].panel)
    return _Pinned(source, completed, key, np.float32(patched.exact))


def _analysis_ids(store: Path) -> list[str]:
    return [a.analysis_id for a in read_analysis_records(store / "analyses.tsv")]


def _cells(store: Path) -> dict[tuple[str, str], _Cell]:
    """Every association of a Store, keyed by (Analysis id, ALID)."""
    group = _ragged(store)
    offsets, codes = group["offsets"][:], group["se"][:]
    flags = group["imputed"][:] if "imputed" in group else np.zeros(len(codes), np.uint8)
    exact = dict(
        zip(
            group["se_exception_index"][:].tolist(),
            group["se_exception_value"][:].tolist(),
            strict=True,
        )
    )
    with closing(VariantAxis(store)) as axis:
        alid = {v.variant_index: v.alid for v in axis.all()}
    cells: dict[tuple[str, str], _Cell] = {}
    for ai, analysis_id in enumerate(_analysis_ids(store)):
        row = stored_rows(store, ai)
        for i, (vi, se) in enumerate(zip(row.variant_index, row.se, strict=True)):
            at = int(offsets[ai]) + i
            held = exact.get(at)
            cells[(analysis_id, alid[int(vi)])] = _Cell(
                np.float32(se), int(codes[at]), bool(flags[at]),
                None if held is None else np.float32(held),
            )
    return cells


def _observed_in_source(completions: _Completions) -> dict[tuple[str, str], _Cell]:
    cells = _cells(completions.source).items()
    observed = {key: cell for key, cell in cells if cell.code != SE_MISSING}
    assert len(observed) == 600 + 596, "fixture: the source holds 1,196 observed SEs"
    return observed


def _bits(value: np.float32) -> int:
    return int(np.float32(value).view(np.uint32))


# --- the fixtures are meaningful before anything is asserted about them ------


@pytest.mark.parametrize("name", COMPLETIONS)
def test_each_completion_reaches_the_residual_path(completions: _Completions, name: str) -> None:
    run = completions.runs[name]
    source_plan = StoreManifest.load(completions.source).encoding
    assert source_plan.se.is_residual, "fixture: the source codes SE as int8_residual"
    assert StoreManifest.load(run.path).encoding.se.is_residual
    assert len(_ragged(completions.source)["se_exception_index"][:]) == 2, (
        "fixture: the source holds one exact exception per Analysis"
    )
    imputed = sum(cell.imputed for cell in _cells(run.path).values())
    assert imputed == run.n_imputed
    if name == "nothing_imputed":
        assert run.n_imputed == 0
    else:
        assert run.n_imputed >= 8, "fixture: completion imputes"


def test_out_of_range_fixture_puts_every_imputed_cell_outside_the_residual_range(
    completions: _Completions,
) -> None:
    """Under the source's own model, no imputed cell of `out_of_range` has a code."""
    run = completions.runs["out_of_range"]
    plan = StoreManifest.load(completions.source).encoding.se
    intercept, slope = (float(v) for v in _ragged(completions.source)["se_coefficients"][1])
    panel_eaf = {alid: eaf for alid, eaf, _ in _skewed_panel_rows()}
    imputed = {key: c for key, c in _cells(run.path).items() if c.imputed}
    assert {analysis_id for analysis_id, _ in imputed} == {"b"}, "fixture: only `b` imputes"
    assert len(imputed) == _N_NEW + len(_B_UNOBSERVED)
    for (_, alid), cell in imputed.items():
        f = float(_panel_eaf_text(panel_eaf[alid]))
        residual = np.log(float(cell.se)) - (intercept + slope * np.log(2 * f * (1 - f)))
        assert abs(np.rint(residual / plan.step)) > 127, (
            f"fixture: {alid}'s residual {residual:.3f} must be outside ±{plan.residual_range}"
        )


# --- observed cells are carried exactly ---------------------------------------


@pytest.mark.parametrize("name", COMPLETIONS)
def test_every_observed_se_decodes_bit_identical(completions: _Completions, name: str) -> None:
    observed = _observed_in_source(completions)
    completed = _cells(completions.runs[name].path)
    moved = {
        key: (float(cell.se), float(completed[key].se))
        for key, cell in observed.items()
        if _bits(completed[key].se) != _bits(cell.se) or completed[key].imputed
    }
    assert not moved, f"{len(moved)} of {len(observed)} observed SEs changed, e.g. " + str(
        list(moved.items())[:3]
    )


@pytest.mark.parametrize("name", COMPLETIONS)
def test_completion_keeps_the_source_se_coefficients(completions: _Completions, name: str) -> None:
    source = _ragged(completions.source)["se_coefficients"][:]
    completed = _ragged(completions.runs[name].path)["se_coefficients"][:]
    assert completed.dtype == np.float32
    assert np.array_equal(completed.view(np.uint32), source.view(np.uint32)), (
        f"refit: {source.tolist()} became {completed.tolist()}"
    )


@pytest.mark.parametrize("name", COMPLETIONS)
def test_observed_exceptions_are_carried_bit_for_bit(completions: _Completions, name: str) -> None:
    """The source's exception cells are exceptions after completion, with the same
    value, and no other observed cell is one."""
    observed = _observed_in_source(completions)
    completed = _cells(completions.runs[name].path)
    source_exceptions = {key: c.exact for key, c in observed.items() if c.code == SE_EXCEPTION}
    assert len(source_exceptions) == 2, "fixture: one exception per Analysis"
    carried = {
        key: completed[key].exact for key in observed if completed[key].code == SE_EXCEPTION
    }
    assert carried.keys() == source_exceptions.keys()
    for key, value in source_exceptions.items():
        assert value is not None and carried[key] is not None
        assert _bits(carried[key]) == _bits(value), key


def test_an_observed_exception_off_the_code_grid_survives_completion(off_grid: _Pinned) -> None:
    """An observed cell held as an exact exception whose value sits half a code step
    from the nearest code reads back exactly only while it stays an exception:
    coded, it would move by half a step. Completion carries every observed cell, so
    the cell arrives in the completed Store as an exception holding its source value
    bit for bit (ADR 0078, issue #301)."""
    completed = _cells(off_grid.completed.path)[off_grid.key]
    assert completed.code == SE_EXCEPTION, "a cell off the code grid must stay an exception"
    assert _bits(completed.se) == _bits(off_grid.value), "the carried SE moved"
    assert completed.exact is not None and _bits(completed.exact) == _bits(off_grid.value)
    assert not completed.imputed


# --- imputed cells are coded against the source's model -----------------------


def test_an_imputed_cell_outside_the_range_is_an_exact_exception(
    completions: _Completions,
) -> None:
    """Each of `b`'s imputed cells is held in the exception table at the value the
    scalar-N model imputes from the source's decoded SE and the panel's frequencies."""
    run = completions.runs["out_of_range"]
    source_cells = _cells(completions.source)
    rows = _skewed_panel_rows()
    se_block = np.array(
        [
            float(source_cells[("b", alid)].se) if ("b", alid) in source_cells else np.nan
            for alid, _, _ in rows
        ]
    )
    eaf_block = np.array([float(_panel_eaf_text(eaf)) for _, eaf, _ in rows])
    seen = np.isfinite(se_block)
    expected = dict(
        zip(
            (alid for alid, _, _ in rows),
            scalar_n_se(se_block[seen], eaf_block[seen], eaf_block).tolist(),
            strict=True,
        )
    )
    imputed = {alid: c for (aid, alid), c in _cells(run.path).items() if c.imputed and aid == "b"}
    assert len(imputed) == _N_NEW + len(_B_UNOBSERVED), "fixture: `b` imputes 16 cells"
    for alid, cell in imputed.items():
        assert cell.code == SE_EXCEPTION, f"{alid}: an out-of-range residual is an exception"
        assert cell.exact is not None and _bits(cell.se) == _bits(cell.exact)
        np.testing.assert_allclose(float(cell.se), expected[alid], rtol=1e-6, err_msg=alid)


def test_the_strict_observed_view_check_passes(completions: _Completions) -> None:
    for name, run in completions.runs.items():
        verdict = check_observed_view(
            completions.source, run.path, seed=301, n_random=10, ld_panel=run.panel,
            regions=run.path.parent / "regions.tsv",  # the table its completion was given
        )
        totals = verdict.record["totals"]
        assert totals["observed_in_source"] == 600 + 596, f"{name}: every Analysis compared"
        assert totals["se_mismatch"] == 0, f"{name}: {verdict.record['max_abs_diff']}"
        assert verdict.passed, f"{name}: {verdict.failures}"


# --- the exception budget is recorded and reported, never handled ---------------


def _plane_tally(store: Path) -> Iterator[tuple[str, dict[str, int]]]:
    group = _ragged(store)
    offsets, codes, flags = group["offsets"][:], group["se"][:], group["imputed"][:]
    for ai, analysis_id in enumerate(_analysis_ids(store)):
        span = slice(int(offsets[ai]), int(offsets[ai + 1]))
        held, imputed = codes[span] != SE_MISSING, flags[span] == 1
        exception = codes[span] == SE_EXCEPTION
        yield analysis_id, {
            "n_se": int(held.sum()),
            "n_exceptions": int(exception.sum()),
            "n_imputed_se": int((held & imputed).sum()),
            "n_imputed_exceptions": int((exception & imputed).sum()),
        }


@pytest.mark.parametrize("name", COMPLETIONS)
def test_each_analysis_exception_rate_is_recorded(completions: _Completions, name: str) -> None:
    store = completions.runs[name].path
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["provenance"]["completion"]["se_exceptions"]
    assert record["budget"] == SE_EXCEPTION_BUDGET
    expected = [
        {
            "analysis_id": analysis_id,
            **counts,
            "exception_rate": counts["n_exceptions"] / counts["n_se"],
        }
        for analysis_id, counts in _plane_tally(store)
    ]
    assert [e["analysis_id"] for e in expected] == ["a", "b"]
    assert all(e["n_exceptions"] >= 1 for e in expected), "fixture: each Analysis has one"
    assert record["analyses"] == expected
    assert record["over_budget"] == _OVER_BUDGET[name]


def test_an_analysis_over_budget_is_named_and_its_plan_kept(completions: _Completions) -> None:
    """`b` holds 17 exceptions in 612 SEs (2.8%). Completion keeps the residual plan
    and its range, and warns naming `b` -- not `a`, which holds 1 in 600."""
    run = completions.runs["out_of_range"]
    rates = {
        analysis_id: counts["n_exceptions"] / counts["n_se"]
        for analysis_id, counts in _plane_tally(run.path)
    }
    assert rates["b"] > SE_EXCEPTION_BUDGET >= rates["a"], f"fixture: {rates}"
    kept = StoreManifest.load(run.path).encoding.se
    assert kept == StoreManifest.load(completions.source).encoding.se, "no float16 fallback"
    named = [message for message in run.warnings if "exception" in message]
    assert len(named) == 1, run.warnings
    assert re.findall(r"(\w+): (\d+\.\d+)% \(", named[0]) == [("b", "2.78")], named[0]
    assert not [m for m in completions.runs["in_range"].warnings if "exception" in m]


def test_validate_accepts_a_completed_store_over_budget(completions: _Completions) -> None:
    """The budget bounds side-table bytes, not accuracy (ADR 0037 §3), so `validate`
    does not enforce it: a completed Store whose imputed cells breach it is valid.
    Enforcing it there would refuse what completion is told to keep (ADR 0078)."""
    run = completions.runs["out_of_range"]
    manifest = json.loads((run.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["completion"]["se_exceptions"]["over_budget"] == ["b"]
    result = validate_store(run.path)
    assert result.ok, result.errors
