"""The write-amplification harness: its fixture's coverage and its combine mode (#249).

The harness builds a synthetic Ragged component to measure the sequence write
amplification.  These pin the two things whose silent failure would weaken the
artifact: that the fixture really reaches the residual SE/EAF branches and every
side table (round 1 found the first fixture took `se float16` / `eaf float32` and
exercised none of them), and that the committed envelope is assembled by the
script rather than by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.measure_write_amplification import _plan, combine_artifacts, measure


def test_the_plan_forces_the_residual_branches() -> None:
    plan = _plan()
    assert plan.se.is_residual, plan.se.kind
    assert plan.eaf.is_residual, plan.eaf.kind
    assert plan.z.kind == "int16_fixed"


def test_the_synthetic_reaches_the_residual_paths_and_side_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A small flush writes the residual planes, every side table, and a real region peak."""
    import benchmarks.measure_write_amplification as amp
    import opengwasdb.store.arrays as store_arrays

    # Several regions in a cheap fixture: a 200-cell shard (one 200-cell inner
    # chunk) with a 200-cell region, so six regions cover 1,200 cells.
    for module in (store_arrays, amp):
        monkeypatch.setattr(module, "RAGGED_SEQUENCE_SHARD_ELEMENTS", 200)
        monkeypatch.setattr(module, "ASSOCIATION_SEQUENCE_CHUNK", 200)
    payload = measure(
        SimpleNamespace(
            legacy_step=False,
            work=tmp_path / "amp",
            total_cells=1200,
            n_analyses=2,
            n_variants=500,
            seed=0,
            region_cells=200,
        )
    )
    assert payload["encoding"]["se"]["kind"] == "int8_residual"
    assert payload["encoding"]["eaf"]["kind"] == "int8_residual"
    for name in (
        "eaf_baseline",
        "eaf_exception_index",
        "se_coefficients",
        "se_exception_index",
        "z_overflow_index",
    ):
        assert name in payload["arrays"], sorted(payload["arrays"])
    for name in ("eaf_exception_index", "se_exception_index", "z_overflow_index"):
        assert payload["arrays"][name]["final_bytes"] > 0, name
    # The planted exact exceptions make the tables non-empty, not just present.
    assert payload["arrays"]["eaf"]["shards"] == 6
    assert payload["arrays"]["eaf"]["max_writes_one_shard"] == 1
    assert payload["seed"] == 0
    # The fixture really is several regions, and each window's allocation peak is
    # the region's: the SE half allocates at least its float32 error plane, so a
    # peak below `cells x 4 B` is a collapsed or out-of-window measurement -- the
    # failure mode the RSS delta cannot show (#249 r3).
    largest = payload["largest_region_cells"]
    assert largest == 200, payload["region_step_cells"]
    assert payload["eaf_region_alloc_peak_bytes"] > 0, payload
    assert payload["se_region_alloc_peak_bytes"] >= largest * 4, payload[
        "se_region_alloc_peak_bytes"
    ]
    assert payload["region_alloc_peak_bytes"] == max(
        payload["eaf_region_alloc_peak_bytes"], payload["se_region_alloc_peak_bytes"]
    )
    assert payload["region_bytes_per_cell"] >= 4, payload["region_bytes_per_cell"]


def _run(legacy_step: bool) -> dict[str, object]:
    return {
        "legacy_step": legacy_step,
        "commit": "abc1234",
        "shard_elements": 50_000_000,
        "inner_chunk": 200_000,
        "region_cells": 1 << 22,
        "total_cells": 1,
        "n_analyses": 1,
        "n_variants": 1,
        "seed": 0,
        "encoding": {"se": {"kind": "int8_residual"}},
    }


def test_combine_artifacts_builds_the_envelope_and_refuses_swapped_runs(tmp_path: Path) -> None:
    legacy, fixed = _run(True), _run(False)
    legacy_path, fixed_path = tmp_path / "legacy.json", tmp_path / "fixed.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    fixed_path.write_text(json.dumps(fixed), encoding="utf-8")
    envelope = combine_artifacts(legacy_path, fixed_path)
    assert envelope["legacy_step"] == legacy
    assert envelope["fixed_step"] == fixed
    assert envelope["fixture"]["seed"] == 0
    assert [run["legacy_step"] for run in envelope["runs"]] == [True, False]
    assert envelope["commit"] == "abc1234"
    with pytest.raises(SystemExit, match="legacy run first"):
        combine_artifacts(fixed_path, legacy_path)


def test_combine_artifacts_refuses_a_different_fixture(tmp_path: Path) -> None:
    """A before/after of two different fixtures must not pair silently (#249 r2)."""
    legacy, fixed = _run(True), _run(False)
    fixed["total_cells"] = 160_000_000
    fixed["seed"] = 7
    legacy_path, fixed_path = tmp_path / "legacy.json", tmp_path / "fixed.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    fixed_path.write_text(json.dumps(fixed), encoding="utf-8")
    with pytest.raises(SystemExit, match="disagree on"):
        combine_artifacts(legacy_path, fixed_path)
