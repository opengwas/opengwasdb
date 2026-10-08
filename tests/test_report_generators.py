"""The generated report blocks must match the committed artifacts.

`scripts/build_finngen_shape_check.py` splices ADR 0058's "Checked on FinnGen
(#250)" block from committed JSON, and `scripts/build_ogs00011_extra_shapes.py`
aggregates the raw one-shape runs. Neither is wired into the suite, so an
artifact edit could leave the committed report/ADR stale and silent; these two
tests make that fail loudly (#250 review r2, nit 9).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adr_0058_finngen_block_matches_the_artifacts(monkeypatch):
    monkeypatch.chdir(ROOT)
    module = _load("build_finngen_shape_check", "scripts/build_finngen_shape_check.py")
    assert module.splice(module.build(), write=False) == 0


def test_extras_aggregate_matches_the_committed_artifact(monkeypatch):
    monkeypatch.chdir(ROOT)
    module = _load("build_ogs00011_extra_shapes", "scripts/build_ogs00011_extra_shapes.py")
    committed = json.loads((ROOT / module.OUT / module.AGG).read_text())
    rebuilt = module._aggregate(module._group(ROOT / module.OUT / module.RAW))
    assert rebuilt == committed


def _extras_record(column: str, shape: str, **overrides) -> dict:
    record = {
        "column": column, "shape": shape, "rep": 1, "elapsed_ms": 1000.0, "timed_out": False,
        "peak_mb": 100.0, "load_start": [1.0, 1.0, 1.0], "result_count": 1, "sha256": "x",
        "python": "/py", "zarr": "3.4.0", "hostname": "h", "opengwasdb_path": "/p",
        "opengwasdb_fingerprint": "f1", "commit": "c", "probe_path": "/probe",
        "probe_sha256": "p", "runner_path": "/runner", "runner_sha256": "r",
        "measured_at": "t", "waited_for_load_s": 0.0, "waited_timed_out": False,
    }
    return record | overrides


def test_extras_environment_refuses_a_column_whose_runs_disagree(monkeypatch):
    """One column label must name one environment, across every shape it ran."""
    monkeypatch.chdir(ROOT)
    module = _load("build_ogs00011_extra_shapes", "scripts/build_ogs00011_extra_shapes.py")
    agreeing = {
        ("a", "phewas_off_axis"): [_extras_record("a", "phewas_off_axis")],
        ("a", "bulk_overflow_heavy"): [_extras_record("a", "bulk_overflow_heavy")],
    }
    assert module._environments(agreeing)["a"]["opengwasdb_fingerprint"] == "f1"
    disagreeing = dict(agreeing)
    disagreeing[("a", "bulk_overflow_heavy")] = [
        _extras_record("a", "bulk_overflow_heavy", opengwasdb_fingerprint="f2")
    ]
    with pytest.raises(SystemExit, match="opengwasdb_fingerprint"):
        module._environments(disagreeing)


def test_adr_0058_verdict_reports_each_way_finngen_could_contradict_it(monkeypatch):
    monkeypatch.chdir(ROOT)
    module = _load("build_finngen_shape_check", "scripts/build_finngen_shape_check.py")
    data = module._inputs()
    assert module._failures(data) == []
    data["guard"] = module.GUARD_LIMIT + 0.05
    data["planes"] = data["planes"] | {"data.zarr/se": data["ref_planes"]["data.zarr/se"] * 1.2}
    failures = module._failures(data)
    assert len(failures) == 2
    assert "contradicts the decision" in module._verdict(data)[0]
