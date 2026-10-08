"""Every timed repetition of the #252 extras must start on a quiet node (#250).

The first confirming run gated each column once, before its first repetition,
and 15 of its 30 repetitions then began at a 1-minute load of 3 or more. The
probe now runs a gate after it has opened the store and immediately before it
samples `load_start` and starts the clock, and the runner passes the load wait
as that gate on every repetition, recording how long it waited and whether it
gave up.
"""

from __future__ import annotations

import json
import sys

import numpy as np

import benchmarks.benchmark_ogs00011_hybrid as hybrid_harness
import opengwasdb.query
from benchmarks import ogs00011_ab, ogs00011_extra_shapes


def _result() -> dict[str, np.ndarray]:
    return {
        "variant_index": np.array([7], dtype=np.int32),
        "analysis_index": np.array([0], dtype=np.int32),
        "z": np.array([1.5], dtype=np.float32),
        "se": np.array([0.1], dtype=np.float32),
        "eaf": np.array([0.2], dtype=np.float32),
        "association_status": np.array(["observed"], dtype=object),
    }


def test_probe_gates_after_opening_the_store_and_before_the_clock(monkeypatch):
    events: list[str] = []

    class _Query:
        def close(self) -> None:
            events.append("close")

    def _open(_store):
        events.append("open")
        return _Query()

    def _shape():
        events.append("timed")
        return _result()

    def _loads():
        events.append("load")
        return [1.0, 1.0, 1.0]

    def _gate():
        events.append("gate")
        return {"gate_wait_s": 30.0, "gate_gave_up": False}

    monkeypatch.setattr(opengwasdb.query, "query_store", _open)
    monkeypatch.setattr(hybrid_harness, "_patterns", lambda query, path: {"probe": _shape})
    monkeypatch.setattr(ogs00011_ab, "_loads", _loads)

    record = ogs00011_ab.measure_one_shape("/store", "probe", 60.0, before_timing=_gate)

    assert events == ["open", "gate", "load", "timed", "load", "close"]
    assert record["gate_wait_s"] == 30.0
    assert record["gate_gave_up"] is False
    assert record["result_count"] == 1


def test_runner_gates_every_repetition_and_records_each_wait(monkeypatch, tmp_path):
    waits = iter([(0.0, False), (45.0, False), (3600.5, True)])
    calls: list[str] = []

    def _wait(max_load):
        calls.append("wait")
        return next(waits)

    def _measure(store, shape, limit, *, before_timing=None):
        gate = before_timing() if before_timing is not None else {}
        return {"shape": shape, "elapsed_ms": 1.0, "timed_out": False, **gate}

    output = tmp_path / "extras.jsonl"
    monkeypatch.setattr(ogs00011_extra_shapes, "wait_for_quiet", _wait)
    monkeypatch.setattr(ogs00011_ab, "measure_one_shape", _measure)
    monkeypatch.setattr(sys, "argv", [
        "ogs00011_extra_shapes.py", "--store", "/store", "--shape", "phewas_off_axis",
        "--column", "c-code-0.2.0", "--reps", "3", "--output", str(output),
    ])

    ogs00011_extra_shapes.main()

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert calls == ["wait", "wait", "wait"]
    assert [r["rep"] for r in records] == [1, 2, 3]
    assert [r["gate_wait_s"] for r in records] == [0.0, 45.0, 3600.5]
    assert [r["gate_gave_up"] for r in records] == [False, False, True]
    assert all(r["column"] == "c-code-0.2.0" and r["runner_sha256"] for r in records)
