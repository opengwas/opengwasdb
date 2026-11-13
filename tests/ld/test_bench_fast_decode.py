"""The #40 record names whole answers and fetches, not just lead counts."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from script_modules import load_script
from store_copies import TEST_STORE

import ogld


def test_benchmark_records_result_bytes_and_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/bench"))
    probe = load_script("bench/fast_decode")
    # Undo every patched attribute; another benchmark test shares this process.
    monkeypatch.setattr(probe.ChunkCache, "_keep_run", probe.ChunkCache._keep_run)
    monkeypatch.setattr(probe.latency.Bench, "timed", probe.latency.Bench.timed)
    monkeypatch.setattr(probe.latency, "clump_query", probe.latency.clump_query)
    probe.install()
    store = ogld.open(TEST_STORE)
    names = store._reader.alids_of(np.array([0, 5, 500])).tolist()
    result = store.clump(names, [3, 2, 1], r2=0.01, kb=3000)
    changed = store.clump(names, [1, 2, 3], r2=0.01, kb=3000)
    assert probe.fingerprint(result) != probe.fingerprint(changed)
    store = ogld.open(TEST_STORE)
    out = io.StringIO()
    bench = probe.latency.Bench(
        TEST_STORE.parent, probe.latency.Target("local", None), None, out, "40", "direct"
    )
    bench.timed(lambda: probe.query(names, [3, 2, 1], 0.01)(store), lambda: store)
    record = json.loads(out.getvalue())
    assert record["result"]["sha256"] == probe.fingerprint(result)
    assert "load average" in record["uptime_before"]
    assert "load average" in record["uptime_after"]
    assert record["chunk_fetches"] >= record["distinct_chunks"] > 0
