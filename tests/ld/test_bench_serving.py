"""ogld#9's serving measurements (`scripts/bench/serving.py`): what they read, when they stop,
and that `docs/benchmarks/serving-tables.md` is their records rendered.

ADR 0008 rests on these numbers: the memory of a process holding the five stores, filled to its
cache's budget, and how late an event loop's ticks wake while ogld reads beside it.
"""

from __future__ import annotations

import asyncio
import gzip
import importlib
import json
import random
import sys
import time

import pytest
from script_modules import SCRIPTS
from testdata import TEST_PANEL

import ogld

for _path in (SCRIPTS / "bench", SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
serving = importlib.import_module("serving")

TEST_STORE = TEST_PANEL / "store"
RESULTS = SCRIPTS.parent / "docs" / "benchmarks" / "results" / "9"


# ------------------------------------------------------------------ what a record reads


def test_status_reads_resident_memory_and_its_peak_in_bytes() -> None:
    text = "Name:\tpython\nVmHWM:\t  2048 kB\nVmRSS:\t  1024 kB\nThreads:\t8\n"
    assert serving.parse_status(text) == {"VmHWM": 2048 * 1024, "VmRSS": 1024 * 1024}
    live = serving.status()
    assert live["VmRSS"] > 0
    assert live["VmHWM"] >= live["VmRSS"]


def test_a_status_without_both_fields_is_refused_not_recorded_short() -> None:
    with pytest.raises(ValueError, match="VmRSS and VmHWM"):
        serving.parse_status("VmRSS:\t1024 kB\n")


def test_the_99th_percentile_is_the_nearest_rank_at_or_below_it() -> None:
    lags = [i / 1000 for i in range(100)]
    random.Random(9).shuffle(lags)
    got = serving.summary(lags)
    assert got["ticks"] == 100
    assert got["lag_p50_ms"] == pytest.approx(49.5)
    assert got["lag_p99_ms"] == pytest.approx(98.0)
    assert got["lag_max_ms"] == pytest.approx(99.0)
    assert got["late_over_100ms"] == 0
    late = serving.summary([*lags, 0.15, 0.2])
    assert late["late_over_100ms"] == 2
    assert late["lag_max_ms"] == pytest.approx(200.0)
    assert serving.summary([0.003])["lag_p99_ms"] == pytest.approx(3.0)


def test_no_tick_is_refused_not_read_as_no_lag() -> None:
    with pytest.raises(ValueError, match="no tick"):
        serving.summary([])


def test_a_tick_wakes_late_by_what_holds_the_loop() -> None:
    def hold_loop(seconds: float) -> None:
        # Busy, not asleep: the loop is held as a thread holding the interpreter would hold it.
        stop = time.perf_counter() + seconds
        while time.perf_counter() < stop:
            pass

    async def measured() -> list[float]:
        ticks = iter(range(6))
        asyncio.get_running_loop().call_later(0.001, hold_loop, 0.06)
        return await serving.lags_while(lambda: next(ticks, None) is not None)

    lags = asyncio.run(measured())
    assert len(lags) == 6, "one lag a tick, until the predicate stops it"
    assert max(lags) >= 0.05, "the tick the held loop delayed must show it"
    assert min(lags) > -0.001


def test_ticking_for_a_time_stops_when_it_is_up() -> None:
    started = time.perf_counter()
    lags = asyncio.run(serving.lags_while(serving.for_seconds(0.03)))
    assert lags, "it must tick at least once"
    assert time.perf_counter() - started < 1.0


# ------------------------------------------------------------------ filling a cache


def test_fill_refuses_to_record_a_cache_short_of_full() -> None:
    store = ogld.open(TEST_STORE)
    places = serving.centres(store)
    assert len(places) == 4, "the Test Store's two chromosomes must give regions to read"
    with pytest.raises(SystemExit, match="ran out of regions after 4"):
        serving.fill(store, places)
    assert serving.held(store) > 0, "the regions were read, and still fell short of 4 GiB"


def test_fill_stops_once_the_cache_holds_its_budget() -> None:
    first = ogld.open(TEST_STORE)
    place = serving.centres(first)[0]
    first.region(place[0], place[1], serving.FILL_RADIUS)
    one_region = serving.held(first)
    assert one_region > 0, "one region must keep chunks for this to mean anything"
    store = ogld.open(TEST_STORE, cache_bytes=one_region)
    assert serving.fill(store, serving.centres(store)) == 1
    assert serving.held(store) >= serving.FULL * store._reader.cache_bytes


# ------------------------------------------------------------------ the committed records


def records(name: str) -> list[dict[str, object]]:
    lines = gzip.decompress((RESULTS / name).read_bytes()).decode().splitlines()
    return [json.loads(line) for line in lines]


def test_the_serving_tables_are_the_committed_records_rendered() -> None:
    paths = sorted(RESULTS.glob("*.jsonl.gz"))
    assert len(paths) == 4, "the clump, read, memory and lag runs"
    committed = (RESULTS.parent.parent / "serving-tables.md").read_text()
    rendered = serving.tables(paths) + "\n"
    assert rendered == committed, "re-render serving-tables.md (bench-serving tables)"


def test_serving_md_shows_the_memory_and_lag_tables_as_rendered() -> None:
    found = serving.runs(sorted(RESULTS.glob("*.jsonl.gz")))
    of = {kind: [r for r in found if r[1].get("measurement") == kind] for kind in ("memory", "lag")}
    assert of["memory"], "the memory run must be committed"
    assert of["lag"], "the lag runs must be committed"
    page = (RESULTS.parent.parent / "serving.md").read_text()
    from report import table

    assert table(serving.MEMORY_HEADER, serving.memory_rows(of["memory"])) in page
    assert table(serving.LAG_HEADER, serving.lag_rows(of["lag"])) in page


def test_the_memory_and_lag_records_name_a_clean_checkout() -> None:
    for name in ("memory.jsonl.gz", "lag.jsonl.gz"):
        runs = [r for r in records(name) if r.get("kind") == "environment"]
        assert runs, f"{name} must hold its runs"
        for run in runs:
            assert run["code"]["clean"], f"{name}: {run['code']['problem']}"


def test_the_engine_s_zarr_settings_changed_no_answer() -> None:
    timed = [r for r in records("lag.jsonl.gz") if r.get("measurement") == "lag" and "result" in r]
    for query in serving.LAG_QUERIES:
        mine = [r for r in timed if r["query"] == query]
        assert {r["engine_zarr"] for r in mine} == {True, False}, f"{query}: both settings ran"
        answers = {json.dumps(r["result"], sort_keys=True) for r in mine}
        assert len(answers) == 1, f"{query} answered differently: {answers}"
    matrix = next(r for r in timed if r["query"] == "matrix_500_genome")
    assert "stored" in matrix["result"], "a matrix's answer counts its stored pairs"


def test_every_committed_file_is_a_bench_run() -> None:
    # A file that is not a run, or a run with no records, would render as nothing.
    for path in sorted(RESULTS.glob("*.jsonl.gz")):
        found = serving.runs([path])
        assert found, path.name
        assert all(written for _, _, written in found), path.name
