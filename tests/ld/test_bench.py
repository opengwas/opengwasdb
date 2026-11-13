"""ogld#8's read-latency benchmark (`scripts/bench/`): what it counts, the wait its proxy adds,
and which records its report keeps.

`docs/benchmarks/latency.md` rests on two things these hold. Its round-trip model assumes the
proxy waits once for each request, and requests in flight together wait together. Its byte and
request columns are the client's counts, which must equal what crossed the proxy. ogld#31 and
ogld#32 re-run the benchmark, so both stay tested, not just checked once.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import importlib
import io
import json
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from script_modules import SCRIPTS
from testdata import TEST_PANEL

for _path in (SCRIPTS / "bench", SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
http_static = importlib.import_module("http_static")
latency = importlib.import_module("latency")
report = importlib.import_module("report")

TEST_STORE = TEST_PANEL / "store"
RESULTS = SCRIPTS.parent / "docs" / "benchmarks" / "results"
#: What a record measured, rather than which read it was.
MEASURED = {"seconds", "load_before", "load_after", "client", "wire", "result", "latency_ms"}


# ------------------------------------------------------------------ serving


@contextmanager
def served(root: Path, wait_ms: float = 0.0) -> Iterator[str]:
    """`http_static.py`'s static host for `root` and its proxy, in this process, as a URL."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    def run(coroutine: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result(10)

    static, port = run(http_static.started(http_static.static_app(root)))
    proxy, proxy_port = run(http_static.started(http_static.proxy_app(f"http://127.0.0.1:{port}")))
    url = f"http://127.0.0.1:{proxy_port}"
    control(url, f"latency/{wait_ms:g}")
    try:
        yield url
    finally:
        run(proxy.cleanup())
        run(static.cleanup())
        loop.call_soon_threadsafe(loop.stop)
        thread.join(10)
        loop.close()


def control(url: str, action: str) -> dict[str, Any]:
    request = urllib.request.Request(f"{url}/__ogld/{action}", method="POST")
    with urllib.request.urlopen(request) as response:
        answer: dict[str, Any] = json.load(response)
    return answer


def fetch(url: str, byte_range: str | None = None) -> tuple[int, bytes]:
    headers = {"Range": f"bytes={byte_range}"} if byte_range else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
        return response.status, response.read()


# ------------------------------------------------------------------ what a request is for


@pytest.mark.parametrize(
    ("path", "array"),
    [
        ("hgdp1kgp-v2-EUR-k10.store/near/22/r/c/3", "near/r"),
        ("/chr22-A.store/near/22/indptr/c/0", "near/indptr"),
        ("/s.store/index/7/partner/c/12", "index/partner"),
        ("index/X/r/c/1", "index/r"),
        ("variants/alid/c/12", "variants/alid"),
        ("s.store/variants/a1_frequency/c/0", "variants/a1_frequency"),
        ("s.store/near/22/r/zarr.json", "metadata"),
        ("s.store/zarr.json", "metadata"),
        ("s.store/.zgroup", "metadata"),
        ("/.zmetadata", "metadata"),
    ],
)
def test_a_request_is_counted_against_its_array(path: str, array: str) -> None:
    assert http_static.array_of(path) == array


def test_counts_add_up_per_array_and_reset_to_nothing() -> None:
    counts = http_static.Counts()
    counts.add("s/near/22/r/c/0", 100)
    counts.add("s/near/21/r/c/4", 50)
    counts.add("s/index/22/r/c/0", 7)
    counts.add("s/zarr.json", 3)
    assert counts.as_json() == {
        "requests": 4,
        "bytes": 160,
        "bytes_by_array": {"near/r": 150, "index/r": 7, "metadata": 3},
        "requests_by_array": {"near/r": 2, "index/r": 1, "metadata": 1},
    }
    counts.reset()
    assert counts.as_json() == {
        "requests": 0,
        "bytes": 0,
        "bytes_by_array": {},
        "requests_by_array": {},
    }


# ------------------------------------------------------------------ the static host and its proxy


def test_the_host_honours_byte_ranges_through_the_proxy(tmp_path: Path) -> None:
    data = bytes(range(256)) * 4
    (tmp_path / "shard").write_bytes(data)
    with served(tmp_path) as url:
        assert fetch(f"{url}/shard", "10-19") == (206, data[10:20])
        assert fetch(f"{url}/shard", "-4") == (206, data[-4:])
        assert fetch(f"{url}/shard") == (200, data)
        stats = control(url, "stats")
    assert stats["requests"] == 3
    assert stats["bytes"] == 10 + 4 + len(data), "the proxy counts the bodies it passes on"


def test_the_proxy_waits_once_per_request_and_requests_in_flight_wait_together(
    tmp_path: Path,
) -> None:
    # The round-trip model (docs/benchmarks/latency.md) is the 0 ms time plus one wait for each
    # request made in sequence; a wait per byte, per connection or per pair of requests breaks it.
    (tmp_path / "chunk").write_bytes(b"x" * 64)
    wait = 0.15
    with served(tmp_path, wait * 1000) as url:
        start = time.perf_counter()
        for _ in range(3):
            assert fetch(f"{url}/chunk", "0-7")[1] == b"x" * 8
        sequential = time.perf_counter() - start
        start = time.perf_counter()
        with ThreadPoolExecutor(6) as pool:
            list(pool.map(lambda _: fetch(f"{url}/chunk", "0-7"), range(6)))
        together = time.perf_counter() - start
    # The lower bounds are exact, since the proxy sleeps. The upper ones only tell one wait from
    # two, and waiting together from waiting in turn, with room for a loaded node.
    assert 3 * wait <= sequential < 5 * wait, f"3 requests in turn took {sequential:.3f} s"
    assert wait <= together < 3 * wait, f"6 requests at once took {together:.3f} s"


# ------------------------------------------------------------------ the client's counts


@pytest.fixture
def instrumented(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The benchmark's counting, put in and taken out again: `instrument` patches classes."""
    monkeypatch.setattr(latency, "TALLY", latency.Tally())
    latency.instrument(None, None, patch=monkeypatch.setattr)
    return latency


def same_counts(client: dict[str, Any], wire: dict[str, Any]) -> None:
    """Every array's requests and bytes the client counted are the proxy's. The one exception is
    the bodies of the 404s zarr's open probes for (`.zgroup` and the like): the client counts a
    missing document as no bytes, the proxy the error page it carried."""
    arrays = set(client["gets"]) | set(wire["requests_by_array"])
    for name in arrays:
        gets, sent = client["gets"].get(name, 0), wire["requests_by_array"].get(name, 0)
        assert gets == sent, f"{name}: the client made {gets} reads, the proxy saw {sent}"
        if name != "metadata":
            got, passed = client["get_bytes"].get(name, 0), wire["bytes_by_array"].get(name, 0)
            assert got == passed, f"{name}: the client read {got} bytes, the proxy passed {passed}"


def test_the_client_counts_what_crossed_the_proxy(instrumented: Any) -> None:
    rows = np.arange(0, 400, 7)
    alids = [str(a) for a in instrumented.opened(TEST_STORE)._reader.alids_of(rows)]
    instrumented.TALLY = instrumented.Tally()
    with served(TEST_STORE) as url:
        control(url, "reset")
        store = instrumented.opened(f"{url}/")
        store.matrix(alids)
        store.proxies(alids[:5], r2=0.8, kb=10_000)
        wire = control(url, "stats")
    reached = set(wire["requests_by_array"])
    assert {"near/r", "index/r", "variants/alid"} <= reached, "the reads must reach these arrays"
    same_counts(instrumented.TALLY.as_json(), wire)


def committed_records() -> list[dict[str, Any]]:
    found = []
    for path in sorted(RESULTS.glob("*.jsonl.gz")):
        lines = gzip.decompress(path.read_bytes()).decode().splitlines()
        found += [json.loads(line) for line in lines if '"kind": "environment"' not in line]
    return found


def test_every_committed_http_record_counts_what_the_proxy_saw() -> None:
    over_http = [r for r in committed_records() if r["wire"] is not None]
    assert len(over_http) > 2000, "the committed results must hold the HTTP runs"
    near = [r for r in over_http if r["wire"]["requests_by_array"].get("near/r")]
    assert len(near) > 1000, "most must have read the Near Band for this to mean anything"
    for record in over_http:
        same_counts(record["client"], record["wire"])


# ------------------------------------------------------------------ the report's records


def run_record(run: str, **read: Any) -> dict[str, Any]:
    fields = {"step": "step2", "reader": "ogld", "target": "http10", "store": "chr22-C.store"}
    return {**fields, "case": "scattered", "state": "cold", "seed": 8, **read, "run": run}


@pytest.mark.parametrize(
    ("field", "other"),
    [
        ("step", "step1"),
        ("reader", "concurrent"),
        ("target", "http50"),
        ("store", "chr22-B.store"),
        ("case", "pair"),
        ("state", "warm"),
        ("seed", 9),
        ("r2", 0.001),
        ("analysis", "OGS-00011__GCST90029004"),
        ("width_mb", 10),
        ("centre", 32_000_000),
        ("chrom", "genome"),
        ("variants", 100),
        ("pair", 3),
    ],
)
def test_a_later_run_supersedes_only_the_same_read(field: str, other: Any) -> None:
    first = run_record("2026-11-12T15:00:00Z")
    again = run_record("2026-11-12T22:00:00Z")
    different = {**run_record("2026-11-12T22:00:00Z"), field: other}
    assert report.latest([first, again]) == [again]
    kept = report.latest([first, different])
    assert kept == [first, different], f"a read with another {field} is another read"


def test_every_field_a_record_names_its_read_by_is_in_the_report_s_key() -> None:
    # A field the benchmark writes that the report does not key on makes two reads one, and
    # the report keeps only the later run's.
    named = set().union(*(set(r) for r in committed_records())) - MEASURED
    assert named - {"cache_bytes"} <= set(report.SAME_READ)


def test_the_tables_are_the_committed_records_rendered(capsys: pytest.CaptureFixture[str]) -> None:
    paths = sorted(str(p) for p in RESULTS.glob("*.jsonl.gz"))
    monkey_argv = ["report.py", *paths, "--layouts", str(RESULTS)]
    old, sys.argv = sys.argv, monkey_argv
    try:
        report.main()
    finally:
        sys.argv = old
    rendered = capsys.readouterr().out
    committed = (RESULTS.parent / "latency-tables.md").read_text()
    assert rendered == committed, "re-render latency-tables.md from results/ (report.py)"


# ------------------------------------------------------------------ what a run is told and records


def test_an_unknown_case_is_refused_not_dropped() -> None:
    args = argparse.Namespace(plan="step1", case=["clump", "regions"], analyses=None)
    with pytest.raises(SystemExit, match="regions"):
        latency.checked(args)


def test_an_unknown_analysis_is_refused_not_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("OGS-1__a", "OGS-1__b"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("OGLD_403_ANALYSES", str(tmp_path))
    assert latency.analyses(["OGS-1__b"]) == [tmp_path / "OGS-1__b"]
    args = argparse.Namespace(plan="step2", case=None, analyses=["OGS-1__a", "OGS-1__z"])
    with pytest.raises(SystemExit, match="OGS-1__z"):
        latency.checked(args)


def test_a_record_names_the_budget_of_the_cache_the_store_read_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ogld.store.location import ChunkCache

    latency.instrument(None, 3, patch=monkeypatch.setattr)
    assert latency.opened(TEST_STORE)._reader._cache._budget == 3 << 20, "--cache-mib sets it"
    # A budget the module's default does not hold: the record must read the cache's own.
    store = latency.opened(TEST_STORE)
    store._reader._cache = ChunkCache(budget=5 << 20)
    out = io.StringIO()
    bench = latency.Bench(TEST_STORE.parent, latency.Target("local", None), None, out, "s", "ogld")
    record = bench.timed(lambda: {}, lambda: store, case="c")
    assert record["cache_bytes"] == 5 << 20
    assert json.loads(out.getvalue())["cache_bytes"] == 5 << 20


@pytest.mark.parametrize(
    ("problem", "clean"), [("tracked files differ from HEAD", False), ("", True)]
)
def test_a_run_records_whether_its_checkout_is_the_code_that_ran(
    monkeypatch: pytest.MonkeyPatch, problem: str, clean: bool
) -> None:
    asked: list[tuple[Path, str]] = []

    def judged(root: Path, installed: str) -> str:
        asked.append((root, installed))
        return problem

    monkeypatch.setattr(latency.checkout, "problem", judged)
    code = latency.code_state("0.0.1.dev1+gabc")
    assert asked == [(SCRIPTS.parent, "0.0.1.dev1+gabc")]
    assert (code["clean"], code["problem"]) == (clean, problem)
    assert len(code["commit"]) == 40


def test_every_lever_off_reads_as_before_ogld_32_and_answers_the_same(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `--without` measures ogld#32's levers one at a time against the reader ogld#8 measured:
    # each lever off is that reader's code, and no lever changes an answer.
    from ogld.store import location

    rows = np.arange(0, 1056, 9)
    alids = [str(a) for a in latency.opened(TEST_STORE)._reader.alids_of(rows)]
    on = latency.opened(TEST_STORE).matrix(alids).r
    assert latency.without([], None, patch=monkeypatch.setattr) is None, "ogld's default"
    assert latency.without(["cache"], 7, patch=monkeypatch.setattr) == 7, "the asked budget"
    budget = latency.without(list(latency.LEVERS), None, patch=monkeypatch.setattr)
    assert budget == 256, "the budget before ogld#32"
    assert location.ChunkCache._chunks is latency._sequential_chunks
    assert location._KeptIndexes.get is latency._index_passed
    assert location._KeptIndexes.get_sync is latency._index_passed_sync
    assert location._counted is latency._counted_per_string
    off = latency.opened(TEST_STORE).matrix(alids).r
    assert np.isnan(on).any(), "the matrix must hold unstored pairs"
    assert np.isfinite(on).sum() > len(rows), "and stored ones beside the diagonal"
    np.testing.assert_array_equal(off, on)
