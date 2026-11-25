"""The host-memory reading and the pool it sizes (#320).

Two rules live here because more than one pass needs them: what the host says it
has available, and how many workers fit in a quarter of that. The reading is
strict on purpose -- a malformed `/proc/meminfo` is `None`, not arithmetic on a
number nobody meant -- because the value it carries decides how much of a shared
host a phase may take.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import opengwasdb.build.ordered_pool as ordered_pool
import opengwasdb.encoding.memory as memory
from opengwasdb.encoding.memory import (
    MEMORY_BUDGET_DIVISOR,
    available_memory_bytes,
    memory_budget_bytes,
    pool_workers,
    release_freed_heap,
    resident_bytes,
    workers_within_memory,
)

_MEMINFO = "MemTotal:       1056123456 kB\nMemAvailable:    900000000 kB\n"


def _meminfo(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "meminfo"
    path.write_text(body, encoding="ascii")
    return path


def test_the_reading_is_the_available_field_in_bytes(tmp_path: Path) -> None:
    assert available_memory_bytes(_meminfo(tmp_path, _MEMINFO)) == 900_000_000 * 1024
    # The total is not the available: a phase that read the wrong field would
    # plan against memory other jobs already hold.
    assert available_memory_bytes(_meminfo(tmp_path, _MEMINFO)) != 1_056_123_456 * 1024


@pytest.mark.parametrize(
    "body",
    [
        "MemAvailable: -5 kB\n",
        "MemAvailable: 5_000 kB\n",
        "MemAvailable: 900000000\n",
        "MemAvailable: 900000000 MB\n",
        "MemAvailable: 900000000 kB extra\n",
        "MemAvailable:\n",
        "MemTotal: 1056123456 kB\n",
        "",
    ],
    ids=[
        "negative",
        "underscore",
        "no-unit",
        "wrong-unit",
        "extra-field",
        "no-value",
        "no-field",
        "empty",
    ],
)
def test_anything_but_a_whole_number_of_kilobytes_is_unreadable(
    tmp_path: Path, body: str
) -> None:
    assert available_memory_bytes(_meminfo(tmp_path, body)) is None


def test_a_missing_file_is_unreadable(tmp_path: Path) -> None:
    assert available_memory_bytes(tmp_path / "absent") is None


def test_the_budget_is_a_quarter_of_what_is_available() -> None:
    assert MEMORY_BUDGET_DIVISOR == 4
    assert memory_budget_bytes(400 * 1024**3) == 100 * 1024**3
    assert memory_budget_bytes(None) is None


def test_workers_are_capped_by_the_budget_and_never_zero() -> None:
    per_band = 100
    # The build's own count is the ceiling.
    assert workers_within_memory(8, per_band, 10 * per_band) == 8
    # The budget is the ceiling when it is smaller.
    assert workers_within_memory(8, per_band, 3 * per_band) == 3
    # A budget too small for one band is one worker, not zero: the band is the
    # unit of the pass, and refusing to run it at all is not a memory policy.
    assert workers_within_memory(8, per_band, per_band - 1) == 1
    assert workers_within_memory(8, per_band, 0) == 1
    # An unknown budget cannot be planned against.
    assert workers_within_memory(8, per_band, None) == 1


# ── the completion write pools' count (#321) ─────────────────────────────────

GIB = 1024**3


def _host(
    monkeypatch: pytest.MonkeyPatch, available: int | None, parent: int, floor: int = 0
) -> None:
    """A host with `available` bytes available, a parent holding `parent`, and a
    forked worker's floor of `floor` -- none, so the arithmetic below is the
    unit's alone, unless a test is about the floor."""
    monkeypatch.setattr(memory, "available_memory_bytes", lambda path=memory.MEMINFO: available)
    monkeypatch.setattr(memory, "resident_bytes", lambda path=memory.SELF_STATUS: parent)
    monkeypatch.setattr(memory, "FORK_FLOOR_BYTES", floor)


def _pool(
    caplog: pytest.LogCaptureFixture,
    requested: int,
    unit: int,
    *,
    units: int = 1000,
    result_bytes: int = 0,
    **kwargs,
) -> int:
    with caplog.at_level(logging.INFO, logger=memory.log.name):
        return pool_workers(
            requested, unit, what="Band write", unit="band", units=units,
            result_bytes=result_bytes, **kwargs,
        )


def test_the_pool_is_what_the_budget_holds_beside_the_parent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """16 GiB available is a 4 GiB budget; the parent holds 1 GiB of it, so
    three 1 GiB workers fit of the eight asked for, and the line says so."""
    _host(monkeypatch, 16 * GIB, 1 * GIB)

    assert _pool(caplog, 8, 1 * GIB) == 3
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "Band write: 3 of 8 worker(s)" in caplog.text
    assert "beside the parent's 1024 MiB, against a budget of 4096 MiB" in caplog.text


def test_every_worker_carries_the_fork_s_floor(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A forked worker holds its interpreter and the parent's pages it writes to
    before its first unit, whatever the pool: 16 GiB available is a 4 GiB budget,
    and beside a 1 GiB parent a 1 GiB unit and the 512 MiB floor fit twice, not
    three times. Every pool counts the same floor, from one definition."""
    _host(monkeypatch, 16 * GIB, 1 * GIB, floor=memory.FORK_FLOOR_BYTES)

    assert memory.FORK_FLOOR_BYTES == 512 * 1024**2
    assert memory.worker_bytes(1 * GIB) == 1 * GIB + memory.FORK_FLOOR_BYTES
    assert _pool(caplog, 8, 1 * GIB) == 2
    assert "a band is 1536 MiB a worker" in caplog.text


def test_the_results_a_worker_leaves_with_the_parent_are_counted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The ordered pool's parent holds two results a worker until it takes
    them: 256 MiB results make each 1 GiB worker cost 1.5 GiB, so two fit
    beside a 1 GiB parent in a 4 GiB budget, not three."""
    _host(monkeypatch, 16 * GIB, 1 * GIB)

    assert memory.RESULTS_HELD_PER_WORKER == 2
    # The ordered pool's window is the share counted here: one constant.
    assert memory.RESULTS_HELD_PER_WORKER is ordered_pool.RESULTS_HELD_PER_WORKER
    assert _pool(caplog, 8, 1 * GIB, result_bytes=GIB // 4) == 2
    assert "512 MiB of results held by the parent" in caplog.text


def test_no_more_workers_than_units(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Three bands at --n-workers 16 fork three workers, and the line says so."""
    _host(monkeypatch, 1024 * GIB, 0)

    assert _pool(caplog, 16, 1 * GIB, units=3) == 3
    assert "3 of 3 worker(s)" in caplog.text
    assert _pool(caplog, 16, 1 * GIB, units=1) == 1


def test_the_parent_given_is_counted_rather_than_its_resident_set(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A parent that will grow (a Ragged writer filling) is counted at its peak."""
    _host(monkeypatch, 16 * GIB, 0)

    assert _pool(caplog, 8, 1 * GIB) == 4
    assert _pool(caplog, 8, 1 * GIB, parent_bytes=2 * GIB) == 2


def test_a_host_that_cannot_be_read_gets_one_worker_and_says_why(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _host(monkeypatch, None, 0)

    assert _pool(caplog, 8, 1) == 1
    assert "cannot read /proc/meminfo" in caplog.text
    assert "1 of 8 worker(s)" in caplog.text


def test_a_budget_the_parent_and_one_worker_overrun_is_one_worker_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _host(monkeypatch, 16 * GIB, 3 * GIB + GIB // 2)

    assert _pool(caplog, 8, 1 * GIB) == 1
    assert "which will overrun it" in caplog.text


def test_one_worker_asked_for_reads_nothing_and_says_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def unread(path: Path = memory.MEMINFO) -> int:
        raise AssertionError("one worker needs no reading")

    monkeypatch.setattr(memory, "available_memory_bytes", unread)
    assert _pool(caplog, 1, 1 * GIB) == 1
    assert caplog.text == ""


def test_the_resident_set_is_vmrss_in_bytes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    status = tmp_path / "status"
    status.write_text("VmPeak:\t 9999 kB\nVmRSS:\t   2048 kB\nVmHWM:\t 4096 kB\n")
    assert resident_bytes(status) == 2048 * 1024
    assert resident_bytes() > 0, "this process's own status is readable"
    assert caplog.text == ""


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ("VmRSS:\t   lots\n", "VmRSS is 'lots', not a whole number of kB"),
        ("VmRSS:\t   2048 MB\n", "VmRSS is '2048 MB', not a whole number of kB"),
        ("VmPeak:\t 9999 kB\n", "it has no VmRSS"),
        (None, "No such file"),
    ],
    ids=["malformed", "wrong-unit", "no-field", "missing"],
)
def test_an_unreadable_resident_set_is_refused_not_counted_as_nothing(
    tmp_path: Path, body: str | None, why: str
) -> None:
    """A parent counted as holding nothing sizes its pool for memory it is using.

    Linux always publishes `VmRSS`, so a status file without it is not a host to
    plan against: the reading fails and names the file and what it found.
    """
    status = tmp_path / "status"
    if body is not None:
        status.write_text(body)
    with pytest.raises(RuntimeError, match="cannot read this process's resident set") as raised:
        resident_bytes(status)
    assert str(status) in str(raised.value)
    assert why in str(raised.value)


_MiB = 1024**2


def test_freed_heap_kept_by_the_allocator_is_handed_back() -> None:
    """A long-lived worker holds the high-water of every Analysis it has read
    (#451): the reader's buffers are freed, but glibc keeps the pages, so each
    worker of a 240-Analysis Pass 2 stepped up 4 GiB a large source and stayed
    there. Freed pages under a live allocation cannot be returned by shrinking
    the heap, and this is that shape: 200 MiB of 1 kB chunks, with one in 64
    kept alive so the heap cannot shrink from its top."""
    before = resident_bytes()
    piece = b"x"
    chunks = [piece * 1000 for _ in range(200_000)]
    allocated = resident_bytes()
    assert allocated - before > 150 * _MiB, "the fixture's chunks must be resident"
    kept = chunks[::64]
    del chunks
    freed = resident_bytes()
    assert allocated - freed < 50 * _MiB, (
        "the allocator must keep the freed chunks, or there is nothing to hand back"
    )

    release_freed_heap()
    assert freed - resident_bytes() > 100 * _MiB
    del kept  # held to here, so the freed chunks lay under live ones
