"""The host-memory reading and the pool it sizes (#320).

Two rules live here because more than one pass needs them: what the host says it
has available, and how many workers fit in a quarter of that. The reading is
strict on purpose -- a malformed `/proc/meminfo` is `None`, not arithmetic on a
number nobody meant -- because the value it carries decides how much of a shared
host a phase may take.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opengwasdb.encoding.memory import (
    MEMORY_BUDGET_DIVISOR,
    available_memory_bytes,
    memory_budget_bytes,
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
