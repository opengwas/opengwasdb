"""`validate --n-workers N` reports what one process does on #387's small release, its
EAF exception table paged and corrupted (#315).

Its eight cases are `worker_passes`' on the small seed, apart from the others' so that
pytest's workers run them side by side.
"""

from __future__ import annotations

from pathlib import Path

from worker_passes import Case, assert_the_same_report, cases_of, worker_floor_pinned

case = cases_of("small_seed")
_pinned = worker_floor_pinned()


def test_the_report_is_the_same_at_one_and_several_workers(case: tuple[Path, Case]) -> None:
    assert_the_same_report(*case)
