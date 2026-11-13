"""`validate --n-workers N` reports what one process does on #387's rich release and
#314's completed and varied ones (#315).

The cases are `worker_passes`'; this file runs those built on these three seeds, so
pytest's workers run it beside the files that run the others.
"""

from __future__ import annotations

from pathlib import Path

from worker_passes import Case, assert_the_same_report, cases_of, worker_floor_pinned

_pinned = worker_floor_pinned()
case = cases_of("rich_seed", "completed_seed", "varied_seed")


def test_the_report_is_the_same_at_one_and_several_workers(case: tuple[Path, Case]) -> None:
    assert_the_same_report(*case)
