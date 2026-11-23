"""A test marked `realdata` or `acceptance` outside its directory is refused (CONTRIBUTING.md).

Both marks are deselected by the default tier, and the directories' conftests add them by
location, so a test that carries one anywhere else runs in no tier. `tests/ld/tier_markers.py`
finds such a test and `tests/conftest.py` makes it a collection error."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from tier_markers import misplaced, refuse_misplaced

ROOT = Path(__file__).resolve().parent


def item(path: str, *marks: str) -> SimpleNamespace:
    return SimpleNamespace(
        path=ROOT / path,
        nodeid=path,
        get_closest_marker=lambda name: name if name in marks else None,
    )


def test_a_test_in_its_own_directory_carries_its_mark_without_complaint() -> None:
    items = [item("realdata/test_a.py", "realdata"), item("acceptance/test_b.py", "acceptance")]
    assert misplaced(items, ROOT) == []


def test_a_test_with_a_mark_outside_its_directory_is_named() -> None:
    items = [item("test_x.py", "realdata"), item("contract/test_y.py", "acceptance")]
    assert misplaced(items, ROOT) == [
        "test_x.py is marked realdata but is not under tests/realdata/",
        "contract/test_y.py is marked acceptance but is not under tests/acceptance/",
    ]


def test_a_mark_under_the_other_tier_s_directory_is_named_too() -> None:
    assert misplaced([item("realdata/test_a.py", "acceptance")], ROOT) == [
        "realdata/test_a.py is marked acceptance but is not under tests/acceptance/"
    ]


def test_an_unmarked_test_is_none_of_its_business() -> None:
    assert misplaced([item("test_x.py"), item("realdata/test_q.py")], ROOT) == []


def test_a_misplaced_test_is_a_collection_error_naming_it() -> None:
    with pytest.raises(pytest.UsageError, match="not under tests/realdata/"):
        refuse_misplaced([item("test_x.py", "realdata")], ROOT)
