"""A test marked `realdata` or `acceptance` outside its directory is refused (CONTRIBUTING.md).

Both marks are deselected by the default tier, and the directories' conftests add them by
location, so a test that carries one anywhere else runs in no tier. `tests/tier_markers.py`
finds such a test and `tests/conftest.py` makes it a collection error."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.ld.tier_markers import misplaced, refuse_misplaced, unasked

ROOT = Path(__file__).resolve().parent


def item(path: str, *marks: str) -> SimpleNamespace:
    return SimpleNamespace(
        path=ROOT / path,
        nodeid=path,
        get_closest_marker=lambda name: name if name in marks else None,
        iter_markers=lambda: iter(SimpleNamespace(name=name) for name in marks),
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


def _dropped(expression: str) -> list[str]:
    items = [item("realdata/a.py", "ld", "realdata"), item("acceptance/b.py", "ld", "acceptance")]
    items.append(item("test_c.py", "ld"))
    return [i.nodeid for i in unasked(items, expression)]


# What the hook takes out. An item it keeps may still be deselected by pytest's own `-m`.
BOTH = ["realdata/a.py", "acceptance/b.py"]


@pytest.mark.parametrize(
    ("expression", "dropped"),
    [
        ("", BOTH),
        ("ld", BOTH),
        ("ld and not slow", BOTH),
        ("ld or not acceptance", BOTH),  # the old rule brought acceptance back
        ("ld or realdata", BOTH),  # `ld` already selects realdata: not asked for
        ("not ld", []),  # nothing for the hook: the expression itself drops them
        ("not acceptance", BOTH),  # realdata is not asked for either
        ("realdata", []),  # the expression drops the others
        ("acceptance", []),
        ("realdata or acceptance", []),
    ],
)
def test_a_tier_is_kept_only_when_the_run_s_m_needs_it(expression: str, dropped: list[str]) -> None:
    assert _dropped(expression) == dropped


def _collect(*arguments: str) -> str:
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/ld/realdata", "-q", "-p", "no:cacheprovider",
         "--collect-only", *arguments],
        cwd=ROOT.parent.parent, capture_output=True, text=True, check=False,
    )  # fmt: skip
    return done.stdout


@pytest.mark.timeout(60)  # fresh pytest processes importing the engine
def test_a_command_line_m_cannot_bring_the_opt_in_tiers_back_but_can_ask_for_one() -> None:
    import re

    assert re.search(r"no tests collected \(\d+ deselected\)", _collect("-m", "ld")), "ld"
    selected = _collect("-m", "realdata")
    assert "test_" in selected and "deselected" not in selected, selected[-400:]
