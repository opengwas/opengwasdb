"""Which collected tests carry a tier mark outside the tier's directory (tests/conftest.py)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

#: Each tier mark and the directory its conftest adds it from.
TIER_DIRECTORIES = {"realdata": "realdata", "acceptance": "acceptance"}


def misplaced(items: Iterable[Any], root: Path) -> list[str]:
    """One line for each collected item marked with a tier outside that tier's directory under
    `root`: such a test is deselected by the default tier and run by no other."""
    found = []
    for item in items:
        for mark, directory in TIER_DIRECTORIES.items():
            home = (root / directory).resolve()
            if item.get_closest_marker(mark) and home not in Path(item.path).resolve().parents:
                found.append(f"{item.nodeid} is marked {mark} but is not under tests/{directory}/")
    return found


def refuse_misplaced(items: Iterable[Any], root: Path) -> None:
    """Make a misplaced test a collection error, naming each (tests/conftest.py calls this)."""
    found = misplaced(items, root)
    if found:
        raise pytest.UsageError("\n".join(found))
