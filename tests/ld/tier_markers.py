"""The test tiers: which collected tests carry a tier mark outside the tier's directory
(tests/ld/conftest.py), and which opt-in tests a run's `-m` did not ask for (tests/conftest.py)."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

#: Each tier mark and the directory, under tests/ld/, its conftest adds it from. `ld` is the area:
#: every test under tests/ld/, the opt-in tiers included.
TIER_DIRECTORIES = {"ld": ".", "realdata": "realdata", "acceptance": "acceptance"}

#: The tiers a run has to ask for by name. They read the HGDP+1kGP panels or run for hours.
OPT_IN_TIERS = ("realdata", "acceptance")


def asked_for(markexpr: str, mark: str) -> bool:
    """Whether a `-m` expression names `mark`. `-m "not realdata"` names it too: the expression
    itself then deselects the tier, so only the default (no `-m`) needs this to say no."""
    return re.search(rf"\b{re.escape(mark)}\b", markexpr) is not None


def unasked(items: Iterable[Any], markexpr: str) -> list[Any]:
    """The items in an opt-in tier whose mark the run's `-m` expression does not name.

    pytest lets a command line's `-m` replace an `addopts` one, so `-m ld` would bring `realdata`
    back (every realdata test is also an `ld` test). The deselection is therefore made here, from
    what the expression names, and holds whatever `-m` the run is given."""
    return [
        item
        for item in items
        if any(
            item.get_closest_marker(tier) and not asked_for(markexpr, tier) for tier in OPT_IN_TIERS
        )
    ]


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
