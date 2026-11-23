"""The test tiers: which collected tests carry a tier mark outside the tier's directory
(tests/ld/conftest.py), and which opt-in tests a run's `-m` did not ask for (tests/conftest.py)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
from _pytest.mark.expression import Expression

#: Each tier mark and the directory, under tests/ld/, its conftest adds it from. `ld` is the area:
#: every test under tests/ld/, the opt-in tiers included.
TIER_DIRECTORIES = {"ld": ".", "realdata": "realdata", "acceptance": "acceptance"}

#: The tiers a run has to ask for by name. They read the HGDP+1kGP panels or run for hours.
OPT_IN_TIERS = ("realdata", "acceptance")


def unasked(items: Iterable[Any], markexpr: str) -> list[Any]:
    """The items in an opt-in tier that the run's `-m` expression did not ask that tier for.

    A tier is asked for when the expression needs it: evaluate the expression on the item's marks
    with the tier's mark removed, and if it still selects the item, something else selects it and
    the tier was not asked for. So no `-m`, `-m ld`, `-m "not ld"` and `-m "ld or not acceptance"`
    leave the opt-in tiers out; `-m realdata` and `-m acceptance` bring their own tier in.
    `-m "ld or realdata"` leaves realdata out, because `ld` already selects every realdata test.

    pytest lets a command line's `-m` replace an `addopts` one, so the default cannot live there:
    `-m ld` would bring `realdata` back (every realdata test is also an `ld` test)."""
    if not markexpr.strip():  # no `-m`: everything is selected, so nothing asked for a tier
        expression = None
    else:
        expression = Expression.compile(markexpr)
    dropped = []
    for item in items:
        marks = {mark.name for mark in item.iter_markers()}
        for tier in OPT_IN_TIERS:
            if tier in marks and (
                expression is None
                or expression.evaluate(lambda name, kept=marks - {tier}: name in kept)
            ):
                dropped.append(item)
                break
    return dropped


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
