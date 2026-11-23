"""Validation and Reference Completion give what they gave before #473 moved their reads.

`tests/golden/engine_read_pins.json` is what the code at `3b1c0c30` -- before
the validator, completion, the top-hit harvest, the overview and Rho read Store
Releases through the Store Reader -- reported and wrote on a Dense, a Ragged and
a Hybrid release pair, and on copies corrupted where the moved reads look
(`engine_read_pins`). The pins were written by the old code, never by this one;
a difference names the release and what differs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from engine_read_pins import CORRUPTIONS, GOLDEN, pins

PINS: dict[str, Any] = json.loads(GOLDEN.read_text(encoding="utf-8"))
LAYOUTS = ("dense", "hybrid", "ragged")


@pytest.fixture(scope="module")
def found(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return pins(tmp_path_factory.mktemp("engine-read-pins"))


def _errors(findings: list[dict[str, str]]) -> set[str]:
    return {f["rule"] for f in findings if f["severity"] == "error"}


def test_the_pins_are_worth_comparing():
    """Each corruption was caught when pinned and each clean release passed: a pin
    of releases no rule fails would hold whatever a rule stopped reading."""
    findings = PINS["findings"]
    assert sorted(findings) == sorted(
        [f"{layout}_{state}" for layout in LAYOUTS for state in ("observed", "completed")]
        + list(CORRUPTIONS)
    )
    for name, recorded in findings.items():
        if name in CORRUPTIONS:
            assert _errors(recorded), name
        else:
            assert not _errors(recorded), name
            assert recorded, name  # the orientation warnings, at least
    for layout in LAYOUTS:
        assert PINS["observed_view"][layout]["passed"] is True, layout
        # Every array, group and table of a completed release, its component's too.
        assert len(PINS["completed"][layout]) > 90, layout


def test_every_finding_matches_the_code_before_the_move(found: dict[str, Any]):
    differing = {
        name: {"before": pinned, "after": found["findings"].get(name)}
        for name, pinned in PINS["findings"].items()
        if found["findings"].get(name) != pinned
    }
    assert differing == {}


def test_every_observed_view_verdict_matches_the_code_before_the_move(found: dict[str, Any]):
    assert found["observed_view"] == PINS["observed_view"]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_completed_array_matches_the_code_before_the_move(
    found: dict[str, Any], layout: str
):
    pinned, now = PINS["completed"][layout], found["completed"][layout]
    differing = sorted(key for key in set(pinned) | set(now) if pinned.get(key) != now.get(key))
    assert differing == []


def test_the_pins_were_written_by_the_code_before_the_move():
    assert PINS["__written_by__"] == "tests/engine_read_pins.py at 3b1c0c30"
    assert Path(GOLDEN).stat().st_size > 10_000
