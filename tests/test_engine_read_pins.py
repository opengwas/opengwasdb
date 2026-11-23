"""Validation and Reference Completion give what they gave before #473 moved their reads.

A migration check, opt-in: a run collects it only with ``OGDB_MIGRATION_PINS=1``
(`tests/conftest.py` deselects it otherwise, as it does the opt-in tiers). The pins
are what the code at `3b1c0c30` -- before the validator, completion, the top-hit
harvest, the overview and Rho read Store Releases through the Store Reader --
reported and wrote (`engine_read_pins`), so once the fixtures change format
(#467 stamps 0.3.0) or a finding is meant to change, they stop describing anything
but that move. The epic's owner runs them when a change to the read path merges,
and they are removed when epic #450 ends.

`tests/golden/engine_read_pins.json` covers a Dense, a Ragged and a Hybrid release
pair; copies corrupted where the moved reads look, each of which must fail the rule
it was corrupted for; a Dense release validated against its source; the EAF
orientation audit; and the observed view, passing and refused. A difference names
the release and what differs.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from engine_read_pins import CORRUPTIONS, GOLDEN, REFUSED_VIEWS, SOURCED, pins

PINS: dict[str, Any] = json.loads(GOLDEN.read_text(encoding="utf-8"))
LAYOUTS = ("dense", "hybrid", "ragged")


@pytest.fixture(scope="module")
def found(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return pins(tmp_path_factory.mktemp("engine-read-pins"))


def _errors(findings: list[dict[str, str]]) -> set[str]:
    return {f["rule"] for f in findings if f["severity"] == "error"}


def test_each_corrupted_copy_fails_the_rule_it_was_corrupted_for():
    """Each corruption was caught, by its own rule, when pinned, and each clean release
    passed: a pin of releases no rule fails would hold whatever a rule stopped reading."""
    findings = PINS["findings"]
    assert sorted(findings) == sorted(
        [f"{layout}_{state}" for layout in LAYOUTS for state in ("observed", "completed")]
        + list(CORRUPTIONS)
    )
    for name, recorded in findings.items():
        if name in CORRUPTIONS:
            assert CORRUPTIONS[name][2] in _errors(recorded), name
        else:
            assert not _errors(recorded), name
            assert recorded, name  # the orientation warnings, at least
    for name, (_corrupt, rule) in SOURCED.items():
        recorded = PINS["sourced_findings"][name]
        skipped = {f["rule"] for f in recorded if f["severity"] == "skip"}
        assert "fidelity.source" not in skipped, name  # it judged the release
        assert _errors(recorded) == ({rule} if rule else set()), name


def test_the_other_pins_are_worth_comparing():
    for layout in LAYOUTS:
        assert PINS["observed_view"][layout]["passed"] is True, layout
        # Every array, group and table of a completed release, its component's too.
        assert len(PINS["completed"][layout]) > 90, layout
    for name in REFUSED_VIEWS:
        assert "completion_quality" in PINS["observed_view"][name]["refused"], name
    audits = PINS["eaf_audit"]
    assert (audits["oriented"]["ok"], audits["flipped"]["ok"]) == (True, False)


def test_every_finding_matches_the_code_before_the_move(found: dict[str, Any]):
    for section in ("findings", "sourced_findings"):
        differing = {
            name: {"before": pinned, "after": found[section].get(name)}
            for name, pinned in PINS[section].items()
            if found[section].get(name) != pinned
        }
        assert differing == {}, section


def test_every_observed_view_verdict_matches_the_code_before_the_move(found: dict[str, Any]):
    assert found["observed_view"] == PINS["observed_view"]


def test_the_eaf_audit_matches_the_code_before_the_move(found: dict[str, Any]):
    assert found["eaf_audit"] == PINS["eaf_audit"]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_completed_array_matches_the_code_before_the_move(
    found: dict[str, Any], layout: str
):
    pinned, now = PINS["completed"][layout], found["completed"][layout]
    differing = sorted(key for key in set(pinned) | set(now) if pinned.get(key) != now.get(key))
    assert differing == []
