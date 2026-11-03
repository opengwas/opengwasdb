"""The Validation Rule registry against the contract that names it (issue #55, ADR 0074).

Spec §20.0 lists every rule by name, in the order the runner runs them, which is
the order findings are reported in. A rule added, renamed, removed or moved
without the spec moving with it fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

from opengwasdb.validation import RULES
from opengwasdb.validation.rule import Fold, Kind, Severity

SPEC = Path(__file__).resolve().parents[1] / "docs" / "spec" / "store-format.md"

#: One row of §20.0's table: `| \\`name\\` | judges | kind | what it checks |`.
_ROW = re.compile(r"^\| `(?P<name>[a-z_.]+)` \| (?P<judges>[^|]+) \| (?P<kind>[^|]+) \|")


def _table() -> list[re.Match[str]]:
    text = SPEC.read_text(encoding="utf-8")
    section = text.split("### 20.0 Validation Rules", 1)[1].split("\n### ", 1)[0]
    rows = [m for line in section.splitlines() if (m := _ROW.match(line))]
    assert len(rows) > 50, "the section must hold the rule table"
    return rows


def test_the_spec_names_every_rule_in_the_order_it_runs() -> None:
    assert [row["name"] for row in _table()] == [rule.name for rule in RULES]


def test_the_spec_says_what_each_rule_judges_and_how() -> None:
    words = {Kind.RELEASE: "release", Kind.GRID: "grid", Kind.CSR: "sequence"}
    for row, rule in zip(_table(), RULES, strict=True):
        judges = ", ".join(word for kind, word in words.items() if kind in rule.kinds)
        how = "fold" if isinstance(rule, Fold) else "check"
        if rule.severity is Severity.WARNING:
            how += ", warning"
        assert (row["judges"].strip(), row["kind"].strip()) == (judges, how), rule.name
