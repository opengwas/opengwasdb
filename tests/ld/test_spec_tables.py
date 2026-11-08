"""The spec's tables and the code's constants say the same thing.

`docs/spec/ld-store.md` is the authority. `ogld.format` carries its provenance keys (§7)
and validation rules (§12) as data, and these tests read both tables out of the spec, so
a rule added to one and not the other fails here instead of drifting.
"""

from __future__ import annotations

import re
from pathlib import Path

import ogld.format as fmt

SPEC = Path(__file__).resolve().parent.parent / "docs" / "spec" / "ld-store.md"


def section(number: int) -> str:
    text = SPEC.read_text(encoding="utf-8")
    match = re.search(rf"^## {number}\. .*?(?=^## {number + 1}\. )", text, re.M | re.S)
    assert match, f"spec §{number} not found"
    return match.group(0)


def first_column(text: str) -> list[str]:
    """The backticked name opening each row of the section's tables."""
    return re.findall(r"^\| `([^`]+)` \|", text, re.M)


def test_the_provenance_keys_are_spec_section_7_s() -> None:
    keys = first_column(section(7))
    assert len(keys) > 30, "the §7 table must have been read for this to mean anything"
    assert tuple(keys) == fmt.REQUIRED_KEYS


def test_the_validation_rules_are_spec_section_12_s_in_order() -> None:
    rules = first_column(section(12))
    assert len(rules) > 10, "the §12 table must have been read for this to mean anything"
    assert tuple(rules) == fmt.VALIDATION_RULES


def test_no_rule_or_key_names_the_retired_far_component() -> None:
    # The Proxy Index holds the Far Pairs (spec §5, §11): no `far/` arrays, no far.* rules,
    # and no PCA settings now that PCs are supplied (ADR 0002).
    assert not [rule for rule in fmt.VALIDATION_RULES if rule.startswith("far.")]
    assert not [key for key in fmt.REQUIRED_KEYS if key.startswith("pc_adjustment.")]
