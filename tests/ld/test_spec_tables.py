"""The spec's tables and the code's constants say the same thing.

`docs/spec/ld-store.md` is the authority. `opengwasdb.ld.format` carries its provenance keys (§7)
and validation rules (§12) as data, and these tests read both tables out of the spec, so
a rule added to one and not the other fails here instead of drifting.
"""

from __future__ import annotations

import re
from pathlib import Path

import opengwasdb.ld.format as fmt

SPEC = Path(__file__).resolve().parents[2] / "docs" / "spec" / "ld-store.md"


def section(number: int) -> str:
    text = SPEC.read_text(encoding="utf-8")
    match = re.search(rf"^## {number}\. .*?(?=^## {number + 1}\. )", text, re.M | re.S)
    assert match, f"ld-store §{number} not found"
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


def test_the_far_pairs_have_the_four_rules_section_11_named() -> None:
    # The Far Pairs are their own component again (ADR 0093): the four far.* rules ld-store §11
    # named before the merge. `far.index` held them equal to the Proxy Index's far entries, and
    # went with the index (#72).
    far = [rule for rule in fmt.VALIDATION_RULES if rule.startswith("far.")]
    assert far == ["far.rows", "far.codes", "far.distance", "far.threshold"]


def test_no_key_names_pca_settings() -> None:
    # PCs are supplied, not computed (ADR 0088).
    assert not [key for key in fmt.REQUIRED_KEYS if key.startswith("pc_adjustment.")]
