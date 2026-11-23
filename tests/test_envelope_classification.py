"""Every Store Release envelope entry is classified in the spec (issue #463, ADR 0102).

ADR 0102's rule decides what a Store Release ships: each component is read by the
Store Reader, kept as documented provenance, kept as a documented convenience, or
dropped. Spec §1.1 holds the classification. The envelope constants in
`opengwasdb.store.open` are the machine-readable list of what a release may carry, so
a name added to them without a row in §1.1 has shipped without anyone applying the
rule. This test fails until it has one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from opengwasdb.store.open import (
    DENSE_ENVELOPE,
    HYBRID_DENSE_COMPONENT_ENVELOPE,
    HYBRID_ENVELOPE,
    RAGGED_ENVELOPE,
)

SPEC = Path(__file__).resolve().parents[1] / "docs" / "spec" / "store-format.md"
HEADING = "### 1.1 What each component is for"
CLASSES = {"read", "provenance", "convenience", "dropped"}

#: One row of §1.1's table: `| component(s) | layouts | read today by | 0.3.0 class | ticket |`.
_ROW = re.compile(r"^\| (?P<component>[^|]+) \| [^|]+ \| [^|]+ \| (?P<klass>[^|]+) \| [^|]+ \|$")

_SEPARATOR = re.compile(r"^\|(\s*:?-+:?\s*\|)+$")


def _rows(text: str | None = None) -> list[tuple[set[str], str]]:
    """§1.1's classification rows: the names in each, and its class cell.

    Every table line must be a row of five cells. A line that is not would otherwise drop out
    of both tests unseen, and the component it names would go unclassified.
    """
    text = SPEC.read_text(encoding="utf-8") if text is None else text
    assert HEADING in text, f"spec has no '{HEADING}' section"
    section = text.split(HEADING, 1)[1].split("\n## ", 1)[0].split("\n### ", 1)[0]
    table = [line for line in section.splitlines() if line.startswith("|")]
    malformed = [line for line in table if not (_ROW.match(line) or _SEPARATOR.match(line))]
    assert malformed == [], "§1.1 table lines that are not five-cell rows"
    rows = [
        (set(re.findall(r"`([^`]+)`", m["component"])), m["klass"].strip())
        for line in table
        if (m := _ROW.match(line)) and m["component"].strip() != "Component"
    ]
    assert len(rows) > 20, "the section must hold the classification table"
    return rows


def test_a_malformed_table_line_fails_rather_than_vanishing() -> None:
    text = SPEC.read_text(encoding="utf-8")
    row = "| `regions.tsv` | Ragged, optional |"
    broken = text.replace(row, row.replace("` |", "`"))
    assert broken != text, "the fixture edit must reach the table"
    with pytest.raises(AssertionError, match="not five-cell rows"):
        _rows(broken)


def test_every_envelope_entry_has_a_classification_row() -> None:
    named = set().union(*(names for names, _ in _rows()))
    envelope = DENSE_ENVELOPE | RAGGED_ENVELOPE | HYBRID_ENVELOPE | HYBRID_DENSE_COMPONENT_ENVELOPE
    assert sorted(envelope - named) == []


def test_every_row_names_one_class() -> None:
    for names, klass in _rows():
        word = re.match(r"[*\s]*([a-z]*)", klass.lower())
        assert word is not None and word[1] in CLASSES, (sorted(names), klass)
