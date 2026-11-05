"""Where a Validation Rule is skipped with no failure behind it (ADR 0074, D1).

`validate` prints `ok`, `errors` and `warnings`, and no skips, so its "valid" is
truthful only while a rule is skipped with no error recorded just where it has
nothing to judge. The runner records a skip behind a prerequisite that failed or
was skipped, behind a read that failed part-way, or behind the fold that reports an
undecodable plane; each failure there is an error. The other skips are the rules
that raise `Skip` themselves, listed here with what each has nothing to judge, and
the rules their skip stops. A new one fails here until it is reviewed, here and in
ADR 0074.
"""

from __future__ import annotations

import ast
import pathlib
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.validation import RULES, Finding, Severity, validate_store

ROOT = Path(__file__).resolve().parents[1] / "opengwasdb"

#: Every function under `validation/` that raises `Skip`, by the rule it judges for.
RAISED = {
    ("validation/rules/associations.py", "_judge_eaf_range"): "associations.eaf_range",
    ("validation/rules/by_variant.py", "present"): "by_variant.present",
    ("validation/rules/variants.py", "variant_aliases"): "sqlite.variant_aliases",
}

#: The rules a skip of one of those stops: the index rules after `by_variant.present`,
#: which have no index to judge on a component that need not carry one.
STOPPED = {
    "by_variant.analyses",
    "by_variant.counts",
    "by_variant.layout",
    "by_variant.members",
    "by_variant.offsets",
    "by_variant.order",
    "by_variant.provenance",
    "by_variant.sample",
}


def _names_skip(node: ast.expr | None) -> bool:
    called = node.func if isinstance(node, ast.Call) else node
    return (isinstance(called, ast.Name) and called.id == "Skip") or (
        isinstance(called, ast.Attribute) and called.attr == "Skip"
    )


class _Raises(ast.NodeVisitor):
    """The innermost function around each `raise Skip`."""

    def __init__(self) -> None:
        self.functions: list[str] = []
        self.found: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_Raise(self, node: ast.Raise) -> None:
        if _names_skip(node.exc):
            self.found.add(self.functions[-1] if self.functions else "<module>")


def _raising() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in sorted((ROOT / "validation").rglob("*.py")):
        raises = _Raises()
        raises.visit(ast.parse(path.read_text(encoding="utf-8")))
        found |= {(str(path.relative_to(ROOT)), name) for name in raises.found}
    return found


def test_only_the_reviewed_rules_raise_skip() -> None:
    assert _raising() == set(RAISED)


def test_every_recorded_skip_names_the_target_and_the_reason(
    dense_store_path: pathlib.Path,
) -> None:
    """A skip says what it could not judge and why, so a reader outside Python can
    tell it from a pass (#378); the fixture holds no `eaf` plane to judge."""
    result = validate_store(dense_store_path)

    assert [f.rule for f in result.skips] == ["associations.eaf_range"]
    assert result.skips == (
        Finding(
            "associations.eaf_range", Severity.SKIP, "no frequencies", "data.zarr",
            ("no frequencies",),
        ),
    )
    assert result.skipped == {"associations.eaf_range": ("no frequencies",)}


def test_a_skip_stops_only_the_rules_that_have_nothing_left_to_judge() -> None:
    by_name = {rule.name: rule for rule in RULES}
    assert set(RAISED.values()) <= by_name.keys()
    stopped: set[str] = set()
    pending = list(RAISED.values())
    while pending:
        name = pending.pop()
        for rule in RULES:
            if name in rule.after and not rule.only_failures_block and rule.name not in stopped:
                stopped.add(rule.name)
                pending.append(rule.name)
    assert stopped == STOPPED


@pytest.mark.parametrize(
    "rules", [None, ("variant_table.representatives",)], ids=["every rule", "selected"]
)
def test_a_row_count_mismatch_fails_the_rows_rule_the_representatives_stand_on(
    dense_store_path: pathlib.Path, rules: tuple[str, ...] | None
) -> None:
    """The representatives are chosen from the offsets, so they stand on the row
    count, `variant_table.rows`, an earlier fold on the same read. Selected alone, they
    bring it with them: the mismatch is reported, not left behind a skip."""
    offsets_path = dense_store_path / "variant_offsets.npy"
    offsets = np.load(offsets_path)
    assert len(offsets) >= 2, "the fixture must have a row to drop"
    np.save(offsets_path, offsets[:-1])

    result = validate_store(dense_store_path, rules=rules)

    assert not result.ok
    assert "variant_table.rows" in result.failed()
    if rules is not None:
        assert result.failed() == {"variant_table.rows"}
    assert any(
        error.startswith("variant_offsets.npy has ") and "but variants.tsv.gz has" in error
        for error in result.errors
    ), result.errors
    assert result.skipped["variant_table.representatives"] == ("variant_table.rows",)
