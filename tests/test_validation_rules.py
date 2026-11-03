"""The Validation Rule runner, driven by rules that read nothing (ADR 0074).

What `evaluate` promises any rule: it runs in registration order on every target it
applies to; a prerequisite that failed or was skipped skips it, and the skip is
recorded rather than reported; an exception fails that rule alone; findings come
out in rule order, then target order; a selection runs what it names and what that
runs after.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from opengwasdb.validation import validate_store
from opengwasdb.validation.rule import (
    Check,
    Finding,
    Kind,
    Severity,
    Target,
    ValidationResult,
    check_registry,
)
from opengwasdb.validation.validate import evaluate


def _check(
    name: str,
    *messages: str,
    kinds: Kind = Kind.RELEASE,
    after: tuple[str, ...] = (),
    severity: Severity = Severity.ERROR,
) -> Check:
    def run(target: Target) -> Iterable[str]:
        return [f"{name}: {m} on {target.label}" for m in messages]

    return Check(name=name, kinds=kinds, after=after, severity=severity, run=run)


def _raises(name: str) -> Check:
    def run(target: Target) -> Iterable[str]:
        raise RuntimeError("boom")

    return Check(name=name, kinds=Kind.RELEASE, run=run)


RELEASE = Target(Kind.RELEASE, ".")
GRID = Target(Kind.GRID, "data.zarr", parent=RELEASE)


def test_a_rule_after_a_failed_rule_is_skipped_and_not_reported() -> None:
    result = evaluate(
        [_check("a", "broken"), _check("b", "noise", after=("a",))],
        [RELEASE],
    )
    assert result.failed() == {"a"}
    assert result.skipped == {"b": ("a",)}
    assert result.errors == ["a: broken on ."]


def test_a_prerequisite_on_the_release_gates_its_grid() -> None:
    rules = [_check("a", "broken"), _check("b", "noise", kinds=Kind.GRID, after=("a",))]
    result = evaluate(rules, [RELEASE, GRID])
    assert result.failed() == {"a"}
    assert result.skipped == {"b": ("a",)}


def test_a_prerequisite_that_does_not_apply_constrains_nothing() -> None:
    rules = [
        _check("a", "never", kinds=Kind.SUBSET),
        _check("b", "ran", kinds=Kind.GRID, after=("a",)),
    ]
    result = evaluate(rules, [RELEASE, GRID])
    assert result.errors == ["b: ran on data.zarr"]
    assert result.skipped == {}


def test_an_exception_fails_its_own_rule_and_no_other() -> None:
    rules = [_raises("a"), _check("b", "ran"), _check("c", after=("a",))]
    result = evaluate(rules, [RELEASE])
    assert result.of("a") == [Finding("a", Severity.ERROR, "validation failed: boom", ".")]
    assert result.failed() == {"a", "b"}
    assert result.skipped == {"c": ("a",)}


def test_findings_come_in_rule_order_then_target_order() -> None:
    both = Kind.RELEASE | Kind.GRID
    rules = [_check("b", "x", kinds=both), _check("a", "y", kinds=both)]
    result = evaluate(rules, [RELEASE, GRID])
    assert result.errors == ["b: x on .", "b: x on data.zarr", "a: y on .", "a: y on data.zarr"]


def test_a_warning_does_not_fail_the_rule_or_its_dependents() -> None:
    rules = [
        _check("a", "unusual", severity=Severity.WARNING),
        _check("b", "ran", after=("a",)),
    ]
    result = evaluate(rules, [RELEASE])
    assert result.warnings == ["a: unusual on ."]
    assert result.errors == ["b: ran on ."]
    assert result.failed() == {"b"}


def test_a_selection_runs_its_rules_and_their_prerequisites_only() -> None:
    rules = [_check("a", "p"), _check("b", "q", after=("a",)), _check("c", "r")]
    result = evaluate(rules, [RELEASE], selection={"b"})
    assert result.failed() == {"a"}
    assert result.skipped == {"b": ("a",)}
    assert evaluate(rules, [RELEASE], selection={"c"}).failed() == {"c"}


def test_an_unknown_rule_name_is_refused_before_the_release_is_read(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no validation rule is named 'no.such.rule'"):
        validate_store(tmp_path / "absent", rules={"no.such.rule"})


def test_the_registry_refuses_a_name_twice_and_a_prerequisite_registered_later() -> None:
    with pytest.raises(ValueError, match="registered twice"):
        check_registry([_check("a"), _check("a")])
    with pytest.raises(ValueError, match="not registered before it"):
        check_registry([_check("b", after=("a",)), _check("a")])


def test_the_projections_read_as_before() -> None:
    result = ValidationResult(
        (
            Finding("a", Severity.WARNING, "w", "."),
            Finding("b", Severity.ERROR, "e", "."),
        )
    )
    assert (result.errors, result.warnings, result.ok) == (["e"], ["w"], False)
    assert ValidationResult().ok


def test_a_release_that_cannot_be_opened_fails_the_manifest_rule(tmp_path: Path) -> None:
    result = validate_store(tmp_path)
    assert result.failed() == {"manifest.loads"}
    assert result.errors == ["missing manifest.json"]
