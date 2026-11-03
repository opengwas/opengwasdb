"""The Validation Rule runner, driven by rules that read nothing (ADR 0074).

What `evaluate` promises any rule: it runs in registration order on every target it
applies to; a prerequisite that failed or was skipped skips it, and the skip is
recorded rather than reported; an exception fails that rule alone; findings come
out in rule order, then target order; a selection runs what it names and what that
runs after.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest

from opengwasdb.validation import validate_store
from opengwasdb.validation.rule import (
    Check,
    Finding,
    Fold,
    Kind,
    Scan,
    Severity,
    Skip,
    Target,
    ValidationResult,
    check_registry,
)
from opengwasdb.validation.run import evaluate


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


def _counting_scan(blocks: list[int], reads: list[int]) -> Scan[int]:
    def read(target: Target) -> Iterator[int]:
        for block in blocks:
            reads.append(block)
            yield block

    return Scan("numbers", read)


def _sum_fold(
    name: str, scan: Scan[int], limit: int, after: tuple[str, ...] = ()
) -> Fold[int, int]:
    def judge(target: Target, total: int) -> Iterable[str]:
        return [f"{name}: total {total}"] if total > limit else []

    return Fold(
        name=name,
        kinds=Kind.RELEASE,
        after=after,
        scan=scan,
        zero=0,
        measure=lambda target, block: block,
        judge=judge,
    )


def test_every_fold_on_a_scan_shares_one_read() -> None:
    reads: list[int] = []
    scan = _counting_scan([1, 2, 3], reads)
    rules = [_sum_fold("a", scan, 5), _check("between"), _sum_fold("b", scan, 1)]
    result = evaluate(rules, [RELEASE])
    assert reads == [1, 2, 3]
    assert result.errors == ["a: total 6", "b: total 6"]


def test_a_fold_that_raises_fails_alone() -> None:
    scan = _counting_scan([1, 2], [])

    def measure(target: Target, block: int) -> int:
        raise RuntimeError("bad block")

    broken = Fold(
        name="broken",
        kinds=Kind.RELEASE,
        scan=scan,
        zero=0,
        measure=measure,
        judge=lambda target, total: [],
    )
    result = evaluate([broken, _sum_fold("fine", scan, 2)], [RELEASE])
    failure = Finding("broken", Severity.ERROR, "validation failed: bad block", ".")
    assert result.of("broken") == [failure]
    assert result.of("fine")[0].message == "fine: total 3"


def test_a_read_that_fails_is_reported_once_and_skips_the_other_folds() -> None:
    def read(target: Target) -> Iterator[int]:
        yield 1
        raise OSError("truncated")

    scan = Scan("numbers", read, lambda target, exc: f"cannot read numbers: {exc}")
    result = evaluate([_sum_fold("a", scan, 0), _sum_fold("b", scan, 0)], [RELEASE])
    assert result.errors == ["cannot read numbers: truncated"]
    assert result.failed() == {"a"}
    assert result.skipped == {"b": ("a",)}


def test_a_rule_that_cannot_judge_its_target_is_skipped_with_its_reason() -> None:
    def run(target: Target) -> Iterable[str]:
        raise Skip("nothing to check")

    rules = [Check(name="a", kinds=Kind.RELEASE, run=run), _check("b", "ran", after=("a",))]
    result = evaluate(rules, [RELEASE])
    assert result.ok
    assert result.skipped == {"a": ("nothing to check",), "b": ("a",)}


def test_the_registry_refuses_a_fold_after_a_rule_registered_after_its_scan_starts() -> None:
    scan = _counting_scan([1], [])
    with pytest.raises(ValueError, match="before the first fold on 'numbers'"):
        check_registry(
            [_sum_fold("a", scan, 0), _check("c"), _sum_fold("b", scan, 0, after=("c",))]
        )
