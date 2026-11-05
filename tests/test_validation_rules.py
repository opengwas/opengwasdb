"""The Validation Rule runner, driven by rules that read nothing (ADR 0074).

What `evaluate` promises any rule: it runs in registration order on every target it
applies to; a prerequisite that failed or was skipped skips it, and the skip is
recorded rather than reported; an exception fails that rule alone; findings come
out in rule order, then target order; a selection runs what it names and what that
runs after.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from opengwasdb.validation import validate_store
from opengwasdb.validation.rule import (
    Check,
    Finding,
    Fold,
    Kind,
    Rule,
    Scan,
    Severity,
    Skip,
    Target,
    ValidationResult,
    check_registry,
)
from opengwasdb.validation.run import evaluate
from opengwasdb.validation.scans import Undecodable


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


def test_a_plane_that_cannot_be_decoded_blocks_only_the_folds_that_read_it() -> None:
    """The first fold to read it reports it, once; a fold that never reads it sees
    every block."""

    def read(target: Target) -> Iterator[int]:
        yield from (1, 2, 3)

    scan = Scan("numbers", read, lambda target, exc: f"cannot decode numbers: {exc}")

    def decoded(target: Target, block: int) -> int:
        raise Undecodable("bad codes")

    def reads_the_plane(name: str) -> Fold[int, int]:
        return Fold(
            name=name,
            kinds=Kind.RELEASE,
            scan=scan,
            zero=0,
            measure=decoded,
            judge=lambda target, total: [],
        )

    rules = [reads_the_plane("first"), _sum_fold("codes", scan, 0), reads_the_plane("later")]
    result = evaluate(rules, [RELEASE])
    assert result.errors == ["cannot decode numbers: bad codes", "codes: total 6"]
    assert result.failed() == {"first", "codes"}
    assert result.skipped == {"later": ("first",)}


def test_a_plane_that_names_its_reporter_is_skipped_behind_it_or_reported_alone() -> None:
    """A rule that already says why the plane cannot be decoded reports it, however
    late it is registered; without that rule, the reader reports it."""

    def read(target: Target) -> Iterator[int]:
        yield from (1, 2, 3)

    scan = Scan("numbers", read, lambda target, exc: f"cannot decode numbers: {exc}")

    def decoded(target: Target, block: int) -> int:
        raise Undecodable("bad codes", reported_by="explains")

    reader = Fold(
        name="reader",
        kinds=Kind.RELEASE,
        scan=scan,
        zero=0,
        measure=decoded,
        judge=lambda target, total: [],
    )
    explains = _sum_fold("explains", scan, 0)

    result = evaluate([reader, explains], [RELEASE])
    assert result.errors == ["explains: total 6"]
    assert result.skipped == {"reader": ("explains",)}

    alone = evaluate([reader, explains], [RELEASE], selection={"reader"})
    assert alone.errors == ["cannot decode numbers: bad codes"]
    assert alone.skipped == {}


def test_a_when_that_raises_fails_its_own_rule_and_no_other() -> None:
    def broken(target: Target) -> bool:
        raise FileNotFoundError("data.zarr")

    rules = [
        Check(name="a", kinds=Kind.RELEASE, when=broken, run=lambda target: ["ran"]),
        _check("b", "ran"),
    ]
    result = evaluate(rules, [RELEASE])
    assert result.of("a") == [
        Finding("a", Severity.ERROR, "validation failed: data.zarr", ".")
    ]
    assert result.errors == ["validation failed: data.zarr", "b: ran on ."]


def test_a_when_is_not_asked_behind_a_failed_prerequisite() -> None:
    """The prerequisite found what `when` would read broken; asking would only repeat it."""
    asked: list[str] = []

    def reads(target: Target) -> bool:
        asked.append(target.label)
        return True

    scan = _counting_scan([1], [])
    rules: list[Rule] = [
        _check("a", "broken"),
        Check(name="b", kinds=Kind.RELEASE, after=("a",), when=reads, run=lambda t: ["ran"]),
        Fold(
            name="c",
            kinds=Kind.RELEASE,
            after=("a",),
            when=reads,
            scan=scan,
            zero=0,
            measure=lambda target, block: block,
            judge=lambda target, total: ["ran"],
        ),
    ]
    result = evaluate(rules, [RELEASE])
    assert asked == []
    assert result.skipped == {"b": ("a",), "c": ("a",)}


def test_what_a_scan_held_for_its_folds_is_released_when_its_pass_ends() -> None:
    """Held for the folds on the scan, not for the rest of the run: a rule after it
    that wants the value reads it again."""
    reads: list[str] = []
    scan = _counting_scan([1, 2], [])

    def held(target: Target) -> int:
        def read() -> int:
            reads.append("read")
            return 10

        return target.remember("held", read, until=scan.name)

    fold = Fold(
        name="a",
        kinds=Kind.RELEASE,
        scan=scan,
        zero=0,
        measure=lambda target, block: held(target) + block,
        judge=lambda target, total: [f"a: total {total}"],
    )
    later = Check(name="b", kinds=Kind.RELEASE, run=lambda target: [f"b: {held(target)}"])

    result = evaluate([fold, later], [RELEASE])

    assert result.errors == ["a: total 23", "b: 10"]
    assert reads == ["read", "read"]


def test_a_rule_that_cannot_judge_its_target_is_skipped_with_its_reason() -> None:
    def run(target: Target) -> Iterable[str]:
        raise Skip("nothing to check")

    rules = [Check(name="a", kinds=Kind.RELEASE, run=run), _check("b", "ran", after=("a",))]
    result = evaluate(rules, [RELEASE])
    assert result.ok
    assert result.skipped == {"a": ("nothing to check",), "b": ("a",)}


def test_a_rule_stopped_only_by_failures_runs_past_a_skip_and_not_past_a_failure() -> None:
    def nothing(target: Target) -> Iterable[str]:
        raise Skip("nothing to check")

    def last(name: str, after: tuple[str, ...]) -> Check:
        return Check(
            name=name,
            kinds=Kind.RELEASE,
            after=after,
            only_failures_block=True,
            run=lambda target: [f"{name} ran"],
        )

    skipped = Check(name="skipped", kinds=Kind.RELEASE, run=nothing)
    rules = [skipped, _check("failed", "bad"), last("past_skip", ("skipped",))]
    rules.append(last("past_failure", ("skipped", "failed")))
    result = evaluate(rules, [RELEASE])
    assert result.errors == ["failed: bad on .", "past_skip ran"]
    assert result.skipped == {"skipped": ("nothing to check",), "past_failure": ("failed",)}


def test_the_registry_refuses_a_fold_after_a_rule_registered_after_its_scan_starts() -> None:
    scan = _counting_scan([1], [])
    with pytest.raises(ValueError, match="before the first fold on 'numbers'"):
        check_registry(
            [_sum_fold("a", scan, 0), _check("c"), _sum_fold("b", scan, 0, after=("c",))]
        )


def test_the_registry_takes_a_fold_after_an_earlier_fold_on_its_scan_not_a_later_one() -> None:
    scan = _counting_scan([1], [])
    earlier, later = _sum_fold("a", scan, 0), _sum_fold("b", scan, 0, after=("a",))
    assert check_registry([earlier, later]) == (earlier, later)
    with pytest.raises(ValueError, match=r"'b' runs after \['a'\], which are not registered"):
        check_registry([later, earlier])


def _judging(
    name: str, scan: Scan[int], says: str, judged: list[str], after: tuple[str, ...] = ()
) -> Fold[int, int]:
    """A summing fold that records being judged, then passes, fails or skips as `says`."""

    def judge(target: Target, total: int) -> Iterable[str]:
        judged.append(name)
        if says == "skip":
            raise Skip("nothing to judge")
        return [f"{name}: total {total}"] if says == "fail" else []

    return replace(_sum_fold(name, scan, 0, after), judge=judge)


@pytest.mark.parametrize("says", ["pass", "fail", "skip"])
def test_a_fold_after_an_earlier_fold_on_its_scan_is_judged_only_once_that_one_passed(
    says: str,
) -> None:
    """Both measure in the one read. The earlier fold is judged first; the later one
    is judged only if it passed, and is skipped behind it otherwise, as behind any
    prerequisite."""
    reads: list[int] = []
    judged: list[str] = []
    scan = _counting_scan([1, 2], reads)
    rules = [
        _judging("a", scan, says, judged),
        _judging("b", scan, "fail", judged, after=("a",)),
    ]

    result = evaluate(rules, [RELEASE])

    assert reads == [1, 2]
    if says == "pass":
        assert judged == ["a", "b"]
        assert (result.errors, result.skipped) == (["b: total 3"], {})
    else:
        assert judged == ["a"]
        assert result.skipped["b"] == ("a",)
        assert result.errors == (["a: total 3"] if says == "fail" else [])
