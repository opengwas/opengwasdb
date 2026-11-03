"""The Validation Rule runner: order, prerequisites, shared scans, containment (ADR 0074).

`evaluate` runs rules in registration order on every target each applies to. A
rule whose prerequisite failed or was skipped on the target, or on a target it
belongs to, is skipped and recorded, not reported. A fold is judged from its
scan, which is read once per target for every fold on it that can run. An
exception fails the one rule that raised it; findings it made before are kept.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Sequence
from functools import partial
from typing import Any

from opengwasdb.validation.rule import (
    Finding,
    Fold,
    Rule,
    Scan,
    Severity,
    Skip,
    Target,
    ValidationResult,
)

#: How a rule ended on one target, as the rules that run after it see it.
_PASSED, _FAILED, _SKIPPED = "passed", "failed", "skipped"


def evaluate(
    rules: Sequence[Rule],
    targets: Sequence[Target],
    selection: Collection[str] | None = None,
) -> ValidationResult:
    """Run `rules` over `targets` in registration order; report findings in that order."""
    run = _Run(rules, targets, selected(rules, selection))
    try:
        for rule in rules:
            if rule.name in run.chosen:
                run.judge(rule)
    finally:
        for target in targets:
            target.close()
    return run.result()


def selected(rules: Sequence[Rule], selection: Collection[str] | None) -> set[str]:
    """The names to run: every rule, or `selection` and every rule it runs after."""
    by_name = {rule.name: rule for rule in rules}
    if selection is None:
        return set(by_name)
    unknown = sorted(set(selection) - by_name.keys())
    if unknown:
        raise ValueError(f"no validation rule is named {', '.join(map(repr, unknown))}")
    chosen: set[str] = set()
    pending = list(selection)
    while pending:
        name = pending.pop()
        if name not in chosen:
            chosen.add(name)
            pending.extend(by_name[name].after)
    return chosen


class _Run:
    """What one run has decided so far: each rule's state and findings on each target."""

    def __init__(self, rules: Sequence[Rule], targets: Sequence[Target], chosen: set[str]) -> None:
        self.rules = rules
        self.targets = targets
        self.chosen = chosen
        self.position = {id(target): i for i, target in enumerate(targets)}
        self.states: dict[tuple[str, int], str] = {}
        self.found: dict[tuple[str, int], list[Finding]] = {}
        self.skipped: dict[str, list[str]] = {}

    def result(self) -> ValidationResult:
        """The findings in rule order, then target order, and the skips."""
        ordered = [
            finding
            for rule in self.rules
            for i in range(len(self.targets))
            for finding in self.found.get((rule.name, i), ())
        ]
        skipped = {name: tuple(why) for name, why in self.skipped.items()}
        return ValidationResult(tuple(ordered), skipped)

    def judge(self, rule: Rule) -> None:
        for i, target in enumerate(self.targets):
            if (rule.name, i) in self.states or not rule.applies_to(target):
                continue
            if not self.ready(rule, i, target):
                continue
            if isinstance(rule, Fold):
                self.scan(rule.scan, i, target)
            else:
                self.settle(rule, i, _contained(rule, target, partial(rule.findings, target)))

    def ready(self, rule: Rule, i: int, target: Target) -> bool:
        """Whether `rule` may run on target `i`; if not, record it as skipped."""
        blockers = self.blockers(rule, target)
        if blockers:
            self.skip(rule.name, i, blockers)
        return not blockers

    def blockers(self, rule: Rule, target: Target) -> list[str]:
        """The prerequisites of `rule` that did not pass on `target` or a target it belongs to."""
        found = []
        for name in rule.after:
            owner: Target | None = target
            while owner is not None:
                state = self.states.get((name, self.position.get(id(owner), -1)))
                if state is not None:
                    if state != _PASSED:
                        found.append(name)
                    break
                owner = owner.parent
        return found

    def settle(self, rule: Rule, i: int, outcome: list[Finding] | Skip) -> None:
        if isinstance(outcome, Skip):
            self.skip(rule.name, i, [str(outcome)])
            return
        self.found[(rule.name, i)] = outcome
        failed = any(f.severity is Severity.ERROR for f in outcome)
        self.states[(rule.name, i)] = _FAILED if failed else _PASSED

    def skip(self, name: str, i: int, why: list[str]) -> None:
        self.states[(name, i)] = _SKIPPED
        held = self.skipped.setdefault(name, [])
        held.extend(reason for reason in why if reason not in held)

    def scan(self, scan: Scan[Any], i: int, target: Target) -> None:
        """Read `scan` once on target `i` for every fold on it that can run there."""
        ready = [fold for fold in self.folds_on(scan, i, target) if self.ready(fold, i, target)]
        totals, failures, broken = _drive(scan, target, ready)
        for fold in ready:
            if fold.name in failures:
                self.settle(fold, i, [failures[fold.name]])
            elif broken is not None:
                self.skip(fold.name, i, [broken])
            else:
                judged = partial(fold.judged, target, totals[fold.name])
                self.settle(fold, i, _contained(fold, target, judged))


    def folds_on(self, scan: Scan[Any], i: int, target: Target) -> list[Fold[Any, Any]]:
        """The selected, undecided folds on `scan` that apply to target `i`."""
        return [
            rule
            for rule in self.rules
            if isinstance(rule, Fold) and rule.scan is scan and self.undecided(rule, i, target)
        ]

    def undecided(self, rule: Rule, i: int, target: Target) -> bool:
        chosen = rule.name in self.chosen and (rule.name, i) not in self.states
        return chosen and rule.applies_to(target)


def _drive(
    scan: Scan[Any], target: Target, folds: list[Fold[Any, Any]]
) -> tuple[dict[str, Any], dict[str, Finding], str | None]:
    """One pass of `scan`: each fold's total, the folds that raised, and who broke the read.

    A fold that raises while measuring fails alone and stops measuring. A read that
    fails part-way is reported by the first fold still measuring; the others are
    skipped, blocked by it, since a part of the data says nothing about the whole.
    """
    totals = {fold.name: fold.zero for fold in folds}
    failures: dict[str, Finding] = {}
    try:
        for block in scan.blocks(target):
            _measure(folds, target, block, totals, failures)
    except Exception as exc:  # the read itself failed, so no fold saw all of it
        measuring = [fold for fold in folds if fold.name not in failures]
        if not measuring:
            return totals, failures, None
        first = measuring[0]
        failures[first.name] = first.finding(target, scan.failure(target, exc), Severity.ERROR)
        return totals, failures, first.name
    return totals, failures, None


def _measure(
    folds: list[Fold[Any, Any]],
    target: Target,
    block: Any,
    totals: dict[str, Any],
    failures: dict[str, Finding],
) -> None:
    """Add one block's measure to each fold's total; a fold that raises stops there."""
    for fold in folds:
        if fold.name in failures:
            continue
        try:
            totals[fold.name] = totals[fold.name] + fold.measure(target, block)
        except Exception as exc:  # this fold's failure, and no other's
            failures[fold.name] = _failure(fold, target, exc)


def _failure(rule: Rule, target: Target, exc: Exception) -> Finding:
    return rule.finding(target, f"validation failed: {exc}", Severity.ERROR)


def _contained(
    rule: Rule, target: Target, findings: Callable[[], Iterable[Finding]]
) -> list[Finding] | Skip:
    """`rule`'s findings on `target`, ending with its failure if it raised.

    An exception says nothing about any other rule, so it fails this one alone; the
    findings the rule made before it are kept. A rule that cannot judge the target
    says so with `Skip`.
    """
    found: list[Finding] = []
    try:
        found.extend(findings())
    except Skip as skip:
        return skip
    except Exception as exc:  # this rule's failure, and no other's
        found.append(_failure(rule, target, exc))
    return found
