"""The Validation Rule runner: order, prerequisites, shared scans, containment (ADR 0074).

`evaluate` runs rules in registration order on every target each applies to. A
rule whose prerequisite failed or was skipped on the target, or on a target it
belongs to, is skipped and recorded, not reported; its `when` is not asked, since
it may read what the prerequisite found broken. A fold is judged from its scan,
which is read once per target for every fold on it that can run. An exception,
from a rule or from its `when`, fails the one rule that raised it; findings it
made before are kept.
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
    Unreadable,
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
        #: The rules whose `when` declined a target: decided, and constraining nothing.
        self.declined: set[tuple[str, int]] = set()
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
            if self.decided(rule, i) or not rule.judges(target):
                continue
            if isinstance(rule, Fold):
                self.scan(rule.scan, i, target)
            elif self.ready(rule, i, target) and self.applies(rule, i, target):
                self.settle(rule, i, _contained(rule, target, partial(rule.findings, target)))

    def decided(self, rule: Rule, i: int) -> bool:
        return (rule.name, i) in self.states or (rule.name, i) in self.declined

    def applies(self, rule: Rule, i: int, target: Target) -> bool:
        """Whether `rule`'s `when` accepts target `i`; a `when` that raises fails the rule."""
        try:
            accepted = rule.when(target)
        except Exception as exc:  # this rule's failure, and no other's
            self.settle(rule, i, [_failure(rule, target, exc)])
            return False
        if not accepted:
            self.declined.add((rule.name, i))
        return accepted

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
                    if _blocks(rule, state):
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
        ready = [
            fold
            for fold in self.folds_on(scan, i, target)
            if self.ready(fold, i, target) and self.applies(fold, i, target)
        ]
        if not ready:
            return  # nothing to measure, so nothing to read
        done = _Pass(scan, target, ready)
        try:
            done.drive()
            self.conclude(done, i, target)
        finally:  # an interruption still ends the pass, and what it held goes with it
            target.release(scan.name)

    def conclude(self, done: _Pass, i: int, target: Target) -> None:
        """Each fold of a finished pass: failed, skipped behind its blocker, or judged."""
        for fold in done.folds:
            if fold.name in done.failures:
                self.settle(fold, i, [done.failures[fold.name]])
            elif fold.name in done.blocked:
                self.skip(fold.name, i, [done.blocked[fold.name]])
            else:
                judged = partial(fold.judged, target, done.totals[fold.name])
                self.settle(fold, i, _contained(fold, target, judged))

    def folds_on(self, scan: Scan[Any], i: int, target: Target) -> list[Fold[Any, Any]]:
        """The selected, undecided folds on `scan` that apply to target `i`."""
        return [
            rule
            for rule in self.rules
            if isinstance(rule, Fold) and rule.scan is scan and self.undecided(rule, i, target)
        ]

    def undecided(self, rule: Rule, i: int, target: Target) -> bool:
        return rule.name in self.chosen and not self.decided(rule, i) and rule.judges(target)


def _blocks(rule: Rule, state: str) -> bool:
    """Whether a prerequisite that ended in `state` stops `rule`."""
    if rule.only_failures_block:
        return state == _FAILED
    return state != _PASSED


class _Pass:
    """One pass of a scan for its ready folds: each fold's total, failure or blocker.

    A fold that raises while measuring fails alone and stops measuring. A read that
    fails part-way is reported by the first fold still measuring, and blocks the
    others, since a part of the data says nothing about the whole. A plane that
    cannot be decoded (`Unreadable`) is reported by the first fold that reads it
    and blocks only the folds that read it after; one that names the rule reporting
    it is skipped behind that rule, which reads the plane's codes, not its values.
    """

    def __init__(self, scan: Scan[Any], target: Target, folds: list[Fold[Any, Any]]) -> None:
        self.scan, self.target, self.folds = scan, target, folds
        self.totals = {fold.name: fold.zero for fold in folds}
        self.failures: dict[str, Finding] = {}
        self.blocked: dict[str, str] = {}
        self.unreadable_by: str | None = None

    def measuring(self) -> list[Fold[Any, Any]]:
        stopped = self.failures.keys() | self.blocked.keys()
        return [fold for fold in self.folds if fold.name not in stopped]

    def drive(self) -> None:
        try:
            for block in self.scan.blocks(self.target):
                for fold in self.measuring():
                    self.measure(fold, block)
        except Exception as exc:  # the read itself failed, so no fold saw all of it
            measuring = self.measuring()
            if measuring:
                self.report(measuring[0], exc)
                for fold in measuring[1:]:
                    self.blocked[fold.name] = measuring[0].name

    def measure(self, fold: Fold[Any, Any], block: Any) -> None:
        """Add one block's measure to `fold`'s total; a fold that raises stops there."""
        try:
            self.totals[fold.name] = self.totals[fold.name] + fold.measure(self.target, block)
        except Unreadable as exc:
            self.unreadable(fold, exc)
        except Exception as exc:  # this fold's failure, and no other's
            self.failures[fold.name] = _failure(fold, self.target, exc)

    def unreadable(self, fold: Fold[Any, Any], exc: Unreadable) -> None:
        """Skip `fold` behind the rule that reports the plane, or report it once here.

        A named reporter that is not measuring in this pass reports nothing, so the
        plane is reported here instead: an undecodable plane is never only a skip.
        """
        if exc.reported_by is not None and exc.reported_by in self.reporting():
            self.blocked[fold.name] = exc.reported_by
        elif self.unreadable_by is None:
            self.unreadable_by = fold.name
            self.report(fold, exc)
        else:
            self.blocked[fold.name] = self.unreadable_by

    def reporting(self) -> set[str]:
        """The folds of this pass that are still measuring, or have failed: they report."""
        return {fold.name for fold in self.folds if fold.name not in self.blocked}

    def report(self, fold: Fold[Any, Any], exc: Exception) -> None:
        message = self.scan.failure(self.target, exc)
        self.failures[fold.name] = fold.finding(self.target, message, Severity.ERROR)


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
