"""The Validation Rule runner: order, prerequisites, shared scans, containment (ADR 0074).

`evaluate` runs rules in registration order on every target each applies to. A
rule whose prerequisite failed or was skipped on the target, or on a target it
belongs to, is skipped and recorded, not reported; its `when` is not asked, since
it may read what the prerequisite found broken. A fold is judged from its scan,
which is read once per target for every fold on it that can run. A fold's
prerequisite may be an earlier fold on the same scan: it is decided when the pass
ends, and the folds are judged in registration order, so the fold is judged after
it or skipped behind it. Its `when` is asked before the read. An exception,
from a rule or from its `when`, fails the one rule that raised it; findings it
made before are kept.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Collection, Iterable, Sequence
from functools import cache, partial
from typing import Any

from opengwasdb.encoding.timing import (
    PhaseTimer,
    format_duration,
    log_phase,
    log_progress,
    log_progress_unknown,
    progress_every,
)
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

log = logging.getLogger(__name__)

#: How a rule ended on one target, as the rules that run after it see it.
_PASSED, _FAILED, _SKIPPED = "passed", "failed", "skipped"

@cache
def untimed_rules() -> frozenset[str]:
    """The rules with nothing to time here: `validate`'s manifest rule, whose Store is
    opened from its manifest before `evaluate` runs (issue #308).

    Read from the rule's own name constant rather than copied, so a rename cannot leave
    a stale exclusion behind; imported here because `validate` imports this module.
    """
    from opengwasdb.validation.validate import MANIFEST_RULE

    return frozenset({MANIFEST_RULE})


def _rule_label(rule: Rule, target: Target) -> str:
    """One rule judging one target, named the same way in the log and the report."""
    return f"{rule.name} on {target.label}"


def _scan_label(scan: Scan[Any], target: Target) -> str:
    """One pass of a scan, named the same way in the log and the report."""
    return f"scan {scan.name} on {target.label}"


def evaluate(
    rules: Sequence[Rule],
    targets: Sequence[Target],
    selection: Collection[str] | None = None,
) -> ValidationResult:
    """Run `rules` over `targets` in registration order; report findings in that order.

    Every rule that runs logs its start and its elapsed time, as does every pass of a
    scan, and the run closes with the wall time each accounted for (issue #308). A
    validation of a full-size Store Release runs for hours; one that logged nothing
    until it finished could not say which rule was slow. The timings go to the log,
    never to the report a caller prints.
    """
    run = _Run(rules, targets, selected(rules, selection))
    log.info("Validation: %d rules over %d targets", len(run.chosen), len(targets))
    try:
        with log_phase(log, "Validation"):
            for rule in rules:
                if rule.name in run.chosen:
                    run.judge(rule)
    finally:
        for target in targets:
            target.close()
        # A failed or interrupted run says where it spent the time before it stopped.
        log.info("Validation phase timings:\n%s", run.timer.format_report())
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
        #: The skips recorded on each target, in the order the run recorded them.
        self.skipped: dict[tuple[str, int], list[Finding]] = {}
        #: Each scan's folds, by name: a prerequisite among them is decided by the pass.
        self.on_scan: dict[str, set[str]] = {}
        #: The wall time each rule and each scan pass accounted for.
        self.timer = PhaseTimer()
        for rule in rules:
            if isinstance(rule, Fold):
                self.on_scan.setdefault(rule.scan.name, set()).add(rule.name)

    def result(self) -> ValidationResult:
        """What the run recorded, in the order it judged: rule order, then target order."""
        ordered: list[Finding] = []
        for rule in self.rules:
            for i in range(len(self.targets)):
                ordered.extend(self.found.get((rule.name, i), ()))
                ordered.extend(self.skipped.get((rule.name, i), ()))
        return ValidationResult(tuple(ordered))

    def judge(self, rule: Rule) -> None:
        for i, target in enumerate(self.targets):
            if self.decided(rule, i) or not rule.judges(target):
                continue
            self.judge_one(rule, i, target)

    def judge_one(self, rule: Rule, i: int, target: Target) -> None:
        """`rule` on target `i`, timed only if it runs.

        A fold costs what its shared read costs, and that read is one phase for every
        fold on it: charging the wait to the first fold would attribute another rule's
        work to it (ADR 0074). A rule the target does not call for, or one a failed
        prerequisite stops, is not timed at all, so the report cannot claim it ran;
        `skip` logs the one line that says it did not (issue #308).
        """
        if isinstance(rule, Fold):
            self.scan(rule.scan, i, target)
        elif self.ready(rule, i, target) and self.applies(rule, i, target):
            timed = rule.name not in untimed_rules()
            with _Phase(self.timer, _rule_label(rule, target), timed=timed) as phase:
                outcome = _contained(rule, target, partial(rule.findings, target))
                if isinstance(outcome, Skip):
                    # It looked and found nothing to judge, so it is not charged: a row
                    # for it would say a rule was judged that was not (issue #308).
                    phase.drop()
                self.settle(rule, i, outcome)

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

    def ready(
        self, rule: Rule, i: int, target: Target, prerequisites: Iterable[str] | None = None
    ) -> bool:
        """Whether `rule` may run on target `i`; if not, record it as skipped.

        `prerequisites` limits which of the rule's prerequisites are looked at;
        by default, all of them.
        """
        blockers = self.blockers(rule, target, prerequisites)
        if blockers:
            self.skip(rule.name, i, blockers)
        return not blockers

    def blockers(
        self, rule: Rule, target: Target, prerequisites: Iterable[str] | None = None
    ) -> list[str]:
        """The prerequisites of `rule` that did not pass on `target` or a target it belongs to."""
        found = []
        for name in rule.after if prerequisites is None else prerequisites:
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
        """Record that `name` could not judge target `i`, and every reason it gave.

        One record per rule and target: the reasons are named in its message, in the
        order they were recorded, and kept apart in `reasons` so `skipped` reads them
        one by one. The log gets one line for it and no phase: a rule that did not run
        has no time to account for (issue #308).
        """
        self.states[(name, i)] = _SKIPPED
        existing = self.skipped.get((name, i))
        reasons = list(existing[0].reasons) if existing else []
        for reason in why:
            if reason not in reasons:
                reasons.append(reason)
        record = Finding(
            name, Severity.SKIP, ", ".join(reasons), self.targets[i].label, tuple(reasons)
        )
        self.skipped[(name, i)] = [record]
        log.info("%s on %s: skipped, %s", name, self.targets[i].label, record.message)

    def scan(self, scan: Scan[Any], i: int, target: Target) -> None:
        """Read `scan` once on target `i` for every fold on it that can run there."""
        ready = [
            fold
            for fold in self.folds_on(scan, i, target)
            if self.ready(fold, i, target, self.before_read(fold)) and self.applies(fold, i, target)
        ]
        if not ready:
            return  # nothing to measure, so nothing to read
        done = _Pass(scan, target, ready)
        label = _scan_label(scan, target)
        try:
            with log_phase(log, label):
                with self.timer.phase(label):
                    # The folds share the one read, so they are charged to it, not
                    # timed apart: naming them says which rules the phase serves.
                    log.info("%s: %s", label, ", ".join(fold.name for fold in ready))
                    done.drive()
                    self.conclude(done, i, target)
        finally:  # an interruption still ends the pass, and what it held goes with it
            target.release(scan.name)

    def before_read(self, fold: Fold[Any, Any]) -> list[str]:
        """`fold`'s prerequisites decided before its scan is read: those not on the scan."""
        return [name for name in fold.after if name not in self.on_scan[fold.scan.name]]

    def on_its_scan(self, fold: Fold[Any, Any]) -> list[str]:
        """`fold`'s prerequisites on its own scan, decided when the pass ends."""
        return [name for name in fold.after if name in self.on_scan[fold.scan.name]]

    def conclude(self, done: _Pass, i: int, target: Target) -> None:
        """Each fold of a finished pass, in registration order: skipped behind a
        prerequisite on the scan that did not pass, failed, skipped behind its blocker,
        or judged. A prerequisite on the scan is registered earlier, so it is decided
        here first; standing on it, the fold is not judged, as behind any other.
        """
        for fold in done.folds:
            behind = self.blockers(fold, target, self.on_its_scan(fold))
            if behind:
                self.skip(fold.name, i, behind)
            elif fold.name in done.failures:
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
        started = time.monotonic()
        label = _scan_label(self.scan, self.target)
        # Counting is not the read: outside the read's `try`, a count that fails leaves
        # every fold to meet the corruption where the blocks do, exactly as a pass with
        # no progress at all would. Inside it, the count's failure would be reported as
        # the read's and block the folds that would have judged (issue #308).
        total = self.counted(label)
        overran = False
        try:
            for done, block in enumerate(self.scan.blocks(self.target), start=1):
                for fold in self.measuring():
                    self.measure(fold, block)
                overran = self.report_progress(label, done, total, started, overran)
        except Exception as exc:  # the read itself failed, so no fold saw all of it
            measuring = self.measuring()
            if measuring:
                self.report(measuring[0], exc)
                for fold in measuring[1:]:
                    self.blocked[fold.name] = measuring[0].name

    def counted(self, label: str) -> int | None:
        """How many blocks the scan expects, or None where the count cannot be made.

        The count reads what the read may never reach -- a Store Variant Table's offsets
        sidecar and its tabix index, say -- and a corrupted one must not turn into a
        failed read, so the failure is logged and the pass goes on without a total.
        """
        if self.scan.total is None:
            return None
        try:
            return self.scan.total(self.target)
        except Exception as exc:  # the read meets it, or does not, as it would alone
            log.info("%s: no block count (%s), so progress without an ETA", label, exc)
            return None

    def report_progress(
        self, label: str, done: int, total: int | None, started: float, overran: bool
    ) -> bool:
        """One progress line for this block, and the first overrun's warning.

        A count taken from a sidecar is what the read expects, not what it will find: a
        table longer than its sidecar runs past it, and the line keeps saying `k/N` so
        the disagreement shows before the rule that reports it.
        """
        if total is None:
            log_progress_unknown(log, label, done, started)
            return overran
        if done > total and not overran:
            log.warning(
                "%s: %d blocks, %d more than its sidecar counted", label, done, done - total
            )
            overran = True
        log_progress(log, label, done, total, started, every=progress_every(total))
        return overran

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


class _Phase:
    """One rule's timed work: its start line now, its elapsed line and report row at the
    end -- unless the rule reports that it judged nothing (issue #308)."""

    def __init__(self, timer: PhaseTimer, label: str, *, timed: bool = True) -> None:
        self.timer, self.label, self.timed = timer, label, timed
        self.started = time.monotonic()
        self.dropped = False

    def __enter__(self) -> _Phase:
        if self.timed:
            log.info("%s: start", self.label)
        return self

    def __exit__(self, *_exc: object) -> None:
        if not self.timed or self.dropped:
            return
        elapsed = time.monotonic() - self.started
        self.timer.record(self.label, elapsed)
        log.info("%s: done in %s", self.label, format_duration(elapsed))

    def drop(self) -> None:
        """The rule judged nothing: no elapsed line and no row in the report."""
        self.dropped = True


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
