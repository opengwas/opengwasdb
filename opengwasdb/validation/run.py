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

A scan that can be read apart is spread over the run's `n_workers` forked workers
(issue #315): each block is measured where it is read, and added here in block
order, so the report is the one a single process makes at any worker count.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Collection, Generator, Iterable, Iterator, Sequence
from concurrent.futures import BrokenExecutor
from contextlib import closing
from dataclasses import dataclass, field
from functools import cache, partial
from typing import Any

from opengwasdb.build.ordered_pool import ordered_map
from opengwasdb.encoding.memory import pool_workers
from opengwasdb.encoding.timing import (
    PhaseTimer,
    format_duration,
    log_phase,
    log_progress,
    log_progress_unknown,
    progress_every,
)
from opengwasdb.validation.rule import (
    Check,
    Finding,
    Fold,
    Rider,
    Rule,
    Scan,
    Severity,
    Skip,
    Split,
    Target,
    Unreadable,
    ValidationResult,
)

log = logging.getLogger(__name__)

#: How a rule ended on one target, as the rules that run after it see it.
_PASSED, _FAILED, _SKIPPED = "passed", "failed", "skipped"
#: A riding check waiting for its scan's pass, which decides it (issue #314).
_WAITING = "waiting"


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
            # A riding check whose scan no fold asked for is decided on its own read.
            run.settle_all_waiting()
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
        #: Where each scan's first fold is registered: a check riding the scan that is
        #: registered before it waits for the pass (issue #314).
        self.first_fold: dict[str, int] = {}
        self.order = {rule.name: n for n, rule in enumerate(rules)}
        #: The `(scan, target)` passes already asked for, whether or not one was read.
        self.scanned: set[tuple[str, int]] = set()
        #: The riding checks waiting for a pass, by `(scan, target)`.
        self.waiting: dict[tuple[str, int], list[Check]] = {}
        for n, rule in enumerate(rules):
            if isinstance(rule, Fold):
                self.on_scan.setdefault(rule.scan.name, set()).add(rule.name)
                self.first_fold.setdefault(rule.scan.name, n)

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
            if isinstance(rule, Check) and rule.ride is not None and self.rides(rule, i, target):
                return
            self.run_check(rule, i, target)

    def run_check(self, rule: Rule, i: int, target: Target) -> None:
        """Judge a check, or any rule that is not a fold, timed as one phase."""
        timed = rule.name not in untimed_rules()
        with _Phase(self.timer, _rule_label(rule, target), timed=timed) as phase:
            outcome = _contained(rule, target, partial(rule.findings, target))
            if isinstance(outcome, Skip):
                # It looked and found nothing to judge, so it is not charged: a row
                # for it would say a rule was judged that was not (issue #308).
                phase.drop()
            self.settle(rule, i, outcome)

    def rides(self, rule: Check, i: int, target: Target) -> bool:
        """Whether `rule` waits for its scan's pass rather than being judged now.

        A check registered before its scan's first fold rides a pass still to come;
        one registered after it rides the pass already read, and a check whose scan
        no fold asked for has its pass read for it here (issue #314).
        """
        assert rule.ride is not None
        scan = rule.ride.scan
        if (scan.name, i) in self.scanned:
            return False
        if self.order[rule.name] < self.first_fold.get(scan.name, len(self.rules)):
            if _carries(rule, target):
                self.states[(rule.name, i)] = _WAITING
                self.waiting.setdefault((scan.name, i), []).append(rule)
                return True
            return False
        self.scan(scan, i, target)
        return False

    def settle_waiting(self, scan: Scan[Any], i: int, target: Target) -> None:
        """Judge the checks waiting for `scan`'s pass over target `i`, which is over.

        A check that no pass carried reads for itself, through the scan's own
        `finish`: it streams what the blocks would have handed it, and nothing else.
        """
        for rule in self.waiting.pop((scan.name, i), []):
            del self.states[(rule.name, i)]
            self.run_check(rule, i, target)

    def settle_all_waiting(self) -> None:
        """Decide every check still waiting: no fold on its scan ran on its target."""
        for scan_name, i in list(self.waiting):
            rules = self.waiting[(scan_name, i)]
            scan = rules[0].ride.scan if rules[0].ride is not None else None
            assert scan is not None
            self.read_alone(scan, i, self.targets[i])

    def read_alone(self, scan: Scan[Any], i: int, target: Target) -> None:
        """The riders' share of a pass no fold reads: what `finish` streams, alone."""
        self.scanned.add((scan.name, i))
        if not self.waiting.get((scan.name, i)):
            return
        try:
            for rule in self.waiting.get((scan.name, i), []):
                _board(rule, target)
            _finish(scan, target)
            self.settle_waiting(scan, i, target)
        finally:
            target.release(scan.name)

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
                at = self.position.get(id(owner), -1)
                state = self.states.get((name, at))
                if state == _WAITING:
                    # Asked for before its pass: it is decided on its own read now.
                    self.read_alone(self.waiting_scan(name, at), at, owner)
                    state = self.states.get((name, at))
                if state is not None:
                    if _blocks(rule, state):
                        found.append(name)
                    break
                owner = owner.parent
        return found

    def waiting_scan(self, name: str, i: int) -> Scan[Any]:
        """The scan the waiting check `name` rides on target `i`."""
        for (_scan_name, at), rules in self.waiting.items():
            for rule in rules:
                if at == i and rule.name == name and rule.ride is not None:
                    return rule.ride.scan
        raise KeyError(f"{name!r} is not waiting on target {i}")

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
        """Read `scan` once on target `i` for every fold on it that can run there.

        The checks riding the scan share the read. One waiting for this pass is
        decided when the read ends, before any fold is judged, and a fold standing on
        it is judged as if it had been decided before the read: skipped behind it with
        every reason it would have named, or judged (issue #314).
        """
        self.scanned.add((scan.name, i))
        waiting = {rule.name for rule in self.waiting.get((scan.name, i), [])}
        ready, held = self.ready_folds(scan, i, target, waiting)
        riders = self.board(scan, i, target, waiting=False)
        if not ready and not riders:
            # Nothing to measure, so no block to read; a waiting check reads alone.
            self.read_alone(scan, i, target)
            self.settle_held(held, i, target)
            return
        riders += self.board(scan, i, target, waiting=True)
        done = _Pass(scan, target, ready, riders)
        label = _scan_label(scan, target)
        try:
            with log_phase(log, label):
                with self.timer.phase(label):
                    # The folds share the one read, so they are charged to it, not
                    # timed apart: naming them says which rules the phase serves.
                    log.info("%s: %s", label, ", ".join(_serves(ready, riders)))
                    done.drive()
                    _finish(scan, target)
                    self.settle_waiting(scan, i, target)
                    self.settle_held(held, i, target)
                    # The read is over, so what the folds held to measure with goes
                    # before they are judged, not after: on a full-size release the
                    # pass keeps 1.6 GiB, chiefly the Top-Hit Index's tiers, and a
                    # judge words its findings from the totals and the small facts
                    # kept beside them (issue #387).
                    target.release(scan.name)
                    self.conclude(done, i, target, waiting)
        finally:  # an interruption still ends the pass, and what it held goes with it
            target.release(scan.name)

    def ready_folds(
        self, scan: Scan[Any], i: int, target: Target, waiting: set[str]
    ) -> tuple[list[Fold[Any, Any]], list[tuple[Fold[Any, Any], object]]]:
        """The folds that measure in this pass, and those held until it ends.

        A fold standing on a waiting check measures provisionally, unless another
        prerequisite already stops it or its `when` declines or raises; those are
        held, and recorded once the waiting check is decided, as they would have been
        had it been decided first.
        """
        ready: list[Fold[Any, Any]] = []
        held: list[tuple[Fold[Any, Any], object]] = []
        for fold in self.folds_on(scan, i, target):
            before = self.before_read(fold)
            if not waiting.intersection(before):
                if self.ready(fold, i, target, before) and self.applies(fold, i, target):
                    ready.append(fold)
                continue
            outcome = self.provisionally(fold, target, [n for n in before if n not in waiting])
            if outcome is True:
                ready.append(fold)
            else:
                held.append((fold, outcome))
        return ready, held

    def provisionally(self, fold: Fold[Any, Any], target: Target, before: list[str]) -> object:
        """True if `fold` measures; else "blocked", "declined", or what its `when` raised."""
        if self.blockers(fold, target, before):
            return "blocked"
        try:
            return True if fold.when(target) else "declined"
        except Exception as exc:  # this rule's failure, unless a prerequisite stops it
            return exc

    def settle_held(
        self, held: list[tuple[Fold[Any, Any], object]], i: int, target: Target
    ) -> None:
        """Record the held folds now that every prerequisite read before the scan is
        decided: skipped behind each that stops it, else declined, else failed."""
        for fold, outcome in held:
            behind = self.blockers(fold, target, self.before_read(fold))
            if behind:
                self.skip(fold.name, i, behind)
            elif outcome == "declined":
                self.declined.add((fold.name, i))
            elif isinstance(outcome, Exception):
                self.settle(fold, i, [_failure(fold, target, outcome)])

    def board(self, scan: Scan[Any], i: int, target: Target, *, waiting: bool) -> list[_Boarded]:
        """Board the checks riding `scan` on target `i`: those waiting for this pass,
        or those judged after it that nothing has stopped. Returns what each hands
        the blocks to, for those that read the blocks."""
        boarded: list[_Boarded] = []
        if waiting:
            rules = self.waiting.get((scan.name, i), [])
        else:
            rules = [rule for rule in self.riders_on(scan) if self.may_ride(rule, i, target)]
        for rule in rules:
            rider = _board(rule, target)
            if rider is not None:
                boarded.append((rule.name, rider))
        return boarded

    def riders_on(self, scan: Scan[Any]) -> list[Check]:
        return [
            rule
            for rule in self.rules
            if isinstance(rule, Check) and rule.ride is not None and rule.ride.scan is scan
        ]

    def may_ride(self, rule: Check, i: int, target: Target) -> bool:
        """A check judged after the pass rides it unless it is decided, or a
        prerequisite already decided stops it; nothing undecided is decided here."""
        if not self.undecided(rule, i, target):
            return False
        for name in rule.after:
            owner: Target | None = target
            while owner is not None:
                state = self.states.get((name, self.position.get(id(owner), -1)))
                if state is not None:
                    if state != _WAITING and _blocks(rule, state):
                        return False
                    break
                owner = owner.parent
        return True

    def before_read(self, fold: Fold[Any, Any]) -> list[str]:
        """`fold`'s prerequisites decided before its scan is read: those not on the scan."""
        return [name for name in fold.after if name not in self.on_scan[fold.scan.name]]

    def on_its_scan(self, fold: Fold[Any, Any]) -> list[str]:
        """`fold`'s prerequisites on its own scan, decided when the pass ends."""
        return [name for name in fold.after if name in self.on_scan[fold.scan.name]]

    def conclude(self, done: _Pass, i: int, target: Target, waiting: set[str]) -> None:
        """Each fold of a finished pass, in registration order: skipped behind a
        prerequisite on the scan that did not pass, failed, skipped behind its blocker,
        or judged. A prerequisite on the scan is registered earlier, so it is decided
        here first; standing on it, the fold is not judged, as behind any other.

        A fold that measured standing on a check that waited for this pass is
        skipped behind it, if it failed, with the reasons it would have given before
        the read, and nothing it met in the read is reported (issue #314).
        """
        for fold in done.folds:
            if waiting.intersection(fold.after):
                before = self.blockers(fold, target, self.before_read(fold))
                if before:
                    self.skip(fold.name, i, before)
                    continue
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


#: A riding check boarded on a pass, by name, and what it hands each block to.
_Boarded = tuple[str, Rider[Any]]


def _carries(rule: Check, target: Target) -> bool:
    """Whether the pass carries anything `rule` reads; a `when` that raises says no,
    and the check meets the same failure in its own place."""
    assert rule.ride is not None
    try:
        return rule.ride.when(target)
    except Exception:  # judged in its place, where it raises again as its own failure
        return False


def _board(rule: Check, target: Target) -> Rider[Any] | None:
    """What `rule` hands each block to; a ride that fails to board is no ride, and
    the check reads for itself."""
    assert rule.ride is not None
    try:
        return rule.ride.board(target)
    except Exception as exc:  # the check reads for itself instead
        log.info("%s on %s: rides no pass (%s)", rule.name, target.label, exc)
        return None


def _finish(scan: Scan[Any], target: Target) -> None:
    """Read what the riders asked for past the last block; a failure leaves them to
    read for themselves, and they meet it there as their own."""
    if scan.finish is None:
        return
    try:
        scan.finish(target)
    except Exception as exc:  # each rider reads for itself instead
        log.info("%s: the riders' remaining read failed (%s)", _scan_label(scan, target), exc)


def _serves(folds: list[Fold[Any, Any]], riders: list[_Boarded]) -> list[str]:
    """The rules one pass serves: its folds, then the checks riding it."""
    return [fold.name for fold in folds] + [f"{name} (riding)" for name, _rider in riders]


class _Pass:
    """One pass of a scan for its ready folds: each fold's total, failure or blocker.

    A fold that raises while measuring fails alone and stops measuring. A read that
    fails part-way is reported by the first fold still measuring, and blocks the
    others, since a part of the data says nothing about the whole. A plane that
    cannot be decoded (`Unreadable`) is reported by the first fold that reads it
    and blocks only the folds that read it after; one that names the rule reporting
    it is skipped behind that rule, which reads the plane's codes, not its values.

    The checks riding the pass are handed each block after the folds; they report
    nothing here, and one that raises is handed no more blocks (issue #314).

    A scan that can be read apart (`Scan.split`) is spread over the run's forked
    workers (issue #315). Each block is measured wherever it is read, into what it
    holds for the pass (`_Measured`), and added here in block order, through the
    same code a pass read in this process adds it with: the totals, the first
    failure, the plane reported unreadable and every rider's parts are the ones a
    single process reaches, at any worker count.
    """

    def __init__(
        self,
        scan: Scan[Any],
        target: Target,
        folds: list[Fold[Any, Any]],
        riders: list[_Boarded] | None = None,
    ) -> None:
        self.scan, self.target, self.folds = scan, target, folds
        self.riders = list(riders or [])
        self.totals = {fold.name: fold.zero for fold in folds}
        self.failures: dict[str, Finding] = {}
        self.blocked: dict[str, str] = {}
        self.unreadable_by: str | None = None
        #: The riders that left the pass, which no worker need hand a block again.
        self.left: set[str] = set()

    def measuring(self) -> list[Fold[Any, Any]]:
        stopped = self.failures.keys() | self.blocked.keys()
        return [fold for fold in self.folds if fold.name not in stopped]

    def stopped(self) -> frozenset[str]:
        """The folds and riders this pass no longer adds a block's outcome to."""
        return frozenset(self.failures.keys() | self.blocked.keys() | self.left)

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
            with closing(self.measured(label, total)) as blocks:
                for done, measured in enumerate(blocks, start=1):
                    if measured.read_failure is not None:
                        self.read_failed(measured.read_failure)
                        return
                    self.add(measured)
                    overran = self.report_progress(label, done, total, started, overran)
        except PoolFailure:  # the run's failure, not the Store's
            raise
        except Exception as exc:  # the read itself failed, so no fold saw all of it
            self.read_failed(self.scan.failure(self.target, exc))

    def measured(self, label: str, total: int | None) -> Generator[_Measured]:
        """What each block holds for the pass, in block order: read here, or by workers."""
        n_workers = self.target.options.n_workers
        split = None
        if n_workers > 1 and self.scan.split is not None:
            split = self.scan.split(self.target)
        if split is None:
            for block in self.scan.blocks(self.target):
                yield _measured(self.scan, self.target, self.measuring(), self.riders, block)
            return
        work = _Work(self.scan, self.target, split, list(self.folds), list(self.riders))
        units = _units(split, self.scan, self.target)
        first = next(units, None)
        if first is None:
            return
        # The first block is read here, so what the folds and riders load on their
        # first block -- a Top-Hit Index's tiers, say -- is loaded once, before the
        # workers fork, and each inherits it rather than loading a copy of its own.
        yield _measure_unit(work, (first, self.stopped()))
        workers = _pool_size(n_workers, split, label, None if total is None else total - 1)
        yield from self.spread(work, units, workers, label)

    def spread(
        self, work: _Work, units: Iterator[Any], workers: int, label: str
    ) -> Generator[_Measured]:
        """The blocks after the first, read by `workers` forked workers, in block order.

        A worker's failure to read its band comes back as that band's outcome, a
        finding like any other. Anything that escapes the pool itself fails the run
        (`PoolFailure`), naming the band whose result the pass was waiting for: a
        worker that died and broke the pool (`WorkerLost`), a fork the host refused.
        The Store is not judged on a read that never finished.
        """
        in_flight: deque[tuple[int, Any]] = deque()

        def items() -> Iterator[tuple[Any, frozenset[str]]]:
            for band, unit in enumerate(units, start=1):
                in_flight.append((band, unit))
                yield unit, self.stopped()

        try:
            for outcome in ordered_map(_measure_unit, items(), workers, shared=work):
                in_flight.popleft()
                yield outcome
        except BrokenExecutor as exc:
            died = "a forked worker died"
            raise WorkerLost(_awaited(label, died, work.split, in_flight)) from exc
        except Exception as exc:  # the pool's own failure, not a band's
            failed = f"the worker pool failed ({exc!r})"
            raise PoolFailure(_awaited(label, failed, work.split, in_flight)) from exc

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

    def add(self, measured: _Measured) -> None:
        """Add what one block held to every fold still measuring, then to every rider."""
        for fold in self.measuring():
            outcome = measured.folds[fold.name]
            if outcome.message is None:
                try:
                    self.totals[fold.name] = self.totals[fold.name] + outcome.measure
                except Exception as exc:  # this fold's failure, and no other's
                    self.failures[fold.name] = _failure(fold, self.target, exc)
            elif outcome.unreadable:
                self.unreadable(fold, outcome.reported_by, outcome.message)
            else:
                self.failures[fold.name] = fold.finding(
                    self.target, outcome.message, Severity.ERROR
                )
        self.join(measured.parts)

    def join(self, parts: dict[str, _Part]) -> None:
        """Join each riding check's part of the block; one that raised, on either side,
        leaves the pass and judges without it, as if it had not ridden it."""
        for name, rider in list(self.riders):
            part = parts[name]
            try:
                if part.raised is not None:
                    raise part.raised
                rider.join(part.value)
            except Exception as exc:  # that check reads for itself instead
                log.info("%s on %s: left the pass (%s)", name, self.target.label, exc)
                self.riders.remove((name, rider))
                self.left.add(name)

    def unreadable(self, fold: Fold[Any, Any], reported_by: str | None, message: str) -> None:
        """Skip `fold` behind the rule that reports the plane, or report it once here.

        A named reporter that is not measuring in this pass reports nothing, so the
        plane is reported here instead: an undecodable plane is never only a skip.
        """
        if reported_by is not None and reported_by in self.reporting():
            self.blocked[fold.name] = reported_by
        elif self.unreadable_by is None:
            self.unreadable_by = fold.name
            self.failures[fold.name] = fold.finding(self.target, message, Severity.ERROR)
        else:
            self.blocked[fold.name] = self.unreadable_by

    def reporting(self) -> set[str]:
        """The folds of this pass that are still measuring, or have failed: they report."""
        return {fold.name for fold in self.folds if fold.name not in self.blocked}

    def read_failed(self, message: str) -> None:
        """The read stopped part-way: the first fold still measuring reports it, and
        the others stand behind that one."""
        measuring = self.measuring()
        if measuring:
            first = measuring[0]
            self.failures[first.name] = first.finding(self.target, message, Severity.ERROR)
            for fold in measuring[1:]:
                self.blocked[fold.name] = first.name


class PoolFailure(RuntimeError):
    """The worker pool itself failed -- a fork the host refused, say -- so the pass
    cannot finish: a failure of the run, not a finding about the Store (issue #315)."""


class WorkerLost(PoolFailure):
    """A forked worker died -- killed for memory, say -- and broke the pool."""


def _awaited(
    label: str, what: str, split: Split[Any], in_flight: deque[tuple[int, Any]]
) -> str:
    """Words for a pool failure: what failed, and the band whose result the pass was
    waiting for -- the oldest still out, not necessarily the one a dead worker held."""
    if not in_flight:
        return f"{label}: {what} before any band was handed to it"
    band, unit = in_flight[0]
    return (
        f"{label}: {what} while the pass was waiting for the result of band {band} "
        f"({split.describe(unit)}), with {len(in_flight) - 1} more band(s) in flight; "
        "the pass cannot say what they hold"
    )


@dataclass(frozen=True)
class _Outcome:
    """What one fold made of one block: its measure, its failure, or a plane it could
    not read, worded where the block was read."""

    measure: Any = None
    #: The failure's words; for a plane it could not read, the scan's words for it.
    message: str | None = None
    unreadable: bool = False
    #: The rule an unreadable plane names as reporting it, if any.
    reported_by: str | None = None


@dataclass(frozen=True)
class _Part:
    """What one riding check took from one block, or what it raised taking it."""

    value: Any = None
    raised: Exception | None = None


@dataclass(frozen=True)
class _Measured:
    """What one block holds for a pass: each fold's outcome and each rider's part,
    or why the block could not be read (issue #315).

    It is made wherever the block was read and added where the pass is driven, so it
    holds what the block adds to the totals, never the block itself.
    """

    folds: dict[str, _Outcome] = field(default_factory=dict)
    parts: dict[str, _Part] = field(default_factory=dict)
    read_failure: str | None = None


def _measured(
    scan: Scan[Any],
    target: Target,
    folds: Iterable[Fold[Any, Any]],
    riders: Iterable[_Boarded],
    block: Any,
) -> _Measured:
    """Measure one block for every fold given, and take every rider's part of it."""
    outcomes: dict[str, _Outcome] = {}
    for fold in folds:
        try:
            outcomes[fold.name] = _Outcome(measure=fold.measure(target, block))
        except Unreadable as exc:
            outcomes[fold.name] = _Outcome(
                message=scan.failure(target, exc), unreadable=True, reported_by=exc.reported_by
            )
        except Exception as exc:  # this fold's failure, and no other's
            outcomes[fold.name] = _Outcome(message=_failed_words(exc))
    parts: dict[str, _Part] = {}
    for name, rider in riders:
        try:
            parts[name] = _Part(value=rider.part(block))
        except Exception as exc:  # that check reads for itself instead
            parts[name] = _Part(raised=_Left(str(exc)))
    return _Measured(outcomes, parts)


class _Left(Exception):
    """Why a rider left the pass, carried as words from the process that read the block."""


@dataclass(frozen=True)
class _Unsplit:
    """A pass whose units stopped part-way: the scan's words for why."""

    message: str


def _units(split: Split[Any], scan: Scan[Any], target: Target) -> Iterator[Any]:
    """The split's units, in block order, ending with why they stopped if they did.

    The units are cut where the pass is driven -- a band's span of a side table among
    them -- so a failure there is the read's failure at that block, after every block
    before it has been added, exactly as it is when one process reads every block.
    """
    try:
        yield from split.units()
    except Exception as exc:  # the read stops at this block, after the ones before it
        yield _Unsplit(scan.failure(target, exc))


@dataclass(frozen=True)
class _Work:
    """What a pass's workers inherit at the fork: the scan, its target and split, and
    the folds and riders each block is measured for."""

    scan: Scan[Any]
    target: Target
    split: Split[Any]
    folds: list[Fold[Any, Any]]
    riders: list[_Boarded]


def _measure_unit(work: _Work, item: tuple[Any, frozenset[str]]) -> _Measured:
    """Read one unit and measure its block for every fold and rider not yet stopped."""
    unit, stopped = item
    if isinstance(unit, _Unsplit):
        return _Measured(read_failure=unit.message)
    try:
        block = work.split.read(unit)
    except Exception as exc:  # the read itself failed at this block
        return _Measured(read_failure=work.scan.failure(work.target, exc))
    folds = [fold for fold in work.folds if fold.name not in stopped]
    riders = [(name, rider) for name, rider in work.riders if name not in stopped]
    return _measured(work.scan, work.target, folds, riders, block)


def _pool_size(n_workers: int, split: Split[Any], label: str, units: int | None) -> int:
    """How many of `n_workers` the pass forks: no more than it has blocks left for,
    and no more than fit a quarter of the host's available memory beside this
    process, one block's working set a worker (issue #321's `pool_workers`)."""
    return pool_workers(
        n_workers,
        split.unit_bytes,
        what=label,
        unit="band",
        units=units,
        result_bytes=split.result_bytes,
    )


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
    return rule.finding(target, _failed_words(exc), Severity.ERROR)


def _failed_words(exc: Exception) -> str:
    """How a rule that raised is reported: the one wording, wherever it raised."""
    return f"validation failed: {exc}"


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
