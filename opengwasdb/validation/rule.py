"""Validation Rules: what each one judges, what must pass first, and what it found (ADR 0074).

A **Validation Rule** has a stable name, the kinds of target it applies to, and the
rules that must pass on that target before it runs. A **Finding** is one thing a
rule reports about one target: an error, which fails the Store Release, or a
warning, which does not. `ValidationResult` holds a run's findings and the rules it
skipped; `errors`, `warnings` and `ok` are projections of them, unchanged from
before rules had names.

A target is the part of a Store Release a rule is judged against: the release
itself, a Dense grid, an association sequence, a Hybrid's two components taken
together, or an Indexed Variant Subset. One rule written once applies to every
target of its kinds, which is what replaces the per-layout copies of a check.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Flag, StrEnum, auto
from functools import cached_property
from pathlib import Path
from typing import Any, Generic, TypeVar

from opengwasdb.admission.rules import Reason
from opengwasdb.encoding import StoreEncoding
from opengwasdb.model.analyses import AnalysesTable, read_analyses
from opengwasdb.model.enums import PrimaryStorageLayout
from opengwasdb.store.arrays import open_group
from opengwasdb.store.open import OpenGWASDBStore
from opengwasdb.variants import VariantAxis

#: What one scan yields: a block of the data it reads.
B = TypeVar("B")
#: What a fold measures in each block and adds up over the scan.
M = TypeVar("M")
#: What a target remembers having read.
T = TypeVar("T")


class Severity(StrEnum):
    """An error fails the release; a warning is reported and does not.

    A skip is neither: the rule could not judge the target, which is not a pass
    (ADR 0074), so it is reported as its own severity rather than as silence.
    """

    ERROR = "error"
    WARNING = "warning"
    SKIP = "skip"


@dataclass(frozen=True)
class Finding:
    """One thing one rule said about one target: a failure, a warning, or a skip.

    `reasons` is what a skip names: the rules that blocked it, or its own reason for
    having nothing to judge, one entry each and in the order they were recorded.
    `message` joins them, which is what every record says to a reader. A failure or a
    warning names none: its message is its whole reason.
    """

    rule: str
    severity: Severity
    message: str
    target: str
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, str]:
        """This finding as the object `validate --format json` prints (spec §20.1)."""
        return {
            "rule": self.rule,
            "severity": self.severity.value,
            "target": self.target,
            "message": self.message,
        }


class Kind(Flag):
    """What a target is, and so which rules apply to it."""

    #: A Store Release: Dense, Ragged, a Hybrid's outer release, or its Dense Component.
    RELEASE = auto()
    #: A Dense grid: a Dense release's planes, or a Hybrid's Dense Component's.
    GRID = auto()
    #: An association sequence: a Ragged release's, or a Hybrid's Ragged Overflow.
    CSR = auto()
    #: A Hybrid release's two components taken together.
    HYBRID = auto()
    #: One Indexed Variant Subset.
    SUBSET = auto()


@dataclass(frozen=True)
class ValidationOptions:
    """What a caller asks of one run beyond the release itself."""

    source: str | Path | Sequence[str | Path] | None = None
    fidelity_samples: int = 2000
    fidelity_seed: int = 0
    source_assembly: str | None = None
    chain_file: str | Path | None = None


class Target:
    """The part of a Store Release one rule is judged against.

    `label` names it in a finding. `parent` is the target it belongs to: a grid's or
    a sequence's release, a Dense Component's Hybrid release; `children` are the
    targets that belong to it. A rule's prerequisites are looked up on the target
    first, then up the `parent` chain. What several rules read about a target is
    read once, here.
    """

    def __init__(
        self,
        kind: Kind,
        label: str,
        store: OpenGWASDBStore | None = None,
        *,
        parent: Target | None = None,
        options: ValidationOptions | None = None,
        group: Any = None,
        encoding: StoreEncoding | None = None,
    ) -> None:
        self.kind = kind
        self.label = label
        self._store = store
        self._group = group
        self._encoding = encoding
        self._read: dict[str, Any] = {}
        self._until: dict[str, set[str]] = {}
        self.parent = parent
        self.children: list[Target] = []
        if parent is not None:
            parent.children.append(self)
        self.options = options if options is not None else ValidationOptions()

    @property
    def store(self) -> OpenGWASDBStore:
        if self._store is None:
            raise ValueError(f"target {self.label!r} has no Store Release")
        return self._store

    @property
    def layout(self) -> PrimaryStorageLayout:
        return self.store.manifest.primary_layout

    @property
    def encoding(self) -> StoreEncoding:
        """The plan this target's planes are written under: its release's manifest's."""
        return self._encoding if self._encoding is not None else self.store.manifest.encoding

    @property
    def is_component(self) -> bool:
        """Whether this target belongs to a Hybrid release's Dense Component."""
        release = self if self.kind is Kind.RELEASE else self.parent
        return release is not None and release.parent is not None

    @property
    def group(self) -> Any:
        """The Zarr group this target's arrays are in, opened on first use.

        A grid's is its release's `data.zarr`; a sequence's, `data.zarr/ragged`.
        """
        if self._group is None:
            path = self.store.data_path
            self._group = open_group(path / "ragged" if self.kind is Kind.CSR else path)
        return self._group

    def remember(self, key: str, read: Callable[[], T], *, until: str | None = None) -> T:
        """`read()`, once per target: what two rules read about it is read once.

        A value remembered `until` a scan is held for the folds on that scan, and
        released when its pass over this target ends, not kept for the whole run.
        """
        if key not in self._read:
            self._read[key] = read()
        if until is not None:
            self._until.setdefault(until, set()).add(key)
        value: T = self._read[key]
        return value

    def forget(self, key: str) -> Any:
        """Drop one remembered value and return it; None if none was remembered."""
        return self._read.pop(key, None)

    def recalled(self, key: str) -> Any:
        """A value already remembered, or None: asking reads nothing."""
        return self._read.get(key)

    def release(self, scan: str) -> None:
        """Drop what was remembered until `scan`'s pass over this target ended."""
        for key in self._until.pop(scan, set()):
            self._read.pop(key, None)

    @cached_property
    def analyses(self) -> AnalysesTable:
        """The release's `analyses.tsv`, read on first use."""
        return read_analyses(self.store.analyses_path)

    @cached_property
    def variant_axis(self) -> VariantAxis:
        """The release's Store Variant Table, opened on first use and closed by `close`."""
        return VariantAxis(self.store.path)

    def close(self) -> None:
        """Release what this target opened for its rules."""
        axis = self.__dict__.pop("variant_axis", None)
        if axis is not None:
            axis.close()

    def __repr__(self) -> str:
        return f"Target({self.kind.name}, {self.label!r})"


def _always(_target: Target) -> bool:
    return True


@dataclass(frozen=True, kw_only=True)
class Rule:
    """A named judgement of every target of its `kinds` that `when` accepts.

    `after` names the rules that must pass on the same target, or on a target it
    belongs to, before this one runs; a rule that failed or was skipped there skips
    this one. A prerequisite that does not apply to the target constrains nothing.
    A rule with `only_failures_block` is skipped only by a prerequisite that failed:
    one that waits on every rule before it names each failure a skip stands behind,
    and a rule skipped for having nothing to judge says nothing against it.

    `witnesses` marks a rule that judges, after the fact, what Association
    Admission guarantees of every cell it admits (ADR 0072): the Admission
    `Reason`s whose drops keep such a cell out of a Store, or none where the
    guarantee is a cell rule no row's `Reason` names. None for every other rule.
    """

    name: str
    kinds: Kind
    after: tuple[str, ...] = ()
    when: Callable[[Target], bool] = _always
    severity: Severity = Severity.ERROR
    witnesses: tuple[str, ...] | None = None
    only_failures_block: bool = False

    def __post_init__(self) -> None:
        unknown = sorted(set(self.witnesses or ()) - ADMISSION_REASONS)
        if unknown:
            raise ValueError(f"rule {self.name!r} witnesses no Admission Reason {unknown}")

    def judges(self, target: Target) -> bool:
        """Whether the target is of a kind this rule judges; reads nothing."""
        return bool(target.kind & self.kinds)

    def applies_to(self, target: Target) -> bool:
        return self.judges(target) and self.when(target)

    def findings(self, target: Target) -> Iterator[Finding]:
        """Judge `target`; raising is a failure of this rule, reported as one."""
        raise NotImplementedError

    def finding(self, target: Target, message: str, severity: Severity | None = None) -> Finding:
        return Finding(self.name, severity or self.severity, message, target.label)


#: Association Admission's `Reason` values (ADR 0072), which a witness rule names.
ADMISSION_REASONS = frozenset(reason.value for reason in Reason)


class Unreadable(Exception):
    """A plane of one block that cannot be read or decoded; the scan words why.

    Raised from a block's lazily decoded plane, so it reaches only the folds that
    read that plane: the first to meet it reports it, the others that meet it are
    skipped, and a fold that never reads the plane still sees every block. When a
    rule on the same scan already says why the plane cannot be decoded, the plane
    names it in `reported_by`, and every fold that meets it is skipped behind it.
    """

    def __init__(self, message: str, *, reported_by: str | None = None) -> None:
        super().__init__(message)
        self.reported_by = reported_by


class Skip(Exception):
    """Raised by a rule that cannot judge this target; recorded with its reason.

    A skip is not a pass: the rules that run after this one are skipped too.
    """


@dataclass(frozen=True, kw_only=True)
class Check(Rule):
    """A rule that reads what it needs for itself: metadata and small arrays.

    A check with a `ride` reads part of what it judges from a scan's pass instead,
    so that read is not made twice; it is judged as any other check (issue #314).
    """

    run: Callable[[Target], Iterable[str]]
    ride: Ride[Any] | None = None

    def findings(self, target: Target) -> Iterator[Finding]:
        for message in self.run(target):
            yield self.finding(target, message)


def _validation_failed(_target: Target, exc: Exception) -> str:
    return f"validation failed: {exc}"


@dataclass(frozen=True)
class Scan(Generic[B]):
    """One streamed read of a target, shared by every fold that names it.

    `blocks` yields the target's data in order, a block at a time. `failure` words
    a read that fails part-way, which is reported once, by the first fold on the
    scan; the others are skipped. A plane that cannot be decoded (`Unreadable`) is
    worded the same way and reported by the first fold that reads it.
    """

    name: str
    blocks: Callable[[Target], Iterator[B]]
    failure: Callable[[Target, Exception], str] = _validation_failed
    #: How many blocks `blocks` will yield for a target, when that is known before
    #: the read; the runner reports progress against it. None silences progress.
    total: Callable[[Target], int] | None = None
    #: Reads what the pass's riders asked for and the blocks did not reach -- a side
    #: table's entries past the last block -- once the blocks are read.
    finish: Callable[[Target], None] | None = None


def _always_rides(_target: Target) -> bool:
    return True


@dataclass(frozen=True)
class Ride(Generic[B]):
    """What a `Check` takes from a scan's pass rather than reading it again (issue #314).

    `board` is asked before a pass over a target, and returns what to hand each
    block, or None when the check reads nothing from the blocks themselves (a side
    table the scan streams, say). What it reads it leaves on the target, where the
    check finds it; a check whose ride failed, or that no pass carried, reads for
    itself. What a ride hands a check is the data itself, except the primary planes
    `subsets.indexed` takes as digests: there a change to one value of one cell is
    always found, and a corruption of several cells or planes in one comparison unit
    is missed with probability about 2^-64 per unit, under a key drawn per run.

    A check registered before its scan's first fold rides a pass that has not
    happened yet, so it is judged when that pass ends, and the folds on the scan
    that run after it are judged as if it had been decided first. `when` says
    whether the pass carries anything the check reads; one that does not is judged
    in its place.
    """

    scan: Scan[B]
    board: Callable[[Target], Callable[[B], None] | None]
    when: Callable[[Target], bool] = _always_rides


@dataclass(frozen=True, kw_only=True)
class Fold(Rule, Generic[B, M]):
    """A rule judged from one pass of a scan: it measures each block and adds up.

    `measure` returns what one block holds (an int, an array, or a value with
    `+`); the runner adds the measures from `zero` in block order and hands the
    total to `judge`. Every fold on one scan shares its single read.
    """

    scan: Scan[B]
    zero: M
    measure: Callable[[Target, B], M]
    judge: Callable[[Target, M], Iterable[str]]

    def findings(self, target: Target) -> Iterator[Finding]:
        raise TypeError(f"fold {self.name!r} is judged from its scan, {self.scan.name!r}")

    def judged(self, target: Target, total: M) -> Iterator[Finding]:
        for message in self.judge(target, total):
            yield self.finding(target, message)


@dataclass(frozen=True)
class ValidationResult:
    """What a run recorded, in the order it judged it: rule order, then target order.

    A rule says one thing about one target in a `Finding`: it failed, it warned, or
    it could not judge the target and was skipped, with the reason ADR 0074 records
    for that. `errors`, `warnings` and `ok` read the failures and warnings as they
    always have; `skips` and `skipped` read the skips, the second the way it always
    did. A skip alone does not fail the release, and says what the release was not
    judged on rather than reading like a pass.
    """

    findings: tuple[Finding, ...] = ()

    @property
    def errors(self) -> list[str]:
        return [f.message for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[str]:
        return [f.message for f in self.findings if f.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not any(f.severity is Severity.ERROR for f in self.findings)

    def failed(self) -> frozenset[str]:
        """The names of the rules that reported an error."""
        return frozenset(f.rule for f in self.findings if f.severity is Severity.ERROR)

    def of(self, rule: str) -> list[Finding]:
        """Every finding `rule` recorded, a skip included."""
        return [f for f in self.findings if f.rule == rule]

    @property
    def skips(self) -> tuple[Finding, ...]:
        """The skips: each a rule, the target it could not judge, and the reason."""
        return tuple(f for f in self.findings if f.severity is Severity.SKIP)

    @property
    def skipped(self) -> Mapping[str, tuple[str, ...]]:
        """Each skipped rule and why, the reasons in the order the run recorded them."""
        reasons: dict[str, list[str]] = {}
        for finding in self.skips:
            held = reasons.setdefault(finding.rule, [])
            for reason in finding.reasons or (finding.message,):
                if reason not in held:
                    held.append(reason)
        return {rule: tuple(why) for rule, why in reasons.items()}


def check_registry(rules: Sequence[Rule]) -> tuple[Rule, ...]:
    """The rules, refused unless names are unique and every prerequisite comes first.

    Running in registration order then decides every prerequisite before the rule
    that names it. A scan runs when its first fold is reached, so every fold on it
    must run after rules registered before that first fold, or after an earlier fold
    on the same scan: that one is decided when the pass ends, before this is judged.
    """
    seen: set[str] = set()
    first_fold: dict[str, set[str]] = {}
    on_scan: dict[str, set[str]] = {}
    for rule in rules:
        if isinstance(rule, Fold):
            earlier = on_scan.setdefault(rule.scan.name, set())
            _check_fold_order(rule, first_fold.setdefault(rule.scan.name, set(seen)) | earlier)
            earlier.add(rule.name)
        if rule.name in seen:
            raise ValueError(f"validation rule {rule.name!r} is registered twice")
        _check_order(rule, seen, "registered before it")
        seen.add(rule.name)
    return tuple(rules)


def _check_fold_order(fold: Fold[Any, Any], before: set[str]) -> None:
    _check_order(
        fold,
        before,
        f"registered before the first fold on {fold.scan.name!r} or earlier folds on it",
    )


def _check_order(rule: Rule, before: set[str], where: str) -> None:
    late = [name for name in rule.after if name not in before]
    if late:
        raise ValueError(f"validation rule {rule.name!r} runs after {late}, which are not {where}")
