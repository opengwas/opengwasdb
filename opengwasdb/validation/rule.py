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
from dataclasses import dataclass, field
from enum import Flag, StrEnum, auto
from functools import cached_property
from pathlib import Path
from typing import Any, Generic, TypeVar

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
    """An error fails the release; a warning is reported and does not."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    """One thing one rule found about one target."""

    rule: str
    severity: Severity
    message: str
    target: str


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

    def remember(self, key: str, read: Callable[[], T]) -> T:
        """`read()`, once per target: what two rules read about it is read once."""
        if key not in self._read:
            self._read[key] = read()
        value: T = self._read[key]
        return value

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


#: Association Admission's `Reason` values (ADR 0072, #357), which a witness rule
#: names. Copied from #357's `opengwasdb.admission.rules.Reason` until it merges;
#: tests/test_admission_witnesses.py holds this copy to that enum once it does.
ADMISSION_REASONS = frozenset(
    {
        "chromosome_missing",
        "allele_invalid",
        "se_unusable",
        "effect_missing",
        "z_nonfinite",
        "info_below_threshold",
        "maf_missing",
        "maf_below_threshold",
        "duplicate_collapsed",
        "duplicate_conflicting",
        "off_axis",
    }
)


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
    """A rule that reads what it needs for itself: metadata and small arrays."""

    run: Callable[[Target], Iterable[str]]

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
    """A run's findings, in rule order, and the rules it skipped.

    `skipped` maps each skipped rule to why: the prerequisites that failed or were
    skipped before it, or the reason the rule gave. `errors`, `warnings` and `ok`
    read as they always have.
    """

    findings: tuple[Finding, ...] = ()
    skipped: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

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
        """Every finding `rule` reported."""
        return [f for f in self.findings if f.rule == rule]


def check_registry(rules: Sequence[Rule]) -> tuple[Rule, ...]:
    """The rules, refused unless names are unique and every prerequisite comes first.

    Running in registration order then decides every prerequisite before the rule
    that names it. A scan runs when its first fold is reached, so every fold on it
    must run after rules registered before that first fold.
    """
    seen: set[str] = set()
    first_fold: dict[str, set[str]] = {}
    for rule in rules:
        if isinstance(rule, Fold):
            _check_fold_order(rule, first_fold.setdefault(rule.scan.name, set(seen)))
        if rule.name in seen:
            raise ValueError(f"validation rule {rule.name!r} is registered twice")
        _check_order(rule, seen, "registered before it")
        seen.add(rule.name)
    return tuple(rules)


def _check_fold_order(fold: Fold[Any, Any], before_scan: set[str]) -> None:
    _check_order(fold, before_scan, f"registered before the first fold on {fold.scan.name!r}")


def _check_order(rule: Rule, before: set[str], where: str) -> None:
    late = [name for name in rule.after if name not in before]
    if late:
        raise ValueError(f"validation rule {rule.name!r} runs after {late}, which are not {where}")
