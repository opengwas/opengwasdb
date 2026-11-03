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
from pathlib import Path

from opengwasdb.model.enums import PrimaryStorageLayout
from opengwasdb.store.open import OpenGWASDBStore


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
    a sequence's release, a Dense Component's Hybrid release. A rule's prerequisites
    are looked up on the target first, then up this chain.
    """

    def __init__(
        self,
        kind: Kind,
        label: str,
        store: OpenGWASDBStore | None = None,
        *,
        parent: Target | None = None,
        options: ValidationOptions | None = None,
    ) -> None:
        self.kind = kind
        self.label = label
        self._store = store
        self.parent = parent
        self.options = options if options is not None else ValidationOptions()

    @property
    def store(self) -> OpenGWASDBStore:
        if self._store is None:
            raise ValueError(f"target {self.label!r} has no Store Release")
        return self._store

    @property
    def layout(self) -> PrimaryStorageLayout:
        return self.store.manifest.primary_layout

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
    """

    name: str
    kinds: Kind
    after: tuple[str, ...] = ()
    when: Callable[[Target], bool] = _always
    severity: Severity = Severity.ERROR

    def applies_to(self, target: Target) -> bool:
        return bool(target.kind & self.kinds) and self.when(target)

    def findings(self, target: Target) -> Iterator[Finding]:
        """Judge `target`; raising is a failure of this rule, reported as one."""
        raise NotImplementedError

    def finding(self, target: Target, message: str, severity: Severity | None = None) -> Finding:
        return Finding(self.name, severity or self.severity, message, target.label)


@dataclass(frozen=True, kw_only=True)
class Check(Rule):
    """A rule that reads what it needs for itself: metadata and small arrays."""

    run: Callable[[Target], Iterable[str]]

    def findings(self, target: Target) -> Iterator[Finding]:
        for message in self.run(target):
            yield self.finding(target, message)


@dataclass(frozen=True)
class ValidationResult:
    """A run's findings, in rule order, and the rules it skipped.

    `skipped` maps each skipped rule to the prerequisites that failed or were
    skipped before it. `errors`, `warnings` and `ok` read as they always have.
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
    that names it.
    """
    seen: set[str] = set()
    for rule in rules:
        if rule.name in seen:
            raise ValueError(f"validation rule {rule.name!r} is registered twice")
        unknown = [name for name in rule.after if name not in seen]
        if unknown:
            raise ValueError(
                f"validation rule {rule.name!r} runs after {unknown}, which are not "
                "registered before it"
            )
        seen.add(rule.name)
    return tuple(rules)
