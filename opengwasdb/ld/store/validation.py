"""The spec's validation rules, by name, on write and on demand (spec §12; ogld#4)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    """One way a store breaks one rule: the rule's name, where, and what is wrong."""

    rule: str
    where: str
    message: str


class ValidationFailed(Exception):
    """A store that breaks a rule. `findings` holds every failing rule's findings."""

    def __init__(self, findings: dict[str, list[Finding]]) -> None:
        failing = sorted(rule for rule, found in findings.items() if found)
        super().__init__(f"the LD Store breaks {', '.join(failing)} (spec §12)")
        self.findings = findings


def validate(location: str | os.PathLike[str]) -> dict[str, list[Finding]]:
    """Every rule of `ogld.format.VALIDATION_RULES` against the store at `location`.

    The result has one key per rule, in the spec's order, whose list is empty when the
    store passes it. Each rule reads in bounded windows and never loads a whole Near
    Band (spec §12; ogld#4).
    """
    raise NotImplementedError("ogld#4")
