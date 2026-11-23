"""The acceptance runs' one reader of a required input (`OPENGWASDB_LD_*`)."""

from __future__ import annotations

from pathlib import Path

from opengwasdb.ld import env


def named(variable: str, what: str) -> Path:
    """The path `$variable` names. An acceptance run with an input missing fails: it was asked
    for, and a pass would claim it ran."""
    value = env.read(variable, "")
    assert value, f"${variable} must name {what}; it is unset"
    assert Path(value).exists(), f"${variable} must name {what}; {value} is not there"
    return Path(value)
