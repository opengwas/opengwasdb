"""Which test an Analysis's source p-value comes from (issue #371, ADR 0085).

A Store keeps z (= beta / se) and SE, and a query answers with the two-sided
normal p of that z (spec §6). The source's own p-value is read once, at
resolve time, as a witness against beta/se (ADR 0069). It is a witness only
when it is the Wald p of the stored coefficient. Whether it is was measured on
OGS-00011's 4,783 sources (#371): no statistic of the file recovers it, so the
Source Collection declares it per Analysis, and no row goes without one.

- `wald_1df`: p is the Wald p of beta/se; the check compares the two.
- `joint_2df`: p is a joint test of the stored (interaction) coefficient and a
  main effect together. beta/se is a valid z, but p cannot witness it, so the
  check is skipped (`joint_test_p_value`).
- `mr_mega`: MR-MEGA scales its SE by the residual mean square, so beta/se is
  a t-statistic on (cohorts - 2) df and not a z, and its p is a separate 2-df
  chi-square. The resolver refuses the Analysis before reading it.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from enum import StrEnum
from pathlib import Path

__all__ = [
    "P_VALUE_TEST_COLUMN",
    "PValueTest",
    "parse_p_value_test",
    "require_p_value_test_column",
]

#: The resolver manifest's column. Required, with no default.
P_VALUE_TEST_COLUMN = "p_value_test"


class PValueTest(StrEnum):
    """The test a source's `p_value` column comes from."""

    WALD_1DF = "wald_1df"
    JOINT_2DF = "joint_2df"
    MR_MEGA = "mr_mega"


_ALLOWED = ", ".join(test.value for test in PValueTest)


def require_p_value_test_column(fieldnames: Collection[str], path: str | Path) -> None:
    """Refuse a manifest that does not declare every Analysis's p-value test."""
    if P_VALUE_TEST_COLUMN not in fieldnames:
        raise ValueError(
            f"analyses manifest {path} is missing required column "
            f"{P_VALUE_TEST_COLUMN!r}: every Analysis declares which test its source "
            f"p-value comes from ({_ALLOWED}); there is no default (issue #371)"
        )


def parse_p_value_test(row: Mapping[str, str]) -> PValueTest:
    """One row's declared test; a blank or unknown value raises."""
    raw = (row.get(P_VALUE_TEST_COLUMN) or "").strip()
    try:
        return PValueTest(raw)
    except ValueError as exc:
        raise ValueError(
            f"invalid {P_VALUE_TEST_COLUMN} {raw!r}; expected one of {_ALLOWED}"
        ) from exc
