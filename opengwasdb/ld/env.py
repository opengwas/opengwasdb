"""The one reader of the `OPENGWASDB_LD_*` environment variables, and the refusal of the old names.

`opengwasdb.ld` came from the `ogld` package, whose scripts and tests read `OGLD_*` variables.
Those are now `OPENGWASDB_LD_*`. A variable under the old name that is still set would be read
by nothing and its value silently ignored, so the default would apply in its place: a build
would run on a root the person did not mean. `read` therefore refuses, naming the new name,
whenever any old one is set. `OGDB_STORE_TIMEOUT_SECONDS`, which no code here reads, is
refused the same way for the same reason (ADR 0094).
"""

from __future__ import annotations

import os
from collections.abc import Mapping

PREFIX = "OPENGWASDB_LD_"
_LEGACY_PREFIX = "OGLD_"
_LEGACY_NAMES = {"OGDB_STORE_TIMEOUT_SECONDS": "OPENGWASDB_STORE_TIMEOUT_SECONDS"}


class LegacyVariable(SystemExit):
    """An old variable name is set; the message names the one that replaces it. A `SystemExit`, so
    a script that reads through `read` stops with the message and no traceback."""


def replacement(name: str) -> str | None:
    """The new name for an old variable `name`, or None when `name` is not an old one."""
    if name.startswith(_LEGACY_PREFIX):
        return PREFIX + name.removeprefix(_LEGACY_PREFIX)
    return _LEGACY_NAMES.get(name)


def refuse_legacy(environ: Mapping[str, str] | None = None) -> None:
    """Raise `LegacyVariable` naming every old variable set and its replacement.

    Covers the dynamic `OGLD_STORE_<panel>` too, by prefix rather than by a list.
    """
    environment = os.environ if environ is None else environ
    old = sorted(name for name in environment if replacement(name) is not None)
    if old:
        renames = ", ".join(f"{name} is now {replacement(name)}" for name in old)
        raise LegacyVariable(f"an old environment variable is set: {renames}")


def read(name: str, default: str) -> str:
    """The value of `name` (an `OPENGWASDB_LD_*` variable), else `default`.

    Refuses first if any old variable is set, so no old name goes unread unnoticed.
    """
    if not name.startswith(PREFIX):
        raise ValueError(f"{name!r} is not an {PREFIX}* variable")
    refuse_legacy()
    return os.environ.get(name, default)
