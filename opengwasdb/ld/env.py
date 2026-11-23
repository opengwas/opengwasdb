"""The one reader of the `OPENGWASDB_LD_*` environment variables, and the refusal of the old names.

`opengwasdb.ld` came from the `ogld` package, whose scripts and tests read `OGLD_*` variables.
Those are now `OPENGWASDB_LD_*`. A variable under the old name that is still set would be read
by nothing and its value silently ignored, so the default would apply in its place: a build
would run on a root the person did not mean. `read` therefore refuses, naming the new name,
whenever any old one is set. `OGDB_STORE_TIMEOUT_SECONDS` is not one of them: it is the
API service's own live setting, which `opengwasdb ld` never reads.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping

PREFIX = "OPENGWASDB_LD_"
_LEGACY_PREFIX = "OGLD_"


class LegacyVariable(SystemExit):
    """An old variable name is set; `message` names the one that replaces it.

    A `SystemExit` with status 2, the status every refusal has (1 is "the store breaks a rule"),
    so a script that reads through `read` stops with no traceback. `read` prints `message` first.
    """

    def __init__(self, message: str) -> None:
        super().__init__(2)
        self.message = message


def replacement(name: str) -> str | None:
    """The new name for an old variable `name`, or None when `name` is not an old one."""
    if name.startswith(_LEGACY_PREFIX):
        return PREFIX + name.removeprefix(_LEGACY_PREFIX)
    return None


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
    try:
        refuse_legacy()
    except LegacyVariable as refused:
        print(refused.message, file=sys.stderr)
        raise
    return os.environ.get(name, default)
