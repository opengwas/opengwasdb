"""The one reader of the `OPENGWASDB_LD_*` environment variables, and the refusal of the old names.

`opengwasdb.ld` came from the `ogld` package, whose scripts and tests read `OGLD_*` variables.
Those are now `OPENGWASDB_LD_*`. A variable under the old name that is still set would be read
by nothing and its value silently ignored, so the default would apply in its place: a build
would run on a root the person did not mean. `read` therefore raises `LegacyVariable`, naming the
new name, whenever any old one is set. It prints nothing: whatever faces the terminal reports
(`opengwasdb ld` for the command line, `scripts/ld/test_panel/roots.py` for the scripts, pytest
for the test helpers). `OGDB_STORE_TIMEOUT_SECONDS` is not one of them: it is the
API service's own live setting, which `opengwasdb ld` never reads.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from opengwasdb.ld.format import InputError

PREFIX = "OPENGWASDB_LD_"
_LEGACY_PREFIX = "OGLD_"


class LegacyVariable(InputError):
    """An old variable name is set; the message names the one that replaces it.

    An ordinary `InputError`, so the CLI's refusal mapping reports it with exit 2 and a test
    helper shows its message under pytest; a script turns it into exit 2 at one place,
    `scripts/ld/test_panel/roots.py`.
    """


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
    refuse_legacy()
    return os.environ.get(name, default)
