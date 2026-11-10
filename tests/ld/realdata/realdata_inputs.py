"""The realdata tier's inputs, each named by an environment variable that `run.sh` passes on.

A missing input is not a defect of ogld. A test that needs one asks `required` first, before it
calls ogld or runs plink2. Unset, or naming nothing on disk, the test stops there as an
expected failure whose reason names the variable and what it is for; `run.sh` prints those
reasons (`-rfEx`). So it never reaches `ogld.open("")`, where a refusal would read as a
defect, and it never counts as a pass. It is an expected failure, not a skip, because the
escapes gate counts every `pytest.skip` site in the tree (quality.json, `escapes`).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def required(name: str, what: str) -> Path:
    """The path `$name` holds; unset or naming nothing, the test stops here, naming both."""
    value = os.environ.get(name, "")
    if not value:
        pytest.xfail(f"input missing: ${name} is unset ({what})")
    if not Path(value).exists():
        pytest.xfail(f"input missing: ${name} names nothing at {value} ({what})")
    return Path(value)


def names_head(version: str, head: str, tags: list[str]) -> bool:
    """Whether an installed ogld `version` names the checkout at commit `head` (tagged `tags`).

    A development version names its commit, `0.0.1.dev71+g95e5f3e7f` (and `.d<date>` when the
    install was built from a changed tree); a release, `0.2.0`, names the commit tagged `v0.2.0`.
    An editable install keeps the version it was built with (CLAUDE.md), so an LD Store built
    after the checkout moved would record a commit whose code did not build it.
    """
    local = version.partition("+")[2]
    commit = local.split(".")[0]
    if commit.startswith("g") and len(commit) > 7:
        return head.startswith(commit[1:])
    return not local and f"v{version}" in tags


def installed_mismatch() -> str:
    """Why the installed ogld does not name this checkout's HEAD, or "" when it does."""
    import subprocess
    from importlib.metadata import version

    root = Path(__file__).resolve().parents[2]
    git = ["git", "-C", str(root)]
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    tags = subprocess.run([*git, "tag", "--points-at", "HEAD"], capture_output=True, text=True)
    installed = version("ogld")
    if names_head(installed, head.strip(), tags.stdout.split()):
        return ""
    return (
        f"the installed ogld is {installed}, which does not name this checkout's HEAD "
        f"{head.strip()[:12]}: an LD Store built now would record the wrong software.ogld. "
        "Refresh it: pixi reinstall --locked -e dev ogld"
    )
