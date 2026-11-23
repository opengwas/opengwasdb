"""Whether a checkout can be cited as the code that ran: the one guard for evidence.

An LD Store records the installed ogld's version (spec §7), and `scripts/make_pcs.py`'s reports
record the commit that made them (build-inputs §8). Either is true only when the code that ran
is that commit's: the editable install's version names HEAD (CLAUDE.md: it is frozen when it is
built), and nothing under `ogld/` or `scripts/` differs from HEAD, tracked or not. The realdata
tier and `make_pcs.py` both call `problem` (review round 2 of ogld#5), so they agree on what
"clean" means.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

#: Where code that runs lives: a change here, committed or not, changes what ran.
CODE = ("opengwasdb", "scripts")


def names_head(version: str, head: str, tags: list[str]) -> bool:
    """Whether an installed ogld `version` names the checkout at commit `head` (tagged `tags`).

    A development version names its commit, `0.0.1.dev71+g95e5f3e7f` (and `.d<date>` when the
    install was built from a changed tree); a release, `0.2.0`, names the commit tagged
    `v0.2.0`.
    """
    local = version.partition("+")[2]
    commit = local.split(".")[0]
    if commit.startswith("g") and len(commit) > 7:
        return head.startswith(commit[1:])
    return not local and f"v{version}" in tags


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed in {root}: {done.stderr.strip()}")
    return done.stdout


def problem(root: Path, installed: str) -> str:
    """Why the checkout at `root`, with ogld `installed`, cannot be cited as the code that ran,
    or "" when it can. It refuses an installed version that does not name HEAD, a tracked file
    changed under `ogld/` or `scripts/`, and an untracked `.py` file there."""
    head = _git(root, "rev-parse", "HEAD").strip()
    return _install_problem(root, installed, head) or _tree_problem(root, head)


def _install_problem(root: Path, installed: str, head: str) -> str:
    if names_head(installed, head, _git(root, "tag", "--points-at", "HEAD").split()):
        return ""
    return (
        f"the installed opengwasdb is {installed}, which does not name this checkout's HEAD "
        f"{head[:12]}: refresh it with pixi reinstall --locked -e dev opengwasdb"
    )


def _tree_problem(root: Path, head: str) -> str:
    changed, loose = _differences(root)
    if changed:
        return f"tracked files differ from HEAD {head[:12]}: {', '.join(changed[:5])}: commit them"
    if loose:
        return f"untracked Python files would run: {', '.join(loose[:5])}: commit or remove them"
    return ""


def _differences(root: Path) -> tuple[list[str], list[str]]:
    """The tracked files under the code that differ from HEAD, and the untracked `.py` there."""
    status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", *CODE)
    changed, loose = [], []
    for line in status.splitlines():
        untracked, path = line.startswith("??"), line[3:]
        if not untracked:
            changed.append(path)
        elif path.endswith(".py"):
            loose.append(path)
    return changed, loose
