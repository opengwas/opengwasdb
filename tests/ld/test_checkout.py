"""`scripts/test_panel/checkout.py`: the one guard the realdata tier and `scripts/make_pcs.py` use
before they record what ran (review round 2 of ogld#5). On a throwaway git repository."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest
from script_modules import SCRIPTS

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
checkout = importlib.import_module("test_panel.checkout")


def git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    """A repository with ogld/, scripts/ and docs/, one commit; the version naming it."""
    for name in ("ogld/__init__.py", "scripts/make.py", "docs/notes.md"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x = 1\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "one")
    return tmp_path, f"0.0.1.dev1+g{git(tmp_path, 'rev-parse', 'HEAD')[:9]}"


@pytest.mark.parametrize(
    ("version", "tags", "names"),
    [
        ("0.0.1.dev71+g95e5f3e7f", [], True),
        ("0.0.1.dev71+g95e5f3e7f.d20261110", [], True),
        ("0.0.1.dev57+g95a6c8276", [], False),
        ("0.2.0", ["v0.2.0"], True),
        ("0.2.0", ["v0.1.0"], False),
        ("0.0.0+unknown", [], False),
    ],
)
def test_an_installed_version_names_the_head_or_does_not(
    version: str, tags: list[str], names: bool
) -> None:
    head = "95e5f3e7f1a2b3c4d5e6f708192a3b4c5d6e7f80"
    assert checkout.names_head(version, head, tags) is names


def test_a_clean_checkout_with_its_own_install_passes(repo: tuple[Path, str]) -> None:
    root, version = repo
    (root / "docs" / "notes.md").write_text("changed, but not code\n")
    (root / "ogld" / "data.tsv").write_text("untracked, but not Python\n")
    assert checkout.problem(root, version) == ""


def test_an_install_that_does_not_name_head_is_refused(repo: tuple[Path, str]) -> None:
    root, _ = repo
    message = checkout.problem(root, "0.0.1.dev1+g000000000")
    assert "0.0.1.dev1+g000000000" in message
    assert git(root, "rev-parse", "HEAD")[:12] in message
    assert "pixi reinstall --locked -e dev ogld" in message


@pytest.mark.parametrize("path", ["ogld/__init__.py", "scripts/make.py"])
def test_a_changed_tracked_file_under_the_code_is_refused(
    repo: tuple[Path, str], path: str
) -> None:
    root, version = repo
    (root / path).write_text("x = 2\n")
    assert path in checkout.problem(root, version)


def test_an_untracked_python_file_under_the_code_is_refused(repo: tuple[Path, str]) -> None:
    root, version = repo
    (root / "scripts" / "loose.py").write_text("y = 1\n")
    assert "scripts/loose.py" in checkout.problem(root, version)
