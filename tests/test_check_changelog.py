"""`scripts/check_changelog.py`: a fragment per pull request (issue #333).

A pull request that changes `opengwasdb/` must add a fragment under
`changelog.d/`, because the merge to `main` becomes a version whose notes are
that pull request's fragments. These tests run the check against a throwaway
git repository, which is the only way to exercise it: what it reads is a diff
between a merge base and `HEAD`, so a fixture that stubbed the diff would
never reach the code that matters.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from script_modules import load_script

changelog = load_script("check_changelog")
release = load_script("release")

SOURCE = "opengwasdb/pkg.py"
FRAGMENT = "changelog.d/333-a-fragment.fixed.md"


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository whose `main` is the merge base of the feature branch."""
    root = tmp_path / "repo"
    (root / "opengwasdb").mkdir(parents=True)
    (root / "changelog.d").mkdir()
    (root / SOURCE).write_text("VALUE = 1\n", encoding="utf-8")
    (root / "changelog.d" / "README.md").write_text("how to write one\n", encoding="utf-8")
    (root / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _commit(root, "base")
    _git(root, "checkout", "-q", "-b", "feature")
    monkeypatch.chdir(root)
    return root


def _commit(root: Path, message: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)


def _write(root: Path, path: str, text: str = "- Something changed.\n") -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_a_fragment_added_alongside_source_changes_is_accepted(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo, SOURCE, "VALUE = 2\n")
    _write(repo, FRAGMENT)
    _commit(repo, "change")

    assert changelog.main(["--base-ref", "main"]) == 0
    assert "changelog.d/" in capsys.readouterr().out


def test_a_fragment_not_committed_yet_still_counts(
    repo: Path,
) -> None:
    """A local run before the first commit of the fragment must not fail.

    CI only ever sees committed files, but the author reads the same message
    locally, and a check that cries wolf teaches people to pass
    `--labels no-changelog` reflexively.
    """
    _write(repo, SOURCE, "VALUE = 2\n")
    _commit(repo, "change")
    _write(repo, FRAGMENT)

    assert changelog.main(["--base-ref", "main"]) == 0


def test_a_staged_fragment_counts(repo: Path) -> None:
    """Staging a fragment is part of writing it, so it counts before the commit.

    Untracked fragments counted but a staged one did not, which is the state
    `git add` leaves a file in and the one an author runs the check from.
    """
    _write(repo, SOURCE, "VALUE = 2\n")
    _commit(repo, "change")
    _write(repo, FRAGMENT)
    _git(repo, "add", FRAGMENT)

    assert changelog.main(["--base-ref", "main"]) == 0


def test_source_changes_without_a_fragment_are_refused(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo, SOURCE, "VALUE = 2\n")
    _commit(repo, "change")

    assert changelog.main(["--base-ref", "main"]) == 1
    error = capsys.readouterr().err
    assert SOURCE in error
    assert "no-changelog" in error
    assert "changelog.d/README.md" in error


def test_editing_the_changelog_is_not_a_fragment(
    repo: Path,
) -> None:
    """The old rule's intent survives: a shared-file edit is not a note.

    `CHANGELOG.md` keeps the history up to the cutover and is not the place a
    change announces itself any more, so touching it must not satisfy the
    check.
    """
    _write(repo, SOURCE, "VALUE = 2\n")
    _write(repo, "CHANGELOG.md", "# Changelog\n\nSomething.\n")
    _commit(repo, "change")

    assert changelog.main(["--base-ref", "main"]) == 1


def test_the_no_changelog_label_exempts_the_pull_request(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo, SOURCE, "VALUE = 2\n")
    _commit(repo, "change")

    assert changelog.main(["--base-ref", "main", "--labels", "chore,no-changelog"]) == 0
    assert "no-changelog" in capsys.readouterr().out


def test_a_change_outside_the_watched_paths_needs_no_fragment(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo, "docs/getting-started.md", "text\n")
    _commit(repo, "docs")

    assert changelog.main(["--base-ref", "main"]) == 0
    assert "not required" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["333-a.security.md", "333-a.md", "333-a.fixed.txt"])
def test_a_fragment_whose_type_cannot_be_read_is_refused(
    repo: Path, name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fragment the release cannot classify is refused by both scripts."""
    _write(repo, SOURCE, "VALUE = 2\n")
    _write(repo, f"changelog.d/{name}")
    _commit(repo, "change")

    assert changelog.main(["--base-ref", "main"]) == 1
    assert name in capsys.readouterr().err


def test_an_empty_fragment_is_refused(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo, SOURCE, "VALUE = 2\n")
    _write(repo, FRAGMENT, "\n\n")
    _commit(repo, "change")

    assert changelog.main(["--base-ref", "main"]) == 1
    assert "empty" in capsys.readouterr().err


def test_a_broken_fragment_is_refused_even_when_the_label_is_applied(
    repo: Path,
) -> None:
    """The label excuses the absence of notes, not a note nothing can read."""
    _write(repo, SOURCE, "VALUE = 2\n")
    _write(repo, "changelog.d/333-a.security.md")
    _commit(repo, "change")

    assert changelog.main(["--base-ref", "main", "--labels", "no-changelog"]) == 1


def test_both_scripts_know_one_fragment_vocabulary() -> None:
    """`check_changelog` accepts exactly what `release` can classify.

    The two scripts are separate entry points -- neither may import the other
    -- so the vocabulary is stated in both and held equal here. A type accepted
    by the check but unknown to the release would fail the release workflow
    after the merge, where the pull request's author can no longer see it.
    """
    assert changelog.FRAGMENT_TYPES == release.FRAGMENT_TYPES
    assert changelog.FRAGMENT_PATTERN.pattern == release.FRAGMENT_PATTERN.pattern
