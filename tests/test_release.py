"""`scripts/release.py`: the next version and its notes (issue #333).

Every merge to `main` is a version. The workflow finds the PR the merge came
from and the fragments that PR added, then calls this script, which owns the
two decisions the workflow must not make itself: how the fragment types set
the bump, and what the release notes say. The tests here are the specification
of both, including the cases with nothing to read -- no previous tag, no PR,
no fragments -- where a silent default would produce a plausible wrong
version.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from script_modules import load_script

release = load_script("release")

REPO = "opengwas/opengwasdb"


def _fragment(tmp_path: Path, name: str, text: str = "- Something changed.\n") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ── The bump rule ───────────────────────────────────────────────────────────


def test_a_fix_is_a_patch() -> None:
    assert release.next_version("v0.3.0", ["fixed"]) == "0.3.1"
    assert release.next_version("v0.3.0", ["internal"]) == "0.3.1"


def test_an_added_changed_or_breaking_fragment_is_a_minor() -> None:
    for kind in ("breaking", "added", "changed"):
        assert release.next_version("v0.3.1", [kind]) == "0.4.0", kind


def test_the_strongest_fragment_type_wins() -> None:
    assert release.next_version("v0.3.0", ["fixed", "added", "internal"]) == "0.4.0"


def test_no_previous_tag_starts_from_zero() -> None:
    """Nothing to bump: a fix is 0.0.1, a feature 0.1.0.

    The workflow always has a tag after the cutover, but a repository whose
    tags were lost must still produce a version, and one that is obviously the
    first rather than a plausible continuation.
    """
    assert release.next_version(None, ["fixed"]) == "0.0.1"
    assert release.next_version(None, ["added"]) == "0.1.0"
    assert release.next_version(None, []) == "0.0.1"


def test_a_pull_request_with_no_fragments_is_a_patch() -> None:
    plan = release.plan_release(
        previous_tag="v0.3.0", fragments=[], pr_number=7, commit_subject="Fix a thing"
    )
    assert plan.version == "0.3.1"
    assert plan.tag == "v0.3.1"
    assert "No changelog fragments" in plan.notes
    assert f"{REPO}/pull/7" in plan.notes


def test_a_push_without_a_pull_request_is_a_patch_that_names_its_commit() -> None:
    plan = release.plan_release(
        previous_tag="v0.3.0",
        fragments=[],
        pr_number=None,
        commit_subject="Correct the allele frequency orientation (#400)",
    )
    assert plan.version == "0.3.1"
    assert "Correct the allele frequency orientation (#400)" in plan.notes
    assert "/pull/" not in plan.notes, "there is no pull request to link to"


def test_an_unknown_fragment_type_is_refused() -> None:
    with pytest.raises(release.ReleaseError, match="security"):
        release.next_version("v0.3.0", ["security"])


@pytest.mark.parametrize("tag", ["0.3.0", "v0.3", "v0.3.0-rc1", "release-v0.3.0", "v"])
def test_a_tag_that_is_not_a_release_is_refused(tag: str) -> None:
    """One tag convention, and it is checked: the bump needs real numbers."""
    with pytest.raises(release.ReleaseError):
        release.next_version(tag, ["fixed"])


# ── The fragments ───────────────────────────────────────────────────────────


def test_fragments_are_read_from_slug_dot_type_dot_md(tmp_path: Path) -> None:
    path = _fragment(tmp_path, "333-release-script.changed.md", "- Versions come from tags.\n")
    fragments = release.read_fragments([path])
    assert [(f.kind, f.text) for f in fragments] == [
        ("changed", "- Versions come from tags.")
    ]


@pytest.mark.parametrize(
    "name", ["fixed.md", "333.fixed.txt", "333.unknown.md", ".fixed.md"]
)
def test_a_name_that_is_not_slug_dot_type_dot_md_is_refused(
    tmp_path: Path, name: str
) -> None:
    path = _fragment(tmp_path, name)
    with pytest.raises(release.ReleaseError, match=name):
        release.read_fragments([path])


def test_an_empty_fragment_is_refused(tmp_path: Path) -> None:
    """A fragment that says nothing is a release note that looks like one."""
    path = _fragment(tmp_path, "333-empty.fixed.md", "  \n\n")
    with pytest.raises(release.ReleaseError, match="empty"):
        release.read_fragments([path])


def test_the_directories_readme_is_not_a_fragment(tmp_path: Path) -> None:
    """`changelog.d/README.md` explains the format, so it is skipped, not read.

    The workflow filters it out of the diff by name, but the script is also run
    by hand, and `--fragment changelog.d/*.md` is the obvious way to try a
    release: refusing that glob would look like a bug in the fragments.
    """
    readme = _fragment(tmp_path, "README.md", "how to write a fragment\n")
    real = _fragment(tmp_path, "333-a.fixed.md", "- Fixed the thing.\n")
    assert [f.kind for f in release.read_fragments([readme, real])] == ["fixed"]


# ── The notes ───────────────────────────────────────────────────────────────


def test_the_notes_are_the_fragments_under_their_type_and_link_the_pr(
    tmp_path: Path,
) -> None:
    fixed = _fragment(tmp_path, "333-a.fixed.md", "- Fixed the thing.\n")
    added = _fragment(tmp_path, "333-b.added.md", "- Added the other thing.\n")
    plan = release.plan_release(
        previous_tag="v0.2.0",
        fragments=release.read_fragments([fixed, added]),
        pr_number=12,
        commit_subject="Merge #12",
    )
    assert plan.version == "0.3.0"
    assert plan.notes.index("### Added") < plan.notes.index("### Fixed")
    assert "- Added the other thing." in plan.notes
    assert "- Fixed the thing." in plan.notes
    assert f"https://github.com/{REPO}/pull/12" in plan.notes


def test_the_notes_keep_every_fragment_of_one_type(tmp_path: Path) -> None:
    first = _fragment(tmp_path, "333-a.fixed.md", "- First fix.\n")
    second = _fragment(tmp_path, "333-b.fixed.md", "- Second fix.\n")
    plan = release.plan_release(
        previous_tag="v0.2.0",
        fragments=release.read_fragments([first, second]),
        pr_number=12,
    )
    assert plan.notes.count("### Fixed") == 1
    assert "- First fix." in plan.notes and "- Second fix." in plan.notes


# ── The command line the workflow calls ────────────────────────────────────


def test_the_cli_prints_the_version_and_writes_the_notes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fragment = _fragment(tmp_path, "333-a.fixed.md", "- Fixed the thing.\n")
    listed = tmp_path / "fragments.txt"
    listed.write_text(f"{fragment}\n\n", encoding="utf-8")
    notes_out = tmp_path / "notes.md"

    status = release.main(
        [
            "--previous-tag",
            "v0.2.0",
            "--fragments-from",
            str(listed),
            "--pr-number",
            "12",
            "--notes-out",
            str(notes_out),
        ]
    )

    assert status == 0
    assert capsys.readouterr().out.strip() == "0.2.1"
    assert "- Fixed the thing." in notes_out.read_text(encoding="utf-8")
