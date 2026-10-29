"""`scripts/release.py`: the releases `main` is owed (issue #333).

Every merge to `main` is a version, so the script has to answer three things
that the workflow must not decide itself:

* **What each merge is**, from the fragment types it added: the bump rule.
* **Which merges are owed a release**, from the tags and the Releases that
  exist: reconciliation. A concurrency queue cannot be relied on to hand every
  merge a run -- GitHub keeps one pending run per group and drops the rest --
  so each run publishes every first-parent commit after the newest tag that has
  no tag yet, and any tagged commit that never got its Release, oldest first.
* **What publication still needs doing** for each of those, so a run that dies
  between pushing a tag and creating its Release is finished by the next one
  instead of leaving half a release behind.

The tests here are the specification of all three. The discovery and
publication tests use throwaway git repositories and a stub `gh` on `PATH`,
because the bugs they guard against (a git failure read as "no tag yet", a tag
read as a finished release) are exactly the ones a stubbed subprocess would
hide.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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


def test_an_unknown_fragment_type_is_refused() -> None:
    with pytest.raises(release.ReleaseError, match="security"):
        release.next_version("v0.3.0", ["security"])


@pytest.mark.parametrize("tag", ["0.3.0", "v0.3", "v0.3.0-rc1", "release-v0.3.0", "v"])
def test_a_tag_that_is_not_a_release_is_refused(tag: str) -> None:
    """One tag convention, and it is checked: the bump needs real numbers."""
    with pytest.raises(release.ReleaseError):
        release.next_version(tag, ["fixed"])


def test_the_latest_tag_is_the_highest_version_not_the_highest_text() -> None:
    tags = {"v0.9.0": "a", "v0.10.0": "b", "v0.2.0": "c"}
    assert release.latest_tag(tags) == "v0.10.0"


def test_no_v_tags_at_all_means_no_previous_tag() -> None:
    assert release.latest_tag({}) is None


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
    """`changelog.d/README.md` explains the format, so it is skipped, not read."""
    readme = _fragment(tmp_path, "README.md", "how to write a fragment\n")
    real = _fragment(tmp_path, "333-a.fixed.md", "- Fixed the thing.\n")
    assert [f.kind for f in release.read_fragments([readme, real])] == ["fixed"]


# ── The notes ───────────────────────────────────────────────────────────────


def _merge(
    *,
    commit: str = "a" * 40,
    subject: str = "Merge #12",
    fragments: list[Path] = (),
    pr_number: int | None = 12,
    tag: str | None = None,
) -> object:
    return release.Merge(
        commit=commit,
        subject=subject,
        fragments=[str(p) for p in fragments],
        pr_number=pr_number,
        tag=tag,
    )


def test_the_notes_are_the_fragments_under_their_type_and_link_the_pr(
    tmp_path: Path,
) -> None:
    fixed = _fragment(tmp_path, "333-a.fixed.md", "- Fixed the thing.\n")
    added = _fragment(tmp_path, "333-b.added.md", "- Added the other thing.\n")
    (planned,) = release.plan_releases(
        tags={"v0.2.0": "0" * 40},
        merges=[_merge(fragments=[fixed, added], pr_number=12)],
        repository=REPO,
    )
    assert planned.version == "0.3.0"
    assert planned.tag == "v0.3.0"
    assert planned.notes.index("### Added") < planned.notes.index("### Fixed")
    assert "- Added the other thing." in planned.notes
    assert "- Fixed the thing." in planned.notes
    assert f"https://github.com/{REPO}/pull/12" in planned.notes


def test_the_notes_keep_every_fragment_of_one_type(tmp_path: Path) -> None:
    first = _fragment(tmp_path, "333-a.fixed.md", "- First fix.\n")
    second = _fragment(tmp_path, "333-b.fixed.md", "- Second fix.\n")
    (planned,) = release.plan_releases(
        tags={"v0.2.0": "0" * 40},
        merges=[_merge(fragments=[first, second])],
        repository=REPO,
    )
    assert planned.notes.count("### Fixed") == 1
    assert "- First fix." in planned.notes and "- Second fix." in planned.notes


def test_a_pull_request_with_no_fragments_is_a_patch_that_says_so(
    tmp_path: Path,
) -> None:
    (planned,) = release.plan_releases(
        tags={"v0.3.0": "0" * 40},
        merges=[_merge(fragments=[], pr_number=7, subject="Fix a thing")],
        repository=REPO,
    )
    assert planned.version == "0.3.1"
    assert "No changelog fragments" in planned.notes
    assert f"{REPO}/pull/7" in planned.notes


def test_a_push_without_a_pull_request_is_a_patch_that_names_its_commit() -> None:
    (planned,) = release.plan_releases(
        tags={"v0.3.0": "0" * 40},
        merges=[
            _merge(
                fragments=[],
                pr_number=None,
                subject="Correct the allele frequency orientation (#400)",
            )
        ],
        repository=REPO,
    )
    assert planned.version == "0.3.1"
    assert "Correct the allele frequency orientation (#400)" in planned.notes
    assert "/pull/" not in planned.notes, "there is no pull request to link to"


def test_a_push_without_a_pull_request_still_uses_its_fragments(
    tmp_path: Path,
) -> None:
    """The fragments set the bump whether or not a pull request was found.

    A direct push that adds an `added` fragment is a minor, its notes are the
    fragment, and the commit subject is named beside it. Saying "no fragments
    were read" there was wrong, and published a patch with empty notes.
    """
    added = _fragment(tmp_path, "400-a.added.md", "- Added the thing.\n")
    (planned,) = release.plan_releases(
        tags={"v0.3.0": "0" * 40},
        merges=[
            _merge(
                fragments=[added],
                pr_number=None,
                subject="Add the thing (#400)",
            )
        ],
        repository=REPO,
    )
    assert planned.version == "0.4.0"
    assert "- Added the thing." in planned.notes
    assert "Add the thing (#400)" in planned.notes
    assert "No changelog fragments" not in planned.notes


# ── Reconciliation: which merges are owed a release ─────────────────────────


def test_three_untagged_merges_are_three_versions_in_ancestry_order(
    tmp_path: Path,
) -> None:
    """One run publishes every merge whose run was lost, oldest first."""
    fix = _fragment(tmp_path, "1-a.fixed.md", "- Fixed one.\n")
    feature = _fragment(tmp_path, "2-b.added.md", "- Added two.\n")
    internal = _fragment(tmp_path, "3-c.internal.md", "- Internal three.\n")
    planned = release.plan_releases(
        tags={"v0.3.0": "0" * 40},
        merges=[
            _merge(commit="1" * 40, fragments=[fix], pr_number=1),
            _merge(commit="2" * 40, fragments=[feature], pr_number=2),
            _merge(commit="3" * 40, fragments=[internal], pr_number=3),
        ],
        repository=REPO,
    )
    assert [(p.commit, p.version) for p in planned] == [
        ("1" * 40, "0.3.1"),
        ("2" * 40, "0.4.0"),
        ("3" * 40, "0.4.1"),
    ]
    assert all(p.needs_tag and p.needs_release for p in planned)


def test_one_untagged_merge_is_one_version(tmp_path: Path) -> None:
    fix = _fragment(tmp_path, "1-a.fixed.md", "- Fixed one.\n")
    planned = release.plan_releases(
        tags={"v0.3.0": "0" * 40},
        merges=[_merge(fragments=[fix])],
        repository=REPO,
    )
    assert [p.version for p in planned] == ["0.3.1"]


def test_nothing_untagged_plans_nothing() -> None:
    assert release.plan_releases(
        tags={"v0.3.0": "0" * 40}, merges=[], repository=REPO
    ) == []


def test_a_tag_without_a_release_is_published_not_retagged(tmp_path: Path) -> None:
    """The state a failed `gh release create` leaves: the tag exists.

    Rerunning creates the missing Release and must not tag again — the plan says
    so, so each publishing step is idempotent on its own.
    """
    fix = _fragment(tmp_path, "1-a.fixed.md", "- Fixed one.\n")
    (planned,) = release.plan_releases(
        tags={"v0.3.1": "1" * 40, "v0.3.0": "0" * 40},
        merges=[_merge(commit="1" * 40, fragments=[fix], tag="v0.3.1")],
        repository=REPO,
    )
    assert planned.version == "0.3.1"
    assert planned.tag == "v0.3.1"
    assert planned.needs_tag is False
    assert planned.needs_release is True
    assert "- Fixed one." in planned.notes


def test_a_version_whose_tag_already_exists_is_refused() -> None:
    """Never publish a version twice.

    `plan_releases` numbers from the highest tag, so it cannot collide with one;
    this is the guard on the step that could — a caller that names the previous
    tag (the `next` command, or a later reconciliation) — and `publish` checks
    the tags actually present before it moves anything.
    """
    with pytest.raises(release.ReleaseError, match="v0.3.1"):
        release._plan_one(
            previous_tag="v0.3.0",
            merge=_merge(commit="1" * 40, fragments=[]),
            tags={"v0.3.0": "0" * 40, "v0.3.1": "9" * 40},
            repository=REPO,
        )


# ── Discovery: the tags, the commits and their fragments ────────────────────


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    (root / "changelog.d").mkdir(parents=True)
    (root / "changelog.d" / "README.md").write_text("how to write one\n", encoding="utf-8")
    (root / "opengwasdb").mkdir()
    (root / "opengwasdb" / "pkg.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "init", "-q", "-b", "main")
    for name, value in (
        ("user.email", "test@example.com"),
        ("user.name", "Test"),
        ("tag.gpgSign", "false"),
    ):
        _git(root, "config", name, value)
    _commit(root, "base")
    _git(root, "tag", "--annotate", "v0.3.0", "--message", "Release v0.3.0")
    monkeypatch.chdir(root)
    return root


def _add(root: Path, name: str, text: str, message: str) -> str:
    path = root / "changelog.d" / name
    path.write_text(text, encoding="utf-8")
    return _commit(root, message)


def test_tags_are_read_from_v_star_only(repo: Path) -> None:
    """A stray `99.0.0` is not a version, and must not become one."""
    _git(repo, "tag", "99.0.0")
    assert list(release.tag_commits()) == ["v0.3.0"]


def test_a_git_failure_is_an_error_not_an_empty_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`|| true` in the shell reads this as "no tag yet" and releases 0.0.1."""
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    monkeypatch.chdir(outside)
    with pytest.raises(release.ReleaseError, match="git"):
        release.tag_commits()


def test_pending_lists_every_first_parent_commit_after_the_latest_tag(
    repo: Path,
) -> None:
    second = _add(repo, "1-a.fixed.md", "- Fixed one.\n", "fix")
    third = _add(repo, "2-b.added.md", "- Added two.\n", "feature")
    assert release.pending(
        tags=release.tag_commits(), released={"v0.3.0"}
    ) == [second, third]


def test_a_tagged_commit_in_the_middle_is_not_published_again(repo: Path) -> None:
    """A merge that already has its tag is not re-released, and the merges after
    it are numbered from it."""
    first = _add(repo, "1-a.fixed.md", "- Fixed one.\n", "fix one")
    _git(repo, "tag", "--annotate", "v0.3.1", "--message", "Release v0.3.1")
    second = _add(repo, "2-b.fixed.md", "- Fixed two.\n", "fix two")
    tags = release.tag_commits()

    assert release.pending(tags=tags, released={"v0.3.0", "v0.3.1"}) == [second]
    planned = release.plan_releases(
        tags=tags,
        merges=[release.Merge(commit=second, subject="fix two", fragments=[])],
        repository=REPO,
    )
    assert [p.version for p in planned] == ["0.3.2"], first


def test_a_tag_whose_release_is_missing_is_pending(repo: Path) -> None:
    second = _add(repo, "1-a.fixed.md", "- Fixed one.\n", "fix")
    _git(repo, "tag", "--annotate", "v0.3.1", "--message", "Release v0.3.1")
    assert release.pending(tags=release.tag_commits(), released={"v0.3.0"}) == [second]


def test_nothing_is_pending_once_every_merge_is_tagged_and_released(repo: Path) -> None:
    _add(repo, "1-a.fixed.md", "- Fixed one.\n", "fix")
    _git(repo, "tag", "--annotate", "v0.3.1", "--message", "Release v0.3.1")
    assert release.pending(tags=release.tag_commits(), released={"v0.3.0", "v0.3.1"}) == []


def test_pending_refuses_when_the_latest_tag_is_off_the_first_parent_line(
    repo: Path,
) -> None:
    """A tag reachable only from a side branch cannot order the merges on main."""
    _git(repo, "checkout", "-q", "-b", "side")
    side = _add(repo, "1-a.fixed.md", "- Fixed one.\n", "fix on side")
    _git(repo, "tag", "--annotate", "v0.4.0", "--message", "Release v0.4.0")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "side", "-m", "merge side")

    assert side not in release.first_parent_history()
    with pytest.raises(release.ReleaseError, match="v0.4.0"):
        release.pending(tags=release.tag_commits(), released=set())


def test_commit_facts_read_the_subject_and_the_added_fragments(repo: Path) -> None:
    commit = _add(repo, "1-a.fixed.md", "- Fixed one.\n", "Fix the thing (#1)")
    assert release.commit_facts(commit) == ("Fix the thing (#1)", ["changelog.d/1-a.fixed.md"])


def test_commit_facts_handle_the_root_commit(repo: Path) -> None:
    """A repository's first commit has no parent to diff against."""
    root = _git(repo, "rev-list", "--max-parents=0", "HEAD")
    assert release.commit_facts(root) == ("base", ["changelog.d/README.md"])


# ── End to end: the `plan` and `publish` commands ───────────────────────────


@pytest.fixture
def stub_gh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A `gh` on `PATH` that answers from a JSON file and logs what it was asked.

    The state lives in a file rather than the environment so a test can change
    it between two runs of the same command, which is how the resumption case
    is written.
    """
    state = tmp_path / "gh-state.json"
    state.write_text(
        json.dumps({"releases": ["v0.3.0"], "prs": {}, "fail_create": False})
    )
    log = tmp_path / "gh.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(
        f"""#!{sys.executable}
import json, os, sys

state = json.loads(open(os.environ["FAKE_GH_STATE"]).read())
args = sys.argv[1:]
with open(os.environ["FAKE_GH_LOG"], "a") as handle:
    handle.write(" ".join(args) + "\\n")
if args[:2] == ["release", "list"]:
    print(json.dumps([{{"tagName": tag}} for tag in state["releases"]]))
elif args[:2] == ["release", "create"]:
    if state["fail_create"]:
        print("could not create the release", file=sys.stderr)
        sys.exit(1)
    print("created")
elif args[:1] == ["api"]:
    sha = args[1].split("/commits/")[1].split("/")[0]
    number = state["prs"].get(sha)
    print(json.dumps([{{"number": number, "merged_at": "2026-10-29"}}] if number else []))
else:
    print("unexpected gh invocation: " + " ".join(args), file=sys.stderr)
    sys.exit(2)
""",
        encoding="utf-8",
    )
    (bin_dir / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_GH_STATE", str(state))
    monkeypatch.setenv("FAKE_GH_LOG", str(log))
    return state


def _gh_calls(state_file: Path) -> list[str]:
    log = Path(os.environ["FAKE_GH_LOG"])
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def _set_gh_state(state_file: Path, **changes: object) -> None:
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state.update(changes)
    state_file.write_text(json.dumps(state), encoding="utf-8")


def _remote(repo: Path) -> Path:
    """A bare repository the test can push to, so publication is real."""
    bare = repo.parent / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    _git(repo, "remote", "add", "origin", str(bare))
    return bare


def _plan(repo: Path, out: Path) -> list[dict[str, object]]:
    assert release.main(["plan", "--repository", REPO, "--json-out", str(out)]) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_plan_lists_every_merge_that_is_owed_a_release(
    repo: Path, stub_gh: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _add(repo, "1-a.fixed.md", "- Fixed one.\n", "Fix one (#1)")
    second = _add(repo, "2-b.added.md", "- Added two.\n", "Add two (#2)")
    _set_gh_state(stub_gh, prs={first: 1, second: 2})

    plan = _plan(repo, tmp_path / "plan.json")

    assert [(e["commit"], e["version"], e["pr_number"]) for e in plan] == [
        (first, "0.3.1", 1),
        (second, "0.4.0", 2),
    ]
    assert all(e["needs_tag"] and e["needs_release"] for e in plan)
    assert "0.3.1" in capsys.readouterr().out


def test_the_plan_resumes_a_tag_whose_release_is_missing(
    repo: Path, stub_gh: Path, tmp_path: Path
) -> None:
    _add(repo, "1-a.fixed.md", "- Fixed one.\n", "Fix one (#1)")
    _git(repo, "tag", "--annotate", "v0.3.1", "--message", "Release v0.3.1")

    (entry,) = _plan(repo, tmp_path / "plan.json")

    assert (entry["tag"], entry["needs_tag"], entry["needs_release"]) == (
        "v0.3.1",
        False,
        True,
    )


def test_publish_tags_pushes_and_creates_the_release(
    repo: Path, stub_gh: Path, tmp_path: Path
) -> None:
    bare = _remote(repo)
    commit = _add(repo, "1-a.fixed.md", "- Fixed one.\n", "Fix one (#1)")
    out = tmp_path / "plan.json"
    _plan(repo, out)

    assert release.main(["publish", "--plan", str(out)]) == 0

    assert _git(repo, "rev-parse", "v0.3.1^{commit}") == commit
    pushed = subprocess.run(
        ["git", "rev-parse", "refs/tags/v0.3.1"],
        cwd=bare, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert pushed == _git(repo, "rev-parse", "refs/tags/v0.3.1")
    assert any(call.startswith("release create v0.3.1") for call in _gh_calls(stub_gh))


def test_publish_finishes_a_release_the_previous_run_could_not_create(
    repo: Path, stub_gh: Path, tmp_path: Path
) -> None:
    """The tag is pushed, `gh release create` fails, and the next run completes it.

    This is the partial publication the reviews found: "tag exists" was read as
    "released", so the Release could never be created afterwards.
    """
    _remote(repo)
    commit = _add(repo, "1-a.fixed.md", "- Fixed one.\n", "Fix one (#1)")
    out = tmp_path / "plan.json"
    _plan(repo, out)

    _set_gh_state(stub_gh, fail_create=True)
    assert release.main(["publish", "--plan", str(out)]) == 1
    assert _git(repo, "rev-parse", "v0.3.1^{commit}") == commit

    _set_gh_state(stub_gh, fail_create=False)
    (entry,) = _plan(repo, out)
    assert (entry["needs_tag"], entry["needs_release"]) == (False, True)

    assert release.main(["publish", "--plan", str(out)]) == 0
    assert _git(repo, "rev-parse", "v0.3.1^{commit}") == commit, "the tag did not move"
    assert any(call.startswith("release create v0.3.1") for call in _gh_calls(stub_gh))


def test_publish_refuses_to_move_a_tag_on_another_commit(
    repo: Path, stub_gh: Path, tmp_path: Path
) -> None:
    _remote(repo)
    _add(repo, "1-a.fixed.md", "- Fixed one.\n", "Fix one (#1)")
    _git(repo, "tag", "--annotate", "v0.3.1", "--message", "already here")
    out = tmp_path / "plan.json"
    out.write_text(
        json.dumps(
            [
                {
                    "commit": "9" * 40,
                    "subject": "elsewhere",
                    "pr_number": None,
                    "version": "0.3.1",
                    "tag": "v0.3.1",
                    "notes": "- Something.\n",
                    "needs_tag": True,
                    "needs_release": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    assert release.main(["publish", "--plan", str(out)]) == 1
    assert not any(call.startswith("release create") for call in _gh_calls(stub_gh))
