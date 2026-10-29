#!/usr/bin/env python3
"""The releases `main` is owed, and what publishing them still needs (#333).

Every merge to `main` is a version: a `vX.Y.Z` tag on the merge commit whose
GitHub Release notes are the changelog fragments that merge added. This script
is the whole of that rule, because a rule spread over YAML is a rule nothing
can test. `.github/workflows/release.yml` gathers nothing and decides nothing;
it runs `plan`, then `publish`.

Three things happen here.

**The bump.** A `breaking`, `added` or `changed` fragment is a minor; anything
else -- `fixed`, `internal`, or no fragment at all -- is a patch. Below `1.0.0`
the minor number carries breaking changes, as `CONTRIBUTING.md` says. With no
previous tag the same rule runs from `0.0.0`, so the first fix is `0.0.1`.

**Reconciliation.** A run per merge is not something the workflow can promise:
GitHub keeps one pending run per concurrency group and discards the rest, so
three quick merges can leave the middle one untagged with nothing to retry it.
Each run therefore publishes *every* first-parent commit after the newest `v*`
tag that has no tag of its own, and any tagged commit whose Release is missing
and is newer than the newest Release there is, oldest first. A lost run is
caught up by the next one, and the cutover's tag is repaired without inventing
Releases for tags from before this workflow existed.

**Resumption.** Publication is two steps per commit -- push the tag, create the
Release -- and each is skipped when the *remote* says it is already done. A run
that dies between them, or a push that never reached the remote, leaves a tag
without its Release, and the next run finishes it rather than concluding from a
local tag that the release exists.

Nothing here treats a git failure as an absence. `git ls-remote --tags origin`
answers "no tags yet" with an empty list and an unreadable remote with an error,
and the error stops the run: an empty answer would publish `0.0.1` over a real
history. Nor does anything read the working tree for a commit's notes: a
fragment is read with `git show <commit>:<path>`, so a later edit cannot change
what an earlier Release said.

Run it by hand, read-only, to see what a merge would release:

    pixi run -e dev python scripts/release.py plan --repository opengwas/opengwasdb
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

REPOSITORY = "opengwas/opengwasdb"

#: The fragment types, in the order their headings appear in the notes.
FRAGMENT_TYPES = ("breaking", "added", "changed", "fixed", "internal")
MINOR_TYPES = frozenset({"breaking", "added", "changed"})
HEADINGS = {
    "breaking": "Breaking",
    "added": "Added",
    "changed": "Changed",
    "fixed": "Fixed",
    "internal": "Internal",
}
TAG_PREFIX = "v"
TAG_PATTERN = re.compile(r"^v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")
FRAGMENT_PATTERN = re.compile(
    r"^(?P<slug>.+)\.(?P<kind>" + "|".join(FRAGMENT_TYPES) + r")\.md$"
)
#: The one file under `changelog.d/` that explains the format instead of being a
#: fragment: `--fragment changelog.d/*.md` must not be refused for it.
FRAGMENT_README = "README.md"
#: `git diff` needs a tree for a commit with no parent; this is git's own empty one.
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
#: How many merges one run may publish. A lost run is caught up here, but a
#: whole merged branch is not a burst of lost runs: at the cutover `dev` is 64
#: first-parent commits past the last tag, and replaying them would publish 64
#: patch versions and no `0.3.0`. Above this many, the run stops and asks for a
#: tag instead.
MAX_PLANNED_RELEASES = 20


class ReleaseError(ValueError):
    """Something a release cannot be planned or published from, named."""


@dataclass(frozen=True)
class Fragment:
    """One added `changelog.d/<slug>.<type>.md` and what it says."""

    path: str
    kind: str
    text: str


@dataclass(frozen=True)
class Merge:
    """What one commit on `main` contributes to a release.

    The fragments are values, not paths: the planner must never read the
    working tree, or a note edited after the fact would rewrite an earlier
    release. Whoever gathers the facts reads them from the commit's own tree.
    """

    commit: str
    subject: str
    fragments: list[Fragment] = field(default_factory=list)
    pr_number: int | None = None
    #: A `v*` tag already on this commit, when its Release was never created.
    tag: str | None = None


@dataclass(frozen=True)
class PlannedRelease:
    """One version, and which of its two publication steps are still owed."""

    commit: str
    subject: str
    pr_number: int | None
    version: str
    tag: str
    notes: str
    needs_tag: bool
    needs_release: bool = True


# ── The rules ───────────────────────────────────────────────────────────────


def parse_tag(tag: str) -> tuple[int, int, int]:
    """The three numbers of a `vX.Y.Z` tag, refusing anything else.

    A tag this cannot read is not a version to bump from: guessing (dropping a
    suffix, taking the last number) would produce a version that looks
    deliberate and is not.
    """
    match = TAG_PATTERN.match(tag)
    if match is None:
        raise ReleaseError(
            f"{tag!r} is not a release tag: expected `v<major>.<minor>.<patch>`, "
            f"the only form the release workflow pushes"
        )
    return int(match["major"]), int(match["minor"]), int(match["patch"])


def next_version(previous_tag: str | None, kinds: list[str]) -> str:
    """The version after `previous_tag` for a change with these fragment types."""
    unknown = sorted(set(kinds) - set(FRAGMENT_TYPES))
    if unknown:
        raise ReleaseError(
            f"unknown fragment type(s) {', '.join(unknown)}; "
            f"expected one of {', '.join(FRAGMENT_TYPES)}"
        )
    major, minor, patch = parse_tag(previous_tag) if previous_tag else (0, 0, 0)
    if any(kind in MINOR_TYPES for kind in kinds):
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def latest_tag(tags: dict[str, str]) -> str | None:
    """The newest `vX.Y.Z` tag by version number, not by text or commit date."""
    parsed = [(parse_tag(tag), tag) for tag in tags]
    return max(parsed)[1] if parsed else None


def fragment_kind(path: Path) -> str:
    """The type a fragment's file name declares, refusing a name that has none."""
    match = FRAGMENT_PATTERN.match(path.name)
    if match is None:
        raise ReleaseError(
            f"{path}: a changelog fragment is named `<slug>.<type>.md` with type "
            f"one of {', '.join(FRAGMENT_TYPES)} (see changelog.d/README.md)"
        )
    return match["kind"]


def _parse_fragments(paths: list[str], read: Callable[[str], str]) -> list[Fragment]:
    """Parse fragments, reading each one however the caller can.

    An empty fragment is refused rather than dropped: a bullet with nothing in
    it would publish a release whose notes silently omit a change.
    """
    fragments = []
    for path in paths:
        if Path(path).name == FRAGMENT_README:
            continue
        kind = fragment_kind(Path(path))
        text = read(path).strip()
        if not text:
            raise ReleaseError(f"{path}: the fragment is empty")
        fragments.append(Fragment(path=path, kind=kind, text=text))
    return fragments


def read_fragments(paths: list[Path]) -> list[Fragment]:
    """Fragments from the working tree -- a dry run of not-yet-committed notes."""
    return _parse_fragments(
        [str(path) for path in paths],
        lambda path: Path(path).read_text(encoding="utf-8"),
    )


def read_commit_fragments(commit: str, paths: list[str]) -> list[Fragment]:
    """Fragments as they were in `commit`'s tree, never as they are now.

    A fragment is never edited, but that is a rule people keep rather than
    something git enforces: reading the working tree let a copy edit in a later
    commit silently change what an earlier version's Release said.
    """
    return _parse_fragments(paths, lambda path: _git("show", f"{commit}:{path}"))


def _bodies(fragments: list[Fragment]) -> str:
    sections = []
    for kind in FRAGMENT_TYPES:
        texts = [fragment.text for fragment in fragments if fragment.kind == kind]
        if texts:
            sections.append(f"### {HEADINGS[kind]}\n\n" + "\n\n".join(texts))
    return "\n\n".join(sections)


def _absent_body(pr_number: int | None) -> str:
    if pr_number is not None:
        return "No changelog fragments were added by this pull request."
    return "No changelog fragments were added by this push."


def _footer(pr_number: int | None, commit_subject: str, repository: str) -> str:
    if pr_number is not None:
        return f"Merged in #{pr_number}: https://github.com/{repository}/pull/{pr_number}"
    named = f": `{commit_subject}`" if commit_subject else "."
    return f"Pushed to `main` without a pull request{named}"


def build_notes(
    *,
    fragments: list[Fragment],
    pr_number: int | None,
    commit_subject: str,
    repository: str,
) -> str:
    """The release notes: the fragments, or a sentence saying why there are none.

    A push with no pull request still uses its fragments; the commit subject is
    named below them, because that push is the only thing that explains itself.
    """
    body = _bodies(fragments) or _absent_body(pr_number)
    footer = _footer(pr_number, commit_subject, repository)
    return f"{body}\n\n---\n\n{footer}\n"


def _plan_one(
    *,
    previous_tag: str | None,
    merge: Merge,
    tags: dict[str, str],
    repository: str,
) -> PlannedRelease:
    fragments = merge.fragments
    notes = build_notes(
        fragments=fragments,
        pr_number=merge.pr_number,
        commit_subject=merge.subject,
        repository=repository,
    )
    if merge.tag is not None:
        version = merge.tag.removeprefix(TAG_PREFIX)
        parse_tag(merge.tag)
        if tags.get(merge.tag) != merge.commit:
            raise ReleaseError(
                f"{merge.tag} is not on {merge.commit}, so it cannot be the tag "
                f"of that commit; refusing to publish it"
            )
        return PlannedRelease(
            commit=merge.commit,
            subject=merge.subject,
            pr_number=merge.pr_number,
            version=version,
            tag=merge.tag,
            notes=notes,
            needs_tag=False,
            needs_release=True,
        )

    version = next_version(
        previous_tag, [fragment.kind for fragment in fragments]
    )
    tag = f"{TAG_PREFIX}{version}"
    if tag in tags:
        raise ReleaseError(
            f"{tag} already exists on {tags[tag]}; the next version for "
            f"{merge.commit} would be {tag}, and a published tag is never moved "
            f"or reused"
        )
    return PlannedRelease(
        commit=merge.commit,
        subject=merge.subject,
        pr_number=merge.pr_number,
        version=version,
        tag=tag,
        notes=notes,
        needs_tag=True,
        needs_release=True,
    )


def plan_releases(
    *, tags: dict[str, str], merges: list[Merge], repository: str = REPOSITORY
) -> list[PlannedRelease]:
    """The versions for these merges, oldest first, numbered from the newest tag.

    Only a *new* version raises the base. A resumed tag takes its own version
    and leaves the base alone: repairing an older missing Release must never
    renumber the next merge below what the remote already published
    (standards review, finding 3: `v0.3.0` released plus a missing `v0.2.0`
    Release gave the next merge `0.2.1`).
    """
    base = latest_tag(tags)
    planned = []
    for merge in merges:
        release = _plan_one(
            previous_tag=base, merge=merge, tags=tags, repository=repository
        )
        planned.append(release)
        if merge.tag is None:
            base = release.tag
    return planned


# ── Discovery: tags, history, fragments, and GitHub ─────────────────────────


def _git(*args: str) -> str:
    """Run git, treating any failure as an error rather than as "nothing there".

    The distinction is the whole point of this function. `git describe ... ||
    true` cannot tell a repository with no tags from a repository that cannot
    be read, and an empty answer for the second publishes `0.0.1` over a real
    history.
    """
    completed = subprocess.run(["git", *args], capture_output=True, text=True)
    if completed.returncode != 0:
        raise ReleaseError(
            f"`git {' '.join(args)}` failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip() or 'no output'}"
        )
    return completed.stdout


def _gh(*args: str, stdin: str | None = None) -> str:
    completed = subprocess.run(
        ["gh", *args], capture_output=True, text=True, input=stdin
    )
    if completed.returncode != 0:
        raise ReleaseError(
            f"`gh {' '.join(args)}` failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip() or 'no output'}"
        )
    return completed.stdout


def _git_quiet(*args: str) -> str | None:
    """Run git, answering `None` for a question whose answer is "there is none".

    Only `rev-parse --verify --quiet` is asked this way: it returns 1 for a ref
    that does not exist, which is an answer, and not a failure.
    """
    completed = subprocess.run(["git", *args], capture_output=True, text=True)
    if completed.returncode == 0:
        return completed.stdout
    return None


def remote_tags(remote: str = "origin") -> dict[str, str]:
    """The `v*` tags the remote has, and the commit each points at.

    The remote is the truth, not the local clone: a plan built from local tags
    cannot tell a tag that was pushed from one a failed push left behind, and it
    cannot see a tag somebody else created. A tag push or a Release that the
    remote never received did not happen.

    `git ls-remote --tags` prints the tag object's own sha for an annotated tag
    and its peeled commit as a second line, `<sha>\trefs/tags/v1.2.3^{}`; the
    commit is what a Release is about, so the peeled line wins.
    """
    listed = _git("ls-remote", "--tags", remote)
    commits: dict[str, str] = {}
    for line in listed.splitlines():
        if not line.strip():
            continue
        sha, _, ref = line.partition("\t")
        name = ref.removeprefix("refs/tags/")
        peeled = name.endswith("^{}")
        name = name.removesuffix("^{}")
        if not is_release_tag(name):
            continue
        if peeled or name not in commits:
            commits[name] = sha.strip()
    return commits


def is_release_tag(name: str) -> bool:
    """Whether `name` is a `vX.Y.Z` tag -- the only kind that is a version."""
    return bool(TAG_PATTERN.match(name))


def local_tag_commit(tag: str) -> str | None:
    """The commit the local clone's `tag` names, or `None` when it has none."""
    found = _git_quiet("rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}")
    return found.strip() if found else None


def first_parent_history() -> list[str]:
    """The first-parent commits of `HEAD`, oldest first: what `main` looks like."""
    listed = _git("rev-list", "--first-parent", "--reverse", "HEAD").splitlines()
    return [sha.strip() for sha in listed if sha.strip()]


def commit_facts(commit: str) -> tuple[str, list[str]]:
    """The commit's subject, and the fragments it added under `changelog.d/`."""
    subject = _git("log", "-1", "--format=%s", commit).strip()
    parents = _git("rev-list", "--parents", "-n", "1", commit).split()
    parent = parents[1] if len(parents) > 1 else EMPTY_TREE
    added = _git(
        "diff", "--diff-filter=A", "--name-only", parent, commit, "--", "changelog.d/"
    )
    return subject, [path for path in added.splitlines() if path.strip()]


def _after_newest(commits: list[str], tags: dict[str, str]) -> list[str]:
    """The first-parent commits after the newest tag, oldest first."""
    newest = latest_tag(tags)
    if newest is None:
        return list(commits)
    try:
        return commits[commits.index(tags[newest]) + 1:]
    except ValueError as error:
        raise ReleaseError(
            f"the latest tag {newest} is on {tags[newest]}, which is not a "
            f"first-parent commit of HEAD; the merges on main cannot be "
            f"ordered from it"
        ) from error


def _untagged_after_newest(
    commits: list[str], tagged: dict[str, str], tags: dict[str, str]
) -> set[str]:
    """First-parent commits past the newest tag that carry no tag themselves."""
    return {commit for commit in _after_newest(commits, tags) if commit not in tagged}


def _newest_and_missing(
    tags: dict[str, str], released: set[str]
) -> tuple[list[tuple[tuple[int, int, int], str]], list[tuple[tuple[int, int, int], str]]]:
    """(missing, existing) tags, each as (version, tag), for the resume rule."""
    missing = [(parse_tag(tag), tag) for tag in tags if tag not in released]
    existing = [(parse_tag(tag), tag) for tag in tags if tag in released]
    return missing, existing


def resume_tags(tags: dict[str, str], released: set[str]) -> set[str]:
    """The tags whose missing Release is worth creating, and no older ones.

    Only tags newer than the newest existing Release are repaired, so the first
    run after the cutover creates the cutover's Release and not one for every
    tag the project ever made. A repository with no Releases at all resumes
    just its newest tag -- the cutover's -- for the same reason: a tag from
    2026 whose history predates fragments is not worth a Release now.
    """
    missing, existing = _newest_and_missing(tags, released)
    if not missing:
        return set()
    if not existing:
        return {max(missing)[1]}
    newest_release = max(existing)[0]
    return {tag for version, tag in missing if version > newest_release}


def pending(
    *, tags: dict[str, str], released: set[str], history: list[str] | None = None
) -> list[str]:
    """The commits that need something published, oldest first.

    A commit needs a release when it is a first-parent commit after the newest
    tag and has no tag of its own, or when it carries a tag whose Release is
    missing and is newer than the newest Release there is. Everything else is
    already published.
    """
    commits = first_parent_history() if history is None else history
    tagged = {commit: tag for tag, commit in tags.items()}
    owed = _untagged_after_newest(commits, tagged, tags)
    owed |= _resumed_commits(tags, released, set(commits))
    return [commit for commit in commits if commit in owed]


def _resumed_commits(
    tags: dict[str, str], released: set[str], known: set[str]
) -> set[str]:
    """The commits carrying a resumed tag, when the commit is on this line."""
    return {
        tags[tag]
        for tag in resume_tags(tags, released)
        if tags[tag] in known
    }


def released_tags(repository: str) -> set[str]:
    """The tags that already have a GitHub Release."""
    payload = json.loads(
        _gh("release", "list", "--repo", repository, "--limit", "1000", "--json", "tagName")
        or "[]"
    )
    return {entry["tagName"] for entry in payload if entry.get("tagName")}


def pull_request(commit: str, repository: str) -> int | None:
    """The merged pull request for `commit`, or `None` when the push had none.

    An empty answer is a fact -- a direct push -- and the release says so. A
    `gh` that fails is an error, because a release whose notes claim no pull
    request when there was one is the plausible wrong answer this repository
    treats as the worst outcome.
    """
    payload = json.loads(
        _gh("api", f"repos/{repository}/commits/{commit}/pulls") or "[]"
    )
    merged = [
        entry
        for entry in payload
        if isinstance(entry, dict) and entry.get("merged_at") and entry.get("number")
    ]
    return int(merged[0]["number"]) if merged else None


# ── Publication ─────────────────────────────────────────────────────────────


def _create_tag(tag: str, commit: str) -> None:
    """Annotate `commit` as `tag`, with an identity of its own.

    The tag's identity comes from here rather than from the runner's git config,
    so a published tag reads the same wherever it was created (`git -c …` must
    precede the subcommand).
    """
    _git(
        "-c",
        "user.name=opengwasdb release",
        "-c",
        "user.email=release@opengwas.io",
        "tag",
        "--annotate",
        tag,
        "--message",
        f"Release {tag}",
        commit,
    )


def _ensure_remote_tag(release: PlannedRelease, remote_commit: str | None) -> bool:
    """Leave the tag on the remote at this release's commit; True if pushed.

    Three states, and only one of them is allowed to do nothing: the remote
    already has the tag on the right commit, the remote has no tag and either
    the clone has none or has one on the right commit (a push that failed leaves
    exactly that), or the tag names another commit — on the remote, or locally —
    which is refused rather than moved.
    """
    if remote_commit is not None:
        if remote_commit != release.commit:
            raise ReleaseError(
                f"{release.tag} exists on the remote at {remote_commit}, not at "
                f"{release.commit}; a published tag is never moved"
            )
        return False
    local_commit = local_tag_commit(release.tag)
    if local_commit is not None and local_commit != release.commit:
        raise ReleaseError(
            f"{release.tag} exists locally at {local_commit}, not at "
            f"{release.commit}; it is never moved by this script"
        )
    if local_commit is None:
        _create_tag(release.tag, release.commit)
    _git("push", "origin", f"refs/tags/{release.tag}")
    return True


def _require_remote_tag(release: PlannedRelease) -> None:
    """Refuse to describe a tag the remote does not have at this commit.

    `gh release create --verify-tag` checks that the tag exists, not which
    commit it names, so the commit is checked here, from the remote, before
    anything is published about it.
    """
    remote_commit = remote_tags().get(release.tag)
    if remote_commit != release.commit:
        raise ReleaseError(
            f"{release.tag} is at {remote_commit} on the remote, not at "
            f"{release.commit}; refusing to create a Release for it"
        )


def publish(
    plan: list[PlannedRelease], *, repository: str = REPOSITORY
) -> list[str]:
    """Do whatever each planned release still needs, oldest first.

    Both steps are idempotent and separate, and both ask the remote rather than
    this clone: a tag that a failed push left behind is pushed on the retry, a
    tag that names another commit is refused, and the Release is created only
    once the remote's tag names the planned commit. A failure part-way leaves
    the steps already done in place and the rest for the next run.
    """
    done = []
    for release in plan:
        remote_commit = remote_tags().get(release.tag)
        pushed = False
        if release.needs_tag or remote_commit is None:
            pushed = _ensure_remote_tag(release, remote_commit)
            if pushed:
                done.append(f"tagged {release.tag} on {release.commit}")
        if release.needs_release:
            _require_remote_tag(release)
            _gh(
                "release",
                "create",
                release.tag,
                "--repo",
                repository,
                "--verify-tag",
                "--title",
                release.tag,
                "--notes-file",
                "-",
                stdin=release.notes,
            )
            done.append(f"released {release.tag}")
    return done


def collect(
    tags: dict[str, str],
    released: set[str],
    repository: str,
    commits: list[str] | None = None,
) -> list[PlannedRelease]:
    """Discover what is owed, gather each commit's facts, and plan it.

    Each commit's fragments are read from that commit's tree, and its tag from
    the remote: the facts are about the commit, not about the clone as it is
    now.
    """
    if commits is None:
        commits = pending(tags=tags, released=released)
    tags_of = {value: key for key, value in tags.items()}
    merges = []
    for commit in commits:
        subject, paths = commit_facts(commit)
        merges.append(
            Merge(
                commit=commit,
                subject=subject,
                fragments=read_commit_fragments(commit, paths),
                pr_number=pull_request(commit, repository),
                tag=tags_of.get(commit),
            )
        )
    return plan_releases(tags=tags, merges=merges, repository=repository)


# ── The command line ────────────────────────────────────────────────────────


def _fragment_paths(fragment_args: list[str], from_file: Path | None) -> list[Path]:
    paths = [Path(path) for path in fragment_args]
    if from_file is not None:
        lines = from_file.read_text(encoding="utf-8").splitlines()
        paths.extend(Path(line.strip()) for line in lines if line.strip())
    return paths


def _plan_json(plan: list[PlannedRelease]) -> str:
    return json.dumps([_entry(release) for release in plan], indent=2)


def _entry(release: PlannedRelease) -> dict[str, object]:
    return {
        "commit": release.commit,
        "subject": release.subject,
        "pr_number": release.pr_number,
        "version": release.version,
        "tag": release.tag,
        "notes": release.notes,
        "needs_tag": release.needs_tag,
        "needs_release": release.needs_release,
    }


def _print_plan(plan: list[PlannedRelease]) -> None:
    if not plan:
        print("Nothing to release: every merge on main has its tag and its Release.")
        return
    for release in plan:
        steps = []
        if release.needs_tag:
            steps.append("tag")
        if release.needs_release:
            steps.append("release")
        print(
            f"{release.version:<10} {release.tag:<10} {release.commit[:12]} "
            f"{release.subject}  [{', '.join(steps)}]"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    following = commands.add_parser(
        "next", help="the version one merge would be, without touching anything"
    )
    following.add_argument("--previous-tag", default="")
    following.add_argument("--fragment", action="append", default=[])
    following.add_argument("--fragments-from", type=Path, default=None)
    following.add_argument("--pr-number", type=int, default=None)
    following.add_argument("--commit-subject", default="")
    following.add_argument("--notes-out", type=Path, default=None)
    following.add_argument("--repository", default=REPOSITORY)

    planning = commands.add_parser(
        "plan", help="every release main is owed, oldest first"
    )
    planning.add_argument("--repository", default=REPOSITORY)
    planning.add_argument("--json-out", type=Path, default=None)
    planning.add_argument("--max-releases", type=int, default=MAX_PLANNED_RELEASES)

    publishing = commands.add_parser(
        "publish", help="tag and release what a plan says is still owed"
    )
    publishing.add_argument("--plan", type=Path, required=True)
    publishing.add_argument("--repository", default=REPOSITORY)
    return parser


def _run_next(args: argparse.Namespace) -> int:
    release = _plan_one(
        previous_tag=args.previous_tag or None,
        merge=Merge(
            commit="",
            subject=args.commit_subject,
            fragments=read_fragments(
                _fragment_paths(args.fragment, args.fragments_from)
            ),
            pr_number=args.pr_number,
        ),
        tags={},
        repository=args.repository,
    )
    if args.notes_out is not None:
        args.notes_out.write_text(release.notes, encoding="utf-8")
    print(release.version)
    return 0


def _run_plan(args: argparse.Namespace) -> int:
    tags = remote_tags()
    released = released_tags(args.repository)
    commits = pending(tags=tags, released=released)
    if len(commits) > args.max_releases:
        raise ReleaseError(
            f"{len(commits)} merges are owed a version, more than the "
            f"{args.max_releases} one run will publish. If this is the cutover "
            f"-- the first push of the integration branch into main -- tag the "
            f"merge commit before pushing main, so the history before it is not "
            f"replayed as versions; otherwise re-run with --max-releases."
        )
    plan = collect(tags, released, args.repository, commits)
    _print_plan(plan)
    if args.json_out is not None:
        args.json_out.write_text(_plan_json(plan), encoding="utf-8")
    return 0


def _run_publish(args: argparse.Namespace) -> int:
    raw = json.loads(args.plan.read_text(encoding="utf-8"))
    plan = [PlannedRelease(**entry) for entry in raw]
    for line in publish(plan, repository=args.repository):
        print(line)
    return 0


_RUNNERS = {"next": _run_next, "plan": _run_plan, "publish": _run_publish}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _RUNNERS[args.command](args)
    except (ReleaseError, OSError) as error:
        print(f"release: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
