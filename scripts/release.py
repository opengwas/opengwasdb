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
tag that has no tag of its own, and any tagged commit that never got its
Release, oldest first. A lost run is caught up by the next one.

**Resumption.** Publication is two steps per commit -- push the tag, create the
Release -- and each is skipped when it is already done. A run that dies between
them leaves a tag with no Release, and the next run creates it instead of
concluding from the tag that the release exists.

Nothing here treats a git failure as an absence. `git tag --list 'v*'` answers
"no tags yet" with an empty list and a broken repository with an error, and the
error stops the run: an empty answer would publish `0.0.1` over a real history.

Run it by hand, read-only, to see what a merge would release:

    pixi run -e dev python scripts/release.py plan --repository opengwas/opengwasdb
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
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
    """What one commit on `main` contributes to a release."""

    commit: str
    subject: str
    fragments: list[str] = field(default_factory=list)
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


def read_fragments(paths: list[Path]) -> list[Fragment]:
    """The fragments at these paths, in the order given.

    An empty fragment is refused rather than dropped: a bullet with nothing in
    it would publish a release whose notes silently omit a change.
    """
    fragments = []
    for path in paths:
        if path.name == FRAGMENT_README:
            continue
        kind = fragment_kind(path)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ReleaseError(f"{path}: the fragment is empty")
        fragments.append(Fragment(path=str(path), kind=kind, text=text))
    return fragments


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
    fragments = read_fragments([Path(path) for path in merge.fragments])
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
    """The versions for these merges, oldest first, numbered from the newest tag."""
    previous = latest_tag(tags)
    planned = []
    for merge in merges:
        release = _plan_one(
            previous_tag=previous, merge=merge, tags=tags, repository=repository
        )
        planned.append(release)
        previous = release.tag
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


def tag_commits() -> dict[str, str]:
    """Every `v*` tag and the commit it points at; `{}` when there are none."""
    names = [name.strip() for name in _git("tag", "--list", "v*").splitlines() if name.strip()]
    return {name: _git("rev-list", "-n", "1", name).strip() for name in names}


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


def _tagged_without_release(
    tagged: dict[str, str], released: set[str], known: set[str]
) -> set[str]:
    """Tagged commits whose GitHub Release does not exist: the resume set."""
    return {
        commit
        for commit, tag in tagged.items()
        if tag not in released and commit in known
    }


def pending(
    *, tags: dict[str, str], released: set[str], history: list[str] | None = None
) -> list[str]:
    """The commits that need something published, oldest first.

    A commit needs a release when it is a first-parent commit after the newest
    tag and has no tag of its own, or when it has a tag whose Release does not
    exist. Everything else is already published.
    """
    commits = first_parent_history() if history is None else history
    tagged = {commit: tag for tag, commit in tags.items()}
    owed = _untagged_after_newest(commits, tagged, tags)
    owed |= _tagged_without_release(tagged, released, set(commits))
    return [commit for commit in commits if commit in owed]


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


def publish(
    plan: list[PlannedRelease], *, repository: str = REPOSITORY
) -> list[str]:
    """Do whatever each planned release still needs, oldest first.

    Both steps are idempotent and separate: an existing tag on the right commit
    is left alone, an existing tag on another commit is refused, and the
    Release is created only when it is missing. A failure part-way leaves the
    steps already done in place and the rest for the next run.
    """
    done = []
    for release in plan:
        existing = tag_commits()
        on_commit = existing.get(release.tag)
        if on_commit is not None and on_commit != release.commit:
            raise ReleaseError(
                f"{release.tag} already exists on {on_commit}, not on "
                f"{release.commit}; a published tag is never moved"
            )
        if release.needs_tag and on_commit is None:
            # The tag's identity comes from here rather than from the runner's
            # git config, so a published tag reads the same wherever it was
            # created (`git -c …` must precede the subcommand).
            _git(
                "-c",
                "user.name=opengwasdb release",
                "-c",
                "user.email=release@opengwas.io",
                "tag",
                "--annotate",
                release.tag,
                "--message",
                f"Release {release.tag}",
                release.commit,
            )
            _git("push", "origin", f"refs/tags/{release.tag}")
            done.append(f"tagged {release.tag} on {release.commit}")
        if release.needs_release:
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
    """Discover what is owed, gather each commit's facts, and plan it."""
    if commits is None:
        commits = pending(tags=tags, released=released)
    merges = []
    for commit in commits:
        subject, fragments = commit_facts(commit)
        merges.append(
            Merge(
                commit=commit,
                subject=subject,
                fragments=fragments,
                pr_number=pull_request(commit, repository),
                tag={value: key for key, value in tags.items()}.get(commit),
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
            fragments=[
                str(path) for path in _fragment_paths(args.fragment, args.fragments_from)
            ],
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
    tags = tag_commits()
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
