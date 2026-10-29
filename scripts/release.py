#!/usr/bin/env python3
"""The next version, and its release notes, from a merge to `main` (issue #333).

Every merge to `main` is a version: `.github/workflows/release.yml` finds the
pull request the pushed commit merged and the changelog fragments that pull
request added, then calls this script and tags the commit with what it returns.
The workflow holds no rule of its own -- a bump rule spread over YAML is a rule
nothing can test.

Two decisions live here:

* **The bump.** A `breaking`, `added` or `changed` fragment is a minor bump;
  anything else -- `fixed`, `internal`, or no fragment at all -- is a patch.
  Below `1.0.0` the minor number carries breaking changes, as `CONTRIBUTING.md`
  says. With no previous tag the same rule applies from `0.0.0`, so the first
  fix is `0.0.1` and the first feature `0.1.0`.

* **The notes**, which are that pull request's fragments under their type
  headings, with a link to the pull request. A push with no pull request gets
  its commit subject instead, and a pull request with no fragments says so:
  an empty release whose notes do not explain why is the silent failure this
  project treats as the worst outcome.

Run it by hand, against the last tag, to see what a merge would release:

    pixi run -e dev python scripts/release.py --previous-tag v0.3.0 \
        --fragment changelog.d/333-release-script.changed.md

It prints the version and, with `--notes-out`, writes the notes.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
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
#: The one file under `changelog.d/` that explains the format instead of being
#: a fragment: a human running `--fragment changelog.d/*.md` must not be refused
#: for it.
FRAGMENT_README = "README.md"
TAG_PATTERN = re.compile(r"^v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")
FRAGMENT_PATTERN = re.compile(
    r"^(?P<slug>.+)\.(?P<kind>" + "|".join(FRAGMENT_TYPES) + r")\.md$"
)


class ReleaseError(ValueError):
    """Something the release cannot be built from, named in the message."""


@dataclass(frozen=True)
class Fragment:
    """One added `changelog.d/<slug>.<type>.md` and what it says."""

    path: str
    kind: str
    text: str


@dataclass(frozen=True)
class Release:
    version: str
    tag: str
    notes: str


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


def _footer(pr_number: int | None, repository: str) -> str:
    if pr_number is None:
        return "Pushed to `main` without a pull request, so no fragments were read."
    return f"Merged in #{pr_number}: https://github.com/{repository}/pull/{pr_number}"


def build_notes(
    *,
    fragments: list[Fragment],
    pr_number: int | None,
    commit_subject: str,
    repository: str,
) -> str:
    """The release notes: the fragments, or what explains their absence."""
    body = _bodies(fragments)
    if not body:
        if pr_number is None:
            body = f"- {commit_subject}" if commit_subject else "No commit subject was given."
        else:
            body = "No changelog fragments were added by this pull request."
    return f"{body}\n\n---\n\n{_footer(pr_number, repository)}\n"


def plan_release(
    *,
    previous_tag: str | None,
    fragments: list[Fragment],
    pr_number: int | None,
    commit_subject: str = "",
    repository: str = REPOSITORY,
) -> Release:
    """The version to tag and the notes to publish for one merge."""
    version = next_version(previous_tag, [fragment.kind for fragment in fragments])
    notes = build_notes(
        fragments=fragments,
        pr_number=pr_number,
        commit_subject=commit_subject,
        repository=repository,
    )
    return Release(version=version, tag=f"{TAG_PREFIX}{version}", notes=notes)


def _fragment_paths(from_file: Path | None) -> list[Path]:
    if from_file is None:
        return []
    lines = from_file.read_text(encoding="utf-8").splitlines()
    return [Path(line.strip()) for line in lines if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--previous-tag",
        default="",
        help="the latest vX.Y.Z tag, or empty when the repository has none",
    )
    parser.add_argument(
        "--fragment", action="append", default=[], help="a fragment the merge added"
    )
    parser.add_argument(
        "--fragments-from",
        type=Path,
        default=None,
        help="a file listing fragment paths, one per line (git diff's output)",
    )
    parser.add_argument("--pr-number", type=int, default=None)
    parser.add_argument("--commit-subject", default="")
    parser.add_argument("--notes-out", type=Path, default=None)
    parser.add_argument("--repository", default=REPOSITORY)
    args = parser.parse_args(argv)

    try:
        fragments = read_fragments(
            [Path(path) for path in args.fragment]
            + _fragment_paths(args.fragments_from)
        )
        release = plan_release(
            previous_tag=args.previous_tag or None,
            fragments=fragments,
            pr_number=args.pr_number,
            commit_subject=args.commit_subject,
            repository=args.repository,
        )
    except (ReleaseError, OSError) as error:
        print(f"release: {error}", file=sys.stderr)
        return 2

    if args.notes_out is not None:
        args.notes_out.write_text(release.notes, encoding="utf-8")
    print(release.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
