#!/usr/bin/env python3
"""Fail when a pull request changes `opengwasdb/` without adding a fragment.

Every merge to `main` is a version whose release notes are the changelog
fragments that pull request added, so a change to the package without a
fragment is a version with nothing to say. A pull request adds
`changelog.d/<slug>.<type>.md`, where the type is one of `breaking`, `added`,
`changed`, `fixed` or `internal`; `changelog.d/README.md` explains what to
write in one.

Two things are checked, in this order:

1. Every fragment the branch *adds* must be readable: `<slug>.<type>.md`,
   directly under `changelog.d/`, with a type `scripts/release.py` can
   classify and a body that is not empty. A file the release cannot classify
   would fail after the merge, in a run nobody is watching, so it is refused
   here instead.
2. If the branch also changes `opengwasdb/`, it must add at least one such
   fragment. `no-changelog` exempts a pull request from this requirement --
   not from the first check, because the label says "no notes are needed", not
   "no note may be read".

Editing `CHANGELOG.md` no longer satisfies anything: it holds the history up
to the cutover, and new entries are the GitHub Releases the workflow publishes.

Runs locally as well as in CI:

    pixi run -e dev python scripts/check_changelog.py --base-ref origin/main

and with `--labels`, as CI runs it, to see the exemption:

    pixi run -e dev python scripts/check_changelog.py --base-ref origin/main \
        --labels no-changelog
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

FRAGMENTS_DIR = "changelog.d/"
FRAGMENT_TYPES = ("breaking", "added", "changed", "fixed", "internal")
FRAGMENT_PATTERN = re.compile(
    r"^(?P<slug>.+)\.(?P<kind>" + "|".join(FRAGMENT_TYPES) + r")\.md$"
)
WATCHED = ("opengwasdb/",)
SKIP_LABEL = "no-changelog"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def changed_files(base_ref: str) -> list[str]:
    merge_base = _git("merge-base", base_ref, "HEAD").strip()
    return [p for p in _git("diff", "--name-only", merge_base, "HEAD").splitlines() if p]


def added_files(base_ref: str) -> list[str]:
    """Files this branch creates, which is what a fragment has to be."""
    merge_base = _git("merge-base", base_ref, "HEAD").strip()
    return [
        p
        for p in _git(
            "diff", "--diff-filter=A", "--name-only", merge_base, "HEAD"
        ).splitlines()
        if p
    ]


def added_fragments(base_ref: str) -> list[str]:
    """The files this branch adds under `changelog.d/`, which claim to be notes.

    Every added file there counts, whatever its extension: a note named
    `333-a.fixed.markdown` would otherwise be ignored in silence and the merge
    would publish a version without it. `README.md` is the one file that is not
    a fragment. A fragment that is staged but not committed counts too, and so
    does one that is not added to the index at all: the author runs this check
    in both states, and refusing a fragment that exists is the kind of false
    alarm that teaches people to reach for the `no-changelog` label.
    """
    merge_base = _git("merge-base", base_ref, "HEAD").strip()
    added = set(added_files(base_ref))
    added.update(
        path
        for path in _git(
            "diff", "--cached", "--diff-filter=A", "--name-only", merge_base
        ).splitlines()
        if path
    )
    added.update(
        path
        for path in _git("ls-files", "--others", "--exclude-standard").splitlines()
        if path
    )
    return sorted(
        path
        for path in added
        if path.startswith(FRAGMENTS_DIR) and path != f"{FRAGMENTS_DIR}README.md"
    )


def fragment_errors(fragments: list[str]) -> list[str]:
    """Why each fragment cannot be read, empty when they all can."""
    errors = []
    for fragment in fragments:
        name = Path(fragment).name
        match = FRAGMENT_PATTERN.match(name)
        if match is None:
            errors.append(
                f"    {fragment}: not named `<slug>.<type>.md` with type one of "
                f"{', '.join(FRAGMENT_TYPES)}"
            )
            continue
        if Path(fragment).parent != Path(FRAGMENTS_DIR.rstrip("/")):
            errors.append(f"    {fragment}: fragments live directly under {FRAGMENTS_DIR}")
            continue
        try:
            text = Path(fragment).read_text(encoding="utf-8").strip()
        except OSError as error:
            errors.append(f"    {fragment}: cannot be read ({error})")
            continue
        if not text:
            errors.append(f"    {fragment}: the fragment is empty")
    return errors


def _watched(paths: list[str]) -> list[str]:
    return [p for p in paths if p.startswith(WATCHED)]


def _listed(paths: list[str]) -> str:
    shown = "".join(f"    {p}\n" for p in paths[:10])
    if len(paths) > 10:
        shown += f"    ... and {len(paths) - 10} more\n"
    return shown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base-ref", required=True, help="branch this change targets, e.g. origin/main"
    )
    parser.add_argument(
        "--labels",
        default="",
        help="the pull request's labels, comma separated, as CI passes them",
    )
    args = parser.parse_args(argv)
    labels = {label.strip() for label in args.labels.split(",") if label.strip()}

    fragments = added_fragments(args.base_ref)
    errors = fragment_errors(fragments)
    if errors:
        print(
            f"changelog.d/: {len(errors)} fragment(s) cannot be read:\n"
            + "\n".join(errors)
            + f"\n\nSee {FRAGMENTS_DIR}README.md for the file name and body a "
            "fragment needs.",
            file=sys.stderr,
        )
        return 1

    watched = _watched(changed_files(args.base_ref))
    if not watched:
        print(f"No changes under {'/'.join(WATCHED)} — a fragment is not required.")
        return 0
    if fragments:
        print(
            f"{len(fragments)} changelog fragment(s) added under {FRAGMENTS_DIR} "
            f"alongside {len(watched)} source file(s)."
        )
        return 0
    if SKIP_LABEL in labels:
        print(
            f"{len(watched)} source file(s) changed and no fragment was added, "
            f"exempted by the '{SKIP_LABEL}' label."
        )
        return 0

    print(
        f"this change alters {len(watched)} file(s) under {'/'.join(WATCHED)} "
        "but adds no changelog fragment:\n"
        + _listed(watched)
        + f"\nAdd {FRAGMENTS_DIR}<slug>.<type>.md — type one of "
        f"{', '.join(FRAGMENT_TYPES)} — describing what a user would notice;\n"
        f"{FRAGMENTS_DIR}README.md says what a fragment should say, and its type "
        "sets the version bump.\n"
        f"If this change genuinely has no user-visible effect, apply the "
        f"'{SKIP_LABEL}' label to the pull request.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
