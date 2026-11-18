#!/usr/bin/env python3
"""Which tagged releases PyPI is owed, and the checks around uploading them (#439).

`scripts/release.py` tags every merge to `main` and creates its GitHub
Release. This script does the rest of a release's life, for
`.github/workflows/publish.yml`, the one workflow PyPI's Trusted Publisher
names. The workflow uploads; every decision is made here, where it is tested.

**Reconcile, never trust the queue.** A run is not promised per release, so
each run asks the index what it holds for every eligible tag and plans every
tag that lacks either of its two files, oldest first. A burst of merges, a lost
run, a failed upload and a partial upload all look the same to the next run:
files missing from the index.

**A boundary, not history.** Only tags at or after `PYPI_FIRST_RELEASE` are
ever eligible for PyPI: `v0.11.1`, the tag rehearsed on TestPyPI (#440).
Set to `None`, it would stop automatic publication to PyPI. TestPyPI is never
published automatically; a run names its tag.

**One set of bytes per version.** The verified archives are attached to the
tag's GitHub Release before anything is uploaded, and a retry downloads them
rather than rebuilding. Whatever the index already holds must match them by
sha256, file for file: a file the index holds that this release did not build,
or holds with other content, stops the run. A published file is never
replaced, and a tag is never moved (`release.py` refuses that).

Every answer from the network is either an answer or an error. A version the
index has never seen is a 404, which means "no files"; any other failure stops
the run instead of reading as "nothing uploaded yet".

    python scripts/pypi_release.py plan --index pypi
    python scripts/pypi_release.py plan --index testpypi --tag v0.12.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# `release` is the sibling script: running this file puts `scripts/` first on
# the import path, and a test loads `release` before loading this.
from release import (
    REPOSITORY,
    TAG_PREFIX,
    ReleaseError,
    _gh,
    parse_tag,
    remote_tags,
)

#: The first tag published to PyPI; no older tag ever is. It is the tag
#: rehearsed on TestPyPI (#440, 18 Nov 2026), so the first PyPI upload is the
#: same bytes the rehearsal installed. `None` would turn PyPI publication off.
PYPI_FIRST_RELEASE: str | None = "v0.11.1"

PROJECT = "opengwasdb"
INDEXES = {"pypi": "https://pypi.org", "testpypi": "https://test.pypi.org"}

#: (index, version) -> {filename: sha256} of the files the index holds.
Lookup = Callable[[str, str], dict[str, str]]


@dataclass(frozen=True)
class OwedRelease:
    """One tag whose files the index does not yet hold in full."""

    tag: str
    version: str
    commit: str
    held: list[str]  # the files the index already has, for the log


def distribution_names(version: str) -> tuple[str, str]:
    """The source archive and wheel a release of `version` consists of."""
    return f"{PROJECT}-{version}.tar.gz", f"{PROJECT}-{version}-py3-none-any.whl"


# ── The index ───────────────────────────────────────────────────────────────


def _fetch_json(url: str) -> object | None:
    """`url` as JSON, or `None` when -- and only when -- the server says 404."""
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "opengwasdb-release"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload: object = json.load(response)
            return payload
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise ReleaseError(f"{url} answered HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise ReleaseError(f"{url} could not be read: {error}") from error


def index_files(
    index: str, version: str, fetch: Callable[[str], object] = _fetch_json
) -> dict[str, str]:
    """{filename: sha256} of the files `index` holds for `version`; {} for none."""
    url = f"{INDEXES[index]}/pypi/{PROJECT}/{version}/json"
    payload = fetch(url)
    if payload is None:
        return {}
    if not isinstance(payload, dict) or not isinstance(payload.get("urls"), list):
        raise ReleaseError(f"{url} did not list files with sha256 digests")
    try:
        return {entry["filename"]: entry["digests"]["sha256"] for entry in payload["urls"]}
    except (KeyError, TypeError) as error:
        raise ReleaseError(f"{url} did not list files with sha256 digests") from error


def files_to_upload(local: dict[str, str], held: dict[str, str]) -> list[str]:
    """The local files the index lacks, refusing anything it holds differently."""
    unexpected = sorted(set(held) - set(local))
    if unexpected:
        raise ReleaseError(
            f"the index holds {unexpected}, which this release did not build; "
            "resolve it by hand before publishing this version"
        )
    conflicting = sorted(name for name in held if held[name] != local[name])
    if conflicting:
        raise ReleaseError(
            f"the index holds {conflicting} with other content than the verified "
            "archives; a published file is never replaced"
        )
    return sorted(set(local) - set(held))


# ── Planning ────────────────────────────────────────────────────────────────


def eligible_tags(
    tags: dict[str, str], *, index: str, first: str | None, only: str | None = None
) -> list[str]:
    """The tags this run may publish to `index`, oldest first."""
    if only is not None:
        return [_named_tag(tags, only, index=index, first=first)]
    if index != "pypi":
        raise ReleaseError(f"{index} is never published automatically; name the tag with --tag")
    if first is None:
        return []
    floor = parse_tag(first)
    return sorted((tag for tag in tags if parse_tag(tag) >= floor), key=parse_tag)


def _named_tag(tags: dict[str, str], tag: str, *, index: str, first: str | None) -> str:
    if tag not in tags:
        raise ReleaseError(f"{tag} is not a release tag on the remote")
    if index == "pypi" and (first is None or parse_tag(tag) < parse_tag(first)):
        raise ReleaseError(f"{tag} is not eligible for PyPI: the first PyPI release is {first}")
    return tag


def plan(
    tags: dict[str, str],
    *,
    index: str,
    first: str | None,
    only: str | None = None,
    lookup: Lookup | None = None,
) -> list[OwedRelease]:
    """Every eligible tag the index does not hold both files of, oldest first."""
    owed = []
    for tag in eligible_tags(tags, index=index, first=first, only=only):
        version = tag.removeprefix(TAG_PREFIX)
        held = (lookup or index_files)(index, version)
        stray = sorted(set(held) - set(distribution_names(version)))
        if stray:
            raise ReleaseError(f"{index} holds {stray} for {tag}, which no release builds")
        if len(held) < len(distribution_names(version)):
            owed.append(OwedRelease(tag, version, tags[tag], sorted(held)))
    return owed


# ── The archives ────────────────────────────────────────────────────────────


def local_files(dist: Path, version: str) -> dict[str, str]:
    """{filename: sha256} of `dist`, which must hold exactly this version's pair."""
    names = sorted(path.name for path in dist.iterdir())
    expected = sorted(distribution_names(version))
    if names != expected:
        raise ReleaseError(f"{dist} holds {names}, not exactly {expected}")
    return {name: hashlib.sha256((dist / name).read_bytes()).hexdigest() for name in names}


def stage(dist: Path, version: str, held: dict[str, str], upload: Path) -> list[str]:
    """Copy into `upload` the verified files the index lacks, after the comparison."""
    missing = files_to_upload(local_files(dist, version), held)
    upload.mkdir(parents=True)
    for name in missing:
        shutil.copy2(dist / name, upload / name)
    return missing


def confirm(
    dist: Path,
    version: str,
    *,
    index: str,
    attempts: int,
    interval: float,
    lookup: Lookup = index_files,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait until the index holds exactly the verified files, byte for byte.

    The index's JSON is served through a cache, so a file uploaded seconds ago
    can be missing from it for a while; a difference in content is never
    waited out.
    """
    local = local_files(dist, version)
    for attempt in range(attempts):
        if not files_to_upload(local, lookup(index, version)):
            return
        if attempt + 1 < attempts:
            sleep(interval)
    raise ReleaseError(
        f"{index} still lacks part of {version} after {attempts} checks; "
        "re-run publish.yml to finish it"
    )


# ── The GitHub Release's copy ───────────────────────────────────────────────


def release_assets(tag: str, repository: str) -> list[str]:
    """The asset names on `tag`'s GitHub Release, which must already exist."""
    payload = json.loads(_gh("release", "view", tag, "--repo", repository, "--json", "assets"))
    return [asset["name"] for asset in payload["assets"]]


def stored_pair(tag: str, version: str, assets: list[str]) -> bool:
    """Whether the Release holds this version's archives; refuses half of them."""
    expected = set(distribution_names(version))
    held = {name for name in assets if name.startswith(f"{PROJECT}-")}
    if not held:
        return False
    if held == expected:
        return True
    raise ReleaseError(
        f"the Release for {tag} holds {sorted(held)}, not the pair {sorted(expected)}; "
        "delete the partial asset by hand and re-run"
    )


def fetch_stored(tag: str, version: str, dist: Path, repository: str) -> bool:
    """Download the Release's verified archives into `dist`; False if it has none."""
    if not stored_pair(tag, version, release_assets(tag, repository)):
        return False
    dist.mkdir(parents=True, exist_ok=True)
    for name in distribution_names(version):
        _gh("release", "download", tag, "--repo", repository, "--pattern", name, "--dir", str(dist))
    return True


def store(tag: str, version: str, dist: Path, repository: str) -> bool:
    """Attach the verified archives to the Release; False when they already are."""
    local_files(dist, version)
    if stored_pair(tag, version, release_assets(tag, repository)):
        return False
    paths = [str(dist / name) for name in distribution_names(version)]
    _gh("release", "upload", tag, *paths, "--repo", repository)
    return True


# ── The command line ────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="the tags the index is owed, oldest first")
    planning.add_argument("--tag", default=None, help="only this tag (required for testpypi)")
    planning.add_argument("--json-out", type=Path, default=None)
    for name, help_text in (
        ("fetch", "download the Release's verified archives, if it has them"),
        ("store", "attach the verified archives to the Release"),
        ("stage", "compare with the index and copy what it lacks to --upload"),
        ("confirm", "wait until the index holds exactly the verified archives"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--tag", required=True)
        command.add_argument("--dist", type=Path, required=True)
    commands.choices["stage"].add_argument("--upload", type=Path, required=True)
    commands.choices["confirm"].add_argument("--attempts", type=int, default=20)
    commands.choices["confirm"].add_argument("--interval", type=float, default=30.0)
    for command in commands.choices.values():
        command.add_argument("--index", choices=sorted(INDEXES), default="pypi")
        command.add_argument("--repository", default=REPOSITORY)
    return parser


def _run_plan(args: argparse.Namespace) -> None:
    owed = plan(remote_tags(), index=args.index, first=PYPI_FIRST_RELEASE, only=args.tag)
    for release in owed:
        print(f"{release.tag} {release.commit[:12]} owed to {args.index}; it holds {release.held}")
    if not owed:
        print(f"Nothing owed to {args.index} (first PyPI release: {PYPI_FIRST_RELEASE}).")
    if args.json_out is not None:
        # Strings only: each entry becomes one leg of a GitHub Actions matrix.
        matrix = [{"tag": r.tag, "version": r.version, "commit": r.commit} for r in owed]
        args.json_out.write_text(json.dumps(matrix))


def _version(args: argparse.Namespace) -> str:
    parse_tag(args.tag)
    return str(args.tag).removeprefix(TAG_PREFIX)


def _run_fetch(args: argparse.Namespace) -> None:
    found = fetch_stored(args.tag, _version(args), args.dist, args.repository)
    print("stored" if found else "absent")


def _run_store(args: argparse.Namespace) -> None:
    attached = store(args.tag, _version(args), args.dist, args.repository)
    print(f"attached to {args.tag}" if attached else f"{args.tag} already holds them")


def _run_stage(args: argparse.Namespace) -> None:
    version = _version(args)
    missing = stage(args.dist, version, index_files(args.index, version), args.upload)
    print(f"{args.index} lacks {missing}" if missing else f"{args.index} holds all of {version}")


def _run_confirm(args: argparse.Namespace) -> None:
    confirm(
        args.dist, _version(args), index=args.index, attempts=args.attempts, interval=args.interval
    )
    print(f"{args.index} holds {args.tag}'s verified archives")


_RUNNERS = {
    "plan": _run_plan,
    "fetch": _run_fetch,
    "store": _run_store,
    "stage": _run_stage,
    "confirm": _run_confirm,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _RUNNERS[args.command](args)
    except (ReleaseError, OSError) as error:
        print(f"pypi-release: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
