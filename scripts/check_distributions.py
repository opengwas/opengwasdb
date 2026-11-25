#!/usr/bin/env python3
"""Hold a built source archive and wheel to what the repository says they hold.

A distribution goes wrong silently: a wheel missing a reader column map
installs, imports and answers `--help`, and fails only when a user reads a
GWAS-SSF file; a source archive carrying the benchmark documents uploads fine
at twenty times the size. `twine check` sees neither. So each archive is
compared with the repository it was built from (#438):

* there is exactly one source archive and one wheel, the wheel is pure Python
  (`py3-none-any`), and both carry one version, which is in both filenames;
* the files under `opengwasdb/` in each archive are exactly the files git
  tracks there -- nothing dropped, nothing stray;
* the source archive holds nothing at its top level beyond what a wheel build
  from it reads (`pyproject.toml`'s `[tool.hatch.build.targets.sdist]`);
* the licence the metadata declares is `pyproject.toml`'s, and the licence
  text in each archive is the repository's `LICENSE`, byte for byte;
* the metadata's project URLs are `pyproject.toml`'s;
* the metadata requires `ogquery`, the Store Reader, with the range `pyproject.toml`
  declares (#475): a wheel without it installs and imports, and fails on the first
  read of a Store Release.

Run it on a `python -m build` output directory, from the checkout the archives
were built from:

    python scripts/check_distributions.py dist/

`--version` additionally requires the archives to carry exactly that version,
which is how a release build proves it was built from its tag.

Standard library only: it runs in the bare build environment CI makes, not
under Pixi.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from pathlib import Path

PACKAGE = "opengwasdb"
LICENSE_FILE = "LICENSE"
WHEEL_TAG = "py3-none-any"
# What hatchling puts at a source archive's top level beyond the include list.
SDIST_GENERATED = frozenset({"PKG-INFO", "pyproject.toml", ".gitignore"})
# Dependencies whose absence from the metadata is a silent failure, not a style
# point: the Store Reader every query reads through (#475).
REQUIRED_DEPENDENCIES = ("ogquery",)


class DistributionError(Exception):
    """The archives do not hold what the repository says they should."""


@dataclass(frozen=True)
class Expected:
    """What the repository says the archives hold."""

    tracked: frozenset[str]  # paths under `opengwasdb/` git tracks, repo-relative
    sdist_top_level: frozenset[str]  # every name allowed at the sdist's top level
    licence_expression: str
    licence_text: bytes
    project_urls: dict[str, str]
    # REQUIRED_DEPENDENCIES name -> the version specifiers pyproject.toml gives it
    requirements: dict[str, frozenset[str]]


@dataclass(frozen=True)
class Archive:
    """One built archive, read into what the checks compare."""

    filename: str
    files: dict[str, bytes]  # repo-relative path -> content, prefix stripped
    metadata: Message


def _metadata(raw: bytes) -> Message:
    return BytesParser().parsebytes(raw)


def read_sdist(path: Path) -> Archive:
    """The source archive's files, with their `<name>-<version>/` prefix removed."""
    files: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            prefix, _, rest = member.name.partition("/")
            if not rest:
                raise DistributionError(f"{path.name}: {member.name} is outside the top directory")
            handle = tar.extractfile(member)
            assert handle is not None  # a regular file always has content
            files[rest] = handle.read()
    if "PKG-INFO" not in files:
        raise DistributionError(f"{path.name}: no PKG-INFO")
    return Archive(path.name, files, _metadata(files["PKG-INFO"]))


def read_wheel(path: Path) -> Archive:
    """The wheel's files, with its `.dist-info/` directory kept as written."""
    with zipfile.ZipFile(path) as wheel:
        files = {name: wheel.read(name) for name in wheel.namelist() if not name.endswith("/")}
    metadata = [name for name in files if name.endswith(".dist-info/METADATA")]
    if len(metadata) != 1:
        raise DistributionError(f"{path.name}: {len(metadata)} METADATA files, expected one")
    return Archive(path.name, files, _metadata(files[metadata[0]]))


def _package_files(files: Iterable[str]) -> set[str]:
    return {name for name in files if name.startswith(f"{PACKAGE}/")}


def _compare_package(archive: Archive, tracked: frozenset[str]) -> list[str]:
    held = _package_files(archive.files)
    problems = [f"{archive.filename}: missing {name}" for name in sorted(tracked - held)]
    problems += [
        f"{archive.filename}: not tracked by git: {name}" for name in sorted(held - tracked)
    ]
    return problems


def _compare_metadata(archive: Archive, expected: Expected) -> list[str]:
    problems = []
    licence = archive.metadata.get("License-Expression")
    if licence != expected.licence_expression:
        problems.append(
            f"{archive.filename}: License-Expression is {licence!r}, "
            f"pyproject.toml declares {expected.licence_expression!r}"
        )
    if archive.metadata.get_all("License-File") != [LICENSE_FILE]:
        problems.append(
            f"{archive.filename}: License-File is {archive.metadata.get_all('License-File')}, "
            f"expected [{LICENSE_FILE!r}]"
        )
    urls = {}
    for entry in archive.metadata.get_all("Project-URL") or []:
        label, _, url = entry.partition(",")
        urls[label.strip()] = url.strip()
    if urls != expected.project_urls:
        problems.append(
            f"{archive.filename}: project URLs {urls} are not pyproject.toml's "
            f"{expected.project_urls}"
        )
    return problems


def _requirement_parts(requirement: str) -> tuple[str, frozenset[str]]:
    """A requirement string as (name, version specifiers): ``ogquery>=0.1,<0.2`` and
    hatchling's ``ogquery<0.2,>=0.1`` are the same requirement."""
    text = requirement.partition(";")[0].replace(" ", "")
    name = text
    for mark in "<>=!~([":
        name = name.partition(mark)[0]
    specifiers = text[len(name):].strip("()")
    return name, frozenset(part for part in specifiers.split(",") if part)


def _compare_requirements(archive: Archive, expected: Expected) -> list[str]:
    declared = {}
    for entry in archive.metadata.get_all("Requires-Dist") or []:
        if "extra ==" in entry:
            continue
        name, specifiers = _requirement_parts(entry)
        declared[name] = specifiers
    problems = []
    for name in REQUIRED_DEPENDENCIES:
        if name not in declared:
            problems.append(f"{archive.filename}: Requires-Dist has no {name}")
        elif declared[name] != expected.requirements[name]:
            problems.append(
                f"{archive.filename}: Requires-Dist {name} {sorted(declared[name])} is not "
                f"pyproject.toml's {sorted(expected.requirements[name])}"
            )
    return problems


def _licence_text(archive: Archive, path: str, expected: bytes) -> list[str]:
    if path not in archive.files:
        return [f"{archive.filename}: no licence text at {path}"]
    if archive.files[path] != expected:
        return [f"{archive.filename}: {path} is not the repository's {LICENSE_FILE}"]
    return []


def check_sdist(sdist: Archive, expected: Expected) -> list[str]:
    """Everything wrong with the source archive."""
    top_level = {name.split("/", 1)[0] for name in sdist.files}
    problems = [
        f"{sdist.filename}: unexpected top-level entry {name}"
        for name in sorted(top_level - expected.sdist_top_level)
    ]
    problems += _compare_package(sdist, expected.tracked)
    problems += _compare_metadata(sdist, expected)
    problems += _compare_requirements(sdist, expected)
    problems += _licence_text(sdist, LICENSE_FILE, expected.licence_text)
    return problems


def check_wheel(wheel: Archive, expected: Expected) -> list[str]:
    """Everything wrong with the wheel."""
    if not wheel.filename.endswith(f"-{WHEEL_TAG}.whl"):
        return [f"{wheel.filename}: not a pure-Python {WHEEL_TAG} wheel"]
    version = wheel.metadata["Version"]
    dist_info = f"{PACKAGE}-{version}.dist-info"
    allowed = {
        f"{dist_info}/{name}" for name in ("METADATA", "WHEEL", "RECORD", "entry_points.txt")
    }
    allowed.add(f"{dist_info}/licenses/{LICENSE_FILE}")
    stray = set(wheel.files) - _package_files(wheel.files) - allowed
    problems = [f"{wheel.filename}: unexpected file {name}" for name in sorted(stray)]
    problems += _compare_package(wheel, expected.tracked)
    problems += _compare_metadata(wheel, expected)
    problems += _compare_requirements(wheel, expected)
    problems += _licence_text(wheel, f"{dist_info}/licenses/{LICENSE_FILE}", expected.licence_text)
    return problems


def check_versions(sdist: Archive, wheel: Archive, required: str | None) -> list[str]:
    """Both archives carry one version, it is in both filenames, and it is `required`."""
    versions = {sdist.metadata["Version"], wheel.metadata["Version"]}
    if len(versions) != 1:
        return [f"the archives disagree on the version: {sorted(versions)}"]
    (version,) = versions
    problems = []
    if sdist.filename != f"{PACKAGE}-{version}.tar.gz":
        problems.append(f"{sdist.filename}: filename does not carry version {version}")
    if not wheel.filename.startswith(f"{PACKAGE}-{version}-"):
        problems.append(f"{wheel.filename}: filename does not carry version {version}")
    if required is not None and version != required:
        problems.append(f"the archives are version {version}, not the required {required}")
    return problems


def find_archives(dist: Path) -> tuple[Path, Path]:
    """The one source archive and the one wheel in `dist`, and nothing else."""
    entries = sorted(dist.iterdir())
    kinds = sorted(_archive_kind(path) for path in entries)
    if kinds != ["sdist", "wheel"]:
        raise DistributionError(
            f"{dist} must hold exactly one .tar.gz and one .whl; it holds "
            f"{[path.name for path in entries]}"
        )
    sdist, wheel = sorted(entries, key=_archive_kind)
    return sdist, wheel


def _archive_kind(path: Path) -> str:
    if path.name.endswith(".tar.gz"):
        return "sdist"
    if path.name.endswith(".whl"):
        return "wheel"
    return "other"


def check(dist: Path, expected: Expected, required_version: str | None = None) -> list[str]:
    """Every problem with the archives in `dist`; empty when they are right."""
    sdist_path, wheel_path = find_archives(dist)
    sdist, wheel = read_sdist(sdist_path), read_wheel(wheel_path)
    return (
        check_versions(sdist, wheel, required_version)
        + check_sdist(sdist, expected)
        + check_wheel(wheel, expected)
    )


def _declared_requirements(dependencies: list[str]) -> dict[str, frozenset[str]]:
    declared = dict(_requirement_parts(entry) for entry in dependencies)
    missing = [name for name in REQUIRED_DEPENDENCIES if name not in declared]
    if missing:
        raise DistributionError(f"pyproject.toml's dependencies lack {', '.join(missing)}")
    return {name: declared[name] for name in REQUIRED_DEPENDENCIES}


def expected_from_repository(root: Path) -> Expected:
    """What the checkout at `root` says the archives hold."""
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", f"{PACKAGE}/"],
        capture_output=True,
        check=True,
    ).stdout.decode()
    files = frozenset(name for name in tracked.split("\0") if name)
    if not files:
        raise DistributionError(f"git tracks nothing under {root / PACKAGE}")
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    project = pyproject["project"]
    includes = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    return Expected(
        tracked=files,
        sdist_top_level=SDIST_GENERATED | {name.strip("/") for name in includes},
        licence_expression=project["license"],
        licence_text=(root / LICENSE_FILE).read_bytes(),
        project_urls=dict(project["urls"]),
        requirements=_declared_requirements(project["dependencies"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("dist", type=Path, help="the directory `python -m build` wrote")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--version", help="the version both archives must carry")
    args = parser.parse_args(argv)
    try:
        problems = check(args.dist, expected_from_repository(args.repo_root), args.version)
    except DistributionError as error:
        problems = [str(error)]
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    sdist, wheel = find_archives(args.dist)
    print(f"ok: {sdist.name} and {wheel.name} hold what the repository says")
    return 0


if __name__ == "__main__":
    sys.exit(main())
