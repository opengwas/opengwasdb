"""The package version comes from the git tag (issue #333).

Every merge to `main` is tagged `vX.Y.Z`, and `hatch-vcs` turns that tag into
the distribution's version at build time. So the only version this package may
report is the installed distribution's: a release checkout reports the tag, a
checkout past the tag reports a development version past it, and a run with no
installed metadata reports a fallback that cannot be mistaken for a release.

The tests below read the *installed* distribution, which pixi built from this
checkout with hatch-vcs, rather than building a throwaway repository: a build
inside a test needs a network fetch of the build backend, and the fact under
test -- that the tag reaches the metadata -- is the same either way.
"""

from __future__ import annotations

import subprocess
import tomllib
from importlib.metadata import PackageNotFoundError
from pathlib import Path

from packaging.version import Version

import opengwasdb
from opengwasdb import version as version_module

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _latest_tag() -> str:
    """The newest `v*` tag, or a failure that says what is missing.

    No skip: the tag is what the version is supposed to come from, so a clone
    without tags cannot show that it does. CI checks out with `fetch-depth: 0`
    and `fetch-tags: true` for exactly this reason.
    """
    try:
        return _git("describe", "--tags", "--abbrev=0", "--match", "v*")
    except (OSError, subprocess.CalledProcessError) as error:
        raise AssertionError(
            f"this clone has no v* tag to compare the installed version against "
            f"({error}); fetch the tags, e.g. `git fetch --tags origin`"
        ) from error


def test_the_package_reports_the_installed_distribution_version() -> None:
    from importlib.metadata import version as installed_metadata_version

    assert opengwasdb.__version__ == installed_metadata_version("opengwasdb")
    assert version_module.installed_version() == installed_metadata_version("opengwasdb")


def test_a_tag_on_head_is_the_version_and_a_commit_past_it_is_newer() -> None:
    """The acceptance criterion, checked against the real tag and HEAD.

    With `v0.3.0` on `HEAD` the package reports `0.3.0`; one commit past it, a
    development version greater than `0.3.0`.

    What this cannot catch is an *editable* install whose metadata was built
    from an earlier commit: it reports a real development version of an older
    commit, not a wrong release, and the freshness it lacks is a property of
    the environment rather than of the version rule. `opengwasdb_git_hash`
    records the commit beside it for exactly that reason (issue #333's audit).
    """
    tag = _latest_tag()
    reported = Version(version_module.installed_version())
    tagged = Version(tag.removeprefix("v"))
    head_is_tagged = _git("rev-parse", "HEAD") == _git("rev-parse", f"{tag}^{{commit}}")
    if head_is_tagged:
        assert reported == tagged, (
            f"HEAD is {tag} but the installed metadata says {reported}; "
            "reinstall the environment (pixi install -e dev)"
        )
    else:
        assert reported > tagged, (
            f"the last tag is {tag} and HEAD is past it, but the installed "
            f"metadata says {reported}; reinstall the environment"
        )
        # A development version past the tag, and nothing else: the `.devN`
        # distance is setuptools-scm's, and the local segment names the commit.
        assert reported.is_devrelease


def test_the_fallback_is_a_version_no_release_can_claim(monkeypatch) -> None:
    def missing(_name: str) -> str:
        raise PackageNotFoundError(_name)

    monkeypatch.setattr(version_module, "version", missing)
    assert version_module.installed_version() == version_module.FALLBACK_VERSION
    assert version_module.FALLBACK_VERSION == "0.0.0+unknown"


def test_the_fallback_parses_and_sorts_below_every_release() -> None:
    """`0.0.0+unknown` is valid PEP 440 and obviously not a release.

    The constant it replaced was `"0.3.0"`, which a reader would take for a
    real build; the local segment says the version is unknown.
    """
    fallback = Version(version_module.FALLBACK_VERSION)
    assert fallback.public == "0.0.0"
    assert fallback.local == "unknown"
    assert fallback < Version("0.0.1")


def test_pyproject_takes_the_version_from_tags_with_the_same_fallback() -> None:
    """The build config and the runtime fallback are one value, not two.

    A wheel built without git metadata carries `fallback_version`; a process
    without installed metadata reads `FALLBACK_VERSION`. If they disagreed, the
    same unknown build would report two versions (issue #333).
    """
    config = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]
    assert project["dynamic"] == ["version"]
    assert "version" not in project, "a static version shadows the tag"

    hatch_version = config["tool"]["hatch"]["version"]
    assert hatch_version["source"] == "vcs"
    raw_options = hatch_version["raw-options"]
    assert raw_options["fallback_version"] == version_module.FALLBACK_VERSION
    assert raw_options["tag_regex"] == r"^v(?P<version>.*)$"
    # setuptools-scm's default node-and-date: a clean checkout carries the
    # commit alone (`+g1a2b3c4`), so the version is stable for as long as the
    # code is; only a dirty tree gains a date. The version is hashed into a
    # resolution record's fingerprint, which decides whether that record may
    # be resumed, so a date on every build would invalidate unfinished work
    # daily without a code change.
    assert raw_options["local_scheme"] == "node-and-date"


def test_a_resolved_manifest_record_reports_the_installed_version() -> None:
    """`opengwasdb_version` in a record is the installed distribution's.

    It is written to every record and, through `_MEASUREMENT_IDENTITY_KEYS`,
    hashed into each Analysis's fingerprint, so a stale or invented value
    either wrongly resumes a record or wrongly rebuilds it.
    """
    from opengwasdb.build import resolve_manifest

    assert resolve_manifest._get_opengwasdb_version() == opengwasdb.__version__


def test_without_installed_metadata_a_record_says_the_version_is_unknown(
    monkeypatch,
) -> None:
    from opengwasdb.build import resolve_manifest

    def missing(_name: str) -> str:
        raise PackageNotFoundError(_name)

    monkeypatch.setattr(version_module, "version", missing)
    assert (
        resolve_manifest._get_opengwasdb_version()
        == version_module.FALLBACK_VERSION
    )


def test_no_release_shaped_fallback_constant_remains() -> None:
    """The old `OPENGWASDB_PACKAGE_VERSION = "0.3.0"` must not come back.

    It was a fallback a reader could not tell from a real release, and it was
    read by two modules. One source of the fallback is the point (#333).
    """
    from opengwasdb.build import resolve_manifest

    assert not hasattr(resolve_manifest, "OPENGWASDB_PACKAGE_VERSION")
