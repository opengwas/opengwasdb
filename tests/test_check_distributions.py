"""`scripts/check_distributions.py` and `scripts/smoke_test_installed.py` (#438).

The archive checks run on small archives built here, shaped as hatchling shapes
them, so each defect a real build can carry -- a dropped resource, a stray
file, a bloated source archive, two versions, the wrong licence -- is put in
one at a time and must be named. CI runs both scripts on the real build; the
smoke test's own checks need an installed wheel, so only the two that must
refuse the developer's editable install are exercised here.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest
from script_modules import load_script

import opengwasdb

check_distributions = load_script("check_distributions")
smoke = load_script("smoke_test_installed")

REPO = Path(__file__).resolve().parent.parent
VERSION = "1.2.3"
LICENCE = b"the licence text\n"
EXPRESSION = "LicenseRef-Test"
URLS = {"Repository": "https://example.org/repo", "Issues": "https://example.org/issues"}
PACKAGE_FILES = {
    "opengwasdb/__init__.py": b"",
    "opengwasdb/readers/column_maps/gwas-ssf.toml": b"[map]\n",
}
EXPECTED = check_distributions.Expected(
    tracked=frozenset(PACKAGE_FILES),
    sdist_top_level=frozenset(
        {"PKG-INFO", "pyproject.toml", ".gitignore", "opengwasdb", "README.md", "LICENSE"}
    ),
    licence_expression=EXPRESSION,
    licence_text=LICENCE,
    project_urls=URLS,
)


def _metadata(version: str = VERSION, expression: str = EXPRESSION) -> bytes:
    lines = [
        "Metadata-Version: 2.4",
        "Name: opengwasdb",
        f"Version: {version}",
        f"License-Expression: {expression}",
        "License-File: LICENSE",
    ]
    lines += [f"Project-URL: {label}, {url}" for label, url in URLS.items()]
    return ("\n".join(lines) + "\n\n").encode()


def _sdist_files(version: str = VERSION) -> dict[str, bytes]:
    return {
        **PACKAGE_FILES,
        "PKG-INFO": _metadata(version),
        "pyproject.toml": b"",
        "README.md": b"",
        "LICENSE": LICENCE,
    }


def _wheel_files(version: str = VERSION) -> dict[str, bytes]:
    info = f"opengwasdb-{version}.dist-info"
    return {
        **PACKAGE_FILES,
        f"{info}/METADATA": _metadata(version),
        f"{info}/WHEEL": b"",
        f"{info}/RECORD": b"",
        f"{info}/licenses/LICENSE": LICENCE,
    }


def _write(
    dist: Path,
    sdist: dict[str, bytes],
    wheel: dict[str, bytes],
    version: str = VERSION,
    wheel_version: str | None = None,
) -> Path:
    dist.mkdir(exist_ok=True)
    with tarfile.open(dist / f"opengwasdb-{version}.tar.gz", "w:gz") as tar:
        for name, content in sdist.items():
            member = tarfile.TarInfo(f"opengwasdb-{version}/{name}")
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    wheel_name = f"opengwasdb-{wheel_version or version}-py3-none-any.whl"
    with zipfile.ZipFile(dist / wheel_name, "w") as archive:
        for name, content in wheel.items():
            archive.writestr(name, content)
    return dist


def _problems(
    tmp_path: Path,
    sdist: dict[str, bytes] | None = None,
    wheel: dict[str, bytes] | None = None,
    required: str | None = None,
) -> list[str]:
    dist = _write(tmp_path / "dist", sdist or _sdist_files(), wheel or _wheel_files())
    result: list[str] = check_distributions.check(dist, EXPECTED, required)
    return result


def test_archives_holding_what_the_repository_says_pass(tmp_path: Path) -> None:
    dist = _write(tmp_path / "dist", _sdist_files(), _wheel_files())
    sdist, wheel = check_distributions.find_archives(dist)
    # The fixture is the shape being judged: a resource beside the code, in both.
    assert (
        "opengwasdb/readers/column_maps/gwas-ssf.toml"
        in check_distributions.read_wheel(wheel).files
    )
    assert (
        "opengwasdb/readers/column_maps/gwas-ssf.toml"
        in check_distributions.read_sdist(sdist).files
    )
    assert check_distributions.check(dist, EXPECTED, VERSION) == []


def test_a_wheel_that_dropped_a_resource_is_refused(tmp_path: Path) -> None:
    wheel = _wheel_files()
    del wheel["opengwasdb/readers/column_maps/gwas-ssf.toml"]
    assert _problems(tmp_path, wheel=wheel) == [
        f"opengwasdb-{VERSION}-py3-none-any.whl: missing "
        "opengwasdb/readers/column_maps/gwas-ssf.toml"
    ]


def test_a_package_file_git_does_not_track_is_refused(tmp_path: Path) -> None:
    sdist = {**_sdist_files(), "opengwasdb/scratch.py": b""}
    assert _problems(tmp_path, sdist=sdist) == [
        f"opengwasdb-{VERSION}.tar.gz: not tracked by git: opengwasdb/scratch.py"
    ]


def test_repository_content_in_the_source_archive_is_refused(tmp_path: Path) -> None:
    sdist = {**_sdist_files(), "docs/benchmark-output/report.html": b"<html>"}
    assert _problems(tmp_path, sdist=sdist) == [
        f"opengwasdb-{VERSION}.tar.gz: unexpected top-level entry docs"
    ]


def test_a_stray_wheel_file_is_refused(tmp_path: Path) -> None:
    wheel = {**_wheel_files(), "tests/test_x.py": b""}
    assert _problems(tmp_path, wheel=wheel) == [
        f"opengwasdb-{VERSION}-py3-none-any.whl: unexpected file tests/test_x.py"
    ]


def test_archives_of_two_versions_are_refused(tmp_path: Path) -> None:
    dist = _write(tmp_path / "dist", _sdist_files(), _wheel_files("1.2.4"), wheel_version="1.2.4")
    assert check_distributions.check(dist, EXPECTED) == [
        "the archives disagree on the version: ['1.2.3', '1.2.4']"
    ]


def test_archives_not_of_the_required_version_are_refused(tmp_path: Path) -> None:
    assert _problems(tmp_path, required="1.2.4") == [
        "the archives are version 1.2.3, not the required 1.2.4"
    ]


def test_a_licence_text_that_is_not_the_repository_s_is_refused(tmp_path: Path) -> None:
    sdist = {**_sdist_files(), "LICENSE": b"MIT\n"}
    assert _problems(tmp_path, sdist=sdist) == [
        f"opengwasdb-{VERSION}.tar.gz: LICENSE is not the repository's LICENSE"
    ]


def test_a_declared_licence_that_is_not_pyproject_s_is_refused(tmp_path: Path) -> None:
    info = f"opengwasdb-{VERSION}.dist-info/METADATA"
    wheel = {**_wheel_files(), info: _metadata(expression="MIT")}
    assert _problems(tmp_path, wheel=wheel) == [
        f"opengwasdb-{VERSION}-py3-none-any.whl: License-Expression is 'MIT', "
        f"pyproject.toml declares {EXPRESSION!r}"
    ]


def test_anything_but_one_sdist_and_one_wheel_is_refused(tmp_path: Path) -> None:
    dist = _write(tmp_path / "dist", _sdist_files(), _wheel_files())
    (dist / "opengwasdb-1.2.2.tar.gz").write_bytes(b"")
    with pytest.raises(check_distributions.DistributionError, match="exactly one"):
        check_distributions.find_archives(dist)


def test_the_repository_expects_its_resources_licence_and_urls() -> None:
    expected = check_distributions.expected_from_repository(REPO)
    assert "opengwasdb/readers/column_maps/gwas-ssf.toml" in expected.tracked
    assert expected.licence_expression == "GPL-3.0-only"
    assert expected.licence_text.lstrip().startswith(b"GNU GENERAL PUBLIC LICENSE\n")
    assert expected.sdist_top_level == {
        "PKG-INFO",
        "pyproject.toml",
        ".gitignore",
        "opengwasdb",
        "README.md",
        "LICENSE",
    }
    assert set(expected.project_urls) >= {"Repository", "Issues", "Documentation"}


def test_the_smoke_test_refuses_a_package_imported_from_a_checkout() -> None:
    # Under Pixi the package is this checkout's, installed editable: exactly
    # what the smoke test exists to tell apart from an installed wheel.
    assert Path(opengwasdb.__file__).resolve().is_relative_to(REPO)
    with pytest.raises(smoke.SmokeTestFailure, match="a checkout is on the import path"):
        smoke.check_import_location()


def test_the_smoke_test_refuses_a_version_other_than_the_built_one() -> None:
    with pytest.raises(smoke.SmokeTestFailure, match="the wheel was built as 0.0.1"):
        smoke.check_version("0.0.1")


def test_the_smoke_test_expects_what_the_checkout_returns_and_documents(tmp_path: Path) -> None:
    # CI runs the smoke test only on a built wheel, and not on a required
    # check, so a change to the query output once left it expecting rows the
    # package no longer returned (#466's `beta`), and publishing would have
    # failed. Run its tiny-Store check here against this checkout's command,
    # and hold getting-started.md to the same rows.
    rows = smoke.EXPECTED_PHEWAS.splitlines()
    assert rows[0].startswith("analysis_id\t") and len(rows) == 3
    documented = (REPO / "docs" / "getting-started.md").read_text()
    assert smoke.EXPECTED_PHEWAS in documented
    smoke.check_tiny_store(smoke.installed_command(), REPO / "docs" / "examples" / "tiny", tmp_path)
