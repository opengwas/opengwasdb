"""What opengwasdb holds of the Store Reader contract bundle beyond reading it (#474).

The bundle (`tests/store_reader_contract/`) is read by its own contract test, which
builds nothing. Two of its claims need opengwasdb itself: its generator re-stamps what a
build records of when and where it ran and refuses a release that keeps any of it, and
`ogdb validate` fails exactly the releases the README names (`LEGACY_STATES`), each on
the rule it names. That a fresh run of the generator writes the committed bundle byte for
byte is `pixi run -e dev contract-check` (`scripts/check_store_reader_contract.py`).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from opengwasdb.store.convert import validate_store

BUNDLE = Path(__file__).resolve().parent / "store_reader_contract"
RELEASES = BUNDLE / "releases"


def load_generator() -> ModuleType:
    """`generate.py`, imported by its path: its directory is not a package. Importing it
    puts the bundle's directory on the path, so `contract_reader` imports after it."""
    spec = importlib.util.spec_from_file_location(
        "store_reader_contract_generate", BUNDLE / "generate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    return load_generator()


def _built(root: Path, manifest: dict) -> Path:
    """A stand-in release: a manifest and an `overview.html`, as a build stamps them."""
    release = root / "release"
    release.mkdir()
    (release / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (release / "overview.html").write_text(
        f"<td>{manifest.get('created_at', '')}</td>", encoding="utf-8"
    )
    return release


def test_normalise_re_stamps_the_fields_it_knows(tmp_path: Path, generator: ModuleType):
    build_root = tmp_path / "build"
    release = _built(tmp_path, {
        "created_at": "2026-11-24T17:33:04.950692+00:00",
        "provenance": {"source_manifest": f"{build_root}/ragged/manifest.tsv"},
        "subset": {"builder_version": "0.42.0", "created_at": "2026-11-24T17:33:05+00:00"},
    })
    generator.normalise(release, build_root)
    text = (release / "manifest.json").read_text(encoding="utf-8")
    assert "17:33" not in text and "0.42.0" not in text and str(build_root) not in text
    assert text.count(generator.STAMP) == 2 and generator.BUILDER_VERSION in text
    assert generator.STAMP in (release / "overview.html").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "field, value",
    [
        ("built_at", "2026-11-25T09:12:44+00:00"),
        ("finished", "2026-11-24 17:33"),
        ("checkout", "/home/someone/repo/opengwasdb"),
        ("host_scratch", "/tmp/pytest-of-someone/build/sources.tsv"),
    ],
)
def test_normalise_refuses_a_time_or_path_in_a_field_it_does_not_know(
    tmp_path: Path, generator: ModuleType, field: str, value: str
):
    """A builder that adds a stamp field fails generation by name, never a byte diff."""
    release = _built(tmp_path, {"created_at": "2026-11-24T17:33:04+00:00", field: value})
    with pytest.raises(AssertionError, match=re.escape(value)):
        generator.normalise(release, tmp_path / "build")


def _readme_refusals() -> dict[str, set[tuple[str, str]]]:
    """`README.md`'s table of what `ogdb validate` fails: release -> (rule, target)."""
    readme = (BUNDLE / "README.md").read_text(encoding="utf-8")
    section = readme.split("<!-- validate-fails -->")[1].split("<!-- /validate-fails -->")[0]
    rows = re.findall(r"^\| `([a-z_]+)` \| `([a-z_.]+)` \| `([a-z_./]+)` \|", section, flags=re.M)
    found: dict[str, set[tuple[str, str]]] = {}
    for name, rule, target in rows:
        found.setdefault(name, set()).add((rule, target))
    return found


def test_ogdb_validate_fails_exactly_the_releases_the_readme_names(generator: ModuleType):
    named = _readme_refusals()
    assert set(named) == set(generator.LEGACY_STATES), sorted(named)
    assert len(named) == 4
    for name in sorted(p.name for p in RELEASES.iterdir() if p.is_dir()):
        result = validate_store(RELEASES / name)
        failed = {(f.rule, f.target) for f in result.findings if f.severity == "error"}
        assert failed == named.get(name, set()), (name, result.errors)
