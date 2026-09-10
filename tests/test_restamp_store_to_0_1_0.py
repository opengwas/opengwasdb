"""Tests for scripts/restamp_store_to_0_1_0.py (issue #143).

The format reset renumbered the store format and deleted every pre-release
decoder. It did not change the bytes a build writes, so a `3.0` release already
holds exactly what `0.1.0` describes -- which is what lets `ukb-b` (13h30m to
rebuild, issue #148) be restamped instead of rebuilt. `0.1`, `1.0` and `2.0` are
genuinely different encodings, and the tool refuses them.

The restamp derives a **new release**: a Store Release is immutable, so the
source is never written, the destination is published from a staging directory
only once the restamped copy validates as a 0.1.0 release, and a failure at any
point leaves the source exactly as it was and nothing at the destination
(spec §21.4, issues #156, #164).

The fixture builds a release the current way and stamps it *back* to 3.0. That
inverse is only a legitimate way to make a pre-reset store because the stamp
really is the whole difference -- the claim the tool rests on -- so each test
first establishes that the store it starts from is one this build refuses.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from opengwasdb.store.open import UnsupportedFormatVersion, open_store
from opengwasdb.validation import validate_store

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "restamp_store_to_0_1_0.py"
_spec = importlib.util.spec_from_file_location("restamp_store_to_0_1_0", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
restamp_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(restamp_module)


def _stamp(store: Path, version: str, *, encoding_version: int | None = None) -> None:
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = version
    if encoding_version is not None:
        manifest["encoding"]["version"] = encoding_version
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@pytest.fixture
def pre_reset_store(dense_store_path: Path) -> Path:
    """A release stamped `3.0`: what every store on disk is, until restamped.

    Meaningful only if it is genuinely unreadable to this build, which is
    asserted here rather than assumed -- a fixture this build could open would
    make every assertion below vacuous.
    """
    assert validate_store(dense_store_path).ok
    _stamp(dense_store_path, "3.0")
    with pytest.raises(UnsupportedFormatVersion):
        open_store(dense_store_path)
    return dense_store_path


def _fingerprint(path: Path, *, exclude: tuple[str, ...] = ()) -> dict[str, bytes]:
    """Content of every file under ``path``, keyed by relative name."""
    return {
        str(file.relative_to(path)): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file() and file.name not in exclude
    }


def test_restamp_publishes_a_0_1_0_release_and_leaves_the_source_untouched(
    pre_reset_store: Path, tmp_path: Path
) -> None:
    before = _fingerprint(pre_reset_store)
    destination = tmp_path / "restamped.opengwasdb"

    assert restamp_module.restamp(pre_reset_store, destination) == 0

    # The source is byte-for-byte what it was: the tool read it and never wrote
    # into it (spec §21.4).
    assert _fingerprint(pre_reset_store) == before
    # Published only once the restamped copy validated; the staging directory
    # is gone.
    assert destination.exists()
    assert not (tmp_path / ".restamped.opengwasdb.tmp").exists()

    manifest = open_store(destination).manifest
    assert manifest.format_version == "0.1.0"
    assert validate_store(destination).ok


def test_the_stamp_is_the_only_thing_that_changes(pre_reset_store: Path, tmp_path: Path) -> None:
    """The claim the whole tool rests on. Every byte outside `manifest.json`
    and the release's own page is identical to the source's: no array, index or
    table is read, let alone rewritten."""
    destination = tmp_path / "restamped.opengwasdb"
    excluded = ("manifest.json", "overview.html")
    before = _fingerprint(pre_reset_store, exclude=excluded)
    assert before, "fixture must have files other than the manifest for this to mean anything"

    restamp_module.restamp(pre_reset_store, destination)

    assert _fingerprint(destination, exclude=excluded) == before


def test_the_restamped_release_has_its_own_identity(
    pre_reset_store: Path, tmp_path: Path
) -> None:
    """A derived release is a new release (issue #164): two releases of one
    store that cannot be told apart is the failure this closes."""
    source_manifest = json.loads((pre_reset_store / "manifest.json").read_text())
    destination = tmp_path / "restamped.opengwasdb"

    restamp_module.restamp(pre_reset_store, destination)

    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["release_id"] != source_manifest["release_id"]
    uuid.UUID(manifest["release_id"])  # a real UUID4, not a decorated copy
    assert manifest["created_at"] != source_manifest["created_at"]
    datetime.fromisoformat(manifest["created_at"])

    restamp = manifest["provenance"]["format_restamp"]
    assert restamp["from"] == "3.0"
    assert restamp["to"] == "0.1.0"
    assert restamp["source_release_id"] == source_manifest["release_id"]


def test_the_new_release_id_reaches_the_page_humans_browse(
    pre_reset_store: Path, tmp_path: Path
) -> None:
    """`overview.html` embeds the release identity (ADR 0032), and the copy
    arrived carrying the source's."""
    destination = tmp_path / "restamped.opengwasdb"

    restamp_module.restamp(pre_reset_store, destination)

    manifest = json.loads((destination / "manifest.json").read_text())
    overview = (destination / "overview.html").read_text(encoding="utf-8")
    assert manifest["release_id"] in overview


@pytest.mark.parametrize("version", ["0.1", "1.0", "2.0", "0.1.0"])
def test_only_a_3_0_release_is_restampable(
    dense_store_path: Path, tmp_path: Path, version: str
) -> None:
    """Every other version is rebuilt, not restamped: `0.1`, `1.0` and `2.0`
    hold different bytes, and `0.1.0` is already current. Refused against the
    source, before anything is copied."""
    _stamp(dense_store_path, version)
    destination = tmp_path / "restamped.opengwasdb"

    with pytest.raises(SystemExit, match="rebuilt"):
        restamp_module.restamp(dense_store_path, destination)

    assert not destination.exists()


def test_a_release_whose_encoding_block_is_not_this_builds_is_refused(
    dense_store_path: Path, tmp_path: Path
) -> None:
    """The stamp says 3.0 and the block says its planes are in an older shape.
    The restamp asserts the planes are already what this build writes, so it
    checks that claim rather than taking the version's word for it."""
    _stamp(dense_store_path, "3.0", encoding_version=2)
    destination = tmp_path / "restamped.opengwasdb"

    with pytest.raises(SystemExit, match="encoding block version"):
        restamp_module.restamp(dense_store_path, destination)

    assert not destination.exists()


def test_a_nested_component_is_restamped_too(tmp_path: Path) -> None:
    """A Hybrid release's Dense Component is a nested Store Release with its
    own manifest. A half-restamped Hybrid store is one this build opens at the
    top and refuses one directory down."""
    store = tmp_path / "hybrid.opengwasdb"
    (store / "dense").mkdir(parents=True)
    for path in (store / "manifest.json", store / "dense" / "manifest.json"):
        path.write_text(json.dumps({"release_id": "v1", "format_version": "3.0"}))

    assert restamp_module._manifest_paths(store) == [
        store / "manifest.json",
        store / "dense" / "manifest.json",
    ]


def test_a_nested_component_in_another_format_is_refused(tmp_path: Path) -> None:
    store = tmp_path / "hybrid.opengwasdb"
    (store / "dense").mkdir(parents=True)
    encoding = {"version": 3, "z": {}, "se": {}, "eaf": {}}
    (store / "manifest.json").write_text(
        json.dumps({"release_id": "v1", "format_version": "3.0", "encoding": encoding})
    )
    (store / "dense" / "manifest.json").write_text(
        json.dumps({"release_id": "v1", "format_version": "2.0", "encoding": encoding})
    )

    with pytest.raises(SystemExit, match="dense/manifest.json"):
        restamp_module._refuse_unless_restampable(store)


def test_a_restamped_store_that_fails_validation_is_not_published(
    pre_reset_store: Path, tmp_path: Path
) -> None:
    """The gate that makes the restamp sound rather than asserted: the staged
    copy is validated as a 0.1.0 release, and one that needed more than a stamp
    is discarded rather than published."""
    shutil.rmtree(pre_reset_store / "data.zarr" / "se")
    before = _fingerprint(pre_reset_store)
    destination = tmp_path / "restamped.opengwasdb"

    with pytest.raises(restamp_module.RestampError, match="does not validate is never published"):
        restamp_module.restamp(pre_reset_store, destination)

    assert not destination.exists()
    assert not (tmp_path / ".restamped.opengwasdb.tmp").exists()
    assert _fingerprint(pre_reset_store) == before


def test_restamp_refuses_when_source_and_destination_are_the_same(
    pre_reset_store: Path,
) -> None:
    with pytest.raises(SystemExit, match="same path"):
        restamp_module.restamp(pre_reset_store, pre_reset_store)


def test_restamp_refuses_an_existing_destination(pre_reset_store: Path, tmp_path: Path) -> None:
    destination = tmp_path / "already-there"
    destination.mkdir()

    with pytest.raises(SystemExit, match="already exists"):
        restamp_module.restamp(pre_reset_store, destination)


def test_main_restamps_into_an_explicit_destination(
    pre_reset_store: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "restamped.opengwasdb"

    assert restamp_module.main([str(pre_reset_store), "--into", str(destination)]) == 0

    assert open_store(destination).manifest.format_version == "0.1.0"


def test_main_requires_a_destination(pre_reset_store: Path) -> None:
    """`--into` is required: the source release is immutable."""
    with pytest.raises(SystemExit) as excinfo:
        restamp_module.main([str(pre_reset_store)])
    assert excinfo.value.code == 2
