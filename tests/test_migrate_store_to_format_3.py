"""Tests for scripts/migrate_store_to_format_3.py (issue #156).

The migration derives a **new release**: the format-3 re-encode rewrites the
`se` plane, the top-hit index and the manifest, which is association data and
therefore outside the Provenance Amendment exception (spec §21.4) — a Store
Release is immutable and a migration must never write into the release it was
given. These tests pin the copy-on-write contract the interim review of
issue-118 asked for: `--into` is required, the destination is published from a
staging directory only when the migrated copy validates, and a failure at any
point leaves the source exactly as it was and nothing at the destination.

There is no format-2.0 builder left in the codebase, so the fixture builds a
Dense release the current way — with data a format-3 build residual-codes —
and hand-reverts it to the 2.0 layout the migration targets: `se` written back
as `float16` (the only `se` 2.0 admits), its side arrays dropped, the top-hit
index rebuilt from that plane, and the manifest stamped 2.0.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from residual_fixtures import residual_eligible_records

from opengwasdb.build.source import NormalisedAssociation
from opengwasdb.encoding.planes import DenseSePlane
from opengwasdb.layouts.dense.build import build_dense_observed_store
from opengwasdb.layouts.dense.top_hits import build_top_hit_indexes
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.validation import ValidationResult, validate_store

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "migrate_store_to_format_3.py"
_spec = importlib.util.spec_from_file_location("migrate_store_to_format_3", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
migrate_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate_module)


def _residual_eligible_source() -> list[NormalisedAssociation]:
    """Data a format-3 build residual-codes, and therefore data a
    format-2.0-era build would have had to store as `float16` without being
    able to express it as a residual. Shared with the encoding suite, which
    needs the same eligible source (`conftest.residual_eligible_records`)."""
    records, _ = residual_eligible_records()
    return records


def _build_format_3_store(path: Path) -> Path:
    """A current (format-3.0) Dense release whose `se` is residual-coded."""
    build_dense_observed_store(
        _residual_eligible_source(),
        path,
        store_id="s",
        release_id="r",
        reference_assembly="GRCh38",
        chunk_shape=(100, 2),
    )
    assert StoreManifest.load(path).encoding.se.is_residual
    return path


def _revert_to_format_2(store: Path) -> Path:
    """Turn a format-3.0 release into the 2.0 release the migration targets:
    `se` back to `float16`, side arrays gone, top-hit index rebuilt from that
    plane, manifest stamped 2.0.

    The fixture is only meaningful if the format-3 build really residual-coded
    the plane: otherwise the migration has nothing to do and every assertion
    below is vacuous.
    """
    root = zarr.open_group(str(store / "data.zarr"), mode="a")
    encoding = StoreManifest.load(store).encoding
    decoded = DenseSePlane.open(root, encoding).band(0, int(root["se"].shape[0]))
    chunks = root["se"].chunks
    del root["se"]
    root.create_dataset("se", data=decoded.astype(np.float16), chunks=chunks, dtype="float16")
    for name in ("se_coefficients", "se_exception_index", "se_exception_value"):
        if name in root:
            del root[name]

    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = "2.0"
    manifest["encoding"]["version"] = 2
    manifest["encoding"]["se"] = {"kind": "float16"}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # A 2.0-era build wrote its top-hit index from its own float16 plane; the
    # one this store carries was written from the residual plane's decode.
    build_top_hit_indexes(store, encoding=StoreManifest.load(store).encoding)
    result = validate_store(store)
    assert result.ok, result.errors
    return store


def _directory_fingerprint(path: Path) -> dict[str, bytes]:
    """Content of every file under ``path``, keyed by relative name."""
    return {
        str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()
    }


def test_migrate_derives_a_3_0_release_and_leaves_the_source_untouched(tmp_path: Path) -> None:
    source = _revert_to_format_2(_build_format_3_store(tmp_path / "two.opengwasdb"))
    before = _directory_fingerprint(source)
    destination = tmp_path / "three.opengwasdb"

    assert migrate_module.migrate(source, destination) == 0

    # The source is byte-for-byte what it was: the migration read it and never
    # wrote into it (spec §21.4).
    assert _directory_fingerprint(source) == before
    # The destination is a separate release, published only once the migrated
    # copy validated; the staging directory is gone.
    assert destination.exists()
    assert not (tmp_path / ".three.opengwasdb.tmp").exists()
    manifest = StoreManifest.load(destination)
    assert manifest.format_version == "3.0"
    # ... and the migration actually re-encoded the plane it exists to re-encode
    # (this is the same data a 2.0-era build had to store as float16).
    assert manifest.encoding.se.is_residual
    assert validate_store(destination).ok


def test_a_migration_needs_a_destination(tmp_path: Path, capsys) -> None:
    """No in-place mode: with no `--into`, the tool refuses rather than
    mutating the release it was given (#156)."""
    source = _revert_to_format_2(_build_format_3_store(tmp_path / "two.opengwasdb"))
    with pytest.raises(SystemExit):
        migrate_module.main([str(source)])
    assert "--into" in capsys.readouterr().err
    # Still 2.0: nothing ran against it.
    assert StoreManifest.load(source).format_version == "2.0"


def test_a_failed_migration_publishes_nothing_and_leaves_the_source_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The destination is built in a staging directory and published by rename;
    a failure mid-migration removes the staging directory and leaves neither a
    half-migrated destination nor a touched source."""
    source = _revert_to_format_2(_build_format_3_store(tmp_path / "two.opengwasdb"))
    before = _directory_fingerprint(source)
    destination = tmp_path / "three.opengwasdb"

    def boom(*args, **kwargs):
        raise RuntimeError("simulated mid-migration failure")

    monkeypatch.setattr(migrate_module, "optimise_dense_se_joint", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        migrate_module.migrate(source, destination)

    assert _directory_fingerprint(source) == before
    assert not destination.exists()
    assert not (tmp_path / ".three.opengwasdb.tmp").exists()


def test_a_migrated_store_that_fails_validation_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staging directory is published only when the migrated copy
    validates. A copy the migration made invalid is refused -- `SystemExit`,
    which the staging context manager leaves in place for inspection -- so
    neither the source nor the destination is ever a half-migrated store.
    """
    source = _revert_to_format_2(_build_format_3_store(tmp_path / "two.opengwasdb"))
    before = _directory_fingerprint(source)
    destination = tmp_path / "three.opengwasdb"
    real_validate = migrate_module.validate_store

    def validate_staged_only(path):
        # The before-migration check runs against the source and passes; the
        # after-migration check runs against the staged copy and fails, as it
        # would if the re-encode had produced an invalid store.
        if Path(path).resolve() == source.resolve():
            return real_validate(path)
        return ValidationResult(errors=["the migration broke something"])

    monkeypatch.setattr(migrate_module, "validate_store", validate_staged_only)
    with pytest.raises(SystemExit, match="introduced 1 error"):
        migrate_module.migrate(source, destination)

    assert _directory_fingerprint(source) == before
    assert not destination.exists()
    assert (tmp_path / ".three.opengwasdb.tmp").exists()  # left for inspection


def test_a_non_migratable_source_is_refused_before_any_copy(tmp_path: Path) -> None:
    """Refusal happens against the source before anything is copied or staged:
    a release that is not migratable leaves neither a destination nor a
    staging directory behind."""
    source = _build_format_3_store(tmp_path / "three.opengwasdb")  # already 3.0
    destination = tmp_path / "waste.opengwasdb"

    with pytest.raises(SystemExit, match="already migrated"):
        migrate_module.migrate(source, destination)

    assert not destination.exists()
    assert not (tmp_path / ".waste.opengwasdb.tmp").exists()


def test_migrate_refuses_when_source_and_destination_are_the_same(tmp_path: Path) -> None:
    source = _revert_to_format_2(_build_format_3_store(tmp_path / "two.opengwasdb"))
    before = _directory_fingerprint(source)
    with pytest.raises(SystemExit, match="same"):
        migrate_module.migrate(source, source)
    assert _directory_fingerprint(source) == before


def test_migrate_refuses_an_existing_destination(tmp_path: Path) -> None:
    source = _revert_to_format_2(_build_format_3_store(tmp_path / "two.opengwasdb"))
    destination = tmp_path / "three.opengwasdb"
    destination.mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        migrate_module.migrate(source, destination)
    # The refusal came before any copy; the pre-existing destination is intact.
    assert destination.is_dir()
    assert not (tmp_path / ".three.opengwasdb.tmp").exists()


def test_main_migrates_into_an_explicit_destination(tmp_path: Path) -> None:
    source = _revert_to_format_2(_build_format_3_store(tmp_path / "two.opengwasdb"))
    destination = tmp_path / "three.opengwasdb"
    assert migrate_module.main([str(source), "--into", str(destination)]) == 0
    assert StoreManifest.load(destination).format_version == "3.0"
    assert StoreManifest.load(source).format_version == "2.0"
