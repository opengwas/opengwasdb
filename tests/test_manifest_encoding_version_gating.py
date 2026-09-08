"""The `encoding` block a manifest declares must be one its own
`format_version` admits (issue #157).

A Store Release's `format_version` is a promise about what a conforming reader
of that version knows. `int8_residual` `se` did not exist before format 3.0,
`int8_residual` `eaf` before 2.0, and the `encoding` block itself before 1.0 —
yet `StoreManifest` parsed whatever the block said without checking it against
the version, so a release stamped 1.0 or 2.0 could declare the format-3
representation and this package would decode it. The damage is to every
*other* reader: a conforming format-2.0 reader is entitled to read `se` as
`float16`, and handed a 2.0 manifest declaring a residual plane it would read
`int8` codes as `float16` and return plausible, wrong standard errors — the
exact failure the explicit `encoding` block exists to prevent.

These tests pin the version checks the parser applies in both directions:
a block where the format requires one (the existing rule, spec §6a), a block
that predates the format, and a kind the format predates.
"""

from __future__ import annotations

import json

import pytest

from opengwasdb.encoding import UnsupportedEncoding
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store import open_store


def _manifest(format_version: str, encoding: dict | None) -> dict:
    manifest = {
        "store_id": "example",
        "release_id": "observed-1",
        "format_version": format_version,
        "primary_layout": "dense",
        "association_coverage": "full",
        "completion_state": "observed_only",
        "reference_assembly": "GRCh37",
    }
    if encoding is not None:
        manifest["encoding"] = encoding
    return manifest


def _block(version: int, se: dict | None = None, eaf: dict | None = None) -> dict:
    """A well-formed encoding block for a format era, with `se`/`eaf`
    overridden. A writer emits `residual_range` beside a residual kind; these
    fixtures do the same so an unrelated parse failure cannot masquerade as a
    version one. `eaf` appears only from block version 2 (format 2.0), exactly
    as the era's manifests were written -- a caller asserting a version-gate
    on `eaf` passes version 2 explicitly."""
    block: dict = {
        "version": version,
        "z": {"kind": "int16_fixed", "scale": 1024},
        "se": se or {"kind": "float16"},
    }
    if version >= 2:
        block["eaf"] = eaf or {"kind": "absent"}
    return block


def test_a_2_0_manifest_cannot_declare_residual_se() -> None:
    """#157's own case: a 2.0 reader is entitled to `float16` `se`, and handed
    a manifest declaring the format-3 residual plane it would read `int8`
    codes as `float16`."""
    manifest = _manifest("2.0", _block(2, se={"kind": "int8_residual", "residual_range": 1.0}))
    with pytest.raises(UnsupportedEncoding) as exc:
        StoreManifest.from_dict(manifest)
    assert "2.0" in str(exc.value)
    assert "int8_residual" in str(exc.value)


def test_a_1_0_manifest_cannot_declare_residual_se() -> None:
    manifest = _manifest("1.0", _block(1, se={"kind": "int8_residual", "residual_range": 1.0}))
    with pytest.raises(UnsupportedEncoding) as exc:
        StoreManifest.from_dict(manifest)
    assert "1.0" in str(exc.value)
    assert "int8_residual" in str(exc.value)


def test_a_3_0_manifest_declaring_residual_se_still_opens() -> None:
    manifest = _manifest("3.0", _block(3, se={"kind": "int8_residual", "residual_range": 1.0}))
    loaded = StoreManifest.from_dict(manifest)
    assert loaded.format_version == "3.0"
    assert loaded.encoding.se.is_residual
    assert loaded.encoding.se.residual_range == 1.0


def test_the_check_covers_the_other_version_gated_kinds() -> None:
    """Not a special case for `se`: each kind records the format that first
    admitted it, so the next encoding to land adds a row rather than a branch.

    Residual `eaf` arrived at format 2.0, so a 1.0 release declaring it is the
    same failure one major down: a conforming 1.0 reader reads ADR 0036's
    `float32` plane and would read the `int8` residual plane as `float32`.
    """
    # A block version 2 (which must declare `eaf`) stamped on a 1.0 release:
    # the kind is format-2.0's and the version says 1.0.
    manifest = _manifest("1.0", _block(2, eaf={"kind": "int8_residual", "residual_range": 1.0}))
    with pytest.raises(UnsupportedEncoding) as exc:
        StoreManifest.from_dict(manifest)
    assert "1.0" in str(exc.value)
    assert "int8_residual" in str(exc.value)
    assert "eaf" in str(exc.value)


def test_a_release_below_1_0_cannot_declare_an_encoding_block() -> None:
    """The block itself is version-gated: below format 1.0 no release ever
    declared one — `float16` throughout *is* what the absence of a block means
    (spec §6a) — so a 0.1 manifest carrying a block claims an encoding its
    format cannot carry, the same mismatch the parser already refuses in the
    other direction (a 1.0 release with no block at all)."""
    manifest = _manifest("0.1", _block(1))
    with pytest.raises(UnsupportedEncoding) as exc:
        StoreManifest.from_dict(manifest)
    assert "0.1" in str(exc.value)


def test_a_manifest_without_a_block_is_unchanged() -> None:
    """The legacy path still works: a release that declares nothing is the
    `float16`-throughout plan, whatever its version."""
    legacy = StoreManifest.from_dict(_manifest("0.1", None))
    assert legacy.encoding.is_legacy
    assert legacy.encoding.se.kind == "float16"


def _stamp_residual_se_onto_a_2_0_store(dense_store_path) -> None:
    """Turn a built 3.0 release into the manifest #157 is about: stamped 2.0
    while declaring the format-3 residual `se`. The arrays are irrelevant --
    the refusal is at open, before anything reads them."""
    manifest_path = dense_store_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = "2.0"
    manifest["encoding"]["version"] = 2
    manifest["encoding"]["se"] = {"kind": "int8_residual", "residual_range": 1.0}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_open_store_refuses_a_2_0_release_declaring_residual_se(dense_store_path) -> None:
    """The refusal stops a read, not merely a parse of a dict: opening the
    release raises before any array is touched."""
    _stamp_residual_se_onto_a_2_0_store(dense_store_path)
    with pytest.raises(UnsupportedEncoding, match="int8_residual"):
        open_store(dense_store_path)


def test_validation_reports_a_2_0_release_declaring_residual_se(
    dense_store_path,
) -> None:
    _stamp_residual_se_onto_a_2_0_store(dense_store_path)
    from opengwasdb.validation import validate_store

    result = validate_store(dense_store_path)
    assert not result.ok
    assert any("int8_residual" in error for error in result.errors), result.errors
