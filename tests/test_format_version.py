"""`format_version` semantics and the reader's obligations (issue #143, ADR 0041).

The format was reset to `0.1.0` at the end of Roadmap 1, and the shape of the
version string is the safety mechanism rather than a cosmetic choice: the
pre-release formats were stamped `0.1`, `1.0`, `2.0` and `3.0`, and a reset
that reused `0.1` would give two different formats one name. Three components
cannot be misread as two, so every pre-reset release is refused loudly instead
of being decoded under a contract it was not written against.

The reader's own obligations -- parse the version, refuse a pre-reset release or
an unknown series, warn on a compatible version from the future -- are tested in
ogquery (`tests/unit/test_format_version.py`, #475). What stays here is what
only opengwasdb can run: `open_store`, validation and the converter against a
store its builders made, and the writer and completion rules.

The completion test covers the hole ADR 0041 §4 exists to close: completion
preserves its source's `format_version` because it writes into the source's
arrays, so a source this build cannot write must be refused before any work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengwasdb.completion.ld_panel import PanelSource
from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.store import open as store_open
from opengwasdb.store_reader import format as store_format


def _set_version(store: Path, version: str) -> None:
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


# --- against a real store --------------------------------------------------


def test_open_store_refuses_a_pre_reset_release(dense_store_path):
    """The headline of the reset: a store stamped with a pre-release format is
    never decoded. Its bytes may or may not be readable -- a `3.0` store's are
    identical to a `0.1.0` store's, a `0.1` store's are not -- and this build
    does not try to tell the two apart."""
    _set_version(dense_store_path, "3.0")

    with pytest.raises(store_format.UnsupportedFormatVersion, match="Rebuild the release"):
        store_open.open_store(dense_store_path)


def test_open_store_rejects_an_unknown_series(dense_store_path):
    _set_version(dense_store_path, "1.0.0")

    with pytest.raises(store_format.UnsupportedFormatVersion):
        store_open.open_store(dense_store_path)


def test_validation_reports_a_pre_reset_release_as_an_error_not_a_crash(dense_store_path):
    from opengwasdb.validation import validate_store

    _set_version(dense_store_path, "2.0")
    result = validate_store(dense_store_path)

    assert result.failed() == {"manifest.loads"}
    assert any("2.0" in error and "Rebuild" in error for error in result.errors)


# --- writing is narrower than reading (ADR 0041 §3) ------------------------


def test_the_version_this_build_completes_is_writable():
    """0.2.0 to 0.2.3 are encoded identically, but only a 0.2.2 or 0.2.3 release spells
    its indels as this build's panel does, so those are the ones a completion writes
    into (ADR 0038 §4, ADR 0079); either completes to 0.2.3 (issue #338)."""
    assert store_open.check_writable_format_version("0.2.2") == "0.2.2"
    assert store_open.check_writable_format_version("0.2.3") == "0.2.3"


def test_the_converter_stamps_by_the_builders_rule_and_keeps_sorted_indels():
    """#247 made the converter's target the builders' version, and #430 stamps both by
    one rule (`format_version_for`). #381 parts their output: the converter reads
    `SOURCE_FORMAT_VERSION` (0.1.0, Zarr v2) and copies its stored codes and ALIDs into
    Zarr v3. A 0.1.0 Store sorts an indel's alleles and says nowhere which is the
    reference, so its copy is stamped as sorted, 0.2.0, where a build writes 0.2.2
    (ADR 0079).
    """
    from opengwasdb.store import convert

    assert convert.format_version_for is store_format.format_version_for
    assert convert.SOURCE_FORMAT_VERSION == "0.1.0"


@pytest.mark.parametrize("version", ["0.1.0", "0.2.0", "0.2.1"])
def test_a_store_that_sorts_its_indels_is_readable_but_not_writable(version):
    """This build reads 0.1.0, 0.2.0 and 0.2.1 and completes only 0.2.2.

    Completion writes into the source's arrays and keeps its format, so each
    source is refused. Each sorts an indel's alleles, and a completion would add
    panel ALIDs spelt reference first beside them, so the remedy is a rebuild:
    a conversion keeps the sorted spelling (ADR 0079).
    """
    store_format.check_format_version(version)
    with pytest.raises(store_format.UnsupportedFormatVersion, match="Rebuild the release"):
        store_open.check_writable_format_version(version, source="source release X")


def test_a_readable_but_unwritable_version_is_refused():
    """A newer remainder of a known series is read, with a warning (§21.2), but not
    written: 0.2.4 spells its indels as 0.2.3 does and is still refused, since this
    build does not know what else it adds."""
    store_format.check_format_version("0.2.4")  # still readable, logged as newer
    with pytest.raises(store_format.UnsupportedFormatVersion, match="reads but cannot write"):
        store_open.check_writable_format_version("0.2.4", source="source release X")


@pytest.mark.parametrize("carries_regions", [False, True])
def test_a_newer_stamp_is_not_judged_by_this_builds_rule(carries_regions):
    """A 0.2.4 release is a later package's format: which stamp its features give is
    that package's rule. This build says to upgrade, never that the stamp is wrong,
    since re-stamping it 0.2.3 would let this build complete a format it does not
    know -- as package 0.9.0 tells a 0.2.2 Store it "is 0.2.1"."""
    with pytest.raises(store_format.UnsupportedFormatVersion) as refused:
        store_open.check_format_version_stamp("0.2.4", carries_regions=carries_regions)
    message = str(refused.value)
    assert "Upgrade opengwasdb" in message and "do not re-stamp" in message, message
    assert "but a release carrying" not in message, message


@pytest.mark.parametrize("version", ["0.1.0", "0.2.0", "0.2.1"])
def test_completion_refuses_a_source_that_sorts_its_indels(tmp_path, dense_store_path, version):
    """The guard ADR 0038 §4 added for exactly this moment now fires.

    A 0.1.0, 0.2.0 or 0.2.1 source is readable but not writable by a build that
    writes 0.2.2, and completion preserves its source's format rather than re-encoding
    it. Its deletions are spelt like insertions, so the refusal names a rebuild
    (ADR 0079).
    """
    _set_version(dense_store_path, version)
    out = tmp_path / "completed.opengwasdb"

    with pytest.raises(store_format.UnsupportedFormatVersion, match="sorts an indel's alleles"):
        complete_dense_store(
            dense_store_path,
            out,
            # Deliberately not a usable panel: the version check must fire
            # first, so completion never reads it.
            PanelSource(tmp_path / "no-such-panel"),
        )

    assert not out.exists()


def test_completion_refuses_a_source_it_cannot_write_before_doing_any_work(
    tmp_path, dense_store_path
):
    """Completion preserves its source's format_version because it writes into
    the source's arrays. A pre-reset source cannot be completed at all -- its
    arrays are in a format this build no longer encodes -- so it fails, and
    fails *before* the imputation rather than at manifest-write time an hour
    later.
    """
    _set_version(dense_store_path, "2.0")
    out = tmp_path / "completed.opengwasdb"

    with pytest.raises(store_format.UnsupportedFormatVersion, match="Rebuild the release"):
        complete_dense_store(
            dense_store_path,
            out,
            # Deliberately not a usable panel: the version check must fire
            # first, so completion never gets far enough to read it.
            PanelSource(tmp_path / "no-such-panel"),
        )

    assert not out.exists()


def test_a_writer_reads_the_layout_from_the_series_not_the_current_version():
    """The layout a release is stored in is its series', not `CURRENT_FORMAT_VERSION`'s.

    Adding arrays to a release's own data group needs the format that group is
    in, so the decision is `ZARR_FORMAT_BY_SERIES`' (issue #313). `0.1.x` is Zarr
    v2 and is refused; every `0.2.x` is Zarr v3 and is accepted -- including one
    the constant has not reached yet, because `CURRENT_FORMAT_VERSION` moves
    *within* a series (issue #381 takes it to `0.2.2`).
    """
    assert store_open.check_writable_zarr_layout("0.2.0") == 3
    assert store_open.check_writable_zarr_layout("0.2.1") == 3
    assert store_open.check_writable_zarr_layout("0.2.2") == 3
    with pytest.raises(store_format.UnsupportedFormatVersion, match="convert_store_to_0_2_0"):
        store_open.check_writable_zarr_layout("0.1.0", source="release at /tmp/x")
    with pytest.raises(store_format.UnsupportedFormatVersion, match="Zarr format 2"):
        store_open.check_writable_zarr_layout("0.1.7")


#: The version tables as they were before #467: the package that does not know 0.3.
_BEFORE_0_3 = {(0, 1): (0,), (0, 2): (3,)}

