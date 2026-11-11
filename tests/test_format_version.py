"""`format_version` semantics and the reader's obligations (issue #143, ADR 0041).

The format was reset to `0.1.0` at the end of Roadmap 1, and the shape of the
version string is the safety mechanism rather than a cosmetic choice: the
pre-release formats were stamped `0.1`, `1.0`, `2.0` and `3.0`, and a reset
that reused `0.1` would give two different formats one name. Three components
cannot be misread as two, so every pre-reset release is refused loudly instead
of being decoded under a contract it was not written against.

These tests pin what a reader owes a store it did not write: refuse a
pre-reset release with an instruction rather than a decode attempt, refuse an
unknown release series, read a known one, and say something when it meets a
compatible version from the future rather than quietly returning a subset of
the data.

The completion test covers the hole ADR 0041 §4 exists to close: completion
preserves its source's `format_version` because it writes into the source's
arrays, so a source this build cannot write must be refused before any work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.store import open as store_open
from opengwasdb.store.open import CURRENT_FORMAT_VERSION, PRE_RESET_FORMAT_VERSIONS


def _set_version(store: Path, version: str) -> None:
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


# --- parsing ---------------------------------------------------------------


def test_version_parses_as_major_minor_patch():
    assert store_open.parse_format_version("0.1.0") == (0, 1, 0)
    assert store_open.parse_format_version("12.34.56") == (12, 34, 56)


@pytest.mark.parametrize("version", sorted(PRE_RESET_FORMAT_VERSIONS))
def test_a_pre_reset_version_is_refused_with_an_instruction(version):
    """`0.1`, `1.0`, `2.0` and `3.0` are the formats this project burned before
    it published anything. None is readable, and the message says what to do
    about it rather than complaining about punctuation: those stores are
    rebuilt, not migrated."""
    with pytest.raises(store_open.UnsupportedFormatVersion, match="Rebuild the release"):
        store_open.parse_format_version(version)


def test_the_pre_reset_versions_are_the_two_component_ones():
    """The reset is safe because no pre-reset stamp can be read as a current
    one. That holds only while every pre-reset version really is two-component
    -- a three-component one would collide with the new numbering."""
    for version in PRE_RESET_FORMAT_VERSIONS:
        assert len(version.split(".")) == 2, version
    assert len(CURRENT_FORMAT_VERSION.split(".")) == 3


@pytest.mark.parametrize(
    "version", ["", "1", "1.2", "1.", ".1", "0.1.0.0", "banana", "0.1.x", "v0.1.0"]
)
def test_a_version_that_is_not_major_minor_patch_is_refused(version):
    """Refused, not coerced. To a caller deciding whether it can read a
    release, an unparseable version and a future one are the same answer."""
    with pytest.raises(store_open.MalformedFormatVersion):
        store_open.parse_format_version(version)
    # And it stops a read, not merely a parse.
    assert issubclass(store_open.MalformedFormatVersion, store_open.UnsupportedFormatVersion)


# --- which component carries an incompatible change ------------------------


def test_the_leftmost_non_zero_component_is_the_breaking_one():
    """Semantic versioning's own rule (ADR 0041 §1). Below 1.0.0 a format
    declares itself unsettled, so MINOR carries a breaking change; from 1.0.0
    it is MAJOR, and MINOR joins the compatible remainder."""
    assert store_open.split_format_version("0.1.0") == ((0, 1), (0,))
    assert store_open.split_format_version("0.1.7") == ((0, 1), (7,))
    assert store_open.split_format_version("0.2.0") == ((0, 2), (0,))
    assert store_open.split_format_version("1.4.2") == ((1,), (4, 2))
    assert store_open.split_format_version("1.0.0") == ((1,), (0, 0))


# --- the reader contract ---------------------------------------------------


def test_the_version_this_build_writes_is_one_it_can_read():
    """Otherwise the build could not open what it had just produced."""
    store_open.check_format_version(CURRENT_FORMAT_VERSION)
    series, remainder = store_open.split_format_version(CURRENT_FORMAT_VERSION)
    assert store_open.SUPPORTED_FORMAT_VERSIONS[series] >= remainder


def test_the_readable_formats_are_the_two_the_epic_carries():
    """Two series: 0.1 (Zarr v2) and 0.2 (Zarr v3 with sharding, since #245). The 0.2
    series reads to 0.2.2: 0.2.1 adds a Ragged release's optional `regions.tsv` (#430),
    and 0.2.2 spells an indel's ALID reference first (#381, ADR 0079). Another entry
    is a decision, not an accident."""
    assert dict(store_open.SUPPORTED_FORMAT_VERSIONS) == {(0, 1): (0,), (0, 2): (2,)}


def test_a_release_is_stamped_the_lowest_version_whose_features_it_uses():
    """The owner's decision of 2026-11-11 (spec §21.3, ADR 0038 §3): 0.2.1 differs from
    0.2.0 only by an optional, manifest-declared `regions.tsv`, and 0.2.2 from 0.2.1
    only by an indel's ALID spelt reference first, the region table still optional. A
    release is stamped the lowest of the three whose features it uses."""
    stamp = store_open.format_version_for
    assert stamp(carries_regions=False, reference_first_indels=False) == "0.2.0"
    assert stamp(carries_regions=True, reference_first_indels=False) == "0.2.1"
    assert stamp(carries_regions=False, reference_first_indels=True) == "0.2.2"
    assert stamp(carries_regions=True, reference_first_indels=True) == "0.2.2"
    assert store_open.WRITTEN_FORMAT_VERSIONS == ("0.2.0", "0.2.1", "0.2.2")


def test_the_current_version_is_the_stamp_of_a_release_using_no_optional_feature():
    """What every build of this package writes for a release without a region table,
    since every build spells an indel reference first, and what opengwasdb-stores'
    engine update probes (`ogstores.engine_update`) and compares with its
    `REQUIRED_FORMAT_VERSION`."""
    assert CURRENT_FORMAT_VERSION == store_open.format_version_for(
        carries_regions=False, reference_first_indels=True
    )
    assert CURRENT_FORMAT_VERSION == "0.2.2"


def test_each_readable_series_has_one_zarr_on_disk_format():
    """The version-to-layout rule lives beside the version tables, so a caller
    does not pattern-match a version string to guess it (#264 review)."""
    assert store_open.zarr_format_for_version("0.1.0") == 2
    assert store_open.zarr_format_for_version(CURRENT_FORMAT_VERSION) == 3
    assert dict(store_open.ZARR_FORMAT_BY_SERIES) == {(0, 1): 2, (0, 2): 3}


def test_an_unreadable_version_has_no_zarr_layout():
    with pytest.raises(store_open.UnsupportedFormatVersion):
        store_open.zarr_format_for_version("9.9.9")


def test_an_unknown_series_is_rejected():
    """A breaking change moves the series, and this build reads two of them."""
    for version in ("0.3.0", "1.0.0", "9.9.9"):
        with pytest.raises(store_open.UnsupportedFormatVersion, match="release series"):
            store_open.check_format_version(version)


def test_a_newer_compatible_version_is_read_but_warned_about(caplog):
    """Reading it is the definition of a compatible change; the warning is how
    one misclassified as compatible becomes visible instead of silently losing
    data."""
    with caplog.at_level("WARNING"):
        store_open.check_format_version("0.1.7")

    assert "0.1.7" in caplog.text
    assert "not visible here" in caplog.text


# --- against a real store --------------------------------------------------


def test_open_store_refuses_a_pre_reset_release(dense_store_path):
    """The headline of the reset: a store stamped with a pre-release format is
    never decoded. Its bytes may or may not be readable -- a `3.0` store's are
    identical to a `0.1.0` store's, a `0.1` store's are not -- and this build
    does not try to tell the two apart."""
    _set_version(dense_store_path, "3.0")

    with pytest.raises(store_open.UnsupportedFormatVersion, match="Rebuild the release"):
        store_open.open_store(dense_store_path)


def test_open_store_rejects_an_unknown_series(dense_store_path):
    _set_version(dense_store_path, "1.0.0")

    with pytest.raises(store_open.UnsupportedFormatVersion):
        store_open.open_store(dense_store_path)


def test_validation_reports_a_pre_reset_release_as_an_error_not_a_crash(dense_store_path):
    from opengwasdb.validation import validate_store

    _set_version(dense_store_path, "2.0")
    result = validate_store(dense_store_path)

    assert result.failed() == {"manifest.loads"}
    assert any("2.0" in error and "Rebuild" in error for error in result.errors)


# --- writing is narrower than reading (ADR 0041 §3) ------------------------


def test_the_version_this_build_completes_is_writable():
    """0.2.0, 0.2.1 and 0.2.2 are encoded identically, but only a 0.2.2 release spells
    its indels as this build's panel does, so it is the one a completion writes into, in
    its own format (ADR 0038 §4, ADR 0079)."""
    assert store_open.check_writable_format_version("0.2.2") == "0.2.2"


def test_the_converter_stamps_by_the_builders_rule_and_keeps_sorted_indels():
    """#247 made the converter's target the builders' version, and #430 stamps both by
    one rule (`format_version_for`). #381 parts their output: the converter reads
    `SOURCE_FORMAT_VERSION` (0.1.0, Zarr v2) and copies its stored codes and ALIDs into
    Zarr v3. A 0.1.0 Store sorts an indel's alleles and says nowhere which is the
    reference, so its copy is stamped as sorted, 0.2.0, where a build writes 0.2.2
    (ADR 0079).
    """
    from opengwasdb.store import convert

    assert convert.format_version_for is store_open.format_version_for
    assert convert.SOURCE_FORMAT_VERSION == "0.1.0"


@pytest.mark.parametrize(
    ("version", "reference_first"),
    [("0.1.0", False), ("0.2.0", False), ("0.2.1", False), ("0.2.2", True), ("0.2.3", True)],
)
def test_a_store_spells_its_indels_reference_first_from_0_2_2(version, reference_first):
    """0.2.1 is package 0.9.0's region table, its indels still sorted; #381's reference-
    first spelling is 0.2.2 (ADR 0079)."""
    assert store_open.indels_reference_first(version) is reference_first


@pytest.mark.parametrize("version", ["0.1.0", "0.2.0", "0.2.1"])
def test_a_store_that_sorts_its_indels_is_readable_but_not_writable(version):
    """This build reads 0.1.0, 0.2.0 and 0.2.1 and completes only 0.2.2.

    Completion writes into the source's arrays and keeps its format, so each
    source is refused. Each sorts an indel's alleles, and a completion would add
    panel ALIDs spelt reference first beside them, so the remedy is a rebuild:
    a conversion keeps the sorted spelling (ADR 0079).
    """
    store_open.check_format_version(version)
    with pytest.raises(store_open.UnsupportedFormatVersion, match="Rebuild the release"):
        store_open.check_writable_format_version(version, source="source release X")


def test_a_readable_but_unwritable_version_is_refused():
    """A newer remainder of a known series is read, with a warning (§21.2), but not
    written: 0.2.3 spells its indels as 0.2.2 does and is still refused, since this
    build does not know what else it adds."""
    store_open.check_format_version("0.2.3")  # still readable, logged as newer
    with pytest.raises(store_open.UnsupportedFormatVersion, match="reads but cannot write"):
        store_open.check_writable_format_version("0.2.3", source="source release X")


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

    with pytest.raises(store_open.UnsupportedFormatVersion, match="sorts an indel's alleles"):
        complete_dense_store(
            dense_store_path,
            out,
            # Deliberately not a usable panel: the version check must fire
            # first, so completion never reads it.
            ld_dir=tmp_path / "no-such-panel",
            ancestry="EUR",
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

    with pytest.raises(store_open.UnsupportedFormatVersion, match="Rebuild the release"):
        complete_dense_store(
            dense_store_path,
            out,
            # Deliberately not a usable panel: the version check must fire
            # first, so completion never gets far enough to read it.
            ld_dir=tmp_path / "no-such-panel",
            ancestry="EUR",
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
    with pytest.raises(store_open.UnsupportedFormatVersion, match="convert_store_to_0_2_0"):
        store_open.check_writable_zarr_layout("0.1.0", source="release at /tmp/x")
    with pytest.raises(store_open.UnsupportedFormatVersion, match="Zarr format 2"):
        store_open.check_writable_zarr_layout("0.1.7")
