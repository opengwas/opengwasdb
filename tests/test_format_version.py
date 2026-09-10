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


def test_there_is_exactly_one_readable_format():
    """The point of the reset (issue #143): one format, one decoder, one
    contract to test. A second entry here is a decision, not an accident."""
    assert dict(store_open.SUPPORTED_FORMAT_VERSIONS) == {(0, 1): (0,)}


def test_an_unknown_series_is_rejected():
    """A breaking change moves the series, and this build reads one series."""
    for version in ("0.2.0", "1.0.0", "9.9.9"):
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
    _set_version(dense_store_path, "0.2.0")

    with pytest.raises(store_open.UnsupportedFormatVersion):
        store_open.open_store(dense_store_path)


def test_validation_reports_a_pre_reset_release_as_an_error_not_a_crash(dense_store_path):
    from opengwasdb.validation import validate_store

    _set_version(dense_store_path, "2.0")
    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("2.0" in error and "Rebuild" in error for error in result.errors)


# --- writing is narrower than reading (ADR 0041 §3) ------------------------


def test_the_current_version_is_writable():
    writable = store_open.check_writable_format_version(CURRENT_FORMAT_VERSION)
    assert writable == CURRENT_FORMAT_VERSION


def test_a_readable_but_unwritable_version_is_refused(monkeypatch):
    """Unreachable by construction today -- this build reads exactly the
    version it writes -- and kept because the invariant is about the *next*
    format. Monkeypatching is what makes it observable at all: the state it
    guards arrives with a second readable version, not before, and a guard
    that has never been executed is a guess.
    """
    monkeypatch.setattr(store_open, "CURRENT_FORMAT_VERSION", "0.1.1")
    monkeypatch.setattr(store_open, "SUPPORTED_FORMAT_VERSIONS", {(0, 1): (1,)})

    store_open.check_format_version("0.1.0")  # still readable
    with pytest.raises(store_open.UnsupportedFormatVersion, match="reads but cannot write"):
        store_open.check_writable_format_version("0.1.0", source="source release X")


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
