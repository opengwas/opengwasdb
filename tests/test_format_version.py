"""`format_version` semantics and the reader's obligations (issue #112, ADR 0038).

Spec §21 has said since v0.1 that a reader "MUST reject Store Releases with
unsupported **major** format versions" and "MAY support older **minor**
versions". The code implemented that as `frozenset({"0.1"})` -- exact-set
membership over opaque strings, with no major, no minor and no ordering -- so
the sentence was not implementable against it, and nothing tested it either way.

These tests pin the three behaviours a reader owes a store it did not write:
reject an unknown major, read a known one, and say something when it meets a
minor from the future rather than quietly returning a subset of the data.

The completion tests cover the hole ADR 0038 §4 exists to close: completion
preserves its source's `format_version` because it writes into the source's
arrays, which is only honest while this build can write that format.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.store import open as store_open
from opengwasdb.store.open import (
    CURRENT_FORMAT_VERSION,
    SUPPORTED_FORMAT_VERSIONS,
    MalformedFormatVersion,
    UnsupportedFormatVersion,
    check_format_version,
    check_writable_format_version,
    open_store,
    parse_format_version,
)


def _set_version(store: Path, version: str) -> None:
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


# --- parsing ---------------------------------------------------------------


def test_version_parses_as_major_minor():
    assert parse_format_version("0.1") == (0, 1)
    assert parse_format_version("12.34") == (12, 34)


@pytest.mark.parametrize("version", ["", "1", "1.", ".1", "1.2.3", "banana", "1.x", "v1.0"])
def test_a_version_that_is_not_major_minor_is_refused(version):
    """Refused, not coerced. To a caller deciding whether it can read a
    release, an unparseable version and a future one are the same answer."""
    with pytest.raises(MalformedFormatVersion):
        parse_format_version(version)
    # And it stops a read, not merely a parse.
    assert issubclass(MalformedFormatVersion, UnsupportedFormatVersion)


# --- the reader contract ---------------------------------------------------


def test_the_version_this_build_writes_is_one_it_can_read():
    """Otherwise the build could not open what it had just produced."""
    check_format_version(CURRENT_FORMAT_VERSION)
    major, minor = parse_format_version(CURRENT_FORMAT_VERSION)
    assert SUPPORTED_FORMAT_VERSIONS[major] >= minor


def test_an_unknown_major_is_rejected():
    unknown = max(SUPPORTED_FORMAT_VERSIONS) + 1
    with pytest.raises(UnsupportedFormatVersion, match=f"major version {unknown}"):
        check_format_version(f"{unknown}.0")


def test_an_older_minor_within_a_known_major_is_accepted():
    major = max(SUPPORTED_FORMAT_VERSIONS)
    check_format_version(f"{major}.0")  # does not raise


def test_a_newer_minor_is_read_but_warned_about(caplog):
    """Reading it is the definition of minor; the warning is how a change
    misclassified as minor becomes visible instead of silently losing data."""
    major = max(SUPPORTED_FORMAT_VERSIONS)
    future = SUPPORTED_FORMAT_VERSIONS[major] + 7

    with caplog.at_level("WARNING"):
        check_format_version(f"{major}.{future}")

    assert f"{major}.{future}" in caplog.text
    assert "not visible here" in caplog.text


# --- against a real store --------------------------------------------------


def test_open_store_rejects_an_unknown_major(dense_store_path):
    _set_version(dense_store_path, f"{max(SUPPORTED_FORMAT_VERSIONS) + 1}.0")

    with pytest.raises(UnsupportedFormatVersion):
        open_store(dense_store_path)


def test_validation_reports_an_unknown_major_as_an_error_not_a_crash(dense_store_path):
    from opengwasdb.validation import validate_store

    _set_version(dense_store_path, "99.0")
    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("99.0" in error for error in result.errors)


# --- writing is narrower than reading (ADR 0038 §3) ------------------------


def test_the_current_version_is_writable():
    assert check_writable_format_version(CURRENT_FORMAT_VERSION) == CURRENT_FORMAT_VERSION


def test_a_readable_but_unwritable_version_is_refused():
    """The state ADR 0038 §4 exists for, which #114 created: `0.1` stays
    readable and stops being written."""
    store_open.check_format_version("0.1")  # still readable
    with pytest.raises(UnsupportedFormatVersion, match="reads but cannot write"):
        store_open.check_writable_format_version("0.1", source="source release X")


def test_completion_refuses_a_source_it_cannot_write_before_doing_any_work(
    tmp_path, dense_store_path
):
    """Completion preserves its source's format_version because it writes into
    the source's arrays. Once this build writes a newer format, stamping the
    old version onto newly encoded arrays would produce a store that lies about
    its own encoding -- so it fails instead, and fails *before* the imputation
    rather than at manifest-write time an hour later.
    """
    _set_version(dense_store_path, "0.1")
    out = tmp_path / "completed.opengwasdb"

    with pytest.raises(UnsupportedFormatVersion, match="rebuild it from source"):
        complete_dense_store(
            dense_store_path,
            out,
            # Deliberately not a usable panel: the version check must fire
            # first, so completion never gets far enough to read it.
            ld_dir=tmp_path / "no-such-panel",
            ancestry="EUR",
        )

    assert not out.exists()
