"""Downloader's external HTTPS seam is exercised without a network (#93)."""

import hashlib
import io
from pathlib import Path

import pytest

from ogld.build.importing.download import Parts


def test_download_retries_checks_bytes_and_remembers_sha256(tmp_path: Path) -> None:
    calls = []

    def fetch(url):
        calls.append(url)
        return io.BytesIO(b"short" if len(calls) == 1 else b"payload")

    parts = Parts(tmp_path, [{"name": "part-1", "bytes": 7}], fetch=fetch, retry_delay=0)
    path = parts.ensure("part-1")
    assert path.read_bytes() == b"payload"
    assert len(calls) == 2
    assert parts.hashes == {"part-1": hashlib.sha256(b"payload").hexdigest()}
    # A different process re-verifies bytes, without touching HTTPS.
    restarted = Parts(tmp_path, [], fetch=fetch)
    assert restarted.ensure("part-1") == path
    assert len(calls) == 2


def test_part_hash_mismatch_refuses_by_name(tmp_path: Path) -> None:
    (tmp_path / "part-1").write_bytes(b"corrupt")
    parts = Parts(tmp_path, [{"name": "part-1", "bytes": 7, "sha256": "0" * 64}])
    with pytest.raises(ValueError, match=r"part-1.*SHA-256"):
        parts.ensure("part-1")
