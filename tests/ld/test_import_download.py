"""Downloader's external HTTPS seam is exercised without a network (#93)."""

import hashlib
import io
import json
import os
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


def _identity(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_ctime_ns]


def _receipt(cache: Path, record: dict[str, object]) -> None:
    receipts = cache / ".sha256"
    receipts.mkdir(exist_ok=True)
    (receipts / "part-1.json").write_text(json.dumps(record))


def test_delete_keeps_a_cache_hard_link_to_a_supplied_part(tmp_path: Path) -> None:
    cache, supplied = tmp_path / "cache", tmp_path / "supplied"
    cache.mkdir()
    supplied.mkdir()
    (supplied / "part-1").write_bytes(b"payload")
    os.link(supplied / "part-1", cache / "part-1")
    _receipt(
        cache,
        {"bytes": 7, "downloaded": True, "download_identity": _identity(cache / "part-1")},
    )
    parts = Parts(cache, [{"name": "part-1", "bytes": 7}], local=supplied)
    assert parts.delete(["part-1"]) == 1
    assert (cache / "part-1").read_bytes() == b"payload"
    assert (supplied / "part-1").read_bytes() == b"payload"


def test_delete_keeps_a_part_with_an_older_receipt_without_ownership(tmp_path: Path) -> None:
    (tmp_path / "part-1").write_bytes(b"payload")
    _receipt(tmp_path, {"bytes": 7, "sha256": hashlib.sha256(b"payload").hexdigest()})
    parts = Parts(tmp_path, [])
    assert parts.delete(["part-1"]) == 1
    assert (tmp_path / "part-1").read_bytes() == b"payload"


@pytest.mark.parametrize("claim", ["ledger", "receipt"])
def test_delete_refuses_an_ownership_claim_for_a_preexisting_cache_part(
    tmp_path: Path, claim: str
) -> None:
    path = tmp_path / "part-1"
    path.write_bytes(b"payload")
    record: dict[str, object] = {
        "name": "part-1",
        "bytes": 7,
        "downloaded": True,
        "download_identity": _identity(path) if claim == "ledger" else None,
    }
    ledger = [record] if claim == "ledger" else [{"name": "part-1", "bytes": 7}]
    if claim == "receipt":
        _receipt(tmp_path, record)
    parts = Parts(tmp_path, ledger)
    assert parts.delete(["part-1"]) == 1
    assert path.read_bytes() == b"payload"
