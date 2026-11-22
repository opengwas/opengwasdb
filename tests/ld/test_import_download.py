"""Downloader's external HTTPS seam is exercised without a network (#93)."""

import hashlib
import io
import json
import os
import urllib.parse
from pathlib import Path

import pytest

from ogld.build.importing.download import Parts, inventory


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


def _listing(*entries: str, truncated: bool = False, token: str = "") -> str:
    continuation = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        + "".join(entries)
        + f"<IsTruncated>{str(truncated).lower()}</IsTruncated>{continuation}"
        + "</ListBucketResult>"
    )


def _entry(name: str, size: int | None) -> str:
    return (
        "<Contents><Key>ld_release/UKBB.EUR.ldadj.bm/parts/"
        + name
        + "</Key>"
        + (f"<Size>{size}</Size>" if size is not None else "")
        + "</Contents>"
    )


def test_inventory_reads_a_single_listing_page() -> None:
    calls = []

    def fetch(url: str) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(_listing(_entry("part-0", 54)).encode())

    assert inventory(fetch) == [{"name": "part-0", "bytes": 54}]
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(calls[0]).query)
    assert query == {
        "list-type": ["2"],
        "prefix": ["ld_release/UKBB.EUR.ldadj.bm/parts/"],
    }


def test_inventory_follows_a_continuation_token() -> None:
    pages = iter(
        [
            _listing(_entry("part-0", 54), truncated=True, token="next page"),
            _listing(_entry("part-1", 47)),
        ]
    )
    calls = []

    def fetch(url: str) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(next(pages).encode())

    assert inventory(fetch) == [
        {"name": "part-0", "bytes": 54},
        {"name": "part-1", "bytes": 47},
    ]
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(calls[1]).query)
    assert query["continuation-token"] == ["next page"]


def test_inventory_refuses_a_truncated_listing_without_a_continuation() -> None:
    def fetch(_url: str) -> io.BytesIO:
        return io.BytesIO(_listing(truncated=True).encode())

    with pytest.raises(ValueError, match="incomplete Pan-UKB bucket listing"):
        inventory(fetch)


def test_inventory_refuses_a_repeated_continuation_token_rather_than_loop() -> None:
    calls = []

    def fetch(_url: str) -> io.BytesIO:
        calls.append(_url)
        assert len(calls) < 10, "inventory looped on a repeating token"
        return io.BytesIO(_listing(_entry("part-0", 54), truncated=True, token="same").encode())

    with pytest.raises(ValueError, match="incomplete Pan-UKB bucket listing"):
        inventory(fetch)
    assert len(calls) == 2


def test_inventory_refuses_a_page_without_is_truncated() -> None:
    unmarked = (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        + _entry("part-0", 54)
        + "<NextContinuationToken>next</NextContinuationToken>"
        + "</ListBucketResult>"
    )
    pages = iter([unmarked, _listing(_entry("part-1", 47))])

    def fetch(_url: str) -> io.BytesIO:
        return io.BytesIO(next(pages).encode())

    with pytest.raises(ValueError, match="incomplete Pan-UKB bucket listing"):
        inventory(fetch)


def test_inventory_marks_a_part_entry_without_a_size_as_invalid(tmp_path: Path) -> None:
    def fetch(_url: str) -> io.BytesIO:
        return io.BytesIO(_listing(_entry("part-0", None)).encode())

    # Parts refuses this sentinel by part name before attempting a download.
    found = inventory(fetch)
    assert found == [{"name": "part-0", "bytes": -1}]
    with pytest.raises(ValueError, match="part-0: invalid listing byte count"):
        Parts(tmp_path, found).ensure("part-0")


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
