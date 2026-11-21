"""Bounded HTTPS downloads and durable per-part SHA-256 receipts (ADR 0015)."""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, cast

from ogld.build.record import sha256

BUCKET = "https://pan-ukb-us-east-1.s3.us-east-1.amazonaws.com/"
PREFIX = "ld_release/UKBB.EUR.ldadj.bm/parts/"
Fetch = Callable[[str], BinaryIO]


def https(url: str) -> BinaryIO:
    """Only the public HTTPS source is a download target, with a finite socket timeout."""
    if not url.startswith(BUCKET):
        raise ValueError(f"not the public Pan-UKB HTTPS bucket: {url}")
    return cast(BinaryIO, urllib.request.urlopen(url, timeout=120))


def atomic_json(path: Path, value: object) -> None:
    """A killed process leaves the last complete receipt, never half of a hash ledger."""
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def inventory(fetch: Fetch = https) -> list[dict[str, Any]]:
    """Paginate the bucket listing; an incomplete page is never a complete inventory."""
    result: list[dict[str, Any]] = []
    token = ""
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    seen: set[str] = set()
    while True:
        query = {"list-type": "2", "prefix": PREFIX}
        if token:
            query["continuation-token"] = token
        with fetch(BUCKET + "?" + urllib.parse.urlencode(query)) as stream:
            root = ET.fromstring(stream.read())
        for obj in root.findall("s:Contents", ns):
            key = obj.findtext("s:Key", namespaces=ns) or ""
            result.append(
                {
                    "name": key.removeprefix(PREFIX),
                    "bytes": int(obj.findtext("s:Size", namespaces=ns) or "-1"),
                }
            )
        truncated = root.findtext("s:IsTruncated", namespaces=ns)
        if truncated == "false":
            return result
        token = root.findtext("s:NextContinuationToken", namespaces=ns) or ""
        if truncated != "true" or not token or token in seen:
            raise ValueError("incomplete Pan-UKB bucket listing")
        seen.add(token)


class Parts:
    """A read-only supplied parts directory plus an owned cache and hash receipts.

    Bucket listings do not publish SHA-256. First acquisition pins the bytes in
    an atomic receipt; subsequent reads verify that pin, never an S3 multipart ETag.
    A supplied ledger's SHA-256 takes precedence over a local receipt.
    """

    def __init__(
        self,
        cache: Path,
        ledger: list[dict[str, Any]],
        *,
        local: Path | None = None,
        fetch: Fetch = https,
        retry_delay: float = 1,
    ) -> None:
        self.cache, self.local = cache, local
        self._supplied: set[tuple[int, int]] = set()
        if local is not None:
            for entry in local.iterdir():
                if entry.is_file():
                    stat = entry.stat()
                    self._supplied.add((stat.st_dev, stat.st_ino))
        self._fetch, self._delay = fetch, retry_delay
        self._ledger = {entry["name"]: entry for entry in ledger}
        self._receipts = cache / ".sha256"
        self._receipts.mkdir(parents=True, exist_ok=True)
        self.hashes: dict[str, str] = {}

    def _record(self, name: str) -> dict[str, Any]:
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"unsafe part name {name!r}")
        supplied = self._ledger.get(name, {})
        receipt = self._receipts / (name + ".json")
        saved = json.loads(receipt.read_text()) if receipt.exists() else {}
        record = {**saved, **supplied}
        # A supplied ledger may pin bytes, but cannot grant permission to unlink them.
        record["downloaded"] = saved.get("downloaded", False)
        record["download_identity"] = saved.get("download_identity")
        if "bytes" not in record:
            raise ValueError(f"{name}: absent from listing and SHA-256 ledger")
        if int(record["bytes"]) <= 0:
            raise ValueError(f"{name}: invalid listing byte count")
        return record

    def _path(self, name: str) -> Path:
        if self.local is not None and (self.local / name).exists():
            return self.local / name
        return self.cache / name

    def ensure(self, name: str) -> Path:
        """Re-verify existing bytes, or fetch and verify before publishing the receipt."""
        record = self._record(name)
        path = self._path(name)
        downloaded = not path.exists()
        if downloaded:
            self._download(name, record, path)
        digest = self._verify(path, record)
        owned = downloaded or self._owned(path, record)
        atomic_json(
            self._receipts / (name + ".json"),
            {
                **record,
                "sha256": digest,
                "downloaded": owned,
                "download_identity": self._identity(path) if owned else None,
            },
        )
        self.hashes[name] = digest
        return path

    def _verify(self, path: Path, record: dict[str, Any]) -> str:
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"{path.name}: byte count differs from listing")
        digest = sha256(path)
        if "sha256" in record and digest != record["sha256"]:
            raise ValueError(f"{path.name}: SHA-256 differs from ledger")
        return digest

    def _download(self, name: str, record: dict[str, Any], path: Path) -> None:
        temporary = path.with_name(name + ".partial")
        for attempt in range(5):
            try:
                with (
                    self._fetch(BUCKET + PREFIX + urllib.parse.quote(name)) as stream,
                    temporary.open("wb") as out,
                ):
                    count = 0
                    while data := stream.read(8 << 20):
                        count += len(data)
                        if count > record["bytes"]:
                            raise ValueError(f"{name}: byte count exceeds listing")
                        out.write(data)
                self._verify(temporary, record)
                temporary.replace(path)
                return
            except (OSError, ValueError) as failed:
                temporary.unlink(missing_ok=True)
                if attempt == 4:
                    raise ValueError(f"{name}: HTTPS download failed: {failed}") from failed
                time.sleep(self._delay * 2**attempt)

    def verify_closed(self, names: list[str]) -> None:
        """Re-verify retained inputs of a closed chromosome, without re-fetching deletions."""
        for name in names:
            record = self._record(name)
            if "sha256" not in record:
                raise ValueError(f"{name}: closed chromosome lacks SHA-256 receipt")
            path = self._path(name)
            if path.exists():
                self._verify(path, record)
            self.hashes[name] = record["sha256"]

    @staticmethod
    def _identity(path: Path) -> list[int]:
        stat = path.stat()
        return [stat.st_dev, stat.st_ino, stat.st_ctime_ns]

    def _owned(self, path: Path, record: dict[str, Any]) -> bool:
        identity = self._identity(path)
        return (
            record.get("downloaded") is True
            and record.get("download_identity") == identity
            and (identity[0], identity[1]) not in self._supplied
            and not path.is_symlink()
        )

    def delete(self, names: list[str]) -> None:
        """Only unlink the recorded downloads, never supplied files or their path aliases."""
        for name in names:
            record = self._record(name)
            path = self.cache / name
            if path.exists() and self._owned(path, record):
                path.unlink()
