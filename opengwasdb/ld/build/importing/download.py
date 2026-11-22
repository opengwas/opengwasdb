"""Bounded HTTPS downloads and durable per-part SHA-256 receipts (ADR 0015)."""

from __future__ import annotations

import http.client
import json
import os
import random
import socket
import ssl
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, BinaryIO, cast

from ogld.build.record import sha256

BUCKET = "https://pan-ukb-us-east-1.s3.us-east-1.amazonaws.com/"
PREFIX = "ld_release/UKBB.EUR.ldadj.bm/parts/"
Fetch = Callable[[str], BinaryIO]


HOST = urllib.parse.urlsplit(BUCKET).hostname or ""
CONNECT_TIMEOUT = 12.0
READ_TIMEOUT = 120.0
ATTEMPTS = 8
Resolve = Callable[[str], list[str]]
OpenAddress = Callable[[str, str, float, float], BinaryIO]


def _resolve(host: str) -> list[str]:
    found = socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM)
    return sorted({str(info[4][0]) for info in found})


class _Pinned(http.client.HTTPSConnection):
    """Connects to a chosen IP while SNI and the certificate check name the real host."""

    def __init__(
        self,
        address: str,
        host: str,
        timeouts: tuple[float, float],
        context: ssl.SSLContext | None = None,
        port: int = 443,
    ):
        context = context or ssl.create_default_context()
        super().__init__(host, port, timeout=timeouts[0], context=context)
        self._address, self._read_timeout = address, timeouts[1]

    def connect(self) -> None:
        raw = socket.create_connection((self._address, self.port), timeout=self.timeout)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)  # type: ignore[attr-defined]
        self.sock.settimeout(self._read_timeout)


class _Body:
    """The response body, closing its connection with it."""

    def __init__(self, response: http.client.HTTPResponse, connection: _Pinned) -> None:
        self._response, self._connection = response, connection

    def read(self, size: int = -1) -> bytes:
        return self._response.read(size)

    def __enter__(self) -> _Body:
        return self

    def __exit__(self, *_: object) -> None:
        self._response.close()
        self._connection.close()


def _open_address(
    address: str,
    url: str,
    connect_timeout: float,
    read_timeout: float,
    *,
    context: ssl.SSLContext | None = None,
    port: int = 443,
) -> BinaryIO:
    split = urllib.parse.urlsplit(url)
    connection = _Pinned(address, HOST, (connect_timeout, read_timeout), context, port)
    try:
        connection.request("GET", split.path + ("?" + split.query if split.query else ""))
        response = connection.getresponse()
    except BaseException:
        connection.close()
        raise
    if response.status != 200:
        connection.close()
        raise _Refused(f"HTTP {response.status} from {address}")
    return cast(BinaryIO, _Body(response, connection))


class _Refused(OSError):
    """The server answered, so the address is alive: only the request failed."""


class PinnedHttps:
    """Fetch the public bucket over addresses that answer; skip those that never do.

    S3's DNS rotates over addresses, some of which never answer. A connect
    failure marks that address dead for this run's life (and is not retried);
    any other failure leaves it alive. TLS still verifies the bucket's hostname.
    """

    def __init__(
        self,
        *,
        resolve: Resolve = _resolve,
        open_address: OpenAddress = _open_address,
        connect_timeout: float = CONNECT_TIMEOUT,
        read_timeout: float = READ_TIMEOUT,
    ) -> None:
        self._resolve, self._open = resolve, open_address
        self._connect, self._read = connect_timeout, read_timeout
        self.dead: set[str] = set()
        self.good: set[str] = set()
        self._lock = threading.Lock()
        self._probe = threading.Lock()

    def _candidates(self) -> list[str]:
        """Addresses that have answered first (DNS rotates, so even if not just returned)."""
        found = self._resolve(HOST)
        with self._lock:
            live = [a for a in found if a not in self.dead]
            if not live and not self.good:  # all dead: forgive them, an outage may have passed
                self.dead.difference_update(found)
                live = list(found)
            known = list(self.good)
        random.shuffle(known)
        return known + [a for a in live if a not in known]

    def _try(self, address: str, url: str) -> BinaryIO:
        stream = self._open(address, url, self._connect, self._read)
        with self._lock:
            self.good.add(address)
        return stream

    def _unknown(self, address: str, url: str) -> BinaryIO | None:
        """Probe an address one caller at a time, so a dead one costs a single timeout."""
        with self._probe:
            with self._lock:
                skipped = address in self.dead
            return None if skipped else self._try(address, url)

    def __call__(self, url: str) -> BinaryIO:
        if not url.startswith(BUCKET):
            raise ValueError(f"not the public Pan-UKB HTTPS bucket: {url}")
        last: OSError = OSError(f"no address resolved for {HOST}")
        for address in self._candidates():
            with self._lock:
                known = address in self.good
            try:
                stream = self._try(address, url) if known else self._unknown(address, url)
            except _Refused:
                raise
            except (TimeoutError, ConnectionError, socket.gaierror) as failed:
                with self._lock:
                    self.dead.add(address)
                    self.good.discard(address)
                last = OSError(f"{address}: {failed}")
                continue
            if stream is not None:
                return stream
        raise last


def https(url: str) -> BinaryIO:
    """One-off fetch; a run shares one `PinnedHttps` so dead addresses are remembered."""
    return PinnedHttps()(url)


def atomic_json(path: Path, value: object) -> None:
    """A killed process leaves the last complete receipt, never half of a hash ledger."""
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


_S3_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _listing_entry(obj: ET.Element) -> dict[str, Any]:
    key = obj.findtext("s:Key", namespaces=_S3_NS) or ""
    return {
        "name": key.removeprefix(PREFIX),
        "bytes": int(obj.findtext("s:Size", namespaces=_S3_NS) or "-1"),
    }


def _incomplete() -> ValueError:
    return ValueError("incomplete Pan-UKB bucket listing")


def _listing_page(fetch: Fetch, token: str) -> tuple[list[dict[str, Any]], str | None]:
    query = {"list-type": "2", "prefix": PREFIX}
    if token:
        query["continuation-token"] = token
    with fetch(BUCKET + "?" + urllib.parse.urlencode(query)) as stream:
        root = ET.fromstring(stream.read())
    entries = [_listing_entry(obj) for obj in root.findall("s:Contents", _S3_NS)]
    truncated = root.findtext("s:IsTruncated", namespaces=_S3_NS)
    if truncated == "false":
        return entries, None
    following = root.findtext("s:NextContinuationToken", namespaces=_S3_NS) or ""
    if truncated != "true" or not following:
        raise _incomplete()
    return entries, following


def inventory(fetch: Fetch = https) -> list[dict[str, Any]]:
    """Paginate the bucket listing; an incomplete page is never a complete inventory."""
    result: list[dict[str, Any]] = []
    token = ""
    seen: set[str] = set()
    while True:
        entries, following = _listing_page(fetch, token)
        result.extend(entries)
        if following is None:
            return result
        if following in seen:
            raise _incomplete()
        seen.add(following)
        token = following


def _settled(futures: list[Future[Path]]) -> list[Path]:
    """Results in order, or the first failure of a future that ran."""
    ran = [future for future in futures if not future.cancelled()]
    for failed in (future.exception() for future in ran):
        if failed is not None:
            raise failed
    return [future.result() for future in futures]


def _in_parallel(work: Callable[[str], Path], names: list[str], workers: int) -> list[Path]:
    """The first failure stops queued work and is raised once in-flight work ends."""
    if workers <= 1 or len(names) <= 1:
        return [work(name) for name in names]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, name) for name in names]
        wait(futures, return_when=FIRST_EXCEPTION)
        for future in futures:
            future.cancel()
    return _settled(futures)


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
            cache_resolved, local_resolved = cache.resolve(), local.resolve()
            if (
                cache_resolved == local_resolved
                or cache_resolved in local_resolved.parents
                or local_resolved in cache_resolved.parents
            ):
                raise ValueError(
                    f"supplied parts directory {local} and download cache {cache} overlap; "
                    "neither may be, contain, or resolve into the other"
                )
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

    def ensure_all(self, names: list[str], workers: int = 8) -> list[Path]:
        """`ensure` each part, missing ones downloaded in parallel through the same path.

        A failure stops new work, lets in-flight downloads finish or fail, and
        refuses by the first failed part's name.
        """
        return _in_parallel(self.ensure, names, workers)

    def _verify(self, path: Path, record: dict[str, Any]) -> str:
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"{path.name}: byte count differs from listing")
        digest = sha256(path)
        if "sha256" in record and digest != record["sha256"]:
            raise ValueError(f"{path.name}: SHA-256 differs from ledger")
        return digest

    def _download(self, name: str, record: dict[str, Any], path: Path) -> None:
        temporary = path.with_name(name + ".partial")
        for attempt in range(ATTEMPTS):
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
                if attempt == ATTEMPTS - 1:
                    raise ValueError(f"{name}: HTTPS download failed: {failed}") from failed
                time.sleep(min(self._delay * 2**attempt, 30 * self._delay))

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

    def delete(self, names: list[str]) -> int:
        """Unlink recorded downloads and count logical parts retained as unowned."""
        kept = 0
        for name in names:
            record = self._record(name)
            path = self.cache / name
            retained = self.local is not None and (self.local / name).exists()
            if path.exists():
                if self._owned(path, record):
                    path.unlink()
                else:
                    retained = True
            kept += int(retained)
        return kept
