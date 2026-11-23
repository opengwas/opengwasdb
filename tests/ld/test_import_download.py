"""Downloader's external HTTPS seam is exercised without a network (opengwas/ogld#93)."""

import contextlib
import hashlib
import io
import json
import os
import socket
import ssl
import subprocess
import threading
import urllib.parse
from pathlib import Path

import pytest

from opengwasdb.ld.build.importing import download
from opengwasdb.ld.build.importing.download import Parts, inventory


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


# --- concurrent downloads and dead addresses (opengwas/ogld#97)
# ---------------------------------------


def _payload(name: str) -> bytes:
    return (name * 4).encode()


def _ledger(*names: str) -> list[dict]:
    return [{"name": n, "bytes": len(_payload(n))} for n in names]


def test_ensure_all_fetches_parts_concurrently_with_receipts(tmp_path: Path) -> None:
    names = [f"part-{i}" for i in range(4)]
    lock, state = threading.Lock(), {"now": 0, "peak": 0}
    gather = threading.Barrier(4, timeout=1)

    def fetch(url: str) -> io.BytesIO:
        name = url.rsplit("/", 1)[-1]
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        with contextlib.suppress(threading.BrokenBarrierError):
            gather.wait()  # four fetches must be in flight together to pass
        with lock:
            state["now"] -= 1
        return io.BytesIO(_payload(name))

    parts = Parts(tmp_path, _ledger(*names), fetch=fetch, retry_delay=0)
    paths = parts.ensure_all(names, workers=4)
    assert state["peak"] == 4
    assert [p.name for p in paths] == names
    for name in names:
        receipt = json.loads((tmp_path / ".sha256" / (name + ".json")).read_text())
        assert receipt["sha256"] == hashlib.sha256(_payload(name)).hexdigest()
        assert receipt["downloaded"] is True
        assert receipt["download_identity"] == parts._identity(tmp_path / name)
        assert parts.hashes[name] == receipt["sha256"]


def test_concurrent_failure_refuses_by_name_and_leaves_nothing(tmp_path: Path) -> None:
    def fetch(url: str) -> io.BytesIO:
        name = url.rsplit("/", 1)[-1]
        if name == "part-2":
            raise OSError("connection reset")
        return io.BytesIO(_payload(name))

    names = [f"part-{i}" for i in range(4)]
    parts = Parts(tmp_path, _ledger(*names), fetch=fetch, retry_delay=0)
    with pytest.raises(ValueError, match=r"part-2: HTTPS download failed"):
        parts.ensure_all(names, workers=3)
    assert not list(tmp_path.glob("*.partial"))
    assert not (tmp_path / "part-2").exists()
    assert not (tmp_path / ".sha256" / "part-2.json").exists()


def test_delete_after_concurrent_download_removes_only_owned(tmp_path: Path) -> None:
    supplied = tmp_path / "supplied"
    supplied.mkdir()
    (supplied / "part-0").write_bytes(_payload("part-0"))
    names = ["part-0", "part-1"]
    parts = Parts(
        tmp_path / "cache",
        _ledger(*names),
        local=supplied,
        fetch=lambda url: io.BytesIO(_payload(url.rsplit("/", 1)[-1])),
    )
    parts.ensure_all(names, workers=2)
    assert parts.delete(names) == 1
    assert (supplied / "part-0").exists()
    assert not (tmp_path / "cache" / "part-1").exists()


def _endpoint(dead: set[str], timeouts: list[str]) -> download.PinnedHttps:
    def resolve(host: str) -> list[str]:
        assert host == download.HOST
        return ["10.0.0.1", "10.0.0.2"]

    def open_address(address, url, connect_timeout, read_timeout, connected):
        if address in dead:
            timeouts.append(address)
            raise download._Unreachable("connect timed out")
        connected()
        return io.BytesIO(_payload(url.rsplit("/", 1)[-1]))

    return download.PinnedHttps(resolve=resolve, open_address=open_address)


def test_a_dead_address_costs_one_timeout_then_is_skipped(tmp_path: Path) -> None:
    timeouts: list[str] = []
    endpoint = _endpoint({"10.0.0.1"}, timeouts)
    names = [f"part-{i}" for i in range(8)]
    parts = Parts(tmp_path, _ledger(*names), fetch=endpoint, retry_delay=0)
    parts.ensure_all(names, workers=4)
    assert timeouts == ["10.0.0.1"]
    assert endpoint.dead == {"10.0.0.1"}
    assert all((tmp_path / n).read_bytes() == _payload(n) for n in names)


def test_every_address_failing_refuses_the_part(tmp_path: Path) -> None:
    endpoint = _endpoint({"10.0.0.1", "10.0.0.2"}, [])
    parts = Parts(tmp_path, _ledger("part-0"), fetch=endpoint, retry_delay=0)
    with pytest.raises(ValueError, match=r"part-0: HTTPS download failed"):
        parts.ensure_all(["part-0"], workers=2)
    assert not list(tmp_path.glob("part-0*"))
    assert not (tmp_path / ".sha256" / "part-0.json").exists()


def test_when_every_address_is_dead_and_none_answered_the_marks_are_cleared() -> None:
    timeouts: list[str] = []
    endpoint = _endpoint({"10.0.0.1", "10.0.0.2"}, timeouts)
    for _ in range(2):
        with pytest.raises(OSError, match="connect timed out"):
            endpoint(download.BUCKET + download.PREFIX + "part-0")
    # The second call probed both again: the first call's dead marks were forgiven.
    assert sorted(timeouts) == ["10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.2"]
    assert endpoint.good == set()


def _after_connect(failure: Exception) -> download.PinnedHttps:
    def open_address(address, url, connect_timeout, read_timeout, connected):
        connected()
        raise failure

    return download.PinnedHttps(resolve=lambda host: ["10.0.0.1"], open_address=open_address)


@pytest.mark.parametrize(
    "failure",
    [TimeoutError("read timed out"), ConnectionResetError("reset"), download._Refused("HTTP 503")],
)
def test_a_failure_after_the_connect_leaves_the_address_alive(failure: Exception) -> None:
    endpoint = _after_connect(failure)
    with pytest.raises(OSError, match=str(failure)):
        endpoint(download.BUCKET + "x")
    assert endpoint.dead == set()
    assert endpoint.good == {"10.0.0.1"}


def test_the_probe_lock_is_released_once_the_connect_succeeds() -> None:
    connected_a, release = threading.Event(), threading.Event()
    calls: list[int] = []

    def open_address(address, url, connect_timeout, read_timeout, connected):
        calls.append(1)
        connected()
        if len(calls) == 1:  # a slow response: headers take their time
            connected_a.set()
            assert release.wait(5)
        return io.BytesIO(b"x")

    endpoint = download.PinnedHttps(resolve=lambda host: ["10.0.0.1"], open_address=open_address)
    url = download.BUCKET + "x"
    first = threading.Thread(target=endpoint, args=(url,))
    first.start()
    assert connected_a.wait(5)
    # A second worker, whose candidate list predates the success, is not queued behind the first.
    second = threading.Thread(target=endpoint, args=(url,))
    second.start()
    second.join(2)
    stuck = second.is_alive()
    release.set()
    first.join(5)
    second.join(5)
    assert not stuck
    assert len(calls) == 2


def test_only_the_bucket_host_is_a_target() -> None:
    with pytest.raises(ValueError, match="not the public Pan-UKB"):
        download.PinnedHttps(resolve=lambda h: [], open_address=lambda *a: io.BytesIO())(
            "https://x/"
        )


# --- the connection seam: timeouts and what counts as unreachable (opengwas/ogld#97)
# ------------------


class _Sock:
    def __init__(self) -> None:
        self.timeouts: list[float | None] = []

    def close(self) -> None:
        pass

    def settimeout(self, value: float | None) -> None:
        self.timeouts.append(value)


class _Context:
    def __init__(self, handshake: Exception | None = None) -> None:
        self.sock, self.hostname, self.handshake = _Sock(), None, handshake

    def wrap_socket(self, raw, *, server_hostname):
        self.hostname = server_hostname
        if self.handshake is not None:
            raise self.handshake
        return self.sock


def _connection(context: _Context) -> download._Pinned:
    return download._Pinned("10.0.0.1", download.HOST, (12.0, 120.0), lambda: None, context)


def test_connect_and_read_timeouts_reach_the_socket_separately(monkeypatch) -> None:
    seen = {}

    def create_connection(address, timeout):
        seen["address"], seen["timeout"] = address, timeout
        return object()

    monkeypatch.setattr(download.socket, "create_connection", create_connection)
    context = _Context()
    _connection(context).connect()
    assert seen == {"address": ("10.0.0.1", 443), "timeout": 12.0}
    assert context.sock.timeouts == [120.0]
    assert context.hostname == download.HOST


def test_the_connected_callback_runs_after_the_handshake(monkeypatch) -> None:
    monkeypatch.setattr(download.socket, "create_connection", lambda *a, **k: object())
    events = []
    context = _Context()
    connection = download._Pinned(
        "10.0.0.1", download.HOST, (12.0, 120.0), lambda: events.append("connected"), context
    )
    connection.connect()
    assert events == ["connected"]


@pytest.mark.parametrize("stage", ["connect", "handshake"])
@pytest.mark.parametrize("failure", [TimeoutError("timed out"), ConnectionRefusedError("refused")])
def test_a_connect_or_handshake_failure_is_unreachable(monkeypatch, stage, failure) -> None:
    def create_connection(address, timeout):
        if stage == "connect":
            raise failure
        return _Sock()

    monkeypatch.setattr(download.socket, "create_connection", create_connection)
    context = _Context(handshake=failure if stage == "handshake" else None)
    with pytest.raises(download._Unreachable):
        _connection(context).connect()


# --- the real socket path, against a local TLS server (opengwas/ogld#97)
# ------------------------------


def _certificate(directory: Path, hostname: str) -> tuple[Path, Path]:
    cert, key = directory / f"{hostname}.pem", directory / f"{hostname}.key"
    subprocess.run(
        [
            *("openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1"),
            *("-subj", f"/CN={hostname}", "-addext", f"subjectAltName=DNS:{hostname}"),
            *("-keyout", str(key), "-out", str(cert)),
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


def _serve(cert: Path, key: Path, status: int, body: bytes) -> int:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    listener = socket.socket()
    listener.settimeout(5)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def serve() -> None:
        with listener, contextlib.suppress(OSError):
            conn, _ = listener.accept()
            with context.wrap_socket(conn, server_side=True) as tls:
                tls.recv(4096)
                tls.sendall(
                    f"HTTP/1.1 {status} X\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
                )

    threading.Thread(target=serve, daemon=True).start()
    return int(listener.getsockname()[1])


def _open(port: int, cert: Path):
    return download._open_address(
        "127.0.0.1",
        download.BUCKET + "x?a=b",
        2,
        2,
        lambda: None,
        context=ssl.create_default_context(cafile=str(cert)),
        port=port,
    )


def test_a_pinned_address_is_verified_against_the_bucket_hostname(tmp_path: Path) -> None:
    cert, key = _certificate(tmp_path, download.HOST)
    with _open(_serve(cert, key, 200, b"bytes"), cert) as stream:
        assert stream.read() == b"bytes"


def test_a_certificate_for_another_hostname_is_refused(tmp_path: Path) -> None:
    cert, key = _certificate(tmp_path, "imposter.example.com")
    with pytest.raises(ssl.SSLCertVerificationError):
        _open(_serve(cert, key, 200, b"bytes"), cert)


def test_an_http_error_is_refused_and_does_not_kill_the_address(tmp_path: Path) -> None:
    cert, key = _certificate(tmp_path, download.HOST)
    port = _serve(cert, key, 503, b"")
    endpoint = download.PinnedHttps(
        resolve=lambda host: ["127.0.0.1"],
        open_address=lambda a, u, c, r, connected: download._open_address(
            a, u, c, r, connected, context=ssl.create_default_context(cafile=str(cert)), port=port
        ),
    )
    with pytest.raises(download._Refused, match="HTTP 503"):
        endpoint(download.BUCKET + "x")
    assert endpoint.dead == set()
    assert endpoint.good == {"127.0.0.1"}
