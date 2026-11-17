"""`scripts/pypi_release.py`: what PyPI is owed, and recovering it (#439).

The index is a dictionary here wherever the question is a rule -- which tags,
which files, which conflicts -- and a real HTTP server where the question is
how an answer is read, because "404 means no files, anything else is an
error" is exactly what a stubbed fetch would get wrong unnoticed. The GitHub
Release's copy goes through a stub `gh` on `PATH`, as `tests/test_release.py`
does.

The recovery cases are #437's acceptance criteria: after the GitHub Release but
before PyPI, after a partial upload, a re-run after success, and a conflicting
file, each with a burst of tags owed at once.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from script_modules import load_script

# `pypi_release` imports its sibling `release`, which must be loaded first.
load_script("release")
pypi = load_script("pypi_release")

TAGS = {"v0.11.0": "a" * 40, "v0.12.0": "b" * 40, "v0.12.1": "c" * 40, "v0.13.0": "d" * 40}


def _names(version: str) -> tuple[str, str]:
    names: tuple[str, str] = pypi.distribution_names(version)
    return names


def _dist(tmp_path: Path, version: str) -> tuple[Path, dict[str, str]]:
    """A directory holding `version`'s two archives, and their sha256s."""
    dist = tmp_path / f"dist-{version}"
    dist.mkdir()
    hashes = {}
    for name in _names(version):
        content = f"the verified bytes of {name}".encode()
        (dist / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    return dist, hashes


class Index:
    """An index as a dictionary: (version) -> {filename: sha256}."""

    def __init__(self, held: dict[str, dict[str, str]] | None = None) -> None:
        self.held = held or {}
        self.asked: list[tuple[str, str]] = []

    def __call__(self, index: str, version: str) -> dict[str, str]:
        self.asked.append((index, version))
        return dict(self.held.get(version, {}))


def _owed(index: Index, first: str | None = "v0.12.0", **kwargs: object) -> list[str]:
    owed = pypi.plan(TAGS, index="pypi", first=first, lookup=index, **kwargs)
    return [release.tag for release in owed]


# ── The boundary ────────────────────────────────────────────────────────────


def test_nothing_is_published_to_pypi_before_a_first_release_is_set() -> None:
    index = Index()
    assert _owed(index, first=None) == []
    assert index.asked == []


def test_tags_before_the_first_release_are_never_planned() -> None:
    # A burst: every eligible tag is owed at once, oldest first, and none older.
    index = Index()
    assert _owed(index) == ["v0.12.0", "v0.12.1", "v0.13.0"]
    assert ("pypi", "0.11.0") not in index.asked


def test_a_tag_named_for_pypi_must_be_eligible() -> None:
    with pytest.raises(pypi.ReleaseError, match="not eligible for PyPI"):
        _owed(Index(), only="v0.11.0")
    with pytest.raises(pypi.ReleaseError, match="not eligible for PyPI"):
        _owed(Index(), first=None, only="v0.13.0")
    with pytest.raises(pypi.ReleaseError, match="not a release tag on the remote"):
        _owed(Index(), only="v9.9.9")


def test_testpypi_publishes_only_a_named_tag_whatever_the_boundary() -> None:
    with pytest.raises(pypi.ReleaseError, match="never published automatically"):
        pypi.plan(TAGS, index="testpypi", first=None, lookup=Index())
    owed = pypi.plan(TAGS, index="testpypi", first=None, only="v0.11.0", lookup=Index())
    assert [(release.tag, release.version) for release in owed] == [("v0.11.0", "0.11.0")]


# ── Recovery ────────────────────────────────────────────────────────────────


def test_a_tag_with_its_github_release_but_nothing_on_pypi_is_owed(tmp_path: Path) -> None:
    dist, hashes = _dist(tmp_path, "0.12.0")
    assert _owed(Index({"0.13.0": {}})) == ["v0.12.0", "v0.12.1", "v0.13.0"]
    assert pypi.stage(dist, "0.12.0", {}, tmp_path / "upload") == sorted(hashes)


def test_a_partial_upload_is_finished_with_only_the_missing_file(tmp_path: Path) -> None:
    dist, hashes = _dist(tmp_path, "0.12.1")
    sdist, wheel = _names("0.12.1")
    index = Index({"0.12.0": {}, "0.12.1": {sdist: hashes[sdist]}})
    (owed,) = [
        r
        for r in pypi.plan(TAGS, index="pypi", first="v0.12.1", lookup=index)
        if r.tag == "v0.12.1"
    ]
    assert owed.held == [sdist]
    upload = tmp_path / "upload"
    assert pypi.stage(dist, "0.12.1", index("pypi", "0.12.1"), upload) == [wheel]
    assert sorted(path.name for path in upload.iterdir()) == [wheel]


def test_a_rerun_after_success_publishes_nothing(tmp_path: Path) -> None:
    held = {}
    for version in ("0.12.0", "0.12.1", "0.13.0"):
        held[version] = _dist(tmp_path, version)[1]
    index = Index(held)
    assert _owed(index) == []
    dist = tmp_path / "dist-0.13.0"
    assert pypi.stage(dist, "0.13.0", held["0.13.0"], tmp_path / "upload") == []
    pypi.confirm(dist, "0.13.0", index="pypi", attempts=1, interval=0, lookup=index)


def test_a_conflicting_file_on_the_index_stops_the_run(tmp_path: Path) -> None:
    dist, hashes = _dist(tmp_path, "0.12.0")
    sdist, _ = _names("0.12.0")
    other = {sdist: "0" * 64}
    assert hashes[sdist] != other[sdist]
    with pytest.raises(pypi.ReleaseError, match="never replaced"):
        pypi.stage(dist, "0.12.0", other, tmp_path / "upload")
    assert not (tmp_path / "upload").exists()
    waits: list[float] = []
    with pytest.raises(pypi.ReleaseError, match="never replaced"):
        pypi.confirm(
            dist,
            "0.12.0",
            index="pypi",
            attempts=5,
            interval=1,
            lookup=Index({"0.12.0": other}),
            sleep=waits.append,
        )
    assert waits == []  # a difference in content is never waited out


def test_a_file_no_release_builds_on_the_index_stops_the_plan() -> None:
    index = Index({"0.12.0": {"opengwasdb-0.12.0-cp312-cp312-linux_x86_64.whl": "1" * 64}})
    with pytest.raises(pypi.ReleaseError, match="which no release builds"):
        _owed(index)


def test_archives_that_are_not_exactly_the_pair_are_refused(tmp_path: Path) -> None:
    dist, _ = _dist(tmp_path, "0.12.0")
    (dist / "opengwasdb-0.12.0.dev1.tar.gz").write_bytes(b"")
    with pytest.raises(pypi.ReleaseError, match="not exactly"):
        pypi.local_files(dist, "0.12.0")


def test_confirmation_waits_out_the_index_cache_and_then_gives_up(tmp_path: Path) -> None:
    dist, hashes = _dist(tmp_path, "0.12.0")
    answers = iter([{}, {}, hashes])
    waits: list[float] = []
    pypi.confirm(
        dist,
        "0.12.0",
        index="pypi",
        attempts=3,
        interval=7,
        lookup=lambda _i, _v: next(answers),
        sleep=waits.append,
    )
    assert waits == [7, 7]
    with pytest.raises(pypi.ReleaseError, match="still lacks part of 0.12.0 after 2 checks"):
        pypi.confirm(
            dist, "0.12.0", index="pypi", attempts=2, interval=0, lookup=Index(), sleep=waits.append
        )


# ── Reading the index over HTTP ─────────────────────────────────────────────


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, tuple[int, bytes]]]:
    """A local index answering each path from a dictionary, 404 otherwise."""
    routes: dict[str, tuple[int, bytes]] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            status, body = routes.get(self.path, (404, b"not found"))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setitem(pypi.INDEXES, "pypi", f"http://127.0.0.1:{server.server_port}")
    yield routes
    server.shutdown()


def test_the_index_s_files_are_read_with_their_digests(
    served: dict[str, tuple[int, bytes]],
) -> None:
    sdist, wheel = _names("0.12.0")
    payload = {
        "urls": [
            {"filename": sdist, "digests": {"sha256": "1" * 64, "md5": "x"}},
            {"filename": wheel, "digests": {"sha256": "2" * 64}},
        ]
    }
    served["/pypi/opengwasdb/0.12.0/json"] = (200, json.dumps(payload).encode())
    assert pypi.index_files("pypi", "0.12.0") == {sdist: "1" * 64, wheel: "2" * 64}


def test_a_version_the_index_never_saw_holds_no_files(served: dict[str, tuple[int, bytes]]) -> None:
    assert served == {}
    assert pypi.index_files("pypi", "0.12.0") == {}


def test_an_index_error_is_not_read_as_no_files(served: dict[str, tuple[int, bytes]]) -> None:
    served["/pypi/opengwasdb/0.12.0/json"] = (503, b"unavailable")
    with pytest.raises(pypi.ReleaseError, match="HTTP 503"):
        pypi.index_files("pypi", "0.12.0")
    served["/pypi/opengwasdb/0.12.0/json"] = (200, b'{"info": {}}')
    with pytest.raises(pypi.ReleaseError, match="did not list files"):
        pypi.index_files("pypi", "0.12.0")


# ── The GitHub Release's copy, through a stub `gh` ──────────────────────────


@pytest.fixture
def release_gh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A `gh` whose one Release's assets live in a directory, and that logs calls."""
    assets = tmp_path / "release-assets"
    assets.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(
        f"""#!{sys.executable}
import json, os, shutil, sys
from pathlib import Path
assets = Path(os.environ["FAKE_ASSETS"])
args = sys.argv[1:]
with open(os.environ["FAKE_GH_LOG"], "a") as handle:
    handle.write(" ".join(args) + "\\n")
if args[:2] == ["release", "view"]:
    print(json.dumps({{"assets": [{{"name": p.name}} for p in sorted(assets.iterdir())]}}))
elif args[:2] == ["release", "upload"]:
    for path in args[3:args.index("--repo")]:
        if (assets / Path(path).name).exists():
            sys.exit("asset already exists")
        shutil.copy(path, assets)
elif args[:2] == ["release", "download"]:
    name = args[args.index("--pattern") + 1]
    shutil.copy(assets / name, args[args.index("--dir") + 1])
else:
    sys.exit("unexpected gh invocation: " + " ".join(args))
""",
        encoding="utf-8",
    )
    (bin_dir / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_ASSETS", str(assets))
    monkeypatch.setenv("FAKE_GH_LOG", str(tmp_path / "gh.log"))
    return assets


def test_verified_archives_are_stored_once_and_fetched_for_a_retry(
    tmp_path: Path, release_gh: Path
) -> None:
    dist, hashes = _dist(tmp_path, "0.12.0")
    retry = tmp_path / "retry"
    assert pypi.fetch_stored("v0.12.0", "0.12.0", retry, "o/r") is False
    assert pypi.store("v0.12.0", "0.12.0", dist, "o/r") is True
    assert pypi.store("v0.12.0", "0.12.0", dist, "o/r") is False
    assert pypi.fetch_stored("v0.12.0", "0.12.0", retry, "o/r") is True
    assert pypi.local_files(retry, "0.12.0") == hashes


def test_a_release_holding_half_the_pair_is_refused(tmp_path: Path, release_gh: Path) -> None:
    dist, _ = _dist(tmp_path, "0.12.0")
    sdist, _ = _names("0.12.0")
    (release_gh / sdist).write_bytes((dist / sdist).read_bytes())
    with pytest.raises(pypi.ReleaseError, match="delete the partial asset"):
        pypi.store("v0.12.0", "0.12.0", dist, "o/r")
    with pytest.raises(pypi.ReleaseError, match="delete the partial asset"):
        pypi.fetch_stored("v0.12.0", "0.12.0", tmp_path / "retry", "o/r")


# ── The command line ────────────────────────────────────────────────────────


def test_the_plan_command_writes_the_owed_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pypi, "remote_tags", lambda: TAGS)
    monkeypatch.setattr(pypi, "index_files", Index())
    monkeypatch.setattr(pypi, "PYPI_FIRST_RELEASE", "v0.12.1")
    out = tmp_path / "plan.json"
    assert pypi.main(["plan", "--index", "pypi", "--json-out", str(out)]) == 0
    entries = json.loads(out.read_text())
    # Each entry is one leg of the workflow's matrix: strings, nothing else.
    assert entries == [
        {"tag": "v0.12.1", "version": "0.12.1", "commit": "c" * 40},
        {"tag": "v0.13.0", "version": "0.13.0", "commit": "d" * 40},
    ]
    assert "v0.12.1" in capsys.readouterr().out


def test_the_shipped_boundary_keeps_pypi_off_until_the_rehearsal() -> None:
    # #440 moves this to the rehearsed tag; until then a merge uploads nothing.
    assert pypi.PYPI_FIRST_RELEASE is None
