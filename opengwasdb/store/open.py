"""The Store Release envelope: paths, manifest, and atomic construction.

``OpenGWASDBStore`` owns the physical shape of a Store Release on disk --
where its artifacts live, how its manifest is read and amended, and how a
release is staged and committed atomically. It does not own the contents of
those artifacts: callers still work with ``zarr.Group`` and
``sqlite3.Connection`` objects for array/table access.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import zarr

from opengwasdb.model.enums import PrimaryStorageLayout
from opengwasdb.model.manifest import StoreManifest

if TYPE_CHECKING:
    from opengwasdb.query.facade import StoreQuery

#: The highest compatible remainder this build fully understands, per readable
#: **release series** (ADR 0041, spec §21). A series absent from this mapping is
#: rejected; a remainder above the one recorded here is read with a warning,
#: because the compatible axis is defined as one an older reader still reads
#: correctly.
#:
#: One entry, and that is the point of the reset (issue #143): there is one
#: format, one decoder, and one contract to test. The four pre-reset versions
#: are not in it and are not readable -- see `PRE_RESET_FORMAT_VERSIONS`.
SUPPORTED_FORMAT_VERSIONS: Mapping[tuple[int, ...], tuple[int, ...]] = MappingProxyType(
    {(0, 1): (0,)}
)

#: format_version stamped on releases written by this build. A build writes
#: exactly one version and reads every one in `SUPPORTED_FORMAT_VERSIONS`
#: (ADR 0041 §3): supporting the *writing* of historical formats would mean
#: keeping every retired encoder alive and tested, for a use case nobody has.
CURRENT_FORMAT_VERSION = "0.1.0"

#: The versions the format carried before the reset, and what each one was.
#: Every one is two-component, so the parser rejects it on shape alone; naming
#: them here is what turns that rejection into an instruction rather than a
#: complaint about punctuation (issue #143). No store in any of them was ever
#: published outside the core team, and this build decodes none of them: the
#: bytes of a `3.0` release are the bytes of a `0.1.0` one, but `0.1`, `1.0`
#: and `2.0` are genuinely different encodings and reading them was deleted
#: rather than deprecated.
PRE_RESET_FORMAT_VERSIONS: Mapping[str, str] = MappingProxyType(
    {
        "0.1": "the first pre-release format, float16 throughout",
        "1.0": "fixed-point z (issue #114)",
        "2.0": "residual-coded eaf (issue #116)",
        "3.0": "residual-coded se (issue #118)",
    }
)

log = logging.getLogger(__name__)


class UnsupportedFormatVersion(Exception):
    """A release declares a format_version this build cannot interpret."""


class MalformedFormatVersion(UnsupportedFormatVersion):
    """A release's format_version is not ``MAJOR.MINOR.PATCH``.

    A subclass rather than a `ValueError`: to a caller deciding whether it can
    read a release, "the version is nonsense" and "the version is from the
    future" are the same answer, and both must stop the read.
    """


def parse_format_version(version: str) -> tuple[int, int, int]:
    """``"0.1.0"`` -> ``(0, 1, 0)``. Raises `MalformedFormatVersion` otherwise.

    Three components, not two, and that shape change is the safety mechanism of
    the reset rather than a cosmetic choice (issue #143, ADR 0041). Pre-reset
    releases are stamped ``0.1``, and a reset that reused that string would
    give two different formats one name -- a store that reads as plausible and
    is wrong. A two-component value has no reading here at all, so every such
    release is refused loudly instead.
    """
    text = str(version)
    was = PRE_RESET_FORMAT_VERSIONS.get(text)
    if was is not None:
        raise UnsupportedFormatVersion(
            f"format_version {text!r} is a pre-release format ({was}), which this "
            "build no longer reads: the format was reset to "
            f"{CURRENT_FORMAT_VERSION} and the decoders for every earlier version "
            "were deleted (ADR 0041, issue #143). Rebuild the release from source "
            "-- a store in one of those formats cannot be validated, completed or "
            "queried by this build"
        )
    parts = text.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise MalformedFormatVersion(
            f"format_version {text!r} is not MAJOR.MINOR.PATCH (ADR 0041, spec §21)"
        )
    major, minor, patch = (int(part) for part in parts)
    return major, minor, patch


def split_format_version(version: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``"0.1.0"`` -> ``((0, 1), (0,))``: a version's series and its remainder.

    The **series** is the part a compatible change may not move, and the
    **remainder** is the part it may. Which is which follows semantic
    versioning's own rule, that the *leftmost non-zero component is the
    breaking one*: below ``1.0.0`` a format is declaring itself unsettled, so
    ``MINOR`` carries an incompatible change and ``PATCH`` does not; from
    ``1.0.0`` it is ``MAJOR``, and ``MINOR`` joins the remainder.

    Both halves come from one function because they are one rule. A caller that
    took the series here and worked out the remainder for itself would be
    stating that rule a second time, and the two would eventually disagree --
    `check_format_version` compares what this returns and knows nothing about
    which component is which.
    """
    major, minor, patch = parse_format_version(version)
    if major == 0:
        return (0, minor), (patch,)
    return (major,), (minor, patch)


def check_format_version(version: str, *, source: str = "release") -> None:
    """Reject a `format_version` this build cannot interpret (ADR 0041 §2).

    Rejection is on the **release series** alone: a build carries the series it
    implements, not a list of every version it has ever seen. A *newer
    remainder* within a known series is accepted and warned about -- accepting
    it is the definition of a compatible change, and the warning is how a
    mis-classified one (a change that should have moved the series) becomes
    visible instead of silently returning partial data.
    """
    series, remainder = split_format_version(version)
    known = SUPPORTED_FORMAT_VERSIONS.get(series)
    if known is None:
        readable = [_series_text(known_series) for known_series in SUPPORTED_FORMAT_VERSIONS]
        raise UnsupportedFormatVersion(
            f"{source} declares format_version={version!r}, a release series this "
            f"build does not implement; readable series: {sorted(readable)}"
        )
    if remainder > known:
        log.warning(
            "%s declares format_version=%s, a newer compatible version than this build "
            "knows (%s). Reading it as that: anything added after it is not visible "
            "here, and if any of it changes how existing arrays decode it was "
            "misclassified as compatible.",
            source,
            version,
            _series_text(series) + "." + ".".join(str(part) for part in known),
        )


def _series_text(series: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in series)


def check_writable_format_version(version: str, *, source: str = "release") -> str:
    """Return `version` if this build can write it; raise otherwise (ADR 0038 §4).

    Reference Completion writes into the source's arrays and therefore into the
    source's encoding, so a completed release keeps its source's
    `format_version` rather than being stamped with the current one. That is
    only honest while this build can actually write that format. Once
    `CURRENT_FORMAT_VERSION` moves ahead of a store on disk, completing it would
    otherwise stamp a version onto arrays that are not in it -- a store that
    lies about its own encoding, which is the failure class this project exists
    to avoid.

    Unreachable while this build reads exactly one format (issue #143): every
    readable version is the one it writes. It is kept because the invariant is
    about the *next* format rather than this one -- the moment a second
    readable version exists, completion writing into arrays it cannot encode is
    live again, and that is not a check to be remembering to add at the time
    (issue #112).
    """
    check_format_version(version, source=source)
    if version != CURRENT_FORMAT_VERSION:
        raise UnsupportedFormatVersion(
            f"{source} is format_version={version!r}, which this build reads but cannot "
            f"write (it writes {CURRENT_FORMAT_VERSION!r}). Completion preserves its "
            "source's format rather than re-encoding it (ADR 0038 §4), so this release "
            "cannot be completed by this build -- rebuild it from source instead"
        )
    return version


# --- Store Release envelope (store-format spec §1/§10/§11/§16/§17) --------
#
# The closed set of top-level entries each Primary Storage Layout
# legitimately produces (issue #80): `opengwasdb.validation.validate` rejects
# a release carrying anything beyond this, the same way it was already
# checking that required entries are present but never that unexpected ones
# are absent -- the gap that let a stray Ragged `traits.tsv.gz` side-file
# (retired in issue #69) go unnoticed until a human spotted it (issues
# #69-#73 / PR #78). Reference Completion (§12) does not currently add any
# new top-level entry for any layout -- it only adds arrays inside
# `data.zarr/` and a `completion_quality` table inside `index.sqlite` -- so
# one set per layout covers both Observed-Only and Reference-Completed
# releases.
_BASE_ENVELOPE: frozenset[str] = frozenset(
    {
        "manifest.json",
        "index.sqlite",
        "analyses.tsv",
        "data.zarr",
        "variants.tsv.gz",
        "variants.tsv.gz.tbi",
        "variant_offsets.npy",
        "variant_alid_bytes.npy",
        "variant_alid_rows.npy",
        # rsid search index (issue #109). Every layout writes it, empty when the
        # source names no variants, so it belongs in the base envelope rather than
        # being optional per layout.
        "variant_rsid_bytes.npy",
        "variant_rsid_rows.npy",
    }
)

#: Dense Observed-Only/Reference-Completed (§10, §16).
DENSE_ENVELOPE: frozenset[str] = _BASE_ENVELOPE | {"overview.html"}

#: Ragged Observed-Only/Reference-Completed (§11, §17) -- no `overview.html`,
#: which is Dense/Hybrid-only today.
RAGGED_ENVELOPE: frozenset[str] = _BASE_ENVELOPE

#: A Hybrid release's own top-level entries (§16 introduces its nested
#: `dense/` Dense Component directory). The Dense Component's *own*
#: top-level entries are `HYBRID_DENSE_COMPONENT_ENVELOPE`, not this one.
HYBRID_ENVELOPE: frozenset[str] = _BASE_ENVELOPE | {"overview.html", "dense"}

#: A Hybrid release's nested `<store>/dense` Dense Component: the ordinary
#: Dense envelope plus `dense_to_shared.npy`, the Dense-row -> shared-table
#: index map only a Dense Component (never a standalone Dense release)
#: carries.
HYBRID_DENSE_COMPONENT_ENVELOPE: frozenset[str] = DENSE_ENVELOPE | {"dense_to_shared.npy"}


def _release_paths(path: Path) -> dict[str, Path]:
    return {
        "manifest": path / "manifest.json",
        "data": path / "data.zarr",
        "index": path / "index.sqlite",
        "analyses": path / "analyses.tsv",
        "variant_table": path / "variants.tsv.gz",
        "variant_tabix": path / "variants.tsv.gz.tbi",
        "variant_offsets": path / "variant_offsets.npy",
    }


class _ReleasePaths:
    """Artifact-path properties shared by an opened release and a staged one.

    Both ``OpenGWASDBStore`` and ``StagedRelease`` set ``_paths`` (via
    ``_release_paths()``) in ``__post_init__`` and inherit these instead of
    each re-deriving the same suffixes off ``self.path``.
    """

    _paths: dict[str, Path]

    @property
    def manifest_path(self) -> Path:
        return self._paths["manifest"]

    @property
    def data_path(self) -> Path:
        return self._paths["data"]

    @property
    def index_path(self) -> Path:
        return self._paths["index"]

    @property
    def analyses_path(self) -> Path:
        return self._paths["analyses"]

    @property
    def variant_table_path(self) -> Path:
        return self._paths["variant_table"]

    @property
    def variant_tabix_path(self) -> Path:
        return self._paths["variant_tabix"]

    @property
    def variant_offsets_path(self) -> Path:
        return self._paths["variant_offsets"]


@contextmanager
def _destination_lock(dst: Path) -> Iterator[None]:
    """Serialise publications of releases into ``dst``'s parent directory.

    The lock is the parent directory's own inode, flocked through an
    ``os.open`` handle, rather than a lock file. A lock file has to live
    somewhere, and both obvious places fail: beside the destination it is
    renamed into the published release when that destination is a Hybrid
    release's nested Dense Component (staged at ``<outer-work>/dense``), and
    under the system temp directory its identity moves with ``TMPDIR`` and
    private temp namespaces -- two callers that do not share a temp root would
    not contend -- while a tmp cleaner can unlink it mid-hold, dropping the
    lock without any process noticing. A directory inode is a stable
    filesystem identity, exists by the time a commit runs, and creates no
    entry that could be published.

    The cost is granularity: every destination in one parent directory shares
    this lock, so commits to different releases in the same directory are
    serialised for the duration of a commit, including deletion of a replaced
    release. Commits happen once per build, so that is accepted in exchange
    for a lock that cannot be moved or reaped out from under it.

    Advisory and host-local: ``flock`` is enforced by the local kernel and is
    released when the holding process dies, so a crashed builder cannot leave
    a destination permanently locked. Whether it also serialises processes on
    another host depends on the filesystem -- NFS and other network
    filesystems may not propagate ``flock``, so this is not a cross-host lock.
    A filesystem that refuses ``flock`` outright fails the commit loudly rather
    than publishing unserialised. Each rename inside the critical section is
    still atomic by the filesystem's own guarantee.

    Two requirements follow, and both fail loudly rather than degrading: the
    parent directory must be openable for reading (``os.open(O_RDONLY)`` needs
    read/search permission on it, so a write-and-execute-only directory cannot
    be locked), and the filesystem must implement ``flock`` on directories.
    """
    fd = os.open(dst.parent, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _new_staging_work_dir(dst: Path) -> Path:
    """Create this invocation's own staging directory beside ``dst``.

    Unique, and never reused. A fixed ``.{name}.tmp`` was the defect this
    replaces: a second invocation for the same destination deleted the
    directory the first was still writing, and the first build then failed
    mid-run with ``variant_offsets.npy`` ENOENT. A name collision raises
    rather than being cleaned up, for the same reason -- deleting a directory
    is only safe when this call created it.

    ``mkdir`` rather than ``tempfile.mkdtemp`` so the published release keeps
    the umask-derived mode an ordinary staging ``mkdir`` gave it, instead of
    ``mkdtemp``'s private 0700.
    """
    token = f"{os.getpid()}.{uuid.uuid4().hex[:12]}"
    work = dst.with_name(f".{dst.name}.tmp.{token}")
    work.mkdir()
    return work


def _commit_staged_release(dst: Path, work: Path, *, overwrite: bool) -> None:
    """Publish the fully-written ``work`` directory at ``dst``, atomically.

    The destination is checked again *under the parent-directory lock* rather
    than trusting the check ``staging()`` made before the build began:
    another process may have published to ``dst`` while this one was writing,
    and a no-overwrite commit must fail loudly instead of clobbering it. That
    entry-time check is fail-fast only -- it cannot see a future publication,
    and it is not the thing that makes the decision.

    Replacement keeps the old release reachable throughout: it is renamed to a
    ``.{name}.old`` sibling, the new release is renamed in, and only then is
    the old one deleted. A failure between the two renames moves the old
    release back, so a failed commit leaves ``dst`` exactly as it was found
    rather than absent or half-swapped.

    ``overwrite=True`` is serialised by the lock but deliberately **last
    writer wins**: both builds commit successfully and the later one replaces
    the earlier, with no error and no record that a release was superseded.
    The lock only makes each replacement atomic and ordered; it does not decide
    which build *should* win, and it cannot stop two callers from doing
    redundant work. A higher-level orchestrator that must not schedule two
    builds for one destination, or that must reap a ``.{name}.tmp.*``
    directory orphaned by a killed build, is therefore complementary to this
    lock, not interchangeable with it (ADR 0043).
    """
    with _destination_lock(dst):
        if dst.exists() and not overwrite:
            raise FileExistsError(
                f"output path already exists: {dst} "
                "(published by another process while this release was staging)"
            )
        if dst.exists():
            old = dst.with_name(f".{dst.name}.old")
            if old.exists():
                shutil.rmtree(old)
            dst.rename(old)
            try:
                work.rename(dst)
            except BaseException:
                old.rename(dst)
                raise
            shutil.rmtree(old, ignore_errors=True)
        else:
            work.rename(dst)


@dataclass(frozen=True)
class OpenGWASDBStore(_ReleasePaths):
    """A local Store Release opened from an explicit path.

    Frozen: the manifest a caller holds never changes under them. In-place
    provenance amendment (``amend_provenance``) rebinds it to a fresh
    instance rather than mutating this one.
    """

    path: Path
    manifest: StoreManifest
    _paths: dict[str, Path] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_paths", _release_paths(self.path))

    # ---- opening -----------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path) -> OpenGWASDBStore:
        """Alias for the module-level :func:`open_store`."""
        return open_store(path)

    def dense_component(self) -> OpenGWASDBStore:
        """Open a Hybrid release's nested Dense Component release.

        The Dense Component (ADR 0026) is a self-contained Dense Store
        Release at ``<store>/dense``, built and completed by the unchanged
        dense machinery -- it is opened the same way any release is.
        """
        if self.manifest.primary_layout is not PrimaryStorageLayout.HYBRID:
            raise ValueError(
                "dense_component() is only valid for a Hybrid release "
                f"(primary_layout={self.manifest.primary_layout})"
            )
        return open_store(self.path / "dense")

    def query(self) -> StoreQuery:
        from opengwasdb.query.facade import query_store

        return query_store(self.path)

    # ---- array / table access -----------------------------------------

    def arrays(self, mode: str = "r") -> zarr.Group:
        """Open this release's ``data.zarr`` group."""
        return zarr.open_group(str(self.data_path), mode=mode)

    def index_connection(self) -> sqlite3.Connection:
        """Open a connection to this release's ``index.sqlite``."""
        conn = sqlite3.connect(str(self.index_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ---- provenance amendment -------------------------------------------

    def amend_provenance(self, patch: dict[str, Any]) -> OpenGWASDBStore:
        """Fold additional facts into ``manifest.json``'s provenance dict, in place.

        The one supported in-place write against a built release -- e.g.
        folding release-level Catalogue facts into a store's manifest after
        the fact (``ancestry/subset.py``), or a format migration updating its
        own provenance (``scripts/migrate_store_to_analyses_tsv.py``).
        Everything else that changes association data or metadata derives a
        new release via ``staging()`` instead.

        Returns a new ``OpenGWASDBStore`` bound to the amended manifest;
        this instance is left untouched.
        """
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        data["provenance"] = {**data.get("provenance", {}), **patch}
        self.manifest_path.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return OpenGWASDBStore(path=self.path, manifest=StoreManifest.from_dict(data))

    # ---- staging / atomic construction ----------------------------------

    @staticmethod
    @contextmanager
    def staging(dest_path: str | Path, *, overwrite: bool = False) -> Iterator[StagedRelease]:
        """Construct a release atomically at ``dest_path``.

        Each invocation writes into its **own** unique ``.{name}.tmp.*``
        sibling, so two builds for one destination -- or a second build after
        a crash -- can never delete or corrupt each other's work. Publication
        happens on successful exit from the ``with`` block, under a
        parent-directory advisory lock that also re-checks the destination: a
        no-overwrite build that loses a race with a concurrent publisher
        fails loudly rather than replacing it.

        Replacing an existing release (``overwrite=True``) is two renames --
        the old release out to a ``.{name}.old`` sibling, the new one in --
        rather than deleting the old release first: a directory can only be
        renamed onto an *empty* destination on POSIX, so a full ``rmtree``
        before the swap would leave no release at ``dest_path`` for however
        long the deletion of a large store takes. Renames are single
        filesystem operations, so that window shrinks to two syscalls, and if
        a crash lands between them the old release is still intact at its
        ``.old`` path rather than gone.

        ``overwrite=True`` is serialised by the lock but deliberately **last
        writer wins**: two concurrent overwrite builds both succeed and the
        later commit replaces the earlier. The lock orders and makes each
        replacement atomic; it does not prevent redundant work or decide which
        build should win, so an orchestrator that must not schedule two builds
        for one destination, or that must reap an orphaned ``.{name}.tmp.*``
        directory, is complementary to this context, not interchangeable with
        it (ADR 0043).

        On error or interruption -- any ``BaseException``, including
        ``KeyboardInterrupt`` and ``SystemExit`` -- this invocation's own work
        directory is discarded and ``dest_path`` is left as it was found.
        """
        dst = Path(dest_path)
        if dst.exists() and not overwrite:
            raise FileExistsError(f"output path already exists: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        work = _new_staging_work_dir(dst)

        staged = StagedRelease(work)
        try:
            yield staged
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise

        try:
            _commit_staged_release(dst, work, overwrite=overwrite)
        except BaseException:
            # `work` is still there exactly when the swap did not complete;
            # if it did, it has been renamed to `dst` and this is a no-op.
            shutil.rmtree(work, ignore_errors=True)
            raise


@dataclass(frozen=True)
class StagedRelease(_ReleasePaths):
    """A release under construction in a ``.tmp`` staging directory.

    Offers the same path/array/table surface as an opened
    ``OpenGWASDBStore``, plus ``write_manifest`` -- staging directories have
    no manifest yet, so they are not themselves an ``OpenGWASDBStore`` until
    committed.
    """

    path: Path
    _paths: dict[str, Path] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_paths", _release_paths(self.path))

    def arrays(self, mode: str = "w") -> zarr.Group:
        return zarr.open_group(str(self.data_path), mode=mode)

    def index_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.index_path))
        conn.row_factory = sqlite3.Row
        return conn

    def write_manifest(self, manifest: StoreManifest) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def open_store(path: str | Path) -> OpenGWASDBStore:
    """Open a local Store Release directory.

    The function intentionally opens exactly the path supplied by the
    caller. Release discovery and default selection belong to a higher-level
    catalogue. Raises ``UnsupportedFormatVersion`` if the release declares a
    format_version this build cannot interpret (ADR 0038 §2).
    """

    store_path = Path(path)
    manifest = StoreManifest.load(store_path)
    check_format_version(manifest.format_version, source=f"release at {store_path}")
    return OpenGWASDBStore(path=store_path, manifest=manifest)
