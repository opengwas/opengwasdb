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
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import zarr

from opengwasdb.model.manifest import normalise_coverage
from opengwasdb.store.arrays import STORE_ZARR_FORMAT, open_group
from opengwasdb.store_reader.alid_index import ALID_INDEX_FILENAMES, alid_index_kind
from opengwasdb.store_reader.alid_sidecar import AlidSidecar

# The version tables and the read check are the Store Reader's (#464); what a writer
# stamps and refuses builds on them. Re-exported: opengwasdb-stores probes
# CURRENT_FORMAT_VERSION here.
from opengwasdb.store_reader.format import CURRENT_FORMAT_VERSION as CURRENT_FORMAT_VERSION
from opengwasdb.store_reader.format import (
    WRITTEN_FORMAT_VERSIONS,
    UnsupportedFormatVersion,
    carries_alid_sidecar,
    check_format_version,
    format_version_for,
    indels_reference_first,
    parse_format_version,
    split_format_version,
    zarr_format_for_version,
)
from opengwasdb.store_reader.manifest import StoreManifest
from opengwasdb.store_reader.release import (
    ReleasePaths,
    StoreRelease,
    open_release,
    release_paths,
)
from opengwasdb.variants.legacy_table import open_legacy_variant_table

if TYPE_CHECKING:
    import sqlite3


#: The first `format_version` whose `analyses.tsv` must record
#: `n_missing_se_out_of_range` on every row (ADR 0068, issue #338). Below it the
#: column may be absent or blank, which reads as unknown, never 0.
SE_OUT_OF_RANGE_COUNTED_SINCE = (0, 2, 3)


def records_se_out_of_range(version: str) -> bool:
    """Whether a Store of ``version`` records `n_missing_se_out_of_range` on every
    Analysis, and so holds no association whose SE is above 65,504 (ADR 0068)."""
    return parse_format_version(version) >= SE_OUT_OF_RANGE_COUNTED_SINCE


def completed_format_version(source_version: str, *, carries_regions: bool) -> str:
    """The `format_version` a Reference Completion of a ``source_version`` release is
    stamped: its source's features, and the count every completion records.

    A completion writes into its source's arrays, so it keeps every feature its
    source's stamp declares (ADR 0038 §4), and adds `n_missing_se_out_of_range`,
    which it counts as it writes (issue #338), and the variable-length ALID sidecar,
    which `write_variant_axis` writes for the completed axis (#467): a 0.2.2 or 0.2.3
    source completes to 0.3.0.
    The source must already be one this build completes (`check_writable_format_version`).
    """
    check_format_version_stamp(source_version, carries_regions=carries_regions)
    return format_version_for(
        carries_regions=carries_regions,
        reference_first_indels=indels_reference_first(source_version),
        records_se_out_of_range=True,
    )


def check_format_version_stamp(
    version: str, *, carries_regions: bool, source: str = "release"
) -> str:
    """Return `version` if it is the stamp a release with these features gets
    (`format_version_for`); raise otherwise.

    A release stamped otherwise disagrees with itself: a 0.2.0 release carrying
    `regions.tsv` is one an older reader takes as fully understood, and a 0.2.1 one
    without it is not what any writer stamps. The stamp itself says whether the
    release's indels are spelt reference first (`indels_reference_first`), which
    `variant_table.indel_order` holds it to. Completion refuses a release stamped
    otherwise rather than re-stamp it, and `validate` fails it (spec §21.3).

    A version newer than every one this build writes (`newer_than_written`) is not
    judged: which stamp its features give is the rule of the package that writes it,
    and naming this build's stamp would send its owner to re-stamp it, the wrong fix,
    instead of upgrading (as package 0.9.0 tells a 0.2.2 release it "is 0.2.1").
    """
    if newer_than_written(version):
        raise UnsupportedFormatVersion(newer_than_written_text(version, source=source))
    stamp = format_version_for(
        carries_regions=carries_regions,
        reference_first_indels=indels_reference_first(version),
        records_se_out_of_range=records_se_out_of_range(version),
        alid_sidecar=carries_alid_sidecar(version),
    )
    if version != stamp:
        carried = "regions.tsv" if carries_regions else "no region table"
        raise UnsupportedFormatVersion(
            f"{source} is stamped {version}, but a release carrying {carried} is {stamp} "
            "(spec §21.3)"
        )
    return version


def newer_than_written(version: str) -> bool:
    """Whether `version` is newer than every `format_version` this build writes in its
    series: a newer remainder of a series it reads (ADR 0041 §2), which a later package
    writes. A 0.2.4 release is one, though this build writes 0.3.0: what 0.2.4 adds to
    the 0.2 series is that package's, not 0.3.0's."""
    newest = _newest_written_in_series(version)
    return newest is not None and parse_format_version(version) > parse_format_version(newest)


def _newest_written_in_series(version: str) -> str | None:
    series = split_format_version(version)[0]
    written = [v for v in WRITTEN_FORMAT_VERSIONS if split_format_version(v)[0] == series]
    return written[-1] if written else None


def newer_than_written_text(version: str, *, source: str = "release") -> str:
    """What this build says of a release newer than it writes: upgrade, not re-stamp."""
    return (
        f"{source} is stamped {version}, newer than any format_version this build writes "
        f"in its series ({_newest_written_in_series(version)}), so which stamp its features "
        "give is the rule of "
        "the package that wrote it. Upgrade opengwasdb to judge its stamp or to complete "
        "it; do not re-stamp the release (spec §21.3)."
    )


def check_writable_zarr_layout(version: str, *, source: str = "release") -> int:
    """Return `version`'s Zarr on-disk format, refusing one this build cannot write.

    A pass that adds arrays to a release's own data group has to write them in
    the format that group already is: a Zarr v3 array in a `0.1.0` release --
    Zarr v2 -- puts a `zarr.json` beside the release's `.zgroup`, which zarr 3
    then reads as the group's format, hiding every v2 array it has (issue #313).

    The series-to-layout rule is `zarr_format_for_version`'s and what this build
    writes is `STORE_ZARR_FORMAT`'s. It is deliberately not a comparison with
    `CURRENT_FORMAT_VERSION`: that constant moves *within* a series (#381, to
    `0.2.2`), and a release of the same series stays writable when it does.
    """
    layout = zarr_format_for_version(version, source=source)
    if layout != STORE_ZARR_FORMAT:
        raise UnsupportedFormatVersion(
            f"{source} declares format_version={version!r}, whose arrays are stored in "
            f"Zarr format {layout}, and this build writes Zarr format {STORE_ZARR_FORMAT}: "
            "arrays written here would sit beside metadata that says otherwise, hiding "
            "every array the release has. Convert the 0.1.0 release with "
            "scripts/convert_store_to_0_2_0.py and write to the converted release; or "
            "rebuild the release from source."
        )
    return layout


log = logging.getLogger(__name__)


def connect_index(path: Path) -> sqlite3.Connection:
    """An `index.sqlite` connection with named columns; `sqlite3` is imported here, on use,
    so opening a Store Release does not load it (#465).

    Read-write, for the writers (`index_connection`), so it creates an index that
    is not there. Every read of a release's index opens it read-only instead,
    through `opengwasdb.variants.legacy_table.open_index_read_only` (#473)."""
    import sqlite3

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


class DestinationExistsError(FileExistsError):
    """A staging/publication destination is already occupied.

    A ``FileExistsError`` subclass so every existing caller that catches
    ``FileExistsError`` (the release stager included) is unchanged, while a
    derived-artifact writer can tell "the destination exists" apart from an
    unrelated ``FileExistsError`` raised inside its staged body (#264 review).
    """


def check_writable_format_version(version: str, *, source: str = "release") -> str:
    """Return `version` if this build can write it; raise otherwise (ADR 0038 §4).

    Reference Completion writes into the source's arrays and therefore into the
    source's encoding, so a completed release keeps its source's
    `format_version` rather than being stamped with the current one. That is
    only honest while this build can actually write that format. A store on disk
    in a version this build does not write (`WRITTEN_FORMAT_VERSIONS`) would
    otherwise be stamped a version onto arrays that are not in it -- a store that
    lies about its own encoding, which is the failure class this project exists
    to avoid. 0.2.0, 0.2.1 and 0.2.2 are written by the same encoders (issue #430).

    Since #381 a source whose indels are sorted -- `0.2.1`, `0.2.0`, or a `0.1.0`
    Zarr v2 one -- is refused here: completing it would add panel ALIDs spelt
    reference first (ADR 0079) beside the source's own deletions spelt like
    insertions, one deletion under two ALIDs. Nothing in such a Store says which of
    its indels are deletions, so it is rebuilt, not converted (the converter keeps
    the sorted spelling), before it is completed.
    """
    check_format_version(version, source=source)
    if not indels_reference_first(version):
        # A completion would add panel ALIDs spelt by ADR 0079 to a Store whose
        # own deletions are spelt like insertions: one deletion, two ALIDs.
        raise UnsupportedFormatVersion(
            f"{source} is format_version={version!r}, which this build reads but does "
            f"not complete (it completes {CURRENT_FORMAT_VERSION!r} releases). A "
            f"{version} Store sorts an indel's alleles, so a deletion in it is spelt "
            "like the insertion of the same sequence; this build spells an indel "
            "reference first (ADR 0079) and cannot tell which of its source's indels "
            "are deletions. Rebuild the release from source with this build, then "
            "complete the rebuilt release."
        )
    if version not in WRITTEN_FORMAT_VERSIONS:
        raise UnsupportedFormatVersion(
            f"{source} is format_version={version!r}, which this build reads but cannot "
            f"write (it writes {', '.join(map(repr, WRITTEN_FORMAT_VERSIONS))}). "
            "Completion preserves its source's format rather than re-encoding it "
            "(ADR 0038 §4), so this release cannot be completed by this build."
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
        # rsid search index (issue #109). Every layout writes it, empty when the
        # source names no variants, so it belongs in the base envelope rather than
        # being optional per layout.
        "variant_rsid_bytes.npy",
        "variant_rsid_rows.npy",
    }
) | set(AlidSidecar.FILENAMES)

#: Dense Observed-Only/Reference-Completed (§10, §16).
DENSE_ENVELOPE: frozenset[str] = _BASE_ENVELOPE | {"overview.html"}

#: Ragged Observed-Only/Reference-Completed (§11, §17) -- no `overview.html`,
#: which is Dense/Hybrid-only today -- and, optionally, `regions.tsv`, the region
#: table its build recorded and completion fills inside (issue #430).
RAGGED_ENVELOPE: frozenset[str] = _BASE_ENVELOPE | {"regions.tsv"}

#: A Hybrid release's own top-level entries (§16 introduces its nested
#: `dense/` Dense Component directory). The Dense Component's *own*
#: top-level entries are `HYBRID_DENSE_COMPONENT_ENVELOPE`, not this one.
HYBRID_ENVELOPE: frozenset[str] = _BASE_ENVELOPE | {"overview.html", "dense"}

#: A Hybrid release's nested `<store>/dense` Dense Component: the ordinary
#: Dense envelope plus `dense_to_shared.npy`, the Dense-row -> shared-table
#: index map only a Dense Component (never a standalone Dense release)
#: carries.
HYBRID_DENSE_COMPONENT_ENVELOPE: frozenset[str] = DENSE_ENVELOPE | {"dense_to_shared.npy"}


def envelope_at(envelope: frozenset[str], format_version: str) -> frozenset[str]:
    """`envelope`, one of the constants above, as a release of `format_version` carries it:
    with the ALID index that version carries (`alid_index_kind`) and no other.

    The constants are what this build writes (0.3.0, the sidecar). A 0.2 release
    carries the fixed-width pair instead, and either one carrying the other's files
    fails its closed envelope (spec §1, §20).
    """
    return (envelope - ALID_INDEX_FILENAMES) | set(alid_index_kind(format_version).FILENAMES)


@contextmanager
def directory_lock(directory: str | Path) -> Iterator[None]:
    """Hold an advisory exclusive ``flock`` on a directory's own inode.

    The lock primitive the publication and removal seams share.  The lock is
    the directory inode itself, flocked through an ``os.open`` handle, rather
    than a lock file.  A lock file has to live somewhere, and both obvious
    places fail: beside the destination it is renamed into the published
    release when that destination is a Hybrid release's nested Dense Component
    (staged at ``<outer-work>/dense``), and under the system temp directory its
    identity moves with ``TMPDIR`` and private temp namespaces -- two callers
    that do not share a temp root would not contend -- while a tmp cleaner can
    unlink it mid-hold, dropping the lock without any process noticing.  A
    directory inode is a stable filesystem identity, exists by the time a
    commit runs, and creates no entry that could be published.

    The cost is granularity: every destination in one directory shares this
    lock, so commits to different names in the same directory are serialised
    for the duration of a commit, including deletion of a replaced one.

    Advisory and host-local: ``flock`` is enforced by the local kernel and is
    released when the holding process dies, so a crashed writer cannot leave a
    destination permanently locked.  Whether it also serialises processes on
    another host depends on the filesystem -- NFS and other network
    filesystems may not propagate ``flock``, so this is not a cross-host lock.

    Two requirements follow, and both fail loudly rather than degrading: the
    directory must be openable for reading (``os.open(O_RDONLY)`` needs
    read/search permission on it), and the filesystem must implement ``flock``
    on directories.
    """
    fd = os.open(Path(directory), os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


@contextmanager
def destination_lock(dst: str | Path) -> Iterator[None]:
    """Serialise writes that publish or remove ``dst``, by locking its parent.

    The lock is held for the publication/removal window only, never the build.

    A filesystem that refuses ``flock`` outright fails the write loudly rather
    than proceeding unserialised.  Each rename inside a critical section is
    still atomic by the filesystem's own guarantee.
    """
    with directory_lock(Path(dst).parent):
        yield


def _adopt_staging_work_dir(work: Path, source: Path) -> None:
    """Replace this invocation's fresh work dir with a retained one.

    ``staging(adopt=...)`` is how a resumed build keeps writing the release a
    previous invocation left behind: the empty directory ``_new_staging_work_dir``
    just created exists only to be replaced, and the retained release is
    renamed into its place so the commit at the end of the context publishes it
    exactly as it publishes any other release. The rename requires one
    filesystem, so the retained release must be a sibling of the destination --
    which is where a checkpoint lives.
    """
    if not source.exists():
        raise FileNotFoundError(f"staging(adopt=...) was given {source}, which does not exist")
    shutil.rmtree(work)
    source.rename(work)


def _discard_staging_work_dir(work: Path, retain_to: Path | None) -> None:
    """Remove this invocation's work dir, or move it out of the way instead.

    ``staging(retain_on_failure_to=...)`` is how a build that crashed keeps the
    release it had written: the caller gets the partial release at ``retain_to``
    to resume from rather than losing it. The move is a rename, so it needs one
    filesystem. A failure to retain is reported and the work directory is
    removed: the exception being handled is the one worth propagating, and a
    silent no-op would leave a build that believes it has a checkpoint.
    """
    if retain_to is None:
        shutil.rmtree(work, ignore_errors=True)
        return
    if not work.exists():
        return
    try:
        if retain_to.exists():
            shutil.rmtree(retain_to)
        retain_to.parent.mkdir(parents=True, exist_ok=True)
        work.rename(retain_to)
    except OSError:
        log.warning("Could not retain staged release at %s; removing %s", retain_to, work)
        shutil.rmtree(work, ignore_errors=True)


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


def _commit_staged_release(
    dst: Path, work: Path, *, overwrite: bool, what: str = "release"
) -> None:
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
    with destination_lock(dst):
        if dst.exists() and not overwrite:
            raise DestinationExistsError(
                f"output path already exists: {dst} "
                f"(another process published a {what} there while this one was staging)"
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


@contextmanager
def _staged_directory(
    dst: Path, *, overwrite: bool, retain_to: Path | None, what: str
) -> Iterator[Path]:
    """The isolation/publication window both staging seams share (ADR 0043).

    Creates a unique ``.{name}.tmp.*`` sibling of ``dst``, yields its path, and
    on a clean exit commits it under the parent-directory lock.  Any
    ``BaseException`` -- from the body or from the commit -- discards this
    invocation's work directory (moving it to `retain_to` when a checkpointed
    build asked for that) and re-raises.
    """
    if dst.exists() and not overwrite:
        raise DestinationExistsError(f"output path already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    work = _new_staging_work_dir(dst)
    try:
        yield work
    except BaseException:
        _discard_staging_work_dir(work, retain_to)
        raise
    try:
        _commit_staged_release(dst, work, overwrite=overwrite, what=what)
    except BaseException:
        # `work` is still there exactly when the swap did not complete; if it
        # did, it has been renamed to `dst` and this is a no-op.
        _discard_staging_work_dir(work, retain_to)
        raise


@dataclass(frozen=True)
class OpenGWASDBStore(StoreRelease):
    """A local Store Release opened from an explicit path.

    The Store Reader's read-only `StoreRelease` (#464), with what writes to a
    release added: its arrays opened in any mode, its ``index.sqlite``,
    provenance amendment and atomic staging. `open_store` opens one with
    opengwasdb's reader of the 0.2.x Store Variant Table
    (`open_legacy_variant_table`).

    Frozen: the manifest a caller holds never changes under them. In-place
    provenance amendment (``amend_provenance``) rebinds it to a fresh
    instance rather than mutating this one.
    """

    # ---- opening -----------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path) -> OpenGWASDBStore:
        """Alias for the module-level :func:`open_store`."""
        return open_store(path)

    def dense_component(self) -> OpenGWASDBStore:
        """`StoreRelease.dense_component`, opened as an `OpenGWASDBStore`."""
        self._require_hybrid()
        return open_store(self.path / "dense")

    # ---- array / table access -----------------------------------------

    def arrays(self, mode: str = "r") -> zarr.Group:
        """Open this release's ``data.zarr`` group."""
        return open_group(self.data_path, mode)

    def index_connection(self) -> sqlite3.Connection:
        """Open a connection to this release's ``index.sqlite``."""
        return connect_index(self.index_path)

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
        # The amendment rewrites the manifest as JSON, so the coverage it leaves
        # behind is whatever the source said. Editing a published release's
        # metadata to the written vocabulary is the one in-place edit the
        # pre-release rule allows (ADR 0064 §5), and it keeps `full` from
        # surviving in a manifest this build just wrote (#298).
        normalise_coverage(data)
        self.manifest_path.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return OpenGWASDBStore(
            path=self.path, manifest=StoreManifest.from_dict(data), legacy=self.legacy
        )

    # ---- staging / atomic construction ----------------------------------

    @staticmethod
    @contextmanager
    def staging(
        dest_path: str | Path,
        *,
        overwrite: bool = False,
        adopt: Path | None = None,
        retain_on_failure_to: Path | None = None,
    ) -> Iterator[StagedRelease]:
        """Construct a release atomically at ``dest_path``.

        Each invocation writes into its **own** unique ``.{name}.tmp.*``
        sibling, so two builds for one destination -- or a second build after
        a crash -- can never delete or corrupt each other's work. Publication
        happens on successful exit from the ``with`` block, under a
        parent-directory advisory lock that re-checks the destination: a
        no-overwrite build that loses a race with a concurrent publisher
        fails loudly rather than replacing it.

        Replacing an existing release (``overwrite=True``) is two renames
        under that lock -- the old release aside, the new one in -- so
        ``dest_path`` is never absent while a large store is deleted
        (`_commit_staged_release` has the rest). It is last writer wins; an
        orchestrator that must not schedule two builds for one destination is
        complementary to this context, not interchangeable with it (ADR 0043).

        On error or interruption -- any ``BaseException``, including
        ``KeyboardInterrupt`` and ``SystemExit`` -- this invocation's own work
        directory is discarded and ``dest_path`` is left as it was found.

        The two opt-in parameters serve a checkpointed build (issue #227)
        without breaking that atomicity: ``adopt`` is the retained work
        directory a resumed build carries on writing, and
        ``retain_on_failure_to`` is where a failure moves this invocation's own
        instead of deleting it. Both are renames, so the path must be on this
        filesystem; each defaults to ``None``, preserving every existing
        caller's behaviour.
        """
        dst = Path(dest_path)
        with _staged_directory(
            dst, overwrite=overwrite, retain_to=retain_on_failure_to, what="release"
        ) as work:
            if adopt is not None:
                _adopt_staging_work_dir(work, Path(adopt))
            yield StagedRelease(work)


@contextmanager
def staged_named_group(
    dest_path: str | Path, *, overwrite: bool = False
) -> Iterator[Path]:
    """Stage one named directory inside a published tree, atomically.

    The named-group companion of `OpenGWASDBStore.staging` (ADR 0043), for a
    derived artifact that lives *inside* an existing release rather than being
    a release of its own -- an Indexed Variant Subset under
    ``data.zarr/indexed_subsets/<name>`` (ADR 0061, #264).  It offers the same
    isolation and publication rules at the group level:

    * the body writes into a fresh, invocation-unique ``.{name}.tmp.*`` sibling
      created by `_new_staging_work_dir`, so two builds for one name can never
      touch each other's work and a killed build's directory is inert;
    * any ``BaseException`` discards exactly this invocation's work directory
      and re-raises, leaving an existing published group untouched;
    * publication runs under the parent-directory advisory lock, re-checks
      ``dest_path`` there, and swaps with the same two-rename sequence and
      rollback as a release commit -- so ``dest_path`` is never absent mid-swap
      and a failed replacement leaves the previous group exactly as it was.

    It yields the work directory's path; the caller has to create the contents
    (a Zarr group, for the derived indexes that use it).  ``dest_path``'s parent
    must already exist as the group the new name is published into.
    """
    with _staged_directory(
        Path(dest_path), overwrite=overwrite, retain_to=None, what="named group"
    ) as work:
        yield work


@dataclass(frozen=True)
class StagedRelease(ReleasePaths):
    """A release under construction in a ``.tmp`` staging directory.

    Offers the same path/array/table surface as an opened
    ``OpenGWASDBStore``, plus ``write_manifest`` -- staging directories have
    no manifest yet, so they are not themselves an ``OpenGWASDBStore`` until
    committed.
    """

    path: Path
    _paths: dict[str, Path] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_paths", release_paths(self.path))

    def arrays(self, mode: str = "w") -> zarr.Group:
        return open_group(self.data_path, mode)

    def index_connection(self) -> sqlite3.Connection:
        return connect_index(self.index_path)

    def write_manifest(self, manifest: StoreManifest) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def open_store(path: str | Path) -> OpenGWASDBStore:
    """Open a local Store Release directory.

    The function intentionally opens exactly the path supplied by the
    caller. Release discovery and default selection belong to a higher-level
    catalogue. Raises ``UnsupportedFormatVersion`` if the release declares a
    format_version this build cannot interpret (ADR 0038 §2). The manifest is
    read, and the version checked, by the Store Reader's `open_release`.
    """
    release = open_release(path)
    return OpenGWASDBStore(
        path=release.path, manifest=release.manifest, legacy=open_legacy_variant_table
    )
