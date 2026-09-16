"""Concurrency, interruption and commit contracts of ``OpenGWASDBStore.staging``.

The staging context is the only thing between a half-written build and a
published Store Release, and every guarantee it makes is about a situation
with more than one participant: a second invocation for the same destination,
an error mid-build, a ``KeyboardInterrupt``, or a failure during the commit
swap itself. These tests drive the real context manager rather than a build,
so the interleavings are exact and the suite stays fast.
"""

from __future__ import annotations

import errno
import multiprocessing
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import pytest

from opengwasdb.store.open import OpenGWASDBStore


def _staging_siblings(dst: Path) -> list[Path]:
    """Every staging work directory ``dst`` could have created beside it.

    The live pattern is ``.{name}.tmp.{pid}.{random}``; matching the ``.tmp``
    prefix also catches the fixed ``.{name}.tmp`` an older build left behind,
    so this asserts "no staging directory remains" for both.
    """
    return sorted(dst.parent.glob(f".{dst.name}.tmp*"))


def test_two_active_contexts_for_one_destination_do_not_share_a_work_directory(
    tmp_path: Path,
) -> None:
    """The regression: two staging contexts for one destination shared a fixed
    ``.{name}.tmp``, so the second deleted the directory the first was still
    writing. On a real build the first's next write failed with ENOENT on
    ``variant_offsets.npy`` -- a build that appeared to be running while its
    output directory was gone."""
    dst = tmp_path / "race.opengwasdb"
    with OpenGWASDBStore.staging(dst, overwrite=True) as first:
        first_artifact = first.path / "variant_offsets.npy"
        first_artifact.write_bytes(b"first")
        with OpenGWASDBStore.staging(dst, overwrite=True) as second:
            assert second.path != first.path, "each invocation owns its own directory"
            assert first.path.is_dir(), "the earlier active directory was removed"
            assert first_artifact.read_bytes() == b"first"
            (second.path / "variant_offsets.npy").write_bytes(b"second")
    # Both commits succeed, serialised; the outer context commits last and wins.
    assert (dst / "variant_offsets.npy").read_bytes() == b"first"
    assert _staging_siblings(dst) == []


def test_error_discards_only_the_failing_invocations_work_directory(tmp_path: Path) -> None:
    """A failed staging run must not take a concurrent run's work with it: the
    cleanup removes exactly the directory this invocation created."""
    dst = tmp_path / "release.opengwasdb"
    with OpenGWASDBStore.staging(dst, overwrite=True) as first:
        (first.path / "kept").write_bytes(b"kept")
        with pytest.raises(RuntimeError, match="boom"):
            with OpenGWASDBStore.staging(dst, overwrite=True) as second:
                (second.path / "discarded").write_bytes(b"discarded")
                raise RuntimeError("boom")
        assert (first.path / "kept").read_bytes() == b"kept"
    assert (dst / "kept").read_bytes() == b"kept"
    assert _staging_siblings(dst) == []


def test_a_second_no_overwrite_commit_fails_loudly_and_leaves_the_winner(
    tmp_path: Path,
) -> None:
    """Two no-overwrite builds may start while the destination is absent; the
    first to publish wins and the second is refused at commit time. The
    entry-time existence check is only fail-fast -- it cannot see a
    destination that is published while this build is still running."""
    dst = tmp_path / "dest.opengwasdb"
    with pytest.raises(FileExistsError, match="already exists"):
        with OpenGWASDBStore.staging(dst) as outer:
            outer_work = outer.path
            with OpenGWASDBStore.staging(dst) as inner:
                inner_work = inner.path
                (inner.path / "which").write_bytes(b"inner")
            assert dst.exists(), "the inner context must publish at its exit"
    assert (dst / "which").read_bytes() == b"inner"
    assert not outer_work.exists(), "the refused run cleans up after itself"
    assert not inner_work.exists(), "the committed work directory moved, not copied"
    assert _staging_siblings(dst) == []


def test_overwrite_publishes_atomically_and_keeps_the_destination_until_commit(
    tmp_path: Path,
) -> None:
    """Replacement is still two renames at exit: readers see the whole old
    release until the swap, and the whole new release afterwards."""
    dst = tmp_path / "release.opengwasdb"
    dst.mkdir()
    (dst / "content").write_bytes(b"old")
    with OpenGWASDBStore.staging(dst, overwrite=True) as staged:
        (staged.path / "content").write_bytes(b"new")
        assert (dst / "content").read_bytes() == b"old"
    assert (dst / "content").read_bytes() == b"new"
    assert not dst.with_name(f".{dst.name}.old").exists()
    assert _staging_siblings(dst) == []


def test_existing_destination_without_overwrite_is_refused_before_any_work(
    tmp_path: Path,
) -> None:
    dst = tmp_path / "release.opengwasdb"
    dst.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        with OpenGWASDBStore.staging(dst):
            pass
    assert _staging_siblings(dst) == []


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_interruption_discards_the_work_directory_and_leaves_the_destination(
    tmp_path: Path, interrupt: type[BaseException]
) -> None:
    """An interruption is a ``BaseException``: catching only ``Exception`` left
    the staging directory on disk. It must be removed without ever touching
    the destination, which still holds the previous release."""
    dst = tmp_path / "release.opengwasdb"
    dst.mkdir()
    (dst / "content").write_bytes(b"old")
    with pytest.raises(interrupt):
        with OpenGWASDBStore.staging(dst, overwrite=True) as staged:
            work = staged.path
            (work / "partial").write_bytes(b"partial")
            raise interrupt("interrupted")
    assert not work.exists()
    assert (dst / "content").read_bytes() == b"old"
    assert _staging_siblings(dst) == []


def test_a_failed_swap_restores_the_previous_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the new release cannot be moved into place, the old one is moved back
    rather than left stranded at ``.{name}.old``, and the work directory is
    discarded."""
    dst = tmp_path / "release.opengwasdb"
    dst.mkdir()
    (dst / "content").write_bytes(b"old")
    real_rename = Path.rename
    work_holder: list[Path] = []

    def flaky_rename(self: Path, target: Path) -> Path:
        if work_holder and self == work_holder[0]:
            raise OSError(errno.EIO, "simulated swap failure")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)
    with pytest.raises(OSError, match="simulated swap failure"):
        with OpenGWASDBStore.staging(dst, overwrite=True) as staged:
            work_holder.append(staged.path)
            (staged.path / "content").write_bytes(b"new")
    assert (dst / "content").read_bytes() == b"old"
    assert not work_holder[0].exists()
    assert not dst.with_name(f".{dst.name}.old").exists()
    assert _staging_siblings(dst) == []


def test_a_nested_destination_lock_is_not_published_with_the_outer_release(
    tmp_path: Path,
) -> None:
    """A Hybrid release's Dense Component is staged inside the outer staging
    directory, so a lock *file* beside it would be renamed into the published
    store. The lock is the destination's parent directory inode, which creates
    no entry at all; this asserts nothing of the sort rode along."""
    dst = tmp_path / "hybrid.opengwasdb"
    with OpenGWASDBStore.staging(dst) as outer:
        component = outer.path / "dense"
        component.mkdir()
        with OpenGWASDBStore.staging(component, overwrite=True) as inner:
            (inner.path / "content").write_bytes(b"component")
        assert (component / "content").read_bytes() == b"component"
    assert (dst / "dense" / "content").read_bytes() == b"component"
    # The outer release carries only its component: a lock file placed beside
    # the nested destination would have been renamed in as `dst/.dense.lock`.
    assert sorted(p.name for p in dst.iterdir()) == ["dense"]
    assert list(dst.rglob("*.lock")) == []


def _hold_destination_lock(
    dst: str, tmpdir: str, attempted: Any, entered: Any, release: Any
) -> None:
    """Child-process body: take the destination lock under a private temp root.

    The private root is the point: a lock keyed by the system temp directory
    would live somewhere this process's peers never look, so it would not
    contend at all. ``attempted`` is set immediately before the acquire, so the
    parent can tell "blocked on the lock" apart from "not scheduled yet". The
    private import keeps the helper-specific private surface out of this
    module's shared namespace.
    """
    os.environ["TMPDIR"] = tmpdir
    tempfile.tempdir = None
    from opengwasdb.store.open import _destination_lock

    attempted.set()
    with _destination_lock(Path(dst)):
        entered.set()
        release.wait(timeout=10)


def test_lock_contends_across_processes_with_different_temp_roots(
    tmp_path: Path,
) -> None:
    """A temp-directory-keyed lock silently fails to serialise callers in
    different ``TMPDIR``/private temp namespaces, and a tmp cleaner can unlink
    it mid-hold. These two processes have different temp roots and must still
    contend on the parent-directory lock."""
    context = multiprocessing.get_context("fork")
    dst = tmp_path / "shared" / "release.opengwasdb"
    dst.parent.mkdir()
    (tmp_path / "tmp-a").mkdir()
    (tmp_path / "tmp-b").mkdir()
    first_entered: Any = context.Event()
    second_attempted: Any = context.Event()
    second_entered: Any = context.Event()
    release: Any = context.Event()

    first = context.Process(
        target=_hold_destination_lock,
        args=(str(dst), str(tmp_path / "tmp-a"), context.Event(), first_entered, release),
    )
    second = context.Process(
        target=_hold_destination_lock,
        args=(
            str(dst),
            str(tmp_path / "tmp-b"),
            second_attempted,
            second_entered,
            release,
        ),
    )
    first.start()
    try:
        assert first_entered.wait(timeout=10), "the first process never took the lock"
        second.start()
        try:
            # Wait until the second process is *at* the acquire: without this,
            # a not-yet-scheduled process would also fail to be "entered" and
            # the block assertion would pass for the wrong reason.
            assert second_attempted.wait(timeout=10), "the second process never tried the lock"
            assert not second_entered.wait(timeout=0.5), (
                "the second process entered while the first held the destination lock"
            )
            release.set()
            assert second_entered.wait(timeout=10), (
                "the second process never acquired the released lock"
            )
        finally:
            release.set()
            second.join(timeout=10)
    finally:
        release.set()
        first.join(timeout=10)
    assert first.exitcode == 0
    assert second.exitcode == 0


def test_destination_lock_serialises_holders(tmp_path: Path) -> None:
    """The commit lock is what makes two same-destination publications
    sequential; without it the check-then-swap sequence is not atomic."""
    from opengwasdb.store.open import _destination_lock

    dst = tmp_path / "release.opengwasdb"
    holder_in = threading.Event()
    release_holder = threading.Event()
    waiter_attempted = threading.Event()
    waiter_in = threading.Event()

    def hold() -> None:
        with _destination_lock(dst):
            holder_in.set()
            release_holder.wait(timeout=10)

    def wait() -> None:
        waiter_attempted.set()
        with _destination_lock(dst):
            waiter_in.set()

    holder = threading.Thread(target=hold)
    waiter = threading.Thread(target=wait)
    holder.start()
    assert holder_in.wait(timeout=10), "the holder never acquired the lock"
    waiter.start()
    assert waiter_attempted.wait(timeout=10), "the waiter never attempted the lock"
    assert not waiter_in.is_set(), "the second holder entered while the first held the lock"
    release_holder.set()
    assert waiter_in.wait(timeout=10), "the waiter never acquired the released lock"
    holder.join(timeout=10)
    waiter.join(timeout=10)
    assert not holder.is_alive()
    assert not waiter.is_alive()
