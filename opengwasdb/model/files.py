"""Replacing a Store Release's small text files in place: the one durable, atomic write.

A release's `manifest.json` and `analyses.tsv` are rewritten in place by the writes spec
§21.4 allows (a provenance amendment, the Clumps' manifest entry and count columns, the
variant-centric index's record). Each goes through `replace_durably`, so a reader sees the
old file or the new one, never part of either, the file keeps the mode and group it had, and
a crash after the call leaves the new file on disk. `write_atomically` is the same write for
a file that may not exist yet: `overview.html`, which a build writes and an in-place step
rewrites (#509).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


def replace_durably(path: Path, text: str) -> None:
    """Replace the file at `path` with `text` atomically and durably.

    The text is written beside it, exactly as given (no newline translation), flushed to
    disk, given the old file's mode and, where the process may set it, its group (a rename
    puts a new inode in its place), and renamed over it; the directory is flushed after.
    """
    _write(path, text, path.stat())


def write_atomically(path: Path, text: str) -> None:
    """`replace_durably` where `path` exists; otherwise the same staged, flushed rename,
    the new file taking the process's default mode. A reader sees no file, the old one or
    the new one, never part of either, and a process killed part-way leaves no half-written
    `path`."""
    _write(path, text, path.stat() if path.exists() else None)


def _write(path: Path, text: str, old: os.stat_result | None) -> None:
    staged = path.with_name(f".{path.name}.writing")
    with open(staged, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    if old is not None:
        os.chmod(staged, stat.S_IMODE(old.st_mode))
        if old.st_gid != os.stat(staged).st_gid and (
            os.geteuid() == 0 or old.st_gid in os.getgroups()
        ):
            os.chown(staged, -1, old.st_gid)
    os.replace(staged, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
