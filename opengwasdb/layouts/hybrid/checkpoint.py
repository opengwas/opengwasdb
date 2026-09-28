"""Phase-granularity checkpoint and resume for the Hybrid build tail (issue #227).

A Hybrid build of a real release spends hours in Pass 2 and in the Dense band
write before it reaches the joint SE fit, and until now a failure in that fit
discarded all of it: ``_build_components`` removes its spill directory in a
``finally`` and the Staged Release context removes its work directory, both by
design. This module is the on-disk record that lets an opt-in build keep them.

**What is recorded, and when.** Everything a later phase reads and cannot cheaply
re-derive is written beside the destination as it is produced, under the same
``.{name}.checkpoint`` convention Reference Completion uses (ADR 0023): the Pass 2
spills themselves, each Analysis's declared-score dispositions, the
off-reference fold's completed columns, the post-Pass-2 axis (its key table,
``old_to_new`` and the shared ALID list), the orientation report, the encoding
plan, and the Dense band write's top-hit harvest. Each phase's completion is a
``<phase>.done`` marker file; the last marker present is the recorded phase.

**What is re-run.** The build tail -- Overflow CSR assembly, the frequency
plane, the joint SE fit, the Dense Component finish, the CSR flush and the
shared metadata -- carries no marker and re-runs wholesale when a crash landed
in it, from the retained spills and the frozen plan. That is why the Overflow
``.ovf`` plates are not consumed under ``--checkpoint``.

**Why the plan and the axis are frozen rather than re-measured.** The encoding
plan is measured from the data, and the Dense bands already on disk were written
under the plan as measured. A resume that re-measured it could choose a
different quantisation for the same cells -- silent, undetectable corruption of
a store that still validates. So `plan.json` is written *before* the first band
write and reloaded verbatim: `resume_hybrid_build` re-enters at the recorded
phase and never calls the measurement. The same holds for the post-Pass-2 axis,
which the Dense row indices and the Overflow plate indices are already keyed on.

**Input identity.** `build_params.json` records the manifest's SHA-256 and each
external input's path, size, mtime and SHA-256, and a resume refuses when any of
them changed: a checkpoint describes one build of one set of inputs, and resuming
it against different ones would produce a store belonging to neither.

**The cost.** The retained spills are the expensive part, and they are the point
-- see the per-cell ceiling asserted in ``tests/test_hybrid_checkpoint.py``. A
checkpoint directory is only ever removed on success or by ``overwrite=True``.

Import surface: the constants and readers/writers here are the checkpoint's
vocabulary; `_build_components` in ``build`` is its only writer.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.completion.checkpoint import checkpoint_dir_for, require_fresh_destination

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "PHASES",
    "RESUME_FUNCTION",
    "CheckpointState",
    "checkpoint_dir_for",
    "completed_phases",
    "input_identities",
    "input_identity",
    "load_npz",
    "mark_phase",
    "read_build_params",
    "read_json",
    "read_lines",
    "read_str_map",
    "record_plates",
    "require_fresh_destination",
    "require_intact_plates",
    "require_matching_params",
    "save_npz",
    "write_build_params",
    "write_json",
    "write_lines",
    "write_str_map",
]

#: The checkpoint record's own format, recorded in ``build_params.json`` beside
#: the build parameters a resumed run reloads (issue #202). A record written
#: without it, or under a different number, is refused before anything in it is
#: read: a resumed run may not reinterpret records its writer did not mean.
CHECKPOINT_FORMAT_VERSION = 1

#: The function a refusal tells an operator to call.
RESUME_FUNCTION = "resume_hybrid_build"

#: The phases Phase 1 records, in the order the build runs them -- Pass 2
#: routing, the off-reference fold, EAF orientation, the joint encoding plan,
#: and the Dense band write. The tail after them carries no marker: it re-runs
#: wholesale, because its inputs (the retained spills and the frozen plan) are
#: all on disk and its own products are the store's last writes.
PHASES: tuple[str, ...] = ("pass2", "fold", "orientation", "plan", "dense_bands")

BUILD_PARAMS = "build_params.json"
INFO_COUNTS = "info_counts.json"
PLATES = "plates.json"
ORIENTATION = "orientation.json"
PLAN = "plan.json"
HITS = "dense_hits.npz"
AXIS = "axis.npz"
AXIS_CANONICAL_RAW = "axis_canonical_raw.txt"
AXIS_PANEL = "axis_panel.txt"
AXIS_OFF_PANEL = "axis_off_panel.txt"
AXIS_SHARED = "axis_shared.txt"
PROVENANCE_SOURCE = "provenance_source.tsv"
PROVENANCE_RSID = "provenance_rsid.tsv"
SPILL_DIR = "spill"
STAGED_DIR = "staged"
FOLD_DIR = "fold"

#: Keys the parameter comparison ignores. `n_workers` is a pure runtime knob with
#: no effect on any computed value, so a resume may change it freely (ADR 0023);
#: `format_version` is the record's own format, checked separately by
#: `read_build_params`, and not something a caller requests.
_COMPARISON_EXCLUSIONS = ("n_workers", "format_version")


def _write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file, then rename)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Record one JSON document where a resume will look for it."""
    _write_text(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    """Read a document `write_json` wrote."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Record arrays atomically, so a crash never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, allow_pickle=False, **arrays)
    tmp.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Read a document `save_npz` wrote, into memory."""
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def write_lines(path: Path, lines: Iterable[str]) -> None:
    """Record an ordered string list, one per line, streamed and atomic."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.writelines(f"{line}\n" for line in lines)
    tmp.replace(path)


def read_lines(path: Path) -> list[str]:
    """Read a document `write_lines` wrote."""
    with path.open(encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def write_str_map(path: Path, mapping: Mapping[str, str | None]) -> None:
    """Record a ``{str: str|None}`` provenance map as a two-column TSV.

    Streamed row by row rather than packed into one array: the map for a
    genome-scale release holds tens of millions of entries, and materialising
    them as a single buffer would cost more than the spill it sits beside. An
    empty value is a ``None`` on read, which no ALID or rsid can spell.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.writelines(f"{key}\t{value or ''}\n" for key, value in mapping.items())
    tmp.replace(path)


def read_str_map(path: Path) -> dict[str, str | None]:
    """Read a document `write_str_map` wrote."""
    out: dict[str, str | None] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            key, _, value = line.rstrip("\n").partition("\t")
            out[key] = value or None
    return out


# ── build parameters and input identity ──────────────────────────────────────


def input_identity(path: str | Path) -> dict[str, Any]:
    """One external input's identity: where it is, how big, and what it hashes to.

    A checkpoint describes one build of one set of inputs. A manifest whose rows
    changed, or a reference panel rewritten in place, invalidates every measured
    value in the checkpoint, so the identity travels with the parameters and a
    resume compares it before reading anything.

    The SHA-256 is a single streamed pass -- the biggest input here is a variant
    reference read once per build, so the pass costs seconds, and a size-and-mtime
    check alone would not notice a rewrite that preserved both.
    """
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def input_identities(paths: Mapping[str, str | Path | None]) -> dict[str, Any]:
    """`input_identity` for each named input that this build actually has."""
    return {
        name: input_identity(path) for name, path in sorted(paths.items()) if path is not None
    }


def require_matching_params(recorded: Mapping[str, Any], requested: Mapping[str, Any]) -> None:
    """Refuse a resume whose requested parameters differ from the recorded ones.

    A parameter that differs changes what the phases already on disk were
    computed under, so this is a refusal and never a warning: a build that
    silently proceeded would keep records from one configuration and write the
    rest of the store under another. `_COMPARISON_EXCLUSIONS` are the exception,
    and the only ones: no computed value depends on them.
    """
    differing = [
        key
        for key in sorted(set(recorded) | set(requested))
        if key not in _COMPARISON_EXCLUSIONS and recorded.get(key) != requested.get(key)
    ]
    if differing:
        raise ValueError(
            "build parameters differ from the recorded checkpoint's; resume with the "
            f"parameters it was built under, or discard it with overwrite=True. "
            f"Differing: {', '.join(differing)}"
        )


# ── the checkpoint directory ─────────────────────────────────────────────────


def marker_path(checkpoint_dir: Path, phase: str) -> Path:
    """Where a phase's completion marker lives."""
    return checkpoint_dir / f"{phase}.done"


def mark_phase(checkpoint_dir: Path, phase: str) -> None:
    """Record that ``phase`` completed, after its product is on disk.

    Written last and atomically: a marker is a promise that everything the next
    phase reads exists in full, so it may never precede that.
    """
    _write_text(marker_path(checkpoint_dir, phase), f"{phase}\n")


def completed_phases(checkpoint_dir: Path) -> tuple[str, ...]:
    """The recorded phases, in build order.

    A set with a gap -- a later marker without an earlier one, which no build
    writes -- is refused rather than interpreted: it cannot come from a build
    this code ran, so the records below it are not trustworthy.
    """
    present = [phase for phase in PHASES if marker_path(checkpoint_dir, phase).exists()]
    if present and present != list(PHASES[: len(present)]):
        raise ValueError(
            f"checkpoint at {checkpoint_dir} has a torn phase record: {', '.join(present)}. "
            f"Its records are not a prefix of {', '.join(PHASES)}; it cannot be resumed."
        )
    return tuple(present)


def write_build_params(checkpoint_dir: Path, params: Mapping[str, Any]) -> None:
    """Write the parameters a resume reloads, with the record's format version."""
    payload = {"format_version": CHECKPOINT_FORMAT_VERSION, **params}
    write_json(checkpoint_dir / BUILD_PARAMS, payload)


def read_build_params(checkpoint_dir: Path) -> dict[str, Any]:
    """Reload a checkpoint's build parameters, refusing an unusable record.

    Absent, versionless and mismatched records are all refusals before anything
    else in the directory is read (issue #202): a resumed run writes into a store
    the original build configured, so it may not proceed on a guess about what
    that configuration was.
    """
    path = Path(checkpoint_dir) / BUILD_PARAMS
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_dir}: {BUILD_PARAMS} is absent. "
            f"{RESUME_FUNCTION}() resumes a build that wrote one; a build started "
            f"without --checkpoint writes none."
        )
    params: dict[str, Any] = read_json(path)
    recorded = params.get("format_version")
    if recorded is None:
        raise ValueError(
            f"The checkpoint at {checkpoint_dir} records no format version: its "
            f"{BUILD_PARAMS} predates this build, which cannot tell what its records "
            f"mean. Start the build again with --checkpoint."
        )
    if recorded != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"The checkpoint at {checkpoint_dir} records format version {recorded}, "
            f"which this build does not read (it writes {CHECKPOINT_FORMAT_VERSION}). "
            f"Start the build again with --checkpoint."
        )
    return params


# ── retained plates ──────────────────────────────────────────────────────────


def record_plates(checkpoint_dir: Path, names: Iterable[str]) -> None:
    """Record the spill plates the phases still to run will read, with sizes.

    The sizes are what makes a torn checkpoint detectable: a spill directory
    that lost or truncated a plate (a full disk, a stray cleanup, a killed
    process) would otherwise be read as if it held every association its owner
    wrote, and the resumed build would publish a store that is short in a way
    nothing checks. A checksum would be stronger and would cost a full pass over
    the spills on every resume; the size, recorded when the plate was final, is
    the cheap version of the same guard.
    """
    spill_dir = Path(checkpoint_dir) / SPILL_DIR
    sizes = {
        name: (spill_dir / name).stat().st_size
        for name in sorted(names)
        if (spill_dir / name).exists()
    }
    write_json(checkpoint_dir / PLATES, {"sizes": sizes})


def require_intact_plates(checkpoint_dir: Path, names: Iterable[str]) -> None:
    """Refuse to resume when a plate the resumed phases read is gone or truncated.

    Only plates the inventory recorded are checked: a name absent from it was
    not on disk when the inventory was taken, which is the ordinary case for a
    column that spilled no off-reference key (no side file) or for a plate a
    later phase creates. A recorded plate is one a phase already reads, so its
    absence -- or a size the writer could not have produced -- means the
    spills are not the ones this checkpoint was written against.

    An absent inventory is itself a refusal: every completed phase writes one,
    so a checkpoint with phases and no inventory has had part of its record
    removed, and the plates it depends on cannot be checked at all.
    """
    inventory = Path(checkpoint_dir) / PLATES
    if not inventory.exists():
        raise ValueError(
            f"The checkpoint at {checkpoint_dir} records no plate inventory, so the "
            f"spills it retains cannot be checked. Start the build again with "
            f"--checkpoint."
        )
    recorded = read_json(inventory).get("sizes", {})
    spill_dir = Path(checkpoint_dir) / SPILL_DIR
    for name in sorted(names):
        expected = recorded.get(name)
        if expected is None:
            continue
        path = spill_dir / name
        if not path.exists() or path.stat().st_size != expected:
            raise ValueError(
                f"The checkpoint at {checkpoint_dir} is missing or has a truncated "
                f"{name} (expected {expected} bytes): the retained spills are not the "
                f"ones this checkpoint was written against, so resuming would silently "
                f"drop associations. Start the build again with --checkpoint."
            )


# ── the state a resume re-enters from ────────────────────────────────────────


@dataclass(frozen=True)
class CheckpointState:
    """A checkpoint directory and the phases it has recorded as complete.

    Read once, before the build begins, so every phase sees one answer to
    "what is already on disk" rather than each re-reading the markers.
    """

    path: Path
    params: dict[str, Any]
    completed: tuple[str, ...]

    @property
    def spill_dir(self) -> Path:
        return self.path / SPILL_DIR

    @property
    def staged_dir(self) -> Path:
        return self.path / STAGED_DIR

    def has(self, phase: str) -> bool:
        """Whether ``phase`` has a completion marker."""
        return phase in self.completed

    def discard(self) -> None:
        """Release the checkpoint: the spills and the retained release with it."""
        shutil.rmtree(self.path, ignore_errors=True)
