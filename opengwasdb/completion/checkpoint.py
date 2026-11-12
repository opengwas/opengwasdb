"""Per-LD-block completion checkpoints -- the on-disk record one process-pool
worker writes for one block, read back by the parent in Phase 3. Shared by
dense and ragged completion so a worker crash or a resumed run behaves
identically in both: each block's result lives at its own path, written
atomically, independent of every other block's.
"""
from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

#: Most ALIDs are short (chr:pos:a1:a2), and a fixed-width byte string holds
#: millions of them compactly. An ALID of more bytes than this (a long indel,
#: ADR 0079) is never cut to fit: it is held whole beside them (`split_alids`),
#: because a width taken from the longest ALID would give every row of a
#: Store's axis that width (OGS-00013's longest is 761 bytes, its median 15).
ALID_DTYPE = "S64"
ALID_WIDTH = 64

#: The format a block checkpoint records (``format`` in its ``.npz``). A resume
#: refuses a directory holding a checkpoint without it, or under another number:
#: it may not reinterpret what its writer did not mean.
#:
#: **2** (ADR 0072): a block leaves the variant at its end that the next block
#: owns out of its fills and counts. A checkpoint written before (by a package
#: that wrote no format) fills and counts it, so resumed beside the next
#: block's it puts two fills on one cell and counts both. Nothing in it says
#: which rule wrote it, so it is refused, and the completion starts again.
#:
#: **3** (#361): a fill's ALID is held whole. Format 2 cut it to 64 bytes, so
#: a checkpoint of it that holds a longer ALID has already lost which variant
#: each such fill was for, and cannot be read right.
CHECKPOINT_FORMAT = 3


def checkpoint_dir_for(dest_path: Path) -> Path:
    dest_path = Path(dest_path)
    return dest_path.parent / f".{dest_path.name}.checkpoint"


def require_fresh_destination(
    dst: Path, checkpoint_dir: Path, overwrite: bool, resume_fn: str
) -> None:
    """Raise unless *dst* and *checkpoint_dir* are clear to write into.

    Shared by `complete_dense_store` and `complete_ragged_store`, which faced
    off identically here before this was extracted: only the resume function
    named in the checkpoint's error message differed between them; the Hybrid
    build passes `resume_hybrid_build` (issue #227).

    A checkpoint directory is never overwritten in silence, whichever build
    finds it, so the refusal names both ways out: the resume function, and the
    flag that discards it. `overwrite=True` is what that flag sets, so both
    spellings are given -- a checkpoint holds the only copy of a release a
    build had already written, which is not something to lose by guessing.
    """
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dst}. Use overwrite=True.")
    if checkpoint_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"A checkpoint directory already exists at {checkpoint_dir}. Use "
                f"{resume_fn}() to continue it, or --overwrite (overwrite=True) to "
                f"discard it."
            )
        shutil.rmtree(checkpoint_dir)


def checkpoint_blocks_dir(checkpoint_dir: Path) -> Path:
    """The directory a completion's block checkpoints live in, created, and
    the checkpoints a resumed run would keep checked (`require_current_checkpoints`)."""
    blocks_dir = Path(checkpoint_dir) / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    require_current_checkpoints(checkpoint_dir)
    return blocks_dir


def require_current_checkpoints(checkpoint_dir: Path) -> None:
    """Refuse to resume from block checkpoints another format wrote.

    Called before a resumed completion schedules its blocks, so neither the
    checkpoints nor the blocks still to run are touched. The refusal names the
    directory and both ways out, as `require_fresh_destination` does.
    """
    written = sorted((Path(checkpoint_dir) / "blocks").glob("*.npz"))
    stale = [path for path in written if _format_of(path) != CHECKPOINT_FORMAT]
    if stale:
        raise ValueError(
            f"The checkpoint directory {checkpoint_dir} holds {len(stale):,} of "
            f"{len(written):,} block checkpoint(s) not written in format "
            f"{CHECKPOINT_FORMAT} (first {stale[0].name}): an older package wrote them, "
            "before a variant two LD blocks share had one owner (ADR 0072) or before "
            "an ALID over 64 bytes was kept whole (#361), and they cannot be resumed. "
            "Restart the completion: delete it, or rerun the completion with "
            "--overwrite (overwrite=True) to discard it."
        )


def _format_of(path: Path) -> int | None:
    with np.load(path, allow_pickle=False) as checkpoint:
        if "format" not in checkpoint.files:
            return None
        return int(checkpoint["format"][0])


def sanitize_block_id(block_id: str) -> str:
    return block_id.replace("/", "__")


class QualityRow(NamedTuple):
    """One ``completion_quality`` row: one (LD block, Analysis) pair's
    imputation outcome. ``pearson_r`` is ``None`` when imputation was never
    attempted (see ``BlockAnalysisResult``); ``n_missing`` is however many of
    the block's positions are still unfilled for this Analysis."""

    analysis_index: int
    pearson_r: float | None
    n_imputed: int
    n_missing: int


class FillRow(NamedTuple):
    """One imputed cell: an Analysis's completed z/se at one ALID."""

    alid: str
    analysis_index: int
    z: float
    se: float


@dataclass(frozen=True)
class BlockCompletionResult:
    block_id: str
    quality_rows: list[QualityRow]
    fills: list[FillRow]


def write_block_checkpoint(path: Path, result: BlockCompletionResult) -> None:
    """Write one block's result to ``path`` atomically, in `CHECKPOINT_FORMAT`."""
    # Typed `Any` because `savez` takes `allow_pickle` beside the named arrays.
    arrays: dict[str, Any] = {
        "format": np.array([CHECKPOINT_FORMAT], dtype=np.int32),
        "block_id": np.array([result.block_id]),
        **_quality_arrays(result.quality_rows),
        **_fill_arrays(result.fills),
    }
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp_path, path)


def _quality_arrays(rows: list[QualityRow]) -> dict[str, np.ndarray]:
    return {
        "q_ai": np.array([r.analysis_index for r in rows], dtype=np.int32),
        "q_pearson": np.array(
            [r.pearson_r if r.pearson_r is not None else np.nan for r in rows], dtype=np.float64
        ),
        "q_nimp": np.array([r.n_imputed for r in rows], dtype=np.int32),
        "q_nmiss": np.array([r.n_missing for r in rows], dtype=np.int32),
    }


def split_alids(alids: Iterable[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``alids`` as ``(short, long_at, long)``, none of them cut.

    ``short`` holds each ALID of `ALID_WIDTH` bytes or fewer, and is empty at
    the position of a longer one; ``long_at`` is the positions of those, and
    ``long`` their ALIDs, as wide as the longest. ALIDs are UTF-8, as the
    Store's variant table writes them.
    """
    encoded = [alid.encode("utf-8") for alid in alids]
    at = [i for i, b in enumerate(encoded) if len(b) > ALID_WIDTH]
    short = np.array([b if len(b) <= ALID_WIDTH else b"" for b in encoded], dtype=ALID_DTYPE)
    long_ = np.array([encoded[i] for i in at], dtype="S1" if not at else None)
    return short, np.array(at, dtype=np.int64), long_


def _fill_arrays(fills: list[FillRow]) -> dict[str, np.ndarray]:
    short, long_at, long_ = split_alids(r.alid for r in fills)
    return {
        "f_alid": short,
        "f_long_at": long_at,
        "f_long_alid": long_,
        "f_ai": np.array([r.analysis_index for r in fills], dtype=np.int32),
        "f_z": np.array([r.z for r in fills], dtype=np.float32),
        "f_se": np.array([r.se for r in fills], dtype=np.float32),
    }
