"""Per-LD-block completion checkpoints -- the on-disk record one process-pool
worker writes for one block, read back by the parent in Phase 3. Shared by
dense and ragged completion so a worker crash or a resumed run behaves
identically in both: each block's result lives at its own path, written
atomically, independent of every other block's.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ALID strings are short and fixed-format (chr:pos:a1:a2); a bounded numpy
# byte-string dtype avoids the overhead of an object array in a checkpoint
# that may hold millions of fill rows.
ALID_DTYPE = "S64"


def checkpoint_dir_for(dest_path: Path) -> Path:
    dest_path = Path(dest_path)
    return dest_path.parent / f".{dest_path.name}.checkpoint"


def require_fresh_destination(
    dst: Path, checkpoint_dir: Path, overwrite: bool, resume_fn: str
) -> None:
    """Raise unless *dst* and *checkpoint_dir* are clear to write into.

    Shared by `complete_dense_store` and `complete_ragged_store`, which faced
    off identically here before this was extracted: only the resume function
    named in the checkpoint's error message differed between them.
    """
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dst}. Use overwrite=True.")
    if checkpoint_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"A checkpoint directory already exists at {checkpoint_dir}. "
                f"Use {resume_fn}() to continue it, or overwrite=True to discard it."
            )
        shutil.rmtree(checkpoint_dir)


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
    q_ai = np.array([r.analysis_index for r in result.quality_rows], dtype=np.int32)
    q_pearson = np.array(
        [r.pearson_r if r.pearson_r is not None else np.nan for r in result.quality_rows],
        dtype=np.float64,
    )
    q_nimp = np.array([r.n_imputed for r in result.quality_rows], dtype=np.int32)
    q_nmiss = np.array([r.n_missing for r in result.quality_rows], dtype=np.int32)
    f_alid = np.array([r.alid.encode("ascii") for r in result.fills], dtype=ALID_DTYPE)
    f_ai = np.array([r.analysis_index for r in result.fills], dtype=np.int32)
    f_z = np.array([r.z for r in result.fills], dtype=np.float32)
    f_se = np.array([r.se for r in result.fills], dtype=np.float32)

    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as fh:
        np.savez(
            fh,
            block_id=np.array([result.block_id]),
            q_ai=q_ai, q_pearson=q_pearson, q_nimp=q_nimp, q_nmiss=q_nmiss,
            f_alid=f_alid, f_ai=f_ai, f_z=f_z, f_se=f_se,
        )
    os.replace(tmp_path, path)


def read_block_checkpoint(path: Path) -> BlockCompletionResult:
    with np.load(path, allow_pickle=False) as d:
        block_id = str(d["block_id"][0])
        quality_rows = [
            QualityRow(int(ai), None if not np.isfinite(p) else float(p), int(ni), int(nm))
            for ai, p, ni, nm in zip(
                d["q_ai"], d["q_pearson"], d["q_nimp"], d["q_nmiss"], strict=True
            )
        ]
        f_alid = d["f_alid"]
        alids = [a.decode("ascii") if isinstance(a, bytes) else str(a) for a in f_alid]
        fills = [
            FillRow(alid, int(ai), float(z), float(se))
            for alid, ai, z, se in zip(alids, d["f_ai"], d["f_z"], d["f_se"], strict=True)
        ]
    return BlockCompletionResult(block_id=block_id, quality_rows=quality_rows, fills=fills)
