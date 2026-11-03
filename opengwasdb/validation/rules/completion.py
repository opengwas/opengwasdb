"""Rules about a Reference-Completed component's imputed mask (ADR 0074).

The mask says which associations are imputed (spec §15). It holds only 0 and 1,
flags no cell whose Z or SE is missing, flags no off-panel row, and agrees with
what each Analysis declares about whether it holds imputed cells at all. The
mask and panel flags are checked for presence and shape first; the value rules
fold over the association scan's blocks, which already hold the mask.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from opengwasdb.associations import Tally
from opengwasdb.model.analyses import read_analyses
from opengwasdb.model.enums import CompletionState
from opengwasdb.validation.rule import Check, Kind, Rule, Target
from opengwasdb.validation.rules.associations import grid_counts, tally, value_fold
from opengwasdb.validation.rules.release import REQUIRED
from opengwasdb.validation.scans import Block

#: The rule every value rule on a completed grid runs after.
MASK = "completion.mask"


def completed(target: Target) -> bool:
    return target.store.manifest.completion_state is CompletionState.REFERENCE_COMPLETED


def mask(target: Target) -> Iterator[str]:
    """A completed grid carries an `imputed` mask of the grid's shape, and binary panel flags.

    `on_panel` is a 1-D array of about `n_variants` bytes, read once for the run.
    """
    group = target.group
    missing = [name for name in ("imputed", "on_panel") if name not in group]
    if missing:
        yield from (f"reference-completed store is missing data.zarr/{name}" for name in missing)
        return
    n_variants, n_analyses = grid_counts(target)
    shape = tuple(group["imputed"].shape)
    if shape != (n_variants, n_analyses):
        yield f"data.zarr/imputed shape {shape} does not match {(n_variants, n_analyses)}"
    on_panel = target.remember("on_panel", lambda: np.asarray(group["on_panel"][:]))
    if len(on_panel) != n_variants:
        yield f"data.zarr/on_panel has {len(on_panel)} entries but expected {n_variants}"
    if not np.all((on_panel == 0) | (on_panel == 1)):
        yield "data.zarr/on_panel contains values other than 0 and 1"


def _judge_binary(_target: Target, total: Tally) -> Iterator[str]:
    if total.flags_not_binary:
        yield "data.zarr/imputed contains values other than 0 and 1"


def _judge_present(_target: Target, total: Tally) -> Iterator[str]:
    """An imputed cell carries both Z and SE (spec §15, ADR 0070)."""
    if total.imputed_z_missing:
        yield "imputed=1 cells have missing z-scores"
    if total.imputed_se_missing:
        yield "imputed=1 cells have NaN se values"


def _off_panel_imputed(_target: Target, block: Block) -> int:
    if block.imputed is None or block.on_panel is None:
        return 0
    off_panel = block.on_panel == 0
    return int(np.count_nonzero(block.imputed[off_panel] == 1))


def _judge_off_panel(_target: Target, n: int) -> Iterator[str]:
    """Off-panel rows can never be imputed: they have no LD structure."""
    if n:
        yield "off-panel (on_panel=0) rows have imputed=1 cells — off-panel is never imputable"


def _imputed_per_analysis(_target: Target, block: Block) -> np.ndarray:
    if block.imputed is None:
        return np.zeros(block.n_analyses, dtype=np.int64)
    return block.per_analysis(block.imputed == 1)


def _judge_declaration(target: Target, total: np.ndarray) -> Iterator[str]:
    counts = np.zeros(grid_counts(target)[1], dtype=np.int64) + total
    errors: list[str] = []
    declaration_matches_mask(target.store.analyses_path, counts, errors)
    yield from errors


def declaration_matches_mask(
    analyses_path: Path, imputed_per_analysis: np.ndarray, errors: list[str]
) -> None:
    """Each Analysis holds imputed cells if and only if it says it does.

    Categorical, not a count comparison: `completion_n_imputed_total` is rolled up
    from per-block quality and the arrays hold what was actually written, so the
    two need not agree to the cell. What they may not do is disagree about
    *whether there are any* -- that is the shape the ancestry-match filter's
    defect took, an Analysis carried through observed-only while its metadata
    claimed thousands of imputed cells, and `eaf_scope` derived from the claim
    (ADR 0028, ADR 0037 §4).
    """
    for row in read_analyses(analyses_path).rows:
        try:
            index = int(row.get("analysis_index", ""))
            declared = int(row.get("completion_n_imputed_total", "") or 0)
        except ValueError:
            continue  # reported by the analyses rules
        if 0 <= index < len(imputed_per_analysis):
            message = _disagreement(row, declared, int(imputed_per_analysis[index]))
            if message is not None:
                errors.append(message)


def _disagreement(row: dict[str, str], declared: int, actual: int) -> str | None:
    """What one Analysis's declaration and mask disagree on, if they do."""
    analysis_id = row.get("analysis_id", "")
    if declared and not actual:
        return (
            f"analysis {analysis_id!r} declares completion_n_imputed_total={declared} "
            "but has no imputed=1 cell in data.zarr/imputed; the metadata describes "
            "cells the release does not contain"
        )
    if actual and not declared:
        return (
            f"analysis {analysis_id!r} has {actual} imputed=1 cell(s) in "
            "data.zarr/imputed but declares completion_n_imputed_total="
            f"{row.get('completion_n_imputed_total', '')!r}; the release holds "
            "imputed cells its metadata does not account for"
        )
    return None


CHECKS: tuple[Rule, ...] = (
    Check(name=MASK, kinds=Kind.GRID, when=completed, after=(REQUIRED,), run=mask),
)

#: The value rules about the mask, in the order they have always reported.
RULES: tuple[Rule, ...] = (
    value_fold("completion.mask_binary", Tally(), tally, _judge_binary, when=completed),
    value_fold("completion.imputed_present", Tally(), tally, _judge_present, when=completed),
    value_fold("completion.off_panel", 0, _off_panel_imputed, _judge_off_panel, when=completed),
    value_fold(
        "completion.declaration",
        np.zeros((), dtype=np.int64),
        _imputed_per_analysis,
        _judge_declaration,
        when=completed,
    ),
)
