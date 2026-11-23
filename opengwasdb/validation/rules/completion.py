"""Rules about a Reference-Completed component's imputed mask and metadata (ADR 0074).

The mask says which associations are imputed (spec §15). It holds only 0 and 1,
flags no cell whose Z or SE is missing, flags no off-panel row, and agrees with
what each Analysis declares about whether it holds imputed cells at all. The
mask, and a grid's panel flags, are checked for presence and shape first; the
value rules fold over the association scan's blocks, which already hold the
mask. A grid and a Ragged release's sequence are judged by the same rules, each
in the words it has always been reported in. A Hybrid's Overflow is off-panel
and never imputed (`csr.observed_only`), so none of these applies to it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from opengwasdb.associations import Tally
from opengwasdb.completion.ancestry_filter import derive_impute_analysis_ids
from opengwasdb.completion.schema import COMPLETION_QUALITY_COLUMNS
from opengwasdb.store_reader.analyses import read_analyses
from opengwasdb.store_reader.arrays import array_length
from opengwasdb.validation.rule import Check, Kind, Rule, Target
from opengwasdb.validation.rules.associations import VALUE_KINDS, grid_counts, tally, value_fold
from opengwasdb.validation.rules.csr import is_overflow
from opengwasdb.validation.rules.encoding import plane_label
from opengwasdb.validation.rules.release import REQUIRED
from opengwasdb.validation.scans import Block, sequence_offsets
from opengwasdb.validation.scans import completed as release_completed

#: The rule every value rule on a completed grid runs after.
MASK = "completion.mask"


def completed(target: Target) -> bool:
    """A Reference-Completed component that can hold imputed cells: not an Overflow."""
    return release_completed(target) and not is_overflow(target)


def mask(target: Target) -> Iterator[str]:
    """A completed component carries an `imputed` mask of its shape; a grid, binary panel flags."""
    if target.kind is Kind.CSR:
        yield from _sequence_mask(target)
    else:
        yield from _grid_mask(target)


def _grid_mask(target: Target) -> Iterator[str]:
    """`on_panel` is a 1-D array of about `n_variants` bytes, read once for the run."""
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


def _sequence_mask(target: Target) -> Iterator[str]:
    """A completed sequence carries an `imputed` flag for every association."""
    group = target.group
    if "imputed" not in group:
        yield "reference-completed store is missing data.zarr/ragged/imputed"
        return
    n_assoc = int(sequence_offsets(target)[-1])
    held = array_length(group["imputed"])
    if held != n_assoc:
        yield f"data.zarr/ragged/imputed has {held} entries but offsets imply {n_assoc}"


def _judge_binary(target: Target, total: Tally) -> Iterator[str]:
    if total.flags_not_binary:
        yield f"{plane_label(target, 'imputed')} contains values other than 0 and 1"


def _judge_present(target: Target, total: Tally) -> Iterator[str]:
    """An imputed cell carries both Z and SE (spec §15, ADR 0070).

    A sequence names its cells rows, and has always repeated the missing-Z count
    in other words. The repeat is kept, since dropping it changes
    `validate --format json`; #378 can drop it (ADR 0074).
    """
    cells = "cells" if target.kind is Kind.GRID else "rows"
    if total.imputed_z_missing:
        yield f"imputed=1 {cells} have missing z-scores"
    if total.imputed_se_missing:
        yield f"imputed=1 {cells} have NaN se values"
    if target.kind is Kind.CSR and total.imputed_z_missing:
        yield "missing z rows have imputed=1 (inconsistent)"


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
    counts = np.zeros(_n_analyses(target), dtype=np.int64) + total
    errors: list[str] = []
    label = plane_label(target, "imputed")
    declaration_matches_mask(target.store.analyses_path, counts, errors, label)
    yield from errors


def declaration_matches_mask(
    analyses_path: Path,
    imputed_per_analysis: np.ndarray,
    errors: list[str],
    label: str,
) -> None:
    """Each Analysis holds imputed cells if and only if it says it does.

    This remains the categorical declaration rule (ADR 0028, ADR 0037 §4): an
    Analysis carried through observed-only must not claim imputed cells and derive
    its `eaf_scope` from that claim. `completion.quality_counts` separately checks
    exact counts where at most one quality row avoids ambiguous overlaps (#307).
    """
    for row in read_analyses(analyses_path).rows:
        try:
            index = int(row.get("analysis_index", ""))
            declared = int(row.get("completion_n_imputed_total", "") or 0)
        except ValueError:
            continue  # analyses.completion_declared and analyses.analysis_index report it
        if 0 <= index < len(imputed_per_analysis):
            message = _disagreement(row, declared, int(imputed_per_analysis[index]), label)
            if message is not None:
                errors.append(message)


def _disagreement(row: dict[str, str], declared: int, actual: int, label: str) -> str | None:
    """What one Analysis's declaration and mask disagree on, if they do."""
    analysis_id = row.get("analysis_id", "")
    if declared and not actual:
        return (
            f"analysis {analysis_id!r} declares completion_n_imputed_total={declared} "
            f"but has no imputed=1 cell in {label}; the metadata describes "
            "cells the release does not contain"
        )
    if actual and not declared:
        return (
            f"analysis {analysis_id!r} has {actual} imputed=1 cell(s) in "
            f"{label} but declares completion_n_imputed_total="
            f"{row.get('completion_n_imputed_total', '')!r}; the release holds "
            "imputed cells its metadata does not account for"
        )
    return None


def _quality_cells(_target: Target, block: Block) -> np.ndarray:
    """Imputed cells and missing panel cells, measured on the shared scan (#361).

    Dense completion's off-panel missing cells have no LD block and do not belong
    to completion_quality. Ragged missing rows are retained completion attempts;
    source-observed rows are present, including off-panel ones (ADR 0039).
    """
    assert block.imputed is not None
    missing = block.z_missing.copy()
    if block.on_panel is not None:
        missing &= block.on_panel[:, None] == 1
    return np.stack((block.per_analysis(block.imputed == 1), block.per_analysis(missing)))


def _quality_totals(target: Target) -> tuple[np.ndarray, np.ndarray]:
    """The Store records block ids and counts, but no block-to-cell membership.

    Count rows as well as attempts: multiple rows cannot prove whether attempted
    cells overlap (owner ruling, 16 Nov 2026, #307). A Store Release must validate
    without an external LD panel or guesses from arbitrary block names.
    """
    n = _n_analyses(target)
    errors: list[str] = []
    totals = np.zeros((2, n), dtype=np.int64)
    row_counts = np.zeros(n, dtype=np.int64)
    with target.store.index_connection() as connection:
        _validate_completion_quality_table(connection, n, errors)
        if errors:
            raise ValueError("completion_quality cannot be counted: " + "; ".join(errors))
        for analysis, block_id, imputed, missing in connection.execute(
            "SELECT analysis_index, block_id, n_imputed, n_missing FROM completion_quality"
        ):
            if not all(isinstance(value, int) and value >= 0 for value in (imputed, missing)):
                raise ValueError(
                    f"completion_quality for analysis {analysis}, block {block_id!r} "
                    "has non-integer or negative n_imputed/n_missing"
                )
            totals[:, analysis] += (imputed, missing)
            row_counts[analysis] += 1
    return totals, row_counts


def _ancestry_exclusions(target: Target) -> set[int]:
    """Only stored routing evidence, not absent quality rows, establishes exclusion."""
    rows = target.analyses.rows
    ancestry = target.store.manifest.provenance.get("completion", {}).get("ancestry")
    if not isinstance(ancestry, str) or not ancestry or not any(
        row.get("assigned_ancestry") for row in rows
    ):
        return set()
    eligible = derive_impute_analysis_ids(rows, ancestry)
    assert eligible is not None, "assigned ancestries must select an explicit set"
    return {
        int(row["analysis_index"]) for row in rows
        if row["analysis_id"] not in eligible and not row.get("completed_against")
    }


def _judge_quality_counts(target: Target, total: np.ndarray) -> Iterator[str]:
    expected, row_counts = _quality_totals(target)
    excluded = _ancestry_exclusions(target)
    actual = np.zeros_like(expected) + total
    for analysis in range(expected.shape[1]):
        if row_counts[analysis] > 1:
            continue
        if expected[0, analysis] != actual[0, analysis]:
            yield (
                f"completion_quality analysis {analysis}: n_imputed={expected[0, analysis]} "
                f"but the imputed plane marks {actual[0, analysis]} cell(s)"
            )
        # Missing rows are not evidence of ancestry exclusion: quality metadata
        # may have disappeared with every failed attempt it described (#307).
        if (
            (row_counts[analysis] or analysis not in excluded)
            and expected[:, analysis].sum() != actual[:, analysis].sum()
        ):
            yield (
                f"completion_quality analysis {analysis}: n_imputed + n_missing="
                f"{expected[:, analysis].sum()} but the planes hold "
                f"{actual[:, analysis].sum()} imputed or missing panel cell(s)"
            )


CHECKS: tuple[Rule, ...] = (
    Check(name=MASK, kinds=Kind.GRID | Kind.CSR, when=completed, after=(REQUIRED,), run=mask),
)

#: The value rules about the mask, in the order they have always reported.
RULES: tuple[Rule, ...] = (
    # Admission's cell guard refuses a flag neither 0 nor 1; no row's Reason names it.
    value_fold(
        "completion.mask_binary", Tally(), tally, _judge_binary, when=completed, witnesses=()
    ),
    # Completion admits an imputed cell only with a finite Z and a finite, positive SE.
    value_fold(
        "completion.imputed_present",
        Tally(),
        tally,
        _judge_present,
        when=completed,
        witnesses=("z_nonfinite", "se_unusable"),
    ),
    value_fold(
        "completion.off_panel",
        0,
        _off_panel_imputed,
        _judge_off_panel,
        when=completed,
        kinds=Kind.GRID,
    ),
    value_fold(
        "completion.declaration",
        np.zeros((), dtype=np.int64),
        _imputed_per_analysis,
        _judge_declaration,
        when=completed,
    ),
    value_fold(
        "completion.quality_counts",
        np.zeros((), dtype=np.int64),
        _quality_cells,
        _judge_quality_counts,
        when=completed,
    ),
)


def _n_analyses(target: Target) -> int:
    """The Analyses a quality row may describe: `analyses.tsv`'s, or the segments'."""
    if target.kind is Kind.CSR:
        return len(sequence_offsets(target)) - 1
    return grid_counts(target)[1]


def quality_table(target: Target) -> Iterator[str]:
    """A completed component's `completion_quality` table, against its Analyses."""
    errors: list[str] = []
    with target.store.index_connection() as connection:
        _validate_completion_quality_table(connection, _n_analyses(target), errors)
    yield from errors


def n_missing_column(target: Target) -> Iterator[str]:
    """`analyses.tsv` says how many of each Analysis's cells completion left missing."""
    columns = set(read_analyses(target.store.analyses_path).fieldnames)
    if "completion_n_missing_total" not in columns:
        yield "analyses.tsv is missing completion_n_missing_total for a reference-completed store"


def _validate_completion_quality_table(
    connection: sqlite3.Connection, n_analyses: int, errors: list[str]
) -> None:
    """Seam: the completion_quality table exists, is complete, and is in range.

    ``completion_quality.analysis_index`` is a positional reference into
    analyses.tsv, not a SQL foreign key (ADR 0030 consequences) -- this is the
    validator rule that decision says must stand in for the database engine's
    constraint, and it is shared by Dense and Ragged Reference-Completed
    stores so a quality row can never describe an Analysis that does not exist.
    """
    tables = {
        r[0]
        for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "completion_quality" not in tables:
        errors.append(
            "index.sqlite is missing the completion_quality table for a reference-completed store"
        )
        return
    cols = {
        r[1] for r in connection.execute("PRAGMA table_info(completion_quality)").fetchall()
    }
    required = COMPLETION_QUALITY_COLUMNS
    missing_cols = required - cols
    if missing_cols:
        errors.append(
            f"completion_quality table is missing columns: {', '.join(sorted(missing_cols))}"
        )
        return
    out_of_range = connection.execute(
        "SELECT COUNT(*) FROM completion_quality "
        "WHERE analysis_index < 0 OR analysis_index >= ?",
        (n_analyses,),
    ).fetchone()[0]
    if out_of_range:
        errors.append(
            f"completion_quality has {out_of_range} row(s) with analysis_index "
            f"outside analyses.tsv's range [0, {n_analyses})"
        )


#: The rules about the metadata completion writes beside the mask.
METADATA_RULES: tuple[Rule, ...] = (
    Check(
        name="completion.quality_table",
        kinds=VALUE_KINDS,
        when=completed,
        after=(REQUIRED,),
        run=quality_table,
    ),
    Check(
        name="completion.n_missing_column",
        kinds=VALUE_KINDS,
        when=completed,
        after=(REQUIRED, "analyses.readable"),
        run=n_missing_column,
    ),
)
