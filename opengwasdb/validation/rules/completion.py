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

import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.completion.ancestry_filter import derive_impute_analysis_ids
from opengwasdb.completion.manifest import PANEL_FIELDS, PANEL_FIELDS_SINCE_327
from opengwasdb.completion.schema import COMPLETION_QUALITY_COLUMNS
from opengwasdb.index.release_index import ReleaseIndex
from opengwasdb.model.analyses import NON_NEGATIVE_INTEGER_OR_BLANK
from opengwasdb.store_reader.analyses import read_analyses
from opengwasdb.store_reader.arrays import array_length
from opengwasdb.store_reader.associations import Tally
from opengwasdb.validation.rule import Check, Kind, Rule, Severity, Target
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
    return block.imputed_cells


def _judge_declaration(target: Target, total: np.ndarray) -> Iterator[str]:
    counts = np.zeros(_n_analyses(target), dtype=np.int64) + total
    errors: list[str] = []
    label = plane_label(target, "imputed")
    for path in _analyses_paths(target):
        declaration_matches_mask(path, counts, errors, label)
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
    a quality row's attempts where at most one row avoids ambiguous overlaps (#307).
    """
    # A value it cannot read is `analyses.completion_declared`'s to report, as it always was,
    # and `completion.imputed_total`'s, which judges every total exactly.
    declared_rows, _unread = _declared_totals(analyses_path, len(imputed_per_analysis))
    for row, index, declared in declared_rows:
        message = _disagreement(row, declared, int(imputed_per_analysis[index]), label)
        if message is not None:
            errors.append(message)


def _analyses_paths(target: Target) -> list[Path]:
    """The `analyses.tsv` files that declare this component's totals: its release's, and
    where the release is a Hybrid's Dense Component, the Hybrid root's too. From 0.3.0
    they are one file; a 0.2.3 Hybrid carries a copy in each, and each is judged."""
    release = target if target.kind is Kind.RELEASE else target.parent
    own = target.store.analyses_path
    if release is None or release.parent is None:
        return [own]
    root = release.parent.store.analyses_path
    return [own] if root == own else [own, root]


def _count(text: str | None) -> int | None:
    """A non-negative integer in the spelling the Store writes (`NON_NEGATIVE_INTEGER_OR_BLANK`,
    blank excluded); None for a blank, a missing value, a sign, a leading zero or anything
    else."""
    if text is None or not text or not re.fullmatch(NON_NEGATIVE_INTEGER_OR_BLANK, text):
        return None
    return int(text)


def _declared_totals(
    analyses_path: Path, n_analyses: int, column: str = "completion_n_imputed_total"
) -> tuple[list[tuple[dict[str, str], int, int]], list[str]]:
    """Each `analyses.tsv` row of an Analysis the planes cover, with its index and its
    declared `column`, and a finding for every value that is not a non-negative integer:
    a blank is not a zero, and a total no rule can read is no total. A table without the
    column has none to judge."""
    found: list[tuple[dict[str, str], int, int]] = []
    problems: list[str] = []
    table = read_analyses(analyses_path)
    if column not in table.fieldnames:
        return found, problems  # a column the table lacks is its own rule's to report
    for row in table.rows:
        index, declared = _count(row.get("analysis_index")), _count(row.get(column))
        if index is None or declared is None:
            name = row.get("analysis_id", "")
            bad, raw = ("analysis_index", row.get("analysis_index")) if index is None else (
                column, row.get(column)
            )
            problems.append(
                f"analysis {name!r} has {bad}={raw!r} in {analyses_path.parent.name}/"
                f"{analyses_path.name}, which is not a non-negative integer"
            )
        elif index < n_analyses:
            found.append((row, index, declared))
    return found, problems


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


def _judge_imputed_total(target: Target, total: np.ndarray) -> Iterator[str]:
    """The declared total is the cells the mask marks, exactly (#362).

    A cell two blocks fill is one imputed cell, and one they disagree on is none;
    a total summed from the blocks' own counts is neither, and nothing else
    notices it is high. Every `analyses.tsv` that declares it is judged.
    """
    counts = np.zeros(_n_analyses(target), dtype=np.int64) + total
    label = plane_label(target, "imputed")
    for path in _analyses_paths(target):
        declared_rows, problems = _declared_totals(path, len(counts))
        yield from problems
        for row, index, declared in declared_rows:
            actual = int(counts[index])
            if declared != actual:
                yield (
                    f"analysis {row.get('analysis_id', '')!r} declares "
                    f"completion_n_imputed_total={declared} but {label} marks {actual} "
                    f"imputed=1 cell(s): {abs(declared - actual)} "
                    f"{'more' if declared > actual else 'fewer'} than the release holds"
                    f" (in {path.parent.name}/{path.name})"
                )


def _missing_cells(target: Target, block: Block) -> np.ndarray:
    """Per Analysis, the cells the release holds no value at that its declaration counts:
    a Dense release's off-panel ones, a Ragged release's missing rows."""
    if target.kind is not Kind.GRID:
        return block.missing_cells
    if block.on_panel is None:
        return np.zeros(block.n_analyses, dtype=np.int64)
    return block.per_analysis(block.z_missing & (block.on_panel[:, None] == 0))


def _judge_missing_total(target: Target, total: np.ndarray) -> Iterator[str]:
    """The declared `completion_n_missing_total` against the cells the release holds no
    value at (#362; spec §12).

    Dense declares exactly the missing cells the release holds off the panel, whatever
    their cause. Ragged declares the rows completion added and left missing, which the
    release cannot count apart from a missing row its source stored and completion carried:
    it must be a non-negative integer no greater than the missing rows the release holds.
    Either way a sum of the blocks' own counts, which count a cell two blocks fail twice,
    exceeds it.
    """
    counts = np.zeros(_n_analyses(target), dtype=np.int64) + total
    grid = target.kind is Kind.GRID
    for path in _analyses_paths(target):
        declared_rows, problems = _declared_totals(path, len(counts), "completion_n_missing_total")
        yield from problems
        for row, index, declared in declared_rows:
            held = int(counts[index])
            if declared != held if grid else declared > held:
                yield (
                    f"analysis {row.get('analysis_id', '')!r} declares "
                    f"completion_n_missing_total={declared} but the release holds {held} "
                    f"{'off-panel ' if grid else ''}cell(s) with no value"
                    f" (in {path.parent.name}/{path.name})"
                )


def _quality_cells(_target: Target, block: Block) -> np.ndarray:
    """Imputed cells and missing panel cells, measured on the shared scan (#361).

    Dense completion's off-panel missing cells have no LD block and do not belong
    to completion_quality. Ragged missing rows are retained completion attempts;
    source-observed rows are present, including off-panel ones (ADR 0039).
    """
    assert block.imputed is not None
    if block.on_panel is None:
        return np.stack((block.imputed_cells, block.missing_cells))
    on_panel = block.z_missing & (block.on_panel[:, None] == 1)
    return np.stack((block.imputed_cells, block.per_analysis(on_panel)))


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
    with ReleaseIndex(target.store.path) as index:
        _validate_completion_quality_table(index, n, errors)
        if errors:
            raise ValueError("completion_quality cannot be counted: " + "; ".join(errors))
        rows = index.completion_quality()
    for analysis, block_id, imputed, missing in rows:
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
    """A quality row is a block's attempts, judged against the cells the release holds.

    An attempt is held as imputed, held as missing, overwritten by an observation (a
    Hybrid's crossover, #99; it records how many in `crossover_overwritten`), or
    superseded by a missing row the source stored and completion carried (Ragged, #424).
    Where the release can count each outcome the row is exact: a Dense release's, and a
    Hybrid Dense Component's with its record. A Ragged row's imputed attempts are bounded
    below by the imputed rows held and above by those plus the missing rows held, which
    are all a superseded fill can be; its attempts in all are exact. The totals in
    `analyses.tsv` judge the release: `completion.imputed_total` and
    `completion.missing_total`.
    """
    expected, row_counts = _quality_totals(target)
    excluded = _ancestry_exclusions(target)
    held = np.zeros_like(expected) + total
    overwritten, problems = _crossover_record(target, expected.shape[1])
    yield from problems
    for analysis in range(expected.shape[1]):
        if row_counts[analysis] > 1:
            continue
        # Missing rows are not evidence of ancestry exclusion: quality metadata
        # may have disappeared with every failed attempt it described (#307).
        judge_sum = bool(row_counts[analysis]) or analysis not in excluded
        yield from _judge_attempts(
            analysis, expected[:, analysis], held[:, analysis], overwritten[:, analysis],
            ragged=target.kind is Kind.CSR, judge_sum=judge_sum,
        )


def _judge_attempts(
    analysis: int, row: np.ndarray, held: np.ndarray, overwritten: np.ndarray,
    *, ragged: bool, judge_sum: bool,
) -> Iterator[str]:
    """One Analysis's row of `(n_imputed, n_missing)` against the `(imputed, missing)` cells
    held and the `(imputed, missing)` attempts a crossover overwrote."""
    imputed, missing = int(held[0]), int(held[1])
    low = imputed + int(overwritten[0])
    high = low + (missing if ragged else 0)
    if not low <= row[0] <= high:
        yield (
            f"completion_quality analysis {analysis}: n_imputed={row[0]} but the imputed "
            f"plane marks {imputed} cell(s)"
            + (f" and a superseded fill could be any of {missing} missing row(s)" if ragged else "")
            + (f" beside {int(overwritten[0])} overwritten by an observation"
               if overwritten[0] else "")
        )
    if judge_sum and row.sum() != imputed + missing + int(overwritten.sum()):
        yield (
            f"completion_quality analysis {analysis}: n_imputed + n_missing={row.sum()} but "
            f"the planes hold {imputed + missing} imputed or missing panel cell(s)"
            + (f" and {int(overwritten.sum())} overwritten by an observation"
               if overwritten.any() else "")
        )


def _crossover_record(target: Target, n_analyses: int) -> tuple[np.ndarray, list[str]]:
    """Per Analysis, the imputed and missing attempts a Hybrid's crossover overwrote, from
    the Dense Component's `provenance.completion.crossover_overwritten`; zeros where it
    records none, as every other component does."""
    none = np.zeros((2, n_analyses), dtype=np.int64)
    record = target.store.manifest.provenance.get("completion", {}).get("crossover_overwritten")
    if not target.is_component or record is None:
        return none, []
    columns = [_counts_column(record, key, n_analyses) for key in ("n_imputed", "n_missing")]
    if any(column is None for column in columns):
        return none, [
            "provenance.completion.crossover_overwritten is not two lists "
            f"(n_imputed, n_missing) of {n_analyses} non-negative integers"
        ]
    return np.array(columns, dtype=np.int64), []


def _counts_column(record: object, key: str, n: int) -> list[int] | None:
    """`record[key]` if it is `n` non-negative integers, else None."""
    column = record.get(key) if isinstance(record, dict) else None
    ok = isinstance(column, list) and len(column) == n and all(
        isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in column
    )
    return column if ok else None


def _panel_record(target: Target) -> Mapping[str, Any]:
    record = target.store.manifest.provenance.get("completion")
    return record if isinstance(record, dict) else {}


def panel_unrecorded(target: Target) -> Iterator[str]:
    """A completion that records none of the panel fields added at #327 (`ld_panel_id` was
    always recorded) was written before they existed; the format is the same, so only a
    warning says so."""
    if not any(key in _panel_record(target) for key in PANEL_FIELDS_SINCE_327):
        yield (
            "provenance.completion records none of the LD panel fields added at #327: the "
            "release names its panel only by `ld_panel_id`, which a completion before then "
            "set to a constant"
        )


def _digest_problems(record: Mapping[str, Any]) -> Iterator[str]:
    digest = record["ld_panel_digest"]
    if not (isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)):
        yield f"provenance.completion.ld_panel_digest {digest!r} is not a sha256"
    elif (
        record["ld_panel_id_source"] == "digest"
        and record["ld_panel_id"] != f"sha256:{digest[:16]}"
    ):
        yield (
            f"provenance.completion.ld_panel_id {record['ld_panel_id']!r} is not the id its "
            f"digest derives ('sha256:{digest[:16]}')"
        )
    if record["ld_panel_id_source"] not in ("caller", "digest"):
        yield (
            f"provenance.completion.ld_panel_id_source {record['ld_panel_id_source']!r} "
            "is neither 'caller' nor 'digest'"
        )


def _component_problems(target: Target, record: Mapping[str, Any]) -> Iterator[str]:
    for component in target.children:
        if component.kind is Kind.RELEASE:
            inner = _panel_record(component)
            for key in PANEL_FIELDS:
                if inner.get(key) != record[key]:
                    yield (
                        f"provenance.completion.{key} is {record[key]!r}, but the Dense "
                        f"Component records {inner.get(key)!r}: one panel per completed release"
                    )


def panel_consistent(target: Target) -> Iterator[str]:
    """A completion that records any panel field added at #327 records all five, consistently: the
    derived id is the digest's prefix, and a Hybrid and its Dense Component name one panel."""
    record = _panel_record(target)
    if not any(key in record for key in PANEL_FIELDS_SINCE_327):
        return
    missing = [key for key in PANEL_FIELDS if not record.get(key)]
    if missing:
        yield f"provenance.completion records no {', '.join(missing)} of its LD panel"
        return
    yield from _digest_problems(record)
    yield from _component_problems(target, record)


CHECKS: tuple[Rule, ...] = (
    Check(name=MASK, kinds=Kind.GRID | Kind.CSR, when=completed, after=(REQUIRED,), run=mask),
    Check(
        name="completion.ld_panel", kinds=Kind.RELEASE, when=release_completed,
        after=(REQUIRED,), run=panel_consistent,
    ),
    Check(
        name="completion.ld_panel_unrecorded", kinds=Kind.RELEASE, when=release_completed,
        after=(REQUIRED,), severity=Severity.WARNING, run=panel_unrecorded,
    ),
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
        "completion.imputed_total",
        np.zeros((), dtype=np.int64),
        _imputed_per_analysis,
        _judge_imputed_total,
        when=completed,
    ),
    value_fold(
        "completion.missing_total",
        np.zeros((), dtype=np.int64),
        _missing_cells,
        _judge_missing_total,
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
    with ReleaseIndex(target.store.path) as index:
        _validate_completion_quality_table(index, _n_analyses(target), errors)
    yield from errors


def n_missing_column(target: Target) -> Iterator[str]:
    """`analyses.tsv` says how many of each Analysis's cells completion gained and left
    missing: it carries both totals, the one place an absent column is reported (the
    rules judging each total skip a table without it, #362)."""
    columns = set(read_analyses(target.store.analyses_path).fieldnames)
    for column in ("completion_n_imputed_total", "completion_n_missing_total"):
        if column not in columns:
            yield f"analyses.tsv is missing {column} for a reference-completed store"


def _validate_completion_quality_table(
    index: ReleaseIndex, n_analyses: int, errors: list[str]
) -> None:
    """Seam: the completion_quality table exists, is complete, and is in range.

    ``completion_quality.analysis_index`` is a positional reference into
    analyses.tsv, not a SQL foreign key (ADR 0030 consequences) -- this is the
    validator rule that decision says must stand in for the database engine's
    constraint, and it is shared by Dense and Ragged Reference-Completed
    stores so a quality row can never describe an Analysis that does not exist.
    """
    if "completion_quality" not in index.names(tables_only=True):
        errors.append(
            "index.sqlite is missing the completion_quality table for a reference-completed store"
        )
        return
    cols = index.completion_quality_columns()
    required = COMPLETION_QUALITY_COLUMNS
    missing_cols = required - cols
    if missing_cols:
        errors.append(
            f"completion_quality table is missing columns: {', '.join(sorted(missing_cols))}"
        )
        return
    out_of_range = index.completion_quality_rows_outside(n_analyses)
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
