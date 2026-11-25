"""Convert a 0.2.0-0.2.2 Dense release to 0.3.0, recording what it cannot fix (#492).

`ogdb rewrite-release --convert-legacy` (ADR 0105, "25 Nov 2026 (#492)"). The rewrite
(`opengwasdb.store.rewrite`) takes a 0.2.3 source; a 0.2.0, 0.2.1 or 0.2.2 one lacks
some of 0.2.3's features. Under the owner's decision of 25 Nov 2026 such a release is
converted fast, with known deviations recorded, and rebuilt from source later. Per gap:

- **An SE above 65,504 (ADR 0068): fixed exactly.** Every cell whose SE decodes above
  `SE_FLOAT16_MAX`, `+inf` included, becomes missing -- z, SE and frequency at their
  missing codes, the imputed flag 0, its exact-value side table entries gone -- in the
  grid and in every Indexed Variant Subset, and leaves every Top-Hit tier. Each
  Analysis counts them in `n_missing_se_out_of_range`. A completion's fill above the
  bound is refused, not masked: its block's `completion_quality` row would then count
  a missing cell as imputed, and the release records no block-to-cell membership to
  correct it by.
- **`source_alid` of an unlifted release: fixed exactly.** The operator says the
  release lifted nothing (`LegacyConversion.lifted`, recorded in provenance): a Dense
  manifest records no Analysis's source assembly, and every one records a chain label
  (`variant_table._judges_source_alid`). Each row a source observed is then given its
  own ALID, state 1, as a build writes it; a row Reference Completion added from the
  panel keeps none, as completion leaves it. Those rows are the rows no cell is
  observed in, and their count must be the completion's `n_variants_new`.
- **An indel's ALID before 0.2.2 (ADR 0079): recorded.** The reference allele of a
  sorted indel is not in the release; the ALIDs are carried as written, and
  `variant_table.indel_order` is a recorded deviation.
- **`source_alid` of a lifted release: recorded.** The pre-liftover ALID is not in a
  release that did not keep it; the table's column is carried, and
  `variant_table.source_alid` is a recorded deviation.

The deviations are recorded by validator rule name in `provenance.converted_from`, and
`validate` reports exactly those rules as warnings that name the conversion
(`opengwasdb.validation.rule.Rule.finding`).

**The proof** asks the source, through the 0.2 reader, the questions the rewrite asks
(`query_digests`), with the deliberate changes applied to its answers
(`expected_query`): the masked cells read as missing, leave the tiers and are counted,
and an unlifted row names its expected `source_alid`. The staged release must then
answer every question identically: each change is compared cell for cell, not left out.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.encoding.codec import SE_FLOAT16_MAX, se_above_bound
from opengwasdb.store.legacy_masking import (
    cell_coordinates,
    cell_positions,
    planes_not_rewritten,
)
from opengwasdb.store.open import OpenGWASDBStore, open_store
from opengwasdb.store.rewrite import RewriteContext, RewriteError
from opengwasdb.store_reader.associations import Cells, DenseGridReader, State
from opengwasdb.store_reader.encoding.codec import SeExceptionTable
from opengwasdb.store_reader.enums import CompletionState, PrimaryStorageLayout
from opengwasdb.store_reader.format import indels_reference_first

#: The source versions `--convert-legacy` takes: the 0.2 series before 0.2.3.
LEGACY_SOURCE_VERSIONS = ("0.2.0", "0.2.1", "0.2.2")

#: The validator rules a conversion can record as deviations (module docstring).
INDEL_ORDER_RULE = "variant_table.indel_order"
SOURCE_ALID_RULE = "variant_table.source_alid"
ROUNDED_SE_RULE = "side_tables.se_exceptions_rounded"

#: Every rule a conversion can record: the only rules `validate` lowers for one.
DEVIATION_RULES = (INDEL_ORDER_RULE, SOURCE_ALID_RULE, ROUNDED_SE_RULE)

#: The provenance key a converted release records, and a rebuild never writes.
CONVERTED_FROM = "converted_from"

#: The engine named in `converted_from`.
ENGINE = "ogdb rewrite-release --convert-legacy"


class LegacyConversionError(RewriteError):
    """A legacy source the conversion refuses."""


@dataclass(frozen=True)
class LegacyConversion:
    """What the operator asks of a conversion: whether the release lifted any variant
    from another assembly, which no Dense manifest records (module docstring)."""

    lifted: bool


@dataclass(frozen=True)
class LegacyPlan:
    """What one read of the source found, which the conversion's steps write and its
    proof compares: the cells it masks, their count per Analysis, and the rows that
    keep no `source_alid`."""

    source_version: str
    lifted: bool
    n_variants: int
    n_analyses: int
    #: Each masked cell's flat position, ``row * n_analyses + analysis``, ascending.
    masked: np.ndarray
    #: Per Analysis, how many cells are masked: the `n_missing_se_out_of_range` written.
    out_of_range: np.ndarray
    #: The rows no cell is observed in, ascending: on a Reference-Completed release the
    #: rows completion added from the panel. Empty on an Observed-Only one, and on a
    #: lifted one, whose rows' `source_alid` is carried and not named from them.
    unobserved_rows: np.ndarray
    #: On a Reference-Completed release, how many masked cells each
    #: `(analysis_index, block_id)` row of `completion_quality` gains in `n_missing`
    #: (`legacy_quality.attribute`).
    quality_missing: tuple[tuple[tuple[int, str], int], ...] = ()
    #: Of the source's SE exceptions, how many are `float16` values and of how many,
    #: where the table was rounded (#355): recorded as a deviation; None where it was not.
    rounded_se: tuple[int, int] | None = None

    @property
    def deviations(self) -> tuple[str, ...]:
        """The validator rules the converted release does not meet, by name."""
        found = []
        if not indels_reference_first(self.source_version):
            found.append(INDEL_ORDER_RULE)
        if self.lifted:
            found.append(SOURCE_ALID_RULE)
        if self.rounded_se is not None:
            found.append(ROUNDED_SE_RULE)
        return tuple(found)

    @property
    def masked_cells(self) -> tuple[np.ndarray, np.ndarray]:
        """The masked cells' rows and Analyses."""
        return cell_coordinates(self.masked, self.n_analyses)

    def source_alid(self, row: int, alid: str, carried: str | None) -> str | None:
        """Row `row`'s `source_alid` after the conversion: `carried`, the table's, on a
        lifted release; on an unlifted one its own ALID, or none on a panel row."""
        if self.lifted:
            return carried
        slot = int(np.searchsorted(self.unobserved_rows, row))
        on_panel = slot < len(self.unobserved_rows) and int(self.unobserved_rows[slot]) == row
        return None if on_panel else alid

    def record(self, source_release_id: str, commit: str | None, now: str) -> dict[str, Any]:
        """`provenance.converted_from`, as every converted manifest records it."""
        return {
            "source_format_version": self.source_version,
            "source_release_id": source_release_id,
            "engine": ENGINE,
            "opengwasdb_git_hash": commit,
            "at": now,
            "deviations": list(self.deviations),
            "lifted": self.lifted,
            "lifted_decided_by": "operator flag " + ("--lifted" if self.lifted else "--not-lifted"),
            "se_out_of_range_masked": int(self.masked.size),
            **_rounded_record(self),
            "note": (
                "Converted by ogdb rewrite-release --convert-legacy (#492): every SE above "
                f"{SE_FLOAT16_MAX:,.0f} was masked and counted, and an unlifted release's "
                "rows named their own ALID as source_alid. The deviations listed are what a "
                "rebuild from source fixes; validate reports exactly those rules as "
                "warnings. A rebuild writes no converted_from."
            ),
        }


def judge_converted_from(record: Any) -> tuple[tuple[str, ...], list[str]]:
    """The deviations a release's `provenance.converted_from` may record, and why each
    other one may not: the one judge of the record, which `validate` lowers rules by
    and fails a release for (`manifest.converted_from`, #492).

    A deviation is accepted only if it is one of `DEVIATION_RULES` and the record says
    why it holds: `variant_table.source_alid` only where `lifted` is true;
    `variant_table.indel_order` only from a source before 0.2.2; and
    `side_tables.se_exceptions_rounded` only with its counts. Anything else is a fault:
    a hand-edited or mistaken record must not silence a rule."""
    if not isinstance(record, dict):
        return (), ["provenance.converted_from is not an object"]
    listed = record.get("deviations")
    if not isinstance(listed, list) or not all(isinstance(rule, str) for rule in listed):
        return (), [f"provenance.converted_from.deviations is {listed!r}, not a list of rules"]
    accepted, faults = [], []
    for rule in listed:
        why_not = _why_not_recorded(rule, record)
        if why_not:
            faults.append(f"provenance.converted_from.deviations lists {rule!r}, {why_not}")
        else:
            accepted.append(rule)
    return tuple(accepted), faults


def _why_not_recorded(rule: str, record: dict[str, Any]) -> str:
    """Why `record` may not list `rule`, or ``""`` where it may."""
    if rule not in DEVIATION_RULES:
        return f"which no conversion records (it records only {', '.join(DEVIATION_RULES)})"
    return _UNJUSTIFIED[rule](record)


def _not_lifted(record: dict[str, Any]) -> str:
    if record.get("lifted") is True:
        return ""
    return "but the record does not say the release was lifted"


def _not_sorted(record: dict[str, Any]) -> str:
    if _sorts_indels(record.get("source_format_version")):
        return ""
    return (
        f"but its source_format_version {record.get('source_format_version')!r} already "
        "spells indels reference first"
    )


def _not_counted(record: dict[str, Any]) -> str:
    counts = (record.get("se_exceptions_rounded"), record.get("se_exceptions"))
    if all(isinstance(n, int) for n in counts):
        return ""
    return "without the counts se_exceptions_rounded and se_exceptions"


#: Each deviation rule, and why a record would not justify it.
_UNJUSTIFIED = {
    INDEL_ORDER_RULE: _not_sorted,
    SOURCE_ALID_RULE: _not_lifted,
    ROUNDED_SE_RULE: _not_counted,
}


def _sorts_indels(version: Any) -> bool:
    """Whether `version` is a 0.2 source before 0.2.2, whose indels are sorted."""
    return version in LEGACY_SOURCE_VERSIONS and not indels_reference_first(str(version))


def _rounded_record(plan: LegacyPlan) -> dict[str, int]:
    if plan.rounded_se is None:
        return {}
    exact, total = plan.rounded_se
    return {"se_exceptions_rounded": exact, "se_exceptions": total}


def refuse_unconvertible(store: OpenGWASDBStore, version: str) -> None:
    """Refuse a source `--convert-legacy` does not take, from its manifest alone."""
    if version not in LEGACY_SOURCE_VERSIONS:
        raise LegacyConversionError(
            f"{store.path}: format_version is {version}; --convert-legacy converts only a "
            f"{', '.join(LEGACY_SOURCE_VERSIONS)} release (a 0.2.3 one is rewritten "
            "without the flag)"
        )
    layout = store.manifest.primary_layout
    if layout is not PrimaryStorageLayout.DENSE:
        raise LegacyConversionError(
            f"{store.path}: primary_layout is {layout.value}; --convert-legacy converts only a "
            "Dense release (#492). Rebuild a Ragged or Hybrid one from source"
        )
    _refuse_other_planes(store)


def _refuse_other_planes(store: OpenGWASDBStore) -> None:
    """Refuse, before a cell is read, a release whose grid or Indexed Variant Subsets
    carry a plane a masked cell is not rewritten in (`legacy_masking`)."""
    from opengwasdb.store_reader.indexed import INDEXED_SUBSETS_GROUP

    root = store.arrays()
    groups = [("data.zarr", root)]
    if INDEXED_SUBSETS_GROUP in root:
        subsets = root[INDEXED_SUBSETS_GROUP]
        groups += [(f"subset {name}", subsets[name]) for name in sorted(subsets.group_keys())]
    for where, group in groups:
        held = planes_not_rewritten(group)
        if held:
            raise LegacyConversionError(
                f"{store.path}: {where} carries {', '.join(held)}, which a cell masked for an "
                "SE above 65,504 is not rewritten in; --convert-legacy does not convert it: "
                "rebuild it from source"
            )


def plan_conversion(
    store: OpenGWASDBStore, version: str, conversion: LegacyConversion
) -> LegacyPlan:
    """Read every cell of the source once: the cells above the bound, and the rows no
    cell is observed in. Refuses a completion's fill above the bound, and a completed
    release whose unobserved rows are not exactly its panel rows (module docstring)."""
    started = time.perf_counter()
    grid = _grid(store)
    n, m = grid.n_variants, grid.n_analyses
    positions, fills, unobserved = _read_cells(grid)
    _refuse_masked_fills(store, fills, m)
    plan = LegacyPlan(
        source_version=version,
        lifted=conversion.lifted,
        n_variants=n,
        n_analyses=m,
        masked=positions,
        out_of_range=np.bincount(positions % m, minlength=m).astype(np.int64),
        unobserved_rows=_NONE if conversion.lifted else _panel_rows(store, unobserved),
        quality_missing=_quality_missing(store, positions, m),
        rounded_se=_rounded_se(store),
    )
    print(
        f"  read {n:,} rows x {m:,} Analyses: {positions.size:,} cell(s) with an SE above "
        f"{SE_FLOAT16_MAX:,.0f}, {plan.unobserved_rows.size:,} row(s) no cell is observed "
        f"in: {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return plan


def _read_cells(grid: DenseGridReader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every cell of the grid, a shard of rows at a time: the flat positions of those
    whose SE is above the bound, of those of them flagged imputed, and the rows no cell
    is observed in, each ascending."""
    m = grid.n_analyses
    over_bound, fills, unobserved = [_NONE], [_NONE], [_NONE]
    for r0, r1 in _bands(grid):
        cells = grid.inspect(Cells.variants(np.arange(r0, r1, dtype=np.int64)))
        rows = cells.variant_index.astype(np.int64)
        flat = cell_positions(rows, cells.analysis_index, m)
        over = se_above_bound(cells.se)
        over_bound.append(flat[over])
        if cells.flags is not None:
            fills.append(flat[over & (cells.flags == 1)])
        observed = np.zeros(r1 - r0, dtype=bool)
        observed[rows[cells.state == State.OBSERVED] - r0] = True
        unobserved.append(np.flatnonzero(~observed) + r0)
    masked, flagged, empty = (
        np.sort(np.concatenate(found)) for found in (over_bound, fills, unobserved)
    )
    return masked, flagged, empty


_NONE = np.empty(0, dtype=np.int64)


def _grid(store: OpenGWASDBStore) -> DenseGridReader:
    completed = store.manifest.completion_state is CompletionState.REFERENCE_COMPLETED
    return DenseGridReader.open(
        store.arrays(), store.manifest.encoding, source=str(store.path), requires_mask=completed
    )


def _bands(grid: DenseGridReader) -> Iterator[tuple[int, int]]:
    """The grid's rows a shard at a time: each band reads one shard of each plane."""
    step = max(grid.scan_window, 1)
    for r0 in range(0, grid.n_variants, step):
        yield r0, min(r0 + step, grid.n_variants)


def _refuse_masked_fills(store: OpenGWASDBStore, fills: np.ndarray, n_analyses: int) -> None:
    if fills.size:
        rows, analyses = cell_coordinates(fills[:1], n_analyses)
        raise LegacyConversionError(
            f"{store.path}: {fills.size:,} imputed cell(s) hold an SE above "
            f"{SE_FLOAT16_MAX:,.0f} (first, row {int(rows[0])}, Analysis "
            f"{int(analyses[0])}). Masking one would leave its block's completion_quality "
            "row counting a missing cell as imputed, and the release records no "
            "block-to-cell membership to correct it by: rebuild it from source"
        )


def _rounded_se(store: OpenGWASDBStore) -> tuple[int, int] | None:
    """The converted release's SE exceptions as `side_tables.se_exceptions_rounded`
    measures them, where that rule would fail it (#355): it is then a deviation, the
    rounding being what the source serves, within 2**-11 relative. They are the
    source's, less the entries above 65,504 the mask removes, none of them a `float16`
    value; None where the rule would pass."""
    from opengwasdb.validation.rules.side_tables import (
        RoundedSeExceptions,
        rounded_se_exceptions,
    )

    group = store.arrays()
    found = rounded_se_exceptions(group, store.manifest.encoding)
    if found is None:
        return None
    above = int(np.count_nonzero(se_above_bound(group[SeExceptionTable.value_name][:])))
    kept = RoundedSeExceptions(found.n_exact, found.n_total - above)
    return (kept.n_exact, kept.n_total) if kept.rounded else None


def _quality_missing(
    store: OpenGWASDBStore, positions: np.ndarray, n_analyses: int
) -> tuple[tuple[tuple[int, str], int], ...]:
    """Each `completion_quality` row's masked panel cells (`legacy_quality`); none on an
    Observed-Only release, which has no such table."""
    if store.manifest.completion_state is not CompletionState.REFERENCE_COMPLETED:
        return ()
    from opengwasdb.store.legacy_quality import UnattributableCell, attribute

    try:
        counted = attribute(store, *cell_coordinates(positions, n_analyses))
    except UnattributableCell as exc:
        raise LegacyConversionError(str(exc)) from exc
    return tuple(sorted(counted.items()))


def _panel_rows(store: OpenGWASDBStore, unobserved: np.ndarray) -> np.ndarray:
    """The rows completion added from the panel: on a completed release, the rows no
    cell is observed in, which must number its `n_variants_new`; none otherwise."""
    manifest = store.manifest
    if manifest.completion_state is not CompletionState.REFERENCE_COMPLETED:
        return np.empty(0, dtype=np.int64)
    added = manifest.provenance.get("completion", {}).get("n_variants_new")
    if not isinstance(added, int) or added != unobserved.size:
        raise LegacyConversionError(
            f"{store.path}: {unobserved.size:,} row(s) hold no observed cell, but the "
            f"completion records n_variants_new={added!r}. The rows a source observed "
            "cannot then be told from those completion added, so no row's source_alid "
            "can be set: rebuild it from source"
        )
    return unobserved.astype(np.int64)


# ── the steps ────────────────────────────────────────────────────────────────


def mask_out_of_range(context: RewriteContext, plan: LegacyPlan) -> None:
    """Mask the plan's cells in the staged grid, its subsets and its tiers, and write
    `n_missing_se_out_of_range` (and the tiers' recount) into `analyses.tsv`."""
    from opengwasdb.store import arrays as store_arrays
    from opengwasdb.store.legacy_masking import mask_cells, mask_subsets, mask_tiers

    staged = context.staged
    root = store_arrays.open_group_for_write(staged / "data.zarr", "a")
    encoding = open_store(staged).manifest.encoding
    if plan.masked.size:
        mask_cells(root, plan.masked, encoding, plan.n_analyses)
        mask_subsets(root, plan.masked, encoding, plan.n_analyses)
        mask_tiers(root, plan.masked, plan.n_analyses)
    _write_counts(staged, plan)


def _write_counts(staged: Path, plan: LegacyPlan) -> None:
    from opengwasdb.layouts.dense.build import write_analyses_tsv
    from opengwasdb.model.analyses import (
        read_analysis_records,
        reset_top_hit_counts,
        with_se_out_of_range,
    )
    from opengwasdb.top_hits import add_hit_counts

    analyses = read_analysis_records(staged / "analyses.tsv")
    counted = with_se_out_of_range(analyses, plan.out_of_range)
    if plan.masked.size:
        counted = add_hit_counts(staged, reset_top_hit_counts(counted))
    write_analyses_tsv(staged, counted)


def count_in_quality(context: RewriteContext, plan: LegacyPlan) -> None:
    """Add each masked panel cell to its block's `n_missing` in the staged
    `completion_quality`, which the `provenance_index` step wrote from the source's."""
    if plan.quality_missing:
        from opengwasdb.store.legacy_quality import count_missing

        count_missing(context.staged, plan.quality_missing)


def name_own_source_alids(context: RewriteContext, plan: LegacyPlan) -> None:
    """Give an unlifted release's rows their `source_alid` in the staged Variant Table
    (`LegacyPlan.source_alid`); the `source_alid_sidecar` step writes the sidecar from
    it. A lifted release's table is left as its source wrote it."""
    if plan.lifted:
        return
    from opengwasdb.variants.axis import rewrite_source_alid_column

    contradicting: list[str] = []

    def named(row: int, alid: str, carried: str | None) -> str | None:
        if carried not in (None, alid) and not contradicting:
            contradicting.append(f"row {row} ({alid}) names {carried}")
        return plan.source_alid(row, alid, carried)

    written = rewrite_source_alid_column(context.staged, named)
    if contradicting:
        raise LegacyConversionError(
            f"{context.source}: the Variant Table names another ALID as a row's source_alid "
            f"({contradicting[0]}), so the release lifted it; it is converted with --lifted"
        )
    if written != plan.n_variants:
        raise LegacyConversionError(
            f"the Variant Table holds {written} rows, the grid {plan.n_variants}"
        )


def stamped(manifest: dict[str, Any], plan: LegacyPlan, record: dict[str, Any]) -> dict[str, Any]:
    """A converted manifest's provenance: `converted_from`, and the completion's parts of
    the count (ADR 0068), all observed since a fill above the bound is refused."""
    provenance = dict(manifest.get("provenance", {}))
    provenance[CONVERTED_FROM] = record
    completion = provenance.get("completion")
    if isinstance(completion, dict):
        provenance["completion"] = {
            **completion,
            "se_out_of_range": {
                "observed": plan.out_of_range.tolist(),
                "imputed": [0] * plan.n_analyses,
            },
        }
    return provenance


# ── the proof's expected answers ─────────────────────────────────────────────


def expected_query(plan: LegacyPlan) -> Any:
    """A `query_digests.QueryOpener` for the source: its 0.2 reader, with the plan's
    deliberate changes applied to every answer (module docstring)."""
    from opengwasdb.store.legacy_expected import open_expected

    def opener(path: Path) -> Any:
        return open_expected(path, plan)

    return opener


__all__ = [
    "CONVERTED_FROM",
    "DEVIATION_RULES",
    "LEGACY_SOURCE_VERSIONS",
    "LegacyConversion",
    "LegacyConversionError",
    "LegacyPlan",
    "judge_converted_from",
    "expected_query",
    "count_in_quality",
    "mask_out_of_range",
    "name_own_source_alids",
    "plan_conversion",
    "refuse_unconvertible",
    "stamped",
]
