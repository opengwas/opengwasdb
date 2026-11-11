"""The Observed-View Check for a Ragged Store (issue #296, stores ADR 0037).

The Ragged counterpart of the Dense check in `observed_view` (issue #289): does the
Reference-Completed Store `complete-ragged` wrote reproduce the Observed-Only Store it
came from? A Ragged Store has no grid to sample, and completion rebuilds the union
variant axis, so the comparison is by variant identity (ALID) over a seeded sample of
Analyses, never by index.

* **Exhaustive axis check.** Every source variant is found in the completed axis by its
  ALID (the exact map the Dense check uses), and both Stores list the same Analyses.
  The map is what lets an Analysis's source associations be looked up in the completed
  one, so a source variant the completed axis lacks fails the check outright.
* **Sampled Analysis check.** For each sampled Analysis, every source-observed
  association is in the completed Store with identical decoded Z, SE and EAF and not
  marked imputed; the completed Store holds no observed association the source lacks;
  every imputed association, and every missing one completion added, lies inside one
  of the Analysis's retained regions. An Analysis the ancestry filter did not admit is
  carried through observed-only (ADR 0028), so it must hold no imputed cell at all.

**Retained regions** (ADR 0014, ADR 0039, issue #430) are not stored as coordinates: a
Store records only block ids and counts in `completion_quality`. They are rebuilt here
by the call completion makes (`completion.regions`): an Analysis's cis region, from the
region table's `cis` row or else its Trait position and the `cis_window_bp` the manifest
records, and its significant trans regions, from the table. The check therefore takes
the region table completion was given (`regions`), refusing a trait-less Analysis
without one as completion does, and the panel, refusing without it: a check that quietly
skipped the region criterion would pass a Store whose imputed cells sit anywhere. A
region's rows are the panel variants of the blocks it overlaps that lie inside it. Block
ids that `completion_quality` records but the rebuilt regions lack fail the check, so a
panel other than the one completion used is noticed rather than trusted.

Rows completion writes inside a retained region that are neither observed nor imputed
(NaN, the panel positions it could not fill) are counted in the verdict and fail
nothing; one outside every region fails (`missing_outside_retained`). The verdict
records the seed, the sample and the counts, so a pass is a measured zero; a refusal
(`ObservedViewRefused`) writes none.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.associations import Cells, CsrReader, Inspection, State, Tally
from opengwasdb.completion.block import panel_positions
from opengwasdb.completion.ld_panel import (
    LDBlock,
    LdPanelNotFoundError,
    canonical_panel_alid,
    check_panel_has_chromosomes,
)
from opengwasdb.completion.regions import (
    Region,
    RegionTable,
    RegionTableError,
    analysis_regions,
    has_trait_position,
    inside,
    read_region_table,
    region_blocks,
)
from opengwasdb.layouts.ragged.zarr_csr import RAGGED_ZARR_PATH
from opengwasdb.model.analyses import Analysis, read_analysis_records
from opengwasdb.model.enums import AssociationCoverage, CompletionState
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store.open import OpenGWASDBStore
from opengwasdb.validation.observed_view import (
    ObservedViewRefused,
    ObservedViewVerdict,
    _AlidIndex,
    _AxisMap,
    _Cells,
    _map_axis,
    _observed,
    _open_checked,
    _plane_difference,
    _store_record,
    count_infinite_cells,
)
from opengwasdb.variants import VariantAxis, is_indexable_alid

__all__ = ["FAILING_COUNTERS", "STRATA", "check_ragged"]

STRATA = ("random", "non_admitted", "largest", "empty", "trait_less")
# Analyses each deterministic stratum adds; non-admitted ones are all included up to a cap,
# drawn with their own generator so that the seed's random stratum does not depend on them.
STRATUM_SIZE = 5
NON_ADMITTED_CAP = 100

FAILING_COUNTERS = (
    "z_mismatch",
    "se_mismatch",
    "eaf_mismatch",
    "observed_marked_imputed",
    "observed_missing_in_target",
    "target_observed_not_in_source",
    "imputed_outside_retained",
    "missing_outside_retained",
    "imputed_in_non_admitted",
    "imputed_mask_invalid",
    "z_se_missingness_mismatch",
    "non_canonical_marker",
    "target_variants_unordered",
    "quality_blocks_not_in_regions",
)
_INFORMATIONAL_COUNTERS = (
    "observed_in_source",
    "imputed",
    "nan_rows",
    "regions",
    "infinite_cells",
)
_TOTAL_KEYS = (*_INFORMATIONAL_COUNTERS, *FAILING_COUNTERS)
_DIFF_PLANES = ("z", "se", "eaf")


def check_ragged(
    source_path: str | Path,
    target_path: str | Path,
    *,
    seed: int,
    n_random: int,
    ld_panel: str | Path | None,
    ancestry: str | None,
    regions: str | Path | None = None,
) -> ObservedViewVerdict:
    """Compare an Observed-Only Ragged Store with the Reference-Completed one made from it.

    Raises `ObservedViewRefused` when the pair cannot be checked (mixed layouts, a source
    that is not Observed-Only, a target that is not Reference-Completed, no LD panel, a
    panel or ancestry the completion was not run with, or a region table completion
    could not have been run with, `regions` missing for a trait-less Analysis among
    them). `ancestry` defaults to the one the completed manifest records. A mismatch is
    a verdict with `passed` false.
    """
    started = time.time()
    pair = _open_pair(Path(source_path), Path(target_path))
    completion = _completion_provenance(pair.target, pair.target_path)
    panel = _open_panel(ld_panel, ancestry, completion, regions, pair.source)
    analyses = _analyses_record(pair)
    axis = _map_axis(pair.source_path, pair.target_path, pair.n_source_rows, pair.n_target_rows)
    record: dict[str, Any] = {
        "seed": seed,
        "n_random": n_random,
        "source": _store_record(pair.source, pair.source_path),
        "target": _store_record(pair.target, pair.target_path),
        "analyses": {k: analyses[k] for k in ("n_parent", "n_child", "identical")},
        "axis": axis.record(pair.n_source_rows, pair.n_target_rows),
        "ld_panel": {
            "path": str(panel.path),
            "ancestry": panel.ancestry,
            "cis_window_bp": panel.cis_window_bp,
        },
        "regions": None if panel.table is None else {
            "path": str(panel.table.path), "sha256": panel.table.sha256,
        },
        "failing_counters": list(FAILING_COUNTERS),
    }
    axis_seconds = time.time() - started
    if axis.unmapped == 0 and analyses["identical"]:
        record.update(_sample_and_compare(pair, analyses["source"], axis, panel, seed, n_random))
    else:
        # Associations that cannot be matched by ALID would only be compared against the
        # wrong ones, so the sample is skipped and the verdict says why it failed.
        record.update(_unsampled_record())
    record["failures"] = _failures(record, axis)
    record["passed"] = not record["failures"]
    record["seconds"] = {"axis": round(axis_seconds, 1), "total": round(time.time() - started, 1)}
    return ObservedViewVerdict(record)


# --- opening and refusing -----------------------------------------------------


@dataclass(frozen=True)
class _Pair:
    source_path: Path
    target_path: Path
    source: OpenGWASDBStore
    target: OpenGWASDBStore
    source_reader: CsrReader
    target_reader: CsrReader
    n_source_rows: int
    n_target_rows: int


def _open_pair(source_path: Path, target_path: Path) -> _Pair:
    source = _open_checked(source_path, "source", CompletionState.OBSERVED_ONLY)
    target = _open_checked(target_path, "target", CompletionState.REFERENCE_COMPLETED)
    if "imputed" not in _open_group(target_path, "target"):
        raise ObservedViewRefused(f"target store {target_path} has no `imputed` mask to check")
    rows = []
    for path in (source_path, target_path):
        with closing(VariantAxis(path)) as axis:
            rows.append(axis.n_variants)
        if rows[-1] == 0:
            raise ObservedViewRefused(f"store {path} has no variants to check")
    return _Pair(
        source_path, target_path, source, target,
        _reader(source_path, "source"), _reader(target_path, "target"), rows[0], rows[1],
    )


def _open_group(path: Path, role: str) -> Any:
    """The Store's Ragged array group; whatever stops it opening is a refusal."""
    try:
        return store_arrays.open_group(path / RAGGED_ZARR_PATH)
    except Exception as exc:
        raise ObservedViewRefused(f"{role} store {path} cannot be opened: {exc!r}") from exc


def _reader(path: Path, role: str) -> CsrReader:
    """The reader, which opens every array and decodes the encoding's side tables."""
    try:
        return CsrReader.open(path, source=f"{role} store {path}")
    except Exception as exc:
        raise ObservedViewRefused(f"{role} store {path} cannot be opened: {exc!r}") from exc


def _completion_provenance(target: OpenGWASDBStore, path: Path) -> dict[str, Any]:
    completion = target.manifest.provenance.get("completion")
    if not isinstance(completion, dict) or not {"ancestry", "cis_window_bp"} <= completion.keys():
        raise ObservedViewRefused(
            f"target store {path} records no completion ancestry and cis_window_bp, "
            "so its retained regions cannot be rebuilt"
        )
    return completion


@dataclass(frozen=True)
class _Panel:
    """What the retained regions are rebuilt from: the panel, the region table, and the
    source's Association Coverage, which says whether the table may be absent."""

    path: Path
    ancestry: str
    cis_window_bp: int
    table: RegionTable | None
    coverage: AssociationCoverage


def _open_panel(
    ld_panel: str | Path | None,
    ancestry: str | None,
    completion: dict[str, Any],
    regions: str | Path | None,
    source: OpenGWASDBStore,
) -> _Panel:
    if ld_panel is None:
        raise ObservedViewRefused(
            "a Ragged pair needs --ld-panel: its retained regions are not stored as "
            "coordinates, so they are rebuilt from the LD panel (and --ancestry) that "
            "complete-ragged was run with"
        )
    recorded = str(completion["ancestry"])
    if ancestry is not None and ancestry != recorded:
        raise ObservedViewRefused(
            f"the target was completed against ancestry {recorded}, not {ancestry}"
        )
    try:
        check_panel_has_chromosomes(ld_panel, recorded)
    except LdPanelNotFoundError as exc:
        raise ObservedViewRefused(f"LD panel {ld_panel}: {exc}") from exc
    try:
        table = (
            None if regions is None
            else read_region_table(regions, source.manifest.reference_assembly)
        )
    except (OSError, RegionTableError) as exc:
        raise ObservedViewRefused(f"region table {regions}: {exc}") from exc
    return _Panel(
        Path(ld_panel), recorded, int(completion["cis_window_bp"]), table,
        source.manifest.association_coverage,
    )


def _analyses_record(pair: _Pair) -> dict[str, Any]:
    source = _sorted_records(pair.source_path)
    target = _sorted_records(pair.target_path)
    columns_agree = (
        pair.source_reader.n_analyses == len(source)
        and pair.target_reader.n_analyses == len(target)
    )
    identical = [a.analysis_id for a in source] == [a.analysis_id for a in target] and columns_agree
    return {
        "n_parent": len(source), "n_child": len(target), "identical": identical,
        "source": source, "target": target,
    }


def _sorted_records(path: Path) -> list[Analysis]:
    """In `analysis_index` order, which is the order of the CSR's rows."""
    return sorted(read_analysis_records(path / "analyses.tsv"), key=lambda a: int(a.analysis_index))


# --- the sample ---------------------------------------------------------------


def _choose_analyses(
    analyses: Sequence[Analysis],
    n_associations: np.ndarray,
    admitted: np.ndarray,
    *,
    seed: int,
    n_random: int,
) -> dict[int, set[str]]:
    """Source Analyses to compare, each with the strata that chose it.

    `default_rng(seed)` serves the random stratum alone, so a seed picks the same
    Analyses however the deliberate strata are defined.
    """
    n = len(analyses)
    chosen: dict[int, set[str]] = {}

    def add(indices: Sequence[int] | np.ndarray, stratum: str) -> None:
        for i in indices:
            chosen.setdefault(int(i), set()).add(stratum)

    add(np.random.default_rng(seed).choice(n, size=min(n_random, n), replace=False), "random")
    not_admitted = np.flatnonzero(~admitted)
    if not_admitted.size > NON_ADMITTED_CAP:
        not_admitted = np.random.default_rng((seed, 1)).choice(
            not_admitted, size=NON_ADMITTED_CAP, replace=False
        )
    add(not_admitted, "non_admitted")
    add(np.argsort(-n_associations, kind="stable")[:STRATUM_SIZE], "largest")
    add(np.flatnonzero(n_associations == 0)[:STRATUM_SIZE], "empty")
    trait_less = [i for i, a in enumerate(analyses) if not has_trait_position(a)]
    add(trait_less[:STRATUM_SIZE], "trait_less")
    return chosen


# --- retained regions ---------------------------------------------------------


class _TargetRows:
    """Resolves ALIDs to rows of the completed axis, exactly (the index is a fast path)."""

    def __init__(self, target_path: Path, n_rows: int) -> None:
        self._index = _AlidIndex.load(target_path)
        loose = self._index.unindexed_rows(n_rows)
        self._loose: dict[str, int] = {}
        if loose.size:
            with closing(VariantAxis(target_path)) as axis:
                self._loose = {r.alid: row for row, r in axis.by_indices(loose.tolist()).items()}

    def aligned(self, alids: Sequence[str | None]) -> np.ndarray:
        """Each ALID's completed-axis row, or -1 where the completed axis holds none."""
        named = [(k, a) for k, a in enumerate(alids) if a is not None]
        found = np.full(len(alids), -1, dtype=np.int64)
        for k, a in named:
            found[k] = self._loose.get(a, -1)
        indexed = [(k, a) for k, a in named if is_indexable_alid(a)]
        if indexed and self._index.alids.size:
            at, rows = self._indexed_rows(indexed)
            found[at] = rows
        return found

    def _indexed_rows(self, indexed: list[tuple[int, str]]) -> tuple[np.ndarray, np.ndarray]:
        """The positions among `indexed` the index holds, and their rows."""
        keys = np.array([a.encode("utf-8") for _k, a in indexed], dtype=self._index.alids.dtype)
        slot = np.minimum(np.searchsorted(self._index.alids, keys), self._index.alids.size - 1)
        hit = self._index.alids[slot] == keys
        at = np.array([k for k, _a in indexed], dtype=np.int64)[hit]
        return at, np.asarray(self._index.rows)[slot[hit]]


@dataclass(frozen=True)
class _Retained:
    """The sampled admitted Analyses' regions and the blocks those regions overlap."""

    regions: dict[int, tuple[Region, ...]]
    blocks: dict[int, list[LDBlock]]


def _retained(
    analyses: Sequence[Analysis], sampled: Sequence[int], admitted: np.ndarray, panel: _Panel,
) -> _Retained:
    """Each sampled admitted Analysis's regions and blocks, by the calls completion makes
    (`analysis_regions`, `region_blocks`), refused where completion would refuse."""
    try:
        regions = analysis_regions(
            analyses, [i for i in sampled if admitted[i]], panel.table, panel.cis_window_bp,
            panel.coverage,
        )
    except RegionTableError as exc:
        raise ObservedViewRefused(str(exc)) from exc
    return _Retained(regions, region_blocks(regions, panel.path, panel.ancestry).blocks)


class _RegionRows:
    """The completed-axis rows of an Analysis's regions: the panel variants of the blocks
    they overlap that lie inside them, each block's rows and positions resolved once."""

    def __init__(self, target_rows: _TargetRows) -> None:
        self._target_rows = target_rows
        self._by_block: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def of(self, blocks: Sequence[LDBlock], regions: Sequence[Region]) -> np.ndarray:
        parts = []
        for block in blocks:
            if block.block_id not in self._by_block:
                alids = [canonical_panel_alid(snp) for snp in block.snp_ids]
                self._by_block[block.block_id] = (
                    self._target_rows.aligned(alids), panel_positions(block)
                )
            rows, positions = self._by_block[block.block_id]
            parts.append(rows[inside(regions, block.chrom, positions) & (rows >= 0)])
        return np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)


def _quality_blocks(target: OpenGWASDBStore) -> dict[int, set[str]]:
    """The blocks `completion_quality` records, per Analysis."""
    blocks: dict[int, set[str]] = {}
    with closing(target.index_connection()) as db:
        try:
            rows = db.execute("SELECT analysis_index, block_id FROM completion_quality")
            for analysis_index, block_id in rows:
                blocks.setdefault(int(analysis_index), set()).add(str(block_id))
        except sqlite3.OperationalError as exc:
            raise ObservedViewRefused(
                f"target store has no completion_quality table: {exc}"
            ) from exc
    return blocks


# --- the comparison -----------------------------------------------------------


def _admitted(pair: _Pair, panel: _Panel) -> np.ndarray:
    """Per Analysis, whether completion imputed it: `completed_against` is the ancestry or empty."""
    target = _sorted_records(pair.target_path)
    foreign = {a.completed_against for a in target} - {"", panel.ancestry}
    if foreign:
        raise ObservedViewRefused(
            f"target analyses are completed against {sorted(foreign)}, not {panel.ancestry}"
        )
    return np.array([a.completed_against != "" for a in target], dtype=bool)


def _flat_offsets(path: Path) -> np.ndarray:
    group = store_arrays.open_group(path / RAGGED_ZARR_PATH)
    return np.asarray(group["offsets"][:], dtype=np.int64)


@dataclass(frozen=True)
class _Sample:
    """What the per-Analysis comparison needs beyond the pair: the regions."""

    admitted: np.ndarray
    retained: _Retained
    rows: _RegionRows
    quality: dict[int, set[str]]


def _sample_and_compare(
    pair: _Pair, source: Sequence[Analysis], axis: _AxisMap, panel: _Panel,
    seed: int, n_random: int,
) -> dict[str, Any]:
    admitted = _admitted(pair, panel)
    chosen = _choose_analyses(
        source, np.diff(_flat_offsets(pair.source_path)), admitted, seed=seed, n_random=n_random
    )
    sampled = sorted(chosen)
    sample = _Sample(
        admitted=admitted,
        retained=_retained(source, sampled, admitted, panel),
        rows=_RegionRows(_TargetRows(pair.target_path, pair.n_target_rows)),
        quality=_quality_blocks(pair.target),
    )
    totals = dict.fromkeys(_TOTAL_KEYS, 0)
    max_diff = dict.fromkeys(_DIFF_PLANES, 0.0)
    listing: list[dict[str, Any]] = []
    for i in sampled:
        counts, diffs = _compare_analysis(pair, axis, sample, i)
        for key in totals:
            totals[key] += counts[key]
        for plane in _DIFF_PLANES:
            max_diff[plane] = max(max_diff[plane], diffs[plane])
        entry = _listing_entry(source[i], i, chosen[i], bool(admitted[i]), counts)
        entry["blocks"] = sorted(b.block_id for b in sample.retained.blocks.get(i, []))
        listing.append(entry)
    return {
        "sample": {
            "analyses_checked": len(sampled),
            "non_admitted": sum(not admitted[i] for i in sampled),
            "by_stratum": {s: sum(s in strata for strata in chosen.values()) for s in STRATA},
            "analyses": listing,
        },
        "totals": totals,
        "max_abs_diff": max_diff,
    }


def _compare_analysis(
    pair: _Pair, axis: _AxisMap, sample: _Sample, analysis: int
) -> tuple[dict[str, int], dict[str, float]]:
    retained = sample.retained.blocks.get(analysis, [])
    region = sample.rows.of(retained, sample.retained.regions.get(analysis, ()))
    counts, diffs = _check_analysis(
        pair, axis, analysis, admitted=bool(sample.admitted[analysis]), region=region,
    )
    counts["regions"] = len(retained)
    counts["quality_blocks_not_in_regions"] = len(
        sample.quality.get(analysis, set()) - {b.block_id for b in retained}
    )
    return counts, diffs


def _listing_entry(
    analysis: Analysis, index: int, strata: set[str], admitted: bool, counts: dict[str, int]
) -> dict[str, Any]:
    return {
        "analysis_index": index, "analysis_id": analysis.analysis_id,
        "strata": sorted(strata), "admitted": admitted, "regions": counts["regions"],
        "observed_in_source": counts["observed_in_source"],
        "imputed": counts["imputed"], "nan_rows": counts["nan_rows"],
        "failing": [k for k in FAILING_COUNTERS if counts[k]],
    }


def _unsampled_record() -> dict[str, Any]:
    """The sample was skipped, so nothing it would have counted is zero (issue #375).

    `analyses_checked` is 0 because no Analysis was read; the strata, `totals` and
    `max_abs_diff` are `null`, a value nobody measured rather than a measured zero. The
    keys stay, so a record's shape does not depend on whether the sample ran.
    """
    return {
        "sample": {
            "analyses_checked": 0, "non_admitted": 0,
            "by_stratum": dict.fromkeys(STRATA), "analyses": [],
        },
        "totals": dict.fromkeys(_TOTAL_KEYS),
        "max_abs_diff": dict.fromkeys(_DIFF_PLANES),
    }


def _failures(record: dict[str, Any], axis: _AxisMap) -> list[str]:
    failures = []
    if axis.unmapped:
        failures.append("parent_rows_unmapped")
    if not record["analyses"]["identical"]:
        failures.append("analyses_mismatch")
    failures.extend(name for name in FAILING_COUNTERS if record["totals"][name])
    return failures


def _decoded(reader: CsrReader, analysis: int, path: Path) -> Inspection:
    """An Analysis's decoded cells, invalid ones included; one that cannot be decoded
    cannot be compared."""
    try:
        return reader.inspect(Cells.analysis(analysis))
    except Exception as exc:
        raise ObservedViewRefused(
            f"store {path} analysis {analysis} cannot be decoded: {exc}"
        ) from exc


def _locate(
    target_vi: np.ndarray, mapped: np.ndarray
) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Where each source association (by its completed-axis row) sits in the target's rows.

    Returns whether it is present, its position, and a gather of a target-row array at those
    positions. An Analysis emptied in the target has nothing to gather: every source cell is
    absent and the gather is zeros, which `present` masks.
    """
    order = np.argsort(target_vi, kind="stable")
    if not len(order):
        absent = np.zeros(len(mapped), dtype=np.int64)
        return (
            np.zeros(len(mapped), dtype=bool),
            absent,
            lambda values: np.zeros(len(mapped), dtype=values.dtype),
        )
    slot = np.minimum(np.searchsorted(target_vi[order], mapped), len(order) - 1)
    at = order[slot]
    return target_vi[order][slot] == mapped, at, lambda values: values[at]


def _check_analysis(
    pair: _Pair,
    axis: _AxisMap,
    analysis: int,
    *,
    admitted: bool,
    region: np.ndarray,
) -> tuple[dict[str, int], dict[str, float]]:
    """One Analysis's cells, compared by ALID through the exhaustive axis map. The
    target's imputed flags are its mask as stored, read with its cells."""
    src = _decoded(pair.source_reader, analysis, pair.source_path)
    tgt = _decoded(pair.target_reader, analysis, pair.target_path)
    raw_imputed = np.zeros(0, dtype=np.uint8) if tgt.flags is None else tgt.flags
    if len(src.variant_index) and int(src.variant_index.max()) >= len(axis.source_to_target):
        raise ObservedViewRefused(
            f"source store {pair.source_path} holds a variant index past its axis"
        )
    source_cells = _Cells(src.z, src.se, src.eaf)
    target_cells = _Cells(tgt.z, tgt.se, tgt.eaf)
    target_vi = np.asarray(tgt.variant_index, dtype=np.int64)
    mapped = axis.source_to_target[np.asarray(src.variant_index, dtype=np.int64)]
    present, at, gather = _locate(target_vi, mapped)

    imputed = raw_imputed.astype(np.uint8)
    source_observed = _observed(source_cells)
    target_observed = _observed(target_cells)
    both = source_observed & present
    matched = np.zeros(len(target_vi), dtype=bool)
    matched[at[both]] = True
    outside, added = ~np.isin(target_vi, region), _added(len(target_vi), at[present])
    # The target's States and counts are the readers' own (spec §15, ADR 0070).
    source_tally, target_tally = src.tally(), tgt.tally()
    clean = tgt.state == State.OBSERVED
    is_imputed = raw_imputed == 1
    counts = {
        "observed_in_source": int(source_observed.sum()),
        "imputed": int(is_imputed.sum()),
        "nan_rows": target_tally.count(State.MISSING),
        "imputed_mask_invalid": target_tally.flags_not_binary,
        **_marker_counts(source_tally, target_tally),
        # Present-but-infinite cells on both sides: reported, never a failure (issue #337).
        "infinite_cells": count_infinite_cells(source_cells) + count_infinite_cells(target_cells),
        "target_variants_unordered": int((np.diff(target_vi) <= 0).sum()),
        "observed_missing_in_target": int(
            (source_observed & ~(present & gather(target_observed))).sum()
        ),
        "observed_marked_imputed": int((both & (gather(imputed) != 0)).sum()),
        "target_observed_not_in_source": int((clean & ~matched).sum()),
        "imputed_in_non_admitted": 0 if admitted else int(is_imputed.sum()),
        "imputed_outside_retained": int((is_imputed & outside).sum()) if admitted else 0,
        "missing_outside_retained": int((added & (tgt.state == State.MISSING) & outside).sum()),
    }
    diffs: dict[str, float] = {}
    for plane in _DIFF_PLANES:
        counts[f"{plane}_mismatch"], diffs[plane] = _plane_difference(
            getattr(source_cells, plane)[both], getattr(target_cells, plane)[at[both]]
        )
    return counts, diffs


def _added(n_target: int, from_source: np.ndarray) -> np.ndarray:
    """Which target rows completion added: those no source row maps to (`from_source`).

    A missing target row is either a source's stored missing row, which completion
    carries wherever it lies, or one completion added, which must lie inside one of the
    Analysis's regions (issue #430).
    """
    added = np.ones(n_target, dtype=bool)
    added[from_source] = False
    return added


def _marker_counts(source: Tally, target: Tally) -> dict[str, int]:
    """The counters both sides' markers decide, over an Analysis's whole sequence.

    A cell with one of Z and SE missing rather than both is `PARTIAL`; one whose marker
    is a NaN other than the canonical one is `NON_CANONICAL`, an invalid state no reader
    accepts. `isnan` alone reads the second as missing, which is why it has its own
    counter rather than falling into `nan_rows` (spec §15, issue #373).
    """
    return {
        "z_se_missingness_mismatch": (
            source.count(State.PARTIAL) + target.count(State.PARTIAL)
        ),
        "non_canonical_marker": (
            source.count(State.NON_CANONICAL) + target.count(State.NON_CANONICAL)
        ),
    }
