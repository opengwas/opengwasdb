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
  and every imputed association lies inside one of the Analysis's retained regions.
  An Analysis the ancestry filter did not admit is carried through observed-only
  (ADR 0028), so it must hold no imputed cell at all.

**Retained regions** (ADR 0014, ADR 0039) are not stored as coordinates: a Store records
only block ids and counts in `completion_quality`, and block extents live in the LD
panel. They are rebuilt here the way completion chose them, from the panel, the
Analysis's Trait position (or, with none, the blocks it already holds enough
observations in) and the `cis_window_bp` the manifest records. The check therefore
takes the panel and refuses without it: a check that quietly skipped the region
criterion would pass a Store whose imputed cells sit anywhere. Block ids that
`completion_quality` records but the rebuilt regions lack fail the check, so a panel
other than the one completion used is noticed rather than trusted.

Rows completion writes inside a retained region that are neither observed nor imputed
(NaN, the panel positions it could not fill) are counted in the verdict and fail
nothing. The verdict records the seed, the sample and the counts, so a pass is a
measured zero; a refusal (`ObservedViewRefused`) writes none.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.completion.ld_panel import (
    LDBlock,
    LdPanelNotFoundError,
    blocks_over_variants,
    canonical_panel_alid,
    check_panel_has_chromosomes,
)
from opengwasdb.layouts.ragged.complete import _cis_window_blocks, _has_trait_position
from opengwasdb.layouts.ragged.zarr_csr import RAGGED_ZARR_PATH, RaggedCSRReader
from opengwasdb.model.analyses import Analysis, read_analysis_records
from opengwasdb.model.enums import CompletionState, PrimaryStorageLayout
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store.open import OpenGWASDBStore, open_store
from opengwasdb.validation.observed_view import (
    _STATE_NAMES,
    ObservedViewRefused,
    ObservedViewVerdict,
    _AlidIndex,
    _AxisMap,
    _Cells,
    _map_axis,
    _observed,
    _partial_cells,
    _plane_difference,
    _store_record,
)
from opengwasdb.variants import VariantAxis, is_indexable_alid

__all__ = ["FAILING_COUNTERS", "STRATA", "check", "involves_ragged"]

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
    "imputed_in_non_admitted",
    "imputed_mask_invalid",
    "z_se_missingness_mismatch",
    "target_variants_unordered",
    "quality_blocks_not_in_regions",
)
_INFORMATIONAL_COUNTERS = ("observed_in_source", "imputed", "nan_rows", "regions")
_TOTAL_KEYS = (*_INFORMATIONAL_COUNTERS, *FAILING_COUNTERS)
_DIFF_PLANES = ("z", "se", "eaf")


def involves_ragged(source_path: Path, target_path: Path) -> bool:
    """Whether either Store is Ragged, so this module owns the pair (a mixed one is refused)."""
    for path in (source_path, target_path):
        try:
            layout = open_store(path).manifest.primary_layout
        except (OSError, ValueError, KeyError):
            continue  # the Dense path reports a Store it cannot open
        if layout is PrimaryStorageLayout.RAGGED:
            return True
    return False


def check(
    source_path: str | Path,
    target_path: str | Path,
    *,
    seed: int,
    n_random: int,
    ld_panel: str | Path | None,
    ancestry: str | None,
) -> ObservedViewVerdict:
    """Compare an Observed-Only Ragged Store with the Reference-Completed one made from it.

    Raises `ObservedViewRefused` when the pair cannot be checked (mixed layouts, a source
    that is not Observed-Only, a target that is not Reference-Completed, no LD panel, or
    a panel or ancestry the completion was not run with). `ancestry` defaults to the one
    the completed manifest records. A mismatch is a verdict with `passed` false.
    """
    started = time.time()
    pair = _open_pair(Path(source_path), Path(target_path))
    completion = _completion_provenance(pair.target, pair.target_path)
    panel = _open_panel(ld_panel, ancestry, completion)
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
    source_reader: RaggedCSRReader
    target_reader: RaggedCSRReader
    n_source_rows: int
    n_target_rows: int


def _open_pair(source_path: Path, target_path: Path) -> _Pair:
    source, target = _open(source_path, "source"), _open(target_path, "target")
    layouts = {source.manifest.primary_layout, target.manifest.primary_layout}
    if layouts != {PrimaryStorageLayout.RAGGED}:
        raise ObservedViewRefused(
            f"the pair mixes layouts (source primary_layout={source.manifest.primary_layout.value}"
            f", target primary_layout={target.manifest.primary_layout.value}); the check "
            "compares two Stores of one layout"
        )
    _require_state(source, source_path, "source", CompletionState.OBSERVED_ONLY)
    _require_state(target, target_path, "target", CompletionState.REFERENCE_COMPLETED)
    if "imputed" not in store_arrays.open_group(target_path / RAGGED_ZARR_PATH):
        raise ObservedViewRefused(f"target store {target_path} has no `imputed` mask to check")
    rows = []
    for path in (source_path, target_path):
        with closing(VariantAxis(path)) as axis:
            rows.append(axis.n_variants)
        if rows[-1] == 0:
            raise ObservedViewRefused(f"store {path} has no variants to check")
    return _Pair(
        source_path, target_path, source, target,
        RaggedCSRReader(source_path), RaggedCSRReader(target_path), rows[0], rows[1],
    )


def _open(path: Path, role: str) -> OpenGWASDBStore:
    try:
        return open_store(path)
    except (OSError, ValueError, KeyError) as exc:
        raise ObservedViewRefused(f"cannot open {role} store {path}: {exc}") from exc


def _require_state(
    store: OpenGWASDBStore, path: Path, role: str, wanted: CompletionState
) -> None:
    state = store.manifest.completion_state
    if state is not wanted:
        raise ObservedViewRefused(
            f"{role} store {path} is not {_STATE_NAMES[wanted]} (completion_state={state.value})"
        )


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
    path: Path
    ancestry: str
    cis_window_bp: int


def _open_panel(
    ld_panel: str | Path | None, ancestry: str | None, completion: dict[str, Any]
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
    return _Panel(Path(ld_panel), recorded, int(completion["cis_window_bp"]))


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
    trait_less = [i for i, a in enumerate(analyses) if not _has_trait_position(a)]
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

    def rows(self, alids: Sequence[str]) -> np.ndarray:
        found = [self._loose[a] for a in alids if a in self._loose]
        indexed = [a for a in alids if is_indexable_alid(a)]
        if indexed and self._index.alids.size:
            keys = np.array([a.encode("utf-8") for a in indexed], dtype=self._index.alids.dtype)
            slot = np.minimum(np.searchsorted(self._index.alids, keys), self._index.alids.size - 1)
            hit = self._index.alids[slot] == keys
            found.extend(np.asarray(self._index.rows)[slot[hit]].tolist())
        return np.unique(np.asarray(found, dtype=np.int64))


def _retained_blocks(
    pair: _Pair, analyses: Sequence[Analysis], sampled: Sequence[int],
    admitted: np.ndarray, panel: _Panel,
) -> dict[int, list[LDBlock]]:
    """Each sampled admitted Analysis's retained blocks, chosen as completion chose them.

    An Analysis with a Trait position is completed against the blocks touching its cis
    window; one without, against the blocks holding enough of its own observations, which
    is resolved for all of them in one ALID-keyed pass so the panel is read once.
    """
    blocks: dict[int, list[LDBlock]] = {}
    by_window: dict[tuple[str, int, int], list[LDBlock]] = {}
    trait_less: list[int] = []
    for i in sampled:
        if not admitted[i]:
            continue
        a = analyses[i]
        if _has_trait_position(a):
            window = (a.trait_chr, int(a.trait_bp), panel.cis_window_bp)
            if window not in by_window:
                by_window[window] = _cis_window_blocks(a, panel.path, panel.ancestry, window[2])
            blocks[i] = by_window[window]
        else:
            trait_less.append(i)
    if trait_less:
        blocks.update(_blocks_over_own_variants(pair, trait_less, panel))
    return blocks


def _blocks_over_own_variants(
    pair: _Pair, indices: Sequence[int], panel: _Panel
) -> dict[int, list[LDBlock]]:
    source_rows = {i: pair.source_reader.variant_indices(i) for i in indices}
    wanted = sorted({int(v) for rows in source_rows.values() for v in rows})
    with closing(VariantAxis(pair.source_path)) as axis:
        records = axis.by_indices(wanted)
    analyses_by_alid: dict[str, list[int]] = {}
    chromosomes: set[str] = set()
    for i, rows in source_rows.items():
        for v in rows.tolist():
            analyses_by_alid.setdefault(records[v].alid, []).append(i)
            chromosomes.add(records[v].chromosome)
    found = blocks_over_variants(panel.path, panel.ancestry, analyses_by_alid, sorted(chromosomes))
    return {i: found.get(i, []) for i in indices}


class _RegionRows:
    """The completed-axis rows of each block's panel positions, resolved once per block."""

    def __init__(self, target_rows: _TargetRows) -> None:
        self._target_rows = target_rows
        self._by_block: dict[str, np.ndarray] = {}

    def of(self, blocks: Sequence[LDBlock]) -> np.ndarray:
        parts = []
        for block in blocks:
            if block.block_id not in self._by_block:
                alids = [a for a in map(canonical_panel_alid, block.snp_ids) if a is not None]
                self._by_block[block.block_id] = self._target_rows.rows(alids)
            parts.append(self._by_block[block.block_id])
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
    """What the per-Analysis comparison needs beyond the pair: regions and the mask."""

    admitted: np.ndarray
    blocks: dict[int, list[LDBlock]]
    regions: _RegionRows
    quality: dict[int, set[str]]
    target_offsets: np.ndarray
    imputed: Any  # the completed Store's `imputed` array


def _sample_and_compare(
    pair: _Pair, source: Sequence[Analysis], axis: _AxisMap, panel: _Panel,
    seed: int, n_random: int,
) -> dict[str, Any]:
    admitted = _admitted(pair, panel)
    chosen = _choose_analyses(
        source, np.diff(_flat_offsets(pair.source_path)), admitted, seed=seed, n_random=n_random
    )
    sampled = sorted(chosen)
    target_group = store_arrays.open_group(pair.target_path / RAGGED_ZARR_PATH)
    sample = _Sample(
        admitted=admitted,
        blocks=_retained_blocks(pair, source, sampled, admitted, panel),
        regions=_RegionRows(_TargetRows(pair.target_path, pair.n_target_rows)),
        quality=_quality_blocks(pair.target),
        target_offsets=_flat_offsets(pair.target_path),
        imputed=target_group["imputed"],
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
        listing.append(_listing_entry(source[i], i, chosen[i], bool(admitted[i]), counts))
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
    retained = sample.blocks.get(analysis, [])
    start, end = int(sample.target_offsets[analysis]), int(sample.target_offsets[analysis + 1])
    counts, diffs = _check_analysis(
        pair, axis, analysis, np.asarray(sample.imputed[start:end]),
        admitted=bool(sample.admitted[analysis]), region=sample.regions.of(retained),
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
    return {
        "sample": {
            "analyses_checked": 0, "non_admitted": 0,
            "by_stratum": dict.fromkeys(STRATA, 0), "analyses": [],
        },
        "totals": dict.fromkeys(_TOTAL_KEYS, 0),
        "max_abs_diff": dict.fromkeys(_DIFF_PLANES, 0.0),
    }


def _failures(record: dict[str, Any], axis: _AxisMap) -> list[str]:
    failures = []
    if axis.unmapped:
        failures.append("parent_rows_unmapped")
    if not record["analyses"]["identical"]:
        failures.append("analyses_mismatch")
    failures.extend(name for name in FAILING_COUNTERS if record["totals"][name])
    return failures


def _decoded(reader: RaggedCSRReader, analysis: int, path: Path) -> Any:
    """An Analysis's decoded cells; one that cannot be decoded cannot be compared."""
    try:
        return reader.get_analysis(analysis)
    except ValueError as exc:
        raise ObservedViewRefused(
            f"store {path} analysis {analysis} cannot be decoded: {exc}"
        ) from exc


def _check_analysis(
    pair: _Pair,
    axis: _AxisMap,
    analysis: int,
    raw_imputed: np.ndarray,
    *,
    admitted: bool,
    region: np.ndarray,
) -> tuple[dict[str, int], dict[str, float]]:
    """One Analysis's cells, compared by ALID through the exhaustive axis map."""
    src = _decoded(pair.source_reader, analysis, pair.source_path)
    tgt = _decoded(pair.target_reader, analysis, pair.target_path)
    if len(src.variant_index) and int(src.variant_index.max()) >= len(axis.source_to_target):
        raise ObservedViewRefused(
            f"source store {pair.source_path} holds a variant index past its axis"
        )
    source_cells = _Cells(src.z, src.se, src.eaf)
    target_cells = _Cells(tgt.z, tgt.se, tgt.eaf)
    target_vi = np.asarray(tgt.variant_index, dtype=np.int64)
    order = np.argsort(target_vi, kind="stable")
    mapped = axis.source_to_target[np.asarray(src.variant_index, dtype=np.int64)]
    slot = np.minimum(np.searchsorted(target_vi[order], mapped), max(len(order) - 1, 0))
    present = (target_vi[order][slot] == mapped) if len(order) else np.zeros(len(mapped), bool)
    at = order[slot] if len(order) else np.zeros(len(mapped), dtype=np.int64)

    imputed = raw_imputed.astype(np.uint8)
    source_observed = _observed(source_cells)
    target_observed = _observed(target_cells)
    both = source_observed & present
    matched = np.zeros(len(target_vi), dtype=bool)
    matched[at[both]] = True
    clean = (raw_imputed == 0) & target_observed
    is_imputed = raw_imputed == 1
    counts = {
        "observed_in_source": int(source_observed.sum()),
        "imputed": int(is_imputed.sum()),
        "nan_rows": int((~np.isfinite(tgt.z) & ~np.isfinite(tgt.se) & (raw_imputed == 0)).sum()),
        "imputed_mask_invalid": int(((raw_imputed != 0) & (raw_imputed != 1)).sum()),
        "z_se_missingness_mismatch": _partial_cells(source_cells) + _partial_cells(target_cells),
        "target_variants_unordered": int((np.diff(target_vi) <= 0).sum()),
        "observed_missing_in_target": int(
            (source_observed & ~(present & target_observed[at])).sum()
        ),
        "observed_marked_imputed": int((both & (imputed[at] != 0)).sum()),
        "target_observed_not_in_source": int((clean & ~matched).sum()),
        "imputed_in_non_admitted": 0 if admitted else int(is_imputed.sum()),
        "imputed_outside_retained": (
            int((is_imputed & ~np.isin(target_vi, region)).sum()) if admitted else 0
        ),
    }
    diffs: dict[str, float] = {}
    for plane in _DIFF_PLANES:
        counts[f"{plane}_mismatch"], diffs[plane] = _plane_difference(
            getattr(source_cells, plane)[both], getattr(target_cells, plane)[at[both]]
        )
    return counts, diffs
