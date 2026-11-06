"""The prominence and LD-block rules over #403's Analyses, shared set and clumps (#409).

Per Analysis: read the 5e-4 Top-Hit tier, observed rows only. `run.read_tier`
issues two Top-Hit Queries for it, the observed-only one the profile is made of
and an unrestricted one that counts the imputed rows, so the read is checked
complete. Its rows at the 5e-8 cut are the candidates, and all of it is the
profile. The candidates' shared set must equal the shared set of the 5e-8 tier,
read now and built as #403 built it, and both must reproduce #403's committed
summaries: its accounting, its HapMap3 row counts, and its distance rules
recomputed on the same clumps (plink2, r^2 0.01). #403 committed no digest of
its rows, so the digest published is of the current reads.

The map is used only for the shared set's cM, so the row sets are identical.
For C2, the HapMap3 restriction applies to the candidates and the profile alike.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from sys import stderr
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd

from opengwasdb.top_hits import z_critical

from .blocks import DEFINITION as BLOCK_DEFINITION
from .blocks import Blocks, block_leads, read_blocks
from .compare import classify
from .genetic_map import LoadedMap, file_digest
from .panel import Panel, PanelVariants
from .profile import DEFINITION as PROFILE_DEFINITION
from .profile import Profile, Threshold, prominence_leads
from .profile_judge import judge
from .profile_valleys import DEFINITION as VALLEY_DEFINITION
from .profile_valleys import read_valley_cells, sample_valleys, summarise_valley_cells
from .rule import locus_leads
from .run import (
    SharedSet,
    TierRows,
    _summary,
    build_shared_set,
    check_tier,
    clump_shared_set,
    read_tier,
)
from .sweep import (
    BP_ONLY,
    PRIMARY_R2,
    TIER,
    W_GRID,
    cached_record,
    code_identity,
    panel_content_digests,
    read_hm3,
)

PROFILE_TIER = 5e-4
BIN_GRID = (10_000, 50_000)
THRESHOLDS = (Threshold("relative", 0.25), Threshold("relative", 0.5),
              Threshold("relative", 0.75), Threshold("fixed", 0.0))
BLOCK_ANCESTRY = "EUR"
R2 = float(PRIMARY_R2)
# #403's distance rules recomputed here and checked against its records: bp-only at every W,
# and the D 0.5 cM / W 1 Mb point the #402 pilot used.
BASELINES = (*((BP_ONLY, w) for w in W_GRID), (0.5, 1_000_000))


@dataclass(frozen=True)
class _KeptVariants(PanelVariants):
    """A chromosome's panel variants whose allele-set table is built once."""

    @cached_property
    def _allele_sets(self) -> pd.DataFrame:
        return super().by_allele_set()

    def by_allele_set(self) -> pd.DataFrame:
        return self._allele_sets


class ReadOncePanel(Panel):
    """The panel, each chromosome's variants read once for the whole sweep.

    Each Analysis matches two shared sets to the panel (its own and #403's, rebuilt), and the
    pvars and their allele-set tables dominate that time, so they are built once.
    """

    def __init__(self, directory: Path) -> None:
        super().__init__(directory)
        self._kept: dict[str, PanelVariants] = {}

    def variants(self, chromosome: str) -> PanelVariants:
        if chromosome not in self._kept:
            read = super().variants(chromosome)
            self._kept[chromosome] = _KeptVariants(read.position, read.variant_id, read.ref,
                                                   read.alt)
        return self._kept[chromosome]


class SharedSetMismatch(ValueError):
    """The rows this sweep ranks do not reproduce the 5e-8 tier's, or #403's summaries."""


class BaselineMismatch(ValueError):
    """A distance rule recomputed here disagrees with #403's committed record."""


@dataclass(frozen=True)
class Rule:
    """A prominence rule (bin width and threshold), or the LD-block rule."""

    name: str
    bin_bp: int | None = None
    threshold: Threshold | None = None

    def label(self) -> dict[str, Any]:
        if self.threshold is None or self.bin_bp is None:
            return {"rule": self.name, "label": f"LD blocks ({BLOCK_ANCESTRY})"}
        return {"rule": self.name, "bin_bp": self.bin_bp, "threshold": self.threshold.kind,
                "value": self.threshold.value,
                "label": f"B {self.bin_bp // 1000} kb, {self.threshold.label()}"}


def rule_grid() -> list[Rule]:
    return [*(Rule("prominence", b, t) for b in BIN_GRID for t in THRESHOLDS), Rule("ld_blocks")]


def shared_set_identity(store_id: str, analysis_id: str,
                        variant_index: np.ndarray) -> dict[str, Any]:
    """A shared set's row count, and a sha256 of its Store, Analysis and sorted Store Variant
    Indices: the same rows in any order give the same digest."""
    rows = np.sort(np.asarray(variant_index, dtype="int64"))
    digest = hashlib.sha256(f"{store_id}\0{analysis_id}\0".encode() + rows.tobytes())
    return {"rows": int(rows.size), "sha256": digest.hexdigest()}


def check_same_shared_set(mine: dict[str, Any], reference: dict[str, Any]) -> None:
    """Refuse a shared set that is not the reference's, row for row."""
    if mine != reference:
        raise SharedSetMismatch(f"the shared set ({mine['rows']:,} rows, {mine['sha256'][:12]}) "
                                f"differs from the 5e-8 tier's ({reference['rows']:,} rows, "
                                f"{reference['sha256'][:12]})")


def reference_shared_set(reference: TierRows, store_id: str, analysis_id: str,
                         committed: dict[str, Any], genetic_map: LoadedMap,
                         panel: Panel) -> dict[str, Any]:
    """The shared set of the 5e-8 tier, built as #403 built it, after its accounting is checked
    against the accounting #403 committed."""
    shared, accounting = build_shared_set(reference, panel, genetic_map.map)
    if accounting != committed["rows"]:
        raise SharedSetMismatch(f"{analysis_id}: the 5e-8 shared set rebuilt now differs from "
                                "#403's committed accounting")
    return shared_set_identity(store_id, analysis_id, shared.rows.variant_index)


def _on(hm3: frozenset[str], alid: np.ndarray) -> np.ndarray:
    return np.fromiter((a in hm3 for a in alid.tolist()), dtype=bool, count=alid.size)


def _n_leads(lead_of: np.ndarray) -> int:
    return int(np.sum(lead_of == np.arange(lead_of.size)))


def _per_lead(shared: SharedSet, lead_of: np.ndarray, index_of: np.ndarray) -> dict[str, Any]:
    """#403's per-lead record at r^2 0.01, without the pair lists."""
    summary = _summary(shared, classify(lead_of, index_of), lead_of, index_of)
    return {k: v for k, v in summary.items()
            if k not in ("spurious_split_pairs", "lost_signal_pairs")}


def check_baselines(shared: SharedSet, index_of: np.ndarray, in_hm3: np.ndarray,
                    committed: dict[str, Any]) -> list[dict[str, Any]]:
    """Recompute #403's distance rules on this shared set and clumps; refuse any difference."""
    rows = shared.rows
    checked = []
    for d_cm, w_bp in BASELINES:
        label = {"d_cm": "bp_only" if d_cm == BP_ONLY else d_cm, "w_bp": w_bp}
        theirs = next(c for c in committed["configs"]
                      if (c["d_cm"], c["w_bp"]) == (label["d_cm"], w_bp))
        lead_of = locus_leads(rows.chromosome, rows.position, shared.cm, rows.abs_z,
                              rows.variant_index, d_cm=d_cm, w_bp=w_bp)
        sparse = rows.take(in_hm3)
        lead_hm3 = locus_leads(sparse.chromosome, sparse.position, shared.cm[in_hm3],
                               sparse.abs_z, sparse.variant_index, d_cm=d_cm, w_bp=w_bp)
        mine = {"per_lead": _per_lead(shared, lead_of, index_of),
                "density": {"leads_full": _n_leads(lead_of), "leads_hm3": _n_leads(lead_hm3)}}
        expected = {"per_lead": {k: v for k, v in theirs["by_r2"][PRIMARY_R2].items()
                                 if k in mine["per_lead"]}, "density": theirs["density"]}
        if json.loads(json.dumps(mine)) != expected:
            raise BaselineMismatch(f"{committed['analysis']['analysis_id']} at {label}: the "
                                   "recomputed distance rule differs from #403's record")
        checked.append(label)
    return checked


@dataclass(frozen=True)
class Candidates:
    """Candidates and the profile that shapes them, at one variant density."""

    rows: TierRows
    profile: Profile


def apply(rule: Rule, at: Candidates, blocks: Blocks) -> tuple[np.ndarray, dict[str, Any]]:
    """A rule's leads, and what it records beside them: splits, or rows outside blocks."""
    r = at.rows
    if rule.threshold is None or rule.bin_bp is None:
        found = block_leads(r.chromosome, r.position, r.abs_z, r.variant_index, blocks)
        return found.lead_of, {"outside": {"rows": found.outside_rows,
                                           "stretches": found.outside_stretches}}
    result = prominence_leads(r.chromosome, r.position, r.abs_z, r.variant_index, at.profile,
                              bin_bp=rule.bin_bp, threshold=rule.threshold)
    kind = result.splits.kind
    return result.lead_of, {"splits": result.splits, "valleys": {
        "splits": int(kind.size), "no_row": int(np.sum(kind == "no_row")),
        "low_z": int(np.sum(kind == "low_z"))}}


def rule_record(rule: Rule, shared: SharedSet, full: Candidates, sparse: Candidates,
                index_of: np.ndarray, blocks: Blocks) -> tuple[dict[str, Any], list[list[Any]]]:
    """One rule on one Analysis, and the (chromosome, bin) of each of its no-row valleys."""
    lead_of, extra = apply(rule, full, blocks)
    lead_hm3, extra_hm3 = apply(rule, sparse, blocks)
    splits = extra.pop("splits", None)
    extra_hm3.pop("splits", None)
    valleys = [] if splits is None else sorted(
        {(str(c), int(b)) for c, b, k in zip(splits.chromosome, splits.bin, splits.kind,
                                             strict=True) if k == "no_row"})
    record = rule.label() | {
        "density": {"leads_full": _n_leads(lead_of), "leads_hm3": _n_leads(lead_hm3)},
        "by_r2": {PRIMARY_R2: _per_lead(shared, lead_of, index_of)},
        **extra, **{f"{k}_hm3": v for k, v in extra_hm3.items()}}
    return record, [list(v) for v in valleys]


def analyse_profile(store: Path, store_id: str, committed: dict[str, Any],
                    genetic_map: LoadedMap, panel: Panel, blocks: Blocks, hm3: frozenset[str],
                    *, work_dir: Path, threads: int) -> dict[str, Any]:
    """One Analysis: its 5e-4 tier and its 5e-8 tier read from the Store, then scored."""
    analysis_id = committed["analysis"]["analysis_id"]
    tier, identity = read_tier(store, analysis_id, PROFILE_TIER)
    reference, _ = read_tier(store, analysis_id, TIER)
    return score_profile(tier, identity, reference, committed, genetic_map, panel, blocks, hm3,
                         store_id=store_id, work_dir=work_dir, threads=threads)


def score_profile(tier: TierRows, identity: dict[str, Any], reference: TierRows,
                  committed: dict[str, Any], genetic_map: LoadedMap, panel: Panel,
                  blocks: Blocks, hm3: frozenset[str], *, store_id: str, work_dir: Path,
                  threads: int) -> dict[str, Any]:
    """One Analysis's 5e-4 tier scored: its candidates' shared set checked against the 5e-8
    tier's and #403's committed summaries, clumped, #403's distance rules reproduced, and every
    rule scored at full density and on HapMap3 alone."""
    analysis_id = committed["analysis"]["analysis_id"]
    check_tier(tier, identity, genetic_map, panel)
    at_tier = tier.abs_z >= z_critical(TIER)
    shared, accounting = build_shared_set(tier.take(at_tier), panel, genetic_map.map)
    mine = shared_set_identity(store_id, analysis_id, shared.rows.variant_index)
    check_same_shared_set(mine, reference_shared_set(reference, store_id, analysis_id, committed,
                                                     genetic_map, panel))
    if accounting != committed["rows"]:
        raise SharedSetMismatch(f"{analysis_id}: the candidates' accounting differs from #403's")
    work_dir.mkdir(parents=True, exist_ok=True)
    index_of = clump_shared_set(shared, panel, [R2], threshold=TIER, work_dir=work_dir,
                                threads=threads).index_of[R2]
    in_hm3 = _on(hm3, shared.rows.alid)
    profile = Profile(tier.chromosome, tier.position, tier.abs_z)
    full = Candidates(shared.rows, profile)
    sparse = Candidates(shared.rows.take(in_hm3), profile.take(_on(hm3, tier.alid)))
    density = {"rows_full": int(in_hm3.size), "rows_hm3": int(in_hm3.sum())}
    if density != committed["density"]:
        raise SharedSetMismatch(f"{analysis_id}: rows on HapMap3 {density}, but #403 "
                                f"{committed['density']}")
    baselines = check_baselines(shared, index_of, in_hm3, committed)
    scored = [rule_record(rule, shared, full, sparse, index_of, blocks) for rule in rule_grid()]
    return {
        "analysis": identity["analysis"], "store": store_id, "query": identity["query"],
        "profile": {"rows": int(tier.abs_z.size), "candidates": int(shared.rows.abs_z.size),
                    "at_5e8_outside_shared_set": int(at_tier.sum()) - int(shared.rows.abs_z.size),
                    "below_5e8": int((~at_tier).sum()),
                    "on_hm3": int(sparse.profile.abs_z.size)},
        "rows": accounting, "shared_set": mine | {"matches_403_summaries": True},
        "density": density,
        "index_variants": _n_leads(index_of), "baselines_equal_403": baselines,
        "rules": [record for record, _ in scored],
        "no_row_valleys": _no_row_valleys(rule_grid(), [valleys for _, valleys in scored]),
    }


def _no_row_valleys(rules: list[Rule], valleys: list[list[list[Any]]]) -> dict[str, list[Any]]:
    """The distinct (chromosome, bin) no-row valleys of every prominence rule, per bin width."""
    found: dict[str, set[tuple[str, int]]] = {}
    for rule, each in zip(rules, valleys, strict=True):
        if rule.bin_bp is not None:
            found.setdefault(f"{rule.bin_bp}", set()).update((c, b) for c, b in each)
    return {b: [list(v) for v in sorted(bins)] for b, bins in found.items()}


def profile_fingerprint(store: Path, analysis_id: str, genetic_map: LoadedMap,
                        panel: dict[str, Any], inputs: dict[str, Path],
                        code: dict[str, str]) -> dict[str, Any]:
    """Everything one Analysis's record is a function of."""
    return {
        "store": str(store), "analysis_id": analysis_id,
        "manifest_sha256": file_digest(store / "manifest.json", "sha256"),
        "genetic_map_sha256": genetic_map.identity["sha256"], "panel": panel,
        **{f"{name}_sha256": file_digest(path, "sha256") for name, path in inputs.items()},
        "profile_tier": PROFILE_TIER, "tier": TIER, "r2": R2, "blocks_ancestry": BLOCK_ANCESTRY,
        "rules": [r.label() for r in rule_grid()],
        "baselines": [[str(d), w] for d, w in BASELINES],
        "code_sha256": code["sha256"], "code_commit": code["commit"],
    }


def _merged_valleys(store_id: str, analysis_id: str,
                    by_bin: dict[str, list[list[Any]]]) -> list[tuple[str, str, int, str, int]]:
    return [(store_id, analysis_id, int(b), str(c), int(n))
            for b, found in by_bin.items() for c, n in found]


def check_inputs(selection: dict[str, Any], committed: dict[str, Any], genetic_map: LoadedMap,
                 hm3_path: Path) -> None:
    """Refuse a selection, map or HapMap3 list other than the ones #403's sweep used."""
    digest = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    checks = {"selection": (digest, committed["selection"]["sha256"]),
              "genetic map": (genetic_map.identity["sha256"], committed["genetic_map"]["sha256"]),
              "HapMap3": (file_digest(hm3_path, "sha256"), committed["hm3"]["sha256"])}
    for name, (here, there) in checks.items():
        if here != there:
            raise ValueError(f"the {name} is not the one #403's sweep used")


def _definitions() -> dict[str, Any]:
    return {"prominence": PROFILE_DEFINITION, "ld_blocks": BLOCK_DEFINITION,
            "valleys": VALLEY_DEFINITION,
            "shared_set": "the candidates' shared set must equal the shared set of the 5e-8 "
                          "tier, read now with #403's read_tier and built with its "
                          "build_shared_set: the same row count and sha256 of Store, Analysis and "
                          "sorted Store Variant Indices. Both must reproduce #403's committed "
                          "summaries: its accounting, its HapMap3 row counts and its distance "
                          "rules' records. #403 committed no digest, so the digest published is "
                          "of the current reads, and matches_403_summaries says only that the "
                          "summaries are reproduced"}


def _inputs(committed: dict[str, Any], committed_path: Path, genetic_map: LoadedMap,
            blocks_path: Path, blocks: Blocks) -> dict[str, Any]:
    return {"selection_sha256": committed["selection"]["sha256"],
            "sweep_403": {"path": str(committed_path),
                          "sha256": file_digest(committed_path, "sha256")},
            "genetic_map": genetic_map.identity, "panel": committed["panel"],
            "hm3": committed["hm3"],
            "blocks": {"path": str(blocks_path), "ancestry": BLOCK_ANCESTRY,
                       "sha256": file_digest(blocks_path, "sha256"), "blocks": blocks.count()},
            "profile_tier": PROFILE_TIER, "tier": TIER, "r2": PRIMARY_R2, "observed_only": True}


def _grid() -> dict[str, Any]:
    return {"bin_bp": list(BIN_GRID), "thresholds": [t.label() for t in THRESHOLDS],
            "rules": [r.label() for r in rule_grid()],
            "baselines": [{"d_cm": "bp_only" if d == BP_ONLY else d, "w_bp": w}
                          for d, w in BASELINES]}


@dataclass(frozen=True)
class SweepInputs:
    """What every Analysis's record is computed from."""

    stores_dir: Path
    genetic_map: LoadedMap
    panel: Panel
    blocks: Blocks
    hm3: frozenset[str]
    files: dict[str, Path]
    panel_digests: dict[str, Any]
    code: dict[str, str]


def _analyse_each(selection: dict[str, Any], committed: dict[str, Any], given: SweepInputs,
                  work_dir: Path, threads: int, only: int | None) -> tuple[
                      list[dict[str, Any]], list[tuple[str, str, int, str, int]],
                      list[dict[str, str]]]:
    """Every chosen Analysis #403 swept, cached; the rest listed as not swept."""
    swept = {(a["store"], a["analysis"]["analysis_id"]): a for a in committed["analyses"]}
    analyses, valleys, not_swept = [], [], []
    started = monotonic()
    for chosen in selection["chosen"][:only]:
        key = (chosen["store"], chosen["analysis_id"])
        if key not in swept:
            not_swept.append({"store": key[0], "analysis_id": key[1]})
            continue
        store = given.stores_dir / key[0] / "store.opengwasdb"

        def compute(store: Path = store, key: tuple[str, str] = key) -> dict[str, Any]:
            return analyse_profile(store, key[0], swept[key], given.genetic_map, given.panel,
                                   given.blocks, given.hm3,
                                   work_dir=work_dir / "work" / f"{key[0]}_{key[1]}",
                                   threads=threads)

        record = cached_record(work_dir / "results" / f"{key[0]}_{key[1]}.json",
                               profile_fingerprint(store, key[1], given.genetic_map,
                                                   given.panel_digests, given.files, given.code),
                               compute)
        valleys += _merged_valleys(key[0], key[1], record.pop("no_row_valleys"))
        analyses.append(record | {"selection": swept[key]["selection"]})
        print(f"{key[0]} {key[1]}: {monotonic() - started:.0f} s", file=stderr)
    return analyses, valleys, not_swept


def conclude(analyses: list[dict[str, Any]], committed: dict[str, Any],
             valleys: dict[str, Any],
             judge: Callable[..., dict[str, Any]] = judge) -> dict[str, Any]:
    """The rules' judgement, or why it is withheld: when more than half of a sample's no-row
    valleys are gaps in coverage, the profile cannot tell a valley from a gap, and the rules
    are not judged."""
    tripped = [name for name, sample in valleys.items() if sample["stop_rule_trips"]]
    if tripped:
        return {"judgement": None, "judgement_withheld": (
            f"the coverage stop rule tripped for the {', '.join(tripped)} valley sample(s): more "
            "than half of their no-row valleys are gaps in coverage, not low |z|")}
    return {"judgement": judge(analyses, committed, rule_grid())}


def profile_sweep(selection: dict[str, Any], committed: dict[str, Any], committed_path: Path,
                  stores_dir: Path, genetic_map: LoadedMap, panel_dir: Path, work_dir: Path, *,
                  hm3_path: Path, blocks_path: Path, threads: int, only: int | None = None,
                  code: dict[str, str] | None = None,
                  read_cells: Callable[[Path, list[dict[str, Any]]], list[dict[str, Any]]]
                  = read_valley_cells) -> dict[str, Any]:
    """Every Analysis #403 swept, each cached under `work_dir`, then pooled and judged."""
    check_inputs(selection, committed, genetic_map, hm3_path)
    panel = ReadOncePanel(panel_dir)
    blocks = read_blocks(blocks_path, BLOCK_ANCESTRY)
    started = monotonic()
    given = SweepInputs(stores_dir, genetic_map, panel, blocks, read_hm3(hm3_path),
                        {"hm3": hm3_path, "blocks": blocks_path, "sweep_403": committed_path},
                        panel_content_digests(panel), code or code_identity())
    print(f"panel digested in {monotonic() - started:.1f} s", file=stderr)
    analyses, valleys, not_swept = _analyse_each(selection, committed, given, work_dir,
                                                 threads, only)
    cells = read_cells(stores_dir, sample_valleys(valleys))
    summary = summarise_valley_cells(cells)
    return {
        "inputs": _inputs(committed, committed_path, genetic_map, blocks_path, blocks),
        "definitions": _definitions(),
        "grid": _grid(),
        "not_swept_by_403": not_swept,
        "valley_cells": {"sample": cells, "summary": summary,
                         "no_row_valleys": len(set(valleys))},
        **conclude(analyses, committed, summary),
        "analyses": analyses,
    }
