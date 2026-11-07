"""The per-locus picture behind the Locus Lead report's region figures (#409).

The count criteria say how many leads each method finds per Analysis. They do not show
where a method puts its leads on a |z| profile. This module records, for a few real
regions, everything a figure of that needs:
- the 5e-8 rows of the shared set, each with its plink2 r^2 0.01 clump;
- the clumps' member extents;
- where each method puts its leads;
- the EUR LD blocks;
- the 50 kb prominence profile.

Nothing here is a new rule. The rows, the clumps and the leads are rebuilt with the
prototype's own functions, as `profile_sweep` built them for #409. They are written only
when every count they imply equals #403's and #409's committed records
(`eur_sweep.json`, `eur_profile.json`), so a figure drawn from the file shows the runs
those reports summarise.

The Analyses and regions are chosen by a stated rule, not by eye (`DEFINITION`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from sys import stderr
from typing import Any

import numpy as np

from opengwasdb.top_hits import z_critical

from .blocks import Blocks, read_blocks
from .genetic_map import LoadedMap, file_digest
from .maps import selection_digest
from .panel import Panel
from .profile import Profile, Threshold
from .profile_sweep import (
    BLOCK_ANCESTRY,
    PROFILE_TIER,
    R2,
    Candidates,
    ReadOncePanel,
    Rule,
    _n_leads,
    _on,
    _per_lead,
    apply,
    check_inputs,
    check_same_shared_set,
    reference_shared_set,
    shared_set_identity,
)
from .rule import locus_leads, rule_order
from .run import SharedSet, TierRows, build_shared_set, check_tier, clump_shared_set, read_tier
from .sweep import BP_ONLY, PRIMARY_R2, TIER, panel_content_digests, read_hm3

WINDOW_BP = 10_000_000
VIEW_PAD_SHARE = 0.04
VIEW_PAD_MIN_BP = 20_000
PROFILE_BIN_BP = 50_000
SEED = 62
PROMINENCE = Rule("prominence", PROFILE_BIN_BP, Threshold("relative", 0.75))
LD_BLOCKS = Rule("ld_blocks")

DEFINITION = {
    "analyses": "the Analysis #409's judgement names, then one Analysis from each other "
                "z_band, drawn with numpy default_rng(seed) in sorted z_band order from the "
                "Analyses of that band with at least as many r^2 0.01 index variants as the "
                "named one",
    "window": f"the {WINDOW_BP:,} bp window [p, p + {WINDOW_BP:,}) starting at one of the "
              "Analysis's index variants that holds the most index variants; ties go to the "
              "first in genome order (chromosome 1-22 by number, then others by name, then "
              "position)",
    "view": f"the window's shared-set rows from first to last, padded by {VIEW_PAD_SHARE:.0%} "
            f"of that span (at least {VIEW_PAD_MIN_BP:,} bp) each side, never below 0",
    "rows": "every 5e-8 row of the shared set in the view, by position",
    "clump": "a clump with a row in the view, numbered 1, 2, ... in the Locus Lead ranking of "
             "its index variant (abs_z descending, then Store Variant Index); rows, start and "
             "end count all its members on the chromosome, in the view or not",
    "leads": "each method's leads in the view; agreed when the lead is also a plink2 index "
             "variant, spurious_split when plink2 puts it inside a stronger variant's clump; "
             "lost counts the index variants in the view that the method merged into a "
             "stronger lead",
    "blocks": "the EUR LD blocks that overlap the view, half-open [start, end)",
    "profile": f"the prominence rule's envelope: the largest |z| of the Analysis's observed "
               f"5e-4 rows in each {PROFILE_BIN_BP:,} bp bin (bin = position // bin_bp) "
               "covering the view; null where the bin holds no 5e-4 row",
}


class RegionMismatch(ValueError):
    """A count rebuilt here differs from #403's or #409's committed record."""


@dataclass(frozen=True)
class Method:
    """One way of placing leads, and the committed record that holds its counts."""

    key: str
    label: str
    lead_of: Callable[[Rebuilt, bool], np.ndarray]
    committed: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    source: str


@dataclass(frozen=True)
class Rebuilt:
    """One Analysis's shared set, its clumps and the 5e-4 profile, as #409 built them."""

    shared: SharedSet
    accounting: dict[str, Any]
    index_of: np.ndarray
    in_hm3: np.ndarray
    profile: Profile
    profile_hm3: Profile


@dataclass(frozen=True)
class Window:
    chromosome: str
    start: int
    index_variants: int


def genome_order(chromosome: str) -> tuple[int, int, str]:
    """Chromosomes 1-22 by number, then any other by name: "2" before "10"."""
    return (0, int(chromosome), "") if chromosome.isdigit() else (1, 0, chromosome)


def densest_window(chromosome: np.ndarray, position: np.ndarray,
                   is_index: np.ndarray) -> Window:
    """The window starting at an index variant that holds the most index variants; ties
    go to the first in genome order."""
    if not is_index.any():
        raise ValueError("no index variant, so no window holds one")
    best = Window("", 0, 0)
    for label in sorted({str(c) for c in chromosome[is_index]}, key=genome_order):
        starts = np.sort(position[is_index & (chromosome == label)])
        held = np.searchsorted(starts, starts + WINDOW_BP, side="left") - np.arange(starts.size)
        first = int(np.argmax(held))  # argmax takes the first of equal counts
        if held[first] > best.index_variants:
            best = Window(label, int(starts[first]), int(held[first]))
    return best


def choose(profile: dict[str, Any]) -> dict[str, Any]:
    """The Analyses the figures show, by `DEFINITION["analyses"]`, and how each was picked."""
    records = {a["analysis"]["analysis_id"]: a for a in profile["analyses"]}
    named = records[profile["judgement"]["named_analysis"]["analysis_id"]]
    floor = named["index_variants"]
    rng = np.random.default_rng(SEED)
    draws = []
    for band in sorted({a["selection"]["z_band"] for a in records.values()}
                       - {named["selection"]["z_band"]}):
        eligible = sorted(i for i, a in records.items()
                          if a["selection"]["z_band"] == band and a["index_variants"] >= floor)
        if not eligible:
            raise ValueError(f"no Analysis in z band {band} has {floor} index variants or more")
        draws.append({"z_band": band, "eligible": eligible,
                      "chosen": eligible[int(rng.integers(len(eligible)))]})
    return {"seed": SEED, "min_index_variants": floor,
            "named": named["analysis"]["analysis_id"], "draws": draws}


def first_difference(mine: Any, theirs: Any, field: str = "") -> tuple[str, Any, Any] | None:
    """The first field at which two JSON values differ, as a dotted path with both values;
    None when they are equal."""
    if isinstance(mine, dict) and isinstance(theirs, dict):
        return _dict_difference(mine, theirs, field)
    if _lists_of_one_length(mine, theirs):
        return _list_difference(mine, theirs, field)
    return None if mine == theirs else (field or "(the whole value)", mine, theirs)


def _lists_of_one_length(mine: Any, theirs: Any) -> bool:
    return isinstance(mine, list) and isinstance(theirs, list) and len(mine) == len(theirs)


def _list_difference(mine: list[Any], theirs: list[Any],
                     field: str) -> tuple[str, Any, Any] | None:
    for k, (a, b) in enumerate(zip(mine, theirs, strict=True)):
        found = first_difference(a, b, f"{field}[{k}]")
        if found is not None:
            return found
    return None


def _dict_difference(mine: dict[str, Any], theirs: dict[str, Any],
                     field: str) -> tuple[str, Any, Any] | None:
    for key in sorted(set(mine) | set(theirs)):
        path = f"{field}.{key}" if field else key
        if key not in mine or key not in theirs:
            return path, mine.get(key, "<absent>"), theirs.get(key, "<absent>")
        found = first_difference(mine[key], theirs[key], path)
        if found is not None:
            return found
    return None


def check_same(what: str, mine: Any, theirs: Any, source: str) -> None:
    """Refuse a rebuilt value that is not the committed one, naming the field that differs."""
    found = first_difference(json.loads(json.dumps(mine)), json.loads(json.dumps(theirs)))
    if found is None:
        print(f"checked: {what} = {source}", file=stderr)
    else:
        field, rebuilt, committed = found
        raise RegionMismatch(f"{what}: field {field} differs from {source} (rebuilt "
                             f"{json.dumps(rebuilt)[:200]}, committed "
                             f"{json.dumps(committed)[:200]}); nothing is written")


def rebuild(tier: TierRows, genetic_map: LoadedMap, panel: Panel, hm3: frozenset[str], *,
            work_dir: Path, threads: int) -> Rebuilt:
    """The 5e-4 tier's 5e-8 rows on the panel and map, clumped at r^2 0.01 as #409 did."""
    at_tier = tier.abs_z >= z_critical(TIER)
    shared, accounting = build_shared_set(tier.take(at_tier), panel, genetic_map.map)
    work_dir.mkdir(parents=True, exist_ok=True)
    index_of = clump_shared_set(shared, panel, [R2], threshold=TIER, work_dir=work_dir,
                                threads=threads).index_of[R2]
    profile = Profile(tier.chromosome, tier.position, tier.abs_z)
    return Rebuilt(shared, accounting, index_of, _on(hm3, shared.rows.alid), profile,
                   profile.take(_on(hm3, tier.alid)))


def methods(named_cm: dict[str, Any], blocks: Blocks) -> list[Method]:
    """The four methods the figures compare, each with where its counts were committed."""
    d_cm, w_bp = named_cm["d_cm"], named_cm["w_bp"]

    def distance(d: float, w: int) -> Callable[[Rebuilt, bool], np.ndarray]:
        def lead_of(r: Rebuilt, hm3: bool) -> np.ndarray:
            keep = r.in_hm3 if hm3 else np.ones(r.in_hm3.size, dtype=bool)
            rows = r.shared.rows.take(keep)
            return locus_leads(rows.chromosome, rows.position, r.shared.cm[keep], rows.abs_z,
                               rows.variant_index, d_cm=d, w_bp=w)
        return lead_of

    def profiled(rule: Rule) -> Callable[[Rebuilt, bool], np.ndarray]:
        def lead_of(r: Rebuilt, hm3: bool) -> np.ndarray:
            at = (Candidates(r.shared.rows.take(r.in_hm3), r.profile_hm3) if hm3
                  else Candidates(r.shared.rows, r.profile))
            return apply(rule, at, blocks)[0]
        return lead_of

    def config(d: Any, w: int) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
        return lambda sweep, _: next(c for c in sweep["configs"]
                                     if (c["d_cm"], c["w_bp"]) == (d, w))

    def rule_of(label: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
        return lambda _, profile: next(r for r in profile["rules"] if r["label"] == label)

    return [
        Method("bp_only", f"bp-only, W {WINDOW_BP // 1_000_000} Mb",
               distance(BP_ONLY, WINDOW_BP), config("bp_only", WINDOW_BP), "eur_sweep.json"),
        Method("named_cm", f"cM rule, D {d_cm:g} cM, W {w_bp / 1e6:g} Mb",
               distance(d_cm, w_bp), config(d_cm, w_bp), "eur_sweep.json"),
        Method("prominence", f"Prominence, {PROMINENCE.label()['label']}",
               profiled(PROMINENCE), rule_of(PROMINENCE.label()["label"]), "eur_profile.json"),
        Method("ld_blocks", "One lead per LD block (EUR)", profiled(LD_BLOCKS),
               rule_of(LD_BLOCKS.label()["label"]), "eur_profile.json"),
    ]


def check_rebuilt(r: Rebuilt, reference: TierRows, committed: tuple[dict[str, Any],
                  dict[str, Any]], context: tuple[str, str, LoadedMap, Panel],
                  every: list[Method]) -> dict[str, np.ndarray]:
    """The 12 checks of one Analysis against #403's (sweep) and #409's (profile) records;
    the methods' full-density leads once all pass."""
    sweep, profile = committed
    store_id, analysis_id, genetic_map, panel = context
    mine = shared_set_identity(store_id, analysis_id, r.shared.rows.variant_index)
    check_same_shared_set(mine, reference_shared_set(reference, store_id, analysis_id, sweep,
                                                     genetic_map, panel))
    check_same(f"{analysis_id} shared set", mine,
               {k: profile["shared_set"][k] for k in ("rows", "sha256")}, "eur_profile.json")
    check_same(f"{analysis_id} accounting", r.accounting, sweep["rows"], "eur_sweep.json")
    check_same(f"{analysis_id} HapMap3 rows", {"rows_full": int(r.in_hm3.size),
                                               "rows_hm3": int(r.in_hm3.sum())},
               sweep["density"], "eur_sweep.json")
    check_same(f"{analysis_id} index variants", {"index_variants": _n_leads(r.index_of)},
               {"index_variants": profile["index_variants"]}, "eur_profile.json")
    leads = {}
    for method in every:
        theirs = method.committed(sweep, profile)
        full = method.lead_of(r, False)
        per_lead = _per_lead(r.shared, full, r.index_of)
        check_same(f"{analysis_id} {method.label} per-lead record", per_lead,
                   {k: v for k, v in theirs["by_r2"][PRIMARY_R2].items() if k in per_lead},
                   method.source)
        check_same(f"{analysis_id} {method.label} density",
                   {"leads_full": _n_leads(full), "leads_hm3": _n_leads(method.lead_of(r, True))},
                   theirs["density"], method.source)
        leads[method.key] = full
    return leads


def view_of(r: Rebuilt) -> tuple[Window, int, int]:
    """The densest window, and the view around its rows (`DEFINITION["view"]`)."""
    rows = r.shared.rows
    window = densest_window(rows.chromosome, rows.position,
                            r.index_of == np.arange(r.index_of.size))
    inside = ((rows.chromosome == window.chromosome) & (rows.position >= window.start)
              & (rows.position < window.start + WINDOW_BP))
    low, high = int(rows.position[inside].min()), int(rows.position[inside].max())
    pad = max(int(VIEW_PAD_SHARE * (high - low)), VIEW_PAD_MIN_BP)
    return window, max(low - pad, 0), high + pad


def region(r: Rebuilt, leads: dict[str, np.ndarray], blocks: Blocks) -> dict[str, Any]:
    """Everything one region figure draws, for the rows `view_of` puts in view."""
    rows, index_of = r.shared.rows, r.index_of
    window, start, end = view_of(r)
    chromosome = window.chromosome
    shown = np.flatnonzero((rows.chromosome == chromosome) & (rows.position >= start)
                           & (rows.position <= end))
    shown = shown[np.lexsort((rows.variant_index[shown], rows.position[shown]))]
    heads = np.unique(index_of[shown])
    heads = heads[rule_order(rows.abs_z[heads], rows.variant_index[heads])]
    number = {int(h): k + 1 for k, h in enumerate(heads)}
    is_index = index_of == np.arange(index_of.size)
    return {
        "window": {"chromosome": chromosome, "start": window.start,
                   "end": window.start + WINDOW_BP, "index_variants": window.index_variants},
        "view": {"chromosome": chromosome, "start": start, "end": end},
        "rows": {"position": rows.position[shown].tolist(), "abs_z": rows.abs_z[shown].tolist(),
                 "clump": [number[int(index_of[k])] for k in shown],
                 "index_variant": is_index[shown].tolist()},
        "clumps": [_clump(r, int(h), number[int(h)]) for h in heads],
        "leads": {key: _leads(lead_of, index_of, shown, number, rows.position)
                  for key, lead_of in leads.items()},
        "blocks": _blocks_in(blocks, chromosome, start, end),
        "profile": _profile_in(r.profile, chromosome, start, end),
    }


def _clump(r: Rebuilt, head: int, number: int) -> dict[str, Any]:
    members = r.shared.rows.position[r.index_of == head]
    return {"clump": number, "index_position": int(r.shared.rows.position[head]),
            "index_abs_z": float(r.shared.rows.abs_z[head]), "rows": int(members.size),
            "start": int(members.min()), "end": int(members.max())}


def _leads(lead_of: np.ndarray, index_of: np.ndarray, shown: np.ndarray, number: dict[int, int],
           position: np.ndarray) -> dict[str, Any]:
    """A method's leads in view, each agreed or a spurious split, and the index variants it
    lost there (`compare.classify`'s definitions)."""
    lead = lead_of[shown] == shown
    index = index_of[shown] == shown
    return {"leads": [{"position": int(position[k]), "clump": number[int(index_of[k])],
                       "kind": "agreed" if i else "spurious_split"}
                      for k, i in zip(shown[lead].tolist(), index[lead].tolist(), strict=True)],
            "lost": int(np.sum(index & ~lead))}


def _blocks_in(blocks: Blocks, chromosome: str, start: int, end: int) -> list[list[int]]:
    starts = blocks.starts.get(chromosome, np.empty(0, dtype="int64"))
    ends = blocks.ends.get(chromosome, np.empty(0, dtype="int64"))
    return [[int(s), int(e)] for s, e in zip(starts, ends, strict=True) if e > start and s <= end]


def _profile_in(profile: Profile, chromosome: str, start: int, end: int) -> dict[str, Any]:
    first, last = start // PROFILE_BIN_BP, end // PROFILE_BIN_BP
    on = profile.chromosome == chromosome
    bins, abs_z = profile.position[on] // PROFILE_BIN_BP, profile.abs_z[on]
    keep = (bins >= first) & (bins <= last)
    envelope = np.full(last - first + 1, np.nan)
    np.fmax.at(envelope, bins[keep] - first, abs_z[keep])
    return {"bin_bp": PROFILE_BIN_BP, "first_bin": int(first),
            "max_abs_z": [None if np.isnan(v) else float(v) for v in envelope]}


@dataclass(frozen=True)
class Inputs:
    """What every region is rebuilt from, loaded once."""

    stores_dir: Path
    genetic_map: LoadedMap
    panel: Panel
    blocks: Blocks
    hm3: frozenset[str]
    work_dir: Path
    threads: int


def analyse(chosen: str, sweep: dict[str, Any], profile: dict[str, Any], given: Inputs,
            every: list[Method]) -> dict[str, Any]:
    """One chosen Analysis: read, rebuilt, checked against both records, then its region."""
    theirs = next(a for a in profile["analyses"] if a["analysis"]["analysis_id"] == chosen)
    store_id = theirs["store"]
    committed = (next(a for a in sweep["analyses"] if (a["store"], a["analysis"]["analysis_id"])
                      == (store_id, chosen)), theirs)
    store = given.stores_dir / store_id / "store.opengwasdb"
    tier, identity = read_tier(store, chosen, PROFILE_TIER)
    reference, _ = read_tier(store, chosen, TIER)
    check_tier(tier, identity, given.genetic_map, given.panel)
    rebuilt = rebuild(tier, given.genetic_map, given.panel, given.hm3,
                      work_dir=given.work_dir / f"{store_id}_{chosen}", threads=given.threads)
    leads = check_rebuilt(rebuilt, reference, committed,
                          (store_id, chosen, given.genetic_map, given.panel), every)
    return {"analysis": {"analysis_id": chosen,
                         "analysis_label": identity["analysis"]["analysis_label"],
                         "store": store_id,
                         "store_manifest_sha256": file_digest(store / "manifest.json", "sha256"),
                         "z_band": theirs["selection"]["z_band"],
                         "max_abs_z": theirs["selection"]["max_abs_z"],
                         "index_variants": theirs["index_variants"]},
            **region(rebuilt, leads, given.blocks)}


def check_records(sweep_path: Path, profile: dict[str, Any], blocks_path: Path) -> None:
    """The two input checks the selection, map and HapMap3 check does not cover."""
    check_same("the #403 sweep", {"sha256": file_digest(sweep_path, "sha256")},
               {"sha256": profile["inputs"]["sweep_403"]["sha256"]}, "eur_profile.json's inputs")
    check_same("the LD-block file", {"sha256": file_digest(blocks_path, "sha256")},
               {"sha256": profile["inputs"]["blocks"]["sha256"]}, "eur_profile.json's inputs")


def _record_inputs(paths: dict[str, Path], selection: dict[str, Any],
                   genetic_map: LoadedMap, panel: Panel) -> dict[str, Any]:
    return {**{name: {"path": str(path), "sha256": file_digest(path, "sha256")}
               for name, path in paths.items()},
            "selection_sha256": selection_digest(selection),
            "genetic_map": {k: genetic_map.identity[k] for k in ("map", "sha256", "assembly")},
            "panel": panel_content_digests(panel),
            "blocks_ancestry": BLOCK_ANCESTRY}


def regions(paths: dict[str, Path], selection: dict[str, Any], given: Inputs,
            output: Path) -> dict[str, Any]:
    """Every chosen Analysis's region, written to `output` only when all checks pass.

    `paths` names #403's sweep JSON (`sweep`), #409's profile JSON (`profile`), the
    HapMap3 list (`hm3`) and the LD-block file (`blocks`).
    """
    sweep = json.loads(paths["sweep"].read_text())
    profile = json.loads(paths["profile"].read_text())
    check_records(paths["sweep"], profile, paths["blocks"])
    check_inputs(selection, sweep, given.genetic_map, paths["hm3"])
    print("checked: the selection, map and HapMap3 list = #403's sweep's", file=stderr)
    every = methods(sweep["counts"]["named_cm"], given.blocks)
    picked = choose(profile)
    chosen = [picked["named"], *(d["chosen"] for d in picked["draws"])]
    found = [analyse(a, sweep, profile, given, every) for a in chosen]
    record = {"definitions": DEFINITION, "selection": picked,
              "tiers": {"rows": TIER, "profile": PROFILE_TIER, "clump_r2": R2},
              "methods": [{"key": m.key, "label": m.label, "source": m.source} for m in every],
              "inputs": _record_inputs(paths, selection, given.genetic_map, given.panel),
              "regions": found}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    return record


def load_inputs(stores_dir: Path, genetic_map: LoadedMap, panel_dir: Path, paths: dict[str, Path],
                work_dir: Path, threads: int) -> Inputs:
    """The panel, blocks and HapMap3 list, read once for every region."""
    return Inputs(stores_dir, genetic_map, ReadOncePanel(panel_dir),
                  read_blocks(paths["blocks"], BLOCK_ANCESTRY), read_hm3(paths["hm3"]),
                  work_dir, threads)

