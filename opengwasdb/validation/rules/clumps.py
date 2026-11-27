"""Rules for the Clumps of the 5e-8 tier (ADR 0106, spec §18.1, §20.0; issue #418).

Each silent failure of the Clumps has its rule. A group the manifest does not declare, as an
interrupted step leaves, and columns without the arrays they count (`clumps.declared`); arrays
of another length than their tier (`clumps.structure`); an imputed row clumped, an ambiguous
indel looked up, a marked row given an r (`clumps.markers`); a lead weaker than its member or
out of its reach (`clumps.greedy`); an Analysis without an LD Store counted as 0
(`clumps.ancestry`); counts that disagree with the arrays (`clumps.counts`); and, given the LD
Stores, anything a re-clump of a seeded sample disagrees with (`clumps.reclump`).

What the rules read is the step's own reading (`opengwasdb.clumps`: `read_clumps`, `Tiers`,
`compute_clumps`), read once per release and remembered on its root target.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from opengwasdb.clumps import (
    ABSENT,
    AMBIGUOUS,
    CLUMPS_FORMAT_VERSIONS,
    CONFIGURATION_ATTRIBUTE,
    IMPUTED,
    MANIFEST_ENTRY,
    MANIFEST_KEY,
    NO_ASSIGNED_ANCESTRY,
    NO_LD_STORE,
    NO_R,
    NOT_CLUMPED,
    TIER,
    WINDOW_BP,
    ClumpRefusal,
    ClumpResult,
    MappedLDStore,
    Tiers,
    ancestries,
    close_ld_stores,
    compute_clumps,
    open_ld_stores,
    read_clumps,
)
from opengwasdb.model.analyses import CLUMP_COUNT_COLUMNS, N_LOCI_5E8, N_UNCLUMPED_5E8
from opengwasdb.store_reader.enums import PrimaryStorageLayout
from opengwasdb.store_reader.manifest import StoreManifest
from opengwasdb.store_reader.top_hits import (
    CLUMPS_ATTRIBUTE,
    CLUMPS_CONFIGURATION,
    CLUMPS_ENTRY_KEYS,
    CLUMPS_GROUP,
    TOP_HIT_GROUP,
    MalformedClumps,
    TierClumps,
    TopHitTier,
    clumps_array_problems,
    clumps_attribute_problem,
    clumps_configuration_problem,
    declares_clumps_of,
)
from opengwasdb.validation.rule import Check, Kind, Skip, Target
from opengwasdb.validation.rules.release import REQUIRED

DECLARED = "clumps.declared"
STRUCTURE = "clumps.structure"

#: How many clumped Analyses `clumps.reclump` re-clumps, at most; the seed is the run's
#: `fidelity_seed`.
RECLUMP_ANALYSES = 25




# ── what the rules read ──────────────────────────────────────────────────────


def _component_label(target: Target) -> str:
    """The component a grid or sequence target is, as `read_clumps` names it."""
    return "data.zarr" if target.kind is Kind.CSR else target.label


def _tiers(target: Target) -> dict[str, TopHitTier]:
    """Each Top-Hit tier of the target's release, by key, as the Store Reader opens it, with
    the release's declaration of its Clumps (`StoreRelease.top_hit_tiers`)."""
    tiers = target.store.top_hit_tiers()
    opened = {key: tiers.by_key(key) for key in tiers.keys()}
    return {key: tier for key, tier in opened.items() if tier is not None}


def _has_group(target: Target) -> bool:
    """Whether this grid's or sequence's 5e-8 tier holds a `clumps/` group."""
    tier = _tiers(target).get(TIER)
    return tier is not None and tier.has_clumps()


def _declares(manifest: StoreManifest) -> bool:
    """Whether a release's manifest declares the Clumps of its 5e-8 tier (spec §2)."""
    return declares_clumps_of(manifest.clumps, TIER)


@dataclass(frozen=True)
class _View:
    """A release's tiers and their Clumps as the Store Reader reads them, one per component,
    its ancestries and its `ld_stores` entries."""

    tiers: Tiers
    clumps: list[TierClumps]
    ancestry: list[str]
    entries: dict[str, Any]

    def component(self, label: str) -> int:
        return [component.label for component in self.tiers.components].index(label)

    def labelled(self) -> Iterator[tuple[str, TierClumps]]:
        for component, clumps in zip(self.tiers.components, self.clumps, strict=True):
            yield component.label, clumps

    def piece(self, i: int, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Component `i`'s `lead` and `r` for Analysis `index`."""
        _, a, b = self.tiers.bounds(index)[i]
        return self.clumps[i].lead[a:b], self.clumps[i].r[a:b]

    def lead_of(self, index: int) -> np.ndarray:
        """Analysis `index`'s `lead` over its clump slice."""
        return np.concatenate([self.piece(i, index)[0] for i in range(len(self.clumps))])

    def r_of(self, index: int) -> np.ndarray:
        return np.concatenate([self.piece(i, index)[1] for i in range(len(self.clumps))])


def _readable(target: Target) -> bool:
    """Whether the Store Reader reads the release's Clumps in every component holding a 5e-8
    tier (`TopHitTier.clumps`). Where not, `clumps.declared` or `clumps.structure` reports
    why, and the rules that judge the values have no values to judge."""
    return _view_or_none(target) is not None


def _view_or_none(target: Target) -> _View | None:
    """The release's view, read once and kept on its root target."""
    root = target.outer_release()
    return root.remember("clumps/view", lambda: _read_view(root))


def _view(target: Target) -> _View:
    view = _view_or_none(target)
    if view is None:
        raise RuntimeError("a clumps rule ran on a release whose Clumps it cannot read")
    return view


def _read_view(root: Target) -> _View | None:
    try:
        stored = read_clumps(root.store)
    except (ClumpRefusal, MalformedClumps):
        return None
    clumps = [c.clumps for c in stored if c.clumps is not None]
    if not stored or len(clumps) != len(stored):
        return None
    return _View(
        tiers=Tiers.read(root.store, [c.tier for c in stored]),
        clumps=clumps,
        ancestry=ancestries(root.store),
        entries=dict(clumps[0].attribute.get("ld_stores", {})),
    )


def _analysis_ids(target: Target) -> list[str]:
    rows = sorted(target.outer_release().analyses.rows, key=lambda row: int(row["analysis_index"]))
    return [row["analysis_id"] for row in rows]


def _clumped(view: _View) -> Iterator[tuple[int, str]]:
    """Each Analysis whose Assigned Ancestry has an `ld_stores` entry, with that ancestry."""
    for index, assigned in enumerate(view.ancestry):
        if assigned in view.entries:
            yield index, assigned


# ── clumps.declared ──────────────────────────────────────────────────────────


def declared(target: Target) -> Iterator[str]:
    store = target.store
    entry = store.manifest.clumps
    where = "manifest.json" if target.label == "." else f"{target.label}/manifest.json"
    tiers = _tiers(target)
    yield from _other_tiers(tiers)
    tier = tiers.get(TIER)
    group = tier.group if tier is not None else None
    names = _configurations(group)
    yield from _declaration(entry, group, names, where)
    if (entry is not None or names is not None) and \
            store.manifest.format_version not in CLUMPS_FORMAT_VERSIONS:
        yield (f"the Clumps are on a release of format_version {store.manifest.format_version}; "
               f"they are a component of {', '.join(CLUMPS_FORMAT_VERSIONS)} alone")
    if not target.is_component:
        yield from _hybrid_declared(target, _declares(store.manifest))
        yield from _columns_declared(target, _declares(store.manifest))


def _other_tiers(tiers: Mapping[str, TopHitTier]) -> Iterator[str]:
    for key, tier in tiers.items():
        if key != TIER and tier.has_clumps():
            yield f"{TOP_HIT_GROUP}/{key} holds {CLUMPS_GROUP}/: only the 5e-8 tier is clumped"


def _configurations(tier: Any) -> list[str] | None:
    """The members of the 5e-8 tier's `clumps/` group; None when it has none."""
    if tier is None or CLUMPS_GROUP not in tier:
        return None
    group = tier[CLUMPS_GROUP]
    if not isinstance(group, zarr.Group):
        return []  # an array where the group belongs: `_declaration` names it
    return sorted(name for name, _ in group.members())


_GROUP = f"{TOP_HIT_GROUP}/{TIER}/{CLUMPS_GROUP}"


def _declaration(
    entry: Any, tier: Any, names: list[str] | None, where: str
) -> Iterator[str]:
    if names is not None:
        problem = clumps_configuration_problem(tier[CLUMPS_GROUP])
        if problem is not None:
            yield f"{TOP_HIT_GROUP}/{TIER}/{problem}"
    yield from _entry_declares(entry, names is not None and CLUMPS_CONFIGURATION in names, where)


def _entry_declares(entry: Any, held: bool, where: str) -> Iterator[str]:
    """The manifest entry is the one §2 names, and is there exactly when the group is."""
    declares = declares_clumps_of(entry, TIER)
    if entry is not None and not declares:
        yield f"{where}'s {MANIFEST_KEY} entry is {entry!r}, not {MANIFEST_ENTRY!r}"
    if held and not declares:
        yield (f"{_GROUP}/{CLUMPS_CONFIGURATION} is not declared by {where}: a step that never "
               "finished, or Clumps left behind")
    if declares and not held:
        yield f"{where} declares {MANIFEST_KEY}, and its 5e-8 tier holds no Clumps"


def _hybrid_declared(target: Target, declares: bool) -> Iterator[str]:
    """A Hybrid declares the Clumps in both manifests or in neither (§2)."""
    if target.store.manifest.primary_layout is not PrimaryStorageLayout.HYBRID:
        return
    if _declares(target.store.dense_component().manifest) != declares:
        yield (f"manifest.json {'declares' if declares else 'does not declare'} {MANIFEST_KEY} "
               "and dense/manifest.json does not agree: a Hybrid declares the Clumps in both")


def _columns_declared(target: Target, declares: bool) -> Iterator[str]:
    columns = [name for name in CLUMP_COUNT_COLUMNS if name in target.analyses.fieldnames]
    if declares and len(columns) != len(CLUMP_COUNT_COLUMNS):
        yield f"analyses.tsv lacks {sorted(set(CLUMP_COUNT_COLUMNS) - set(columns))}"
    if not declares and columns:
        yield f"analyses.tsv has {columns}, and the release declares no Clumps"


# ── clumps.structure ─────────────────────────────────────────────────────────


def structure(target: Target) -> Iterator[str]:
    """The configuration laid out as the Store Reader reads it (`clumps_array_problems`,
    `clumps_attribute_problem`), and the attribute's values those checks leave to
    validation: `tier_rows`, the fixed configuration and each `ld_stores` entry."""
    tier = _tiers(target)[TIER].group
    default = tier[CLUMPS_GROUP][CLUMPS_CONFIGURATION]
    n_rows = int(tier["z"].shape[0])
    yield from clumps_array_problems(default, n_rows)
    problem = clumps_attribute_problem(default)
    if problem is not None:
        yield problem
        return
    attribute = dict(default.attrs[CLUMPS_ATTRIBUTE])
    yield from _attribute_problems(attribute, n_rows)
    for ancestry, entry in sorted(dict(attribute.get("ld_stores", {})).items()):
        yield from _entry_problems(ancestry, entry, target.store.manifest.reference_assembly)


def _attribute_problems(attribute: dict[str, Any], n_rows: int) -> Iterator[str]:
    if attribute.get("tier_rows") != n_rows:
        yield (f"the {CLUMPS_ATTRIBUTE} attribute's tier_rows is {attribute.get('tier_rows')}, "
               f"the tier {n_rows}")
    for key, value in CONFIGURATION_ATTRIBUTE.items():
        if attribute[key] != value:
            yield f"the {CLUMPS_ATTRIBUTE} attribute's {key} is {attribute[key]!r}, not {value!r}"


def _entry_problems(ancestry: str, entry: dict[str, Any], assembly: str) -> Iterator[str]:
    lacking = sorted(CLUMPS_ENTRY_KEYS - set(entry))
    if lacking:
        yield f"ld_stores[{ancestry!r}] lacks {lacking}"
    if entry.get("r2") != entry.get("t_far"):
        yield (f"ld_stores[{ancestry!r}]'s r2 {entry.get('r2')} is not its t_far "
               f"{entry.get('t_far')}")
    if entry.get("reference_assembly") != assembly:
        yield (f"ld_stores[{ancestry!r}] is reference assembly "
               f"{entry.get('reference_assembly')}, the release {assembly}")


# ── clumps.markers ───────────────────────────────────────────────────────────


def markers(target: Target) -> Iterator[str]:
    view = _view(target)
    i = view.component(_component_label(target))
    tiers = view.tiers
    imputed = tiers.components[i].imputed
    ambiguous = tiers.identity.ambiguous[tiers.identity_rows[i]] & ~imputed
    ids = _analysis_ids(target)
    for index, assigned in _clumped(view):
        _, a, b = tiers.bounds(index)[i]
        lead, r = view.piece(i, index)
        problem = _marker_problem(lead, r, imputed[a:b], ambiguous[a:b],
                                  int(view.entries[assigned]["r_scale"]))
        if problem:
            yield f"{ids[index]}: {problem}"


def _marker_problem(
    lead: np.ndarray, r: np.ndarray, imputed: np.ndarray, ambiguous: np.ndarray, scale: int
) -> str | None:
    if not np.array_equal(lead == IMPUTED, imputed):
        return "−2 is not exactly on the tier's imputed rows"
    if not np.array_equal(lead == AMBIGUOUS, ambiguous):
        return ("−3 is not exactly on the observed rows whose indel shares its sorted spelling "
                "with another variant")
    if not np.array_equal(r == NO_R, lead < 0):
        return "r is not −32768 exactly where lead is negative"
    if (np.abs(r[lead >= 0].astype(np.int64)) > scale).any():
        return f"an r beyond ±{scale}, its LD Store's r_scale"
    return None


# ── clumps.greedy ────────────────────────────────────────────────────────────


def greedy(target: Target) -> Iterator[str]:
    view = _view(target)
    ids = _analysis_ids(target)
    for index, assigned in _clumped(view):
        lead = view.lead_of(index).astype(np.int64)
        rows = np.flatnonzero(lead >= 0)
        if not len(rows):
            continue
        problem = _lead_problem(view, index, lead, rows) or \
            _reach_problem(view, index, lead, rows, view.entries[assigned])
        if problem:
            yield f"{ids[index]}: {problem}"


def _lead_problem(view: _View, index: int, lead: np.ndarray, rows: np.ndarray) -> str | None:
    """A lead that is not a lead: past the slice, led by another, imputed, or weaker."""
    leads = lead[rows]
    if (leads >= len(lead)).any():
        return f"a lead offset past its clump slice of {len(lead)} rows"
    if (lead[leads] != leads).any():
        return "a row's lead is not its own lead"
    part = view.tiers.slice(index)
    if part.imputed[leads].any():
        return "an imputed row leads a clump"
    z, shared = part.abs_z, part.shared
    weaker = (z[leads] < z[rows]) | ((z[leads] == z[rows]) & (shared[leads] > shared[rows]))
    if weaker.any():
        return "a lead is weaker than its member by |z|, then Store Variant Index"
    return None


def _reach_problem(
    view: _View, index: int, lead: np.ndarray, rows: np.ndarray, entry: dict[str, Any]
) -> str | None:
    """A member out of its lead's reach, or an r a lead or member cannot have."""
    leads = lead[rows]
    ident = view.tiers.slice(index).identity
    chrom = view.tiers.identity.chromosome[ident]
    position = view.tiers.identity.position[ident]
    if (chrom[leads] != chrom[rows]).any() or \
            (np.abs(position[leads] - position[rows]) >= WINDOW_BP).any():
        return f"a member off its lead's chromosome or {WINDOW_BP:,} bp or more from it"
    r = view.r_of(index).astype(np.int64)
    scale = int(entry["r_scale"])
    own = leads == rows
    if (r[rows[own]] != scale).any():
        return f"a lead's r is not +{scale}"
    if ((r[rows[~own]] / scale) ** 2 <= float(entry["r2"])).any():
        return f"a member's r² is not above its entry's r2 {entry['r2']}"
    return None


# ── clumps.ancestry ──────────────────────────────────────────────────────────


def ancestry(target: Target) -> Iterator[str]:
    view = _view(target)
    yield from _recorded_ancestries(view)
    rows = {int(r["analysis_index"]): r for r in target.analyses.rows}
    for index, assigned in enumerate(view.ancestry):
        blank = [rows[index].get(c, "") == "" for c in CLUMP_COUNT_COLUMNS]
        marked = view.lead_of(index) == NOT_CLUMPED
        problem = (_clumped_problem(bool(marked.any()), any(blank)) if assigned in view.entries
                   else _unclumped_problem(bool(marked.all()), all(blank)))
        if problem:
            yield f"{rows[index]['analysis_id']}: assigned_ancestry {assigned!r} {problem}"


def _recorded_ancestries(view: _View) -> Iterator[str]:
    """Every component records the same `ld_stores` keys, and the Analyses not clumped."""
    expected = _not_clumped(view)
    for label, clumps in view.labelled():
        attribute = clumps.attribute
        recorded = set(dict(attribute.get("ld_stores", {})))
        if recorded != set(view.entries):
            yield f"{label} records ld_stores {sorted(recorded)}, another {sorted(view.entries)}"
        if attribute.get("not_clumped") != expected:
            yield (f"{label} records not_clumped {attribute.get('not_clumped')}, "
                   f"the Analyses {expected}")


def _clumped_problem(any_marked: bool, any_blank: bool) -> str | None:
    if any_marked:
        return "has an LD Store, and rows marked −4"
    if any_blank:
        return f"has an LD Store, and a blank {N_LOCI_5E8} or {N_UNCLUMPED_5E8}"
    return None


def _unclumped_problem(all_marked: bool, all_blank: bool) -> str | None:
    if not all_marked:
        return "has no LD Store, and rows not marked −4"
    if not all_blank:
        return f"has no LD Store, and records {N_LOCI_5E8} or {N_UNCLUMPED_5E8}: blank, never 0"
    return None


def _not_clumped(view: _View) -> dict[str, Any]:
    """The Analyses not clumped, by reason, as the attribute must record them (§18.1)."""
    unmapped = [a for a in view.ancestry if a not in view.entries]
    by_ancestry: dict[str, int] = {}
    for assigned in filter(None, unmapped):
        by_ancestry[assigned] = by_ancestry.get(assigned, 0) + 1
    return {NO_ASSIGNED_ANCESTRY: unmapped.count(""), NO_LD_STORE: by_ancestry}


# ── clumps.counts ────────────────────────────────────────────────────────────


def counts(target: Target) -> Iterator[str]:
    view = _view(target)
    rows = {int(r["analysis_index"]): r for r in target.analyses.rows}
    for index, _assigned in _clumped(view):
        problem = _count_problem(view.lead_of(index), rows[index])
        if problem:
            yield f"{rows[index]['analysis_id']}: {problem}"
    for i, (label, clumps) in enumerate(view.labelled()):
        yield from _tally_problems(label, clumps, _tallies(view, i))


def _tally_problems(
    label: str, clumps: TierClumps, tallies: Mapping[str, Mapping[str, int]]
) -> Iterator[str]:
    """Each `ld_stores` entry's row counts are its rows' markers' counts."""
    recorded = dict(clumps.attribute.get("ld_stores", {}))
    for assigned, tally in tallies.items():
        for key, value in tally.items():
            found = recorded.get(assigned, {}).get(key)
            if found != value:
                yield (f"{label}: ld_stores[{assigned!r}].{key} is {found}, "
                       f"its rows {value}")


def _count_problem(lead: np.ndarray, row: Mapping[str, str]) -> str | None:
    loci = int((lead == np.arange(len(lead))).sum())
    unclumped = int(np.isin(lead, (ABSENT, AMBIGUOUS)).sum())
    if row.get(N_LOCI_5E8) != str(loci) or row.get(N_UNCLUMPED_5E8) != str(unclumped):
        return (f"{N_LOCI_5E8} {row.get(N_LOCI_5E8)!r} and {N_UNCLUMPED_5E8} "
                f"{row.get(N_UNCLUMPED_5E8)!r}; its rows say {loci} and {unclumped}")
    hits = row.get("n_hits_5e8", "")
    if hits.isdigit() and loci + unclumped > int(hits):
        return f"more loci and unclumped rows than n_hits_5e8 {hits}"
    return None


def _tallies(view: _View, i: int) -> dict[str, dict[str, int]]:
    """Component `i`'s rows marked −1 and −3, by the ancestry of their Analysis."""
    out = {a: {"rows_absent": 0, "rows_ambiguous": 0} for a in view.entries}
    for index, assigned in _clumped(view):
        lead, _ = view.piece(i, index)
        out[assigned]["rows_absent"] += int((lead == ABSENT).sum())
        out[assigned]["rows_ambiguous"] += int((lead == AMBIGUOUS).sum())
    return out


# ── clumps.reclump ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Reclump:
    problems: list[str]
    result: ClumpResult | None
    sample: list[int]
    seed: int


def reclump(target: Target) -> Iterator[str]:
    if not target.options.ld_stores:
        raise Skip("not run: validate was given no LD Store (--ld-store ANCESTRY=PATH)")
    view = _view(target)
    i = view.component(_component_label(target))
    found = target.outer_release().remember("clumps/reclump", lambda: _reclumped(target, view))
    yield from found.problems
    if found.result is None:
        return
    ids = _analysis_ids(target)
    again = found.result.clumps[i]
    for index in found.sample:
        _, a, b = view.tiers.bounds(index)[i]
        lead, r = view.piece(i, index)
        if not (np.array_equal(lead, again.lead[a:b]) and np.array_equal(r, again.r[a:b])):
            yield (f"{ids[index]}: lead and r differ from a re-clump (seed {found.seed}, "
                   f"sample {[ids[k] for k in found.sample]})")


def _reclumped(target: Target, view: _View) -> _Reclump:
    """A seeded sample of the clumped Analyses whose ancestry was given, re-clumped."""
    root = target.outer_release()
    seed = target.options.fidelity_seed
    given = dict(target.options.ld_stores or {})
    problems = [f"--ld-store {a} has no ld_stores entry to check"
                for a in given if a not in view.entries]
    opened = _open_given(given, view, root.store.manifest.reference_assembly, problems)
    if opened is None:
        return _Reclump(problems, None, [], seed)
    try:
        problems += _provenance_problems(opened, view.entries)
        sample = _sample([k for k, a in enumerate(view.ancestry) if a in opened], seed)
        result = compute_clumps(root.store, opened, tiers=view.tiers, analyses=sample)
    finally:
        close_ld_stores(opened)
    return _Reclump(problems, result, sample, seed)


def _open_given(
    given: Mapping[str, str | Path], view: _View, assembly: str, problems: list[str]
) -> dict[str, MappedLDStore] | None:
    """The given LD Stores that have an entry, opened; None, with the refusal added to
    `problems`, when one is refused."""
    try:
        return open_ld_stores({a: p for a, p in given.items() if a in view.entries},
                              reference_assembly=assembly)
    except ClumpRefusal as exc:
        problems.append(str(exc))
        return None


def _provenance_problems(
    opened: Mapping[str, MappedLDStore], entries: Mapping[str, Any]
) -> list[str]:
    """Each given LD Store whose recomputed digest is not the one its entry records."""
    problems = []
    for assigned, mapped in opened.items():
        recorded = entries[assigned].get("provenance_sha256")
        digest = mapped.identity["provenance_sha256"]
        if digest != recorded:
            problems.append(f"--ld-store {assigned} {mapped.location} is not the LD Store "
                            f"recorded: provenance_sha256 {digest}, recorded {recorded}")
    return problems


def _sample(candidates: list[int], seed: int) -> list[int]:
    size = min(RECLUMP_ANALYSES, len(candidates))
    if not size:
        return []
    chosen = np.random.default_rng(seed).choice(candidates, size, replace=False)
    return sorted(int(k) for k in chosen)


# ── the registry's entries ───────────────────────────────────────────────────


def _release_declares(target: Target) -> bool:
    return not target.is_component and _declares(target.store.manifest) \
        and _readable(target)


def _component_readable(target: Target) -> bool:
    return _has_group(target) and _readable(target)


RULES = (
    Check(name=DECLARED, kinds=Kind.RELEASE, after=(REQUIRED,), run=declared),
    Check(name=STRUCTURE, kinds=Kind.GRID | Kind.CSR, when=_has_group,
          after=(REQUIRED, DECLARED), run=structure),
    Check(name="clumps.markers", kinds=Kind.GRID | Kind.CSR, when=_component_readable,
          after=(STRUCTURE,), run=markers),
    Check(name="clumps.greedy", kinds=Kind.RELEASE, when=_release_declares,
          after=(DECLARED,), run=greedy),
    Check(name="clumps.ancestry", kinds=Kind.RELEASE, when=_release_declares,
          after=(DECLARED,), run=ancestry),
    Check(name="clumps.counts", kinds=Kind.RELEASE, when=_release_declares,
          after=(DECLARED,), run=counts),
    Check(name="clumps.reclump", kinds=Kind.GRID | Kind.CSR, when=_component_readable,
          after=(STRUCTURE,), run=reclump),
)
