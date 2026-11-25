"""The Clumps of a Store Release's 5e-8 tier, and `n_loci_5e8` (ADR 0106, spec §18.1, #418).

One step writes them: `clump_store`, which the CLI's `opengwasdb clump` calls. It clumps each
Analysis's observed rows of the 5e-8 Top-Hit tier against the LD Store its Assigned Ancestry
maps to, through `opengwasdb.ld` (`LDStore.clump`, plink2's greedy), and writes, in place:

1. per component holding the tier, `top_hits/p_5e_08/clumps/default/` with `lead` and `r`, one
   entry per tier row, and the `clumps` attribute (§18.1);
2. `n_loci_5e8` and `n_unclumped_5e8` in the release's `analyses.tsv`, and `overview.html`;
3. the manifest's `clumps` entry, last, in each manifest of a Hybrid (§2).

The mapping from Assigned Ancestry to LD Store is an input: the engine has no default LD Store
(ADR 0106 §2). Each mapped store answers at r2 = its own t_far and kb = 10,000; one that refuses
that, or whose reference assembly is not the release's, fails the step before anything is
written, as does a release of any format but 0.3.0, one with no 5e-8 tier, and a
Reference-Completed one whose tier has no `imputed` array (§18.1, *Refusals*).

`compute_clumps` is the compute path without the writes and without the format refusal, so the
same code measures a published 0.2 Store (#418's parity run). `read_clumps` reads what was
written, through the Store Reader (`TopHitTier.clumps`), for the validation rules
(`opengwasdb.validation.rules.clumps`) and the tests; #419's query builds on the same read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from opengwasdb.ld.format import InputError, Refusal, REncoding
from opengwasdb.ld.query.kind import open_kind
from opengwasdb.ld.query.ldstore import LDStore
from opengwasdb.ld.query.ldstore import open as open_ld_store
from opengwasdb.model.analyses import N_LOCI_5E8, N_UNCLUMPED_5E8, write_analysis_columns
from opengwasdb.overview import write_overview_html
from opengwasdb.store.arrays import (
    create_array,
    create_group,
    open_group_for_write,
    refuse_under_consolidated_metadata,
)
from opengwasdb.store.open import OpenGWASDBStore, open_store, staged_named_group
from opengwasdb.store_reader.analyses import read_analyses
from opengwasdb.store_reader.enums import CompletionState, PrimaryStorageLayout
from opengwasdb.store_reader.format import UnsupportedFormatVersion
from opengwasdb.store_reader.hybrid import dense_to_shared_path
from opengwasdb.store_reader.layout import ArrayRole
from opengwasdb.store_reader.release import StoreRelease
from opengwasdb.store_reader.top_hits import (
    CLUMPS_ARRAYS,
    CLUMPS_ATTRIBUTE,
    CLUMPS_CONFIGURATION,
    CLUMPS_GROUP,
    TOP_HIT_GROUP,
    TierClumps,
    TierRows,
    TopHitTier,
    analysis_hit_counts,
    threshold_key,
)
from opengwasdb.top_hits import TOP_HIT_CHUNK_SIZE, recomputed_abs_z_and_p
from opengwasdb.variants.sorted_spelling import SortedLookup, sorted_lookup
from opengwasdb.version import installed_version

log = logging.getLogger(__name__)

#: The tier clumped, and its group's name (§18.1).
THRESHOLD: Final = 5e-8
TIER: Final = threshold_key(THRESHOLD)
#: The one stored Clumping Configuration's window: kb, and W in bp (ADR 0106 §2).
KB: Final = 10_000
WINDOW_BP: Final = KB * 1_000
#: The formats a release carrying Clumps may be (ADR 0106 §9, §10).
CLUMPS_FORMAT_VERSIONS: Final = ("0.3.0",)

#: The markers a negative `lead` holds (§18.1), decided in this order of precedence.
NOT_CLUMPED: Final = -4
IMPUTED: Final = -2
AMBIGUOUS: Final = -3
ABSENT: Final = -1
MARKERS: Final = {"absent": ABSENT, "imputed": IMPUTED, "ambiguous": AMBIGUOUS,
                  "not_clumped": NOT_CLUMPED}
#: The `r` of a row whose `lead` is negative: the LD Store's reserved code (ADR 0090).
NO_R: Final = REncoding.reserved_code

#: The `clumps` attribute's values that are the one stored Clumping Configuration (§18.1), the
#: same on every release: what the step writes and `clumps.structure` holds it to.
CONFIGURATION_ATTRIBUTE: Final = {
    "configuration": CLUMPS_CONFIGURATION,
    "threshold": THRESHOLD,
    "score": "abs(z)",
    "ties": "store_variant_index",
    "kb": KB,
    "join": "r2_gt_distance_lt",
    "markers": MARKERS,
}

#: The manifest's `clumps` entry (§2).
MANIFEST_KEY: Final = "clumps"
MANIFEST_ENTRY: Final = {"tier": TIER, "configuration": CLUMPS_CONFIGURATION}

#: An Analysis's rows are not clumped for one of these reasons (§18.1, `not_clumped`).
NO_ASSIGNED_ANCESTRY: Final = "no_assigned_ancestry"
NO_LD_STORE: Final = "no_ld_store"


class ClumpRefusal(ValueError):
    """The step refuses a Store or an LD Store, naming it and why; nothing is written."""


# ── the Store's side: its tiers ──────────────────────────────────────────────


@dataclass(frozen=True)
class TierComponent:
    """One component's 5e-8 tier, as the step reads it.

    `label` names the component as validation does (`data.zarr`, or `dense/data.zarr` for a
    Hybrid's Dense Component). `offsets` addresses each Analysis's rows; `shared` is each row's
    shared Store Variant Index, a Dense Component's panel row mapped through
    `dense_to_shared.npy` (ADR 0106 §5). `overflow` is a Hybrid's Ragged Overflow, which is
    observed by construction and carries no `imputed`. `imputed` is each row's imputed flag:
    False throughout an Observed-Only tier and an Overflow's, and refused missing elsewhere
    (`tier_components`).
    """

    label: str
    release: StoreRelease
    tier: TopHitTier
    rows: TierRows
    offsets: np.ndarray
    shared: np.ndarray
    overflow: bool
    imputed: np.ndarray

    @property
    def tier_path(self) -> Path:
        return self.release.data_path / TOP_HIT_GROUP / TIER

    @property
    def n_rows(self) -> int:
        return int(len(self.rows.z))


def component_releases(release: StoreRelease) -> list[tuple[str, StoreRelease]]:
    """The releases a Store Release's Clumps live in, in clump-slice order, with the label
    validation gives each component: a Hybrid's Dense Component, then the Hybrid's root, which
    holds the Ragged Overflow; every other layout's one release."""
    if release.manifest.primary_layout is PrimaryStorageLayout.HYBRID:
        return [("dense/data.zarr", release.dense_component()), ("data.zarr", release)]
    return [("data.zarr", release)]


def tier_components(release: StoreRelease, n_analyses: int) -> list[TierComponent]:
    """The release's 5e-8 tiers, in clump-slice order. Refused, naming the component (§18.1,
    *Refusals*): a component without the tier, and a Reference-Completed one whose tier has no
    `imputed` array, which is what says which of its rows are observed. Every path that clumps
    or judges Clumps reads the tiers here, so none reads such a tier as all observed."""
    completed = release.manifest.completion_state is CompletionState.REFERENCE_COMPLETED
    to_shared = None
    if release.manifest.primary_layout is PrimaryStorageLayout.HYBRID:
        to_shared = np.load(dense_to_shared_path(release.path)).astype(np.int64)
    components = []
    for label, each in component_releases(release):
        overflow = to_shared is not None and each is release
        component = _tier_component(each, label, n_analyses,
                                    None if each is release else to_shared, overflow=overflow)
        if completed and not overflow and component.rows.imputed is None:
            raise ClumpRefusal(
                f"{component.tier_path} is the 5e-8 tier of a Reference-Completed release and "
                "carries no `imputed` array, so its observed rows cannot be told from its "
                "imputed ones (§18.1); rebuild its Top-Hit Index"
            )
        components.append(component)
    return components


def _tier_component(
    release: StoreRelease, label: str, n_analyses: int, to_shared: np.ndarray | None,
    *, overflow: bool,
) -> TierComponent:
    root = release.arrays()
    tier = release.top_hit_tiers(root).tier(THRESHOLD)
    if tier is None:
        raise ClumpRefusal(
            f"{release.path} has no 5e-8 Top-Hit tier ({TOP_HIT_GROUP}/{TIER}) to clump "
            "(spec §18.1); build its Top-Hit Index first"
        )
    rows = tier.read(None)
    counts = np.asarray(analysis_hit_counts(root, THRESHOLD, n_analyses), dtype=np.int64)
    if len(counts) != n_analyses:
        raise ClumpRefusal(
            f"{release.path}: its 5e-8 tier addresses {len(counts)} Analyses, and "
            f"analyses.tsv holds {n_analyses}"
        )
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    vi = np.asarray(rows.variant_index, dtype=np.int64)
    shared = vi if to_shared is None else to_shared[vi]
    imputed = (np.zeros(len(vi), dtype=bool) if rows.imputed is None
               else np.asarray(rows.imputed, dtype=bool))
    return TierComponent(label, release, tier, rows, offsets, shared, overflow, imputed)


def refuse_format(release: StoreRelease) -> None:
    """Refuse a release of any format but 0.3.0, its Hybrid Dense Component's included,
    naming its `format_version` (§18.1, *Refusals*; ADR 0106 §10)."""
    for _label, each in component_releases(release):
        version = each.manifest.format_version
        if version not in CLUMPS_FORMAT_VERSIONS:
            raise ClumpRefusal(
                f"{each.path} is format_version {version}; the Clumps are a component of "
                f"format {', '.join(CLUMPS_FORMAT_VERSIONS)} alone (ADR 0106 §10). Rebuild "
                "the Store at 0.3.0, then clump it"
            )


# ── the LD Stores ────────────────────────────────────────────────────────────


@dataclass
class MappedLDStore:
    """One Assigned Ancestry's LD Store, opened, and what the step records of it (§18.1)."""

    ancestry: str
    location: str
    store: LDStore
    encoding: REncoding
    identity: dict[str, Any]

    @property
    def r2(self) -> float:
        return float(self.identity["r2"])


def provenance_sha256(record: Mapping[str, Any]) -> str:
    """The SHA-256 of an LD Store's root provenance record, canonical JSON (ADR 0106 §4)."""
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def open_ld_stores(
    locations: Mapping[str, str | Path], *, reference_assembly: str
) -> dict[str, MappedLDStore]:
    """Open each mapped LD Store and judge it before anything is clumped: its reference
    assembly must be the release's, it must record every value §18.1's entry copies, and it
    must answer r2 = its t_far at kb 10,000 (ADR 0106 §2). A refusal names the store and the
    ancestry that maps to it."""
    opened: dict[str, MappedLDStore] = {}
    try:
        for ancestry, location in locations.items():
            opened[ancestry] = _open_mapped(ancestry, str(location), reference_assembly)
    except BaseException:
        close_ld_stores(opened)
        raise
    return opened


def _open_mapped(ancestry: str, location: str, reference_assembly: str) -> MappedLDStore:
    where = f"LD Store {location} (for {ancestry})"
    if not ancestry.strip():
        raise ClumpRefusal(f"LD Store {location} is mapped to a blank ancestry; an ancestry is an "
                           "assigned_ancestry value, and a blank one is Unassigned")
    try:
        store = open_ld_store(location)
    except (InputError, Refusal, OSError, KeyError) as exc:
        raise ClumpRefusal(f"{where} does not open: {exc}") from exc
    try:
        encoding, identity = _judged(store, location, ancestry, reference_assembly)
        mapped = MappedLDStore(ancestry, location, store, encoding, identity)
    except BaseException:
        store.close()
        raise
    return mapped


def _judged(
    store: LDStore, location: str, ancestry: str, reference_assembly: str
) -> tuple[REncoding, dict[str, Any]]:
    """The store's encoding and its `ld_stores` entry's identity and settings (§18.1), once
    its assembly and the configuration are judged. A value it does not record is refused by
    name, never read as a default; `samples_sha256` may be null, but must be recorded."""
    where = f"LD Store {location} (for {ancestry})"
    record: dict[str, Any] = dict(store.provenance)
    assembly = _recorded(record, where, "reference_assembly")
    if assembly != reference_assembly:
        raise ClumpRefusal(
            f"{where} is reference assembly {assembly}, and the Store is {reference_assembly}; "
            "a Store is clumped only against an LD Store of its own assembly (ADR 0106 §2)"
        )
    for key in ("t_far", "near_bp", "far_bp", "encoding"):
        _recorded(record, where, key)  # recorded, and not null, or refused by name
    try:
        t_far, near_bp, far_bp = float(store.t_far), int(store.near_bp), int(store.far_bp)
        encoding = REncoding.of(record["encoding"])
    except (TypeError, ValueError) as exc:  # InputError is a ValueError
        raise ClumpRefusal(
            f"{where} records a t_far, near_bp, far_bp or encoding this package cannot read: "
            f"{exc} (ld-store §3, §7)"
        ) from exc
    try:
        # The request itself, with no hits: `clump` judges the configuration first.
        store.clump([], np.empty(0), r2=t_far, kb=KB)
    except Refusal as refusal:
        raise ClumpRefusal(
            f"{where} refuses the Clumping Configuration, r2 = its t_far {t_far} at kb {KB}: "
            f"{refusal}"
        ) from refusal
    identity = {
        "name": str(_recorded(record, where, "panel", "id")),
        "kind": open_kind(location).kind,
        "format_version": str(_recorded(record, where, "format_version")),
        "samples_sha256": _recorded(record, where, "panel", "samples_sha256", nullable=True),
        "provenance_sha256": provenance_sha256(record),
        "location": location,
        "reference_assembly": assembly,
        "n": int(_recorded(record, where, "panel", "n")),
        "k": int(_recorded(record, where, "adjustment", "k")),
        "t_far": t_far,
        "t_partner": float(_recorded(record, where, "t_partner")),
        "near_bp": near_bp,
        "far_bp": far_bp,
        "r_scale": int(encoding.scale),
        "r2": t_far,
    }
    return encoding, identity


def _recorded(record: Mapping[str, Any], where: str, *path: str, nullable: bool = False) -> Any:
    """The value at `path` in an LD Store's provenance record; refused, naming the store and
    the key, when it is not recorded, or recorded null where null is not allowed."""
    value: Any = record
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ClumpRefusal(f"{where} records no {'.'.join(path)} (ld-store §7)")
        value = value[key]
    if value is None and not nullable:
        raise ClumpRefusal(f"{where} records {'.'.join(path)} as null (ld-store §7)")
    return value


def close_ld_stores(stores: Mapping[str, MappedLDStore]) -> None:
    for mapped in stores.values():
        mapped.store.close()


# ── the compute path ─────────────────────────────────────────────────────────

#: An `ld_stores` entry's per-component counts (§18.1), in the order they are recorded.
_ROW_COUNTS: Final = ("analyses", "rows_clumped", "rows_absent", "rows_ambiguous",
                      "indel_rows_by_sorted_spelling")


@dataclass
class ComponentClumps:
    """One component's Clumps: `lead` and `r`, one entry per tier row, and its rows' counts by
    the ancestry whose LD Store clumped them."""

    lead: np.ndarray
    r: np.ndarray
    counts: dict[str, dict[str, int]]

    @classmethod
    def unclumped(cls, n_rows: int, ancestries: Sequence[str]) -> ComponentClumps:
        """Every row −4, until an Analysis is clumped."""
        return cls(
            np.full(n_rows, NOT_CLUMPED, dtype=np.int32),
            np.full(n_rows, NO_R, dtype=np.int16),
            {a: dict.fromkeys(_ROW_COUNTS, 0) for a in ancestries},
        )


@dataclass
class ClumpResult:
    """What `compute_clumps` found: each component's Clumps, each Analysis's two counts (None
    for one that was not clumped), and the Analyses not clumped, by reason."""

    components: list[TierComponent]
    clumps: list[ComponentClumps]
    n_loci: list[int | None]
    n_unclumped: list[int | None]
    not_clumped: dict[str, Any]

    def markers(self) -> dict[str, int]:
        """How many rows hold each marker, every component together."""
        lead = np.concatenate([c.lead for c in self.clumps]) if self.clumps else np.empty(0)
        return {name: int((lead == value).sum()) for name, value in MARKERS.items()}


@dataclass(frozen=True)
class Identity:
    """Each distinct shared Store Variant Index of the tiers, looked up once: its chromosome
    and position, its sorted spelling, whether that swaps A1, whether it is an indel, and
    whether its sorted spelling is also another variant's of the shared Variant Axis."""

    chromosome: np.ndarray
    position: np.ndarray
    sorted_alid: np.ndarray
    swapped: np.ndarray
    indel: np.ndarray
    ambiguous: np.ndarray


def identities(release: StoreRelease, shared: np.ndarray) -> Identity:
    """`Identity` of each of `shared`'s variants, through the release's shared Variant Axis
    (ADR 0106 §5, §6)."""
    axis = release.variant_axis(aliases=False)
    try:
        found = axis.identity_by_indices(np.asarray(shared, dtype=np.int64))
        lookups = [sorted_lookup(str(alid)) for alid in found["alid"]]
        ambiguous = _shares_its_spelling(axis.alid_index, lookups)
    finally:
        axis.close()
    return Identity(
        chromosome=np.asarray(found["chromosome"], dtype=object),
        position=np.asarray(found["position"], dtype=np.int64),
        sorted_alid=np.array([lookup.alid for lookup in lookups], dtype=object),
        swapped=np.array([lookup.swapped for lookup in lookups], dtype=bool),
        indel=np.array([lookup.reversed_indel is not None for lookup in lookups], dtype=bool),
        ambiguous=ambiguous,
    )


def _shares_its_spelling(alid_index: Any, lookups: Sequence[SortedLookup]) -> np.ndarray:
    """Whether each variant is an indel whose reversed indel the axis also holds."""
    asked = [(k, lookup.reversed_indel) for k, lookup in enumerate(lookups)
             if lookup.reversed_indel is not None]
    ambiguous = np.zeros(len(lookups), dtype=bool)
    if asked:
        rows = np.asarray(alid_index.search([alid for _, alid in asked]))
        ambiguous[[k for k, _ in asked]] = rows >= 0
    return ambiguous


@dataclass
class Slice:
    """One Analysis's clump slice: its rows of every component, Dense Component first.

    `parts` is each component's (index, start, stop); `identity` each row's index into the
    `Identity` arrays."""

    parts: list[tuple[int, int, int]]
    shared: np.ndarray
    abs_z: np.ndarray
    imputed: np.ndarray
    identity: np.ndarray

    def __len__(self) -> int:
        return int(len(self.shared))


@dataclass(frozen=True)
class Tiers:
    """A release's 5e-8 tiers, in clump-slice order, with what clumping reads of their rows:
    each row's |z| from its `z`, and the identity of every variant they hold. `identity_rows`
    is, per component, each tier row's index into the `Identity` arrays."""

    components: list[TierComponent]
    identity: Identity
    abs_z: list[np.ndarray]
    identity_rows: list[np.ndarray]

    @classmethod
    def read(cls, release: StoreRelease, components: Sequence[TierComponent]) -> Tiers:
        comps = list(components)
        shared = np.concatenate([c.shared for c in comps]) if comps else np.empty(0, np.int64)
        distinct, back = np.unique(shared, return_inverse=True)
        starts = np.cumsum([0, *(c.n_rows for c in comps)])
        return cls(
            components=comps,
            identity=identities(release, distinct),
            abs_z=[recomputed_abs_z_and_p(c.rows.z)[0] for c in comps],
            identity_rows=[back[starts[i]:starts[i + 1]] for i in range(len(comps))],
        )

    def bounds(self, index: int) -> list[tuple[int, int, int]]:
        """Analysis `index`'s rows in each component: (component, start, stop)."""
        return [(i, int(c.offsets[index]), int(c.offsets[index + 1]))
                for i, c in enumerate(self.components)]

    def slice(self, index: int) -> Slice:
        parts = self.bounds(index)
        comps = self.components
        return Slice(
            parts=parts,
            shared=np.concatenate([comps[i].shared[a:b] for i, a, b in parts]),
            abs_z=np.concatenate([self.abs_z[i][a:b] for i, a, b in parts]),
            imputed=np.concatenate([comps[i].imputed[a:b] for i, a, b in parts]),
            identity=np.concatenate([self.identity_rows[i][a:b] for i, a, b in parts]),
        )


def ancestries(release: StoreRelease) -> list[str]:
    """Each Analysis's `assigned_ancestry`, by `analysis_index`, from the release's own
    `analyses.tsv` (a Hybrid's root's). A blank value is an Unassigned Analysis; a table
    without the column is refused rather than read as every Analysis Unassigned."""
    table = read_analyses(release.analyses_path)
    if "assigned_ancestry" not in table.fieldnames:
        raise ClumpRefusal(
            f"{release.analyses_path} has no assigned_ancestry column, so no Analysis can be "
            "mapped to an LD Store; a blank value is Unassigned, a missing column is not "
            "(spec §7a, §18.1)"
        )
    by_index = {int(row["analysis_index"]): row["assigned_ancestry"] for row in table.rows}
    if sorted(by_index) != list(range(len(by_index))):
        raise ClumpRefusal(f"{release.analyses_path}: analysis_index is not 0..n-1")
    return [by_index[i].strip() for i in range(len(by_index))]


def compute_clumps(
    release: StoreRelease,
    ld_stores: Mapping[str, MappedLDStore],
    *,
    tiers: Tiers | None = None,
    analyses: Sequence[int] | None = None,
) -> ClumpResult:
    """Clump each Analysis's 5e-8 rows as §18.1 says, writing nothing.

    `tiers` is the release's tiers already read (`Tiers.read`); they are read here if not.
    `analyses` restricts the work to those Analyses (validation's seeded re-clump); every
    other Analysis's rows are left −4 and its counts None, and are not counted. The format is
    not judged here (`refuse_format`), so a 0.2 Store's tier can be measured too."""
    assigned = ancestries(release)
    if tiers is None:
        tiers = Tiers.read(release, tier_components(release, len(assigned)))
    result = ClumpResult(
        tiers.components,
        [ComponentClumps.unclumped(c.n_rows, list(ld_stores)) for c in tiers.components],
        [None] * len(assigned),
        [None] * len(assigned),
        {NO_ASSIGNED_ANCESTRY: 0, NO_LD_STORE: {}},
    )
    wanted = range(len(assigned)) if analyses is None else sorted(set(analyses))
    for done, index in enumerate(wanted):
        _clump_analysis(result, tiers, ld_stores, assigned[index], index)
        if (done + 1) % 500 == 0:
            log.info("clumped %d of %d Analyses", done + 1, len(wanted))
    return result


def _clump_analysis(
    result: ClumpResult, tiers: Tiers, ld_stores: Mapping[str, MappedLDStore], ancestry: str,
    index: int,
) -> None:
    """Clump Analysis `index` into `result`, or count it among those not clumped."""
    mapped = ld_stores.get(ancestry) if ancestry else None
    if mapped is None:
        _count_not_clumped(result.not_clumped, ancestry)
        return
    part = tiers.slice(index)
    lead, r = _clump_slice(part, tiers.identity, mapped)
    result.n_loci[index] = int((lead == np.arange(len(part))).sum())
    result.n_unclumped[index] = int(np.isin(lead, (ABSENT, AMBIGUOUS)).sum())
    _scatter(part, lead, r, tiers.identity, result.clumps, ancestry)


def _count_not_clumped(not_clumped: dict[str, Any], ancestry: str) -> None:
    if not ancestry:
        not_clumped[NO_ASSIGNED_ANCESTRY] += 1
    else:
        by_ancestry = not_clumped[NO_LD_STORE]
        by_ancestry[ancestry] = by_ancestry.get(ancestry, 0) + 1


def clump_order(abs_z: np.ndarray, shared: np.ndarray) -> np.ndarray:
    """The score `clump` is given for rows of |z| `abs_z` at shared Store Variant Index
    `shared`: n − rank, rank 0 the largest |z|, ties by Store Variant Index ascending (§18.1)."""
    order = np.lexsort((shared, -np.asarray(abs_z, dtype=np.float64)))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return (len(order) - rank).astype(np.float64)


def _clump_slice(
    part: Slice, identity: Identity, mapped: MappedLDStore
) -> tuple[np.ndarray, np.ndarray]:
    """One clumped Analysis's `lead` and `r`, by offset in its clump slice."""
    n = len(part)
    lead = np.full(n, ABSENT, dtype=np.int32)
    r = np.full(n, NO_R, dtype=np.int16)
    lead[part.imputed] = IMPUTED
    ambiguous = ~part.imputed & identity.ambiguous[part.identity]
    lead[ambiguous] = AMBIGUOUS
    given = np.flatnonzero(~part.imputed & ~ambiguous)
    if len(given) == 0:
        return lead, r
    names = identity.sorted_alid[part.identity[given]]
    if len(set(names)) != len(names):
        raise ValueError(
            "two rows of one clump slice share a sorted spelling and neither is ambiguous: "
            f"{sorted({n for n in names if list(names).count(n) > 1})[:5]}"
        )
    score = clump_order(part.abs_z[given], part.shared[given])
    found = mapped.store.clump(list(names), score, r2=mapped.r2, kb=KB)
    absent = set(found.absent)
    present = given[np.array([name not in absent for name in names], dtype=bool)]
    offset_of = {str(name): int(offset) for name, offset in zip(names, given, strict=True)}
    leads = np.array([offset_of[str(name)] for name in found.lead], dtype=np.int64)
    swapped = identity.swapped[part.identity]
    sign = np.where(swapped[present] != swapped[leads], -1, 1).astype(np.int64)
    codes = mapped.encoding.quantise(found.r).astype(np.int64) * sign
    lead[present] = leads
    r[present] = codes.astype(np.int16)
    return lead, r


def _scatter(
    part: Slice, lead: np.ndarray, r: np.ndarray, identity: Identity,
    clumps: list[ComponentClumps], ancestry: str,
) -> None:
    """Write one Analysis's slice back into each component's arrays, and count its rows."""
    at = 0
    indel = identity.indel[part.identity]
    for i, start, stop in part.parts:
        width = stop - start
        piece = slice(at, at + width)
        clumps[i].lead[start:stop] = lead[piece]
        clumps[i].r[start:stop] = r[piece]
        counts = clumps[i].counts[ancestry]
        counts["analyses"] += 1
        given = lead[piece] >= ABSENT
        counts["rows_clumped"] += int(given.sum())
        counts["rows_absent"] += int((lead[piece] == ABSENT).sum())
        counts["rows_ambiguous"] += int((lead[piece] == AMBIGUOUS).sum())
        counts["indel_rows_by_sorted_spelling"] += int((given & indel[piece]).sum())
        at += width


# ── writing ──────────────────────────────────────────────────────────────────


def group_attribute(
    component: TierComponent, clumps: ComponentClumps, result: ClumpResult,
    ld_stores: Mapping[str, MappedLDStore], created_at: str,
) -> dict[str, Any]:
    """The `clumps` attribute of one component's group (§18.1)."""
    return {
        **CONFIGURATION_ATTRIBUTE,
        "markers": dict(MARKERS),
        "tier_rows": component.n_rows,
        "ld_stores": {
            ancestry: {**mapped.identity, **clumps.counts[ancestry]}
            for ancestry, mapped in sorted(ld_stores.items())
        },
        "not_clumped": result.not_clumped,
        "opengwasdb": installed_version(),
        "created_at": created_at,
    }


def _write_group(
    component: TierComponent, clumps: ComponentClumps, attribute: dict[str, Any]
) -> None:
    """One component's `clumps/` group, staged beside the tier and renamed into it whole."""
    refuse_under_consolidated_metadata(component.tier_path, "writing the Clumps into")
    with staged_named_group(component.tier_path / CLUMPS_GROUP, overwrite=True) as work:
        root = open_group_for_write(work, "w", zarr_format=3)
        group = create_group(root, CLUMPS_CONFIGURATION)
        values = {"lead": clumps.lead, "r": clumps.r}
        for name, dtype in CLUMPS_ARRAYS.items():
            create_array(group, name, ArrayRole.TOP_HIT_INDEX, data=values[name], dtype=dtype,
                         hint=TOP_HIT_CHUNK_SIZE)
        group.attrs[CLUMPS_ATTRIBUTE] = attribute


def _manifest_holders(release: OpenGWASDBStore) -> list[OpenGWASDBStore]:
    """The releases whose manifests declare the Clumps (§2), in clump-slice order: a Hybrid's
    Dense Component, then its root, last; every other layout's one."""
    holders = []
    for _label, each in component_releases(release):
        if not isinstance(each, OpenGWASDBStore):
            raise TypeError(f"{each.path} was opened read-only; the Clumps are written to a "
                            "release opened with opengwasdb.store.open_store")
        holders.append(each)
    return holders


def _count_column(counts: Sequence[int | None]) -> list[str]:
    """A count per Analysis as `analyses.tsv` spells it: blank, never 0, where None."""
    return ["" if n is None else str(n) for n in counts]


def write_clumps(
    release: OpenGWASDBStore, result: ClumpResult, ld_stores: Mapping[str, MappedLDStore]
) -> None:
    """Write `result` into `release` in place, in §18.1's order. Every manifest stops
    declaring any earlier Clumps first, so a step interrupted anywhere leaves a group no
    manifest declares, which `validate` fails (`clumps.declared`), never a declaration of
    stale columns or half a group."""
    holders = _manifest_holders(release)
    for holder in holders:
        holder.set_manifest_entry(MANIFEST_KEY, None)
    created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    for component, clumps in zip(result.components, result.clumps, strict=True):
        _write_group(component, clumps,
                     group_attribute(component, clumps, result, ld_stores, created_at))
    write_analysis_columns(release.analyses_path, {
        N_LOCI_5E8: _count_column(result.n_loci),
        N_UNCLUMPED_5E8: _count_column(result.n_unclumped),
    })
    if (release.path / "overview.html").exists():
        write_overview_html(release.path, read_analyses(release.analyses_path))
    for holder in holders:
        holder.set_manifest_entry(MANIFEST_KEY, dict(MANIFEST_ENTRY))


@dataclass(frozen=True)
class ClumpRun:
    """What one run of the step did, for the CLI and a build record (§18.1, *Written last*).

    `n_loci_5e8` and `n_unclumped_5e8` are the totals over the clumped Analyses, and None
    when no Analysis was clumped: a 0 would claim the Store has no locus. `not_clumped`
    gives the Analyses not clumped by reason, as the group's attribute does."""

    store: str
    seconds: float
    analyses: int
    clumped: int
    n_loci_5e8: int | None
    n_unclumped_5e8: int | None
    markers: dict[str, int]
    not_clumped: dict[str, Any]
    ld_stores: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "store": self.store, "seconds": round(self.seconds, 3), "analyses": self.analyses,
            "clumped": self.clumped, "n_loci_5e8": self.n_loci_5e8,
            "n_unclumped_5e8": self.n_unclumped_5e8, "markers": self.markers,
            "not_clumped": self.not_clumped, "ld_stores": self.ld_stores,
        }


def clump_store(store_path: str | Path, ld_stores: Mapping[str, str | Path]) -> ClumpRun:
    """Clump the 5e-8 tier of the 0.3.0 Store Release at `store_path` in place, each Analysis
    against the LD Store its Assigned Ancestry maps to in `ld_stores` (ADR 0106, §18.1).

    Every refusal comes before anything is written. Running the step again replaces the
    Clumps, the two columns, `overview.html` and the manifest entry whole."""
    started = time.perf_counter()
    try:
        release = open_store(store_path)
    except UnsupportedFormatVersion as exc:
        raise ClumpRefusal(str(exc)) from exc
    refuse_format(release)
    n_analyses = len(ancestries(release))
    components = tier_components(release, n_analyses)
    opened = open_ld_stores(ld_stores, reference_assembly=release.manifest.reference_assembly)
    try:
        result = compute_clumps(release, opened, tiers=Tiers.read(release, components))
        write_clumps(release, result, opened)
        return ClumpRun(
            store=str(release.path),
            seconds=time.perf_counter() - started,
            analyses=n_analyses,
            clumped=sum(n is not None for n in result.n_loci),
            n_loci_5e8=_total(result.n_loci),
            n_unclumped_5e8=_total(result.n_unclumped),
            markers=result.markers(),
            not_clumped=result.not_clumped,
            ld_stores={a: m.identity for a, m in opened.items()},
        )
    finally:
        close_ld_stores(opened)


def _total(counts: Sequence[int | None]) -> int | None:
    """The sum over the clumped Analyses; None when none was clumped, never a false 0."""
    clumped = [n for n in counts if n is not None]
    return int(sum(clumped)) if clumped else None


# ── reading ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StoredComponent:
    """One component's tier and the Clumps stored beside it, read by the Store Reader
    (`TopHitTier.clumps`): None where the tier holds no `clumps/` group."""

    tier: TierComponent
    clumps: TierClumps | None


def read_clumps(release: StoreRelease) -> list[StoredComponent]:
    """Each component's tier and stored Clumps, in clump-slice order. A group the Store
    Reader cannot read as §18.1 lays it out is refused (`MalformedClumps`)."""
    n_analyses = len(ancestries(release))
    return [StoredComponent(c, c.tier.clumps()) for c in tier_components(release, n_analyses)]

