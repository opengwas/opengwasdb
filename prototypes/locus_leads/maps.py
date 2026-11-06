"""The comparison genetic maps for the EUR sweep: HapMap II lifted, and the pyrho median (#404).

#403 measured Locus Leads against the equal-weight mean of the five pyrho EUR
population maps. #404 asks whether another map does better, which needs two more
maps in the one layout the sweep reads — `chromosome position cM` on GRCh38,
autosomes only, cM non-decreasing within each chromosome:

- **HapMap II combined, lifted to GRCh38**, as Eagle ships it. An LD-based map
  lifted from hg19 (#62's revision, "The global map"): a lift can put a point's
  cM below the point before it, and the map may not lose more than a tenth of a
  percent of a chromosome's points putting it back in order.
- **The median of the same five pyrho maps**, in place of their mean. #402 found
  two single-population spikes that survive the mean, and one population's jump
  cannot move the middle of five.

The maps are also checked against each other: every map must leave exactly the
same rows to judge, so the shared sets are compared by count and by a digest of
(Store, Analysis, Store Variant Index).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .eur_map import AUTOSOMES, EUR_POPULATIONS, read_population_maps
from .genetic_map import (
    ARCHIVE_RECORD,
    INTERPOLATION,
    GeneticMap,
    GeneticMapError,
    file_digest,
    sidecar_path,
    verify_archive,
    write_points,
)
from .pairs import Spike
from .panel import Panel, PanelMatch, PanelVariants, match_to_panel
from .run import TierRows, read_tier
from .sweep import EUR_SPIKES, TIER

if TYPE_CHECKING:
    from .genetic_map import LoadedMap

# A lifted map's cM may fall where the lift moved a point; a tenth of a
# chromosome's points may go restoring the order, and past that the lift is
# broken rather than noisy.
MAX_DROPPED_SHARE = 0.001

MEDIAN_RULE = (
    "the median of the five population maps' cM at the union of their positions, each "
    "interpolated linearly between its points and held flat beyond its ends"
)

EAGLE_HAPMAP2_URL = ("https://alkesgroup.broadinstitute.org/Eagle/downloads/tables/"
                     "genetic_map_hg38_withX.txt.gz")
EAGLE_HAPMAP2_RETRIEVED = "2026-11-06T00:20:00Z"
EAGLE_HAPMAP2_LICENCE = (
    "HapMap II combined map, reformatted by the Eagle authors and shipped with Eagle; Eagle's "
    "manual (section 4.1.2) says the reference maps are 'downloaded from HapMap and "
    "reformatted', and Eagle is GPLv3 (section 7). HapMap's own data are public research data"
)

# Chromosome labels that are not autosomes, and what they are. Anything else is
# refused: a map spelled `chr1` would otherwise read as a map of nothing.
NON_AUTOSOME = {"23": "chrX", "X": "chrX", "24": "chrY", "Y": "chrY",
                "25": "chrMT", "MT": "chrMT", "M": "chrMT"}


class MapBuildError(ValueError):
    """A downloaded map cannot be turned into the map the sweep reads."""


class SharedSetMismatch(ValueError):
    """Two maps would judge different rows, so their counts are not comparable."""


@dataclass(frozen=True)
class Layout:
    """How a downloaded map spells the three columns the prototype needs."""

    name: str
    header: tuple[str, ...]
    chromosome: str
    position: str
    cm: str


EAGLE_HAPMAP2 = Layout(
    name="eagle-hapmap2",
    header=("chr", "position", "COMBINED_rate(cM/Mb)", "Genetic_Map(cM)"),
    chromosome="chr", position="position", cm="Genetic_Map(cM)",
)


@dataclass(frozen=True)
class ParsedPoints:
    """An autosome's points, and how many points each other chromosome held."""

    points: dict[str, tuple[np.ndarray, np.ndarray]]
    excluded: dict[str, int]


@dataclass(frozen=True)
class Dropped:
    """The points a map lost to a cM decrease, per chromosome."""

    positions: dict[str, list[int]]
    total_points: dict[str, int]

    def counts(self) -> dict[str, int]:
        return {chromosome: len(found) for chromosome, found in self.positions.items()}

    def record(self) -> dict[str, Any]:
        return {"share_limit": MAX_DROPPED_SHARE, "points_dropped": self.counts(),
                "positions": self.positions, "points_before": self.total_points}


def parse_points(text: str, layout: Layout) -> ParsedPoints:
    """Read a downloaded map as points per autosome, refusing any other layout."""
    frame = _read_frame(text, layout)
    labels = _labels(frame, layout)
    position = frame[layout.position].to_numpy(dtype="int64")
    cm = frame[layout.cm].to_numpy(dtype="float64")
    points = {}
    for chromosome in AUTOSOMES:
        on = np.array([label == chromosome for label in labels])
        if on.any():
            points[chromosome] = (position[on], cm[on])
    excluded = {label: labels.count(label) for label in NON_AUTOSOME if label in labels}
    return ParsedPoints(points, excluded)


def _read_frame(text: str, layout: Layout) -> pd.DataFrame:
    """The file, only if its header is this layout's, column for column.

    Matched whole rather than by column position: Beagle and Eagle ship other
    spellings of the same map, and a file read by column number comes out
    plausible and wrong.
    """
    frame = pd.read_csv(StringIO(text), sep=r"\s+")
    if tuple(frame.columns) != layout.header:
        raise MapBuildError(f"{layout.name}: the header is {tuple(frame.columns)}, not "
                            f"{layout.header}")
    return frame


def _labels(frame: pd.DataFrame, layout: Layout) -> list[str]:
    """Each row's chromosome label, refusing one that is neither autosome nor known."""
    labels = [str(label) for label in frame[layout.chromosome]]
    unknown = sorted(set(labels) - set(AUTOSOMES) - set(NON_AUTOSOME))
    if unknown:
        raise MapBuildError(f"{layout.name}: chromosome labels {unknown} are neither an autosome "
                            f"nor one of {sorted(NON_AUTOSOME)}")
    return labels


def autosome_map(parsed: ParsedPoints, *, name: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The points on all 22 autosomes, or a refusal naming those that are missing."""
    missing = [chromosome for chromosome in AUTOSOMES if chromosome not in parsed.points]
    if missing:
        raise MapBuildError(f"{name}: the map is missing autosomes {', '.join(missing)}; a "
                            "chromosome absent from the map would silently get NaN cM")
    return {chromosome: parsed.points[chromosome] for chromosome in AUTOSOMES}


def drop_decreasing(position: np.ndarray,
                    cm: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """The points at or above the running maximum of cM, and the positions dropped.

    Dropping every point that falls below the running maximum is the smallest set
    that restores order and it moves nothing: a dropped point costs its own
    interval, not a shift of the points after it. `build_genetic_map` decides
    whether the map may afford it.
    """
    keep = cm >= np.maximum.accumulate(cm)
    dropped = np.flatnonzero(~keep)
    return (position[keep], cm[keep]), position[dropped]


def build_genetic_map(points: Mapping[str, tuple[np.ndarray, np.ndarray]]
                      ) -> tuple[GeneticMap, Dropped]:
    """The map, once every chromosome's cM is non-decreasing; and what that cost.

    The points a cM decrease costs are dropped here and nowhere else: a lift that
    costs more than `MAX_DROPPED_SHARE` of a chromosome is broken, and the build
    stops rather than hand back a map whose distances are invented.
    """
    built, dropped, before = {}, {}, {}
    for chromosome, (position, cm) in points.items():
        _check_positions(chromosome, position)
        (kept_position, kept_cm), lost = drop_decreasing(position, cm)
        if lost.size > MAX_DROPPED_SHARE * position.size:
            raise MapBuildError(
                f"chromosome {chromosome}: {lost.size:,} of {position.size:,} points "
                f"({lost.size / position.size:.2%}) would go to restore a non-decreasing cM, past "
                f"the {MAX_DROPPED_SHARE:.1%} a lift may cost")
        built[chromosome] = (kept_position, kept_cm)
        dropped[chromosome] = [int(value) for value in lost]
        before[chromosome] = int(position.size)
    return GeneticMap(built), Dropped(dropped, before)


def _check_positions(chromosome: str, position: np.ndarray) -> None:
    repeats = np.flatnonzero(np.diff(position) <= 0)
    if repeats.size:
        raise MapBuildError(
            f"chromosome {chromosome}: {repeats.size:,} positions repeat or fall, the first at "
            f"{int(position[repeats[0] + 1]):,}; which point is which is reported, not guessed")


def median_map(maps: Mapping[str, GeneticMap]) -> GeneticMap:
    """The median of `maps`, at the union of their positions, per chromosome.

    The maps must cover the same chromosomes: a chromosome some lack would be a
    median of fewer populations without saying so.
    """
    covered = {name: sorted(genetic_map.chromosomes) for name, genetic_map in maps.items()}
    chromosomes = next(iter(maps.values())).chromosomes
    if any(labels != sorted(chromosomes) for labels in covered.values()):
        raise GeneticMapError(f"population maps cover different chromosomes: {covered}")
    median = {}
    for chromosome in chromosomes:
        each = [genetic_map.points(chromosome) for genetic_map in maps.values()]
        union = np.unique(np.concatenate([position for position, _ in each]))
        cm = np.median([np.interp(union, position, cm) for position, cm in each], axis=0)
        median[chromosome] = (union, cm)
    return GeneticMap(median)


def write_source_map(points: Mapping[str, tuple[np.ndarray, np.ndarray]], map_path: Path, *,
                     provenance: Mapping[str, Any],
                     excluded: Mapping[str, int]) -> dict[str, Any]:
    """Write `<name>.tsv.gz` and its identity, and return the identity.

    `provenance` names what the map came from — an `archive` (checked against its
    md5 on every load) or a `source` download (checked against its sha256) — plus
    how the map was made from it. What the non-decreasing check cost is part of
    the identity, so a dropped point can never be found only in a report prose.
    """
    genetic_map, dropped = build_genetic_map(points)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "map": map_path.name,
        "sha256": write_points(genetic_map, map_path),
        "assembly": "GRCh38",
        **provenance,
        "interpolation": INTERPOLATION,
        "points_per_chromosome": {chromosome: int(genetic_map.points(chromosome)[0].size)
                                  for chromosome in genetic_map.chromosomes},
        "dropped_to_a_non_decreasing_cM": dropped.record(),
        "excluded_chromosomes": dict(excluded),
    }
    sidecar_path(map_path).write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def build_hapmap2_map(original: Path, map_path: Path) -> dict[str, Any]:
    """HapMap II lifted to GRCh38, as Eagle ships it, normalised and checked."""
    text = gzip.decompress(original.read_bytes()).decode() if original.suffix == ".gz" \
        else original.read_text()
    parsed = parse_points(text, EAGLE_HAPMAP2)
    return write_source_map(
        autosome_map(parsed, name=str(original)), map_path,
        provenance={"source": {
            "path": str(original.relative_to(map_path.parent)),
            "sha256": file_digest(original, "sha256"), "url": EAGLE_HAPMAP2_URL,
            "retrieved": EAGLE_HAPMAP2_RETRIEVED, "licence": EAGLE_HAPMAP2_LICENCE,
            "layout": EAGLE_HAPMAP2.name}},
        excluded=parsed.excluded)


def build_median_map(archive: Path, map_path: Path) -> dict[str, Any]:
    """The median of the five pyrho EUR population maps, written beside the mean."""
    verify_archive(archive)
    populations = read_population_maps(archive, EUR_POPULATIONS)
    median = median_map(populations)
    return write_source_map(
        {chromosome: median.points(chromosome) for chromosome in median.chromosomes}, map_path,
        provenance={
            "archive": {"path": str(archive.relative_to(map_path.parent)),
                        "md5": file_digest(archive, "md5"), "record": ARCHIVE_RECORD},
            "populations": list(populations), "combination": MEDIAN_RULE},
        excluded={})


@dataclass(frozen=True)
class MatchedTier:
    """One Analysis's tier rows, and each row's panel match, which every map shares."""

    rows: TierRows
    match: PanelMatch


def match_tiers(tiers: Mapping[str, TierRows],
                variants: Mapping[str, PanelVariants]) -> dict[str, MatchedTier]:
    """Match each tier's rows to the panel once, so three maps cost one match."""
    return {key: MatchedTier(tier, match_to_panel(tier.chromosome, tier.position, tier.allele_1,
                                                  tier.allele_2, variants))
            for key, tier in tiers.items()}


def shared_set_rows(tier: TierRows, variants: Mapping[str, PanelVariants],
                    genetic_map: GeneticMap) -> np.ndarray:
    """The tier's rows this map can judge: matched to the panel, with a finite cM.

    The same rule `run.build_shared_set` applies, spelled out here because the
    maps are compared on one read of each tier and matching the panel is the
    expensive half. `test_the_shared_set_spells_out_the_rule_build_shared_set_applies`
    holds the two together.
    """
    return _kept(MatchedTier(tier, match_to_panel(tier.chromosome, tier.position, tier.allele_1,
                                                  tier.allele_2, variants)), genetic_map)


def _kept(matched: MatchedTier, genetic_map: GeneticMap) -> np.ndarray:
    cm = genetic_map.cm_at(matched.rows.chromosome, matched.rows.position)
    return np.asarray((matched.match.status == "matched") & ~np.isnan(cm), dtype=bool)


def variant_index_digest(store_id: str, analysis_id: str, variant_index: np.ndarray) -> str:
    """sha256 over (Store, Analysis, Store Variant Index), the indices in order."""
    digest = hashlib.sha256()
    digest.update(f"{store_id}\t{analysis_id}\n".encode())
    digest.update(np.sort(np.asarray(variant_index, dtype="<i8")).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MapSharedSet:
    """One map's shared set per Analysis, and the digest over all of them."""

    name: str
    analyses: dict[str, dict[str, Any]]

    def digest(self) -> str:
        digest = hashlib.sha256()
        for key, record in self.analyses.items():
            digest.update(f"{key}\t{record['shared_set']}\t{record['sha256']}\n".encode())
        return digest.hexdigest()

    def record(self) -> dict[str, Any]:
        return {"map": self.name, "analyses": self.analyses, "sha256": self.digest(),
                "shared_set_rows": sum(r["shared_set"] for r in self.analyses.values()),
                "tier_rows": sum(r["rows"] for r in self.analyses.values())}


def shared_set_of(matched: Mapping[str, MatchedTier], genetic_map: GeneticMap, *,
                  name: str) -> MapSharedSet:
    """Each Analysis's shared-set size and digest on one map, keyed `STORE/analysis`."""
    analyses = {}
    for key, one in matched.items():
        store_id, analysis_id = key.split("/", 1)
        index = one.rows.variant_index[_kept(one, genetic_map)]
        analyses[key] = {"rows": int(one.rows.variant_index.size),
                         "shared_set": int(index.size),
                         "sha256": variant_index_digest(store_id, analysis_id, index)}
    return MapSharedSet(name, analyses)


def check_same_shared_set(reference: MapSharedSet,
                          others: Mapping[str, MapSharedSet]) -> None:
    """Refuse any map whose shared set is not `reference`'s, row for row."""
    for name, other in others.items():
        if set(other.analyses) != set(reference.analyses):
            raise SharedSetMismatch(f"{name}: swept Analyses {sorted(other.analyses)} are not "
                                    f"{sorted(reference.analyses)}")
        for key, mine in reference.analyses.items():
            theirs = other.analyses[key]
            if theirs["shared_set"] != mine["shared_set"]:
                raise SharedSetMismatch(
                    f"{name}: {key}: {mine['shared_set']:,} rows are in {reference.name}'s shared "
                    f"set but {theirs['shared_set']:,} rows are in this map's, so the two maps "
                    "would judge different hits")
            if theirs["sha256"] != mine["sha256"]:
                raise SharedSetMismatch(
                    f"{name}: {key}: the same {mine['shared_set']:,} rows but another sha256, so "
                    "the two maps do not rank the same Store Variant Indices")


def measure_shared_sets(selection: dict[str, Any], stores_dir: Path, panel_dir: Path,
                        maps_by_name: Mapping[str, LoadedMap], *, tier: float = TIER,
                        only: int | None = None) -> dict[str, MapSharedSet]:
    """Every map's shared sets over the drawn Analyses, reading each tier once.

    The panel is read once, not once per Analysis: matching against it is the
    expensive half of building the shared set.
    """
    panel = Panel(panel_dir)
    variants = {chromosome: panel.variants(chromosome) for chromosome in panel.chromosomes}
    matched: dict[str, MatchedTier] = {}
    for chosen in selection["chosen"][:only]:
        key = f"{chosen['store']}/{chosen['analysis_id']}"
        store = stores_dir / chosen["store"] / "store.opengwasdb"
        rows, _ = read_tier(store, chosen["analysis_id"], tier)
        matched |= match_tiers({key: rows}, variants)
    return {name: shared_set_of(matched, loaded.map, name=name)
            for name, loaded in maps_by_name.items()}


def interval_jump(genetic_map: GeneticMap, chromosome: str, position: int) -> dict[str, Any]:
    """The map's interval around `position`, and the cM it gains there.

    #402's two spikes are exactly this: one interval narrower than a window that
    gains more cM than the rule's D can cross. On a map with a point at
    `position` that is the jump from the point to the next; on a map with its own
    points, HapMap II's for one, it is the interval that contains the position.
    Measured from the map, so no report types it.
    """
    points, cm = genetic_map.points(chromosome)
    at = int(np.searchsorted(points, position))
    start = at if at < points.size and int(points[at]) == position else at - 1
    if start < 0 or start + 1 >= points.size:
        raise MapBuildError(f"chr{chromosome}:{position:,} has no interval after it on the map")
    return {"chromosome": chromosome, "position": position,
            "interval_start": int(points[start]), "interval_end": int(points[start + 1]),
            "interval_bp": int(points[start + 1] - points[start]),
            "cm_gain": float(cm[start + 1] - cm[start]),
            "at_a_map_point": int(points[start]) == position}


def map_measurements(loaded: Mapping[str, LoadedMap],
                     spikes: tuple[Spike, ...] = EUR_SPIKES) -> dict[str, dict[str, Any]]:
    """Each map's coverage, genetic length, and what it gains across each spike interval."""
    measured = {}
    for name, one in loaded.items():
        measured[name] = {
            "chromosomes": len(one.map.chromosomes),
            "points": sum(int(one.map.points(c)[0].size) for c in one.map.chromosomes),
            "total_cM": sum(float(one.map.points(c)[1][-1]) for c in one.map.chromosomes),
            "spike_intervals": [interval_jump(one.map, s.chromosome, s.position)
                                for s in spikes],
        }
    return measured


PER_LEAD_KEYS = ("locus_leads", "index_variants", "spurious_splits", "lost_signals",
                 "spurious_split_rate", "lost_signal_rate", "disagreement_rate",
                 "median_spurious_split_rate", "median_lost_signal_rate")


def per_lead_rows(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each (D, W)'s pooled per-lead rates at each r^2, from a sweep's evaluation block.

    The rates are diagnostics since #403's owner decision; they are kept because
    a map that wins on counts by losing half the signals has to be visible.
    """
    return [{"d_cm": config["d_cm"], "w_bp": config["w_bp"],
             "by_r2": {r2: {key: pooled[key] for key in PER_LEAD_KEYS}
                       for r2, pooled in config["by_r2"].items()},
             "clearly_better_than_bp_only": config.get("clearly_better_than_bp_only"),
             "qualifies_pooled": config.get("qualifies_pooled")}
            for config in evaluation["configs"]]


def selection_digest(selection: Mapping[str, Any]) -> str:
    """sha256 of a selection record, as the sweep and the map comparison spell it."""
    return hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()


def comparison_record(selection: Mapping[str, Any], sweeps: Mapping[str, tuple[Path, dict]],
                      shared_sets: Mapping[str, MapSharedSet], *,
                      reference: str,
                      measurements: Mapping[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The maps' count criteria side by side, with the shared-set check beside them.

    Each sweep is passed with its path, so the record names the file its numbers
    came from and that file's sha256. The shared sets are checked here, before
    anything is written: two maps measured on different rows are not comparable,
    and a report over them would be a plausible wrong answer.
    """
    check_same_shared_set(shared_sets[reference], {name: found for name, found
                                                  in shared_sets.items() if name != reference})
    maps_record = {name: _map_record(name, path, sweep, shared_sets[name])
                   for name, (path, sweep) in sweeps.items()}
    return {
        "selection": {"sha256": selection_digest(selection),
                      "chosen": len(selection["chosen"]),
                      "swept": {name: record["analyses"]
                                for name, record in maps_record.items()}},
        "reference_map": reference,
        "shared_sets_equal": True,
        "map_measurements": dict(measurements or {}),
        "maps": maps_record,
    }


def _map_record(name: str, path: Path, sweep: dict[str, Any],
                whole: MapSharedSet) -> dict[str, Any]:
    """One map's counts and diagnostics, and the rows the maps are checked on."""
    if name != whole.name:
        raise SharedSetMismatch(f"{name}: the shared sets are {whole.name}'s")
    swept = {f"{a['store']}/{a['analysis']['analysis_id']}" for a in sweep["analyses"]}
    below = sweep["left_out_below_shared_set_floor"]
    return {
        "sweep": str(path), "sweep_sha256": file_digest(path, "sha256"),
        "analyses": sweep["counts"]["analyses"],
        "grid": sweep["grid"], "genetic_map": sweep["genetic_map"],
        "spikes": sweep.get("spikes", []),
        "criteria": sweep["counts"]["criteria"],
        "counts": sweep["counts"],
        "best_by_per_lead": sweep["evaluation"]["best"],
        "per_lead": per_lead_rows(sweep["evaluation"]),
        "below_shared_set_floor": {
            "floor": below["floor"],
            "analyses": [f"{b['store']}/{b['analysis']['analysis_id']}"
                         for b in below["analyses"]]},
        # The swept Analyses' rows, and the digest the equality check ran on,
        # which covers every drawn Analysis, the ones below the floor included.
        "shared_set": MapSharedSet(name, {key: rows for key, rows in whole.analyses.items()
                                          if key in swept}).record(),
        "shared_set_checked": {"analyses": len(whole.analyses), "sha256": whole.digest()},
    }
