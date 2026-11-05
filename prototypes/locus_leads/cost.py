"""What genetic-distance pruning costs, against plink2 clumping (#406).

Three measurements, on one machine on one date:

1. the per-variant cM array over a Store's whole Variant Table — what a Store
   would carry, and what an augment step would pay to write it;
2. query-time pruning of a Top-Hit tier, stage by stage: the tier's own query,
   the positions, the cM lookup, the sort, and the rule;
3. plink2 clumping of the same Analyses' shared set, genotypes included.

The ratio in the report is 2 against 3 on identical rows. Everything here is a
measurement: no stage is timed from a record that was not produced, and a run
refuses to publish when the faster rule disagrees with the rule.

`fast_locus_leads` is #406's one addition to the rule. The rule visits each row
in Python, which costs about 1.6 us a row — around 15 s on a 9.4M-row tier. It
reaches the same leads by asking a block of rows at once whether the accepted
leads already exclude them: within a chromosome the rows in position order have
non-decreasing cM, so a lead's exclusion zone (within W bp *and* within D cM) is
the intersection of two rank intervals, and therefore one rank interval. Only
the rows no interval covers are visited one at a time — and those are the leads.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import platform
import resource
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from opengwasdb.query.facade import query_store
from opengwasdb.variants.axis import VariantAxis

from . import clump, rule
from . import run as run_mod
from .genetic_map import LoadedMap, load_map
from .panel import Panel, plink2
from .run import SharedSet, TierRows, build_shared_set, read_tier

STAGES = ("tier", "positions", "cm", "sort", "rule")
# What a budget or a ratio adds up. `positions_array` is the same read from a
# per-variant array — the follow-up #62 names if the Variant Table proves the
# bottleneck. `sort` is absent deliberately: it is timed for information, and
# both rules sort inside their own time, so charging it would charge a sort twice.
PRUNE_STAGES = ("positions", "cm", "rule")
FAST_PRUNE_STAGES = ("positions_array", "cm", "rule_fast")
SORT_NOTE = ("the `sort` stage is timed for information only: both rules sort inside their "
             "own time, so no total, budget or ratio adds the separate stage")
BUDGET = {
    "quantity": "the default Top-Hit Query's own wall time for the same slice, "
                "imputed rows included, and the same query resolved",
    "judged": "pruning runs per query when its stages cost no more than the query they "
              "follow (fast_ratio <= 1), so the reader waits at most twice as long",
}
# Blocks start small. The rows a block leaves uncovered are handled one at a
# time and re-checked against each other, so the first block of a chromosome is
# the expensive one: it starts with nothing covered. Once coverage is dense a
# block can be large — the vectorised test costs the same either way, and a
# larger one is asked fewer times.
BLOCK_MIN, BLOCK_MAX = 1 << 12, 1 << 18
_INTERVAL = 0.005
_PAGE_MB = resource.getpagesize() / 1024.0 / 1024.0


# ------------------------------------------------------------------ measuring

def load_average() -> dict[str, float]:
    """The node's 1, 5 and 15 minute load averages."""
    one, five, fifteen = os.getloadavg()
    return {"one": round(one, 2), "five": round(five, 2), "fifteen": round(fifteen, 2)}


def rss_mb(pid: int) -> float | None:
    """`pid`'s current RSS in MB, or None when it has already gone."""
    try:
        with open(f"/proc/{pid}/statm") as handle:
            return int(handle.read().split()[1]) * _PAGE_MB
    except (OSError, IndexError):
        return None


def child_pids(parent: int) -> list[int]:
    """The pids whose parent is `parent`, from the kernel's own list.

    `/proc/<pid>/task/<pid>/children` costs microseconds; walking all of /proc
    costs about 50 ms here, which is too much to sample with.
    """
    try:
        listed = Path(f"/proc/{parent}/task/{parent}/children").read_text()
        return [int(pid) for pid in listed.split()]
    except OSError:
        return _child_pids_by_scan(parent)


def _child_pids_by_scan(parent: int) -> list[int]:
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            stat = Path(f"/proc/{name}/stat").read_bytes()
        except OSError:
            continue
        # The command name may hold spaces and parentheses, so the fields that
        # matter start after the last ')'.
        fields = stat[stat.rfind(b")") + 2:].split()
        if len(fields) > 1 and int(fields[1]) == parent:
            pids.append(int(name))
    return pids


def own_rss() -> float:
    return rss_mb(os.getpid()) or 0.0


def child_rss(expected: str) -> float | None:
    """The highest RSS among this process's children running the `expected` executable.

    A child counts only once it has executed that program. Between `fork` and
    `exec` it is a copy-on-write clone of this process: its RSS is this
    process's, several gigabytes on a large shared set that plink2 has not
    touched. That clone is recognised by a command line identical to this
    process's own, which `exec` replaces.

    None when no such child was alive at a poll. A short-lived child can start and
    finish between polls, and a peak nobody observed is missing evidence — never 0
    MB, which would read as "it used no memory".
    """
    seen = [value for value in (rss_mb(pid) for pid in child_pids(os.getpid())
                                if _runs(pid, expected)) if value is not None]
    return max(seen) if seen else None


def _runs(pid: int, expected: str) -> bool:
    """Whether `pid` has executed the `expected` program, rather than being a fork of us."""
    try:
        executable = Path(os.readlink(f"/proc/{pid}/exe")).name
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    if not executable.startswith(expected):
        return False
    return command != Path("/proc/self/cmdline").read_bytes()


class PeakRss:
    """Poll RSS on a background thread and keep the maximum seen, and how often it was seen.

    Current RSS from /proc/<pid>/statm, never `ru_maxrss`: under `pixi run` the
    high-water mark starts near 2 GiB and cannot be reset, and a query's
    intermediate buffers are freed before it returns, so a before-and-after read
    misses the peak entirely.

    `peak_mb` is None until a poll observes something, and stays None if none
    does; `observations` counts the polls that did, so a missing peak is
    distinguishable from a small one.
    """

    def __init__(self, reader: Callable[[], float | None], interval: float = _INTERVAL) -> None:
        self._reader = reader
        self._interval = interval
        self._stop = threading.Event()
        self.peak_mb: float | None = None
        self.observations = 0

    def __enter__(self) -> PeakRss:
        self._observe()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _observe(self) -> None:
        value = self._reader()
        if value is not None:
            self.observations += 1
            self.peak_mb = value if self.peak_mb is None else max(self.peak_mb, value)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._observe()
            self._stop.wait(self._interval)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._observe()


def peak_child_rss(argv: Sequence[str], interval: float = _INTERVAL) -> dict[str, Any]:
    """The highest RSS a child running `argv` reaches, and how many polls saw it."""
    with PeakRss(lambda: child_rss(Path(argv[0]).name), interval) as sampler:
        subprocess.run(list(argv), check=True)
    return {"peak_mb": sampler.peak_mb, "observations": sampler.observations,
            "interval_s": interval}


def time_stages(stages: Mapping[str, Callable[[], object]], *, repeats: int,
                child: str | None = None) -> dict[str, Any]:
    """Each stage timed `repeats` times in order, with its loads and its peak RSS.

    `child` names an executable whose child processes carry the peak to sample —
    plink2's, when clumping, is not this process's. Without it the peak is this
    process's own.
    """
    reader = (lambda: child_rss(child)) if child else own_rss
    return {name: _timed_stage(stage, repeats=repeats, reader=reader, child=child)
            for name, stage in stages.items()}


def _timed_stage(stage: Callable[[], object], *, repeats: int,
                 reader: Callable[[], float | None], child: str | None) -> dict[str, Any]:
    """One stage's repeats: every sample, every load, and the highest peak observed."""
    samples: list[float] = []
    loads: list[dict[str, float]] = []
    peak: float | None = None
    observations = 0
    for _ in range(repeats):
        loads.append(load_average())
        with PeakRss(reader) as sampler:
            started = time.perf_counter()
            stage()
            samples.append(time.perf_counter() - started)
        observations += sampler.observations
        peak = _higher(peak, sampler.peak_mb)
    return {
        "samples_s": [round(value, 4) for value in samples],
        "median_s": round(float(statistics.median(samples)), 4),
        "min_s": round(min(samples), 4),
        "max_s": round(max(samples), 4),
        "peak_mb": None if peak is None else round(peak, 1),
        "rss_observations": observations,
        "peak_source": "children" if child else "process",
        "child": child,
        "loads": loads,
    }


def _higher(peak: float | None, observed: float | None) -> float | None:
    """The higher of two peaks; an unobserved one never displaces an observed one."""
    if observed is None:
        return peak
    return observed if peak is None else max(peak, observed)


# ----------------------------------------------------------------- provenance

def _git(*arguments: str) -> str:
    root = Path(__file__).resolve().parents[2]
    done = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else "unknown"


def machine() -> dict[str, Any]:
    """The node a measurement was taken on."""
    cpuinfo = Path("/proc/cpuinfo").read_text().splitlines()
    model = next((line.split(":", 1)[1].strip() for line in cpuinfo
                  if line.startswith("model name")), "unknown")
    total_kb = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemTotal"))
    return {
        "hostname": socket.gethostname(),
        "cpu_model": model,
        "cores": os.cpu_count() or 0,
        "physical_cores": len(set(_cpu_sockets(cpuinfo))),
        "memory_gb": round(total_kb / 1024 / 1024, 1),
        "platform": platform.platform(),
    }


def _cpu_sockets(cpuinfo: list[str]) -> list[tuple[str, str]]:
    """Each processor's (package, core), so a hyperthreaded node states both counts."""
    pairs, package = [], ""
    for line in cpuinfo:
        if line.startswith("physical id"):
            package = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            pairs.append((package, line.split(":", 1)[1].strip()))
    return pairs


def provenance() -> dict[str, Any]:
    """What the numbers beside it were measured at.

    The commit alone does not say which code ran — this measurement was taken from
    a working tree — so the prototype's own sources are digested beside it.
    """
    version = subprocess.run([plink2(), "--version"], check=True, capture_output=True, text=True)
    return {
        "commit": _git("rev-parse", "--short", "HEAD"),
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": machine(),
        "sources": sources(),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "opengwasdb_path": str(Path(sys.modules["opengwasdb"].__file__).resolve().parent),
            "plink2": version.stdout.strip().splitlines()[0],
        },
    }


def sources() -> dict[str, str]:
    """A digest of each module of this prototype, so an artifact names the code it ran."""
    package = Path(__file__).resolve().parent
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            for path in sorted(package.glob("*.py"))}


# ------------------------------------------------------------------ the budget

def _median(stages: Mapping[str, Mapping[str, Any]], name: str) -> float:
    """One stage's median; a stage that was not timed is a KeyError, never a zero."""
    return float(stages[name]["median_s"])


def budget_record(stages: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """What pruning costs against the Top-Hit Query it follows, from the medians.

    Two baselines: the tier's own query, and the same query resolved — the
    workload a reader waits for when it wants positions and ALIDs, which is what
    pruning needs. `sort` is charged to neither total (see `SORT_NOTE`).
    """
    query = _median(stages, "tier")
    resolved = _median(stages, "resolved_query")
    pruning = sum(_median(stages, name) for name in PRUNE_STAGES)
    fast = sum(_median(stages, name) for name in FAST_PRUNE_STAGES)
    return {
        "definition": BUDGET["quantity"],
        "judged": BUDGET["judged"],
        "sort_note": SORT_NOTE,
        "charged": {"as_written": list(PRUNE_STAGES), "fast": list(FAST_PRUNE_STAGES)},
        "query_s": round(query, 4),
        "resolved_query_s": round(resolved, 4),
        "pruning_s": round(pruning, 4),
        "ratio": round(pruning / query, 4) if query else None,
        "within_budget_as_written": bool(query) and pruning <= query,
        "fast_pruning_s": round(fast, 4),
        "fast_ratio": round(fast / query, 4) if query else None,
        "fast_ratio_of_resolved": round(fast / resolved, 4) if resolved else None,
        "within_budget": bool(query) and fast <= query,
    }


# ------------------------------------------------------------ the faster rule

class _Covered:
    """One chromosome's excluded ranks, as disjoint ascending [start, end) intervals.

    Each interval carries the lead that claimed it. A later lead's interval is
    added only where nothing covers it yet, so a rank keeps the strongest lead
    that reaches it — the one the rule accepts first.
    """

    def __init__(self) -> None:
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.owner: list[int] = []

    def covering(self, ranks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each rank, whether it is already excluded, and by which row."""
        if not self.starts:
            return np.zeros(ranks.size, dtype=bool), np.zeros(ranks.size, dtype="int64")
        at = np.searchsorted(np.asarray(self.starts), ranks, side="right") - 1
        safe = np.clip(at, 0, None)
        covered = (at >= 0) & (ranks < np.asarray(self.ends)[safe])
        return covered, np.where(covered, np.asarray(self.owner)[safe], -1)

    def add(self, start: int, end: int, row: int) -> None:
        """Claim [start, end) for `row`, keeping an earlier lead wherever they overlap."""
        at = bisect.bisect_left(self.ends, start)
        cursor = start
        while at < len(self.starts) and self.starts[at] < end:
            if cursor < self.starts[at]:
                self._insert(at, cursor, min(self.starts[at], end), row)
                at += 1
            cursor = max(cursor, self.ends[at])
            at += 1
        if cursor < end:
            self._insert(at, cursor, end, row)

    def _insert(self, at: int, start: int, end: int, row: int) -> None:
        self.starts.insert(at, start)
        self.ends.insert(at, end)
        self.owner.insert(at, row)


def fast_locus_leads(
    chromosome: np.ndarray, position: np.ndarray, cm: np.ndarray, abs_z: np.ndarray,
    variant_index: np.ndarray, *, d_cm: float, w_bp: int,
) -> np.ndarray:
    """The same leads as `rule.locus_leads`, a block of rows at a time.

    The leads are identical, not approximate: the exclusion zone of a lead is
    computed from the same two conditions the rule tests, as the exact set of
    row ranks that satisfy them. cM that falls with position is refused, because
    the interval arithmetic rests on it not falling (as `check_cm` does for a
    run).
    """
    chromosome = np.asarray(chromosome)
    position = np.asarray(position, dtype="int64")
    cm = np.asarray(cm, dtype="float64")
    if not np.all(np.isfinite(cm)):
        raise ValueError("every row needs a finite cM; rows without one are ranked")
    lead_of = np.full(cm.shape, -1, dtype="int64")
    rank_in_order = np.empty(cm.shape, dtype="int64")
    rank_in_order[rule.rule_order(abs_z, variant_index)] = np.arange(cm.size)
    for label in dict.fromkeys(chromosome.tolist()):
        rows = np.flatnonzero(chromosome == label)
        _solve_chromosome(label, rows, position, cm, rank_in_order, lead_of,
                          d_cm=d_cm, w_bp=w_bp)
    return lead_of


def _solve_chromosome(label: str, rows: np.ndarray, position: np.ndarray, cm: np.ndarray,
                      rank_in_order: np.ndarray, lead_of: np.ndarray, *, d_cm: float,
                      w_bp: int) -> None:
    """Assign one chromosome's rows, in the rule's order, against its merged intervals."""
    by_position = rows[np.argsort(position[rows], kind="stable")]
    positions, cms = position[by_position], cm[by_position]
    if np.any(np.diff(cms) < 0):
        raise ValueError(f"chromosome {label}: cM falls with position; the faster rule needs "
                         "cM non-decreasing along the chromosome")
    rank_of_row = np.empty(position.size, dtype="int64")
    rank_of_row[by_position] = np.arange(by_position.size)
    covered = _Covered()
    sequence = rows[np.argsort(rank_in_order[rows], kind="stable")]
    start, size = 0, BLOCK_MIN
    while start < sequence.size:
        block = sequence[start:start + size]
        ranks = rank_of_row[block]
        excluded, owner = covered.covering(ranks)
        lead_of[block[excluded]] = owner[excluded]
        accepted = _accept(block[~excluded], ranks[~excluded], lead_of, positions, cms,
                           d_cm=d_cm, w_bp=w_bp)
        for low, high, row in accepted:
            covered.add(low, high, row)
        start += size
        size = _next_block(size, len(accepted))


def _next_block(size: int, uncovered: int) -> int:
    """The next block's size: larger while coverage is dense, smaller while it is not.

    A block's size moves no result — only how many rows reach the one-at-a-time
    path — so it is free to follow the data.
    """
    if uncovered * 8 <= size:
        return min(size * 2, BLOCK_MAX)
    return max(size // 2, BLOCK_MIN)


def _accept(block: np.ndarray, ranks: np.ndarray, lead_of: np.ndarray, positions: np.ndarray,
            cms: np.ndarray, *, d_cm: float, w_bp: int) -> list[tuple[int, int, int]]:
    """The rows of one block no earlier lead reaches, and their exclusion intervals."""
    accepted: list[tuple[int, int, int]] = []
    for row, rank in zip(block.tolist(), ranks.tolist(), strict=True):
        low, high = _exclusion(int(positions[rank]), float(cms[rank]), positions, cms,
                               d_cm=d_cm, w_bp=w_bp)
        lead = next((lead for low_there, high_there, lead in accepted
                     if low_there <= rank < high_there), None)
        lead_of[row] = row if lead is None else lead
        if lead is None:
            accepted.append((low, high, row))
    return accepted


def _exclusion(position: int, cm: float, positions: np.ndarray, cms: np.ndarray, *,
               d_cm: float, w_bp: int) -> tuple[int, int]:
    """The rank interval a lead at (`position`, `cm`) excludes, both bounds included."""
    low = max(int(np.searchsorted(positions, position - w_bp, side="left")),
              int(np.searchsorted(cms, cm - d_cm, side="left")))
    high = min(int(np.searchsorted(positions, position + w_bp, side="right")),
               int(np.searchsorted(cms, cm + d_cm, side="right")))
    return low, high


# ---------------------------------------------------- the per-variant cM array

@dataclass(frozen=True)
class CmAxis:
    """The per-variant arrays a Store would carry, in Store Variant Index order."""

    store: Path
    labels: tuple[str, ...]
    chromosome: np.ndarray  # int8 code into `labels`, one per variant
    position: np.ndarray  # int32
    cm: np.ndarray  # float32, NaN off the map
    record: dict[str, Any]

    def cm_at(self, variant_index: np.ndarray) -> np.ndarray:
        """cM for these Store Variant Indices, as a Store would look them up."""
        index = np.asarray(variant_index, dtype="int64")
        return self.cm[index]


def build_cm_axis(store: Path, genetic_map: LoadedMap, *, repeats: int) -> CmAxis:
    """Interpolate cM at every variant of `store`'s axis, and keep what a Store would carry."""
    held: dict[str, Any] = {}
    stages = {
        "read_variant_table": lambda: held.update(coordinates=_read_coordinates(store)),
        "interpolate": lambda: held.update(
            cm=genetic_map.map.cm_at(_on_label(held["coordinates"]), held["coordinates"].position)),
    }
    record = time_stages(stages, repeats=repeats)
    coordinates = held["coordinates"]
    cm = np.asarray(held["cm"], dtype="float32")
    labels = np.asarray(coordinates.labels, dtype=object)
    off_map = ~np.isin(labels[coordinates.chromosome], genetic_map.map.chromosomes)
    stages_record = {
        "stages": record,
        "variants": int(coordinates.position.size),
        "off_map_variants": int(off_map.sum()),
        "off_map_chromosomes": sorted(set(labels[coordinates.chromosome][off_map].tolist())),
        "mapped_bytes": int(sum(array.nbytes for array in
                                (coordinates.chromosome, coordinates.position, cm))),
        "cm_dtype": "float32",
        "store": store_identity(store),
        "genetic_map": genetic_map.identity,
    }
    return CmAxis(store, coordinates.labels, coordinates.chromosome, coordinates.position, cm,
                  stages_record)


def _on_label(coordinates: _Coordinates) -> np.ndarray:
    """The chromosome labels as the map's `cm_at` reads them: one label per variant."""
    return np.asarray(coordinates.labels, dtype=object)[coordinates.chromosome]


@dataclass(frozen=True)
class _Coordinates:
    labels: tuple[str, ...]
    chromosome: np.ndarray
    position: np.ndarray


def _read_coordinates(store: Path) -> _Coordinates:
    """Chromosome and position from the Store Variant Table, in Store Variant Index order."""
    frame = pd.read_csv(store / "variants.tsv.gz", sep="\t",
                        usecols=["#chromosome", "position", "variant_index"],
                        dtype={"#chromosome": "category", "position": "int64",
                               "variant_index": "int64"})
    variant_index = frame["variant_index"].to_numpy()
    if not np.array_equal(variant_index, np.arange(variant_index.size)):
        raise ValueError(f"{store}: the Variant Table is not in Store Variant Index order")
    return _Coordinates(tuple(str(label) for label in frame["#chromosome"].cat.categories),
                        frame["#chromosome"].cat.codes.to_numpy().astype("int8"),
                        frame["position"].to_numpy().astype("int32"))


def store_identity(store: Path) -> dict[str, Any]:
    """The manifest fields a measurement is meaningless without."""
    manifest = _json(store / "manifest.json")
    return {"path": str(store), "store_id": manifest["store_id"],
            "release_id": manifest["release_id"],
            "format_version": manifest["format_version"],
            "primary_layout": str(manifest["primary_layout"]),
            "completion_state": str(manifest["completion_state"]),
            "reference_assembly": manifest["reference_assembly"]}


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


# ------------------------------------------------------------------ the slices

@dataclass(frozen=True)
class SliceSpec:
    """One Analysis at one Top-Hit tier, and why it was measured."""

    name: str
    store: Path
    analysis_id: str
    threshold: float
    why: str


def largest_tiers(root: Path, thresholds: Sequence[float] = (5e-8, 5e-4),
                  top: int = 5) -> dict[str, list[dict[str, Any]]]:
    """The largest single-Analysis tiers under `root`, from each `analyses.tsv`.

    The Top-Hit Counts include imputed rows, which is the row count a reader of
    the default Top-Hit Query prunes — so the largest slice is chosen on them.
    """
    return {f"{threshold:.0e}": _tier_counts(root, threshold)[:top] for threshold in thresholds}


def _tier_counts(root: Path, threshold: float) -> list[dict[str, Any]]:
    column = "n_hits_" + f"{threshold:.0e}".replace("e-0", "e").replace("e-", "e")
    rows = [row for path in sorted(root.glob("OGS-*/store.opengwasdb/analyses.tsv"))
            for row in _counts(path, column, threshold)]
    return sorted(rows, key=lambda row: -row["rows"])


def _counts(path: Path, column: str, threshold: float) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, sep="\t", usecols=["analysis_id", column],
                        dtype={"analysis_id": str})
    store = path.parent
    return [{"store": str(store), "store_id": store.parent.name,
             "analysis_id": str(row.analysis_id), "threshold": threshold,
             "rows": int(getattr(row, column))}
            for row in frame.itertuples() if int(getattr(row, column) or 0) > 0]


def slice_specs(root: Path) -> list[SliceSpec]:
    """The slices this run measures: the two largest tiers, mid-size ones, and the tracer."""
    specs = [SliceSpec(name=f"largest_{name}", store=Path(row["store"]),
                       analysis_id=row["analysis_id"], threshold=row["threshold"],
                       why=f"the largest single-Analysis {name} tier under {root}")
             for name, rows in largest_tiers(root).items() if rows
             for row in rows[:1]]
    specs += [
        SliceSpec("mvp_5e-4", Path("/data/opengwasdb/stores/OGS-00012/store.opengwasdb"),
                  "GCST90475362", 5e-4, "the largest 5e-4 tier outside OGS-00011"),
        SliceSpec("mvp_5e-8", Path("/data/opengwasdb/stores/OGS-00012/store.opengwasdb"),
                  "GCST90475362", 5e-8, "a Dense Reference-Completed tier an eighth the size"),
        SliceSpec("height_5e-8", Path("/data/opengwasdb/stores/OGS-00011/store.opengwasdb"),
                  "GCST90245992", 5e-8, "a Hybrid tier most of whose rows are imputed"),
        SliceSpec("observed_only_5e-8",
                  Path("/data/opengwasdb/stores/OGS-00011/store.opengwasdb"),
                  "GCST002047", 5e-8, "the largest Observed-Only tier: no imputed rows at all"),
        SliceSpec("tracer", Path("/data/opengwasdb/stores/OGS-00003/store.opengwasdb"),
                  "finngen-r13-BMI_IRN", 5e-8, "#402's tracer Analysis"),
    ]
    return specs


# ---------------------------------------------------------------- one slice

@dataclass(frozen=True)
class _ClumpParameters:
    panel: Panel
    work_dir: Path
    repeats: int
    threads: int
    r2: float
    threshold: float


def measure_slice(spec: SliceSpec, *, genetic_map: LoadedMap, panel: Panel, axis: CmAxis,
                  work_dir: Path, repeats: int, threads: int, d_cm: float, w_bp: int,
                  r2: float) -> dict[str, Any]:
    """One tier's pruning stages, its clumping, and the ratio between them."""
    held: dict[str, Any] = {}
    stages = _tier_stages(spec, axis, held, d_cm=d_cm, w_bp=w_bp)
    stages["read_tier"] = lambda: held.update(prototype=read_tier(
        spec.store, spec.analysis_id, spec.threshold))
    timed = time_stages(stages, repeats=repeats)
    rows, leads = _tier_record(held, timed, axis)
    clumping = _clump_measurement(held["prototype"], genetic_map=genetic_map, panel=panel,
                                  axis=axis,
                                  parameters=_ClumpParameters(panel=panel, work_dir=work_dir,
                                                              repeats=repeats, threads=threads,
                                                              r2=r2, threshold=spec.threshold),
                                  d_cm=d_cm, w_bp=w_bp)
    return {
        "name": spec.name, "why": spec.why, "store": store_identity(spec.store),
        "analysis_id": spec.analysis_id, "threshold": spec.threshold,
        "rows": rows, "stages": timed, "budget": budget_record(timed), "leads": leads,
        "clumping": clumping, "ratio": clumping["ratio"],
    }


def _tier_stages(spec: SliceSpec, axis: CmAxis, held: dict[str, Any], *, d_cm: float,
                 w_bp: int) -> dict[str, Callable[[], object]]:
    """Every stage a Store would run to prune the tier it has just read."""
    return {
        "tier": lambda: held.update(hits=_read_hits(spec)),
        "positions": lambda: held.update(positions=_positions_from_table(spec.store, held["hits"])),
        "positions_array": lambda: held.update(positions_array=_positions_from_array(
            axis, held["hits"])),
        "cm": lambda: held.update(cm=axis.cm_at(held["hits"]["variant_index"])),
        "sort": lambda: held.update(order=rule.rule_order(held["hits"]["abs_z"],
                                                          held["hits"]["variant_index"])),
        "rule": lambda: held.update(leads=_leads(held, rule.locus_leads, d_cm=d_cm, w_bp=w_bp)),
        "rule_fast": lambda: held.update(fast=_leads(held, fast_locus_leads, d_cm=d_cm, w_bp=w_bp)),
        # The baseline a reader actually waits for when it needs positions and
        # ALIDs: the same default query, every row resolved, imputed included.
        "resolved_query": lambda: held.update(resolved=_read_resolved(spec)),
    }


def _read_hits(spec: SliceSpec) -> dict[str, np.ndarray]:
    """The default Top-Hit Query: variant index, z and status, imputed rows included."""
    query = query_store(spec.store)
    try:
        hits = query.top_hits(analysis_id=spec.analysis_id, threshold=spec.threshold)
        return {"variant_index": np.asarray(hits["variant_index"], dtype="int64"),
                "abs_z": np.abs(np.asarray(hits["z"], dtype="float64")),
                "status": np.asarray(hits["association_status"]).astype(str)}
    finally:
        query.close()


def _read_resolved(spec: SliceSpec) -> int:
    """The default Top-Hit Query with every row resolved, as `run.read_tier` does for observed."""
    query = query_store(spec.store)
    try:
        hits = query.top_hits(analysis_id=spec.analysis_id, threshold=spec.threshold)
        resolved = list(query.resolve(hits))
    finally:
        query.close()
    if len(resolved) != hits.count:
        raise ValueError(f"{spec.analysis_id}: {len(resolved):,} rows resolved of "
                         f"{hits.count:,} in the tier")
    return len(resolved)


def _positions_from_table(store: Path, hits: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Chromosome and position for the tier's Store Variant Indices, from the Variant Table."""
    identity = VariantAxis(store).identity_by_indices(hits["variant_index"])
    if identity is None:
        raise ValueError(f"{store}: no ALID index, so positions need a full Variant Table scan")
    return {"chromosome": identity["chromosome"].astype(str),
            "position": identity["position"].astype("int64")}


def _positions_from_array(axis: CmAxis, hits: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The same, from the per-variant arrays built beside the cM array."""
    index = np.asarray(hits["variant_index"], dtype="int64")
    return {"chromosome": np.asarray(axis.labels, dtype=object)[axis.chromosome[index]],
            "position": axis.position[index].astype("int64")}


def _on_map(held: Mapping[str, Any]) -> np.ndarray:
    """The rows with finite cM: the only ones the rule ranks."""
    finite = np.isfinite(held["cm"])
    if not finite.any():
        raise ValueError("no row has a finite cM; the map and the tier do not overlap")
    return finite


def _leads(held: Mapping[str, Any], leads_at: Callable[..., np.ndarray], *, d_cm: float,
           w_bp: int) -> np.ndarray:
    """One of the two rules, over the rows that have a cM."""
    keep = _on_map(held)
    hits, positions = held["hits"], held["positions"]
    return leads_at(positions["chromosome"][keep], positions["position"][keep], held["cm"][keep],
                    hits["abs_z"][keep], hits["variant_index"][keep], d_cm=d_cm, w_bp=w_bp)


def _tier_record(held: dict[str, Any], timed: dict[str, Any],
                 axis: CmAxis) -> tuple[dict[str, Any], dict[str, Any]]:
    """The tier's row counts and its leads, the two rules agreeing before either is reported."""
    hits, keep = held["hits"], _on_map(held)
    slow, fast = held["leads"], held["fast"]
    differing = np.flatnonzero(slow != fast)
    if differing.size:
        raise ValueError(f"the faster rule differs from the rule at {differing.size:,} rows, "
                         f"first {differing[0]}")
    status = hits["status"][keep]
    rows = np.arange(slow.size)
    off_map = _off_map_labels(axis, hits["variant_index"][~keep])
    return (
        {"tier": int(hits["abs_z"].size), "observed": int(np.sum(status == "observed")),
         "imputed": int(np.sum(status == "imputed")), "finite_cm": int(keep.sum()),
         "off_map": int((~keep).sum()), "ranked": int(slow.size),
         "off_map_chromosomes": off_map,
         # A two-sided float64 p is 0 from |z| 38.5, where plink2 can no longer
         # rank by it; the rule ranks on abs_z for exactly that reason (#62).
         "max_abs_z": round(float(hits["abs_z"].max()), 4),
         "rows_abs_z_over_38": int(np.sum(hits["abs_z"] > 38))},
        {"rule": int(np.sum(slow == rows)), "identical_fast_rule": True},
    )


def _off_map_labels(axis: CmAxis, index: np.ndarray) -> list[str]:
    """The chromosome labels of the rows the map does not cover."""
    labels = np.asarray(axis.labels, dtype=object)[axis.chromosome[index]]
    return sorted({str(label) for label in np.unique(labels)})


def _clump_measurement(prototype: tuple[TierRows, dict[str, Any]], *, genetic_map: LoadedMap,
                       panel: Panel, axis: CmAxis, parameters: _ClumpParameters, d_cm: float,
                       w_bp: int) -> dict[str, Any]:
    """plink2's clumping of the shared set, per chromosome, and pruning of the same rows.

    The shared set is timed whole — what a Store run pays — and its panel match
    and its cM interpolation are timed apart from it, because clumping needs the
    match and not the cM work. The ratio is taken against all three.
    """
    tier, _ = prototype
    held: dict[str, Any] = {}
    stages: dict[str, Callable[[], object]] = {
        "shared_set": lambda: held.update(shared=build_shared_set(
            tier, parameters.panel, genetic_map.map)),
        "matching": lambda: held.update(match=run_mod.match_rows(tier, panel)),
        "shared_cm": lambda: held.update(shared_cm=_shared_cm(tier, genetic_map)),
    }
    stages["ranked"] = lambda: held.update(ranked=clump.ranked_neg_log10_p(
        held["shared"][0].rows.abs_z, held["shared"][0].rows.variant_index))
    timed = time_stages(stages, repeats=parameters.repeats)
    shared = held["shared"][0]
    clumped = _clump_by_chromosome(shared, held["ranked"], parameters)
    pruning = _shared_pruning(shared, axis, repeats=parameters.repeats, d_cm=d_cm, w_bp=w_bp)
    bills = {"matching_s": timed["matching"]["median_s"],
             "shared_cm_s": timed["shared_cm"]["median_s"],
             "shared_set_s": timed["shared_set"]["median_s"],
             "ranked_s": timed["ranked"]["median_s"],
             "plink2_s": clumped["clump_s"]}
    return {
        "shared_rows": int(shared.rows.chromosome.size),
        "shared_set_accounting": held["shared"][1],
        "stages": timed | clumped["stages"],
        "per_chromosome": clumped["per_chromosome"],
        "index_variants": clumped["index_variants"],
        "members": clumped["members"],
        "clump_s": clumped["clump_s"],
        "order_followed": clumped["order_followed"],
        "r2": clumped["r2"],
        "clump_kb": clumped["clump_kb"],
        "total_s": round(timed["shared_set"]["median_s"] + timed["ranked"]["median_s"]
                         + clumped["clump_s"], 4),
        "rows": int(shared.rows.chromosome.size),
        "shared_pruning": pruning,
        "ratio": ratio_record(pruning, bills, int(shared.rows.chromosome.size)),
    }


def _shared_cm(rows: TierRows, genetic_map: LoadedMap) -> np.ndarray:
    """The shared set's cM interpolation and its check, which only pruning needs."""
    cm = genetic_map.map.cm_at(rows.chromosome, rows.position)
    run_mod.check_cm(rows.chromosome, rows.position, cm, genetic_map.map)
    return cm


def _clump_by_chromosome(shared: SharedSet, ranked: clump.RankedScores,
                         parameters: _ClumpParameters) -> dict[str, Any]:
    """One plink2 run per chromosome, which is what `clump_rows` does internally.

    Timing the chromosomes separately therefore measures the same work as one
    call over the shared set, and gives the cost against row count the report
    needs. Every row must come back in a clump, and every clump's index variant
    must be the one the rule would accept first.
    """
    rows = shared.rows
    index_of = np.full(rows.chromosome.size, -1, dtype="int64")
    kept = list(dict.fromkeys(rows.chromosome.tolist()))
    stages = {}
    for label in kept:
        keep = np.flatnonzero(rows.chromosome == label)
        stages[f"chr{label}"] = _clump_stage(shared, ranked, index_of, keep, parameters)
    timed = time_stages(stages, repeats=parameters.repeats, child=Path(plink2()).name)
    clump.check_clumps_follow_rule_order(index_of, rows.abs_z, rows.variant_index)
    is_index = index_of == np.arange(index_of.size)
    return {
        "stages": timed,
        "per_chromosome": [{"chromosome": label, "rows": int(np.sum(rows.chromosome == label)),
                            "index_variants": int(np.sum(is_index[rows.chromosome == label])),
                            **timed[f"chr{label}"]} for label in kept],
        "index_variants": int(is_index.sum()),
        "members": int(np.sum(index_of >= 0)),
        "clump_s": round(sum(timed[f"chr{label}"]["median_s"] for label in kept), 4),
        "order_followed": True,
        "r2": parameters.r2,
        "clump_kb": clump.CLUMP_KB,
    }


def _clump_stage(shared: SharedSet, ranked: clump.RankedScores, index_of: np.ndarray,
                 keep: np.ndarray, parameters: _ClumpParameters) -> Callable[[], object]:
    """One chromosome's clumping, written into the whole tier's index array."""
    rows = shared.rows

    def run() -> object:
        # `clump_rows` answers in the row order of the arrays it was given, which
        # here is this chromosome's subset: map each answer back to its own row.
        index_of[keep] = keep[clump.clump_rows(
            parameters.panel, rows.chromosome[keep], shared.panel_id[keep],
            ranked.neg_log10_p[keep], r2=parameters.r2, work_dir=parameters.work_dir,
            threads=parameters.threads, p1=parameters.threshold, p2=parameters.threshold)]
        return None

    return run


def _shared_pruning(shared: SharedSet, axis: CmAxis, *, repeats: int, d_cm: float,
                    w_bp: int) -> dict[str, Any]:
    """The pruning stages over the shared set alone: the rows the ratio is taken on."""
    rows = shared.rows
    held: dict[str, Any] = {}
    stages = {
        "cm": lambda: held.update(cm=axis.cm_at(rows.variant_index)),
        "sort": lambda: held.update(order=rule.rule_order(rows.abs_z, rows.variant_index)),
        "rule": lambda: held.update(leads=_leads_at(rows, held["cm"], rule.locus_leads,
                                                    d_cm=d_cm, w_bp=w_bp)),
        "rule_fast": lambda: held.update(fast=_leads_at(rows, held["cm"], fast_locus_leads,
                                                        d_cm=d_cm, w_bp=w_bp)),
    }
    timed = time_stages(stages, repeats=repeats)
    differing = np.flatnonzero(held["leads"] != held["fast"])
    if differing.size:
        raise ValueError(f"the faster rule differs from the rule on the shared set at "
                         f"{differing.size:,} rows, first {differing[0]}")
    return timed


def _leads_at(rows: TierRows, cm: np.ndarray, leads_at: Callable[..., np.ndarray], *,
              d_cm: float, w_bp: int) -> np.ndarray:
    return leads_at(rows.chromosome, rows.position, cm, rows.abs_z, rows.variant_index,
                    d_cm=d_cm, w_bp=w_bp)


def _bill(name: str) -> str:
    """A clumping bill's ratio key: `plink2_s` is `plink2_only`, the rest lose the unit."""
    return "plink2_only" if name == "plink2_s" else name.removesuffix("_s")


def ratio_record(pruning: Mapping[str, Mapping[str, Any]],
                 clumping: Mapping[str, float], rows: int) -> dict[str, Any]:
    """Pruning over clumping on the identical rows, against three clumping bills.

    `plink2_s` is plink2 alone, genotypes and all. `matched_s` adds the panel
    matching clumping cannot start without — the rows have to be named on the
    panel before they can be clumped. `end_to_end_s` adds what a Store run pays
    to build the shared set, which includes the cM interpolation and check that
    only pruning needs. The sort is charged to neither side (see `SORT_NOTE`).
    """
    pruning_s = sum(_median(pruning, name) for name in ("cm", "rule"))
    fast_pruning_s = sum(_median(pruning, name) for name in ("cm", "rule_fast"))
    matched_s = clumping["matching_s"] + clumping["ranked_s"] + clumping["plink2_s"]
    shared_set_s = clumping.get("shared_set_s",
                                clumping["matching_s"] + clumping["shared_cm_s"])
    bills = {"plink2_s": round(clumping["plink2_s"], 4), "matched_s": round(matched_s, 4),
             "end_to_end_s": round(shared_set_s + clumping["ranked_s"] + clumping["plink2_s"], 4)}
    return {
        "rows": rows,
        "pruning_s": round(pruning_s, 4),
        "fast_pruning_s": round(fast_pruning_s, 4),
        "clumping": bills,
        "shared_set_split_s": round(clumping["matching_s"] + clumping["shared_cm_s"], 4),
        "ratios": {_bill(name): round(pruning_s / total, 4) for name, total in bills.items()},
        "ratios_fast": {_bill(name): round(fast_pruning_s / total, 4)
                        for name, total in bills.items()},
        "sort_note": SORT_NOTE,
        "definition": "pruning (cM lookup, rule) over clumping on the shared set of the same "
                      "Analysis: plink2 alone, plink2 with the panel matching, and the whole "
                      "shared set",
    }


# ---------------------------------------------------------------- the whole run

def _stores(specs: Sequence[SliceSpec]) -> list[Path]:
    """Each Store once, in the order the slices name them."""
    return list(dict.fromkeys(spec.store for spec in specs))


def measure(*, store_root: Path, genetic_map_path: Path, panel_dir: Path, work_dir: Path,
            repeats: int, threads: int, d_cm: float = 0.5, w_bp: int = 1_000_000,
            r2: float = 0.01, specs: Sequence[SliceSpec] | None = None) -> dict[str, Any]:
    """The whole measurement: the cM arrays, then every slice's pruning and clumping."""
    started = time.monotonic()
    record = provenance()
    loaded: dict[str, Any] = {}
    map_timing = time_stages({"load_map": lambda: loaded.update(map=load_map(genetic_map_path))},
                             repeats=repeats)
    genetic_map = loaded["map"]
    chosen = list(specs) if specs is not None else slice_specs(store_root)
    work_dir.mkdir(parents=True, exist_ok=True)
    axes: dict[Path, CmAxis] = {}
    for store in _stores(chosen):
        axes[store] = build_cm_axis(store, genetic_map, repeats=repeats)
    panel = Panel(panel_dir)
    slices = []
    for spec in chosen:
        work = work_dir / spec.name
        work.mkdir(parents=True, exist_ok=True)
        slices.append(measure_slice(spec, genetic_map=genetic_map, panel=panel,
                                    axis=axes[spec.store], work_dir=work, repeats=repeats,
                                    threads=threads, d_cm=d_cm, w_bp=w_bp, r2=r2))
    return {
        "schema": 1,
        "provenance": record,
        "parameters": {"d_cm": d_cm, "w_bp": w_bp, "r2": r2, "clump_kb": clump.CLUMP_KB,
                       "repeats": repeats, "threads": threads,
                       "plink2_memory_mib": clump.PLINK2_MEMORY_MIB,
                       "panel": str(panel_dir), "genetic_map": str(genetic_map_path),
                       "store_root": str(store_root), "budget": BUDGET},
        "genetic_map_load": map_timing,
        "selection": {"largest_tiers": largest_tiers(store_root),
                      "slices": [{"name": spec.name, "store": str(spec.store),
                                  "analysis_id": spec.analysis_id,
                                  "threshold": spec.threshold, "why": spec.why}
                                 for spec in chosen]},
        "per_variant_cm": [axes[store].record for store in _stores(chosen)],
        "slices": slices,
        "run_seconds": round(time.monotonic() - started, 1),
    }
