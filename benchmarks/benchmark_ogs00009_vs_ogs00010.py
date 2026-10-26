#!/usr/bin/env python3
"""Benchmark the rebuilt OGS-00009 (format 0.2.0) against the old OGS-00010 (0.1.0).

The owner deletes the old Stores once this comparison says what the rebuild
changed (opengwas/opengwasdb-stores#312). The two Reference-Completed releases
differ in five independent ways, and a reader must be able to keep them apart:

* **membership** -- 2,514 Analyses against 2,024, none removed;
* **the completion panel** -- the UKB EUR LD panel against HGDP+1KGP hg38 EUR;
* **the hg19 lift** -- #288's strand-aware projection renames reverse-strand
  variants and re-orients reverse-strand palindromes;
* **the format** -- 0.2.0 sharded Zarr v3 against 0.1.0 Zarr v2;
* **the identity model** -- a single-id completion (ADR 0037) against a
  two-id child of the old OGS-00009.

So every section holds the others fixed where it can and says which it cannot:
timings run one selection on both Stores (the shared Analyses, ALIDs both axes
carry, one region); the observed-cell fidelity compares by ALID over the shared
Analyses; the MR and regional sections are where the panel shows, and they say
so. Build collisions (opengwas/opengwasdb#325) are read from each Store on its
own, because a collision both Stores share is invisible to any comparison.

Sections, in the order the run measures them (every stop condition first):

1. identity and provenance of each Store, read from its manifest and records;
2. the variant axes, exhaustively: which observed variants each Store alone
   carries, the chain block each lies in, and the orientation each shared
   palindrome must keep, decided from the chain alone. The run stops if a
   one-sided variant lies outside every reverse-strand block, or the
   package's lift fails to reproduce a rename, a collapse or a re-orientation;
3. observed-cell fidelity over a seeded sample of windows, plus every observed
   variant the lift could have touched and every build-collision row. Each old
   observed cell is identical, (a) re-oriented by #288, (b) present in only one
   Store under that spelling (renamed), (d) at a build collision, a palindrome
   the chain cannot orient, or (c) unexplained. Any (c) cell stops the run and
   writes the partial result beside the output;
4. build collisions (`_build_collisions`): every Observed-Only row with a
   blank `source_alid`, its hg19 sources reconstructed with that Store's own
   lift, and whose statistics it holds. The run stops if a row's statistics
   match neither source;
5. MR positive controls (`benchmark_ogs00010_completed.mr_condition`) on all,
   observed-only and imputed-only variants. The run stops if an observed-only
   estimate changes sign or significance;
6. regional profiles around each exposure's strongest imputed hit in each Store,
   and APOE for statin use;
7. per-array structure (`compare_store_releases.discover_arrays`), with each
   Dense plane's Blosc typesize set against its element size;
8. query time, by the generic harness `benchmark_store_comparison` run on both
   Stores with one shared selection (its own artifact, beside this one): the
   seven shapes, the observed-only and shared-Analysis twins, first call apart
   from the warm repeats, per-shape RSS, load average, HM3 extraction and the
   footprint, from which the size section's bytes per cell are computed.

Usage:
  pixi run -e dev python benchmarks/benchmark_ogs00009_vs_ogs00010.py --reps 7
  cd docs/benchmark-output && pixi run -e report quarto render \
      opengwasdb_ogs00009_vs_ogs00010.qmd
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numcodecs
import numpy as np
import pandas as pd
import zarr
from pyliftover import LiftOver

from benchmarks import _build_collisions, _query_shapes
from benchmarks._artifact import provenance, write_artifact
from benchmarks.benchmark_ogs00010_completed import (
    EXPOSURE,
    MR_CONDITIONS,
    MR_PAIRS,
    REGION,
    REGION_HALF_WINDOW,
    exposure_columns,
    mr_condition,
    strongest_imputed_hit,
    window_profile,
)
from benchmarks.benchmark_store_comparison import CACHE_NOTE, effective_reader_settings
from benchmarks.compare_store_releases import discover_arrays
from opengwasdb.build.liftover import lifted_rows
from opengwasdb.query import query_store

REPO = Path(__file__).resolve().parents[1]
STORES = Path("/data/opengwasdb/stores")
CHAIN = Path("/data/opengwasdb/reference/liftover/hg19ToHg38.over.chain.gz")
OUTPUT = REPO / "docs/benchmark-output/opengwasdb_ogs00009_vs_ogs00010.json"
QUERIES_OUTPUT = REPO / "docs/benchmark-output/opengwasdb_ogs00009_vs_ogs00010_queries.json"
# The PheWAS variant every earlier ukb-b report timed (APOE rs429358), read
# from the OGS-00010 artifact so this run times the same variant.
OLD_ARTIFACT = REPO / "docs/benchmark-output/opengwasdb_ogs00010_completed_benchmark.json"

LABELS = ("new", "old")
FIDELITY_SEED = 20261024
# 1,000 windows of 500 consecutive observed variants: 500,000 variants, about
# 5% of the observed axis, so a clean result bounds the unexplained-variant
# rate at 3 / 500,000 (the rule of three) on top of the exhaustive stratum.
FIDELITY_WINDOWS = 1000
FIDELITY_WINDOW_ROWS = 500
EXAMPLE_CAP = 25

_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")
# Fidelity categories of an old observed cell: identical; (a) `reoriented` by
# #288; (b) `renamed`, present in only one Store under that spelling; (d)
# `collision`, a build-collision row in either Store (opengwas/opengwasdb#325);
# `palindrome_ambiguous`, a palindrome the chain cannot orient; and (c)
# `unexplained`, which stops the run. The `*_new_only` categories count new
# observed cells that no old cell accounts for.
OLD_CELL_CATEGORIES = (
    "identical", "reoriented", "renamed", "collision", "palindrome_ambiguous", "unexplained",
)
#: What each completion panel is, keyed by the path its completion step was
#: given: the step record names the panel only by path, and the manifest not at
#: all (its `ld_panel_id` is a constant). A path not listed reads as itself.
PANEL_NAMES = {
    "/data/opengwasdb/reference/ld": "UKB EUR LD panel (ukb-hg38-eur-ld)",
    "/data/opengwasdb/reference/hgdp1kgp-hg38/panel": "HGDP+1kGP hg38 EUR panel",
}
# Two Analyses whose source GWAS-VCFs say whose statistics a collision row holds.
COLLISION_ANALYSES = ("ukb-b-17805", "ukb-b-19953")
COLLISION_ISSUE = "opengwas/opengwasdb#325"


@dataclass(frozen=True)
class StoreSpec:
    """One Store under comparison and the records that say how it was made."""

    label: str
    store: Path
    records: Path
    parent: Path
    parent_records: Path


def _specs(args: argparse.Namespace) -> dict[str, StoreSpec]:
    return {
        "new": StoreSpec(
            "new", args.new_store, args.new_records, args.new_parent, args.new_parent_records
        ),
        "old": StoreSpec(
            "old", args.old_store, args.old_records, args.old_parent, args.old_parent_records
        ),
    }


def loadavg() -> list[float]:
    """The 1-, 5- and 15-minute load averages; the host is shared."""
    with open("/proc/loadavg") as fh:
        return [float(x) for x in fh.read().split()[:3]]


def _json(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _argv_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _count(text: str) -> int:
    return int(text.replace(",", ""))


# --- identity -----------------------------------------------------------------


def panel_counts(stdout: str) -> dict[str, int]:
    """Blocks and variants from a completion step's `LD panel: ...` line.

    The manifest cannot say which panel completed a Store: `ld_panel_id` is a
    constant of the completion code, so both Stores record `eur-hg38-gpm`
    although they were completed against different panels. The step record's
    argv and stdout are the only provenance that distinguishes them.
    """
    line = next(row for row in stdout.splitlines() if row.startswith("LD panel: "))
    blocks, variants = line.removeprefix("LD panel: ").split(", ")
    return {"n_blocks": _count(blocks.split()[0]), "n_variants": _count(variants.split()[0])}


def _step(record: dict[str, Any]) -> dict[str, Any]:
    argv = record["argv"]
    return {
        "opengwasdb_rev": record["opengwasdb_rev"],
        "opengwasdb_version": record["opengwasdb_version"],
        "start_time": record["start_time"],
        "end_time": record["end_time"],
        "elapsed_seconds": record["elapsed_seconds"],
        "n_workers": int(_argv_value(argv, "--n-workers")),
        "argv": argv,
    }


def identity_section(spec: StoreSpec, analysis_ids: list[str]) -> dict[str, Any]:
    """Manifest fields, runtime revisions, panel and derived artifacts of one Store."""
    manifest = _json(spec.store / "manifest.json")
    prov = manifest["provenance"]
    complete = _json(spec.records / "complete.json")
    root = spec.store / "data.zarr"
    return {
        "store": str(spec.store),
        "parent_store": str(spec.parent),
        "manifest": {
            key: manifest[key]
            for key in (
                "store_id",
                "release_id",
                "format_version",
                "completion_state",
                "association_coverage",
                "created_at",
                "reference_assembly",
                "primary_layout",
            )
        },
        "encoding": manifest["encoding"],
        "builder": prov["builder"],
        "chain_file": prov["chain_file"],
        "source_release_id": prov["source_release_id"],
        "dense": prov["dense"],
        "completion": prov["completion"],
        "panel": {
            "name": PANEL_NAMES.get(_argv_value(complete["argv"], "--ld-panel"),
                                    _argv_value(complete["argv"], "--ld-panel")),
            "path": _argv_value(complete["argv"], "--ld-panel"),
            "ancestry": _argv_value(complete["argv"], "--ancestry"),
            "manifest_ld_panel_id": prov["completion"]["ld_panel_id"],
            **panel_counts(complete["stdout"]),
        },
        "runtime": {
            "build": _step(_json(spec.parent_records / "build.json")),
            "complete": _step(complete),
        },
        "n_analyses": len(analysis_ids),
        "derived": {
            "hm3_subset": (root / "indexed_subsets" / "hm3").is_dir(),
            "top_hits": (root / "top_hits").is_dir(),
        },
    }


# --- the variant axes and the lift ---------------------------------------------


def load_axis(store: Path, expected_rows: int) -> pd.DataFrame:
    """A Store Variant Table's chromosome, position, row and ALID columns."""
    frame = pd.read_csv(
        store / "variants.tsv.gz",
        sep="\t",
        usecols=["#chromosome", "position", "variant_index", "alid"],
        dtype={"#chromosome": str, "alid": str},
    )
    frame.columns = ["chrom", "pos", "vi", "alid"]
    if len(frame) != expected_rows or not np.array_equal(
        frame["vi"].to_numpy(), np.arange(expected_rows)
    ):
        raise SystemExit(
            f"{store}: variants.tsv.gz holds {len(frame):,} rows, the manifest says "
            f"{expected_rows:,}, or its variant_index is not 0..n-1"
        )
    return frame


class ChainStrands:
    """Where the hg19 -> hg38 chain lands on hg38, by block orientation.

    #288 projects a lifted variant's alleles onto the strand of its chain block,
    so a variant's fate under the new lift is fixed by whether its hg38
    position lies in a reverse-strand block: there a non-palindromic variant is
    renamed (its alleles complemented) and a palindromic one keeps its ALID with
    its statistics re-oriented. A position can lie under blocks of both
    orientations (a region the chain maps twice); which one a source variant
    used then depends on its hg19 position, which the Store does not keep.
    """

    def __init__(self, path: Path) -> None:
        rows: list[tuple[str, int, int, str, str, int, int, int]] = []
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as fh:
            header: list[str] = []
            t_pos = q_pos = 0
            for line in fh:
                fields = line.split()
                if not fields:
                    continue
                if fields[0] == "chain":
                    header, t_pos, q_pos = fields, int(fields[5]), int(fields[10])
                    continue
                size, q_size, strand = int(fields[0]), int(header[8]), header[9]
                start = q_pos if strand == "+" else q_size - q_pos - size
                rows.append(
                    (header[7].removeprefix("chr"), start, start + size, strand, header[2],
                     t_pos, q_pos, q_size)
                )
                if len(fields) == 3:
                    t_pos += size + int(fields[1])
                    q_pos += size + int(fields[2])
        self.blocks = pd.DataFrame(
            rows, columns=["chrom", "start", "end", "strand", "t_chrom", "t_pos", "q_pos", "q_size"]
        )
        self._union = {strand: self._merged(strand) for strand in "+-"}

    def _merged(self, strand: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        merged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for chrom, group in self.blocks[self.blocks.strand == strand].groupby("chrom"):
            spans: list[list[int]] = []
            for start, end in sorted(zip(group.start, group.end, strict=True)):
                if spans and start <= spans[-1][1]:
                    spans[-1][1] = max(spans[-1][1], end)
                else:
                    spans.append([start, end])
            array = np.asarray(spans, dtype=np.int64)
            merged[str(chrom)] = (array[:, 0], array[:, 1])
        return merged

    def covered(self, strand: str, chrom: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Whether each 1-based hg38 position lies under a block of `strand`."""
        out = np.zeros(len(pos), dtype=bool)
        zero_based = np.asarray(pos, dtype=np.int64) - 1
        chrom = np.asarray(chrom).astype(str)
        for name, (starts, ends) in self._union[strand].items():
            on = chrom == name
            if not on.any():
                continue
            p = zero_based[on]
            slot = np.searchsorted(starts, p, side="right") - 1
            hit = slot >= 0
            inside = np.zeros(len(p), dtype=bool)
            inside[hit] = p[hit] < ends[slot[hit]]
            out[on] = inside
        return out

    def sources(self, chrom: str, pos: int) -> list[tuple[str, int, str]]:
        """Every 1-based hg19 position a chain block maps onto `chrom:pos`, with its strand."""
        p = pos - 1
        hits = self.blocks[
            (self.blocks.chrom == chrom) & (self.blocks.start <= p) & (self.blocks.end > p)
        ]
        out = []
        for row in hits.itertuples():
            offset = p - row.start if row.strand == "+" else row.q_size - 1 - p - row.q_pos
            out.append((str(row.t_chrom).removeprefix("chr"), int(row.t_pos + offset) + 1,
                        str(row.strand)))
        return out

    def reverse_sources(self, chrom: str, pos: int) -> list[tuple[str, int]]:
        """The 1-based hg19 positions a reverse-strand block maps onto `chrom:pos`."""
        return [(c, p) for c, p, strand in self.sources(chrom, pos) if strand == "-"]


def complement_alid(alid: str) -> str:
    """The ALID of the same position with both alleles complemented."""
    chrom, pos, a1, a2 = alid.split(":")
    c1, c2 = sorted((a1.translate(_COMPLEMENT), a2.translate(_COMPLEMENT)))
    return f"{chrom}:{pos}:{c1}:{c2}"


def complement_swaps(alid: str) -> bool:
    """Whether complementing the alleles makes the other allele A1.

    The old lift stored a reverse-strand variant under its hg19-strand alleles
    with z signed against their A1; #288 complements them and signs against the
    complemented pair's A1. When the complement of the old A1 is not the new A1
    the statistics are negated and EAF becomes 1 - EAF (``statistics_swap``).
    """
    _, _, a1, a2 = alid.split(":")
    c1, c2 = a1.translate(_COMPLEMENT), a2.translate(_COMPLEMENT)
    return c1 != min(c1, c2)


def is_palindromic(alid: str) -> bool:
    _, _, a1, a2 = alid.split(":")
    return len(a1) == len(a2) == 1 and a1.translate(_COMPLEMENT) == a2


@dataclass
class LiftMap:
    """What the lift did to every observed variant, from the two observed axes.

    `old_axis`/`new_axis` are the observed-variant ALIDs of each Store -- its
    Observed-Only parent's axis, which the completed axis contains in order
    (the `check-observed-view` verdicts). `collided_*` are each Store's build
    collisions (`_build_collisions`). `expected` holds the orientation a
    shared variant should keep where it is not plainly `same`: `flip`, or
    `ambiguous` for a palindrome the chain cannot orient.
    """

    old_axis: set[str]
    new_axis: set[str]
    collided_old: set[str] = field(default_factory=set)
    collided_new: set[str] = field(default_factory=set)
    renamed: dict[str, str] = field(default_factory=dict)
    collapsed: dict[str, str] = field(default_factory=dict)
    dropped: set[str] = field(default_factory=set)
    unpaired_new: set[str] = field(default_factory=set)
    expected: dict[str, str] = field(default_factory=dict)

    @property
    def collapse_targets(self) -> dict[str, str]:
        return {target: source for source, target in self.collapsed.items()}


def _pair_one_sided(lift: LiftMap, old_only: set[str], new_only: set[str]) -> None:
    for alid in old_only:
        partner = complement_alid(alid)
        if partner in new_only:
            lift.renamed[alid] = partner
        elif partner in lift.new_axis:
            lift.collapsed[alid] = partner
        else:
            lift.dropped.add(alid)
    lift.unpaired_new = new_only - set(lift.renamed.values())


def chain_orientation(engine: Any, source_alid: str | None, alid: str) -> str:
    """How the chain lifts a palindrome's hg19 source onto it: `flip`, `same` or `ambiguous`.

    Decided from the chain alone, never from the cells: a palindrome's two
    orientations can both fit its data, so choosing by fit would file a real
    strand error as a re-orientation. `ambiguous` when there is no source,
    when the source maps under blocks of both orientations, or when its first
    mapping -- the one the builders take -- does not land on the ALID.
    """
    if not source_alid or source_alid == ".":
        return "ambiguous"
    chrom, pos = source_alid.split(":")[:2]
    mapped = engine.convert_coordinate(f"chr{chrom}", int(pos) - 1) or []
    strands = {m[2] for m in mapped}
    target_chrom, target_pos = alid.split(":")[:2]
    lands = bool(mapped) and (mapped[0][0], int(mapped[0][1]) + 1) == (
        f"chr{target_chrom}", int(target_pos))
    if len(strands) != 1 or not lands:
        return "ambiguous"
    return "flip" if strands == {"-"} else "same"


def _expected_orientations(
    lift: LiftMap, shared: pd.DataFrame, chain: ChainStrands,
    source_of: Callable[[str], str | None], engine: Any,
) -> dict[str, int]:
    """Orientation each shared observed variant should keep, from its chain blocks.

    A palindrome only under reverse blocks re-orients (`flip`). One under
    blocks of both orientations is decided by lifting its hg19 `source_alid`
    (from the old parent) through the chain (`chain_orientation`); a
    build-collision row needs no orientation, because its cells are (d).
    """
    chrom, pos = shared["chrom"].to_numpy(), shared["pos"].to_numpy()
    minus = chain.covered("-", chrom, pos)
    plus = chain.covered("+", chrom, pos)
    counts: Counter[str] = Counter()
    for alid, also_plus in zip(shared["alid"].to_numpy()[minus], plus[minus], strict=True):
        if is_palindromic(alid) and not also_plus:
            kind = "flip"
        elif is_palindromic(alid) and (alid in lift.collided_old or alid in lift.collided_new):
            kind = "either_collision"
        elif is_palindromic(alid):
            kind = f"either_{chain_orientation(engine, source_of(alid), alid)}"
        else:
            kind = "same" if also_plus else "unrenamed_reverse"
        counts[kind] += 1
        expected = {"flip": "flip", "either_flip": "flip", "either_ambiguous": "ambiguous"}
        if kind in expected:
            lift.expected[alid] = expected[kind]
    return {
        "shared_in_reverse_block": int(minus.sum()),
        "shared_in_both_orientations": int((minus & plus).sum()),
        **{f"expected_{kind}": n for kind, n in sorted(counts.items())},
        "expected_reoriented_total": counts["flip"] + counts["either_flip"],
    }


def verify_with_lift(
    pairs: list[tuple[str, str]], chain: ChainStrands, chain_path: Path
) -> set[str]:
    """New-spelling ALIDs the package's own lift reproduces from the old spelling.

    Each old spelling is read back to hg19 through the reverse-strand blocks
    over its position and lifted again with `lifted_rows`, the lift every
    builder uses since #288; a pair verifies when that lift lands on the same
    position, on the reverse strand, under the new spelling. A variant the
    chain maps twice could lift through its other block and stay unverified;
    the run stops on any shortfall against the renames, collapses and
    re-orientations it expects (`gate_failures`).
    """
    wanted: list[tuple[tuple[str, int, str, str], str]] = []
    for old_alid, new_alid in pairs:
        chrom, pos, a1, a2 = old_alid.split(":")
        for t_chrom, t_pos in chain.reverse_sources(chrom, int(pos)):
            wanted.append(((t_chrom, t_pos, a1, a2), new_alid))
    rows, _, _ = lifted_rows((key for key, _ in wanted), chain_file=chain_path)
    verified: set[str] = set()
    for key, new_alid in wanted:
        row = rows.get(key)
        if row is not None and row.strand == "-" and row.alid == new_alid:
            verified.add(new_alid)
    return verified


def lift_section(
    old_parent: pd.DataFrame, new_parent: pd.DataFrame, chain: ChainStrands, chain_path: Path,
    collided: dict[str, set[str]], source_of: Callable[[str], str | None],
) -> tuple[LiftMap, dict[str, Any], pd.DataFrame]:
    """Exhaustive: every observed variant only one Store carries, and why."""
    lift = LiftMap(set(old_parent["alid"]), set(new_parent["alid"]),
                   collided_old=collided["old"], collided_new=collided["new"])
    old_only, new_only = lift.old_axis - lift.new_axis, lift.new_axis - lift.old_axis
    _pair_one_sided(lift, old_only, new_only)
    shared = old_parent[old_parent["alid"].isin(lift.new_axis)]
    orientation = _expected_orientations(lift, shared, chain, source_of,
                                         LiftOver(str(chain_path)))

    one_sided = sorted(old_only | new_only)
    one_chrom = np.array([a.split(":")[0] for a in one_sided])
    one_pos = np.array([int(a.split(":")[1]) for a in one_sided])
    in_reverse = dict(zip(one_sided, chain.covered("-", one_chrom, one_pos), strict=True))
    flips = [a for a, kind in lift.expected.items() if kind == "flip"]
    verified = verify_with_lift(
        [*lift.renamed.items(), *lift.collapsed.items(), *((a, a) for a in flips)],
        chain,
        chain_path,
    )
    summary = {
        "old_observed_variants": len(lift.old_axis),
        "new_observed_variants": len(lift.new_axis),
        "shared_observed_variants": len(shared),
        "old_only": len(old_only),
        "new_only": len(new_only),
        "renamed_pairs": len(lift.renamed),
        # The cross-Store view of #288's collisions: an old spelling whose
        # complement is an ALID the new Store already holds. Each target is a
        # row of `build_collisions`, which has the sources and statistics.
        "collapsed": [
            {"old_alid": source, "kept_alid": target,
             "source_hg19": chain.reverse_sources(source.split(":")[0], int(source.split(":")[1])),
             "see": f"build_collisions.stores.new.rows[{target}]"}
            for source, target in sorted(lift.collapsed.items())
        ],
        "dropped_old": sorted(lift.dropped),
        "unpaired_new": sorted(lift.unpaired_new),
        "one_sided_in_reverse_block": int(sum(in_reverse.values())),
        "one_sided_outside_reverse_block": sorted(a for a, v in in_reverse.items() if not v),
        "renamed_swapping_statistics": sum(complement_swaps(a) for a in lift.renamed),
        "lift_verified_renamed": sum(new in verified for new in lift.renamed.values()),
        "lift_verified_collapsed": sum(t in verified for t in lift.collapsed.values()),
        "lift_verified_flip": sum(a in verified for a in flips),
        **orientation,
    }
    return lift, summary, shared


# --- observed-cell fidelity ----------------------------------------------------


@dataclass(frozen=True)
class Tolerance:
    """How far two encodings of one source value may decode apart.

    z is `int16_fixed` (exact, so compared bit for bit). SE and EAF are
    `int8_residual`: each Store decodes within half a quantisation step of the
    source value (log SE; logit EAF) against its own baseline, and the two
    baselines differ with membership, so two Stores may differ by the sum of
    their half-steps. The 1e-6 covers float32 rounding of a decoded SE.

    A decoded frequency is rounded to float32 once by the decoder, and once
    more where the lift re-oriented it (``1 - eaf``); near 0 or 1 that moves its
    logit by up to ``2^-24 / min(f, 1 - f)`` per rounding, which at a frequency
    of 0.0015 is 4e-5 -- more than the 1e-6. `agreement` therefore adds
    `EAF_ROUNDING` ``/ min(f, 1 - f)`` for each of the two frequencies compared.
    A real difference is orders of magnitude larger: a wrong orientation moves
    the logit by ``2 |logit(f)|``.
    """

    se_log: float
    eaf_logit: float


def tolerance(encodings: list[dict[str, Any]]) -> Tolerance:
    def half_steps(plane: str) -> float:
        total = 0.0
        for encoding in encodings:
            spec = encoding[plane]
            if spec["kind"] != "int8_residual":
                raise SystemExit(f"{plane} is {spec['kind']}; the tolerance assumes int8_residual")
            total += float(spec["residual_range"]) / 127.0 / 2.0
        return total + 1e-6

    return Tolerance(se_log=half_steps("se"), eaf_logit=half_steps("eaf"))


#: Two float32 roundings of a frequency, in units of its distance from 0 or 1
#: (see `Tolerance`).
EAF_ROUNDING = 2.0 * 2.0**-24


def _logit(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(np.log(v / (1.0 - v)))


def agreement(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray], *, flipped: bool, tol: Tolerance
) -> np.ndarray:
    """Whether each cell of `b` reproduces `a`, re-oriented when `flipped`.

    Re-orientation negates z exactly and takes EAF to 1 - EAF; SE is
    orientation-free. A frequency of exactly 0 or 1 is an exact exception in
    both encodings and is compared exactly; NaN EAF agrees only with NaN.
    """
    sign = -1.0 if flipped else 1.0
    z_ok = b["z"] == sign * a["z"]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_gap = np.abs(np.log(b["se"].astype(np.float64)) - np.log(a["se"].astype(np.float64)))
    se_ok = (b["se"] == a["se"]) | (log_gap <= tol.se_log)
    ea, eb = a["eaf"], b["eaf"]
    target = (1.0 - ea) if flipped else ea
    inside = (ea > 0) & (ea < 1) & (eb > 0) & (eb < 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        logit_gap = np.abs(_logit(eb) - sign * _logit(ea))
        rounding = EAF_ROUNDING * (1.0 / np.minimum(ea, 1.0 - ea).astype(np.float64)
                                   + 1.0 / np.minimum(eb, 1.0 - eb).astype(np.float64))
    both_nan = np.isnan(ea) & np.isnan(eb)
    eaf_ok = both_nan | (eb == target) | (inside & (logit_gap <= tol.eaf_logit + rounding))
    return np.asarray(z_ok & se_ok & eaf_ok)


@dataclass
class Cells:
    """One Store's decoded cells over a set of rows, the shared Analyses in one order."""

    alid: np.ndarray
    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray
    observed: np.ndarray

    def take(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        return {"z": self.z[rows], "se": self.se[rows], "eaf": self.eaf[rows]}


def read_cells(q: Any, rows: np.ndarray, cols: np.ndarray) -> Cells:
    """Decode z, SE, EAF and the imputed mask for `rows` x every Analysis.

    A contiguous run reads as a band; a scattered set by orthogonal index. A
    cell is observed exactly as the query facade labels it: finite z and SE,
    imputed flag 0.
    """
    rows = np.asarray(rows, dtype=np.int64)
    z = q._z.rows(rows)
    eaf = q._eaf.read_rows(rows, want_imputed=True)
    if eaf.imputed is None:
        raise SystemExit("a Reference-Completed Store returned no imputed mask")
    se = q._se.rows(rows, eaf=eaf.values)
    observed = np.isfinite(z) & np.isfinite(se) & (eaf.imputed == 0)
    ident = q._variant_axis.identity_by_indices(rows)
    if ident is None:
        raise SystemExit("no mmap'd ALID index; refusing per-row identity seeks")
    return Cells(
        ident["alid"].astype(str), z[:, cols], se[:, cols], np.asarray(eaf.values)[:, cols],
        observed[:, cols],
    )


@dataclass
class Tally:
    """Observed cells and variants by category, for one stratum.

    Every observed cell of the old Store gets exactly one of
    `OLD_CELL_CATEGORIES` -- the comparison checks that after every read -- and
    a new observed cell no old one accounts for is `collision_new_only` at a
    build-collision row and `unexplained_new_only` anywhere else.
    """

    cells: Counter[str] = field(default_factory=Counter)
    variants: Counter[str] = field(default_factory=Counter)
    bit_identical: Counter[str] = field(default_factory=Counter)
    max_gap: dict[str, float] = field(default_factory=lambda: {"se_log": 0.0, "eaf_logit": 0.0})
    examples: list[dict[str, Any]] = field(default_factory=list)
    collision_agreeing: int = 0
    reads: int = 0
    old_rows: int = 0
    new_rows: int = 0
    old_observed_variants: int = 0
    old_observed_cells: int = 0
    new_observed_cells: int = 0

    def record(self) -> dict[str, Any]:
        return {
            "reads": self.reads,
            "old_rows_read": self.old_rows,
            "new_rows_read": self.new_rows,
            "old_observed_variants": self.old_observed_variants,
            "old_observed_cells": self.old_observed_cells,
            "new_observed_cells": self.new_observed_cells,
            "cells": dict(sorted(self.cells.items())),
            "variants": dict(sorted(self.variants.items())),
            "summary": {
                "identical": self.cells["identical"],
                "a_reoriented": self.cells["reoriented"],
                "b_one_store_renamed": self.cells["renamed"],
                "d_build_collision": self.cells["collision"] + self.cells["collision_new_only"],
                "palindrome_strand_ambiguous": self.cells["palindrome_ambiguous"],
                "c_unexplained": self.cells["unexplained"] + self.cells["unexplained_new_only"],
            },
            # Of the (d) old cells, those the other Store reproduces in the
            # orientation the lift implies: informative, not a verdict.
            "d_build_collision_agreeing": self.collision_agreeing,
            "bit_identical_among_identical": dict(sorted(self.bit_identical.items())),
            "max_gap": self.max_gap,
            "unexplained_examples": self.examples,
        }


def _example(
    reason: str, analysis_id: str, old: tuple[str, dict[str, Any]] | None,
    new: tuple[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    def side(entry: tuple[str, dict[str, Any]] | None) -> dict[str, Any] | None:
        if entry is None:
            return None
        alid, values = entry
        return {"alid": alid, **{name: float(values[name]) for name in ("z", "se", "eaf")}}

    return {"reason": reason, "analysis_id": analysis_id, "old": side(old), "new": side(new)}


def _record_gaps(tally: Tally, a: dict[str, np.ndarray], b: dict[str, np.ndarray],
                 good: np.ndarray, flipped: bool) -> None:
    """The widest SE and EAF gaps among cells that agreed, and bit-identity counts."""
    if not good.any():
        return
    sign = -1.0 if flipped else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        se_gap = np.abs(np.log(b["se"][good].astype(np.float64) / a["se"][good]))
    eaf_gap = np.abs(_logit(b["eaf"][good]) - sign * _logit(a["eaf"][good]))
    for key, gap in (("se_log", se_gap), ("eaf_logit", eaf_gap)):
        finite = gap[np.isfinite(gap)]
        if finite.size:
            tally.max_gap[key] = max(tally.max_gap[key], float(finite.max()))
    if not flipped:
        tally.bit_identical["z"] += int((good & (a["z"] == b["z"])).sum())
        tally.bit_identical["se"] += int((good & (a["se"] == b["se"])).sum())
        same_eaf = (a["eaf"] == b["eaf"]) | (np.isnan(a["eaf"]) & np.isnan(b["eaf"]))
        tally.bit_identical["eaf"] += int((good & same_eaf).sum())


def _note_bad(tally: Tally, old: Cells, new: Cells, i: np.ndarray, j: np.ndarray,
              bad: np.ndarray, match: str, analyses: list[str]) -> None:
    a, b = old.take(i), new.take(j)
    for r, k in np.argwhere(bad)[: max(0, EXAMPLE_CAP - len(tally.examples))]:
        reason = f"{match}: " + ("mismatch" if new.observed[j[r], k] else "observed only in old")
        tally.examples.append(_example(
            reason, analyses[k],
            (str(old.alid[i[r]]), {n: a[n][r, k] for n in a}),
            (str(new.alid[j[r]]), {n: b[n][r, k] for n in b}),
        ))


def score_rows(
    tally: Tally, old: Cells, new: Cells, rows_old: list[int], rows_new: list[int], *,
    flipped: bool, match: str, analyses: list[str], tol: Tolerance,
) -> np.ndarray:
    """Classify old rows' observed cells against their new counterparts, in one batch.

    An old observed cell is `match` when the new cell is observed and agrees
    (re-oriented when `flipped`), and `unexplained` otherwise. Returns the new
    cells the batch accounted for -- those observed on both sides -- so a new
    cell left over is known to have no old counterpart.
    """
    i, j = np.asarray(rows_old, dtype=np.int64), np.asarray(rows_new, dtype=np.int64)
    if len(set(j.tolist())) != len(j):
        raise SystemExit(f"{match}: two old rows paired with one new row in one batch")
    a, b = old.take(i), new.take(j)
    obs_o, obs_n = old.observed[i], new.observed[j]
    good = obs_o & obs_n & agreement(a, b, flipped=flipped, tol=tol)
    bad = obs_o & ~good
    tally.cells[match] += int(good.sum())
    tally.cells["unexplained"] += int(bad.sum())
    bad_rows = bad.any(axis=1)
    tally.variants[match] += int((~bad_rows).sum())
    tally.variants["unexplained"] += int(bad_rows.sum())
    _record_gaps(tally, a, b, good, flipped)
    _note_bad(tally, old, new, i, j, bad, match, analyses)
    return np.asarray(obs_o & obs_n)


def _score_collision(tally: Tally, old: Cells, new: Cells, i: int, j: int, flipped: bool,
                     tol: Tolerance) -> np.ndarray:
    """(d): every observed cell of an old row that is, or lands on, a build collision.

    A collision row holds one of two sources' statistics, so a difference there
    is the collision, not an unexplained one. Whether the other Store reproduces
    the cell is still counted (`collision_agreeing`), and every new cell of the
    row is accounted for.
    """
    obs_o, obs_n = old.observed[i], new.observed[j]
    ok = agreement(old.take(np.array([i])), new.take(np.array([j])), flipped=flipped, tol=tol)[0]
    tally.cells["collision"] += int(obs_o.sum())
    tally.variants["collision"] += 1
    tally.collision_agreeing += int((obs_o & obs_n & ok).sum())
    return np.asarray(obs_n)


def _score_ambiguous(tally: Tally, old: Cells, new: Cells, i: int, j: int, tol: Tolerance,
                     analyses: list[str]) -> np.ndarray:
    """A palindrome the chain cannot orient: counted apart when either orientation holds."""
    a, b = old.take(np.array([i])), new.take(np.array([j]))
    either = agreement(a, b, flipped=False, tol=tol)[0] | agreement(a, b, flipped=True, tol=tol)[0]
    obs_o, obs_n = old.observed[i], new.observed[j]
    good = obs_o & obs_n & either
    bad = obs_o & ~good
    tally.cells["palindrome_ambiguous"] += int(good.sum())
    tally.cells["unexplained"] += int(bad.sum())
    tally.variants["unexplained" if bad.any() else "palindrome_ambiguous"] += 1
    _note_bad(tally, old, new, np.array([i]), np.array([j]), bad[None, :],
              "palindrome_ambiguous", analyses)
    return np.asarray(obs_o & obs_n)


def _on_axis(cells: Cells, axis: set[str], label: str) -> np.ndarray:
    """Which rows of a read are observed-axis variants; a panel row must hold none."""
    on = np.array([alid in axis for alid in cells.alid], dtype=bool)
    stray = ~on & cells.observed.any(axis=1)
    if stray.any():
        k = int(np.flatnonzero(stray)[0])
        raise SystemExit(f"{label} Store: {cells.alid[k]} is a panel-only row yet holds "
                         f"{int(cells.observed[k].sum())} observed cells")
    return on


@dataclass
class _Plan:
    """Old row -> new row pairings of one read, grouped by how they are scored."""

    batches: dict[tuple[bool, str], tuple[list[int], list[int]]] = field(default_factory=dict)
    collisions: list[tuple[int, int, bool]] = field(default_factory=list)
    ambiguous: list[tuple[int, int]] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)

    def add(self, flipped: bool, match: str, i: int, j: int) -> None:
        rows = self.batches.setdefault((flipped, match), ([], []))
        rows[0].append(i)
        rows[1].append(j)


def _counterpart(alid: str, lift: LiftMap) -> tuple[str | None, bool]:
    """The new ALID that should hold an old row's cells, and whether they re-orient."""
    if alid in lift.renamed:
        return lift.renamed[alid], complement_swaps(alid)
    if alid in lift.collapsed:
        return lift.collapsed[alid], complement_swaps(alid)
    if alid in lift.dropped:
        return None, False
    return alid, lift.expected.get(alid) == "flip"


def _plan_read(old: Cells, new_row: dict[str, int], lift: LiftMap, on_old: np.ndarray) -> _Plan:
    plan = _Plan()
    for i in np.flatnonzero(on_old):
        alid = str(old.alid[i])
        partner, flipped = _counterpart(alid, lift)
        if partner is None:
            plan.dropped.append(int(i))
            continue
        if partner not in new_row:
            raise SystemExit(f"{alid}: its counterpart {partner} is not in the same read")
        j = new_row[partner]
        if alid in lift.collided_old or partner in lift.collided_new:
            plan.collisions.append((int(i), j, flipped))
        elif alid in lift.collapsed:
            # Two source rows on one ALID are a build collision by definition;
            # one its parent does not record would otherwise score as (a).
            raise SystemExit(f"{alid} collapses onto {partner}, which the new Store's parent "
                             "does not record as a build collision")
        elif alid in lift.renamed:
            plan.add(flipped, "renamed", int(i), j)
        elif lift.expected.get(alid) == "ambiguous":
            plan.ambiguous.append((int(i), j))
        else:
            plan.add(flipped, "reoriented" if flipped else "identical", int(i), j)
    return plan


def compare_reads(
    old: Cells, new: Cells, lift: LiftMap, tol: Tolerance, analyses: list[str], tally: Tally
) -> None:
    """Classify every observed cell of one read on both Stores (see `Tally`).

    Each old observed row is paired with the new row that should hold its
    cells -- itself, its renamed spelling, or the ALID it collapsed onto -- in
    the orientation the lift implies. A row that is a build collision in either
    Store is (d), whatever its cells hold. An old row with no counterpart and no
    lift evidence for its absence is (c).
    """
    on_old, on_new = _on_axis(old, lift.old_axis, "old"), _on_axis(new, lift.new_axis, "new")
    new_row = {alid: j for j, alid in enumerate(new.alid)}
    before = sum(tally.cells[c] for c in OLD_CELL_CATEGORIES)
    tally.reads += 1
    tally.old_rows += len(old.alid)
    tally.new_rows += len(new.alid)
    old_cells = int(old.observed[on_old].sum())
    tally.old_observed_variants += int(on_old.sum())
    tally.old_observed_cells += old_cells
    tally.new_observed_cells += int(new.observed[on_new].sum())

    plan = _plan_read(old, new_row, lift, on_old)
    used = np.zeros_like(new.observed)
    for (flipped, match), (rows_old, rows_new) in plan.batches.items():
        used[rows_new] |= score_rows(tally, old, new, rows_old, rows_new, flipped=flipped,
                                     match=match, analyses=analyses, tol=tol)
    for i, j, flipped in plan.collisions:
        used[j] |= _score_collision(tally, old, new, i, j, flipped, tol)
    for i, j in plan.ambiguous:
        used[j] |= _score_ambiguous(tally, old, new, i, j, tol, analyses)
    for i in plan.dropped:
        tally.cells["unexplained"] += int(old.observed[i].sum())
        tally.variants["unexplained"] += 1
        if len(tally.examples) < EXAMPLE_CAP:
            tally.examples.append({"reason": "present only in old, no lift evidence for its "
                                   "absence", "alid": str(old.alid[i])})

    collided = np.array([alid in lift.collided_new for alid in new.alid], dtype=bool)
    extra = new.observed & ~used & on_new[:, None]
    tally.cells["collision_new_only"] += int((extra & collided[:, None]).sum())
    extra &= ~collided[:, None]
    tally.cells["unexplained_new_only"] += int(extra.sum())
    for j, k in np.argwhere(extra)[: max(0, EXAMPLE_CAP - len(tally.examples))]:
        values = new.take(np.array([j]))
        tally.examples.append(_example("observed only in new", analyses[k], None,
                                       (str(new.alid[j]), {n: values[n][0, k] for n in values})))
    tally.variants["unexplained_new_only"] += len(set(new.alid) & lift.unpaired_new)
    after = sum(tally.cells[c] for c in OLD_CELL_CATEGORIES)
    if after - before != old_cells:
        raise SystemExit(f"fidelity bookkeeping: {after - before:,} old cells classified, "
                         f"{old_cells:,} observed")


def fidelity_windows(
    old_parent: pd.DataFrame, n_windows: int, window_rows: int, seed: int
) -> list[tuple[str, int, int]]:
    """Seeded, non-overlapping windows of consecutive observed variants.

    Slots of `window_rows` old observed rows are drawn without replacement; a
    slot spanning two chromosomes becomes one window per chromosome. Each
    window is read on both Stores by position, so it carries every variant
    either Store holds there, renamed spellings included.
    """
    n_slots = len(old_parent) // window_rows
    picks = np.sort(np.random.default_rng(seed).choice(n_slots, size=n_windows, replace=False))
    windows: list[tuple[str, int, int]] = []
    for slot in picks:
        part = old_parent.iloc[slot * window_rows:(slot + 1) * window_rows]
        for chrom, group in part.groupby("chrom", sort=False):
            windows.append((str(chrom), int(group["pos"].min()), int(group["pos"].max())))
    return windows


def _stratum_rows(
    lift: LiftMap, shared: pd.DataFrame, chain: ChainStrands, axes: dict[str, pd.DataFrame]
) -> dict[str, np.ndarray]:
    """Every observed variant the lift could have touched, and every collision row."""
    minus = chain.covered("-", shared["chrom"].to_numpy(), shared["pos"].to_numpy())
    touched = set(shared["alid"].to_numpy()[minus])
    old_only = set(lift.renamed) | set(lift.collapsed) | lift.dropped
    collapse_partners = set(lift.collapsed.values())
    # Every build-collision row of either Store, with the row that should hold
    # its cells in the other, so a collision both Stores share is read too.
    renamed_from = {new: old for old, new in lift.renamed.items()}
    collided_old = lift.collided_old | {renamed_from.get(a, a) for a in lift.collided_new}
    collided_new = lift.collided_new | {lift.renamed.get(a, a) for a in lift.collided_old}
    wanted = {
        "old": touched | old_only | collapse_partners | (collided_old & lift.old_axis),
        "new": touched | set(lift.renamed.values()) | lift.unpaired_new | collapse_partners
        | (collided_new & lift.new_axis),
    }
    rows = {}
    for label, alids in wanted.items():
        axis = axes[label]
        rows[label] = np.sort(axis.loc[axis["alid"].isin(alids), "vi"].to_numpy())
    return rows


def fidelity_section(
    queries: dict[str, Any], cols: dict[str, np.ndarray], lift: LiftMap, shared: pd.DataFrame,
    chain: ChainStrands, axes: dict[str, pd.DataFrame], old_parent: pd.DataFrame,
    analyses: list[str], tol: Tolerance, args: argparse.Namespace,
) -> dict[str, Any]:
    """The observed-cell fidelity check over two strata (see the module docstring)."""
    started = time.perf_counter()
    targeted = Tally()
    rows = _stratum_rows(lift, shared, chain, axes)
    compare_reads(
        read_cells(queries["old"], rows["old"], cols["old"]),
        read_cells(queries["new"], rows["new"], cols["new"]),
        lift, tol, analyses, targeted,
    )
    print(f"fidelity targeted stratum: {targeted.record()['summary']}", flush=True)
    sampled = Tally()
    windows = fidelity_windows(old_parent, args.fidelity_windows, args.fidelity_window_rows,
                               args.fidelity_seed)
    for n, (chrom, start, end) in enumerate(windows):
        read = {
            label: read_cells(
                queries[label],
                queries[label]._variant_axis.range_indices(chrom, start, end),
                cols[label],
            )
            for label in LABELS
        }
        compare_reads(read["old"], read["new"], lift, tol, analyses, sampled)
        if n % 100 == 0:
            print(f"  window {n:,}/{len(windows):,}: {sampled.record()['summary']}", flush=True)
    print(f"fidelity random windows: {sampled.record()['summary']}", flush=True)
    return {
        "design": {
            "shared_analyses": len(analyses),
            "seed": args.fidelity_seed,
            "n_windows_drawn": args.fidelity_windows,
            "window_old_observed_rows": args.fidelity_window_rows,
            "n_windows_read": len(windows),
            "old_observed_axis": len(old_parent),
            "tolerance": {"se_log": tol.se_log, "eaf_logit": tol.eaf_logit,
                          "eaf_rounding_per_value": "2 * 2^-24 / min(f, 1 - f)"},
        },
        "strata": {"targeted": targeted.record(), "random_windows": sampled.record()},
        "seconds": round(time.perf_counter() - started, 1),
        "loadavg_after": loadavg(),
    }


# --- MR and regions --------------------------------------------------------------


def mr_section(
    queries: dict[str, Any], analyses_by_id: dict[str, dict[str, int]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """The three MR pairs on both Stores, each under all three conditions."""
    hits: dict[str, dict[str, Any]] = {label: {} for label in LABELS}
    out = []
    for pair in MR_PAIRS:
        stores = {}
        for label in LABELS:
            if pair["exposure_id"] not in hits[label]:
                hits[label][pair["exposure_id"]] = exposure_columns(
                    queries[label], pair["exposure_id"]
                )
            cols = hits[label][pair["exposure_id"]]
            stores[label] = {
                cond: mr_condition(queries[label], analyses_by_id[label], pair, cond, cols)
                for cond in MR_CONDITIONS
            }
            for cond, c in stores[label].items():
                print(f"MR {label} {pair['key']:13s} {cond:9s} n={c['n_instruments']:5d} "
                      f"beta={c['ivw_beta']:+.4f} se={c['ivw_se']:.4f} p={c['ivw_pval']:.2g}",
                      flush=True)
        out.append({**pair, "stores": stores})
    return out, hits


def observed_mr_agreement(mr: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The observed-only estimate must keep its sign and its significance at 0.05."""
    rows = []
    for pair in mr:
        new, old = pair["stores"]["new"]["observed"], pair["stores"]["old"]["observed"]
        rows.append({
            "key": pair["key"],
            "new_beta": new["ivw_beta"],
            "old_beta": old["ivw_beta"],
            "new_pval": new["ivw_pval"],
            "old_pval": old["ivw_pval"],
            "new_instruments": new["n_instruments"],
            "old_instruments": old["n_instruments"],
            "same_sign": bool(np.sign(new["ivw_beta"]) == np.sign(old["ivw_beta"])),
            "same_significance": bool((new["ivw_pval"] < 0.05) == (old["ivw_pval"] < 0.05)),
        })
    return rows


def _strongest_imputed(cols: dict[str, Any], analysis_id: str) -> tuple[str, int, str, float]:
    k = strongest_imputed_hit(cols, analysis_id)
    return str(cols["chrom"][k]), int(cols["pos"][k]), str(cols["alid"][k]), float(cols["z"][k])


def profile(q: Any, analysis_id: str, window: dict[str, Any], axis: set[str]) -> dict[str, Any]:
    """One Store's cells over a window, coded o/i/p/m per variant.

    `o` observed, `i` imputed at a variant of the observed axis, `p` imputed at
    a panel-only variant completion added, `m` still missing. Positions are
    offsets from the window start, z to four significant digits.
    """
    w = window_profile(q, analysis_id, window["chrom"], window["start"], window["end"])
    codes = []
    for alid, status in zip(w["alid"], w["status"], strict=True):
        observed_axis = str(alid) in axis
        if status == "observed" and not observed_axis:
            raise SystemExit(f"{alid}: observed cell at a panel-only variant")
        codes.append({"observed": "o", "missing": "m"}.get(status, "i" if observed_axis else "p"))
    text = "".join(codes)
    return {
        "n_points": len(text),
        "counts": {code: text.count(code) for code in "oipm"},
        "pos": [int(p) - window["start"] for p in w["position"]],
        "z": [float(f"{v:.4g}") for v in w["z"]],
        "status": text,
    }


def region_section(
    queries: dict[str, Any], hits: dict[str, dict[str, Any]], lift: LiftMap
) -> list[dict[str, Any]]:
    """Each exposure's strongest imputed hit in each Store, and APOE for statin use."""
    axes = {"new": lift.new_axis, "old": lift.old_axis}
    traits = {row["analysis_id"]: row["analysis_label"]
              for row in queries["new"].analyses_table().values()}
    windows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for pair in MR_PAIRS:
        exposure = pair["exposure_id"]
        for label in LABELS:
            chrom, pos, alid, z = _strongest_imputed(hits[label][exposure], exposure)
            start = max(1, pos - REGION_HALF_WINDOW)
            key = (exposure, chrom, start)
            windows.setdefault(key, {
                "exposure_id": exposure, "pair": pair["key"], "kind": "strongest_imputed_hit",
                "chrom": chrom, "start": start, "end": pos + REGION_HALF_WINDOW,
                "centre": {}, "defined_by": [],
            })
            windows[key]["centre"][label] = {"alid": alid, "pos": pos, "z": z}
            windows[key]["defined_by"].append(label)
    windows[(EXPOSURE, REGION[0], REGION[1])] = {
        "exposure_id": EXPOSURE, "pair": "ldl_chd", "kind": "apoe",
        "chrom": REGION[0], "start": REGION[1], "end": REGION[2], "centre": {}, "defined_by": [],
    }
    out = []
    for window in windows.values():
        window["trait"] = traits[window["exposure_id"]]
        window["stores"] = {
            label: profile(queries[label], window["exposure_id"], window, axes[label])
            for label in LABELS
        }
        out.append(window)
    return out


#: Inner chunks re-encoded per plane whose stored Blosc typesize is not its
#: element size -- enough for the ratio, seconds to read.
TYPESIZE_SAMPLE_CHUNKS = 200


def typesize_section(
    queries: dict[str, Any], structure: dict[str, dict[str, Any]], seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Each Dense plane's stored Blosc `typesize` against its element size.

    Zarr v3 stores `typesize` in the codec configuration and Blosc shuffles by
    it; zarr v2 took it from the buffer at encode time, so a 0.1.0 plane cannot
    get it wrong. A 0.2.0 plane whose codec was copied from a wider array keeps
    the wider typesize, and bit-shuffling an int8 plane in groups of four bytes
    interleaves four cells' bits instead of one's. Where they differ, a seeded
    sample of inner chunks is re-encoded both ways with the stored codec's
    parameters, so the report can say what it costs.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for label, q in queries.items():
        rows: list[dict[str, Any]] = []
        for plane in ("z", "se", "eaf", "imputed"):
            meta = structure[label].get(f"data.zarr/{plane}")
            if meta is None:
                continue
            itemsize = np.dtype(meta["dtype"]).itemsize
            stored = meta["compressor"].get("typesize")
            row: dict[str, Any] = {"plane": plane, "dtype": meta["dtype"], "itemsize": itemsize,
                                   "stored_typesize": stored, "zarr_format": meta["zarr_format"]}
            if stored is not None and int(stored) != itemsize:
                row["recompressed"] = _recompress_sample(
                    q._root[plane], meta, int(stored), np.random.default_rng(seed)
                )
            rows.append(row)
        out[label] = rows
    return out


#: Blosc shuffle names a v3 codec records, as numcodecs numbers them, and the
#: compressors Blosc knows. Anything else is refused rather than approximated.
_BLOSC_SHUFFLES = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
_BLOSC_CNAMES = ("blosclz", "lz4", "lz4hc", "zlib", "zstd")


def _blosc_of(config: dict[str, Any]) -> Any:
    """A numcodecs Blosc with the stored codec's parameters; anything else is refused."""
    from numcodecs import Blosc

    if "cname" not in config:
        raise SystemExit(f"typesize sample: no Blosc codec in {config}; refusing to guess one")
    if config["shuffle"] not in _BLOSC_SHUFFLES:
        raise SystemExit(f"typesize sample: unknown Blosc shuffle {config['shuffle']!r}")
    if config["cname"] not in _BLOSC_CNAMES:
        raise SystemExit(f"typesize sample: unknown Blosc compressor {config['cname']!r}")
    return Blosc(cname=config["cname"], clevel=int(config["clevel"]),
                 shuffle=_BLOSC_SHUFFLES[config["shuffle"]],
                 blocksize=int(config.get("blocksize", 0)))


def _recompress_sample(
    array: Any, meta: dict[str, Any], stored: int, rng: np.random.Generator
) -> dict[str, Any]:
    blosc = _blosc_of(meta["compressor"])
    rows, cols = meta["inner_chunk"]
    n_rows, n_cols = meta["shape"][0] // rows, meta["shape"][1] // cols
    picks = rng.choice(n_rows * n_cols, size=TYPESIZE_SAMPLE_CHUNKS, replace=False)
    as_stored = as_element = 0
    for pick in picks:
        r, c = divmod(int(pick), n_cols)
        chunk = np.ascontiguousarray(array[r * rows:(r + 1) * rows, c * cols:(c + 1) * cols])
        wide = np.dtype(f"V{stored}")
        as_stored += len(blosc.encode(chunk.view(wide)))
        as_element += len(blosc.encode(chunk))
    return {
        "chunks": TYPESIZE_SAMPLE_CHUNKS,
        "bytes_at_stored_typesize": as_stored,
        "bytes_at_element_size": as_element,
        "ratio": as_stored / as_element,
        "plane_bytes_on_disk": meta["nbytes_on_disk"],
        # An estimate: the plane's bytes scaled by the sample's ratio.
        "plane_bytes_at_element_size_estimate": round(
            meta["nbytes_on_disk"] * as_element / as_stored),
    }


# --- query time -------------------------------------------------------------------


def resolve_selection(axes: dict[str, pd.DataFrame], shared_analyses: list[str]) -> dict[str, Any]:
    """One selection for both Stores, in `benchmark_store_comparison`'s format.

    The random draws keep the historical `default_rng(0)` order (variants,
    then Analyses) but draw from the ALIDs both completed axes carry and from
    the shared Analyses, and the region's ALIDs are pinned to the ones both
    carry, so every lookup asks both Stores the same thing. It opts into the
    observed-only shapes and the shared-Analysis twins of `phewas`/`regional`.
    """
    new, old = axes["new"], axes["old"]
    both = new[new["alid"].isin(old["alid"])]
    rng = np.random.default_rng(0)
    random_alids = rng.choice(both["alid"].to_numpy(), _query_shapes.RANDOM_AXIS_SIZE,
                              replace=False)
    random_analyses = rng.choice(np.asarray(shared_analyses), _query_shapes.RANDOM_AXIS_SIZE,
                                 replace=False)
    chrom, start, end = REGION
    in_region = both[(both["chrom"] == chrom) & (both["pos"] >= start) & (both["pos"] <= end)]
    phewas_alid = _json(OLD_ARTIFACT)["selection"]["phewas_alid"]
    if not (both["alid"] == phewas_alid).any():
        raise SystemExit(f"PheWAS variant {phewas_alid} is not on both axes")
    return {
        "exposure_analysis_id": EXPOSURE,
        "phewas_alid": phewas_alid,
        "region": {"chrom": chrom, "start": start, "end": end},
        "random_lookup_shapes": _query_shapes.RANDOM_LOOKUP_SHAPES,
        "random_alids": [str(a) for a in random_alids],
        "random_analyses": [str(a) for a in random_analyses],
        "regional_alids": [str(a) for a in in_region["alid"]],
        "shared_analyses": shared_analyses,
        "observed_only_shapes": True,
        "resolved_from": {
            "label": "both",
            "rule": "ALIDs on both completed axes; the shared Analyses; default_rng(0)",
            "n_alids_on_both_axes": len(both),
            "n_shared_analyses": len(shared_analyses),
        },
    }


def query_time_section(
    args: argparse.Namespace, specs: dict[str, StoreSpec], selection: dict[str, Any]
) -> dict[str, Any]:
    """Time both Stores with the generic harness, `benchmark_store_comparison`.

    The harness owns the protocol every store comparison shares -- the shapes,
    warm-up and repeats, fresh-interpreter RSS, HM3 extraction, the footprint
    and the load gate -- so this report's timings are measured the way #240's
    and #246's were. It runs as its own process because its RSS probes
    re-invoke the script that is running. The two Stores hold different data,
    so their results are recorded as differing rather than refused.
    """
    started = time.perf_counter()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(selection, fh)
    command = [
        sys.executable, str(Path(__file__).with_name("benchmark_store_comparison.py")),
        "--store", f"new={specs['new'].store}", "--store", f"old={specs['old'].store}",
        "--selection", fh.name, "--allow-different-results", "--reps", str(args.reps),
        "--max-start-load", str(args.max_start_load), "--output", str(args.queries_output),
    ]
    if args.skip_rss:
        command.append("--skip-rss")
    selection_sha = _sha256(Path(fh.name))
    try:
        subprocess.run(command, check=True)
    finally:
        os.unlink(fh.name)
    return {
        "artifact": args.queries_output.name,
        "command": [part if part != fh.name else "<selection>" for part in command[1:]],
        # The selection the harness timed lives in its committed artifact
        # (`selection`); the temporary file that carried it there does not.
        "selection_source": {"artifact": args.queries_output.name,
                             "sha256": _sha256(args.queries_output),
                             "selection_sha256": selection_sha},
        "seconds": round(time.perf_counter() - started, 1),
    }


# --- size and structure -----------------------------------------------------------


def size_section(
    spec: StoreSpec, identity: dict[str, Any], n_variants: int, fp: dict[str, Any]
) -> dict[str, Any]:
    """The cell budget, bytes per cell and the variant axis, over the harness footprint."""
    comp = identity["completion"]
    cells = n_variants * identity["n_analyses"]
    imputed, failed, off = (int(comp[k]) for k in (
        "n_imputed", "n_missing_imputation_failed", "n_missing_off_panel"))
    observed = cells - imputed - failed - off
    observed_rows = int(_json(spec.store / "manifest.json")["provenance"]["n_variants"])
    panel_rows = int(comp["n_variants_new"])
    if observed_rows + panel_rows != n_variants or int(comp["n_variants_total"]) != n_variants:
        raise SystemExit(f"{spec.store}: observed {observed_rows:,} + panel {panel_rows:,} rows "
                         f"is not the axis {n_variants:,}")
    array_bytes = sum(int(a["apparent_bytes"]) for a in fp["arrays"])
    return {
        "apparent_bytes": fp["apparent_bytes"],
        "allocated_bytes": fp["allocated_bytes"],
        "array_apparent_bytes": array_bytes,
        "envelope_apparent_bytes": fp["envelope"]["apparent_bytes"],
        "n_files": fp["n_files"],
        "arrays": [{k: a[k] for k in ("node", "n_files", "apparent_bytes", "largest_file_bytes")}
                   for a in fp["arrays"]],
        "cells": {
            "total": cells, "observed": observed, "imputed": imputed,
            "missing_imputation_failed": failed, "missing_off_panel": off,
        },
        "bytes_per_cell": fp["apparent_bytes"] / cells,
        "array_bytes_per_cell": array_bytes / cells,
        "bytes_per_non_missing_cell": fp["apparent_bytes"] / (observed + imputed),
        "variant_axis": {"n_variants": n_variants, "observed_rows": observed_rows,
                         "panel_rows": panel_rows},
    }


# --- the run ------------------------------------------------------------------------


def _environment(queries: dict[str, Any]) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "zarr": zarr.__version__,
        "numcodecs": numcodecs.__version__,
        "hostname": socket.gethostname(),
        "nproc": os.cpu_count(),
        "blosc_nthreads": numcodecs.blosc.get_nthreads(),
        "thread_env": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS")},
        "effective_reader": {label: effective_reader_settings(q._root)
                             for label, q in queries.items()},
        "cache": CACHE_NOTE,
    }


def _stop(result: dict[str, Any], args: argparse.Namespace, message: str) -> None:
    partial = args.output.with_suffix(".partial.json")
    write_artifact(partial, result)
    raise SystemExit(f"STOP: {message} (partial result in {partial})")


def _membership(analyses: dict[str, list[str]]) -> dict[str, Any]:
    new, old = set(analyses["new"]), set(analyses["old"])
    if old - new:
        raise SystemExit(f"{len(old - new)} Analyses removed; the comparison assumes none")
    return {"shared": len(new & old), "added": len(new - old), "removed": len(old - new),
            "added_ids": sorted(new - old)}


def _open(specs: dict[str, StoreSpec]) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    queries = {label: query_store(spec.store) for label, spec in specs.items()}
    by_id = {
        label: {v["analysis_id"]: k for k, v in q.analyses_table().items()}
        for label, q in queries.items()
    }
    return queries, by_id


# --- build collisions -------------------------------------------------------------


def _source_vcfs(manifest: Path, analyses: tuple[str, ...]) -> dict[str, Path]:
    """The source GWAS-VCF of each Analysis, from the build's input manifest."""
    frame = pd.read_csv(manifest, sep="\t", dtype=str, usecols=["analysis_id", "source_file"])
    paths = dict(zip(frame["analysis_id"], frame["source_file"], strict=True))
    missing = [a for a in analyses if a not in paths]
    if missing:
        raise SystemExit(f"{manifest}: no source_file for {missing}")
    return {a: Path(paths[a]) for a in analyses}


def _stored_z(q: Any) -> Callable[[str, str], float]:
    def stored(alid: str, analysis: str) -> float:
        look = q.lookup([alid], [analysis])
        return float(look["z"][0]) if len(look["z"]) else float("nan")

    return stored


@dataclass(frozen=True)
class OtherStore:
    """What the other Store holds, to place a collision row there.

    `observed` is its observed axis, `collisions` its collision rows (ALID to
    rsid), and `axis_row` looks an ALID up on its whole completed axis, panel
    rows included.
    """

    observed: set[str]
    collisions: dict[str, str]
    axis_row: Callable[[str], Any]


def _relation(alid: str, other: OtherStore, label: str) -> dict[str, Any]:
    """Where a collision row's variant stands in the other Store.

    Looked up on the other Store's whole axis, not only among its collisions:
    first as an observed variant, in the row's own spelling and then in the one
    #288's complement gives it, then as a panel-only row. A row that collides
    here but was held uncollided there is the case a reader most needs to see:
    the same ALID returned another variant's association before.
    """
    spellings = [(alid, False)]
    if complement_alid(alid) != alid:
        spellings.append((complement_alid(alid), True))
    found = next(((a, renamed, "observed") for a, renamed in spellings if a in other.observed),
                 None)
    if found is None:
        found = next(((a, renamed, "panel-only") for a, renamed in spellings
                      if other.axis_row(a) is not None), None)
    one_sided = "new" if label == "new" else "gone"
    if found is None:
        return {"relation": one_sided, "other_store_alid": None, "other_store_rsid": None,
                "other_store_row": None, "other_store_collision": None}
    other_alid, renamed, row = found
    collided = other_alid in other.collisions
    kind = ("shared" if collided else one_sided) + (", renamed by #288" if renamed else "")
    record = other.axis_row(other_alid)
    return {"relation": kind, "other_store_alid": other_alid,
            "other_store_rsid": None if record is None else record.rsid,
            "other_store_row": row, "other_store_collision": collided}


def build_collisions_section(
    specs: dict[str, StoreSpec], rows: dict[str, pd.DataFrame], queries: dict[str, Any],
    identity: dict[str, Any], chain: ChainStrands, args: argparse.Namespace,
    observed: dict[str, set[str]],
) -> dict[str, Any]:
    """Every build-collision row of each Store, its sources and whose statistics it holds.

    Rows come from each Observed-Only parent's blank `source_alid`s and must be
    on the published completed axis under the same rsid. Each Store's lift is
    reproduced from the revision its build record names (`store_lift`).
    """
    vcfs = _source_vcfs(args.source_manifest, COLLISION_ANALYSES)
    rsids = {label: dict(zip(rows[label]["alid"], rows[label]["rsid"], strict=True))
             for label in LABELS}
    others = {label: OtherStore(observed[label], rsids[label],
                                queries[label]._variant_axis.by_identifier)
              for label in LABELS}
    stores: dict[str, Any] = {}
    lifts: dict[str, Any] = {}
    for label in LABELS:
        q, other = queries[label], others["old" if label == "new" else "new"]
        revision = identity[label]["runtime"]["build"]["opengwasdb_rev"]
        lift = _build_collisions.store_lift(REPO, revision, args.lift_workdir, args.chain)
        lifts[label] = lift.record()
        records = []
        for alid, rsid in rsids[label].items():
            published = q._variant_axis.by_identifier(alid)
            if published is None or published.rsid != rsid:
                raise SystemExit(f"{label}: collision row {alid} ({rsid}) is not on the published "
                                 f"axis under that rsid")
            record = _build_collisions.describe_collision(
                alid, lift=lift.lift, sources=chain.sources, vcfs=vcfs, stored_z=_stored_z(q))
            donor = next((c for c in record["sources"] if c["source"] == record["statistics_from"]),
                         None)
            records.append({
                **record, "rsid": rsid,
                "rsid_names_statistics_source": None if donor is None else donor["rsid"] == rsid,
                "observed_cells": len(q.phewas(alid, observed_only=True)["z"]),
                **_relation(alid, other, label),
            })
        n_analyses = int(identity[label]["n_analyses"])
        stores[label] = {
            "n_rows": len(records),
            "n_analyses": n_analyses,
            "cells": len(records) * n_analyses,
            "observed_cells": sum(r["observed_cells"] for r in records),
            "rows": records,
        }
    return {
        "issue": COLLISION_ISSUE,
        "analyses_checked": list(vcfs),
        "source_vcfs": {a: str(path) for a, path in vcfs.items()},
        "lift": lifts,
        "stores": stores,
    }


def earlier_runs(entries: list[str]) -> list[dict[str, Any]]:
    """Each earlier run's warm timings, read from its own query-time artifact.

    `LABEL=PATH` names a superseded artifact (extracted from git). Its numbers
    are copied with the file's sha256 and the commit it records, so the report
    can show how far the timings moved between runs without anyone typing them.
    """
    out = []
    for entry in entries:
        label, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"--earlier-queries wants LABEL=PATH, got {entry!r}")
        artifact = _json(Path(path))
        timings = {
            store["label"]: {
                row["query"]: {"median_ms": row["median_ms"], "min_ms": min(row["warm_ms"]),
                               "max_ms": max(row["warm_ms"]), "reps": len(row["warm_ms"])}
                for row in store["timings"] if row.get("warm_ms")
            }
            for store in artifact["stores"]
        }
        out.append({"label": label, "sha256": _sha256(Path(path)), "commit": artifact.get("commit"),
                    "measured_at": artifact.get("measured_at"), "timings": timings})
    return out


def gate_failures(result: dict[str, Any]) -> list[str]:
    """The lift checks that must hold before the fidelity counts mean anything."""
    lift = result["lift_axis"]
    failures = []
    if lift["one_sided_outside_reverse_block"]:
        failures.append(f"{len(lift['one_sided_outside_reverse_block'])} variant(s) only one "
                        "Store carries lie outside every reverse-strand chain block")
    for verified, total in (("lift_verified_renamed", lift["renamed_pairs"]),
                            ("lift_verified_collapsed", len(lift["collapsed"])),
                            ("lift_verified_flip", lift["expected_reoriented_total"])):
        if lift[verified] < total:
            failures.append(f"{verified}: {lift[verified]:,} of {total:,}")
    return failures


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Laps:
    """Wall time of each section of the run, for the report's run-time line."""

    def __init__(self) -> None:
        self.started = self.last = time.perf_counter()
        self.sections: dict[str, float] = {}

    def __call__(self, name: str) -> None:
        now = time.perf_counter()
        self.sections[name] = round(now - self.last, 1)
        self.last = now

    def record(self) -> dict[str, float]:
        return {"total": round(time.perf_counter() - self.started, 1), **self.sections}


def _source_alid_lookup(parent: Path) -> tuple[Any, Callable[[str], str | None]]:
    """An open Observed-Only release, and its `source_alid` for one ALID."""
    q = query_store(parent)

    def source_of(alid: str) -> str | None:
        row = q._variant_axis.by_identifier(alid)
        return None if row is None else row.source_alid

    return q, source_of


def _axes_and_lift(
    args: argparse.Namespace, specs: dict[str, StoreSpec], identity: dict[str, Any],
    queries: dict[str, Any], collided: dict[str, set[str]],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], ChainStrands, LiftMap,
           dict[str, Any], pd.DataFrame]:
    axes = {label: load_axis(specs[label].store, int(queries[label]._variant_axis.n_variants))
            for label in LABELS}
    parents = {
        label: load_axis(specs[label].parent, int(identity[label]["completion"]["n_variants_total"])
                         - int(identity[label]["completion"]["n_variants_new"]))
        for label in LABELS
    }
    chain = ChainStrands(args.chain)
    old_parent, source_of = _source_alid_lookup(specs["old"].parent)
    try:
        lift, summary, shared = lift_section(parents["old"], parents["new"], chain, args.chain,
                                             collided, source_of)
    finally:
        old_parent.close()
    print(f"lift: {json.dumps({k: v for k, v in summary.items() if k != 'collapsed'})}",
          flush=True)
    return axes, parents, chain, lift, summary, shared


def main() -> None:
    args = _parser().parse_args()
    laps = _Laps()
    specs = _specs(args)
    queries, by_id = _open(specs)
    analyses = {label: sorted(by_id[label]) for label in LABELS}
    shared_analyses = sorted(set(analyses["new"]) & set(analyses["old"]))
    identity = {label: identity_section(specs[label], analyses[label]) for label in LABELS}
    result: dict[str, Any] = {
        "stores": identity,
        "membership": _membership(analyses),
        "environment": _environment(queries),
        "loadavg_start": loadavg(),
    }
    laps("identity")

    collision_rows = {label: _build_collisions.collision_rows(specs[label].parent)
                      for label in LABELS}
    collided = {label: set(collision_rows[label]["alid"]) for label in LABELS}
    axes, parents, chain, lift, summary, shared = _axes_and_lift(args, specs, identity, queries,
                                                                 collided)
    result["lift_axis"] = {**summary, "chain": str(args.chain), "chain_sha256": _sha256(args.chain)}
    laps("axes_and_lift")
    failures = gate_failures(result)
    if failures:
        _stop(result, args, "; ".join(failures))

    cols = {label: np.array([by_id[label][a] for a in shared_analyses]) for label in LABELS}
    tol = tolerance([identity[label]["encoding"] for label in LABELS])
    result["fidelity"] = fidelity_section(queries, cols, lift, shared, chain, axes,
                                          parents["old"], shared_analyses, tol, args)
    laps("fidelity")
    unexplained = sum(s["summary"]["c_unexplained"] for s in result["fidelity"]["strata"].values())
    if unexplained:
        _stop(result, args, f"{unexplained:,} unexplained observed cells")

    result["build_collisions"] = build_collisions_section(
        specs, collision_rows, queries, identity, chain, args,
        {"new": lift.new_axis, "old": lift.old_axis})
    laps("build_collisions")
    neither = [r["alid"] for st in result["build_collisions"]["stores"].values()
               for r in st["rows"] if r["matches_neither"]]
    if neither:
        _stop(result, args, f"collision rows whose statistics match neither source: {neither}")

    result["mr"], hits = mr_section(queries, by_id)
    result["mr_observed_agreement"] = observed_mr_agreement(result["mr"])
    laps("mr")
    if not all(r["same_sign"] and r["same_significance"] for r in result["mr_observed_agreement"]):
        _stop(result, args, "an observed-only MR estimate changed sign or significance")

    result["regions"] = region_section(queries, hits, lift)
    laps("regions")
    result["structure"] = {label: discover_arrays(specs[label].store) for label in LABELS}
    result["typesize"] = typesize_section(queries, result["structure"], args.fidelity_seed)
    laps("structure")

    selection = resolve_selection(axes, shared_analyses)
    result["selection"] = selection
    result["queries"] = query_time_section(args, specs, selection)
    result["earlier_runs"] = earlier_runs(args.earlier_queries)
    laps("queries")
    harness = {s["label"]: s for s in _json(args.queries_output)["stores"]}
    result["size"] = {
        label: size_section(specs[label], identity[label], int(queries[label]._variant_axis
                                                                .n_variants),
                            harness[label]["footprint"])
        for label in LABELS
    }
    result["loadavg_end"] = loadavg()
    result["wall_seconds"] = laps.record()
    write_artifact(args.output, {**result, **provenance()})


def _parser() -> argparse.ArgumentParser:
    s9 = STORES / "OGS-00009"
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--new-store", type=Path, default=s9 / "store.opengwasdb")
    ap.add_argument("--new-records", type=Path, default=s9 / "records")
    ap.add_argument("--new-parent", type=Path,
                    default=s9 / "replaced/20261023T190459Z/store.opengwasdb",
                    help="the Observed-Only release the new Store was completed from")
    ap.add_argument("--new-parent-records", type=Path, default=s9 / "records")
    ap.add_argument("--old-store", type=Path, default=STORES / "OGS-00010/store.opengwasdb")
    ap.add_argument("--old-records", type=Path, default=STORES / "OGS-00010/records")
    ap.add_argument("--old-parent", type=Path,
                    default=s9 / "replaced/20261021T223815Z/store.opengwasdb",
                    help="the old OGS-00009, which OGS-00010 was completed from")
    ap.add_argument("--old-parent-records", type=Path,
                    default=s9 / "replaced/20261021T223815Z/records")
    ap.add_argument("--chain", type=Path, default=CHAIN)
    ap.add_argument("--source-manifest", type=Path, default=s9 / "work/analyses.tsv",
                    help="the build input manifest naming each Analysis's source GWAS-VCF")
    ap.add_argument("--lift-workdir", type=Path,
                    default=Path(tempfile.gettempdir()) / "opengwasdb-lift-archives",
                    help="where an older revision's liftover.py is git-archived to run it")
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--queries-output", type=Path, default=QUERIES_OUTPUT,
                    help="where the generic harness writes the query-time artifact")
    ap.add_argument("--earlier-queries", action="append", default=[], metavar="LABEL=PATH",
                    help="an earlier run's query-time artifact, to report how far timings "
                    "moved between runs; repeatable")
    ap.add_argument("--fidelity-windows", type=int, default=FIDELITY_WINDOWS)
    ap.add_argument("--fidelity-window-rows", type=int, default=FIDELITY_WINDOW_ROWS)
    ap.add_argument("--fidelity-seed", type=int, default=FIDELITY_SEED)
    ap.add_argument("--reps", type=int, default=7, help="warm repeats per shape (at least 5)")
    ap.add_argument("--skip-rss", action="store_true", help="skip the per-shape RSS probes")
    ap.add_argument("--max-start-load", type=float, default=48.0,
                    help="the harness waits for a 1-minute load below this before each "
                    "Store's column (0 disables); 48 is about a fifth of this node's 224 cores")
    return ap


if __name__ == "__main__":
    main()
