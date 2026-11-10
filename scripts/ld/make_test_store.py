#!/usr/bin/env python3
"""Write the Test Store, `tests/data/test-panel/store/` (opengwas/ogld#12).

The Test Store is the LD Store of the Test Panel (`CONTEXT.md`), written from the panel's
independent `reference_r(10)`, so a reader's tests have real data in the real format before any
builder exists. Its variants are the rows that pass the build's filters, with the engine's ALID,
A1 and A2 (`ogld.identity`), reversed indels resolved by the engine's rule; its r is the
reference r signed to A1 and quantised (spec §3); its Near Band is every pair within 3 Mb and
its Proxy Index every pair within 10 Mb whose Stored r reaches t_far for n = 771 and the k it
was written at (spec §4, §5). It is deterministic: the same inputs write the same bytes.

    pixi run -e dev python scripts/make_test_store.py [--k 10] [--force] [out]

`write_test_store(k, out)` writes the store at either k the Test Panel holds a reference for.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import math
import resource
import shutil
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import zarr

import ogld
from ogld.build.near_band import read_alt_counts
from ogld.format import (
    ENCODING,
    FAR_BP,
    FORMAT_VERSION,
    NEAR_BP,
    RESERVED_CODE,
    T_FAR_Q,
    Provenance,
    far_code_floor,
    quantise,
    t_far,
)
from ogld.format.geometry import chromosome_ranges, near_row_lengths
from ogld.identity import Orientation, orient, resolve_reversed_indels, variant_order_key
from ogld.store import StoreWriter, Variants

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "tests" / "data" / "test-panel" / "store"
TEST_PANEL = ROOT / "tests" / "data" / "test-panel"
PANEL_ID = "hgdp1kgp-v2-EUR-test-panel"
#: The seed ogld#405's cross-chromosome pair set is drawn with (spec §7).
DIAGNOSTIC_SEED = 405
RATE_R2 = 0.01
NEAR_100KB = 100_000
QUANTILES = (0, 0.5, 0.9, 0.99, 1)
LONG_RANGE = ROOT / "docs" / "spec" / "long-range-ld-regions-grch38.tsv"


@dataclass(frozen=True)
class VariantSet:
    """The Test Store's Variant Set: Test Panel rows in the engine's order, and their signs."""

    panel_rows: npt.NDArray[np.int64]
    variants: Variants
    sign: npt.NDArray[np.float64]
    dropped_reversed: int


def panel_data() -> ModuleType:
    """The Test Panel's loader, `tests/testdata.py`: the one way to read the panel's files."""
    tests = str(ROOT / "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    return importlib.import_module("testdata")


def variant_set() -> VariantSet:
    """The rows that pass the filters, reversed indels resolved, in the engine's order."""
    v = panel_data().variants()
    passing = np.flatnonzero(v.passes_filters)
    records = [(str(v.chrom[r]), int(v.pos[r]), str(v.ref[r]), str(v.alt[r])) for r in passing]
    kept, dropped = resolve_reversed_indels(records)
    found = {i: orient(*records[i]) for i in kept}
    order = sorted(kept, key=lambda i: variant_order_key(found[i]))
    rows = passing[order].astype(np.int64)
    oriented = [found[i] for i in order]
    is_alt = np.array([o.a1_is_alt for o in oriented])
    table = variants_table(oriented, a1_frequency(rows, is_alt))
    return VariantSet(rows, table, np.where(is_alt, 1.0, -1.0), len(dropped))


def a1_frequency(
    rows: npt.NDArray[np.int64], is_alt: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """A1's frequency among the kept samples: the ALT frequency, or its complement."""
    panel = panel_data().panel()
    alt = read_alt_counts(panel.pgen, panel.psam, panel.keep, rows)
    alt_frequency = alt.mean(axis=1) / 2
    return np.where(is_alt, alt_frequency, 1 - alt_frequency)


def variants_table(oriented: list[Orientation], frequency: npt.NDArray[np.float64]) -> Variants:
    return Variants(
        chrom=np.array([o.chrom for o in oriented], dtype=np.str_),
        position=np.array([o.position for o in oriented], dtype=np.int64),
        alid=np.array([o.alid for o in oriented], dtype=np.str_),
        a1=np.array([o.a1 for o in oriented], dtype=np.str_),
        a2=np.array([o.a2 for o in oriented], dtype=np.str_),
        a1_frequency=frequency,
    )


def code_matrices(
    vs: VariantSet, k: int, ranges: dict[str, tuple[int, int]]
) -> dict[str, tuple[npt.NDArray[np.int16], npt.NDArray[np.bool_]]]:
    """Per chromosome, the reference r as codes for local rows (lo, hi), lo < hi, and where held."""
    ref = panel_data().reference_r(k)
    store_row = np.full(len(panel_data().variants()), -1, dtype=np.int64)
    store_row[vs.panel_rows] = np.arange(len(vs.panel_rows))
    a, b = store_row[ref.row_i], store_row[ref.row_j]
    if (a < 0).any() or (b < 0).any():
        raise ValueError("reference pairs name a row the Variant Set does not hold")
    r = ref.r * vs.sign[a] * vs.sign[b]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    codes = quantise(r)
    out = {}
    for chrom, (start, stop) in ranges.items():
        mine = (lo >= start) & (lo < stop)
        size = stop - start
        dense = np.full((size, size), RESERVED_CODE, dtype=np.int16)
        held = np.zeros((size, size), dtype=np.bool_)
        dense[lo[mine] - start, hi[mine] - start] = codes[mine]
        held[lo[mine] - start, hi[mine] - start] = True
        out[chrom] = (dense, held)
    return out


def require_every_pair(
    held: npt.NDArray[np.bool_], position: npt.NDArray[np.int64], chrom: str
) -> None:
    """Every pair within far_bp must have a reference r, or a Bound would hide a missing one."""
    gap = position[None, :] - position[:, None]
    wanted = np.triu(gap <= FAR_BP, k=1)
    if (wanted & ~held).any():
        raise ValueError(f"the reference lacks a pair within {FAR_BP} bp on chromosome {chrom}")


def write_chromosome(
    writer: StoreWriter,
    chrom: str,
    start: int,
    position: npt.NDArray[np.int64],
    dense: npt.NDArray[np.int16],
    held: npt.NDArray[np.bool_],
    floor: int,
) -> tuple[int, int, int, int]:
    """Write one chromosome's Near Band and Proxy Index; return the counts provenance records."""
    require_every_pair(held, position, chrom)
    lengths = near_row_lengths(position, NEAR_BP)
    codes = np.concatenate([dense[row, row + 1 : row + 1 + n] for row, n in enumerate(lengths)])
    writer.write_near(start, lengths, codes)
    both = held | held.T
    symmetric = np.where(held, dense, dense.T)
    keep = both & (np.abs(symmetric.astype(np.int32)) >= floor)
    partner = np.nonzero(keep)[1].astype(np.int64) + start
    indptr = np.concatenate([[0], np.cumsum(keep.sum(axis=1))]).astype(np.int64)
    writer.write_index(chrom, indptr, partner, symmetric[keep])
    far = np.triu(position[None, :] - position[:, None] > NEAR_BP, k=1) & held
    return len(codes), int(far.sum()), int((far & keep).sum()), int(keep.sum())


INPUTS = ("panel.pgen", "panel.pvar", "panel.psam", "panel.keep", "reference/r.npz")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adjustment(k: int) -> dict[str, object]:
    if k == 0:
        return {
            "covariates": [],
            "k": 0,
            "k_target": 0,
            "k_min": None,
            "pc_file": None,
            "pc_check": None,
        }
    manifest = panel_data().manifest()
    check = {key: manifest["pc_check"][key] for key in _PC_CHECK_KEYS}
    pcs = manifest["inputs"]["pcs"]
    return {
        "covariates": [f"PC{i}" for i in range(1, k + 1)],
        "k": k,
        "k_target": k,
        "k_min": 5,
        "pc_file": {"sha256": pcs["sha256"], "source": pcs["declared_source"]},
        "pc_check": _plain(check),
    }


_PC_CHECK_KEYS = (
    "window_bp",
    "step_bp",
    "windows",
    "min_share",
    "max_ratio",
    "scan",
    "dropped",
    "null",
)


def _plain(value: object) -> object:
    """A manifest value as plain dicts and lists, which is what a JSON attribute holds."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def provenance_inputs(k: int) -> dict[str, object]:
    """The provenance keys that say what the LD was computed from, and by what."""
    panel = panel_data().panel()
    n = panel.n_samples
    iids = sorted(line.strip() for line in panel.keep.read_text().splitlines()[1:])
    return {
        "source": {
            "kind": "genotypes",
            "description": (
                f"the Test Panel (tests/data/test-panel): HGDP+1kGP v2 EUR chr21 and chr22, "
                f"{n} samples; r is the panel's independent reference_r({k}), not a build's"
            ),
            "sha256": {name: sha256(TEST_PANEL / name) for name in INPUTS},
        },
        "panel": {
            "id": PANEL_ID,
            "ancestry": "EUR",
            "n": n,
            "samples_sha256": hashlib.sha256(("\n".join(iids) + "\n").encode()).hexdigest(),
        },
        "reference_assembly": "GRCh38",
        "variant_filters": {
            "filter": ["PASS", "."],
            "mac_min": 20,
            "info_min": None,
            "autosomes_only": True,
        },
    }


def provenance(
    k: int,
    vs: VariantSet,
    counts: tuple[int, int, int, int],
    diagnostic: dict[str, object] | None,
    build: dict[str, object],
) -> Provenance:
    far_t = t_far(panel_data().panel().n_samples, k)
    near, candidates, far_pairs, entries = counts
    record: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "encoding": dict(ENCODING),
        **provenance_inputs(k),
        "variants_dropped": {
            "zero_variance": 0,
            "duplicate_reversed_indel": vs.dropped_reversed,
        },
        "n_variants": len(vs.panel_rows),
        "near_bp": NEAR_BP,
        "far_bp": FAR_BP,
        "adjustment": adjustment(k),
        "t_far": far_t,
        "t_far_q": T_FAR_Q,
        "far_code_floor": far_code_floor(far_t),
        "pc_diagnostic": diagnostic,
        "counts": {
            "near_pairs": near,
            "far_candidates": candidates,
            "far_pairs": far_pairs,
            "index_entries": entries,
        },
        "software": {
            "ogld": ogld.__version__,
            "numpy": np.__version__,
            "zarr": zarr.__version__,
        },
        "build": build,
    }
    return cast(Provenance, record)


def build_record(started: float) -> dict[str, object]:
    """What this run cost: when it finished, how long it took, its peak memory, its threads."""
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "created_at": now,
        "wall_seconds": time.perf_counter() - started,
        "peak_memory_bytes": int(peak_kib) * 1024,
        "threads": 1,
    }


def unit_columns(vs: VariantSet, k: int) -> npt.NDArray[np.float64]:
    """Each variant's A1 dosages across the kept samples, after regressing out an intercept and
    the first `k` PCs, as a unit vector: a row product is r signed to A1 (spec §3)."""
    data = panel_data()
    panel = data.panel()
    alt = read_alt_counts(panel.pgen, panel.psam, panel.keep, vs.panel_rows).astype(np.float64)
    z = alt - alt.mean(axis=1, keepdims=True)
    if k:
        z = z - residual_fit(z, kept_pcs(k))
    z /= np.sqrt((z * z).sum(axis=1, keepdims=True))
    return z * vs.sign[:, None]


def kept_pcs(k: int) -> npt.NDArray[np.float64]:
    """The intercept and first `k` PCs of the kept samples, in the genotypes' sample order."""
    data = panel_data()
    panel = data.panel()
    kept = {line.strip() for line in panel.keep.read_text().splitlines()[1:]}
    ordered = [line.split("\t")[0] for line in panel.psam.read_text().splitlines()[1:]]
    wanted = [iid for iid in ordered if iid in kept]
    pcs = data.pcs()
    row = {iid: i for i, iid in enumerate(pcs.iids)}
    values = pcs.values[[row[iid] for iid in wanted], :k]
    return np.column_stack([np.ones(len(wanted)), values])


def residual_fit(z: npt.NDArray[np.float64], design: npt.NDArray[np.float64]) -> Any:
    """The part of each row of `z` the design's columns explain (least squares)."""
    beta, *_ = np.linalg.lstsq(design, z.T, rcond=None)
    return (design @ beta).T


def long_range_mask(vs: VariantSet) -> npt.NDArray[np.bool_]:
    """True for each variant inside a Long-Range LD Region (build-inputs §9)."""
    rows = [
        line.split("\t")
        for line in LONG_RANGE.read_text().splitlines()
        if line and not line.startswith("#")
    ][1:]
    inside = np.zeros(len(vs.variants), dtype=np.bool_)
    for _, chrom, start, end, *_ in rows:
        on = (vs.variants.chrom == chrom) & (vs.variants.position >= int(start))
        inside |= on & (vs.variants.position <= int(end))
    return inside


def pc_diagnostic(vs: VariantSet, k: int) -> dict[str, object]:
    """What PC Adjustment did to chance LD, measured on the Test Panel (spec §7): every
    chr21 × chr22 pair, the share at r² ≥ 0.01 before and after, what sampling predicts, and the
    quantiles of the change in r within 100 kb outside the Long-Range LD Regions."""
    before, after = unit_columns(vs, 0), unit_columns(vs, k)
    chrom, pos = vs.variants.chrom, vs.variants.position
    across = np.triu(chrom[:, None] != chrom[None, :], k=1)
    r_before, r_after = before @ before.T, after @ after.T
    near = np.triu(
        (chrom[:, None] == chrom[None, :]) & (pos[None, :] - pos[:, None] <= NEAR_100KB), k=1
    )
    outside = ~long_range_mask(vs)
    near &= outside[:, None] & outside[None, :]
    change = np.quantile((r_after - r_before)[near], QUANTILES)
    n = panel_data().panel().n_samples
    return {
        "pairs": {
            "source": (
                "every chr21 × chr22 pair of the Test Panel's variants: the Test Panel holds no "
                "other cross-chromosome pairs, so none is drawn and the seed is recorded only"
            ),
            "seed": DIAGNOSTIC_SEED,
            "n_pairs": int(across.sum()),
        },
        "rate_r2_ge_0_01_unadjusted": float((r_before[across] ** 2 >= RATE_R2).mean()),
        "rate_r2_ge_0_01_adjusted": float((r_after[across] ** 2 >= RATE_R2).mean()),
        "rate_r2_ge_0_01_sampling": math.erfc(math.sqrt(RATE_R2 * (n - k - 1) / 2)),
        "near_100kb_change": {f"{q:g}": float(c) for q, c in zip(QUANTILES, change, strict=True)},
    }


def write_test_store(k: int, out: str | Path) -> Path:
    """Write the Test Store at `out` from the Test Panel's reference r at `k`; return `out`."""
    started = time.perf_counter()
    vs = variant_set()
    writer = StoreWriter.create(out, variants=vs.variants, near_bp=NEAR_BP, far_bp=FAR_BP)
    ranges = chromosome_ranges(vs.variants.chrom)
    floor = far_code_floor(t_far(panel_data().panel().n_samples, k))
    totals = np.zeros(4, dtype=np.int64)
    for chrom, (dense, held) in code_matrices(vs, k, ranges).items():
        start, stop = ranges[chrom]
        position = vs.variants.position[start:stop]
        totals += write_chromosome(writer, chrom, start, position, dense, held, floor)
    counts = (int(totals[0]), int(totals[1]), int(totals[2]), int(totals[3]))
    diagnostic = pc_diagnostic(vs, k)
    writer.finish(provenance(k, vs, counts, diagnostic, build_record(started)))
    return Path(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", nargs="?", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--k", type=int, default=10, choices=panel_data().KS)
    parser.add_argument("--force", action="store_true", help="replace a store already at `out`")
    args = parser.parse_args(argv)
    if args.out.exists():
        if not args.force:
            parser.error(f"{args.out} exists; pass --force to write it again")
        shutil.rmtree(args.out)
    out = write_test_store(args.k, args.out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
