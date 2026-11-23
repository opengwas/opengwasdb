#!/usr/bin/env python3
"""Write the Test Store, `tests/data/test-panel/store/` (opengwas/ogld#12).

The Test Store is the LD Store of the Test Panel (`CONTEXT.md`), written from the panel's
independent `reference_r(10)`, so a reader's tests have real data in the real format before any
builder exists. Its variants are the rows that pass the build's filters, with the engine's ALID,
A1 and A2 in the sorted spelling LD Stores keep until #462 (`opengwasdb.variants.sorted_spelling`),
reversed indels resolved by that spelling's rule; its r is the
reference r signed to A1. The script supplies that r and what only its source knows, and the
store makes the rest (`StoreWriter`, ogld#17). That is its codes,
quantised at × 511, HGDP+1kGP EUR's encoding (spec §3; ADR 0004), the Near Band, the Far Pairs at
t_far for n = 771 and the k it was written at (spec §4, §5), the Partner Rows, and the counts. It
holds no Proxy Index: #72 removed the component (ADR 0011). It is "import a
precomputed r" in miniature. It is deterministic: the same inputs write the same bytes.

    pixi run -e dev python scripts/make_test_store.py --r-scale=<scale> --t-partner=<r²> \
        [--k 10] [--force] [out]

`write_test_store(k, out, r_scale, t_partner)` writes the store at either k the Test Panel holds a
reference for, at any scale the format admits: the committed Test Store is × 511, and tests write
× 32767 stores beside it. The scale and the Partner Threshold are always given, as a build's are:
neither has a default.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
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

import opengwasdb.ld
from opengwasdb.ld.build import diagnostic, record, tiles
from opengwasdb.ld.build.inputs import fileset, read_keep
from opengwasdb.ld.build.pc_check import K_MIN
from opengwasdb.ld.format import FAR_BP, NEAR_BP, Adjustment, PcCheck, REncoding, Sourced
from opengwasdb.ld.format.geometry import chromosome_ranges, local_rows
from opengwasdb.ld.genotypes import Genotypes
from opengwasdb.ld.store import StoreWriter, Variants
from opengwasdb.variants.axis import alid_sort_key
from opengwasdb.variants.sorted_spelling import (
    SortedIdentity,
    resolve_sorted_reversed_indels,
    sorted_identity,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "tests" / "ld" / "data" / "test-panel" / "store"
TEST_PANEL = ROOT / "tests" / "ld" / "data" / "test-panel"
PANEL_ID = "hgdp1kgp-v2-EUR-test-panel"


@dataclass(frozen=True)
class VariantSet:
    """The Test Store's Variant Set: Test Panel rows in the engine's order, and their signs."""

    panel_rows: npt.NDArray[np.int64]
    variants: Variants
    sign: npt.NDArray[np.float64]
    dropped_reversed: int


def panel_data() -> ModuleType:
    """The Test Panel's loader, `tests/testdata.py`: the one way to read the panel's files."""
    tests = str(ROOT / "tests" / "ld")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    return importlib.import_module("testdata")


def variant_set() -> VariantSet:
    """The rows that pass the filters, reversed indels resolved, in the engine's order."""
    v = panel_data().variants()
    passing = np.flatnonzero(v.passes_filters)
    records = [(str(v.chrom[r]), int(v.pos[r]), str(v.ref[r]), str(v.alt[r])) for r in passing]
    kept, dropped = resolve_sorted_reversed_indels(records)
    found = {i: sorted_identity(*records[i]) for i in kept}
    order = sorted(kept, key=lambda i: alid_sort_key(found[i].alid))
    rows = passing[order].astype(np.int64)
    oriented = [found[i] for i in order]
    is_alt = np.array([o.a1_is_alt for o in oriented])
    table = variants_table(oriented, a1_frequency(rows, is_alt))
    return VariantSet(rows, table, np.where(is_alt, 1.0, -1.0), len(dropped))


def alt_counts(rows: npt.NDArray[np.int64], pgen: Path | None = None) -> npt.NDArray[np.int8]:
    """The ALT allele count (0, 1, 2) of each row of the Test Panel (or of `pgen`, a fileset
    of rows of it), for the kept samples: [rows, samples], read by the pgen reader
    (`opengwasdb.ld.genotypes.Genotypes`). A missing call is refused, never averaged in (ADR
    0002)."""
    panel = panel_data().panel()
    name = "the Test Panel"
    first = fileset(pgen or panel.pgen, read_keep(panel.keep, name), None, name)
    with Genotypes(first) as genotypes:
        counts = genotypes.rows(rows)
    missing = np.flatnonzero((counts < 0).any(axis=1))
    if len(missing):
        raise opengwasdb.ld.InputError(f"missing call in row {int(rows[missing[0]])} of {name}")
    return counts.astype(np.int8)


def a1_frequency(
    rows: npt.NDArray[np.int64], is_alt: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """A1's frequency among the kept samples: the ALT frequency, or its complement."""
    alt = alt_counts(rows)
    alt_frequency = alt.mean(axis=1) / 2
    return np.where(is_alt, alt_frequency, 1 - alt_frequency)


def variants_table(oriented: list[SortedIdentity], frequency: npt.NDArray[np.float64]) -> Variants:
    return Variants(
        chrom=np.array([o.chrom for o in oriented], dtype=np.str_),
        position=np.array([o.position for o in oriented], dtype=np.int64),
        alid=np.array([o.alid for o in oriented], dtype=np.str_),
        a1=np.array([o.a1 for o in oriented], dtype=np.str_),
        a2=np.array([o.a2 for o in oriented], dtype=np.str_),
        a1_frequency=frequency,
    )


def r_matrices(
    vs: VariantSet, k: int, ranges: dict[str, tuple[int, int]]
) -> dict[str, npt.NDArray[np.float64]]:
    """Per chromosome, the reference r signed to A1 as a matrix over its rows: r[lo, hi] for each
    pair the reference holds, lo < hi, and NaN for every pair it does not. The store cuts each
    whole; a NaN within far_bp of its row is refused there, so a pair the reference lacks is never
    read as a Bound."""
    ref = panel_data().reference_r(k)
    store_row = np.full(len(panel_data().variants()), -1, dtype=np.int64)
    store_row[vs.panel_rows] = np.arange(len(vs.panel_rows))
    a, b = store_row[ref.row_i], store_row[ref.row_j]
    if (a < 0).any() or (b < 0).any():
        raise ValueError("reference pairs name a row the Variant Set does not hold")
    r = ref.r * vs.sign[a] * vs.sign[b]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    out = {}
    for chrom, (start, stop) in ranges.items():
        mine = (lo >= start) & (lo < stop)
        dense = np.full((stop - start, stop - start), np.nan)
        dense[local_rows(lo[mine], start), local_rows(hi[mine], start)] = r[mine]
        out[chrom] = dense
    return out


INPUTS = ("panel.pgen", "panel.pvar", "panel.psam", "panel.keep", "reference/r.npz")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adjustment(k: int) -> Adjustment:
    """The `adjustment` record a build makes (`opengwasdb.ld.build.record.adjustment`) of the Test
    Panel's
    first k PCs, which its manifest's PC Check kept."""
    if k == 0:
        return record.adjustment([], 0, K_MIN, None, None)
    manifest = panel_data().manifest()
    check = {key: manifest["pc_check"][key] for key in _PC_CHECK_KEYS}
    pcs = manifest["inputs"]["pcs"]
    return record.adjustment(
        [f"PC{i}" for i in range(1, k + 1)],
        k,
        K_MIN,
        (pcs["sha256"], pcs["declared_source"]),
        cast(PcCheck, _plain(check)),
    )


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
        "variant_filters": record.variant_filters(20, None),
    }


def sourced(
    k: int, vs: VariantSet, diagnostic: dict[str, object] | None, build: dict[str, object]
) -> Sourced:
    """What only the Test Store's source knows of its provenance; `StoreWriter.finish` adds what
    the store derives from itself (ogld#17)."""
    record: dict[str, object] = {
        **provenance_inputs(k),
        "variants_dropped": {
            "zero_variance": 0,
            "duplicate_reversed_indel": vs.dropped_reversed,
        },
        "adjustment": adjustment(k),
        "pc_diagnostic": diagnostic,
        "software": {
            "ogld": opengwasdb.ld.__version__,
            "numpy": np.__version__,
            "zarr": zarr.__version__,
        },
        "build": build,
    }
    return cast(Sourced, record)


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
    alt = alt_counts(vs.panel_rows).astype(np.float64)
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


def change_pairs(vs: VariantSet) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The PC Diagnostic's pairs for the change in r, as rows (i, j) of the Variant Set: within
    100 kb, both variants outside the Long-Range LD Regions. The build's own definition, called
    rather than copied: `tiles.change_pairs` over `diagnostic.outside_regions` (spec §7)."""
    first, second = [], []
    for chrom, (start, stop) in chromosome_ranges(vs.variants.chrom).items():
        position = vs.variants.position[start:stop]
        band = tiles.Band.of(position, FAR_BP, diagnostic.CHANGE_BP)
        outside = diagnostic.outside_regions(chrom, position)
        t, partner, clear = tiles.change_pairs(band, outside, 0, len(position))
        first.append(start + t[clear])
        second.append(start + partner[clear])
    return np.concatenate(first), np.concatenate(second)


def pc_diagnostic(vs: VariantSet, k: int) -> dict[str, object]:
    """What PC Adjustment did to chance LD, measured on the Test Panel (spec §7): every
    chr21 × chr22 pair, the share at r² ≥ 0.01 before and after, what sampling predicts, and the
    exact quantiles of the change in r over `change_pairs`. The rate, the window, the regions,
    the quantiles and the sampling formula are `opengwasdb.ld.build.diagnostic`'s; only the pairs
    differ
    from a build's, because the Test Panel holds every chr21 × chr22 pair and no other."""
    before, after = unit_columns(vs, 0), unit_columns(vs, k)
    chrom = vs.variants.chrom
    across = np.triu(chrom[:, None] != chrom[None, :], k=1)
    r_before, r_after = before @ before.T, after @ after.T
    i, j = change_pairs(vs)
    change = np.quantile((r_after - r_before)[i, j], diagnostic.QUANTILES)
    n = panel_data().panel().n_samples
    return {
        "pairs": {
            "source": (
                "every chr21 × chr22 pair of the Test Panel's variants: the Test Panel holds no "
                "other cross-chromosome pairs, so none is drawn and the seed is recorded only"
            ),
            "seed": diagnostic.SEED,
            "n_pairs": int(across.sum()),
        },
        "rate_r2_ge_0_01_unadjusted": diagnostic.chance_rate(r_before[across]),
        "rate_r2_ge_0_01_adjusted": diagnostic.chance_rate(r_after[across]),
        "rate_r2_ge_0_01_sampling": diagnostic.sampling_rate(n, k),
        "near_100kb_change": {
            f"{q:g}": float(c) for q, c in zip(diagnostic.QUANTILES, change, strict=True)
        },
    }


def write_test_store(k: int, out: str | Path, r_scale: int, t_partner: float) -> Path:
    """Write the Test Store at `out` from the Test Panel's reference r at `k`, at × `r_scale` and
    with the Partner Threshold `t_partner`, as a build takes both; return `out`."""
    started = time.perf_counter()
    vs = variant_set()
    writer = StoreWriter.create(
        out,
        variants=vs.variants,
        encoding=REncoding(r_scale),
        near_bp=NEAR_BP,
        far_bp=FAR_BP,
        n=panel_data().panel().n_samples,
        k=k,
        t_partner=t_partner,
    )
    ranges = chromosome_ranges(vs.variants.chrom)
    for chrom, r in r_matrices(vs, k, ranges).items():
        writer.write(writer.cut(ranges[chrom][0], r))
    diagnostic = pc_diagnostic(vs, k)
    writer.finish(sourced(k, vs, diagnostic, build_record(started)))
    return Path(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", nargs="?", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--r-scale",
        type=int,
        required=True,
        help="the store's r scale: 511 for the committed Test Store, HGDP+1kGP EUR's (ADR 0004)",
    )
    parser.add_argument(
        "--t-partner",
        type=float,
        required=True,
        help="the r² a pair within near_bp must reach to be in the Partner Rows: 0.01 for the "
        "committed Test Store, the HGDP+1kGP panels'",
    )
    parser.add_argument("--k", type=int, default=10, choices=panel_data().KS)
    parser.add_argument("--force", action="store_true", help="replace a store already at `out`")
    args = parser.parse_args(argv)
    if args.out.exists():
        if not args.force:
            parser.error(f"{args.out} exists; pass --force to write it again")
        shutil.rmtree(args.out)
    out = write_test_store(args.k, args.out, args.r_scale, args.t_partner)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
