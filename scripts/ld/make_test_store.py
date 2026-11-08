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
import hashlib
import importlib
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import numpy.typing as npt
import zarr

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
from ogld.identity import Orientation, orient, resolve_reversed_indels
from ogld.store import StoreWriter, Variants
from ogld.store.writer import chromosome_ranges

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "tests" / "data" / "test-panel" / "store"
PANEL_ID = "hgdp1kgp-v2-EUR-test-panel"
#: The time and cost of a build the Test Store was not: fixed, so regeneration is byte-identical.
NOT_A_BUILD = "1970-01-01T00:00:00Z"


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


def chromosome_order(label: str) -> int:
    return {"X": 23, "Y": 24, "MT": 25}.get(label) or int(label)


def engine_order(o: Orientation) -> tuple[int, int, str]:
    """The engine's variant order: chromosome, position, then ALID bytewise (spec §2)."""
    return chromosome_order(o.chrom), o.position, o.alid


def variant_set() -> VariantSet:
    """The rows that pass the filters, reversed indels resolved, in the engine's order."""
    v = panel_data().variants()
    passing = np.flatnonzero(v.passes_filters)
    records = [(str(v.chrom[r]), int(v.pos[r]), str(v.ref[r]), str(v.alt[r])) for r in passing]
    kept, dropped = resolve_reversed_indels(records)
    found = {i: orient(*records[i]) for i in kept}
    order = sorted(
        kept, key=lambda i: (chromosome_order(records[i][0]), records[i][1], found[i].alid)
    )
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
    size = len(position)
    reach = np.searchsorted(position, position + NEAR_BP, side="right")
    lengths = (reach - np.arange(size) - 1).astype(np.int64)
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
            "sha256": {f.name: sha256(f) for f in (panel.pgen, panel.pvar, panel.psam)},
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


def provenance(k: int, vs: VariantSet, counts: tuple[int, int, int, int]) -> Provenance:
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
        # Not measured: the diagnostic needs the build's PC Adjustment (ogld#5).
        "pc_diagnostic": None,
        "counts": {
            "near_pairs": near,
            "far_candidates": candidates,
            "far_pairs": far_pairs,
            "index_entries": entries,
        },
        "software": {
            "ogld": "unversioned: written by scripts/make_test_store.py",
            "numpy": np.__version__,
            "zarr": zarr.__version__,
        },
        "build": {
            "created_at": NOT_A_BUILD,
            "wall_seconds": 0.0,
            "peak_memory_bytes": 0,
            "threads": 1,
        },
    }
    return cast(Provenance, record)


def write_test_store(k: int, out: str | Path) -> Path:
    """Write the Test Store at `out` from the Test Panel's reference r at `k`; return `out`."""
    vs = variant_set()
    writer = StoreWriter.create(out, variants=vs.variants, near_bp=NEAR_BP, far_bp=FAR_BP)
    ranges = chromosome_ranges(vs.variants.chrom)
    floor = far_code_floor(t_far(panel_data().panel().n_samples, k))
    totals = np.zeros(4, dtype=np.int64)
    for chrom, (dense, held) in code_matrices(vs, k, ranges).items():
        start, stop = ranges[chrom]
        position = vs.variants.position[start:stop]
        totals += write_chromosome(writer, chrom, start, position, dense, held, floor)
    writer.finish(
        provenance(k, vs, (int(totals[0]), int(totals[1]), int(totals[2]), int(totals[3])))
    )
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
