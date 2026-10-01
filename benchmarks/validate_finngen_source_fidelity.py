#!/usr/bin/env python3
"""Compare a FinnGen R13 Dense Store cell by cell against the source files it was built from.

`opengwasdb validate` checks a store's internal consistency; it never opens a
source file. This harness closes that loop. For each sampled Analysis it parses
the whole source `.gz`, matches every row to the store's variant axis with its
own allele-pair matcher (deliberately not the builder's), decodes the store's
cells through the public query API, and compares

  * z    against beta / sebeta, negated when the store's canonical effect allele
         is the source REF;
  * se   against sebeta;
  * eaf  against af_alt, or 1 - af_alt when the orientation is flipped;

and reports which source rows have no cell of their own, which store cells have
no source row, and where the store and source disagree about missingness.

Gates come from the encoding the store declares in its manifest (ADR 0037):
z is int16 fixed-point (half a step of 1/scale), se is held to the ADR's 1%
ordinary-cell bound, eaf to the half-step of its int8 residual code.

Usage:
  pixi run -e dev python benchmarks/validate_finngen_source_fidelity.py \
      --store /data/opengwasdb/stores/OGS-00016/store.opengwasdb \
      --source-dir /data/opengwasdb/raw/finngen-r13/releases/r13-full/source \
      --manifest /data/opengwasdb/stores/OGS-00016/work/analyses.tsv \
      --output docs/benchmark-output/opengwasdb_ogs00016_source_fidelity.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

from benchmarks._artifact import provenance, write_artifact
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.query import query_store

PREFIX = "finngen-r13-"
CHROMOSOMES = {**{str(n): n for n in range(1, 23)}, "X": 23, "Y": 24, "MT": 25, "23": 23}

# Named so the report can say why each is sampled: the two inverse-rank-normalised
# quantitative traits, the flagship endpoints of the query benchmark, an endpoint
# and its WIDE twin, and the smallest case count in the release.
ANCHORS = [
    "finngen-r13-BMI_IRN",
    "finngen-r13-HEIGHT_IRN",
    "finngen-r13-T2D",
    "finngen-r13-T2D_WIDE",
    "finngen-r13-I9_CHD",
    "finngen-r13-G6_ALZHEIMER",
]

FLOAT32_RESOLVABLE = 2e-3

# Shared with forked workers, so the 21M-row axis is parsed once.
_AXIS: dict[str, Any] = {}


def _load_axis(store: Path) -> None:
    started = time.perf_counter()
    table = pd.read_csv(
        store / "variants.tsv.gz",
        sep="\t",
        usecols=["#chromosome", "position", "effect_allele", "other_allele"],
        dtype=str,
    )
    chrom = table["#chromosome"].map(CHROMOSOMES).to_numpy(np.int64)
    position = table["position"].astype(np.int64).to_numpy()
    vocab = pd.Index(pd.unique(np.concatenate([table.effect_allele, table.other_allele])))
    width = len(vocab) + 1
    effect = vocab.get_indexer(table.effect_allele)
    other = vocab.get_indexer(table.other_allele)
    keys = _pair_key(chrom, position, effect, other, width)
    order = np.argsort(keys)
    sorted_keys = keys[order]
    if not (np.diff(sorted_keys) > 0).all():
        raise SystemExit("the store's variant axis has duplicate (site, allele pair) keys")
    _AXIS.update(
        vocab=vocab, width=width, effect=effect, order=order, sorted_keys=sorted_keys,
        n_variants=len(table), load_seconds=time.perf_counter() - started,
    )


def _pair_key(chrom: np.ndarray, pos: np.ndarray, a: np.ndarray, b: np.ndarray, width: int):
    """One int64 per (site, unordered allele pair): the store's identity for a variant."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return ((chrom << 28 | pos) * width + lo) * width + hi


def _as_codes(column: pd.Series, vocab: pd.Index) -> np.ndarray:
    categories = column.cat.categories
    return vocab.get_indexer(categories)[column.cat.codes.to_numpy()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"max": 0.0, "p9999": 0.0, "p99": 0.0, "median": 0.0}
    return {
        "max": float(values.max()),
        "p9999": float(np.quantile(values, 0.9999)),
        "p99": float(np.quantile(values, 0.99)),
        "median": float(np.median(values)),
    }


def compare_analysis(task: dict[str, Any]) -> dict[str, Any]:
    """Compare one Analysis. Runs in a forked worker sharing the parsed axis."""
    analysis_id, store, source = task["analysis_id"], Path(task["store"]), Path(task["source"])
    gates = task["gates"]
    ax = _AXIS
    started = time.perf_counter()
    src = pd.read_csv(
        source, sep="\t",
        usecols=["#chrom", "pos", "ref", "alt", "pval", "beta", "sebeta", "af_alt"],
        dtype={"#chrom": str, "ref": "category", "alt": "category"},
    )
    parse_seconds = time.perf_counter() - started

    chrom = src["#chrom"].map(CHROMOSOMES).to_numpy(np.int64)
    ref, alt = _as_codes(src.ref, ax["vocab"]), _as_codes(src.alt, ax["vocab"])
    known = (ref >= 0) & (alt >= 0)
    key = np.where(
        known,
        _pair_key(chrom, src.pos.to_numpy(np.int64), np.maximum(ref, 0), np.maximum(alt, 0),
                  ax["width"]),
        -1,
    )
    slot = np.minimum(np.searchsorted(ax["sorted_keys"], key), ax["n_variants"] - 1)
    hit = known & (ax["sorted_keys"][slot] == key)
    variant = np.where(hit, ax["order"][slot], -1)
    # The store's effect allele is the alphabetically first of the pair; the
    # source's effect allele is ALT.
    flipped = hit & (alt != ax["effect"][np.maximum(variant, 0)])

    beta, se_src, af, pval = (
        src[c].to_numpy(np.float64) for c in ("beta", "sebeta", "af_alt", "pval")
    )
    usable = hit & np.isfinite(beta) & np.isfinite(se_src) & (se_src > 0)
    z_src = beta / se_src * np.where(flipped, -1.0, 1.0)
    af_src = np.where(flipped, 1.0 - af, af)

    started = time.perf_counter()
    q = query_store(store)
    cells = q.analysis(analysis_id)
    read_seconds = time.perf_counter() - started
    q.close()
    where = np.full(ax["n_variants"], -1, np.int64)
    where[cells["variant_index"]] = np.arange(len(cells["z"]))
    cell = np.where(hit, where[np.maximum(variant, 0)], -1)

    # Two source rows that reduce to one (site, allele pair) share a cell.
    frame = pd.DataFrame({"variant": variant})
    collapsed = hit & frame.variant.duplicated(keep=False).to_numpy()
    n_groups = int(frame.variant[collapsed].nunique())
    clean = usable & ~collapsed & (cell >= 0)

    z = cells["z"].astype(np.float64)[np.maximum(cell, 0)]
    se = cells["se"].astype(np.float64)[np.maximum(cell, 0)]
    eaf = cells["eaf"].astype(np.float64)[np.maximum(cell, 0)]

    z_err = np.abs(z - z_src)[clean]
    se_rel = (np.abs(se - se_src) / se_src)[clean]
    frequency = clean & np.isfinite(af_src) & (af_src > 0) & (af_src < 1) & np.isfinite(eaf)
    # ADR 0037: the logit code bounds the relative error of EAF and of 1 - EAF alike. The
    # store returns float32, whose own rounding swamps a relative bound on 1 - EAF once
    # EAF is within ~1e-3 of 1, so that side is checked only where float32 can resolve it.
    minor_resolvable = frequency & (1.0 - af_src > FLOAT32_RESOLVABLE)
    eaf_rel = np.abs(eaf - af_src)[frequency] / af_src[frequency]
    other_rel = (np.abs(eaf - af_src) / (1.0 - af_src))[minor_resolvable]
    eaf_rel = np.concatenate([eaf_rel, other_rel])
    z_abs = np.abs(z_src[clean])
    overflow = z_abs > gates["z_overflow"]
    large = overflow & (np.abs(z - z_src)[clean] <= 1e-4 * z_abs)
    logp_store = -(log_ndtr(-np.abs(z)) + math.log(2.0)) / math.log(10.0)
    with np.errstate(divide="ignore"):
        logp_src = -np.log10(pval)
    p_ok = clean & np.isfinite(logp_src) & (logp_src > 2)

    later_wins = earlier_wins = neither = both = 0
    if collapsed.any():
        rows = np.flatnonzero(collapsed)
        order = np.lexsort((rows, variant[rows]))
        rows = rows[order]
        first, second = rows[0::2], rows[1::2]
        ok_first = np.abs(z[first] - z_src[first]) <= gates["z_abs"]
        ok_second = np.abs(z[second] - z_src[second]) <= gates["z_abs"]
        later_wins = int((ok_second & ~ok_first).sum())
        earlier_wins = int((ok_first & ~ok_second).sum())
        neither = int((~ok_first & ~ok_second).sum())
        both = int((ok_first & ok_second).sum())

    allele_len = np.array([len(a) for a in ax["vocab"]] + [0])
    snv_row = (allele_len[ref] == 1) & (allele_len[alt] == 1)
    example = None
    if collapsed.any():
        group = np.flatnonzero(collapsed)[:2]
        example = [
            {"chrom": str(src["#chrom"].iloc[i]), "pos": int(src.pos.iloc[i]),
             "ref": str(src.ref.iloc[i]), "alt": str(src.alt.iloc[i]),
             "beta": float(beta[i]), "sebeta": float(se_src[i])}
            for i in group
        ]
    invalid = hit & ~usable
    result = {
        "analysis_id": analysis_id,
        "source": source.name,
        "source_sha256": task.get("source_sha256"),
        "manifest_sha256": task.get("manifest_sha256"),
        "source_rows": int(len(src)),
        "rows_matched_to_axis": int(hit.sum()),
        "rows_not_on_axis": int((~hit).sum()),
        "rows_unusable": int(invalid.sum()),
        "unusable_rows_with_a_store_cell": int((invalid & (cell >= 0)).sum()),
        "collapsed_rows": int(collapsed.sum()),
        "collapsed_cells": n_groups,
        "collapsed_snv_rows": int((collapsed & snv_row).sum()),
        "collapsed_example": example,
        "collapsed_later_row_wins": later_wins,
        "collapsed_earlier_row_wins": earlier_wins,
        "collapsed_neither": neither,
        "collapsed_indistinguishable": both,
        "store_cells": int(len(cells["z"])),
        "store_cells_without_a_source_row": int(len(cells["z"]) - np.unique(cell[cell >= 0]).size),
        "usable_rows_without_a_store_cell": int((usable & ~collapsed & (cell < 0)).sum()),
        "cells_compared": int(clean.sum()),
        "orientation_flipped": int((flipped & clean).sum()),
        "z_abs_error": _quantiles(z_err),
        "se_rel_error": _quantiles(se_rel),
        "eaf_cells_compared": int(frequency.sum()),
        "eaf_rel_error": _quantiles(eaf_rel),
        "eaf_cells_beyond_float32_minor_side": int((frequency & ~minor_resolvable).sum()),
        "eaf_missing_in_store": int((clean & np.isfinite(af_src) & ~np.isfinite(eaf)).sum()),
        "sign_disagreements": int(
            ((np.sign(z) != np.sign(z_src)) & (np.abs(z_src) > gates["z_abs"]) & clean).sum()
        ),
        "overflow_cells": int(overflow.sum()),
        "overflow_cells_exact": int(large.sum()),
        "logp_cells_compared": int(p_ok.sum()),
        "logp_abs_error": _quantiles(np.abs(logp_store - logp_src)[p_ok]),
        "statuses": {
            str(status): int(count)
            for status, count in zip(
                *np.unique(cells["association_status"], return_counts=True), strict=True
            )
        },
        "parse_seconds": round(parse_seconds, 2),
        "store_read_seconds": round(read_seconds, 2),
    }
    result["passed"] = _passes(result, gates)
    return result


def _passes(r: dict[str, Any], gates: dict[str, float]) -> dict[str, bool]:
    checks = {
        "every_source_row_on_the_axis": r["rows_not_on_axis"] == 0,
        "unusable_rows_are_missing": r["unusable_rows_with_a_store_cell"] == 0,
        "no_usable_row_lost_except_collapsed": r["usable_rows_without_a_store_cell"] == 0,
        "every_store_cell_has_a_source_row": r["store_cells_without_a_source_row"] == 0,
        "z_within_half_step": r["z_abs_error"]["max"] <= gates["z_abs"],
        "se_within_adr_bound": r["se_rel_error"]["max"] <= gates["se_rel"],
        "eaf_within_half_step": r["eaf_rel_error"]["max"] <= gates["eaf_rel"],
        "no_sign_disagreement": r["sign_disagreements"] == 0,
        "overflow_cells_exact": r["overflow_cells"] == r["overflow_cells_exact"],
    }
    return checks


def _gates(store: Path) -> dict[str, float]:
    encoding = StoreManifest.load(store).encoding.to_manifest()
    z, se, eaf = encoding["z"], encoding["se"], encoding["eaf"]
    kinds = (z["kind"], se["kind"], eaf["kind"])
    if kinds != ("int16_fixed", "int8_residual", "int8_residual"):
        raise SystemExit(f"gates are derived for the v3 int16/int8 residual encoding; got {kinds}")
    return {
        "z_abs": 0.5 / z["scale"] + 1e-6,
        "z_overflow": 32767 / z["scale"],
        "se_rel": 0.01,
        "eaf_rel": math.expm1(eaf["residual_range"] / 254.0) + 1e-4,
    }


def _choose(
    rows: list[dict[str, str]], n_random: int, seed: int, explicit: list[str] | None
) -> dict[str, list[str]]:
    ids = [row["analysis_id"] for row in rows]
    if explicit:
        return {"explicit": explicit}
    chosen = [a for a in ANCHORS if a in ids]
    cases = {r["analysis_id"]: int(r["n_cases"]) for r in rows if r["n_cases"]}
    smallest = min((a for a in cases if cases[a] > 0 and a not in chosen), key=cases.get)
    rest = [a for a in ids if a not in chosen and a != smallest]
    picks = np.random.default_rng(seed).choice(len(rest), size=n_random, replace=False)
    return {
        "anchors": chosen,
        "fewest_cases": [smallest],
        "random": [rest[i] for i in sorted(picks)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="build manifest with analysis_id and checksum (sha256) columns")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-random", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--analyses", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-checksum", action="store_true")
    args = ap.parse_args()

    with open(args.manifest, newline="") as fh:
        manifest = {row["analysis_id"]: row for row in csv.DictReader(fh, delimiter="\t")}
    with open(args.store / "analyses.tsv", newline="") as fh:
        store_rows = list(csv.DictReader(fh, delimiter="\t"))
    in_store = [row["analysis_id"] for row in store_rows]
    missing = [a for a in (args.analyses or []) if a not in in_store]
    if missing:
        raise SystemExit(f"not in the store: {missing}")

    roles = _choose(store_rows, args.n_random, args.seed, args.analyses)
    selected = [a for group in roles.values() for a in group]
    gates = _gates(args.store)
    print(f"{len(selected)} analyses; gates {gates}", flush=True)

    _load_axis(args.store)
    print(f"axis: {_AXIS['n_variants']:,} variants in {_AXIS['load_seconds']:.0f} s", flush=True)

    tasks = []
    for analysis_id in selected:
        source = args.source_dir / f"finngen_R13_{analysis_id.removeprefix(PREFIX)}.gz"
        if not source.exists():
            raise SystemExit(f"{source}: missing")
        digest = None if args.no_checksum else _sha256(source)
        recorded = manifest.get(analysis_id, {}).get("checksum")
        if digest and recorded and digest != recorded:
            raise SystemExit(f"{source}: sha256 {digest} != manifest {recorded}")
        tasks.append({
            "analysis_id": analysis_id, "store": str(args.store), "source": str(source),
            "gates": gates, "source_sha256": digest, "manifest_sha256": recorded,
        })
    print("source checksums verified" if not args.no_checksum else "checksums skipped", flush=True)

    started = time.perf_counter()
    context = multiprocessing.get_context("fork")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for record in pool.map(compare_analysis, tasks):
            failed = [k for k, ok in record["passed"].items() if not ok]
            verdict = "PASS" if not failed else f"FAIL {failed}"
            print(f"{record['analysis_id']:45s} compared {record['cells_compared']:>11,}  "
                  f"z<={record['z_abs_error']['max']:.2e} se<={record['se_rel_error']['max']:.2e} "
                  f"eaf<={record['eaf_rel_error']['max']:.2e}  {verdict}", flush=True)
            results.append(record)

    payload = {
        "store": str(args.store),
        "source_dir": str(args.source_dir),
        "n_variants": _AXIS["n_variants"],
        "gates": gates,
        "seed": args.seed,
        "selection": roles,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "analyses": results,
        **provenance(),
    }
    write_artifact(args.output, payload)
    if not all(all(r["passed"].values()) for r in results):
        raise SystemExit("fidelity gates failed")


if __name__ == "__main__":
    main()
