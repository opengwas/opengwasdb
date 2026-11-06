"""Are the prominence rule's empty valleys low |z|, or gaps in coverage? (#409)

A bin with no 5e-4 tier row has envelope 0 whether the Analysis's Z there is
below the tier's floor or its cells are missing. For a seeded sample of the
valleys with no tier row, the Store's cells in the bin are read for that
Analysis: a valley with an observed cell is low |z|; one whose variants carry
no observed Z, or that holds no variant of the Store at all, is a gap the
envelope cannot tell from a valley.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.query.facade import query_store
from opengwasdb.top_hits import z_critical
from opengwasdb.variants import VariantAxis

SAMPLE_SIZE = 200
SAMPLE_SEED = 409
# A no-row bin cannot hold an observed |z| at or above the profile tier's floor.
PROFILE_TIER = 5e-4

DEFINITION = {
    "sample": f"per bin width B, up to {SAMPLE_SIZE} of the distinct (Store, Analysis, "
              "chromosome, bin) valleys with no tier row, over every prominence rule at full "
              f"density, drawn by random.Random({SAMPLE_SEED}).sample from their sorted list",
    "cells": "every variant of the Store's Variant Index in the bin [bin x B, (bin + 1) x B - "
             "1], read for the Analysis through the query facade's lookup",
    "low_z": "the bin holds at least one observed cell (all below the tier's floor, |z| < "
             f"{z_critical(PROFILE_TIER):.4f})",
    "missing_cells": "the bin holds Store variants but none with an observed cell for the "
                     "Analysis (missing, or imputed only)",
    "no_variant": "the Store's Variant Index holds no variant in the bin",
    "share_gap": "(missing_cells + no_variant) / sampled valleys",
    "stop_rule": "more than half of a sample's valleys are gaps: the profile cannot tell a "
                 "valley from a gap in coverage",
}


def sample_valleys(valleys: list[tuple[str, str, int, str, int]],
                   size: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> list[dict[str, Any]]:
    """Up to `size` distinct (Store, Analysis, B, chromosome, bin) valleys per B, seeded."""
    sample = []
    for bin_bp in sorted({v[2] for v in valleys}):
        pool = sorted({v for v in valleys if v[2] == bin_bp})
        drawn = sorted(random.Random(seed).sample(pool, min(size, len(pool))))
        sample += [{"store": s, "analysis_id": a, "bin_bp": b, "chromosome": c, "bin": n}
                   for s, a, b, c, n in drawn]
    return sample


def read_valley_cells(stores_dir: Path, sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each sampled valley's Store cells for its Analysis: variants, observed and imputed."""
    out = []
    for store_id in sorted({v["store"] for v in sample}):
        path = stores_dir / store_id / "store.opengwasdb"
        query, axis = query_store(path), VariantAxis(path)
        try:
            out += [valley | _cells(query, axis, valley)
                    for valley in sample if valley["store"] == store_id]
        finally:
            axis.close()
            query.close()
    order = {(v["store"], v["analysis_id"], v["bin_bp"], v["chromosome"], v["bin"]): i
             for i, v in enumerate(sample)}
    return sorted(out, key=lambda v: order[(v["store"], v["analysis_id"], v["bin_bp"],
                                            v["chromosome"], v["bin"])])


def _cells(query: Any, axis: VariantAxis, valley: dict[str, Any]) -> dict[str, Any]:
    start = valley["bin"] * valley["bin_bp"]
    alids = [r.alid for r in axis.range(valley["chromosome"], start, start + valley["bin_bp"] - 1)]
    if not alids:
        return {"variants": 0, "observed": 0, "imputed": 0, "max_observed_abs_z": None}
    found = query.lookup(alids, [valley["analysis_id"]])
    status = np.asarray(found["association_status"], dtype=object)
    observed = np.abs(np.asarray(found["z"], dtype="float64"))[status == "observed"]
    if observed.size and observed.max() >= z_critical(PROFILE_TIER):
        raise ValueError(f"{valley}: an observed |z| of {observed.max():.3f} in a bin the "
                         "profile holds no tier row in")
    return {"variants": len(alids), "observed": int(observed.size),
            "imputed": int(np.sum(status == "imputed")),
            "max_observed_abs_z": float(observed.max()) if observed.size else None}


def summarise_valley_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Shares of low-|z| valleys and gaps, per bin width and pooled, and of the cells."""
    groups = {f"{b}": [c for c in cells if c["bin_bp"] == b]
              for b in sorted({c["bin_bp"] for c in cells})} | {"pooled": cells}
    return {name: _summary(group) for name, group in groups.items()}


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(cells)
    low = sum(c["observed"] > 0 for c in cells)
    no_variant = sum(c["variants"] == 0 for c in cells)
    missing = n - low - no_variant
    gap = _share(missing + no_variant, n)
    return {
        "valleys": n, "low_z": low, "missing_cells": missing, "no_variant": no_variant,
        "missing_cells_with_an_imputed_cell": sum(c["observed"] == 0 < c["imputed"]
                                                  for c in cells),
        "share_low_z": _share(low, n), "share_gap": gap,
        "stop_rule_trips": gap is not None and gap > 0.5,
        "cells": _cell_counts(cells),
    }


def _cell_counts(cells: list[dict[str, Any]]) -> dict[str, int]:
    variants = sum(c["variants"] for c in cells)
    observed = sum(c["observed"] for c in cells)
    imputed = sum(c["imputed"] for c in cells)
    return {"variants": variants, "observed": observed, "imputed": imputed,
            "missing": variants - observed - imputed}
