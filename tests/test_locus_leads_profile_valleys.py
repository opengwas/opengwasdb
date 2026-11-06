"""Sampled no-row valleys read from a Store (prototypes/locus_leads, #409).

The fixture Store holds chromosome 1 variants at 100, 200 and 300. Analysis a1 has an
observed Z at 100 and 200, and none at 300.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from prototype_modules import load_prototype

from opengwasdb.query.facade import query_store

profile_valleys = load_prototype("locus_leads", "profile_valleys")

BIN_BP = 100


def _valley(store: Path, bin_index: int) -> dict[str, Any]:
    return {"store": store.parent.name, "analysis_id": "a1", "bin_bp": BIN_BP,
            "chromosome": "1", "bin": bin_index}


def test_a_valley_without_a_store_variant_is_kept_and_counted_as_a_gap(
        dense_store_path: Path) -> None:
    with query_store(dense_store_path) as query:
        a1 = query.analysis("a1", observed_only=True)
        positions = sorted(int(str(r["position"])) for r in query.resolve(a1))
    assert positions == [100, 200], "a1 must be observed at 100 and 200 only"
    # Bins 1, 3 and 5: an observed cell, a missing cell, and no variant of the Store.
    sample = [_valley(dense_store_path, b) for b in (1, 3, 5)]

    cells = profile_valleys.read_valley_cells(dense_store_path.parent.parent, sample)

    assert [(c["bin"], c["variants"], c["observed"], c["imputed"]) for c in cells] == [
        (1, 1, 1, 0), (3, 1, 0, 0), (5, 0, 0, 0)]
    summary = profile_valleys.summarise_valley_cells(cells)["pooled"]
    assert (summary["valleys"], summary["low_z"], summary["missing_cells"],
            summary["no_variant"]) == (3, 1, 1, 1)
    assert summary["share_gap"] == pytest.approx(2 / 3)
    assert summary["stop_rule_trips"]
