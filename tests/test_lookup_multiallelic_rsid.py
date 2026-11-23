"""`lookup` by an rsID that names several rows returns every row, on every layout (#465).

A multi-allelic site, or one position stored under both allele orders, gives one
rsID several Store Variant Table rows. Dense `lookup` has always returned all of
them (`indices_by_identifiers`); Ragged returned only the lowest, and a Hybrid's
off-panel half did too. One contract across layouts: every row.

The fixture releases name no rsID, so two rows of a copy are made to share one,
the way `rsid_releases` gives every row one, a Hybrid's Dense Component too.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from rsid_releases import RELEASE_NAMES, with_rsids

from opengwasdb.query import query_store
from opengwasdb.variants.axis import VariantAxis


def _plant(store: Path, pairs: list[tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """Give each pair of shared Variant Indices one rsID; every other row its own."""
    axis = VariantAxis(store)
    try:
        records = axis.all()
    finally:
        axis.close()
    names = {r.alid: f"rs{1000 + r.variant_index}" for r in records}
    planted = {}
    for k, (a, b) in enumerate(pairs):
        rsid = f"rs9{k}"
        names[records[a].alid] = names[records[b].alid] = rsid
        planted[rsid] = (a, b)
    with_rsids(store, names)
    return planted


def _pairs(store: Path) -> list[tuple[int, int]]:
    """Two rows to share an rsID: a Hybrid gets one on-panel and one off-panel pair."""
    n = VariantAxis(store).n_variants
    panel_file = store / "dense" / "dense_to_shared.npy"
    if not panel_file.exists():
        return [(n // 2, n // 2 + 1)]
    panel = np.load(panel_file)
    on = [int(i) for i in panel[len(panel) // 2 :][:2]]
    off = [i for i in range(n // 2, n) if i not in set(int(p) for p in panel)][:2]
    # A Hybrid with no two off-panel variants would leave half the case unasked.
    assert len(on) == 2 and len(off) == 2, (len(on), len(off))
    return [(on[0], on[1]), (off[0], off[1])]


@pytest.fixture(scope="module")
def planted(
    rsid_releases: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, tuple[Path, dict]]:
    """A private copy of each shared release, with the pairs planted in it."""
    private = tmp_path_factory.mktemp("multiallelic-rsid")
    out = {}
    for name, shared in rsid_releases.items():
        path = Path(shutil.copytree(shared, private / name))
        out[name] = (path, _plant(path, _pairs(path)))
    return out


@pytest.mark.parametrize("name", RELEASE_NAMES)
def test_lookup_by_a_multi_allelic_rsid_returns_every_row(
    planted: dict[str, tuple[Path, dict]], name: str
):
    path, pairs = planted[name]
    with query_store(path) as q:
        table = q.variants_table()
        ids = [row["analysis_id"] for _, row in sorted(q.analyses_table().items())]
        for rsid, rows in pairs.items():
            assert {table[i]["rsid"] for i in rows} == {rsid}
            by_alid = q.lookup([table[i]["alid"] for i in rows], ids)
            wanted = {int(v) for v in by_alid["variant_index"]}
            # Both rows carry data, or "returns every row" is asked of one row.
            assert wanted == set(rows), (rsid, rows, wanted)
            found = q.lookup([rsid], ids)
            assert {int(v) for v in found["variant_index"]} == set(rows), (rsid, rows)
            assert len(found["variant_index"]) == len(by_alid["variant_index"])
