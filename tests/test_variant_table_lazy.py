"""A query that names no row of the Variant Table never opens it (#465).

In format 0.2.x the Store Variant Table is a BGZF/tabix file only pysam reads and
a Store built before the rsid index names rsids in `index.sqlite`. Opening either
costs a file open and a connection per query, whatever the query asks. Each
question below is answered from the mmap'd search indexes and the arrays alone, in
a fresh interpreter, and the test requires that neither `pysam` nor `sqlite3` was
ever imported. `range_phewas` was the one that needed the table (a position range
was a tabix fetch); on 0.3.0 the position sidecar answers it (#468), so it loads
neither pysam nor sqlite3. A 0.2 release's range is still a tabix fetch
(`tests/test_range_phewas_0_3_0.py`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from rsid_releases import RELEASE_NAMES

from opengwasdb.store_reader.alid import ALID_WIDTH
from opengwasdb.variants.axis import VariantAxis

ROOT = Path(__file__).resolve().parents[1]
RELEASES = RELEASE_NAMES

PROBE = r"""
import json, sys
from opengwasdb.query import query_store

path, chromosome = sys.argv[1], sys.argv[4]
alids, rsids = json.loads(sys.argv[2]), json.loads(sys.argv[3])
loaded = lambda: sorted(n for n in ("pysam", "sqlite3") if n in sys.modules)
seen = {}
with query_store(path) as q:
    analyses = q.analyses_table()
    ids = [row["analysis_id"] for _, row in sorted(analyses.items())]
    seen["opened"] = loaded()
    shapes = {
        "analysis": lambda: q.analysis(ids[0]),
        "top_hits": lambda: q.top_hits(threshold=5e-4),
    }
    for kind, names in (("alid", alids), ("rsid", rsids)):
        for i, name in enumerate(names):
            shapes[f"phewas_{kind}_{i}"] = lambda n=name: q.phewas(n)
            shapes[f"lookup_{kind}_{i}"] = lambda n=name: q.lookup([n], ids)
    if hasattr(q, "range_by_analysis"):
        shapes["range_by_analysis"] = lambda: q.range_by_analysis(chromosome, 1, 10**9)
    sizes = {}
    for name, shape in shapes.items():
        result = shape()
        sizes[name] = len(result["variant_index"])
        seen[name] = loaded()
        if name.startswith(("phewas_alid_0", "lookup_rsid_0")):
            sizes[name + "_rows"] = len(list(q.resolve(result)))
            seen[name + "_resolved"] = loaded()
    # Once the one question that needed the table: on 0.3.0 the position sidecar answers it.
    ranged = q.range_phewas(chromosome, 1, 10**9)
    sizes["range_phewas"] = len(ranged["variant_index"])
    after = loaded()
print(json.dumps({"loaded": seen, "sizes": sizes, "after_range_phewas": after}))
"""


def _run(code: str, *args: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", code, *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def _asked(path: Path) -> list[dict]:
    """Variants that name a row of the release and carry an rsid: for a Hybrid, one the
    Dense Component's panel holds and one it does not (an off-panel variant, whose rsid
    the Dense Component's own indexes lack)."""
    axis = VariantAxis(path)
    try:
        records = [dict(r) for r in axis.all() if r.rsid]
    finally:
        axis.close()
    panel_file = path / "dense" / "dense_to_shared.npy"
    if panel_file.exists():
        panel = {int(i) for i in np.load(panel_file)}
        on = [r for r in records if r["variant_index"] in panel]
        off = [r for r in records if r["variant_index"] not in panel]
        # Fixtures asserted meaningful: a Hybrid with no off-panel variant would
        # leave the case this test exists for unasked.
        assert on and off, (len(on), len(off))
        chosen = [on[len(on) // 2], off[len(off) // 2]]
    else:
        assert records, "the fixture release carries no rsid"
        chosen = [records[len(records) // 2]]
    for record in chosen:
        assert len(record["alid"].encode()) <= ALID_WIDTH
    return chosen


@pytest.mark.parametrize("name", RELEASES)
def test_questions_a_search_index_answers_never_load_pysam_or_sqlite(
    rsid_releases: dict[str, Path], name: str
):
    chosen = _asked(rsid_releases[name])
    probe = _run(
        PROBE,
        str(rsid_releases[name]),
        json.dumps([r["alid"] for r in chosen]),
        json.dumps([r["rsid"] for r in chosen]),
        chosen[0]["chromosome"],
    )
    assert probe["loaded"] == {key: [] for key in probe["loaded"]}
    assert len(probe["loaded"]) >= 8, probe["loaded"]
    # The questions were answered, not empty: every phewas and lookup found its row.
    sizes = probe["sizes"]
    assert all(n > 0 for key, n in sizes.items() if key.startswith(("phewas", "lookup")))
    assert sizes["phewas_alid_0_rows"] > 0 and sizes["lookup_rsid_0_rows"] > 0
    # `range_phewas` answers (its digests are pinned in tests/golden/store_reader_digests.json,
    # written by the code before #464), and on 0.3.0 without opening the table (#468).
    assert sizes["range_phewas"] > 0
    assert probe["after_range_phewas"] == []


@pytest.mark.parametrize("name", ["hybrid_completed", "hybrid_observed"])
def test_a_hybrid_phewas_names_the_panel_row_without_inverting_the_alid_index(
    rsid_releases: dict[str, Path], name: str
):
    """Naming the one Dense row must not build `variant_index -> ALID slot` for the whole
    shared axis (23.5 M rows on OGS-00004: phewas went from 48 to 503 ms)."""
    from opengwasdb.query import query_store

    on_panel, off_panel = _asked(rsid_releases[name])
    with query_store(rsid_releases[name]) as q:
        for chosen in (on_panel, off_panel):
            assert len(q.phewas(chosen["alid"])["variant_index"]) > 0
        # The 0.3.0 sidecar names a row by two reads and has nothing to invert; a 0.2
        # release's fixed-width index must not have built its inverse either.
        assert getattr(q._variant_axis.alid_index, "_inverse", None) is None
