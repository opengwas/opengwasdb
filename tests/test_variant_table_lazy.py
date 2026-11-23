"""A query that names no row of the Variant Table never opens it (#465).

In format 0.2.x the Store Variant Table is a BGZF/tabix file only pysam reads and
a Store built before the rsid index names rsids in `index.sqlite`. Opening either
costs a file open and a connection per query, whatever the query asks. Each
question below is answered from the mmap'd search indexes and the arrays alone, in
a fresh interpreter, and the test requires that neither `pysam` nor `sqlite3` was
ever imported. `range_phewas` is the one that needs the table (a position range
is a tabix fetch); it must still answer as before, with pysam loaded.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from residual_stores import copied
from store_reader_digests import build_releases

from opengwasdb.query import query_store
from opengwasdb.variants.axis import VariantAxis, write_variant_axis
from opengwasdb.variants.normalise import CanonicalVariant

ROOT = Path(__file__).resolve().parents[1]
HEAVY = ("pysam", "sqlite3")
RELEASES = [
    f"{layout}_{state}"
    for layout in ("dense", "hybrid", "ragged")
    for state in ("completed", "observed")
]

PROBE = r"""
import hashlib, json, sys
from opengwasdb.query import query_store

def result_digests(result):
    # Not `benchmarks.benchmark_store_comparison`'s: importing that loads pysam.
    return {k: hashlib.sha256(repr(v.tolist()).encode()).hexdigest() for k, v in sorted(result.items())}

path, alid, rsid = sys.argv[1:4]
loaded = lambda: sorted(n for n in ("pysam", "sqlite3") if n in sys.modules)
seen = {}
with query_store(path) as q:
    analyses = q.analyses_table()
    ids = [row["analysis_id"] for _, row in sorted(analyses.items())]
    seen["opened"] = loaded()
    shapes = {
        "analysis": lambda: q.analysis(ids[0]),
        "top_hits": lambda: q.top_hits(threshold=0.2),
        "phewas_alid": lambda: q.phewas(alid),
        "phewas_rsid": lambda: q.phewas(rsid),
        "lookup_alid": lambda: q.lookup([alid], ids),
        "lookup_rsid": lambda: q.lookup([rsid], ids),
    }
    if hasattr(q, "range_by_analysis"):
        shapes["range_by_analysis"] = lambda: q.range_by_analysis(
            alid.split(":")[0], 1, 10**9
        )
    answers = {}
    for name, shape in shapes.items():
        result = shape()
        answers[name] = result_digests(dict(result))
        seen[name] = loaded()
        if name in ("phewas_alid", "lookup_rsid"):
            rows = list(q.resolve(result))
            answers[name + "_rows"] = len(rows)
            seen[name + "_resolved"] = loaded()
print(json.dumps({"loaded": seen, "answers": {k: str(v) for k, v in answers.items()}}))
"""

RANGE_PROBE = r"""
import hashlib, json, sys
from opengwasdb.query import query_store

def result_digests(result):
    return {k: hashlib.sha256(repr(v.tolist()).encode()).hexdigest() for k, v in sorted(result.items())}

path, chromosome = sys.argv[1:3]
with query_store(path) as q:
    digest = str(result_digests(dict(q.range_phewas(chromosome, 1, 10**9))))
print(json.dumps({"pysam": "pysam" in sys.modules, "digest": digest}))
"""


def _rewrite_with_rsids(store: Path, rsid_by_alid: dict[str, str] | None) -> dict[str, str]:
    """Rewrite one Variant Table and its search indexes so every row names an rsid."""
    axis = VariantAxis(store)
    try:
        records = axis.all()
    finally:
        axis.close()
    names = rsid_by_alid or {r.alid: f"rs{1000 + r.variant_index}" for r in records}
    write_variant_axis(
        store,
        [
            CanonicalVariant(r.chromosome, r.position, r.effect_allele, r.other_allele)
            for r in records
        ],
        names,
        [r.source_alid for r in records],
    )
    return names


def _with_rsids(store: Path) -> None:
    """Give a release's rows rsids, a Hybrid's Dense Component too (it resolves its own).

    No fixture release carries one, and a query by rsid on a release that names
    none is an empty answer that proves nothing about where it was resolved.
    """
    names = _rewrite_with_rsids(store, None)
    if (store / "dense" / "variants.tsv.gz").exists():
        _rewrite_with_rsids(store / "dense", names)


@pytest.fixture(scope="module")
def releases(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    built = build_releases(tmp_path_factory.mktemp("lazy-variant-table"))
    private = tmp_path_factory.mktemp("lazy-variant-table-rsids")
    out = {}
    for name, path in built.items():
        out[name] = copied(path, private / name) if (private / name).mkdir() is None else path
        _with_rsids(out[name])
    return out


def _run(code: str, *args: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", code, *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def _asked(path: Path) -> tuple[str, str]:
    """An ALID and an indexed rsid that name a row of the release."""
    # A Hybrid resolves every identifier in its Dense Component first, and an
    # rsid that component's index lacks (an off-panel variant) is the one case
    # that consults `variant_aliases`: ask for a variant the panel holds.
    axis = VariantAxis(path / "dense" if (path / "dense").is_dir() else path)
    try:
        table = axis.all()
    finally:
        axis.close()
    records = [dict(r) for r in table if r.rsid]
    # Fixtures asserted meaningful: a release with no rsid or only long ALIDs would
    # make every answer below an empty one.
    assert records, "the fixture release carries no rsid"
    record = records[len(records) // 2]
    assert len(record["alid"].encode()) <= 32
    return record["alid"], record["rsid"]


@pytest.mark.parametrize("name", RELEASES)
def test_questions_a_search_index_answers_never_load_pysam_or_sqlite(
    releases: dict[str, Path], name: str
):
    alid, rsid = _asked(releases[name])
    probe = _run(PROBE, str(releases[name]), alid, rsid)
    assert probe["loaded"] == {key: [] for key in probe["loaded"]}
    assert len(probe["loaded"]) >= 9, probe["loaded"]
    # The questions were answered, not empty: one row at least, resolved.
    assert probe["answers"]["phewas_alid_rows"] != "0"
    assert probe["answers"]["lookup_rsid_rows"] != "0"


@pytest.mark.parametrize("name", RELEASES)
def test_range_phewas_still_answers_and_does_load_the_table(
    releases: dict[str, Path], name: str
):
    alid, _ = _asked(releases[name])
    probe = _run(RANGE_PROBE, str(releases[name]), alid.split(":")[0])
    assert probe["pysam"] is True
    with query_store(releases[name]) as q:
        result = dict(q.range_phewas(alid.split(":")[0], 1, 10**9))
        want = str(
            {
                k: hashlib.sha256(repr(v.tolist()).encode()).hexdigest()
                for k, v in sorted(result.items())
            }
        )
    assert len(result["variant_index"]) > 0
    assert probe["digest"] == want
