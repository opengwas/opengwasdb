"""The reads `clump`, `proxies` and `region` make (spec §10; ogld#14), counted where they happen.

Each test counts at the reader's chunk cache: a `Counting` cache is put on one reader instance,
so no reader is subclassed and no module global is patched. "Resolved" is the Stored r the
reader takes, the elements of every `near/<c>/r` and `index/<c>/r` it asks the cache for.
What a query needs comes from an independent greedy over the Test Panel's reference r.
"""

from __future__ import annotations

import tracemalloc
from collections import Counter
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import zarr
from contract import support
from contract.support import alids, panel_data, passing_rows

from ogld.format import dequantise
from ogld.query import clumping, matrices
from ogld.query import proxies as proxies_query
from ogld.store import StoreReader, location
from ogld.store import reader as reader_module


class Counting(location.ChunkCache):
    """A reader's chunk cache that counts the elements of each array it hands the reader."""

    def __init__(self) -> None:
        super().__init__()
        self.taken: Counter[str] = Counter()
        self.gathered: Counter[str] = Counter()
        self.sliced: dict[str, list[tuple[int, int]]] = {}

    def take(self, array: zarr.Array[Any], coords: npt.ArrayLike) -> npt.NDArray[Any]:
        got = super().take(array, coords)
        self.taken[array.path] += len(got)
        self.gathered[array.path] += len(got)
        return got

    def runs(
        self, array: zarr.Array[Any], first: npt.ArrayLike, length: npt.ArrayLike
    ) -> list[npt.NDArray[Any]]:
        got = super().runs(array, first, length)
        self.taken[array.path] += sum(len(run) for run in got)
        pairs = zip(np.asarray(first).tolist(), np.asarray(length).tolist(), strict=True)
        self.sliced.setdefault(array.path, []).extend(pairs)
        return got

    def resolved(self) -> int:
        """The Stored r taken: elements of every Near Band and Proxy Index `r` array."""
        return sum(n for path, n in self.taken.items() if path.endswith("/r"))


def counted(where: object) -> tuple[StoreReader, Counting]:
    """A reader of the store at `where` whose own chunk cache counts what it is asked for."""
    reader = StoreReader.open(str(where))
    cache = Counting()
    reader._cache = cache
    return reader, cache


def reference_greedy(
    score: npt.NDArray[np.float64], r2: float, kb: float
) -> tuple[npt.NDArray[np.int64], int]:
    """plink2's greedy over the Test Store's variants from the reference r, quantised: each
    variant's lead, and the pairs the leads need, each lead with every hit still open within
    W on its chromosome. Ties go to the row, the engine's order (spec §2)."""
    rows = passing_rows()
    pos = panel_data().variants().pos.astype(np.int64)[rows]
    chrom = np.asarray(panel_data().variants().chrom)[rows]
    ref = support.reference_matrix(support.K, tuple(rows.tolist()))
    held = ~np.isnan(ref)
    r = np.full(ref.shape, np.nan)
    r[held] = dequantise(support.quantise(ref[held]))
    lead = np.full(len(rows), -1, dtype=np.int64)
    needed = 0
    for i in np.lexsort((np.arange(len(rows)), -score)):
        if lead[i] >= 0:
            continue
        lead[i] = i
        open_near = (lead < 0) & (chrom == chrom[i]) & (np.abs(pos - pos[i]) < kb * 1000)
        needed += int(open_near.sum())
        lead[open_near & (r[i] * r[i] > r2)] = i
    return lead, needed


#: What the leads need, on the Test Store at f0e90de: the tip's square read resolved 1,495,403
#: pairs at r² 0.1 (64 times as many), and 33,187 Near Band entries are the leads' whole rows
#: and columns at r² 0.01, below t_far.
NEEDED = {0.1: 23_327, 0.01: 7_246}


@pytest.mark.parametrize("r2", sorted(NEEDED))
def test_clump_resolves_at_most_twice_the_pairs_it_needs(r2: float) -> None:
    rows = passing_rows()
    z = np.abs(panel_data().gwas().z)[rows]
    lead, needed = reference_greedy(z, r2, 3000)
    assert needed == NEEDED[r2], "the leads' candidates, as the tip counted them"
    assert len(np.unique(lead)) > 200, "many leads, each reading its own pairs"
    reader, cache = counted(support.TEST_STORE)
    ours = clumping.clump(reader, alids(rows), z, r2=r2, kb=3000)
    assert list(ours.lead) == alids(rows[lead])
    assert cache.resolved() <= 2 * needed, f"resolved {cache.resolved()} for {needed} needed"


#: The Variant Set's string columns (spec §2).
STRING_COLUMNS = ("variants/alid", "variants/a1", "variants/a2")


@pytest.mark.parametrize(("r2", "kb"), [(0.5, 3000), (0.8, 10_000)])
def test_a_warm_proxies_decodes_no_string_column_before_its_rows_are_chosen(
    r2: float, kb: float
) -> None:
    # The reader filters on the positions it keeps; the strings read are the queries' ALIDs,
    # to find them, and then the kept proxies' ALIDs, to name them (ogld#14).
    rows = passing_rows()[::25]
    reader, cache = counted(support.TEST_STORE)
    proxies_query.proxies(reader, alids(rows), r2=r2, kb=kb)
    cache.taken.clear()
    found = proxies_query.proxies(reader, alids(rows), r2=r2, kb=kb)
    stored = panel_data().variants().pos.astype(np.int64)[passing_rows()]
    at_queries = int(np.isin(stored, panel_data().variants().pos[rows]).sum())
    near = panel_data().variants().pos.astype(np.int64)[rows]
    candidates = int((np.abs(stored[:, None] - near[None, :]) <= kb * 1000).sum())
    assert candidates > 20 * len(found), "far more candidates than proxies kept"
    assert len(found) > 0, "and some kept"
    strings = {path: n for path, n in cache.taken.items() if path in STRING_COLUMNS}
    assert strings == {"variants/alid": at_queries + len(found)}


@pytest.mark.parametrize(("centre", "radius"), [(20_771_005, 1_000_000), (21_021_005, 5_000_000)])
def test_a_region_reads_each_row_as_one_contiguous_slice(centre: int, radius: int) -> None:
    reader, cache = counted(support.TEST_STORE)
    got = matrices.region(reader, "22", centre, radius, fill=None)
    # Where each of chr22's rows begins in near/22/r, and which are in the region, from the
    # Test Panel's positions and near_bp alone (spec §4).
    pos = panel_data().variants().pos.astype(np.int64)[passing_rows()]
    chrom = np.asarray(panel_data().variants().chrom)[passing_rows()]
    pos = pos[chrom == "22"]
    later = (pos[None, :] - pos[:, None] <= 3_000_000) & (
        np.arange(len(pos))[None, :] > np.arange(len(pos))[:, None]
    )
    begins = np.concatenate([[0], np.cumsum(later.sum(axis=1))[:-1]])
    inside = np.flatnonzero(np.abs(pos - centre) <= radius)
    assert len(inside) > 50, "a region of many rows"
    expected = [(int(begins[i]), int(later[i, inside].sum())) for i in inside]
    assert sum(n for _, n in expected) > 1000, "holding many Near Band pairs"
    assert got.r.shape == (len(inside), len(inside))
    assert cache.sliced.get("near/22/r") == expected, "one slice per row, inside the region"
    assert cache.gathered["near/22/r"] == 0, "and no Near Band pair gathered one by one"


@pytest.mark.parametrize("query", ["matrix", "region"])
def test_a_matrix_query_holds_one_float64_copy_of_its_result(
    query: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The matrix is written in place and filled in place (ogld#14): never a second n × n
    # float64 beside the result. A step of 97 pairs keeps the working memory, which the step
    # bounds (spec §10), out of the measurement.
    monkeypatch.setattr(reader_module, "_PAIRS_PER_STEP", 97)
    reader = StoreReader.open(str(support.TEST_STORE))
    rows = passing_rows()[468:]  # chr22, so pair_r's per-chromosome block would be n × n

    def run() -> npt.NDArray[np.float64]:
        if query == "matrix":
            return matrices.matrix(reader, alids(rows), fill=0.0).r
        return matrices.region(reader, "22", 21_021_005, 5_000_000, fill=0.0).r

    run()  # warm: the chunks are read and kept before anything is measured
    tracemalloc.start()
    try:
        r = run()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert r.shape[0] > 300, "a matrix of many rows"
    assert not np.isnan(r).any(), "every unstored pair filled"
    # Two float64 copies alone would be twice the result; the rest is the mask, one step's
    # pairs and the variant table's strings.
    assert peak < 2 * r.nbytes, f"peak {peak:,} bytes for a {r.nbytes:,}-byte result"
