"""The reads `clump`, `proxies` and `region` make (spec §10; ogld#14), counted where they happen.

Each test counts at the reader's chunk cache: a `Counting` cache is put on one reader instance,
so no reader is subclassed and no module global is patched. "Resolved" is the Stored r the
reader takes, the elements of every `near/<c>/r` it asks the cache for. What a query needs comes
from an independent greedy over the Test Panel's reference r.
"""

from __future__ import annotations

import os
import tracemalloc
from collections import Counter
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import zarr
from contract import support
from contract.support import alids, panel_data, passing_rows
from store_copies import TEST_STORE, store_reads

import ogld
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
        """The Stored r taken: elements of every Near Band `r` array."""
        return sum(n for path, n in self.taken.items() if path.endswith("/r"))


def counted(where: object) -> tuple[StoreReader, Counting]:
    """A reader of the store at `where` whose own chunk cache counts what it is asked for."""
    reader = StoreReader.open(str(where))
    cache = Counting()
    # One cache: the reader and the variant table it composes hold the same `OpenedStore`, so this
    # one assignment is what both halves read through (#94).
    reader._cache = cache
    return reader, cache


def reference_greedy(
    score: npt.NDArray[np.float64], r2: float, kb: float
) -> tuple[npt.NDArray[np.int64], int, int]:
    """plink2's greedy over the Test Store's variants from the reference r, quantised: each
    variant's lead; the pairs the leads need, each lead with every hit still open within W
    on its chromosome; and every partner within far_bp whose code reaches the floor, which is
    the rows' entries at the far code floor (spec §5). Ties go to the row, the engine's order."""
    rows = passing_rows()
    pos = panel_data().variants().pos.astype(np.int64)[rows]
    chrom = np.asarray(panel_data().variants().chrom)[rows]
    ref = support.reference_matrix(support.K, tuple(rows.tolist()))
    held = ~np.isnan(ref)
    r = np.full(ref.shape, np.nan)
    r[held] = support.stored_at(ref[held])
    code = np.zeros(ref.shape, dtype=np.int64)
    code[held] = np.abs(support.codes_at(ref[held]))
    lead = np.full(len(rows), -1, dtype=np.int64)
    needed = indexed = 0
    for i in np.lexsort((np.arange(len(rows)), -score)):
        if lead[i] >= 0:
            continue
        lead[i] = i
        same = chrom == chrom[i]
        open_near = (lead < 0) & same & (np.abs(pos - pos[i]) < kb * 1000)
        needed += int(open_near.sum())
        far = same & (np.abs(pos - pos[i]) <= 10_000_000) & (code[i] >= support.FLOOR)
        indexed += int(far.sum()) - int(far[i])
        lead[open_near & (r[i] * r[i] > r2)] = i
    return lead, needed, indexed


#: What the leads need, on the Test Store at × 511 (ogld#16), from the reference. The square
#: read at aa5d4d7 resolved 1,489,250 pairs at r² 0.1 and took 1,587,155 Stored r (68 times
#: as many). At × 32767 (f0e90de) the leads needed 23,327 and 7,246.
NEEDED = {0.1: 23_228, 0.01: 7_311}


@pytest.mark.parametrize("r2", sorted(NEEDED))
def test_clump_reads_one_partner_row_per_lead_and_no_near_band_or_index_element(
    r2: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    # From 0.2.0 a lead's partners are its Partner Rows row, one frame read and one row's
    # unshuffle (partner-rows §7, §8.2): no element of `near/` is read at any r2, and
    # no hit another lead took is read, so the reads are one a lead.
    rows = passing_rows()
    z = np.abs(panel_data().gwas().z)[rows]
    lead, needed, _ = reference_greedy(z, r2, 3000)
    assert needed == NEEDED[r2], "the leads' candidates, as the tip counted them"
    assert len(np.unique(lead)) > 200, "many leads, each reading its own pairs"
    reader, cache = counted(support.TEST_STORE)
    read = reader.partner_row
    asked: list[tuple[str, int]] = []

    def counted_row(chrom: str, local_row: int) -> Any:
        asked.append((chrom, local_row))
        return read(chrom, local_row)

    monkeypatch.setattr(reader, "partner_row", counted_row)
    with store_reads() as keys:
        ours = clumping.clump(reader, alids(rows), z, r2=r2, kb=3000)
    assert list(ours.lead) == alids(rows[lead])
    assert len(asked) == len(np.unique(lead)), "one Partner Rows row per lead, and no other"
    assert len(set(asked)) == len(asked), "no row is read twice"
    assert cache.resolved() == 0, "no Near Band or Proxy Index code is read at any r2"
    offending = [key for key in keys if key.startswith("near/")]
    assert offending == [], f"clump read {offending[:5]}"


#: The Variant Set's string columns (spec §2).
STRING_COLUMNS = ("variants/alid", "variants/a1", "variants/a2")


@pytest.mark.parametrize(("r2", "kb"), [(0.5, 3000), (0.8, 10_000)])
def test_a_warm_proxies_decodes_no_string_column_before_its_rows_are_chosen(
    r2: float, kb: float
) -> None:
    # The reader filters on the positions it keeps. The queries are found in the ALID index,
    # which reads no string column (ogld#42); the strings read are the kept proxies' ALIDs, to
    # name them (ogld#14).
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
    assert strings == {"variants/alid": len(found)}
    assert at_queries > 0, "the queries' own rows are there to be read, and are not"


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


@pytest.mark.parametrize("one_against_many", [False, True])
def test_r_of_pairs_holds_what_its_docstring_states(
    one_against_many: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The result, 8 bytes a pair, and one step of pairs, 64 bytes each, with its index rows
    # (ogld#14): the broadcast is never copied, so one row against many costs its result.
    monkeypatch.setattr(reader_module, "_PAIRS_PER_STEP", 997)
    reader = StoreReader.open(str(support.TEST_STORE))
    rng = np.random.default_rng(14)
    second = rng.integers(0, 1056, 150_000)
    first = np.array([500]) if one_against_many else rng.integers(0, 1056, 150_000)
    reader.r_of_pairs(first, second)
    tracemalloc.start()
    try:
        r = reader.r_of_pairs(first, second)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak <= 1.25 * r.nbytes + 64 * 1024, f"peak {peak:,} for a {r.nbytes:,}-byte result"


def test_proxies_read_the_partner_rows_and_nothing_of_near_or_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`rows_of` answers from the ALID index, and from 0.2.0 a variant's proxies are its Partner
    Rows row: no `near/` array of any kind, and none of its metadata (partner-rows §8.3, §8.4;
    ld-store §10). The frames are the one chunk file it reads."""
    names = ogld.open(TEST_STORE)._reader.alids_of(np.array([500, 520])).tolist()
    preads: list[int] = []
    original = os.pread

    def recording_pread(fd: int, length: int, offset: int) -> bytes:
        preads.append(length)
        return original(fd, length, offset)

    monkeypatch.setattr(os, "pread", recording_pread)
    store = ogld.open(TEST_STORE)
    with store_reads() as keys:
        assert (store._reader.rows_of(names) >= 0).all()
        store.proxies(names, r2=0.8, kb=3000)
    assert preads, "a local frame is one pread of the chunk file (partner-rows §7)"
    offending = [key for key in keys if key.startswith("near/")]
    assert offending == [], f"proxies read {offending}"
