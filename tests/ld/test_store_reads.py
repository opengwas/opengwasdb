"""Bounded reads (spec §10; ogld#4): a reader takes only the chunks a request needs, reads a
re-sharded store the same, and refuses whatever a lost chunk reads as.

The committed Test Store holds each array in one shard of one inner chunk, so every read of it
touches every chunk. These tests read a copy re-sharded into small chunks, where a bounded read
and a whole-array read differ, and are expected to: each asserts the copy has many chunks first.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import shutil
import threading
import tracemalloc
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import zarr
from contract import support
from contract.support import RangeHandler
from numpy.dtypes import StringDType
from store_copies import TEST_STORE, copy_store, resharded

import ogld
from ogld.format import InputError, Refusal, RefusalReason, judge_request, quantise
from ogld.query import clumping
from ogld.query import proxies as proxies_query
from ogld.store import StoreReader, location, reader
from ogld.store.validation import validate_group

#: Inner chunks of 512 elements, 4 to a shard: chr22's Near Band is 136 chunks in 34 shards.
INNER, PER_SHARD = 512, 4


@pytest.fixture(scope="module")
def small_chunks(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return resharded(tmp_path_factory.mktemp("resharded") / "store", INNER, PER_SHARD)


def whole(location: Path | str) -> ogld.LDMatrix:
    store = ogld.open(location)
    rows = np.arange(store.provenance["n_variants"])
    return store.matrix(list(StoreReader.open(location).variants(rows).alid))


def shards(store: Path, path: str) -> list[Path]:
    found = sorted((store / path / "c").iterdir(), key=lambda p: int(p.name))
    assert len(found) > 4, f"{path} must span several shards for this to mean anything"
    return found


def test_a_re_sharded_store_reads_the_same_as_the_committed_one(small_chunks: Path) -> None:
    near = zarr.open_array(small_chunks / "near" / "22" / "r", mode="r")
    assert near.chunks == (INNER,)
    assert near.shape[0] > 50 * INNER, "the copy's Near Band must span many chunks"
    np.testing.assert_array_equal(whole(small_chunks).r, whole(TEST_STORE).r)


def reference_pairs(rows: np.ndarray) -> np.ndarray:
    """What the store must answer among its `rows`, from the Test Panel's reference r at k = 10:
    the A1-signed code over 32,767 where the store holds the pair, NaN where it does not."""
    names, _ = support.oriented()
    panel = {str(name): i for i, name in enumerate(names)}
    variants = StoreReader.open(TEST_STORE).variants(rows)
    ref = support.reference_matrix(support.K, tuple(panel[a] for a in variants.alid))
    gap = np.abs(variants.position[:, None] - variants.position[None, :])
    codes = quantise(np.where(np.isnan(ref), 0.0, ref)).astype(np.int32)
    held = ~np.isnan(ref) & ((gap <= 3_000_000) | (np.abs(codes) >= support.FLOOR))
    expected = np.where(held, codes / 32767, np.nan)
    np.fill_diagonal(expected, 1.0)
    return expected


@pytest.mark.parametrize("budget", [0, 1 << 28])
def test_a_matrix_read_a_few_pairs_at_a_time_is_the_reference(
    small_chunks: Path, budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # pair_r resolves 97 pairs a step, so this matrix takes over twenty steps; a budget of 0
    # keeps no chunk, so every step fetches its chunks again. Each must give the reference r.
    rows = np.arange(0, 1056, 7)
    monkeypatch.setattr(location, "CACHE_BYTES", budget)
    monkeypatch.setattr(reader, "_PAIRS_PER_STEP", 97)
    expected = reference_pairs(rows)
    stored = (np.isfinite(expected).sum() - len(rows)) // 2
    assert stored > 20 * 97, "the matrix must take many steps for this to mean anything"
    assert np.isnan(expected).any(), "and hold unstored pairs"
    np.testing.assert_array_equal(StoreReader.open(small_chunks).pair_r(rows), expected)


@pytest.mark.parametrize("budget", [0, 1 << 28])
def test_index_and_band_rows_read_the_same_from_any_layout(
    small_chunks: Path, budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = np.arange(0, 1056, 7)
    expected = StoreReader.open(TEST_STORE)
    monkeypatch.setattr(location, "CACHE_BYTES", budget)
    got = StoreReader.open(small_chunks)
    for read in ("index", "band"):
        mine, theirs = getattr(got, read)(rows), getattr(expected, read)(rows)
        assert len(theirs.partner) > 1000, f"{read} must return many partners"
        np.testing.assert_array_equal(mine.indptr, theirs.indptr)
        np.testing.assert_array_equal(mine.partner, theirs.partner)
        np.testing.assert_array_equal(mine.r, theirs.r)


def stored_positions() -> tuple[np.ndarray, np.ndarray]:
    """The chromosome and position of each Test Store row, from the Test Panel."""
    rows = support.passing_rows()
    variants = support.panel_data().variants()
    return np.asarray(variants.chrom)[rows], variants.pos.astype(np.int64)[rows]


def test_r_of_pairs_is_the_reference_r_of_each_pair_and_of_one_row_against_many(
    small_chunks: Path,
) -> None:
    rows = np.arange(1056)
    expected = reference_pairs(rows)
    chrom, pos = stored_positions()
    rng = np.random.default_rng(14)
    i, j = rng.integers(0, len(rows), 6000), rng.integers(0, len(rows), 6000)
    i[:10] = j[:10]
    # The Far Pairs are rare: every one the reference holds is added, in both orders.
    far_i, far_j = np.nonzero(~np.isnan(expected) & (np.abs(pos[:, None] - pos) > 3_000_000))
    i, j = np.concatenate([i, far_i]), np.concatenate([j, far_j])
    gap = np.abs(pos[rows[i]] - pos[rows[j]])
    same = chrom[rows[i]] == chrom[rows[j]]
    want = expected[i, j]
    assert (~np.isnan(want) & same & (gap > 3_000_000)).sum() > 10, "Far Pairs among them"
    assert (np.isnan(want) & same & (gap <= 10_000_000)).sum() > 100, "and the Bound"
    assert (~same).sum() > 1000, "and pairs across chromosomes"
    reader = StoreReader.open(small_chunks)
    np.testing.assert_array_equal(reader.r_of_pairs(rows[i], rows[j]), want)
    np.testing.assert_array_equal(reader.r_of_pairs(rows[3:4], rows), expected[3])


@pytest.mark.parametrize(
    ("r2", "kb", "listed"),
    [(0.05, 10_000, False), (support.T_FAR, 3000, True), (0.01, 3000, False), (0.0, 500, True)],
)
def test_partners_are_the_reference_pairs_at_r2_within_the_window(
    small_chunks: Path, r2: float, kb: float, listed: bool
) -> None:
    every = np.arange(1056)
    expected = reference_pairs(every)
    chrom, pos = stored_positions()
    rows = np.concatenate([np.arange(0, 1056, 11), [33, 33]])
    among = np.arange(0, 1056, 3) if listed else None
    got = StoreReader.open(small_chunks).partners(rows, r2=r2, window_bp=kb * 1000, among=among)
    for k, row in enumerate(rows):
        r = expected[row]
        keep = ~np.isnan(r) & (r * r >= r2) & (np.abs(pos - pos[row]) <= kb * 1000)
        keep &= (chrom == chrom[row]) & (every != row)
        if among is not None:
            keep &= np.isin(every, among)
        part = slice(got.indptr[k], got.indptr[k + 1])
        np.testing.assert_array_equal(got.partner[part], every[keep])
        np.testing.assert_array_equal(got.r[part], r[keep])
        np.testing.assert_array_equal(got.distance[part], np.abs(pos[keep] - pos[row]))
    assert len(got.partner) > 500, "many partners in all"


@pytest.mark.parametrize(
    ("r2", "window_bp", "reason"),
    [
        (0.01, 3_000_000.9, RefusalReason.THRESHOLD_BELOW_T_FAR),
        (0.01, 3_000_001, RefusalReason.THRESHOLD_BELOW_T_FAR),
        (0.5, 10_000_000.5, RefusalReason.WINDOW_OVER_FAR),
        (0.5, -1.0, RefusalReason.INVALID),
        (1.5, 1000.0, RefusalReason.INVALID),
    ],
)
def test_partners_refuses_what_judge_request_refuses(
    r2: float, window_bp: float, reason: RefusalReason
) -> None:
    # One rule (spec §9): the reader asks ogld.format, so a window 0.9 bp past near_bp below
    # t_far is refused there as it is by judge_request, not answered to ⌊W⌋ (ogld#14).
    t_far = StoreReader.open(TEST_STORE).provenance["t_far"]
    judged = judge_request(r2=r2, kb=window_bp / 1000, t_far=t_far, near_bp=3_000_000,
                           far_bp=10_000_000)  # fmt: skip
    assert judged.refusal is reason, "judge_request refuses it so"
    with pytest.raises(Refusal) as refused:
        StoreReader.open(TEST_STORE).partners(np.array([10, 600]), r2=r2, window_bp=window_bp)
    assert refused.value.reason is reason


def test_partners_answers_exactly_near_bp_below_t_far_and_far_bp_above() -> None:
    reader = StoreReader.open(TEST_STORE)
    rows = np.array([10, 600])
    assert len(reader.partners(rows, r2=0.01, window_bp=3_000_000).indptr) == 3
    assert len(reader.partners(rows, r2=0.5, window_bp=10_000_000).indptr) == 3


def test_partners_refuses_a_list_that_is_not_ascending_distinct_rows() -> None:
    reader = StoreReader.open(TEST_STORE)
    rows = np.array([10, 600])
    with pytest.raises(ValueError, match="ascending, distinct"):
        reader.partners(rows, r2=0.5, window_bp=1000, among=np.array([5, 5]))
    with pytest.raises(ValueError, match="Absent ALID"):
        reader.partners(rows, r2=0.5, window_bp=1000, among=np.array([-1, 5]))


def test_a_region_reads_the_same_from_any_layout(small_chunks: Path) -> None:
    # One run per row, across the copy's 512-code chunks; a whole chromosome is wider than
    # far_bp, so its Far Pairs are scattered from the index and the rest is NaN.
    reader = StoreReader.open(small_chunks)
    for start, stop in ((468, 1056), (0, 468), (500, 520)):
        np.testing.assert_array_equal(
            reader.region_r(start, stop), reference_pairs(np.arange(start, stop))
        )


@pytest.mark.parametrize(("start", "stop"), [(6, 39), (36, 221), (840, 1020)])
def test_a_region_keeps_none_of_its_rows_far_pairs_that_leave_it(start: int, stop: int) -> None:
    # Each span has rows with a Far Pair outside it, which the scatter must not place on a
    # row inside it; (840, 1020) also holds one inside (spec §10; ogld#14).
    every = np.arange(1056)
    expected = reference_pairs(every)
    _, pos = stored_positions()
    far = ~np.isnan(expected) & (np.abs(pos[:, None] - pos) > 3_000_000)
    inside = (every >= start) & (every < stop)
    assert far[inside][:, ~inside].any(), "a Far Pair leaving the region"
    got = StoreReader.open(TEST_STORE).region_r(start, stop)
    np.testing.assert_array_equal(got, expected[start:stop, start:stop])


def test_a_region_is_a_span_of_one_chromosome() -> None:
    reader = StoreReader.open(TEST_STORE)
    assert reader.region_r(5, 5).shape == (0, 0)
    with pytest.raises(ValueError, match="leave their chromosome"):
        reader.region_r(460, 470)
    for start, stop in ((1000, 1100), (-1, 3), (5, 4)):
        with pytest.raises(ValueError, match="not a span"):
            reader.region_r(start, stop)


def retained(cache: location.ChunkCache) -> int:
    """The bytes the cache's kept chunks actually hold alive: each one's owning buffer, once."""
    owners = {}
    for chunk, _ in cache._kept.values():
        owner = chunk if chunk.base is None else chunk.base
        owners[id(owner)] = owner.nbytes
    return sum(owners.values())


def test_the_cache_holds_no_more_than_its_budget(small_chunks: Path) -> None:
    # One read of 40 neighbouring chunks is one run; keeping 4 of them must not keep the run.
    near = zarr.open_array(small_chunks / "near" / "22" / "r", mode="r")
    budget = 4 * INNER * near.dtype.itemsize
    cache = location.ChunkCache(budget)
    got = cache.take(near, np.arange(0, 40 * INNER))
    np.testing.assert_array_equal(got, np.asarray(near[: 40 * INNER]))
    assert len(cache._kept) == 4, "the cache must keep a few chunks for this to mean anything"
    assert retained(cache) <= budget


def measured(chunk: np.ndarray) -> int:
    """The bytes a chunk's storage takes, measured by tracemalloc as a copy of it allocates
    them: its buffer, and for numpy's variable-width strings their arena, which nbytes omits."""
    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    copy = chunk.copy()
    taken = tracemalloc.get_traced_memory()[0] - before
    del copy
    if not tracing:
        tracemalloc.stop()
    return taken


def text(i: int, length: int) -> str:
    """A string of `length` characters, ALID-like when long enough."""
    return (
        f"{i:0{length}d}"[-length:]
        if length < 12
        else (f"22:{i}:" + "A" * length)[: length - 2] + ":C"
    )


#: Chunks of 512 strings: all 8 characters, held inside the array; all 200, each in the arena;
#: and every tenth 200 among 8s. Each budget keeps a few chunks.
STRINGS = {
    "short 8": (lambda i: text(i, 8), 40_000),
    "long 200": (lambda i: text(i, 200), 450_000),
    "mixed": (lambda i: text(i, 200 if i % 10 == 0 else 8), 100_000),
}


@pytest.mark.parametrize("kind", list(STRINGS))
def test_string_chunks_count_an_upper_bound_on_what_they_hold(tmp_path: Path, kind: str) -> None:
    # numpy's variable-width strings keep each string over 15 bytes in an arena that nbytes
    # (16 bytes an element) leaves out, so a budget counted by nbytes would be no bound.
    make, budget = STRINGS[kind]
    names = np.array([make(i) for i in range(20 * 512)], dtype=StringDType())
    group = zarr.open_group(tmp_path / "g", mode="w")
    alid = group.create_array("alid", shape=names.shape, dtype=str, chunks=(512,), fill_value="")
    alid[:] = names
    cache = location.ChunkCache(budget)
    np.testing.assert_array_equal(cache.take(alid, np.arange(len(names))), names)
    kept = list(cache._kept.values())
    assert len(kept) >= 2, "the cache must keep chunks for this to mean anything"
    assert all(chunk.dtype.kind == "T" for chunk, _ in kept), "kept as variable-width strings"
    for chunk, counted in kept:
        assert measured(chunk) <= counted, "a chunk counts at least what it holds"
    assert sum(measured(chunk) for chunk, _ in kept) <= cache._bytes <= budget
    assert cache._bytes == sum(counted for _, counted in kept)


def test_threads_missing_one_chunk_store_it_once(small_chunks: Path) -> None:
    # 16 threads each read 8 chunks, overlapping their neighbours', 54 chunks in all, into a
    # 24-chunk budget: each read fits the budget, so threads race on chunks still kept, and
    # the cache evicts as they do.
    near = zarr.open_array(small_chunks / "near" / "22" / "r", mode="r")
    everything = np.asarray(near[:])
    budget = 24 * INNER * near.dtype.itemsize
    cache = location.ChunkCache(budget)
    start = threading.Barrier(16)
    wrong: list[int] = []

    def read(first: int) -> None:
        start.wait()
        coords = np.arange(first * INNER, (first + 8) * INNER, 5)
        if not np.array_equal(cache.take(near, coords), everything[coords]):
            wrong.append(first)

    workers = [threading.Thread(target=read, args=(3 * i,)) for i in range(16)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert wrong == []
    assert len(cache._kept) == 24, "the cache must be full, having evicted 30 of 54 chunks"
    counted = [size for _, size in cache._kept.values()]
    assert cache._bytes == sum(counted) == retained(cache)
    assert cache._bytes <= budget


def test_a_lost_shard_refuses_the_pairs_it_held_and_no_others(
    small_chunks: Path, tmp_path: Path
) -> None:
    copy = tmp_path / "lost"
    shutil.copytree(small_chunks, copy)
    lost = shards(copy, "near/22/r")[-1]
    lost.unlink()
    store, chr22 = ogld.open(copy), StoreReader.open(copy).rows_in("22", 0, 10**9)
    names = list(StoreReader.open(copy).variants(chr22).alid)
    # The first rows' codes sit in the first shard: answered, as the committed store answers.
    np.testing.assert_array_equal(
        store.matrix(names[:20]).r, ogld.open(TEST_STORE).matrix(names[:20]).r
    )
    with pytest.raises(ValueError, match=r"near/22/r read \d+ element\(s\) .* -32768"):
        store.matrix(names[-20:])


@pytest.mark.parametrize(
    ("path", "fill"),
    [
        ("variants/alid", '""'),
        ("variants/position", "0"),
        ("index/22/partner", "-1"),
        ("index/22/r", "-32768"),
        ("near/22/indptr", "-1"),
    ],
)
def test_a_lost_chunk_of_any_array_is_refused_never_read_as_data(
    small_chunks: Path, tmp_path: Path, path: str, fill: str
) -> None:
    # An ALID that read as "" would make a present variant Absent; a partner of −1 would name
    # the last variant. Each is refused, naming the array.
    copy = tmp_path / "lost"
    shutil.copytree(small_chunks, copy)
    every = list(StoreReader.open(copy).variants(np.arange(1056)).alid)
    for shard in sorted((copy / path / "c").iterdir()):
        shard.unlink()
    message = rf"{path} read \d+ element\(s\) no LD Store holds there, as its fill value {fill} is"
    with pytest.raises(ValueError, match=message):
        ogld.open(copy).matrix(every)


def fill_findings(store: Path) -> list[ogld.Finding]:
    root = zarr.open_group(store, mode="r")
    return validate_group(root, root.attrs["ld_store"], ["envelope.fill"])["envelope.fill"]


def set_fill(store: Path, path: str, fill: object) -> None:
    meta = store / path / "zarr.json"
    doc = json.loads(meta.read_text())
    doc["fill_value"] = fill
    meta.write_text(json.dumps(doc))


def test_a_partner_whose_lost_chunk_would_read_as_row_0_is_refused(
    small_chunks: Path, tmp_path: Path
) -> None:
    # The review's case: fill 0 (zarr's int32 default) and a lost shard read as partner row 0,
    # chr22's first variant, and proxies named it 377 times in place of the true partners.
    copy = tmp_path / "store"
    shutil.copytree(small_chunks, copy)
    set_fill(copy, "index/22/partner", 0)
    shards(copy, "index/22/partner")[1].unlink()
    names = list(StoreReader.open(TEST_STORE).variants(np.arange(468, 1056, 15)).alid)
    with pytest.raises(
        ValueError, match=r"index/22/partner has the fill value 0, which valid data"
    ):
        ogld.open(copy).matrix(names)
    assert fill_findings(copy), "validate fails what open refuses"


def test_every_fill_is_judged_before_any_element_is_decoded(tmp_path: Path) -> None:
    # A lost indptr chunk and a partner fill of 0: the fill is refused first, so no element of
    # the chromosome was decoded from an array whose lost chunk could read as data.
    copy = copy_store(tmp_path / "store")
    set_fill(copy, "index/22/partner", 0)
    (copy / "near" / "22" / "indptr" / "c" / "0").unlink()
    with pytest.raises(ValueError, match=r"index/22/partner has the fill value 0"):
        ogld.open(copy).region("22", 30_000_000, 1_000_000)


@pytest.mark.parametrize(
    ("path", "fill"),
    [("variants/position", 5), ("variants/alid", "22:1:A:C"), ("near/22/indptr", 0)],
)
def test_an_array_whose_fill_valid_data_can_hold_is_refused(
    tmp_path: Path, path: str, fill: object
) -> None:
    copy = copy_store(tmp_path / "store")
    set_fill(copy, path, fill)
    with pytest.raises(ValueError, match=rf"{path} has the fill value"):
        ogld.open(copy).region("22", 30_000_000, 1_000_000)
    assert fill_findings(copy), "validate fails what open refuses"


def test_an_r_array_whose_lost_chunk_would_read_as_a_number_is_refused(tmp_path: Path) -> None:
    copy = copy_store(tmp_path / "store")
    meta = copy / "near" / "22" / "r" / "zarr.json"
    meta.write_text(meta.read_text().replace('"fill_value": -32768', '"fill_value": 0'))
    with pytest.raises(ValueError, match=r"near/22/r has the fill value 0, which valid data"):
        ogld.open(copy).region("22", 30_000_000, 100_000)
    chr21 = StoreReader.open(copy).variants(np.array([100])).position[0]
    assert ogld.open(copy).region("21", int(chr21), 1_000_000).r.size > 1, "chr21 is intact"


Served = tuple[str, list[tuple[str, int]]]


@contextlib.contextmanager
def serving(directory: Path) -> Iterator[Served]:
    """`directory` over HTTP, with the path and body length of every response it sends."""
    sent: list[tuple[str, int]] = []

    class Counting(RangeHandler):
        def send_header(self, keyword: str, value: str) -> None:
            if keyword == "Content-Length":
                sent.append((self.path, int(value)))
            super().send_header(keyword, value)

    handler = partial(Counting, directory=str(directory))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        worker = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
        worker.start()
        yield f"http://127.0.0.1:{server.server_address[1]}/", sent
        server.shutdown()
        worker.join()


@pytest.fixture
def served(small_chunks: Path) -> Iterator[Served]:
    """The re-sharded copy over HTTP."""
    with serving(small_chunks) as found:
        yield found


def test_a_few_pairs_over_http_fetch_a_small_part_of_the_near_band(
    served: Served, small_chunks: Path
) -> None:
    url, sent = served
    store = ogld.open(url)
    names = list(StoreReader.open(small_chunks).variants(np.array([500, 520, 900])).alid)
    got = store.matrix(names)
    np.testing.assert_array_equal(got.r, ogld.open(TEST_STORE).matrix(names).r)
    near = sum(p.stat().st_size for p in (small_chunks / "near" / "22" / "r").rglob("*"))
    assert len(shards(small_chunks, "near/22/r")) > 20, "the Near Band must span many shards"
    read = sum(length for path, length in sent if path.startswith("/near/22/r/c/"))
    assert 0 < read < near / 10, f"{read:,} of chr22's {near:,} Near Band bytes were read"


@pytest.mark.parametrize(("budget", "again"), [(1 << 28, False), (0, True)])
def test_a_repeated_read_fetches_its_chunks_once(
    served: Served,
    small_chunks: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget: int,
    again: bool,
) -> None:
    # Clumping reads a few pairs per lead, many times over: each chunk is fetched once and kept.
    url, sent = served
    monkeypatch.setattr(location, "CACHE_BYTES", budget)
    store = ogld.open(url)
    names = list(StoreReader.open(small_chunks).variants(np.arange(500, 560, 3)).alid)

    def near_bytes() -> int:
        return sum(length for path, length in sent if path.startswith("/near/22/r/c/"))

    first = store.matrix(names).r
    once = near_bytes()
    assert once > 0, "the first read must fetch Near Band chunks"
    np.testing.assert_array_equal(store.matrix(names).r, first)
    assert (near_bytes() > once) is again


@pytest.mark.parametrize("query", ["clump", "proxies"])
def test_at_r2_at_or_above_t_far_no_near_band_chunk_is_fetched(query: str) -> None:
    # At r2 ≥ t_far a variant's partners are its Proxy Index row, both sides, and the reader
    # chooses that part for both queries (spec §10; ogld#14). r2 is t_far exactly, the edge.
    # The committed store holds each array in one chunk, so any Near Band read fetches one.
    rows = support.passing_rows()
    names = support.alids(rows)
    with serving(TEST_STORE) as (url, sent):
        reader = StoreReader.open(url)
        if query == "clump":
            z = np.abs(support.panel_data().gwas().z)[rows]
            clumping.clump(reader, names, z, r2=support.T_FAR, kb=3000)
        else:
            proxies_query.proxies(reader, names[::5], r2=support.T_FAR, kb=10_000)
    fetched = {path for path, _ in reader._cache._kept}
    assert {"index/21/r", "index/22/r"} <= fetched, "the Proxy Index rows must be read"
    assert not [path for path in fetched if path.startswith("near/")], "no Near Band chunk"
    chunks = ("/near/21/r/c/", "/near/22/r/c/")
    assert not [path for path, _ in sent if path.startswith(chunks)], "nor any Near Band byte"


def test_the_test_store_passes_every_rule_over_http() -> None:
    with serving(TEST_STORE) as (url, sent):
        assert not any(ogld.validate(url).values())
    assert any(path.startswith("/index/22/partner/c/") for path, _ in sent), "read over HTTP"


def test_a_url_with_no_store_is_an_input_error(
    served: Served,
) -> None:
    url, _ = served
    with pytest.raises(InputError, match="no LD Store"):
        ogld.open(url + "nothing/")


def test_a_zarr_v2_hierarchy_is_not_read_as_a_store(tmp_path: Path) -> None:
    zarr.open_group(tmp_path / "v2", mode="w", zarr_format=2)
    with pytest.raises(InputError, match="Zarr v2"):
        ogld.open(tmp_path / "v2")


def test_a_region_takes_the_chr_prefix_a_query_may_carry() -> None:
    store = ogld.open(TEST_STORE)
    plain = store.region("22", 30_000_000, 1_000_000)
    assert len(plain.variants) > 3, "the region must hold variants"
    np.testing.assert_array_equal(store.region("chr22", 30_000_000, 1_000_000).r, plain.r)


def test_a_repeated_alid_is_a_repeated_row_and_an_empty_request_an_empty_matrix() -> None:
    store = ogld.open(TEST_STORE)
    one = list(StoreReader.open(TEST_STORE).variants(np.array([3])).alid)
    twice = store.matrix(one * 2)
    np.testing.assert_array_equal(twice.r, np.ones((2, 2)))
    assert (twice.kinds() == "stored").all()
    empty = store.matrix(["not-an-alid"])
    assert empty.r.shape == (0, 0)
    assert empty.absent == ("not-an-alid",)


def test_a_fill_of_zero_is_the_caller_s_and_fills_only_the_unstored_cells() -> None:
    store = ogld.open(TEST_STORE)
    rows = np.arange(0, 1056, 5)
    names = list(StoreReader.open(TEST_STORE).variants(rows).alid)
    plain, zero = store.matrix(names), store.matrix(names, fill=0)
    unstored = np.isnan(plain.r)
    assert unstored.any(), "the matrix must hold unstored pairs"
    assert (zero.r[unstored] == 0).all()
    np.testing.assert_array_equal(zero.r[~unstored], plain.r[~unstored])
    assert zero.fill == 0
