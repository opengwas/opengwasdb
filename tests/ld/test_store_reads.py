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
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import zarr
from contract import support
from contract.support import RangeHandler
from store_copies import TEST_STORE, copy_store, resharded

import ogld
from ogld.format import InputError, quantise
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


def retained(cache: location.ChunkCache) -> int:
    """The bytes the cache's kept chunks actually hold alive: each one's owning buffer, once."""
    owners = {}
    for chunk in cache._kept.values():
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


def test_threads_missing_one_chunk_store_it_once(small_chunks: Path) -> None:
    near = zarr.open_array(small_chunks / "near" / "22" / "r", mode="r")
    everything = np.asarray(near[:])
    cache = location.ChunkCache(64 * INNER * near.dtype.itemsize)
    start = threading.Barrier(16)
    wrong: list[int] = []

    def read(first: int) -> None:
        start.wait()
        coords = np.arange(first * INNER, (first + 48) * INNER, 5)
        if not np.array_equal(cache.take(near, coords), everything[coords]):
            wrong.append(first)

    workers = [threading.Thread(target=read, args=(i % 8,)) for i in range(16)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert wrong == []
    assert len(cache._kept) > 32, "the threads' reads must overlap in many kept chunks"
    assert cache._bytes == sum(chunk.nbytes for chunk in cache._kept.values()) == retained(cache)


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
