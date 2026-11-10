"""Bounded reads (spec §10; ogld#4): a reader takes only the chunks a request needs, reads a
re-sharded store the same, and refuses whatever a lost chunk reads as.

The committed Test Store holds each array in one shard of one inner chunk, so every read of it
touches every chunk. These tests read a copy re-sharded into small chunks, where a bounded read
and a whole-array read differ, and are expected to: each asserts the copy has many chunks first.
"""

from __future__ import annotations

import contextlib
import http.server
import shutil
import threading
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import zarr
from contract.support import RangeHandler
from store_copies import TEST_STORE, copy_store, resharded

import ogld
from ogld.format import InputError
from ogld.store import StoreReader, location, reader

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


@pytest.mark.parametrize("budget", [0, 1 << 28])
def test_reads_agree_with_and_without_the_chunk_cache(
    small_chunks: Path, budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A budget of 0 keeps no chunk, so every read fetches its chunks again. A tiny step makes
    # pair_r resolve its pairs a few at a time. Each must give the committed store's answers.
    rows = np.arange(0, 1056, 7)
    expected = StoreReader.open(TEST_STORE)
    monkeypatch.setattr(location, "CACHE_BYTES", budget)
    monkeypatch.setattr(reader, "_PAIRS_PER_STEP", 97)
    got = StoreReader.open(small_chunks)
    np.testing.assert_array_equal(got.pair_r(rows), expected.pair_r(rows))
    for read in ("index", "band"):
        mine, theirs = getattr(got, read)(rows), getattr(expected, read)(rows)
        assert len(theirs.partner) > 1000, f"{read} must return many partners"
        np.testing.assert_array_equal(mine.indptr, theirs.indptr)
        np.testing.assert_array_equal(mine.partner, theirs.partner)
        np.testing.assert_array_equal(mine.r, theirs.r)


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
    with pytest.raises(ValueError, match=r"near/22/r read the reserved code -32768"):
        store.matrix(names[-20:])


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("variants/alid", r"variants/alid read \d+ element\(s\) as its fill value"),
        ("variants/position", r"variants/position read \d+ element\(s\) as its fill value 0"),
        ("index/22/partner", r"index/22/partner read \d+ element\(s\) as its fill value -1"),
        ("index/22/r", "reserved code"),
        ("near/22/indptr", r"near/22/indptr read \d+ element\(s\) as its fill value -1"),
    ],
)
def test_a_lost_chunk_of_any_array_is_refused_never_read_as_data(
    small_chunks: Path, tmp_path: Path, path: str, message: str
) -> None:
    # An ALID that read as "" would make a present variant Absent; a partner of −1 would name
    # the last variant. Each is refused, naming the array.
    copy = tmp_path / "lost"
    shutil.copytree(small_chunks, copy)
    every = list(StoreReader.open(copy).variants(np.arange(1056)).alid)
    for shard in sorted((copy / path / "c").iterdir()):
        shard.unlink()
    with pytest.raises(ValueError, match=message):
        ogld.open(copy).matrix(every)


def test_an_r_array_whose_lost_chunk_would_read_as_a_number_is_refused(tmp_path: Path) -> None:
    copy = copy_store(tmp_path / "store")
    meta = copy / "near" / "22" / "r" / "zarr.json"
    meta.write_text(meta.read_text().replace('"fill_value": -32768', '"fill_value": 0'))
    with pytest.raises(ValueError, match=r"near/22/r has the fill value 0"):
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
