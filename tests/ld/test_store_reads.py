"""Bounded reads (ld-store §10; ogld#4): a reader takes only the chunks a request needs, reads a
re-sharded store the same, and refuses whatever a lost chunk reads as.

The committed Test Store holds each array in one shard of one inner chunk, so every read of it
touches every chunk. These tests read a copy re-sharded into small chunks, where a bounded read
and a whole-array read differ, and are expected to: each asserts the copy has many chunks first.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import json
import threading
import tracemalloc
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from contract import support
from contract.support import RangeHandler
from store_copies import TEST_STORE, copy_store, resharded

import opengwasdb.ld
from ogquery.ld.store import StoreReader, decoding, location
from opengwasdb.ld.store.validation import validate_group

#: Inner chunks of 512 elements, 4 to a shard: chr22's Near Band is 136 chunks in 34 shards.
INNER, PER_SHARD = 512, 4


@pytest.fixture(scope="module")
def small_chunks(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return resharded(tmp_path_factory.mktemp("resharded") / "store", INNER, PER_SHARD)


def whole(location: Path | str) -> opengwasdb.ld.LDMatrix:
    store = opengwasdb.ld.open(location)
    rows = np.arange(store.provenance["n_variants"])
    return store.matrix(list(StoreReader.open(location).variants(rows).alid))


def shards(store: Path, path: str) -> list[Path]:
    found = sorted((store / path / "c").iterdir(), key=lambda p: int(p.name))
    assert len(found) > 4, f"{path} must span several shards for this to mean anything"
    return found


def reference_pairs(rows: np.ndarray) -> np.ndarray:
    """What the store must answer among its `rows`, from the Test Panel's reference r at k = 10:
    the A1-signed code over 32,767 where the store holds the pair, NaN where it does not."""
    names, _ = support.oriented()
    panel = {str(name): i for i, name in enumerate(names)}
    variants = StoreReader.open(TEST_STORE).variants(rows)
    ref = support.reference_matrix(support.K, tuple(panel[a] for a in variants.alid))
    gap = np.abs(variants.position[:, None] - variants.position[None, :])
    codes = support.codes_at(np.where(np.isnan(ref), 0.0, ref))
    held = ~np.isnan(ref) & ((gap <= 3_000_000) | (np.abs(codes) >= support.FLOOR))
    expected = np.where(held, codes / support.R_SCALE, np.nan)
    np.fill_diagonal(expected, 1.0)
    return expected


def stored_positions() -> tuple[np.ndarray, np.ndarray]:
    """The chromosome and position of each Test Store row, from the Test Panel."""
    rows = support.passing_rows()
    variants = support.panel_data().variants()
    return np.asarray(variants.chrom)[rows], variants.pos.astype(np.int64)[rows]


def kind(key: object) -> str:
    """What the cache keeps under `key`: a shard's index, or a chunk."""
    return "index" if isinstance(key, location._ShardIndex) else "chunk"


def retained(cache: location.ChunkCache) -> int:
    """The bytes the cache's kept chunks actually hold alive: each one's owning buffer, once."""
    owners = {}
    for chunk, _ in cache._kept.values():
        owner = chunk if chunk.base is None else chunk.base
        owners[id(owner)] = owner.nbytes
    return sum(owners.values())


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
ASCII = {
    "short 8": (lambda i: text(i, 8), 40_000),
    "long 200": (lambda i: text(i, 200), 450_000),
    "mixed": (lambda i: text(i, 200 if i % 10 == 0 else 8), 100_000),
}


class Measured(location.ChunkCache):
    """A chunk cache that measures what it holds after every chunk or index it keeps, inside
    the lock it keeps them under: the most it ever held, not a sample of it."""

    def __init__(self, budget: int) -> None:
        super().__init__(budget)
        self.held: list[int] = []
        self.kinds: set[str] = set()

    def _keep(self, key: tuple[str, int], chunk: np.ndarray, counted: int) -> np.ndarray:
        kept = super()._keep(key, chunk, counted)
        self.held.append(retained(self))
        self.kinds.update(kind(key) for key in self._kept)
        return kept


def fill_findings(store: Path) -> list[opengwasdb.ld.Finding]:
    root = zarr.open_group(store, mode="r")
    return validate_group(root, root.attrs["ld_store"], ["envelope.fill"])["envelope.fill"]


def set_fill(store: Path, path: str, fill: object) -> None:
    meta = store / path / "zarr.json"
    doc = json.loads(meta.read_text())
    doc["fill_value"] = fill
    meta.write_text(json.dumps(doc))


def a_far_pair(chrom: str) -> tuple[int, int]:
    """One pair on `chrom` beyond near_bp that the Test Store holds: a read of it opens the
    chromosome's Far Pairs."""
    rows = np.arange(1056)
    names, position = stored_positions()
    expected = reference_pairs(rows)
    apart = (
        ~np.isnan(expected)
        & (np.abs(position[:, None] - position) > 3_000_000)
        & (names[:, None] == chrom)
        & (names[None, :] == chrom)
    )
    i, j = (int(k[0]) for k in np.nonzero(apart))
    return i, j


def test_a_partner_whose_lost_chunk_would_read_as_row_0_is_refused(tmp_path: Path) -> None:
    # The review's case: fill 0 (zarr's int32 default) is a partner row a valid store holds —
    # chr22's first variant — so a lost chunk would read as data. The read that opens the array
    # refuses the fill rather than decode it, and validation fails it too.
    copy = copy_store(tmp_path / "store")
    set_fill(copy, "far/22/partner", 0)
    lo, hi = a_far_pair("22")
    reader = StoreReader.open(copy)
    with pytest.raises(ValueError, match=r"far/22/partner has the fill value 0, which valid data"):
        reader.r_of_pairs(np.array([lo]), np.array([hi]))
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
        opengwasdb.ld.open(copy).region("22", 30_000_000, 1_000_000)
    assert fill_findings(copy), "validate fails what open refuses"


def test_an_r_array_whose_lost_chunk_would_read_as_a_number_is_refused(tmp_path: Path) -> None:
    copy = copy_store(tmp_path / "store")
    meta = copy / "near" / "22" / "r" / "zarr.json"
    meta.write_text(meta.read_text().replace('"fill_value": -32768', '"fill_value": 0'))
    with pytest.raises(ValueError, match=r"near/22/r has the fill value 0, which valid data"):
        opengwasdb.ld.open(copy).region("22", 30_000_000, 100_000)
    chr21 = StoreReader.open(copy).variants(np.array([100])).position[0]
    assert opengwasdb.ld.open(copy).region("21", int(chr21), 1_000_000).r.size > 1, (
        "chr21 is intact"
    )


async def reached(event: threading.Event) -> None:
    """Wait until `event` is set, or fail: the test's own worker has to be in flight."""
    async with asyncio.timeout(2):
        while not event.is_set():
            await asyncio.sleep(0.001)


class Sent(list[tuple[str, int]]):
    """The path and body length of every response a server sent, and each response's status.

    A list of the pairs the tests already read, so a server that answers a range past a file's end
    with 416 (a length of 0) is told apart from one that answers it with an empty 206 (partner-rows
    §5.1).
    """

    def __init__(self) -> None:
        super().__init__()
        self.statuses: list[int] = []


Served = tuple[str, Sent]


@contextlib.contextmanager
def serving(directory: Path) -> Iterator[Served]:
    """`directory` over HTTP, with the path, body length and status of every response it sends."""
    sent = Sent()

    class Counting(RangeHandler):
        def send_response(self, code: int, message: str | None = None) -> None:
            sent.statuses.append(code)
            super().send_response(code, message)

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


@dataclass
class InFlight:
    """The chunk requests a server is answering at once: now, and at most. Each is held until
    `together` are in flight, or `patience` seconds pass, so requests made together are
    answered together and requests made one after another each wait out the patience."""

    together: int
    patience: float = 0.3
    now: int = 0
    peak: int = 0
    arrived: threading.Condition = field(default_factory=threading.Condition)

    def start(self) -> None:
        with self.arrived:
            self.now += 1
            self.peak = max(self.peak, self.now)
            self.arrived.notify_all()
            self.arrived.wait_for(lambda: self.now >= self.together, self.patience)

    def end(self) -> None:
        with self.arrived:
            self.now -= 1


@contextlib.contextmanager
def holding(directory: Path, flight: InFlight) -> Iterator[str]:
    """`directory` over HTTP, each chunk or shard-index request held as `flight` says."""

    class Held(RangeHandler):
        def send_head(self) -> Any:
            if "/c/" not in self.path:
                return super().send_head()
            flight.start()
            try:
                return super().send_head()
            finally:
                flight.end()

    handler = partial(Held, directory=str(directory))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        worker = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
        worker.start()
        yield f"http://127.0.0.1:{server.server_address[1]}/"
        server.shutdown()
        worker.join()


@dataclass
class Held:
    """What a read holds of what it fetched, measured each time it fetches a batch, as the
    cache counts memory (`location._counted`): the pieces of earlier batches still alive, and
    this batch's."""

    earlier: list[int] = field(default_factory=list)
    batches: list[int] = field(default_factory=list)
    alive: list[weakref.ref[np.ndarray]] = field(default_factory=list)

    def still_held(self) -> int:
        # Direct chunks transfer their sole buffers to the cache, so those references are
        # kept chunks now, not a prior batch's uncounted run still pinned beside the cache.
        return sum(
            location._counted(p)
            for p in (ref() for ref in self.alive)
            if p is not None and not isinstance(p, decoding.DecodedChunk)
        )


@pytest.fixture
def served(small_chunks: Path) -> Iterator[Served]:
    """The re-sharded copy over HTTP."""
    with serving(small_chunks) as found:
        yield found


def test_the_test_store_passes_every_rule_over_http() -> None:
    # The chunk files' ends are probed one byte past their length, which a server answers with 416
    # (RFC 9110 §15.5.17): an unsatisfiable range says the file ends there, and a store must not be
    # refused for it (partner-rows §5.1, §5.3). This store passed only when the probe read an empty
    # 206.
    with serving(TEST_STORE) as (url, sent):
        found = opengwasdb.ld.validate(url)
    assert not any(found.values()), {rule: f for rule, f in found.items() if f}
    assert any(path.startswith("/far/22/partner/c/") for path, _ in sent), "read over HTTP"
    assert 416 in sent.statuses, sorted(set(sent.statuses))
