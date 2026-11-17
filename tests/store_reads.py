"""Counting the chunk reads one query makes at a Store Release's zarr store.

`test_query_eaf_reads` counts chunk reads of `eaf`, `eaf_baseline` and
`imputed` the way `test_query_metadata_reads` counts metadata reads: by
wrapping zarr's store for the duration of one query. The instrument lives here
so the counting logic is written once, and the metadata test is free to keep
its own narrower key filter.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from opengwasdb.top_hits import TopHitTier

#: Zarr keys that hold metadata, not chunk bytes; a metadata read is not a read
#: of the array's data.
_METADATA_SUFFIXES = (".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json")


def array_of(key: object, names: tuple[str, ...]) -> str | None:
    """Which of `names` a store key holds a chunk of, or None."""
    text = str(key)
    if text.endswith(_METADATA_SUFFIXES):
        return None
    for name in names:
        if name in text.split("/"):
            return name
    return None


@contextmanager
def chunk_reads(
    monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]
) -> Iterator[dict[str, list[str]]]:
    """Record every chunk read of each array in `names` through a LocalStore.

    zarr 3 routes a synchronous read through `LocalStore.get_sync` and an
    asynchronous one through `LocalStore.get`; both are counted, so a read via
    either path is seen. The wrapper returns whatever the original returns, so
    the same code covers the sync method and the coroutine one.
    """
    reads: dict[str, list[str]] = {name: [] for name in names}
    for method in ("get", "get_sync"):
        original = getattr(LocalStore, method)

        def wrapper(
            self: LocalStore,
            key: str,
            *args: Any,
            _original: Any = original,
            **kwargs: Any,
        ) -> Any:
            name = array_of(key, names)
            if name is not None:
                reads[name].append(str(key))
            return _original(self, key, *args, **kwargs)

        monkeypatch.setattr(LocalStore, method, wrapper)
    yield reads
    monkeypatch.undo()


#: zarr's selection entry points. Every array read, `[...]`, `.oindex` and `.vindex`
#: alike, goes through exactly one of them.
_SELECTIONS = (
    "get_basic_selection",
    "get_orthogonal_selection",
    "get_coordinate_selection",
    "get_mask_selection",
    "get_block_selection",
)


@contextmanager
def element_reads(monkeypatch: pytest.MonkeyPatch) -> Iterator[Counter[str]]:
    """Count the elements every array returns, keyed by the array's store path.

    A sharded plane's inner chunks share one shard file, and zarr fetches a whole
    shard without a byte range, so store keys cannot tell one read of a plane from
    two. Elements returned can: a plane read once returns `plane.size` elements, and
    `times_read` divides by it.
    """
    counts: Counter[str] = Counter()
    for method in _SELECTIONS:
        original = getattr(zarr.Array, method)

        def wrapper(self: zarr.Array, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            result = _original(self, *args, **kwargs)
            counts[array_key(self)] += int(np.size(result))
            return result

        monkeypatch.setattr(zarr.Array, method, wrapper)
    yield counts
    monkeypatch.undo()


def array_key(array: zarr.Array) -> str:
    """The key `element_reads` files an array's reads under."""
    return str(array.store_path)


def times_read(counts: Counter[str], array: zarr.Array) -> float:
    """How many times over `array`'s elements were returned while `counts` was counting."""
    return counts[array_key(array)] / array.size


@contextmanager
def largest_read(monkeypatch: pytest.MonkeyPatch) -> Iterator[Counter[str]]:
    """The most elements any one read returns, keyed by the array's store path.

    `element_reads` says how many elements a pass read in total; this says how much
    of an array it held at once, which is what a rule that pages through a table
    instead of loading it is for.
    """
    sizes: Counter[str] = Counter()
    for method in _SELECTIONS:
        original = getattr(zarr.Array, method)

        def wrapper(self: zarr.Array, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            result = _original(self, *args, **kwargs)
            sizes[array_key(self)] = max(sizes[array_key(self)], int(np.size(result)))
            return result

        monkeypatch.setattr(zarr.Array, method, wrapper)
    yield sizes
    monkeypatch.undo()


@contextmanager
def old_index_without_fields(
    monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]
) -> Iterator[None]:
    """Make a top-hit reader look like an index built before it carried fields.

    A current tier stores decoded `z`, `se`, `eaf` and `imputed`, so a top-hit
    query reads no plane at all and a shared read is not exercised. The path
    #253 is about is the fallback for an index that predates those fields, so
    the reader is told it has none and the caller derives them from the planes.
    """
    original = TopHitTier.has

    def has(self: TopHitTier, name: str) -> bool:
        if name in names or name == "se":
            return False
        return original(self, name)

    monkeypatch.setattr(TopHitTier, "has", has)
    yield


def duplicate_chunk_keys(reads: dict[str, list[str]]) -> dict[str, list[str]]:
    """Each array's chunk keys read more than once in one query."""
    out: dict[str, list[str]] = {}
    for name, keys in reads.items():
        duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
        if duplicates:
            out[name] = duplicates
    return out
