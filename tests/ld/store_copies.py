"""Copies of the Test Store for tests that must change one: re-sharded, or with a piece lost.

No test writes into `tests/data/`; each works on a copy under its own `tmp_path`.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

TEST_STORE = Path(__file__).resolve().parent / "data" / "test-panel" / "store"


def copy_store(to: Path) -> Path:
    """A plain copy of the committed Test Store at `to`."""
    shutil.copytree(TEST_STORE, to)
    return to


def arrays(group: zarr.Group, prefix: str = "") -> Iterator[tuple[str, zarr.Array[Any]]]:
    """Every array under `group`, by its path."""
    for name, member in group.members():
        if isinstance(member, zarr.Array):
            yield prefix + name, member
        else:
            yield from arrays(member, f"{prefix}{name}/")


def resharded(to: Path, inner: int, per_shard: int) -> Path:
    """The Test Store rewritten with inner chunks of `inner` elements and `per_shard` of them
    to a shard: the same data and metadata on another layout, as ogld#9 may serve it (spec §6).
    """
    copy_store(to)
    source = zarr.open_group(TEST_STORE, mode="r")
    target = zarr.open_group(to, mode="r+")
    for path, array in arrays(source):
        data = np.asarray(array[:])
        group, name = path.rsplit("/", 1)
        shutil.rmtree(to / path)
        chunk = min(inner, max(1, len(data)))
        shard = min(chunk * per_shard, -(-max(1, len(data)) // chunk) * chunk)
        made = zarr.open_group(to / group, mode="r+").create_array(
            name,
            shape=data.shape,
            dtype=str if data.dtype.kind in "OT" else data.dtype,
            chunks=(chunk,),
            shards=(shard,),
            compressors=[ZstdCodec(level=3)],
            fill_value=array.fill_value,
        )
        made[:] = data
    assert dict(target.attrs) == dict(source.attrs)
    return to
