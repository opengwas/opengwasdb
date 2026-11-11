"""Copies of the Test Store for tests that must change one: re-sharded, or with a piece lost.

No test writes into `tests/data/`; each works on a copy under its own `tmp_path`.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import zarr
from zarr.codecs import ZstdCodec

TEST_STORE = Path(__file__).resolve().parent / "data" / "test-panel" / "store"
#: The Test Store's encoding, r̂ = code / R_SCALE, as a literal: every test that needs the Test
#: Store's step, or builds the Test Panel to compare with it, takes it from here, never from
#: ogld (spec §3; ADR 0004). A test holds the store's record to it.
R_SCALE = 32767
#: Half a step: |r̂ − r| never exceeds it.
HALF_STEP = 1 / (2 * R_SCALE)
#: The far code floor at R_SCALE for n = 771 (spec §3's table): FLOOR at the Test Store's
#: k = 10, FLOOR_K0 at k = 0, and FLOOR_K7 at k = 7, where a build that drops a local PC lands.
FLOOR = 5426
FLOOR_K0 = 5391
FLOOR_K7 = 5416


def codes_at(r: npt.ArrayLike, scale: int = R_SCALE) -> npt.NDArray[np.int64]:
    """r as a store at × `scale` codes it, computed here rather than by ogld: spec §3's
    clamp(rint(r × scale), −scale, +scale), half to even."""
    return np.clip(np.rint(np.asarray(r, dtype=np.float64) * scale), -scale, scale).astype(np.int64)


def stored_at(r: npt.ArrayLike, scale: int = R_SCALE) -> npt.NDArray[np.float64]:
    """r as a store at × `scale` reads it back: its code / scale."""
    return codes_at(r, scale) / scale


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
