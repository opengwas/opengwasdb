"""Where an LD Store is, and how its elements are read (spec §1, §10).

A store is a local path or an HTTP(S) URL, opened as one Zarr hierarchy. A reader and
validation open it the same way, so a store served from a static host that honours byte
ranges reads exactly as it does from disk. zarr opens a URL through fsspec's HTTP filesystem
(aiohttp). Nothing here lists a directory: every node is found by its own metadata document,
which a static host serves like any other file.

Every element a reader or a rule reads goes through `take`, which fetches only the inner
chunks holding them, so how a store is read is decided in one place, which ogld#8 tunes when
it times the reads.
"""

from __future__ import annotations

import os
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import zarr

from ogld.format import InputError

#: A read of `coords` takes the slice from the first to the last when that slice is at most
#: this many times as long as `coords`; otherwise it selects the coordinates themselves.
_DENSE: Final = 4


def open_group(location: str | os.PathLike[str]) -> zarr.Group:
    """The root group of the hierarchy at `location`, read-only, whatever its Zarr format.

    A location with no Zarr group at all is an `InputError` naming it. A Zarr v2 group opens,
    so validation can say it is not v3 (spec §12, `envelope.arrays`) rather than not there.
    """
    try:
        return zarr.open_group(str(location), mode="r")
    except FileNotFoundError as missing:
        raise InputError(f"no LD Store at {location}: there is no Zarr group there") from missing


def take(array: zarr.Array[Any], coords: npt.ArrayLike) -> npt.NDArray[Any]:
    """The elements of a 1-D array at `coords`, in that order, reading only the inner chunks
    that hold them: one slice when they are dense, the coordinates themselves when not."""
    at = np.asarray(coords, dtype=np.int64)
    if len(at) == 0:
        return np.asarray(array[0:0])
    lo, hi = int(at.min()), int(at.max()) + 1
    if hi - lo <= _DENSE * len(at):
        return np.asarray(array[lo:hi])[at - lo]
    distinct, back = np.unique(at, return_inverse=True)
    return np.asarray(array.get_coordinate_selection(distinct))[back]
