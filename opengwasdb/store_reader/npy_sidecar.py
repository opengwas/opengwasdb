"""Opening a sidecar of plain `.npy` files beside the Store Variant Table (format 0.3.0).

The ALID sidecar (`alid_sidecar`) and the position sidecar (`position_sidecar`) are
each a fixed set of files that is written whole. A directory holds all of a sidecar's
files or none of them. A partial set is refused rather than read: a half-written
sidecar answers some questions and not others, and that reads as absence.

The two index pairs older than format 0.3.0, the fixed-width ALID pair and the rsID
pair, open more leniently (`open_pair_or_none`): a release built before one existed
has neither file. Both pairs map a key to a row, and both are inverted, to map a row
to its key, the same way (`invert_rows`).

This module imports numpy and the standard library only, as every codec does.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def holds_sidecar(store_path: str | Path, filenames: Sequence[str], what: str) -> bool:
    """Whether `store_path` holds the sidecar named `what`, made of `filenames`.

    False when none of the files is there. Raises FileNotFoundError when only some are.
    """
    store = Path(store_path)
    present = [name for name in filenames if (store / name).exists()]
    if present and len(present) != len(filenames):
        missing = sorted(set(filenames) - set(present))
        raise FileNotFoundError(f"{store}: {what} is incomplete, missing {missing}")
    return bool(present)


def open_sidecar_arrays(
    store_path: str | Path, filenames: Sequence[str], what: str
) -> list[np.ndarray] | None:
    """Each of `filenames` memory-mapped, in order. None when `store_path` holds none
    of them; a partial set is refused (`holds_sidecar`)."""
    if not holds_sidecar(store_path, filenames, what):
        return None
    store = Path(store_path)
    return [np.load(store / name, mmap_mode="r") for name in filenames]


def open_pair_or_none(
    store_path: str | Path, filenames: Sequence[str]
) -> list[np.ndarray] | None:
    """Each of `filenames` memory-mapped, in order, or None unless every one is there.

    The 0.2 series' indexes -- the fixed-width ALID pair and the rsID pair -- are read
    so: a release built before an index existed has none of its files, and opens.
    """
    store = Path(store_path)
    paths = [store / name for name in filenames]
    if not all(path.exists() for path in paths):
        return None
    return [np.load(path, mmap_mode="r") for path in paths]


def invert_rows(rows: np.ndarray, n_rows: int, dtype: np.typing.DTypeLike) -> np.ndarray:
    """Each of `n_rows` rows' position in `rows`, -1 for a row `rows` does not name.

    `rows` is an index's row column: entry `k` names row `rows[k]`. Where `rows` names a
    row twice the later entry wins; a caller that must refuse that compares
    ``inverse[rows]`` with ``arange``. Every row must be in ``[0, n_rows)``.
    """
    rows = np.asarray(rows, dtype=np.int64)
    inverse = np.full(n_rows, -1, dtype=dtype)
    inverse[rows] = np.arange(rows.size, dtype=dtype)
    return inverse
