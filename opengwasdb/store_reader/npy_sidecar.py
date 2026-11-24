"""Opening a sidecar of plain `.npy` files beside the Store Variant Table (format 0.3.0).

The ALID sidecar (`alid_sidecar`) and the position sidecar (`position_sidecar`) are
each a fixed set of files that is written whole. A directory holds all of a sidecar's
files or none of them. A partial set is refused rather than read: a half-written
sidecar answers some questions and not others, and that reads as absence.

The two index pairs older than format 0.3.0, the fixed-width ALID pair and the rsID
pair, open more leniently (`open_pair_or_none`): a release built before one existed
has neither file. Both pairs map a key to a row, and both are inverted, to map a row
to its key, the same way (`invert_rows`).

Two sidecars hold text the same way, as UTF-8 bytes and `<u8` offsets: the ALID
sidecar (every row's ALID) and the `source_alid` sidecar (each lifted row's). Both
encode (`encode_utf8_offsets`), slice (`slice_texts`) and check (`offsets_fault`)
through this module.

This module imports numpy and the standard library only, as every codec does.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: The dtype of a text sidecar's offsets: entry `k` is where text `k` starts.
OFFSET_DTYPE = np.dtype("<u8")

#: Rows a reader names, or a scan reads, at a time, where one pass over a whole sidecar
#: would hold an object or an index array the size of the axis.
WINDOW_ROWS = 65_536


def encode_utf8_offsets(encoded: Sequence[bytes]) -> tuple[np.ndarray, np.ndarray]:
    """`encoded`'s texts as one ``uint8`` array and `OFFSET_DTYPE` offsets, one more than
    the texts: text ``k`` is ``utf8[offsets[k]:offsets[k + 1]]``."""
    lengths = np.fromiter((len(item) for item in encoded), dtype=OFFSET_DTYPE, count=len(encoded))
    offsets = np.zeros(len(encoded) + 1, dtype=OFFSET_DTYPE)
    np.cumsum(lengths, out=offsets[1:])
    utf8 = np.frombuffer(b"".join(encoded), dtype=np.uint8)
    return utf8, offsets


def slice_texts(text: memoryview, offsets: np.ndarray, at: np.ndarray) -> list[str]:
    """The texts at positions `at` (each in ``0..len(offsets) - 2``), decoded strictly:
    a text that is not UTF-8 raises, never reads as another."""
    at = np.asarray(at, dtype=np.int64)
    starts = np.asarray(offsets[at], dtype=np.int64).tolist()
    stops = np.asarray(offsets[at + 1], dtype=np.int64).tolist()
    return [str(text[a:b], "utf-8") for a, b in zip(starts, stops, strict=True)]


def dtype_fault(expected: Sequence[tuple[str, np.ndarray, np.dtype[Any]]]) -> str:
    """The first of `expected`'s ``(file, array, dtype)`` that is not a 1-d array of its
    dtype, or ``""``: the check every sidecar makes of its arrays before it reads them."""
    for name, array, dtype in expected:
        if array.dtype != dtype or array.ndim != 1:
            return f"{name} is {array.dtype} of {array.ndim} dimension(s), not 1-d {dtype}"
    return ""


def offsets_fault(
    offsets: np.ndarray,
    n_bytes: int,
    *,
    names: tuple[str, str],
    rising: tuple[str, str] | None,
    unit: str = "bytes",
) -> str:
    """Why `offsets` cannot slice `n_bytes` of text into non-empty texts, or ``""``.

    They start at 0 and end at `n_bytes`. With `rising`, ``(what a text is, why it is
    never empty)``, each also lies above the one before: the check reads every offset,
    so a reader that opens a sidecar leaves it out and the validator makes it. `names`
    is ``(offsets file, text file)``, and `unit` what the text file holds `n_bytes` of
    (the position sidecar's offsets slice rows, not bytes).
    """
    offsets_name, utf8_name = names
    if not len(offsets) or int(offsets[0]) != 0:
        return f"{offsets_name} does not start at 0"
    if rising is not None:
        values = np.asarray(offsets, dtype=np.uint64)
        flat = np.flatnonzero(values[1:] <= values[:-1])
        if flat.size:
            what, why = rising
            return f"{offsets_name} does not rise at {what} {int(flat[0])}: {why}"
    if int(offsets[-1]) != n_bytes:
        return (
            f"{offsets_name} ends at {int(offsets[-1])} but {utf8_name} holds {n_bytes} {unit}"
        )
    return ""


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
