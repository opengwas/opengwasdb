"""Opening a sidecar of plain `.npy` files beside the Store Variant Table (format 0.3.0).

The ALID sidecar (`alid_sidecar`) and the position sidecar (`position_sidecar`) are
each a fixed set of files that is written whole. A directory holds all of a sidecar's
files or none of them. A partial set is refused rather than read: a half-written
sidecar answers some questions and not others, and that reads as absence.

This module imports numpy and the standard library only, as both codecs do.
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
