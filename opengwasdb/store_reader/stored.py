"""A Store Release's stored arrays, asked for by what they hold (#473).

The query facade reads a release through its decoders. The engine's other
readers -- the validator, Reference Completion, the top-hit, Rho and Indexed
Variant Subset builders, `overview.html` -- also need what a release stores as
it is stored: a validation rule that judges the storage itself reads stored
codes, chunking and attributes, and a builder that adds to a release, or
completes one, reads the side arrays its source was coded against rather than
refitting them. They ask for them here, and a release's whole `data.zarr` group
from `opengwasdb.store_reader.release.release_arrays` (what `StoreRelease.arrays`
is), so where a release keeps its arrays and how they are opened (read-only,
through `opengwasdb.store_reader.arrays.open_group`) are known to the Store
Reader alone.

A group these functions return is zarr's, opened read-only: a caller that
writes into a release opens it through `opengwasdb.store.arrays`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.store_reader.arrays import open_group
from opengwasdb.store_reader.csr import RAGGED_ZARR_PATH
from opengwasdb.store_reader.encoding.codec import EAF_BASELINE, SE_COEFFICIENTS


def sequence_arrays(release: str | Path) -> Any:
    """The Association Sequence group of the release at `release`, read-only.

    A Ragged release's CSR, or a Hybrid release's Overflow: `data.zarr/ragged`.
    """
    return open_group(Path(release) / RAGGED_ZARR_PATH)


def stored_attributes(group_path: str | Path) -> dict[str, Any]:
    """The attributes the stored group at `group_path` records, read-only: a
    release's layout recordings on its `data.zarr` root, an Indexed Variant
    Subset's recorded facts on its own group."""
    return dict(open_group(group_path).attrs)


def optional_array(group: Any, name: str) -> Any:
    """`group`'s array `name`, unread, or None when the group does not carry it.

    For a reader that gathers from the array a row at a time (the By-Variant
    Index); `eaf_baseline` and `se_coefficients` read the whole array.
    """
    return group[name] if name in group else None


def eaf_baseline(group: Any) -> np.ndarray | None:
    """The per-variant `eaf_baseline` a group's EAF codes are offsets from, as
    stored (float32), or None when the group holds none."""
    array = optional_array(group, EAF_BASELINE)
    return None if array is None else np.asarray(array[:], dtype=np.float32)


def se_coefficients(group: Any) -> np.ndarray | None:
    """The per-Analysis `se_coefficients` a group's residual SE codes are coded
    against, as stored (float32), or None when the group holds none."""
    array = optional_array(group, SE_COEFFICIENTS)
    return None if array is None else np.asarray(array[:], dtype=np.float32)
