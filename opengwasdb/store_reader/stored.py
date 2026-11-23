"""A Store Release's stored arrays, asked for by what they hold (#473).

The query facade reads a release through its decoders. The engine's other
readers -- the validator, Reference Completion, the top-hit, Rho and Indexed
Variant Subset builders, `overview.html` -- also need what a release stores as
it is stored: a validation rule that judges the storage itself reads stored
codes, chunking and attributes, and a builder that adds to a release, or
completes one, reads the side arrays its source was coded against rather than
refitting them. They ask for them here, so where a release keeps its arrays
(`data.zarr`, its `ragged` group) and how they are opened (read-only, through
`opengwasdb.store_reader.arrays.open_group`) are known to the Store Reader
alone.

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

#: Where a release, and a Hybrid's Dense Component (itself a release), keeps its arrays.
DATA_ZARR = "data.zarr"


def release_arrays(release: str | Path) -> Any:
    """The `data.zarr` group of the release at `release`, read-only.

    A Dense release's grid and its side arrays, a Hybrid release's Overflow
    parent group, the Top-Hit Index and Rho of either. The path may be a Hybrid
    Dense Component's, or a staged release's that is not yet published.
    """
    return open_group(Path(release) / DATA_ZARR)


def sequence_arrays(release: str | Path) -> Any:
    """The Association Sequence group of the release at `release`, read-only.

    A Ragged release's CSR, or a Hybrid release's Overflow: `data.zarr/ragged`.
    """
    return open_group(Path(release) / RAGGED_ZARR_PATH)


def stored_attributes(group: str | Path) -> dict[str, Any]:
    """The attributes the stored group at `group` records, read-only: a
    release's layout recordings on its `data.zarr` root, an Indexed Variant
    Subset's recorded facts on its own group."""
    return dict(open_group(group).attrs)


def eaf_baseline(group: Any) -> np.ndarray | None:
    """The per-variant `eaf_baseline` a group's EAF codes are offsets from, as
    stored (float32), or None when the group holds none."""
    if EAF_BASELINE not in group:
        return None
    return np.asarray(group[EAF_BASELINE][:], dtype=np.float32)


def se_coefficients(group: Any) -> np.ndarray | None:
    """The per-Analysis `se_coefficients` a group's residual SE codes are coded
    against, as stored (float32), or None when the group holds none."""
    if SE_COEFFICIENTS not in group:
        return None
    return np.asarray(group[SE_COEFFICIENTS][:], dtype=np.float32)
