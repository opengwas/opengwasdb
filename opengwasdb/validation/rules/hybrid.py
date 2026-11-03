"""Rules about how a Hybrid release's two components partition one variant space (ADR 0074).

`dense_to_shared.npy` maps each Dense Component row to its row of the shared
variant table (spec §16). It must name rows that exist, and no shared row twice,
before anything can be said about which shared rows are on the panel.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from opengwasdb.layouts.hybrid.layout import dense_to_shared_path
from opengwasdb.model.enums import PrimaryStorageLayout
from opengwasdb.validation.rule import Check, Kind, Rule, Target
from opengwasdb.validation.rules.release import REQUIRED

#: The rule every rule reading the Dense Component's place in the shared space runs after.
DENSE_TO_SHARED = "hybrid.dense_to_shared"


def is_hybrid(target: Target) -> bool:
    return target.layout is PrimaryStorageLayout.HYBRID and not target.is_component


def _release(target: Target) -> Target:
    """The outer release a Hybrid target belongs to."""
    while target.parent is not None:
        target = target.parent
    return target


def shared_rows(target: Target) -> int:
    """The rows of the shared variant table, by its offset sidecar."""
    release = _release(target)
    return release.remember("n_shared", lambda: release.variant_axis.n_variants)


def dense_to_shared(target: Target) -> np.ndarray:
    """The shared row of each Dense Component row, read once for the run."""
    release = _release(target)
    return release.remember(
        "dense_to_shared", lambda: np.load(dense_to_shared_path(release.store.path))
    )


def mapping(target: Target) -> Iterator[str]:
    """`dense_to_shared` is a strictly-increasing set of valid shared rows."""
    rows = dense_to_shared(target)
    n_shared = shared_rows(target)
    if len(rows) and (int(rows.min()) < 0 or int(rows.max()) >= n_shared):
        yield "dense_to_shared references rows outside the shared variant table"
    elif len(rows) != len(np.unique(rows)):
        yield "dense_to_shared maps two Dense Component rows to one shared row"


RULES: tuple[Rule, ...] = (
    Check(name=DENSE_TO_SHARED, kinds=Kind.RELEASE, when=is_hybrid, after=(REQUIRED,), run=mapping),
)
