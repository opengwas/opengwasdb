"""How a gate records what Variant Routing resolved among an axis's duplicates.

Routing places each (variant, Analysis) once (`collapse_duplicates`): rows that
agree collapse to the first, and rows that disagree are all dropped. Both
leave the gate's admitted rows; the conflicting ones are logged by the ALID
of the variant they reached.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from opengwasdb.admission.gate import Gate

__all__ = ["record_duplicates"]


class PlacedDuplicates(Protocol):
    """What routing reports of one axis's duplicates: `Placed`'s fields."""

    @property
    def dropped(self) -> np.ndarray: ...

    @property
    def collapsed(self) -> int: ...


def record_duplicates(gate: Gate, placed: PlacedDuplicates, alids: Sequence[str]) -> None:
    """Count one axis's collapsed rows and count and log its conflicting ones.

    ``alids`` is that axis, so ``placed.dropped``'s indices name variants.
    """
    gate.duplicates([alids[i] for i in placed.dropped.tolist()], placed.collapsed)
