"""A deterministic bounded sample of rows, chosen by the hash of their ALID.

One mechanism for every piece of resolution evidence that must stay bounded in
memory (ADR 0044): the phenotype-SD evidence and the effect-consistency check
(ADR 0069) both keep the bottom-`k` rows by `site_hash`, the rule
`opengwasdb.build.eaf_orientation.select_sites` uses to pick an Analysis's own
variants, so a sample drawn by one and a sample drawn by another agree on which
variants exist. Equal hashes are broken by source row order, which keeps a
repeated ALID's two rows both eligible.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from opengwasdb.build.eaf_orientation import site_hash

__all__ = ["HashSample"]

Payload = TypeVar("Payload")


@dataclass
class HashSample(Generic[Payload]):
    """The `k` offered rows with the smallest `site_hash`, in source row order.

    Below the bound nothing is sampled: every offered row is retained, so a
    consumer reading `retained()` sees the whole file's rows rather than a
    subset.
    """

    k: int
    offered: int = 0
    _sequence: int = 0
    # `(-hash, sequence, payload)`. The sequence is unique, so the heap never
    # compares payloads.
    _heap: list[tuple[int, int, Payload]] = field(default_factory=list)

    def offer(self, alid: str, payload: Payload) -> None:
        """Retain the row if it outranks the current bottom-`k`."""
        self.offered += 1
        self._sequence += 1
        entry = (-site_hash(alid), self._sequence, payload)
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
        elif entry[:2] > self._heap[0][:2]:
            heapq.heapreplace(self._heap, entry)

    @property
    def sampled(self) -> bool:
        """Whether the bound dropped any offered row."""
        return self.offered > self.k

    def retained(self) -> list[Payload]:
        """The retained payloads, in source row order."""
        return [payload for _hash, _sequence, payload in sorted(self._heap, key=_sequence_of)]


def _sequence_of(entry: tuple[int, int, Payload]) -> int:
    return entry[1]
