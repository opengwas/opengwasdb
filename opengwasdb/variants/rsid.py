"""The rs identifier rules readers and builders share (issue #306).

Two rules live here because no one reader or builder owns them: how a source's
identifier cell becomes the one rs identifier a Store Variant Table records,
and what a Hybrid build does -- and reports -- when its sources name one stored
variant with different rs identifiers.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from opengwasdb.variants.normalise import chromosome_sort_key

#: How many conflicting routing keys a build's summary names. The count covers
#: every one; the sample is for a reader to look at, so it is short and stable.
RSID_CONFLICT_SAMPLE_SIZE = 20


def normalise_rsid(cell: str) -> str:
    """The rs identifier a source cell names, or "" if it names none.

    Harmonised GWAS Catalog files write one name several ways (stores #225): a
    compound ``rsN:pos:a:b`` id appends a coordinate to the identifier, and a
    ``;``-separated list carries several identifiers. An rs identifier ends at
    its first ``:``, and a list names its first rs item, so
    ``rs199633771:13122834:C:T`` and ``rs199633771`` are one name and two
    sources spelling it differently agree on it. A cell with no rs item --
    another kind of identifier, ``.``, or nothing -- names nothing.
    """
    if ";" in cell:
        for item in cell.split(";"):
            name = normalise_rsid(item)
            if name:
                return name
        return ""
    name = cell.strip()
    if not name.startswith("rs"):
        return ""
    return name.partition(":")[0]


class RsidConflictPolicy(StrEnum):
    """What a Hybrid build does with a routing key its sources name differently.

    ``refuse`` fails the build, naming the key and both names (issue #255).
    ``drop`` stores that variant with no rsid and reports the key: absence is
    kept distinct from a name that may be wrong, and the variant is still found
    by ALID and by position (issue #306).
    """

    REFUSE = "refuse"
    DROP = "drop"


@dataclass(frozen=True)
class RsidConflict:
    """One routing key and the distinct names its sources gave it, in
    ``(rank, site)`` order."""

    key: str
    names: tuple[str, ...]


@dataclass(frozen=True)
class RsidConflicts:
    """What a build's rsid harvest found, for the build's summary.

    ``n_keys`` is ``None`` when the build compared no source names -- a
    ``--reference-panel`` build, or a reference that names its own variants --
    so a count that was never taken cannot be read as a count of zero.
    """

    policy: RsidConflictPolicy
    n_keys: int | None
    sample: tuple[RsidConflict, ...] = ()

    @classmethod
    def found(
        cls, policy: RsidConflictPolicy, conflicts: Mapping[str, Sequence[str]]
    ) -> RsidConflicts:
        """Every conflicting key counted; the first few in genomic order named."""
        first = heapq.nsmallest(RSID_CONFLICT_SAMPLE_SIZE, conflicts, key=_genomic_order)
        return cls(
            policy=policy,
            n_keys=len(conflicts),
            sample=tuple(RsidConflict(key, tuple(conflicts[key])) for key in first),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RsidConflicts:
        """Read back what `as_dict` wrote (a checkpoint's record)."""
        return cls(
            policy=RsidConflictPolicy(payload["policy"]),
            n_keys=payload["n_keys"],
            sample=tuple(
                RsidConflict(entry["key"], tuple(entry["names"])) for entry in payload["sample"]
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """The JSON form a build's summary carries."""
        return {
            "policy": self.policy.value,
            "n_keys": self.n_keys,
            "sample": [{"key": item.key, "names": list(item.names)} for item in self.sample],
        }


def _genomic_order(key: str) -> tuple[tuple[int, str], int, str, str]:
    """A ``chrom:pos:ref:alt`` routing key's place on the genome."""
    chrom, pos, ref, alt = key.split(":")
    return chromosome_sort_key(chrom), int(pos), ref, alt
