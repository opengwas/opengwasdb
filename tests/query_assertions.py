"""The SE the public query surface answers with, for the Dense and Hybrid seam tests.

Two suites ask the same question of a built release -- what does a lookup decode
at this cell, and what does a Top-Hit Query read there -- because a Dense build
and a Hybrid's Dense Component are written by the same band write (issue #355).
The reads live here rather than as two copies the duplication gate would flag.
Both assert the release answered at all: a query that returns nothing proves
nothing about the value it did not return.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from opengwasdb.query import query_store

#: The loosest indexed tier: a hit there is a hit in every tier above it.
DEFAULT_THRESHOLD = 5e-8


def decoded_se(store: Path, alid: str, analysis_id: str) -> float:
    """The SE a lookup decodes at `(alid, analysis_id)`, which must be one cell."""
    with query_store(store) as query:
        cell = query.lookup([alid], [analysis_id])
        assert len(cell["se"]) == 1, (
            f"{alid} in {analysis_id} must decode to one association, or this proves nothing"
        )
        return float(cell["se"][0])


def indexed_se(
    store: Path, alid: str, analysis_id: str, threshold: float = DEFAULT_THRESHOLD
) -> float:
    """The SE a Top-Hit Query reads at `(alid, analysis_id)`, which must be a hit there.

    The tier's `se` is `float32` by the index's own contract (spec §18); the value
    is what a query decodes at the cell, which rule `top_hits.se` holds validation
    to.
    """
    with query_store(store) as query:
        hits = query.top_hits(threshold=threshold, analysis_id=analysis_id)
        assert hits["se"].dtype == np.dtype("float32")
        variants = query.variants_table()
        names = [variants[int(index)]["alid"] for index in hits["variant_index"]]
        assert alid in names, (
            f"{alid} must clear {threshold} in {analysis_id}, or the index is not under test"
        )
        return float(hits["se"][np.array([name == alid for name in names])][0])
