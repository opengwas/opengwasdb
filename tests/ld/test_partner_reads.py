"""The reads `clump`, `proxies` and `region` make (spec §10; ogld#14), counted where they happen.

Each test counts at the reader's chunk cache: a `Counting` cache is put on one reader instance,
so no reader is subclassed and no module global is patched. "Resolved" is the Stored r the
reader takes, the elements of every `near/<c>/r` and `index/<c>/r` it asks the cache for.
What a query needs comes from an independent greedy over the Test Panel's reference r.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import zarr
from contract import support
from contract.support import alids, panel_data, passing_rows

from ogld.format import dequantise
from ogld.query import clumping
from ogld.store import StoreReader, location


class Counting(location.ChunkCache):
    """A reader's chunk cache that counts the elements of each array it hands the reader."""

    def __init__(self) -> None:
        super().__init__()
        self.taken: Counter[str] = Counter()

    def take(self, array: zarr.Array[Any], coords: npt.ArrayLike) -> npt.NDArray[Any]:
        got = super().take(array, coords)
        self.taken[array.path] += len(got)
        return got

    def resolved(self) -> int:
        """The Stored r taken: elements of every Near Band and Proxy Index `r` array."""
        return sum(n for path, n in self.taken.items() if path.endswith("/r"))


def counted(where: object) -> tuple[StoreReader, Counting]:
    """A reader of the store at `where` whose own chunk cache counts what it is asked for."""
    reader = StoreReader.open(str(where))
    cache = Counting()
    reader._cache = cache
    return reader, cache


def reference_greedy(
    score: npt.NDArray[np.float64], r2: float, kb: float
) -> tuple[npt.NDArray[np.int64], int]:
    """plink2's greedy over the Test Store's variants from the reference r, quantised: each
    variant's lead, and the pairs the leads need, each lead with every hit still open within
    W on its chromosome. Ties go to the row, the engine's order (spec §2)."""
    rows = passing_rows()
    pos = panel_data().variants().pos.astype(np.int64)[rows]
    chrom = np.asarray(panel_data().variants().chrom)[rows]
    ref = support.reference_matrix(support.K, tuple(rows.tolist()))
    held = ~np.isnan(ref)
    r = np.full(ref.shape, np.nan)
    r[held] = dequantise(support.quantise(ref[held]))
    lead = np.full(len(rows), -1, dtype=np.int64)
    needed = 0
    for i in np.lexsort((np.arange(len(rows)), -score)):
        if lead[i] >= 0:
            continue
        lead[i] = i
        open_near = (lead < 0) & (chrom == chrom[i]) & (np.abs(pos - pos[i]) < kb * 1000)
        needed += int(open_near.sum())
        lead[open_near & (r[i] * r[i] > r2)] = i
    return lead, needed


#: What the leads need, on the Test Store at f0e90de: the tip's square read resolved 1,495,403
#: pairs at r² 0.1 (64 times as many), and 33,187 Near Band entries are the leads' whole rows
#: and columns at r² 0.01, below t_far.
NEEDED = {0.1: 23_327, 0.01: 7_246}


@pytest.mark.parametrize("r2", sorted(NEEDED))
def test_clump_resolves_at_most_twice_the_pairs_it_needs(r2: float) -> None:
    rows = passing_rows()
    z = np.abs(panel_data().gwas().z)[rows]
    lead, needed = reference_greedy(z, r2, 3000)
    assert needed == NEEDED[r2], "the leads' candidates, as the tip counted them"
    assert len(np.unique(lead)) > 200, "many leads, each reading its own pairs"
    reader, cache = counted(support.TEST_STORE)
    ours = clumping.clump(reader, alids(rows), z, r2=r2, kb=3000)
    assert list(ours.lead) == alids(rows[lead])
    assert cache.resolved() <= 2 * needed, f"resolved {cache.resolved()} for {needed} needed"
