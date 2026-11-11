"""What goes in an LD Store, from the r its producer supplies (spec §3–§7, §12; ogld#17).

PROPOSED, ogld#17 step 1: signatures for the owner to confirm before any producer moves. Nothing
calls this module yet and every body raises NotImplementedError. The annotations use today's
`Encoding` record and int16 arrays; once ogld#16 merges they become its encoding value and its
`Codes`, and this module names no scale, dtype or reserved code of its own.

`StoreWriter` is the seam a producer calls (its proposed methods are `_StoreWriterProposal` in
`ogld/store/writer.py`). This module, behind it, owns what goes in a store (owner, 10 November
2026; `docs/architecture/ownership.md`, "Provenance and what goes in a store"):

- quantising r at the store's encoding (spec §3);
- the Near Band's row lengths and `indptr`, from the positions (spec §4);
- the threshold at the far code floor, judged on codes, for every pair within far_bp (§3, §5);
- both directions of the Proxy Index, partners ascending, one code per pair (spec §5);
- the counts, and the rest of the provenance a store derives from itself (spec §7);
- and validation at `finish`, as today: `validate_group`, all 18 rules (spec §12).

A producer supplies r for blocks of rows in order, and at `finish` what only the source knows
(`ogld.format.Sourced`).

**Memory is bounded by the block and the far_bp window, never the chromosome.** On chr6 of the
whole EUR panel (lane #5's `EUR-k10-t16.store`, n = 771: 602,466 rows, 6,826,805,033 Near Band
codes, 731,767,394 Proxy Index entries), measured from that store's arrays:

- Near Band carry: less than one shard of `r`, 67,108,864 codes: 128 MiB.
- Proxy Index carry: the pairs whose later row is not yet written, at most those spanning one
  row within far_bp: 33,869,608 pairs at 10 B, 323 MiB, at the MHC.
- Proxy Index output: less than one shard of `partner` and `r` (1,048,576 entries), 6 MiB, and
  one block's rows, at most 2,612,379 entries, 25 MiB.
- Per-row arrays: both `indptr`s and the stops, n_c + 1 int64 each: 18 MiB.
- Each block in flight: its codes and pairs, at most 8,878,531 and 1,468,811 in a 256-row
  block: 31 MiB.

About 0.5 GiB, plus 31 MiB for each block a producer holds between cutting and writing it. Today
the writer holds chr6's band twice, 2 × 13.65 GB, and the builder lays its Proxy Index out whole:
365,883,697 pairs at 10 B, 731,767,394 entries at 10 B and a 2.9 GB sort order, 13.9 GB at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import zarr

from ogld.format import Counts, Encoding, Provenance, Sourced


@dataclass(frozen=True)
class Cut:
    """A block of rows cut from its producer's r: what `StoreWriter.write` writes.

    `first` is the block's first global row and `rows` how many it holds, all on one chromosome.
    `near` holds their Near Band codes, row after row (spec §4). `lo`, `hi` and `codes` are
    their Proxy Index pairs, as local rows lo < hi: each partner hi within far_bp of row lo whose
    code reaches the far code floor, once, with its one code (spec §3, §5), ascending by
    (lo, hi). `far_candidates` counts the block's pairs more than near_bp and at most far_bp
    apart, all judged, and `far_pairs` those of them kept (spec §7).

    A producer neither builds nor reads one: it hands what `StoreWriter.cut` returned to
    `StoreWriter.write`.
    """

    first: int
    rows: int
    near: npt.NDArray[np.int16]
    lo: npt.NDArray[np.int32]
    hi: npt.NDArray[np.int32]
    codes: npt.NDArray[np.int16]
    far_candidates: int
    far_pairs: int


class Cutter:
    """What every block of one store is cut by, fixed when the store is created: the Variant
    Set's positions and chromosome ranges, near_bp and far_bp, the encoding, and t_far with its
    far code floor (spec §3–§5). Nothing in it changes after `create`, so blocks are cut on any
    thread, in any order."""

    def __init__(
        self,
        position: npt.NDArray[np.int64],
        ranges: dict[str, tuple[int, int]],
        distances: tuple[int, int],
        encoding: Encoding,
        adjusted: tuple[int, int],
    ) -> None:
        """`adjusted` is (n, k): t_far = `ogld.format.t_far(n, k)`, and the floor is the
        encoding's for it. Blocks are thresholded as they arrive, so both are fixed before the
        first; an n and k t_far refuses (n − k − 1 ≤ q) are refused here, before any row."""
        raise NotImplementedError

    @property
    def t_far(self) -> float:
        raise NotImplementedError

    @property
    def floor(self) -> int:
        """The far code floor: a pair within far_bp is in the Proxy Index exactly when
        |code| reaches it (spec §3, §5)."""
        raise NotImplementedError

    def cut(self, first: int, r: npt.NDArray[np.floating[Any]]) -> Cut:
        """`StoreWriter.cut`'s work, which says what `r` holds.

        For each row l of the block it reads r(l, m) for l < m < far stop(l) only. Every m
        within near_bp is quantised, in order, for the Near Band. Beyond near_bp only a pair
        whose |r| can reach the floor is quantised, to judge it on its code; a value that is not
        a number is such a pair, so quantising refuses it rather than let it pass as below the
        floor. So a pair the producer did not measure is refused, never stored or read as a
        Bound (spec §8).
        """
        raise NotImplementedError


class NearBandStream:
    """One chromosome's Near Band, written as its blocks arrive (spec §4, §6).

    Lane #4's measured design (comment on ogld#17). `r`'s length, the sum of the row lengths,
    is known from the positions before any code arrives. So `indptr` is written whole, and `r`
    is created at its full length, when the chromosome's first block arrives. Codes are written
    in whole shards of `r`, the only writes zarr makes without reading a shard back (zarr 3.4.2,
    `_is_complete_shard_write`), and less than one shard is carried to the next block. The last
    shard is written when the last row arrives.
    """

    def __init__(self, group: zarr.Group, lengths: npt.NDArray[np.int64]) -> None:
        raise NotImplementedError

    def add(self, codes: npt.NDArray[np.int16]) -> None:
        """Append the next rows' codes; write each shard they complete."""
        raise NotImplementedError

    def close(self) -> None:
        """Write the carried codes; refuse a band whose codes are not the sum of its lengths."""
        raise NotImplementedError


class ProxyIndexStream:
    """One chromosome's Proxy Index, both directions, written row by row as blocks arrive (§5).

    Row l holds every pair (m, l) with m < l, from earlier blocks, then every pair (l, m) with
    m > l, from l's own block. So row l is complete when its own block arrives, and each block's
    rows are written then: partners below ascending, then partners above, each with its pair's
    one code. The block's pairs whose later row is still to come are carried: at most the pairs
    spanning one row within far_bp, however long the chromosome.

    `partner` and `r` are written in whole shards. Their length is known only at the
    chromosome's end, so they grow shard by shard (`zarr.Array.resize`); an index under one
    shard is written at its clipped shape at the end (spec §6). Step 1 checked on zarr 3.4.2
    that a grown array's shard files and metadata are byte-identical to the same array written
    whole, and that no whole-shard write reads a shard back. `indptr` is held and written at the
    end.
    """

    def __init__(self, group: zarr.Group, rows: int) -> None:
        raise NotImplementedError

    def add(self, cut: Cut) -> None:
        """Write the rows of `cut`, which continue the rows already written, and carry its
        pairs whose later row is still to come."""
        raise NotImplementedError

    def close(self) -> int:
        """Write `indptr` and the last entries; return the entries written. A carried pair left
        over would be a row never written, and is refused."""
        raise NotImplementedError


@dataclass
class Tally:
    """The counts provenance records (spec §7), added up block by block."""

    near_pairs: int = 0
    far_candidates: int = 0
    far_pairs: int = 0
    index_entries: int = 0

    def add(self, cut: Cut) -> None:
        raise NotImplementedError

    def counts(self) -> Counts:
        raise NotImplementedError


def provenance(sourced: Sourced, cutter: Cutter, n_variants: int, tally: Tally) -> Provenance:
    """The store's provenance: the producer's `sourced`, and what the store derives from itself:
    `format_version`, `encoding`, `n_variants`, `near_bp`, `far_bp`, `t_far`, `t_far_q`,
    `far_code_floor` and `counts` (spec §7). Refuses a `sourced` whose `panel.n` or
    `adjustment.k` is not the n and k the store was thresholded with, naming both."""
    raise NotImplementedError
