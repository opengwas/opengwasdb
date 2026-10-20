"""Chain fixtures shared by the liftover and BESD-build tests (issue #284).

A chain file is the only place a lift's strand comes from, so a test about
strand handling must supply one whose strands are known by construction rather
than read back from the code under test.
"""

from __future__ import annotations

from pathlib import Path

#: Length of both fixture chromosomes, in bp. Positions are held inside it.
STRAND_CHAIN_SIZE = 1_000_000


def write_two_strand_chain(path: Path) -> Path:
    """A real UCSC chain `pyliftover` reads itself: chr2 reverse, chr3 forward.

    Both chains cover a whole chromosome, so the mapping is the identity on
    chr3 and the mirror image on chr2: a 1-based hg19 position ``p`` lifts to
    ``STRAND_CHAIN_SIZE - p + 1`` with every allele complemented. The reverse
    chain is written with the target's own (reverse-complemented) coordinates,
    as the UCSC format requires: ``qStart 0``/``qEnd size`` with ``qStrand -``.
    """
    size = STRAND_CHAIN_SIZE
    path.write_text(
        f"chain 1000 chr2 {size} + 0 {size} chr2 {size} - 0 {size} 1\n{size}\n\n"
        f"chain 1000 chr3 {size} + 0 {size} chr3 {size} + 0 {size} 2\n{size}\n\n",
        encoding="utf-8",
    )
    return path
