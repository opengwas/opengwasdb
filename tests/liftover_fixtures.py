"""Chain fixtures shared by the liftover and BESD-build tests (issue #284).

A chain file is the only place a lift's strand comes from, so a test about
strand handling must supply one whose strands are known by construction rather
than read back from the code under test.
"""

from __future__ import annotations

from pathlib import Path

from opengwasdb.query import StoreQuery

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


#: One fixture row per orientation case: name, source contig, hg19 position,
#: REF, ALT, the hg38 ALID the chain arithmetic gives, and the z and EAF the
#: projected alleles require. ES/SE is 0.2/0.1 for the effect allele `A`, and
#: AF is its frequency 0.2, so the source-canonical statistic is z = +2, eaf =
#: 0.2 in every row; the reverse-strand rows store -2 and 0.8 because `A`
#: becomes `T`, which is `A2` of the projected pair.
STRAND_ROWS: tuple[tuple[str, str, int, str, str, str, float, float], ...] = (
    ("reverse", "2", 1_000, "G", "A", "2:999001:C:T", -2.0, 0.8),
    ("reverse_palindromic", "2", 2_000, "T", "A", "2:998001:A:T", -2.0, 0.8),
    ("forward", "3", 1_000, "G", "A", "3:1000:A:G", 2.0, 0.2),
)


def strand_vcf_rows() -> list[str]:
    """`STRAND_ROWS` as GWAS-VCF body rows (REF, ALT, `ES:SE:AF`)."""
    return [
        f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tES:SE:AF\t0.2:0.1:0.2\n"
        for _name, chrom, pos, ref, alt, _alid, _z, _eaf in STRAND_ROWS
    ]


def assert_strand_statistics(query: StoreQuery, analysis_id: str) -> None:
    """Assert each `STRAND_ROWS` ALID is stored in `query`'s store as the
    projection requires."""
    table = {variant["alid"] for variant in query.variants_table().values()}
    for name, _chrom, _pos, _ref, _alt, alid, expected_z, expected_eaf in STRAND_ROWS:
        assert alid in table, f"{name}: {alid} missing from {sorted(table)}"
        result = query.lookup([alid], [analysis_id])
        assert len(result["z"]) == 1, f"{name}: no association at {alid}"
        assert abs(float(result["z"][0]) - expected_z) < 0.01, name
        assert abs(float(result["eaf"][0]) - expected_eaf) < 0.01, name
