"""The hg19 rows of a lift collision, and the ledger a build writes of them (#325).

`liftover_fixtures.write_collision_chain` lifts hg19 501 and 1001 both to
1:5501:A:G, and 1501 to 1:6001:C:T alone. The collision suites, the plain-list
suite (#260) and the extraction suite build from these rows and judge the ledger
against these lines, so the fixture and its expected record cannot drift apart.
"""

from __future__ import annotations

from liftover_fixtures import vcf_row

#: The ALID two hg19 loci lift onto, and the one 1501 reaches alone.
COLLIDED = "1:5501:A:G"
CLEAN = "1:6001:C:T"


def named_row(pos: int, ref: str, alt: str, rsid: str, *, es: float = 0.2) -> str:
    """A chr1 GWAS-VCF row whose ID column is ``rsid``."""
    return vcf_row("1", pos, ref, alt, es).replace("\t.\t", f"\t{rsid}\t", 1)


#: hg19 501 and 1001 both lift to 1:5501:A:G; 1501 lifts to 1:6001:C:T alone.
ROWS = [
    named_row(501, "A", "G", "rs_first"),
    named_row(1_001, "A", "G", "rs_second"),
    named_row(1_501, "C", "T", "rs_clean"),
]
#: The ``lift_collisions.tsv`` lines a build that names its sources writes of `ROWS`.
LEDGER = [
    f"{COLLIDED}\t1:501:A:G\trs_first",
    f"{COLLIDED}\t1:1001:A:G\trs_second",
]
