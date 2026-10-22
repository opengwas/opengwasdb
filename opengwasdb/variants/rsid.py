"""The rs identifier rule every GWAS-SSF read shares (issue #306).

How a source's identifier cell becomes the one rs identifier a Store Variant
Table records lives here, not in one reader, so the variant projection and the
full-row parser cannot disagree about it.
"""

from __future__ import annotations


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
