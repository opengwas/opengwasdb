"""Fixture Store Releases whose rows name rsIDs (#465).

No release `store_reader_digests.build_releases` builds carries an rsID, and a
query by rsID on a release that names none is an empty answer that proves nothing
about where it was resolved. `rewrite_with_rsids` rewrites a release's Variant
Table and search indexes (a Hybrid's Dense Component too) so every row, or a
chosen set of rows, names one; the `rsid_releases` session fixture
(`conftest.py`) builds the six releases once and gives them rsIDs.
"""

from __future__ import annotations

from pathlib import Path

from opengwasdb.variants.axis import VariantAxis, write_variant_axis
from opengwasdb.variants.normalise import CanonicalVariant

RELEASE_NAMES = [
    f"{layout}_{state}"
    for layout in ("dense", "hybrid", "ragged")
    for state in ("completed", "observed")
]


def rewrite_with_rsids(store: Path, rsid_by_alid: dict[str, str] | None = None) -> dict[str, str]:
    """Rewrite one Variant Table and its search indexes with an rsID per row.

    `rsid_by_alid` names them; by default row `i` is `rs{1000 + i}`. Returns the names used.
    """
    axis = VariantAxis(store)
    try:
        records = axis.all()
    finally:
        axis.close()
    names = rsid_by_alid or {r.alid: f"rs{1000 + r.variant_index}" for r in records}
    write_variant_axis(
        store,
        [
            CanonicalVariant(r.chromosome, r.position, r.effect_allele, r.other_allele)
            for r in records
        ],
        names,
        [r.source_alid for r in records],
    )
    return names


def with_rsids(store: Path, rsid_by_alid: dict[str, str] | None = None) -> dict[str, str]:
    """Give a release's rows rsIDs, a Hybrid's Dense Component too (it resolves its own)."""
    names = rewrite_with_rsids(store, rsid_by_alid)
    if (store / "dense" / "variants.tsv.gz").exists():
        rewrite_with_rsids(store / "dense", names)
    return names
