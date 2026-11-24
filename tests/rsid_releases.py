"""Fixture Store Releases whose rows name rsIDs (#465).

No release `store_reader_digests.build_releases` builds carries an rsID, and a
query by rsID on a release that names none is an empty answer that proves nothing
about where it was resolved. `rewrite_with_rsids` rewrites a release's Variant
Table and search indexes (a Hybrid's Dense Component too) so every row, or a
chosen set of rows, names one; the `rsid_releases` session fixture
(`conftest.py`) builds the six releases once and gives them rsIDs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from opengwasdb.variants.axis import VariantAxis, write_variant_axis
from opengwasdb.variants.normalise import CanonicalVariant

RELEASE_NAMES = [
    f"{layout}_{state}"
    for layout in ("dense", "hybrid", "ragged")
    for state in ("completed", "observed")
]


def rsid_names(alids: Sequence[str], planted: Sequence[str | None] = ()) -> dict[str, str]:
    """The rsID of each of `alids` (one per row, in row order): row `i` is `rs{1000 + i}`,
    except the first rows, which `planted` names in order, None leaving a row unnamed."""
    names = {alid: f"rs{1000 + i}" for i, alid in enumerate(alids)}
    for alid, name in zip(alids, planted, strict=False):
        if name is None:
            del names[alid]
        else:
            names[alid] = name
    return names


def rewrite_with_rsids(
    store: Path,
    rsid_by_alid: dict[str, str] | None = None,
    *,
    planted: Sequence[str | None] = (),
) -> dict[str, str]:
    """Rewrite one Variant Table and its search indexes with an rsID per row.

    `rsid_by_alid` names them; by default `rsid_names` does, with `planted` naming the
    first rows. Returns the names used.
    """
    axis = VariantAxis(store)
    try:
        records = axis.all()
    finally:
        axis.close()
    assert len(records) >= len(planted), f"{store} has too few rows to plant {len(planted)}"
    names = rsid_by_alid or rsid_names([r.alid for r in records], planted)
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


def with_rsids(
    store: Path,
    rsid_by_alid: dict[str, str] | None = None,
    *,
    planted: Sequence[str | None] = (),
) -> dict[str, str]:
    """Give a release's rows rsIDs, a Hybrid's Dense Component too (it resolves its own)."""
    names = rewrite_with_rsids(store, rsid_by_alid, planted=planted)
    if (store / "dense" / "variants.tsv.gz").exists():
        rewrite_with_rsids(store / "dense", names)
    return names
