"""Dense releases whose EAF exception table spans many pages, for the table's readers.

Shared by the tests of the rules that read a release's EAF exception table (#387)
and of the one read a validation pass makes of it (#314): a residual release in
which every twentieth variant carries, in its last Analysis, a frequency no residual
coding can express, so its table holds one exact entry for each.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from dense_sources import build_dense_from_records

from opengwasdb.encoding.codec import EAF_EXCEPTION_INDEX, EAF_EXCEPTION_VALUE
from opengwasdb.layouts.dense.build import NormalisedAssociation
from opengwasdb.layouts.dense.indexed_subsets import build_indexed_subset
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store.arrays import create_array
from opengwasdb.variants import CanonicalVariant

#: The subset every one of these releases publishes.
SUBSET = "hm3"
#: Every twentieth variant carries, in one Analysis, a frequency no residual
#: coding can express: 3,280 exact cells on the wide fixture and 400 on the small
#: one, enough to span several pages without making the residual plan lose to a
#: plain float32 plane.
EXCEPTION_EVERY = 20
SMALL_VARIANTS = 4_000
ANALYSES = 2


def table_records(variants: int) -> list[NormalisedAssociation]:
    freqs = np.linspace(0.05, 0.95, variants, dtype=np.float32)
    records: list[NormalisedAssociation] = []
    for col in range(ANALYSES):
        for row in range(variants):
            eaf = float(freqs[row])
            if col == ANALYSES - 1 and row % EXCEPTION_EVERY == 0:
                eaf = 0.999  # far from every baseline: an exception, stored exactly
            records.append(
                NormalisedAssociation(
                    analysis_id=f"a{col}",
                    variant=CanonicalVariant("1", row + 1, "A", "G"),
                    z=1.0,
                    se=0.2,
                    eaf=eaf,
                )
            )
    return records


def build_table_store(path: Path, variants: int, store_id: str) -> Path:
    build_dense_from_records(
        table_records(variants),
        path,
        store_id=store_id,
        release_id=f"{store_id}-v1",
        reference_assembly="GRCh38",
        chunk_shape=(1_000, ANALYSES),
    )
    return path


def publish_every_variant(store: Path, variants: int) -> Path:
    """Publish the subset of every variant the release holds."""
    listing = store.parent / f"{SUBSET}.alid.txt"
    listing.write_text(
        "\n".join(f"1:{row + 1}:A:G" for row in range(variants)) + "\n", encoding="utf-8"
    )
    build_indexed_subset(store, SUBSET, listing, reference_assembly="GRCh38")
    return store


def rewrite_primary_table(store: Path, positions: np.ndarray, values: np.ndarray) -> None:
    """Replace the release's own exception table, through the array seam."""
    root = zarr.open_group(str(store / "data.zarr"), mode="r+")
    for name, data, dtype in (
        (EAF_EXCEPTION_INDEX, positions, "int64"),
        (EAF_EXCEPTION_VALUE, values, "float32"),
    ):
        del root[name]
        create_array(
            root,
            name,
            store_arrays.ArrayRole.EXCEPTION_TABLE,
            data=np.asarray(data, dtype=dtype),
            compressor=None,
        )
