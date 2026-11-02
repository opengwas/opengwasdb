"""Small Dense, Ragged and Hybrid Store Releases with one association's value planted.

Shared by the query and validation tests of the presence rule (ADR 0070). A plant
writes every copy a read may use: a Ragged component's by-variant index holds its
own copy of each row.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copytree

import numpy as np
import zarr
from conftest import SOURCE_HEADER, SOURCE_ROWS
from test_query_metadata_reads import _BUILDERS

from opengwasdb.build.observed import build_dense_observed_from_sources
from opengwasdb.layouts.hybrid.layout import dense_component_path, dense_to_shared_path

#: (builder, Analysis, shared variant index, analysis index) of one present cell.
CELLS = {
    "dense": ("dense", "a1", 1, 0),
    "ragged": ("ragged", "trait_a", 1, 0),
    "hybrid_on_panel": ("hybrid", "trait_a", 2, 0),
    "hybrid_overflow": ("hybrid", "trait_a", 1, 0),
}


def build_stores(base: Path) -> dict[str, Path]:
    """One Store Release of each layout, built under `base`."""
    source = base / "associations.tsv"
    source.write_text(SOURCE_HEADER + "\n" + "\n".join(SOURCE_ROWS) + "\n", encoding="utf-8")
    dense = base / "store.opengwasdb"
    build_dense_observed_from_sources(
        [source], dense, store_id="f", release_id="v", reference_assembly="GRCh37"
    )
    built = {}
    for name, build in _BUILDERS.items():
        (base / name).mkdir()
        built[name] = build(base / name, dense)
    return built


def csr_position(group: zarr.Group, variant: int, analysis: int) -> int:
    offsets = np.asarray(group["offsets"][:], dtype=np.int64)
    start, end = int(offsets[analysis]), int(offsets[analysis + 1])
    found = np.flatnonzero(np.asarray(group["variant_index"][start:end]) == variant)
    return start + int(found[0])


def plant_csr(store: Path, variant: int, analysis: int, se: float) -> None:
    """The SE of one CSR row, in the sequence and in its by-variant copy."""
    ragged = zarr.open_group(store / "data.zarr" / "ragged", mode="r+")
    ragged["se"][csr_position(ragged, variant, analysis)] = se
    index = ragged["by_variant"]
    start, end = (int(x) for x in index["offsets"][variant : variant + 2])
    rows = np.flatnonzero(np.asarray(index["analysis_index"][start:end]) == analysis)
    index["se"][start + int(rows[0])] = se


def plant(stores: dict[str, Path], case: str, se: float, tmp_path: Path) -> Path:
    builder, _, variant, analysis = CELLS[case]
    store = Path(copytree(stores[builder], tmp_path / "store"))
    if case == "dense":
        zarr.open_group(store / "data.zarr", mode="r+")["se"][variant, analysis] = se
    elif case == "hybrid_on_panel":
        panel_row = int(np.flatnonzero(np.load(dense_to_shared_path(store)) == variant)[0])
        grid = zarr.open_group(dense_component_path(store) / "data.zarr", mode="r+")
        grid["se"][panel_row, analysis] = se
    else:
        plant_csr(store, variant, analysis, se)
    return store
