"""Small Dense, Ragged and Hybrid Store Releases with one association's value planted.

Shared by the query and validation tests of the presence rule (ADR 0070). A plant
writes every copy a read may use: a Ragged component's by-variant index holds its
own copy of each row.
"""

from __future__ import annotations

from json import dumps, loads
from pathlib import Path
from shutil import copytree

import numpy as np
import zarr
from conftest import SOURCE_HEADER, SOURCE_ROWS
from dense_sources import build_dense_fixture
from test_query_metadata_reads import _BUILDERS

from opengwasdb.encoding.planes import DenseZPlane
from opengwasdb.layouts.hybrid.layout import dense_component_path, dense_to_shared_path
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.model.manifest import StoreManifest

#: A float16 NaN other than `np.nan`'s 0x7e00: the payload spec §15 forbids.
PAYLOAD_NAN = np.array([0x7E01], np.uint16).view(np.float16)[0]

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
    build_dense_fixture(
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


#: The `z` group of a Dense release (or a Hybrid's Dense Component), and of a
#: Ragged one (or a Hybrid's Overflow), relative to that component's Store root.
DENSE_Z_GROUP = "data.zarr"
RAGGED_Z_GROUP = "data.zarr/ragged"


def float_plane_z(store: Path, group_path: str = DENSE_Z_GROUP) -> zarr.Group:
    """Rewrite one component's `z` plane as the float plane its manifest then declares.

    Every plan this build decides writes Z as `int16_fixed` (`StoreEncoding.decide`),
    an integer plane whose missing marker is a sentinel and which holds no NaN at all,
    so a float16 Z has to be made here before a payload NaN can be planted in one. The
    same edit `test_query_presence` makes to a Ragged Store for its top-hit scan. The
    values, the chunks and -- where the plane is sharded -- the shards are kept, so the
    Store keeps the geometry a check derives from it. Returns the reopened group.
    """
    manifest_path = store / "manifest.json"
    manifest = loads(manifest_path.read_text(encoding="utf-8"))
    group = zarr.open_group(store / group_path, mode="r+")
    plane = group["z"]
    raw = np.asarray(plane[:])
    encoding = StoreManifest.load(store).encoding
    if raw.ndim == 2:
        decoded = DenseZPlane.open(group, encoding).band(0, raw.shape[0])
    else:
        decoded = RaggedCSRReader(store, encoding).z_slice(0, len(raw))
    chunks, shards = plane.chunks, plane.shards
    del group["z"]
    group.create_array(
        "z", data=decoded.astype(np.float16), chunks=chunks, shards=shards, dtype=None
    )
    manifest["encoding"]["z"] = {"kind": "float16"}
    manifest_path.write_text(dumps(manifest), encoding="utf-8")
    reopened = zarr.open_group(store / group_path, mode="r+")
    assert reopened["z"].dtype == np.dtype("float16"), "the plane must now be a float one"
    return reopened


def plant_payload_nan(group: zarr.Group, index: tuple[int, ...] | int) -> None:
    """Write `PAYLOAD_NAN` into one cell of a float16 `z` plane, and check it stuck."""
    group["z"][index] = PAYLOAD_NAN
    stored = np.asarray(group["z"][index], dtype=np.float16)
    assert int(stored.view(np.uint16)) == 0x7E01


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
