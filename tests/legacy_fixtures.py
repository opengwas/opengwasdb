"""Produce a format 0.1.0 (Zarr v2) Dense store for the converter's tests (#247).

From #247 the builders write **only** 0.2.0 (ADR 0041 §3): Zarr v3 with the
sharding codec.  That leaves the Dense converter (#245), whose job is to turn a
0.1.0 release into a 0.2.0 one, without a 0.1.0 source to test against -- the
builders that used to write one now write the target format.

`relayout_dense_as_0_1_0` bridges that gap without keeping a retired encoder
alive in the package.  It takes a release the *current* builders wrote and
re-lays it out as 0.1.0: every array is read through zarr and written back one
chunk per file into a Zarr v2 group, carrying the same values, dtype, fill,
inner chunk and Blosc configuration, and the three layout recordings are
rewritten to the v2 compressor.  The result is a genuine 0.1.0 Store Release --
the layout the converter is defined against -- so a converter test exercises the
real thing rather than a hand-built approximation.

It is a test helper, not a migration: it lives in `tests/` and nothing in
`opengwasdb/` imports it.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.index.sqlite import get_metadata, set_metadata
from opengwasdb.store.arrays import (
    COMPRESSOR_RECORD,
    compressor,
    create_array,
    open_group,
    open_group_for_write,
    require_group,
    role_for_array_path,
)

#: The format_version `relayout_dense_as_0_1_0` stamps.
LEGACY_FORMAT_VERSION = "0.1.0"


def _iter_arrays(root: Any, prefix: str = "") -> Iterator[str]:
    for name in sorted(root.array_keys()):
        yield prefix + name
    for name in sorted(root.group_keys()):
        yield from _iter_arrays(root[name], prefix + name + "/")


def _iter_groups(root: Any, prefix: str = "") -> Iterator[str]:
    """Every group path under `root`, outermost first."""
    for name in sorted(root.group_keys()):
        path = prefix + name
        yield path
        yield from _iter_groups(root[name], path + "/")


def _v2_codec(array: Any) -> Any:
    """The numcodecs codec for a source array, or `None` if it is uncompressed.

    The Store format has one Blosc configuration, so a compressed array maps to
    the seam's `compressor()`; the exception/overflow tables are deliberately
    uncompressed and map to `None`.
    """
    return None if not tuple(array.compressors or ()) else compressor()


def _read_plan(root: Any) -> tuple[list[tuple[Any, ...]], dict[str, dict[str, Any]]]:
    """Snapshot every array and group attr of a Dense `data.zarr` before rewrite."""
    plan: list[tuple[Any, ...]] = []
    z_chunk: tuple[int, ...] | None = None
    for path in _iter_arrays(root):
        array = root[path]
        role = role_for_array_path(path)
        if role is None:
            raise AssertionError(
                f"{path}: no ArrayRole; relayout_dense_as_0_1_0 only handles Dense releases"
            )
        chunk = tuple(int(size) for size in array.chunks)
        if path == "z":
            z_chunk = chunk
        plan.append(
            (
                path,
                role,
                tuple(int(size) for size in array.shape),
                str(array.dtype),
                array.fill_value,
                chunk,
                _v2_codec(array),
                np.asarray(array[:]),
            )
        )
    if z_chunk is None:
        raise AssertionError("data.zarr has no 'z'; not a Dense release")
    groups = {path: dict(root[path].attrs) for path in _iter_groups(root)}
    return plan, groups


def _write_v2_tree(
    destination_data: Path,
    plan: list[tuple[Any, ...]],
    groups: dict[str, dict[str, Any]],
    root_attrs: dict[str, Any],
) -> None:
    shutil.rmtree(destination_data)
    v2 = open_group_for_write(destination_data, "w", zarr_format=2)
    for group_path in groups:
        require_group(v2, group_path)
    for path, role, _shape, dtype, fill, chunk, codec, values in plan:
        parent_path, _, leaf = path.rpartition("/")
        group = v2 if not parent_path else v2[parent_path]
        create_array(
            group,
            leaf,
            role,
            data=values,
            dtype=dtype,
            fill_value=fill,
            compressor=codec,
            inner_chunk=chunk,
            overwrite=True,
        )
    for group_path, attrs in groups.items():
        if attrs:
            v2[group_path].attrs.update(attrs)
    root_attrs.pop("shard_shape", None)
    root_attrs.pop("zarr_format", None)
    root_attrs["compressor"] = COMPRESSOR_RECORD
    v2.attrs.update(root_attrs)


def _rewrite_recordings(release: Path, z_chunk: tuple[int, ...]) -> None:
    """Point the manifest, the `index.sqlite` blob and the root attrs at v2."""
    manifest_path = release / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["format_version"] = LEGACY_FORMAT_VERSION
    dense = dict(data.get("provenance", {}).get("dense", {}))
    dense["chunk_shape"] = list(z_chunk)
    dense["compressor"] = COMPRESSOR_RECORD
    dense.pop("shard_shape", None)
    dense.pop("zarr_format", None)
    data["provenance"] = {**data.get("provenance", {}), "dense": dense}
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    connection = sqlite3.connect(str(release / "index.sqlite"))
    connection.row_factory = sqlite3.Row
    with connection:
        blob = get_metadata(connection, "dense", default=None)
        if isinstance(blob, dict):
            blob = dict(blob)
            blob["chunk_shape"] = list(z_chunk)
            blob["compressor"] = COMPRESSOR_RECORD
            blob.pop("shard_shape", None)
            blob.pop("zarr_format", None)
            set_metadata(connection, "dense", blob)
        connection.commit()
    connection.close()


def relayout_dense_as_0_1_0(source: str | Path, destination: str | Path) -> Path:
    """Write a 0.1.0 copy of a Dense release the current builders wrote.

    `source` is copied, then its `data.zarr` is replaced by a Zarr v2 tree
    holding the same arrays; `manifest.json`, the `index.sqlite` `dense` blob and
    the root attrs are restamped to 0.1.0 and the v2 compressor record.  The
    source is never written.
    """
    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"{destination} already exists")
    shutil.copytree(source, destination)
    root = open_group(source / "data.zarr", "r")
    plan, groups = _read_plan(root)
    root_attrs = dict(root.attrs)
    z_chunk = next(entry[5] for entry in plan if entry[0] == "z")
    _write_v2_tree(destination / "data.zarr", plan, groups, root_attrs)
    _rewrite_recordings(destination, z_chunk)
    return destination
