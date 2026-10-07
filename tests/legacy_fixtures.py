"""Produce a format 0.1.0 (Zarr v2) release for the converters' tests (#247, #248).

From #247 the builders write **only** 0.2.0 (ADR 0041 §3): Zarr v3 with the
sharding codec.  That leaves the converters, whose job is to turn a 0.1.0
release into a 0.2.0 one, without a 0.1.0 source to test against -- the builders
that used to write one now write the target format.

`relayout_as_0_1_0` bridges that gap without keeping a retired encoder alive in
the package.  It takes a release the *current* builders wrote and re-lays it out
as 0.1.0: every array is read through zarr and written back one chunk per file
into a Zarr v2 group, carrying the same values, dtype, fill, inner chunk and
Blosc configuration, and each release's layout recordings are rewritten to the
v2 compressor.  The result is a genuine 0.1.0 Store Release -- the layout the
converters are defined against -- so a converter test exercises the real thing
rather than a hand-built approximation.

It handles a whole release, not one tree: a Hybrid release is two Store Releases
(the outer one and its nested Dense Component at `dense/`), so every
`data.zarr` under the release is re-laid out and every `manifest.json` and
`index.sqlite` beside one is downgraded.  A Ragged root carries no Dense
recording and gets none added.

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

#: The format_version `relayout_as_0_1_0` stamps.
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


def _read_plan(
    root: Any,
) -> tuple[list[tuple[Any, ...]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Snapshot every array, group attr and root attr of a `data.zarr`."""
    plan: list[tuple[Any, ...]] = []
    z_chunk: tuple[int, ...] | None = None
    for path in _iter_arrays(root):
        array = root[path]
        role = role_for_array_path(path)
        if role is None:
            raise AssertionError(
                f"{path}: no ArrayRole; relayout_as_0_1_0 cannot lay out this release"
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
    groups = {path: dict(root[path].attrs) for path in _iter_groups(root)}
    root_attrs = dict(root.attrs)
    if z_chunk is not None:
        root_attrs["chunk_shape"] = list(z_chunk)
    return plan, groups, root_attrs


def _write_v2_tree(
    destination_data: Path,
    plan: list[tuple[Any, ...]],
    groups: dict[str, dict[str, Any]],
    root_attrs: dict[str, Any],
) -> tuple[int, ...] | None:
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
    z_chunk = next((entry[5] for entry in plan if entry[0] == "z"), None)
    had_compressor = "compressor" in root_attrs
    root_attrs.pop("shard_shape", None)
    root_attrs.pop("zarr_format", None)
    root_attrs.pop("compressor", None)
    if z_chunk is not None:
        root_attrs["chunk_shape"] = list(z_chunk)
    if had_compressor:
        root_attrs["compressor"] = COMPRESSOR_RECORD
    v2.attrs.update(root_attrs)
    return z_chunk


def _downgrade_recording(block: dict[str, Any], z_chunk: tuple[int, ...] | None) -> dict[str, Any]:
    """A recorded layout as 0.1.0 records it: inner chunk, v2 compressor."""
    block = dict(block)
    block["compressor"] = COMPRESSOR_RECORD
    block.pop("shard_shape", None)
    block.pop("zarr_format", None)
    if z_chunk is not None:
        block["chunk_shape"] = list(z_chunk)
    return block


def _rewrite_manifest(manifest_path: Path, z_chunk: tuple[int, ...] | None) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["format_version"] = LEGACY_FORMAT_VERSION
    provenance = dict(data.get("provenance", {}))
    for key in ("dense", "hybrid"):
        block = provenance.get(key)
        if isinstance(block, dict) and "chunk_shape" in block:
            provenance[key] = _downgrade_recording(block, z_chunk)
    data["provenance"] = provenance
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_index(index_path: Path, z_chunk: tuple[int, ...] | None) -> None:
    if not index_path.is_file():
        return
    connection = sqlite3.connect(str(index_path))
    connection.row_factory = sqlite3.Row
    with connection:
        has_metadata = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'"
        ).fetchone()
        if has_metadata is None:
            return
        blob = get_metadata(connection, "dense", default=None)
        if isinstance(blob, dict):
            set_metadata(connection, "dense", _downgrade_recording(blob, z_chunk))
        connection.commit()
    connection.close()


def relayout_as_0_1_0(source: str | Path, destination: str | Path) -> Path:
    """Write a 0.1.0 copy of a release the current builders wrote.

    `source` is copied, then every `data.zarr` under the copy is replaced by a
    Zarr v2 tree holding the same arrays, and every `manifest.json` and
    `index.sqlite` beside one is restamped to 0.1.0 and the v2 compressor
    record.  The source is never written.
    """
    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"{destination} already exists")
    shutil.copytree(source, destination)
    for data_zarr in sorted(destination.rglob("data.zarr")):
        root = open_group(data_zarr, "r")
        plan, groups, root_attrs = _read_plan(root)
        z_chunk = _write_v2_tree(data_zarr, plan, groups, root_attrs)
        release = data_zarr.parent
        _rewrite_manifest(release / "manifest.json", z_chunk)
        _rewrite_index(release / "index.sqlite", z_chunk)
    return destination


def relayout_dense_as_0_1_0(source: str | Path, destination: str | Path) -> Path:
    """`relayout_as_0_1_0` under its original name; a Dense release is one root."""
    return relayout_as_0_1_0(source, destination)
