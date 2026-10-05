"""Convert a Dense Observed-Only Store Release to format 0.2.0 (Zarr v3, sharded).

Format 0.1.0 is Zarr v2 with one chunk per file; 0.2.0 is Zarr v3 with the
sharding codec, so the unit a query reads (the *inner chunk*) is decoupled from
the unit stored as a file (the *shard*), and the Dense Analysis-axis inner chunk
can narrow without multiplying the file count (epic #240, ADR 0057).  The
conversion does **not** re-encode any value: every array is read as raw stored
codes and written as the same codes, so the derived release holds bit-identical
values under the new physical layout.

Design rules, all of them deliberately narrow:

* **Dense Observed-Only only.**  Ragged, Hybrid and Dense Reference-Completed
  releases are refused *by name*, saying #248 adds them.  A 0.2.0 source is
  refused as already converted.  Nothing is guessed.
* **Every array is mapped to an `ArrayRole` by its path** (`role_for_array_path`,
  the seam's path -> role table).  An array with no role fails the conversion;
  it is never copied with a default layout.
* **Inner chunk and shard come from the seam.**  `chunk_layout` gives the inner
  chunk, `shard_layout` the shard; the converter keeps no private copy of either
  policy, so #247's builders and this tool cannot disagree.
* **Each destination shard is written whole, once.**  Shard-writing tasks are
  distributed over a process pool, and no two workers own one shard.
* **Bit-exact or nothing.**  `verify_conversion` compares every array's path,
  shape, dtype, fill value and stored values (raw bytes, so NaN payloads count)
  against the source before the staged release is validated and published.  A
  mismatch raises, the staging directory is discarded, and nothing is published.
* **The source is never written**, a destination that exists is refused, and the
  result is a new release: fresh `release_id` and `created_at`, `store_id` kept,
  a `zarr_v3_conversion` provenance block, and `overview.html` regenerated.

The module is the logic; ``scripts/convert_store_to_0_2_0.py`` is the thin CLI.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.index.sqlite import get_metadata, set_metadata
from opengwasdb.layouts.dense.overview import write_overview_html
from opengwasdb.model.analyses import read_analyses
from opengwasdb.store.arrays import (
    COMPRESSOR_RECORD,
    DENSE_CHUNK_SHAPE,
    DENSE_SHARD_SHAPE,
    SHARDED_COMPRESSOR_RECORD,
    ArrayRole,
    chunk_layout,
    create_array,
    open_group,
    open_group_for_write,
    require_group,
    role_for_array_path,
    shard_layout,
    sharded_compressor,
)
from opengwasdb.store.open import (
    CURRENT_FORMAT_VERSION,
    SHARDED_FORMAT_VERSION,
    OpenGWASDBStore,
    StagedRelease,
)
from opengwasdb.validation import validate_store

#: The only layout the converter understands until #248.
DENSE_OBSERVED_ONLY = "dense"

#: The `ArrayRole`s whose inner chunk narrows on the Analysis axis and whose
#: shard is the conversion's `(V_s, A_s)` parameter.  The manifest's recorded
#: `chunk_shape` describes these planes.
_DENSE_GRID_ROLES = frozenset({ArrayRole.DENSE_STATISTIC_PLANE, ArrayRole.DENSE_IMPUTED_MASK})

#: Root attributes that are *expected* to differ between a source `data.zarr`
#: and its converted copy: they describe the physical layout, which the
#: conversion deliberately changes (issue #245).
_REWRITTEN_ROOT_ATTRS = frozenset({"chunk_shape", "shard_shape", "compressor", "zarr_format"})

#: Verification block, in rows and columns.  One block per read keeps the
#: verifier's memory bounded on a 40 GB plane.
_VERIFY_ROWS = 50_000
_VERIFY_COLS = 1_024

#: numcodecs Blosc shuffle codes -> the v3 codec's spelling.
_SHUFFLE_NAMES = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}


class ConversionError(Exception):
    """A refusal raised inside the staging block.

    An `Exception` subclass on purpose: `OpenGWASDBStore.staging` discards the
    staging directory on any `Exception` (issue #164).  A `SystemExit` would
    escape that cleanup and leave a half-converted release behind.
    """


class ConversionVerificationError(ConversionError):
    """The converted release is not bit-identical to its source."""


@dataclass(frozen=True)
class _ArrayPlan:
    """One source array, and the 0.2.0 layout it is converted into.

    Built in the parent from metadata only, then used both to create the
    destination array and to schedule its shard writes.  `codec` is the v3
    Blosc parameter triple, or `None` for an array the source stored
    uncompressed (the Z/EAF exception tables are deliberately uncompressed).
    """

    path: str
    role: ArrayRole
    shape: tuple[int, ...]
    dtype: str
    fill_value: Any
    inner_chunk: tuple[int, ...]
    shard_shape: tuple[int, ...]
    codec: tuple[str, int, str] | None


def _manifest_data(release: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    return data


def _refuse_unconvertible(source: Path) -> dict[str, Any]:
    """Fail against the *source* manifest, before any bytes are copied.

    The converter is a migration for one thing: a Dense Observed-Only release in
    0.1.0.  Everything else is refused by name, because a wrong conversion of a
    layout this tool does not understand would produce a plausible store.
    """
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        raise ConversionError(f"{source}: no manifest.json; this is not a Store Release")
    data = _manifest_data(source)
    version = str(data.get("format_version"))
    if version == SHARDED_FORMAT_VERSION:
        raise ConversionError(
            f"{manifest_path}: format_version is already {SHARDED_FORMAT_VERSION!r}; this "
            "release is already converted (issue #245)"
        )
    if version != CURRENT_FORMAT_VERSION:
        raise ConversionError(
            f"{manifest_path}: format_version is {version!r}, not "
            f"{CURRENT_FORMAT_VERSION!r}. The converter derives a "
            f"{SHARDED_FORMAT_VERSION} release from a {CURRENT_FORMAT_VERSION} one and is "
            "not a general migration tool; every other format is rebuilt (ADR 0041, "
            "spec §21.4)."
        )
    layout = str(data.get("primary_layout"))
    if layout != DENSE_OBSERVED_ONLY:
        raise ConversionError(
            f"{manifest_path}: primary_layout is {layout!r}; this converter accepts only a "
            "Dense release. Converting the remaining layouts (Ragged, Hybrid) adds them: "
            "see opengwas/opengwasdb#248."
        )
    completion = str(data.get("completion_state"))
    if completion != "observed_only":
        raise ConversionError(
            f"{manifest_path}: completion_state is {completion!r}; this converter accepts "
            "only a Dense Observed-Only release. Converting Dense Reference-Completed "
            "releases adds them: see opengwas/opengwasdb#248."
        )
    if (source / "dense" / "manifest.json").exists():
        raise ConversionError(
            f"{source}: carries a nested component manifest (a Hybrid Dense Component); "
            "this converter accepts only a standalone Dense Observed-Only release. "
            "Converting Hybrid releases adds them: see opengwas/opengwasdb#248."
        )
    return data


def _array_paths(root: Any, prefix: str = "") -> Iterator[str]:
    """Every array path under `root`, depth-first, groups included."""
    for name in sorted(root.array_keys()):
        yield prefix + name
    for name in sorted(root.group_keys()):
        yield from _array_paths(root[name], prefix + name + "/")


def _group_paths(root: Any, prefix: str = "") -> Iterator[str]:
    """Every group path under `root`, outermost first."""
    for name in sorted(root.group_keys()):
        path = prefix + name
        yield path
        yield from _group_paths(root[name], path + "/")


def _v3_codec(source_array: Any) -> tuple[str, int, str] | None:
    """The v3 Blosc parameters for a source array, or `None` if uncompressed.

    The Store format defines **one** compressor configuration, which the
    manifest publishes.  A source array stored with a different one would make
    that published claim false, so it is refused rather than silently converted
    under a record that does not describe it.  A source array with more than one
    codec is refused for the same reason.
    """
    codecs = tuple(source_array.compressors)
    if not codecs:
        return None
    if len(codecs) != 1:
        raise ConversionError(
            f"{source_array.path}: stored with {len(codecs)} codecs; a Store Release "
            "array has exactly one"
        )
    codec = codecs[0]
    shuffle = _SHUFFLE_NAMES.get(int(getattr(codec, "shuffle", -1)))
    if shuffle is None or not hasattr(codec, "cname"):
        raise ConversionError(
            f"{source_array.path}: cannot map codec {codec!r} to Zarr v3; the Store "
            "format stores Blosc arrays only"
        )
    observed = (str(codec.cname), int(codec.clevel), shuffle)
    expected = (
        str(COMPRESSOR_RECORD["cname"]),
        int(COMPRESSOR_RECORD["clevel"]),
        str(COMPRESSOR_RECORD["shuffle"]),
    )
    if observed != expected:
        raise ConversionError(
            f"{source_array.path}: stored with Blosc {observed!r}, but the Store format's "
            f"one compressor is {expected!r}. The manifest publishes that record, so "
            "converting this array under it would be a false claim; rebuild the source "
            "or extend the format's compressor policy (issue #245)."
        )
    return observed


def _plan_arrays(
    source_root: Any, *, dense_analysis_chunk: int, dense_shard: tuple[int, int]
) -> list[_ArrayPlan]:
    """Map every source array to a role and its 0.2.0 inner chunk and shard.

    The Dense plane's variant-axis inner chunk comes from `chunk_layout` (the
    fixed 1,000 rows the epic keeps); only the Analysis axis narrows.  The
    `PER_VARIANT` side arrays then follow that plane's variant chunk through the
    same policy, so they are never coarser than the plane they serve (issue
    #135, spec §6).
    """
    if "z" not in source_root:
        raise ConversionError(
            "data.zarr has no 'z' array; a Dense release must carry one, and this "
            "converter uses its variant-axis chunk to lay out the per-variant arrays"
        )
    z_shape = tuple(int(size) for size in source_root["z"].shape)
    plane_inner = chunk_layout(
        ArrayRole.DENSE_STATISTIC_PLANE,
        z_shape,
        hint=(DENSE_CHUNK_SHAPE[0], dense_analysis_chunk),
    )
    plans = [
        _plan_one_array(
            source_root,
            path,
            dense_analysis_chunk=dense_analysis_chunk,
            dense_shard=dense_shard,
            component_chunk=plane_inner[0],
        )
        for path in _array_paths(source_root)
    ]
    if not plans:
        raise ConversionError("data.zarr holds no arrays; nothing to convert")
    return plans


def _plan_one_array(
    source_root: Any,
    path: str,
    *,
    dense_analysis_chunk: int,
    dense_shard: tuple[int, int],
    component_chunk: int,
) -> _ArrayPlan:
    """One source array's role, inner chunk and shard; an unknown path refuses."""
    role = role_for_array_path(path)
    if role is None:
        raise ConversionError(
            f"data.zarr/{path}: no ArrayRole is registered for this path. A conversion "
            "never guesses a layout; known Dense arrays: z, se, eaf, eaf_baseline, "
            "eaf_reference, imputed, on_panel, se_coefficients, the z/se/eaf exception "
            "and overflow tables, top_hits/<tier>/*, rho/*. Add the role to the seam's "
            "path table (issue #245)."
        )
    array = source_root[path]
    shape = tuple(int(size) for size in array.shape)
    is_grid = role in _DENSE_GRID_ROLES
    inner = chunk_layout(
        role,
        shape,
        hint=(DENSE_CHUNK_SHAPE[0], dense_analysis_chunk) if is_grid else None,
        component_chunk=component_chunk,
    )
    shard = shard_layout(
        role,
        shape,
        inner_chunk=inner,
        dense_shard=dense_shard if is_grid else None,
    )
    return _ArrayPlan(
        path=path,
        role=role,
        shape=shape,
        dtype=str(array.dtype),
        fill_value=array.fill_value,
        inner_chunk=inner,
        shard_shape=shard,
        codec=_v3_codec(array),
    )


def _codec_object(codec: tuple[str, int, str] | None) -> Any:
    """The v3 compressor for a plan, or `None` for an uncompressed array."""
    return None if codec is None else sharded_compressor()


def _create_destination_arrays(
    destination_root: Any, source_root: Any, plans: list[_ArrayPlan]
) -> None:
    """Create every group and array in the destination v3 tree.

    Creation happens once, in the parent, before any forked shard writer: a
    worker only ever writes chunks into an array that already exists, so two
    processes never race on metadata.  Subgroup attributes (a top-hit tier's
    `threshold` and `order`) are copied, because the query facade reads them and
    the verifier requires group attrs to survive the conversion unchanged.
    """
    for group_path in _group_paths(source_root):
        group = require_group(destination_root, group_path)
        source_attrs = dict(source_root[group_path].attrs)
        if source_attrs:
            group.attrs.update(source_attrs)
    for plan in plans:
        parent_path, _, leaf = plan.path.rpartition("/")
        group = destination_root if not parent_path else destination_root[parent_path]
        create_array(
            group,
            leaf,
            plan.role,
            shape=plan.shape,
            dtype=plan.dtype,
            fill_value=plan.fill_value,
            compressor=_codec_object(plan.codec),
            inner_chunk=plan.inner_chunk,
            shards=plan.shard_shape,
        )


def _shard_starts(shape: tuple[int, ...], shard: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """The top-left corner of every shard in an array of `shape`."""
    axes = [range(0, max(dim, 1), step) for dim, step in zip(shape, shard, strict=True)]
    yield from itertools.product(*axes)


def _slice_for(
    starts: tuple[int, ...], shape: tuple[int, ...], shard: tuple[int, ...]
) -> tuple[slice, ...]:
    return tuple(
        slice(start, min(start + step, dim))
        for start, step, dim in zip(starts, shard, shape, strict=True)
    )


_OPEN_ROOTS: dict[tuple[int, str, str], Any] = {}


def _root(path: str, mode: str) -> Any:
    """A cached group per (pid, path, mode), so a forked child never reuses one."""
    pid = os.getpid()
    key = (pid, path, mode)
    root = _OPEN_ROOTS.get(key)
    if root is None:
        root = open_group(path, mode)
        # A fork child inherits the parent's cache; drop every other pid's
        # entries so it cannot touch a handle that belonged to the parent.
        for stale in [entry for entry in _OPEN_ROOTS if entry[0] != pid]:
            _OPEN_ROOTS.pop(stale, None)
        _OPEN_ROOTS[key] = root
    return root


def _write_shard(
    source_data: str, destination_data: str, plan: _ArrayPlan, starts: tuple[int, ...]
) -> None:
    """Read one source block and write exactly one destination shard."""
    source_array = _root(source_data, "r")[plan.path]
    destination_array = _root(destination_data, "r+")[plan.path]
    block = _slice_for(starts, plan.shape, plan.shard_shape)
    destination_array[block] = source_array[block]


def _write_shards(
    source_data: Path, destination_data: Path, plans: list[_ArrayPlan], *, workers: int
) -> None:
    """Write every shard of every array, over `workers` processes.

    Each task owns one whole shard, so no two processes touch one, and memory is
    bounded at one shard block per worker in flight.  With one worker the writes
    run in this process: forking for one task buys nothing.
    """
    tasks = [
        (plan, starts)
        for plan in plans
        for starts in _shard_starts(plan.shape, plan.shard_shape)
    ]
    print(f"  writing {len(tasks)} shards over {max(workers, 1)} worker(s)", flush=True)
    if workers <= 1:
        for plan, starts in tasks:
            _write_shard(str(source_data), str(destination_data), plan, starts)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_write_shard, str(source_data), str(destination_data), plan, starts)
            for plan, starts in tasks
        ]
        for future in futures:
            future.result()


def _recorded_layouts(plans: list[_ArrayPlan]) -> dict[str, dict[str, Any]]:
    """The inner chunk and shard of every array, keyed by its path."""
    recorded: dict[str, dict[str, Any]] = {}
    for plan in sorted(plans, key=lambda item: item.path):
        recorded[plan.path] = {
            "role": plan.role.value,
            "chunk_shape": list(plan.inner_chunk),
            "shard_shape": list(plan.shard_shape),
        }
    return recorded


def _rewrite_manifest(
    staged_path: Path,
    *,
    plans: list[_ArrayPlan],
    dense_analysis_chunk: int,
    dense_shard: tuple[int, int],
    now: str,
    source_release_id: str,
) -> str:
    """Restamp the manifest as a new 0.2.0 release and record the new layout.

    Returns the fresh `release_id`.  The Dense `chunk_shape` moves to the new
    inner chunk and the compressor to the v3 record; `shard_shape` and the
    per-array layout are added, so a later reader can tell what the arrays are
    without opening them.
    """
    path = staged_path / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    release_id = str(uuid.uuid4())
    data["release_id"] = release_id
    data["created_at"] = now
    data["format_version"] = SHARDED_FORMAT_VERSION
    dense = dict(data.get("provenance", {}).get("dense", {}))
    plane = next(plan for plan in plans if plan.role is ArrayRole.DENSE_STATISTIC_PLANE)
    dense["chunk_shape"] = list(plane.inner_chunk)
    dense["shard_shape"] = list(plane.shard_shape)
    dense["compressor"] = SHARDED_COMPRESSOR_RECORD
    dense["zarr_format"] = 3
    data["provenance"] = {
        **data.get("provenance", {}),
        "dense": dense,
        "zarr_v3_conversion": {
            "source_release_id": source_release_id,
            "from_format_version": CURRENT_FORMAT_VERSION,
            "to_format_version": SHARDED_FORMAT_VERSION,
            "source_zarr_format": 2,
            "target_zarr_format": 3,
            "tool": "scripts/convert_store_to_0_2_0.py",
            "at": now,
            "opengwasdb_git_hash": _installed_commit(),
            "dense_analysis_chunk": dense_analysis_chunk,
            "dense_shard": list(dense_shard),
            "layouts": _recorded_layouts(plans),
            "note": (
                "Derived by scripts/convert_store_to_0_2_0.py: every array was re-written "
                "as Zarr v3 with the sharding codec, holding the same stored codes as the "
                "source. The source release was not modified."
            ),
        },
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return release_id


def _installed_commit() -> str:
    """The installed opengwasdb commit, via the fingerprint #176 added.

    Imported here rather than at module scope: `resolve_manifest` pulls in the
    ancestry reference and the readers, and a conversion is not the only thing
    that imports this module.
    """
    from opengwasdb.build.resolve_manifest import _get_git_hash

    return _get_git_hash()


def _rewrite_dense_index(staged_path: Path, plans: list[_ArrayPlan]) -> None:
    """Re-point the `index.sqlite` `dense` blob at the new arrays."""
    plane = next(plan for plan in plans if plan.role is ArrayRole.DENSE_STATISTIC_PLANE)
    connection = StagedRelease(staged_path).index_connection()
    with connection:
        blob = get_metadata(connection, "dense", default={})
        if not isinstance(blob, dict):
            raise ConversionError(
                f"{staged_path}/index.sqlite: 'dense' metadata is not an object"
            )
        blob = dict(blob)
        blob["chunk_shape"] = list(plane.inner_chunk)
        blob["shard_shape"] = list(plane.shard_shape)
        blob["compressor"] = SHARDED_COMPRESSOR_RECORD
        blob["zarr_format"] = 3
        set_metadata(connection, "dense", blob)


def _write_root_attrs(destination_root: Any, source_root: Any, plans: list[_ArrayPlan]) -> None:
    """Copy the source root attrs and rewrite the ones describing the layout."""
    attrs = dict(source_root.attrs)
    plane = next(plan for plan in plans if plan.role is ArrayRole.DENSE_STATISTIC_PLANE)
    attrs["chunk_shape"] = list(plane.inner_chunk)
    attrs["shard_shape"] = list(plane.shard_shape)
    attrs["compressor"] = SHARDED_COMPRESSOR_RECORD
    attrs["zarr_format"] = 3
    destination_root.attrs.update(attrs)


def _raw_bytes(values: Any) -> np.ndarray:
    """`values` as a C-contiguous uint8 view, so NaN payloads count as bytes."""
    return np.ascontiguousarray(values).view(np.uint8)


def _verify_block(shape: tuple[int, ...]) -> tuple[int, ...]:
    """The verification block for an array of `shape`."""
    block = tuple(max(1, min(_VERIFY_ROWS, dim)) for dim in shape)
    if len(shape) == 2:
        block = (block[0], max(1, min(_VERIFY_COLS, shape[1])))
    return block


def _verify_array_values(source_array: Any, destination_array: Any) -> None:
    """Compare two arrays' stored values block by block, as raw bytes."""
    shape = tuple(int(size) for size in source_array.shape)
    block = _verify_block(shape)
    axes = [range(0, max(dim, 1), step) for dim, step in zip(shape, block, strict=True)]
    for starts in itertools.product(*axes):
        slices = tuple(
            slice(start, min(start + step, dim))
            for start, step, dim in zip(starts, block, shape, strict=True)
        )
        source_bytes = _raw_bytes(np.asarray(source_array[slices]))
        destination_bytes = _raw_bytes(np.asarray(destination_array[slices]))
        if source_bytes.shape != destination_bytes.shape or not np.array_equal(
            source_bytes, destination_bytes
        ):
            raise ConversionVerificationError(
                f"data.zarr/{source_array.path}: block {slices!r} is not bit-identical to "
                "the source; refusing to publish a converted release"
            )


def _attrs_differ_only_where_expected(
    source_attrs: dict[str, Any], destination_attrs: dict[str, Any], *, root: bool
) -> str | None:
    """A message when the two attr sets differ beyond the rewritten keys."""
    for key in sorted(set(source_attrs) | set(destination_attrs)):
        if root and key in _REWRITTEN_ROOT_ATTRS:
            continue
        if source_attrs.get(key) != destination_attrs.get(key):
            return (
                f"group attribute {key!r} differs: source {source_attrs.get(key)!r} vs "
                f"destination {destination_attrs.get(key)!r}"
            )
    return None


def _verify_root_layout(destination_root: Any) -> None:
    """The root attrs must describe the destination's real plane layout."""
    plane = destination_root["z"]
    for key, expected in (
        ("chunk_shape", list(int(size) for size in plane.chunks)),
        ("shard_shape", list(int(size) for size in plane.shards)),
    ):
        actual = destination_root.attrs.get(key)
        if list(actual) != expected:
            raise ConversionVerificationError(
                f"data.zarr root attr {key!r} is {actual!r}, but data.zarr/z has {expected!r}"
            )
    if dict(destination_root.attrs.get("compressor", {})) != SHARDED_COMPRESSOR_RECORD:
        raise ConversionVerificationError(
            "data.zarr root attr 'compressor' is not the v3 sharded record"
        )
    if destination_root.attrs.get("zarr_format") != 3:
        raise ConversionVerificationError("data.zarr root attr 'zarr_format' does not say 3")


def verify_conversion(source: str | Path, destination: str | Path) -> None:
    """Fail unless `destination` is a bit-exact 0.2.0 copy of `source`.

    Checks, for every array: the path set, shape, dtype and fill value; then the
    stored values, block by block, as raw bytes.  Every group's attributes must
    be equal, except the root's rewritten layout keys, which must instead name
    the destination's real inner chunk, shard and compressor.  Raises
    `ConversionVerificationError` on any difference.

    Separately callable so a test can corrupt one destination shard and watch it
    fail, and so the CLI can run it before publishing.
    """
    source_root = open_group(Path(source) / "data.zarr", "r")
    destination_root = open_group(Path(destination) / "data.zarr", "r")
    _require_same_paths(
        "array", set(_array_paths(source_root)), set(_array_paths(destination_root))
    )
    _require_same_paths(
        "group", set(_group_paths(source_root)), set(_group_paths(destination_root))
    )
    _verify_array_headers(source_root, destination_root)
    _verify_group_attrs(source_root, destination_root)
    _verify_root_layout(destination_root)


def _require_same_paths(kind: str, source: set[str], destination: set[str]) -> None:
    """Refuse a path set that differs, naming both sides."""
    if source == destination:
        return
    missing = sorted(source - destination)
    extra = sorted(destination - source)
    raise ConversionVerificationError(
        f"{kind} path sets differ; missing from destination: {missing}; unexpected: {extra}"
    )


def _verify_array_headers(source_root: Any, destination_root: Any) -> None:
    """Every array's shape, dtype, fill value, sharding and stored values match."""
    for path in sorted(_array_paths(source_root)):
        print(f"  verifying {path}", flush=True)
        source_array = source_root[path]
        destination_array = destination_root[path]
        for what, source_value, destination_value in (
            ("shape", tuple(source_array.shape), tuple(destination_array.shape)),
            ("dtype", str(source_array.dtype), str(destination_array.dtype)),
            (
                "fill value",
                source_array.fill_value,
                destination_array.fill_value,
            ),
        ):
            if source_value != destination_value:
                raise ConversionVerificationError(
                    f"data.zarr/{path}: {what} {destination_value!r} != source {source_value!r}"
                )
        if getattr(destination_array, "shards", None) is None:
            raise ConversionVerificationError(
                f"data.zarr/{path}: destination is not sharded; a 0.2.0 release stores "
                "every array with the sharding codec"
            )
        _verify_array_values(source_array, destination_array)


def _verify_group_attrs(source_root: Any, destination_root: Any) -> None:
    """Subgroup attrs must be equal; the root's only where they describe the layout."""
    for path in sorted(_group_paths(source_root)):
        difference = _attrs_differ_only_where_expected(
            dict(source_root[path].attrs), dict(destination_root[path].attrs), root=False
        )
        if difference:
            raise ConversionVerificationError(f"data.zarr/{path}: {difference}")
    difference = _attrs_differ_only_where_expected(
        dict(source_root.attrs), dict(destination_root.attrs), root=True
    )
    if difference:
        raise ConversionVerificationError(f"data.zarr: {difference}")


def _reflink_copy(source: Path, destination: Path) -> None:
    """Copy every top-level entry of `source` except `data.zarr` into `destination`.

    Reflink where the filesystem can, so the side files cost no space until one
    side is written.  `data.zarr` is deliberately left out: the destination gets
    a fresh v3 tree, and copying 119,118 v2 chunk files into a release that never
    reads them is work for nothing.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        if entry.name == "data.zarr":
            continue
        subprocess.run(
            ["cp", "-a", "--reflink=auto", str(entry), str(destination / entry.name)],
            check=True,
        )


def _require_valid_staged_release(staged_path: Path, destination: Path) -> None:
    """Publish only a staged copy that validates with no errors (issue #164)."""
    result = validate_store(staged_path)
    if not result.ok:
        for error in result.errors:
            print(f"  ERROR {error}")
        raise ConversionError(
            f"the converted release has {len(result.errors)} validation error(s); a release "
            "that does not validate is never published. The staging directory was removed, "
            f"the source is unchanged, and nothing was published to {destination}."
        )


def convert_dense_release(
    source: str | Path,
    destination: str | Path,
    *,
    dense_analysis_chunk: int = 64,
    dense_shard: tuple[int, int] = DENSE_SHARD_SHAPE,
    workers: int = 1,
) -> Path:
    """Derive a 0.2.0 release at `destination` from a Dense 0.1.0 one at `source`."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination:
        raise ConversionError(
            f"source and --into are the same path ({source}); a conversion derives a new "
            "release and cannot write into the one it was given (spec §21.4)"
        )
    if destination.exists():
        raise ConversionError(
            f"{destination}: already exists; refusing to overwrite. A conversion never "
            "replaces an existing release."
        )
    if int(dense_analysis_chunk) < 1:
        raise ConversionError("--dense-analysis-chunk must be at least 1")
    dense_shard = (int(dense_shard[0]), int(dense_shard[1]))
    _refuse_unconvertible(source)
    _stage_and_convert(
        source,
        destination,
        dense_analysis_chunk=int(dense_analysis_chunk),
        dense_shard=dense_shard,
        workers=int(workers),
    )
    print(f"Published {destination} as {SHARDED_FORMAT_VERSION}", flush=True)
    return destination


def _stage_and_convert(
    source: Path,
    destination: Path,
    *,
    dense_analysis_chunk: int,
    dense_shard: tuple[int, int],
    workers: int,
) -> None:
    """Build, verify and validate the converted release in staging.

    Split out of `convert_dense_release` so the refusal checks and this,
    the long-running half, are one function each.  Staging publishes by rename
    on clean exit and discards everything on any exception.
    """
    with OpenGWASDBStore.staging(destination) as staged:
        started = time.perf_counter()
        _reflink_copy(source, staged.path)
        _log_phase("copied the release envelope", started)
        started = time.perf_counter()
        source_root = open_group(source / "data.zarr", "r")
        plans = _plan_arrays(
            source_root,
            dense_analysis_chunk=dense_analysis_chunk,
            dense_shard=dense_shard,
        )
        print(f"  {len(plans)} arrays to convert", flush=True)
        destination_root = open_group_for_write(staged.data_path, "w", zarr_format=3)
        _create_destination_arrays(destination_root, source_root, plans)
        destination_root = None  # drop the write handle before forked writers run
        _log_phase("created the destination arrays", started)
        started = time.perf_counter()
        _write_shards(source / "data.zarr", staged.data_path, plans, workers=workers)
        _log_phase(f"wrote every shard over {workers} worker(s)", started)
        started = time.perf_counter()
        _rewrite_staged_metadata(
            source, staged.path, plans, dense_analysis_chunk, dense_shard, source_root
        )
        _log_phase("rewrote the manifest, index blob and root attrs", started)
        started = time.perf_counter()
        verify_conversion(source, staged.path)
        _log_phase("verified the conversion bit-exact", started)
        started = time.perf_counter()
        write_overview_html(staged.path, read_analyses(staged.path / "analyses.tsv"))
        _require_valid_staged_release(staged.path, destination)
        _log_phase("regenerated overview.html and validated the release", started)


def _rewrite_staged_metadata(
    source: Path,
    staged_path: Path,
    plans: list[_ArrayPlan],
    dense_analysis_chunk: int,
    dense_shard: tuple[int, int],
    source_root: Any,
) -> None:
    """Restamp the manifest, re-point the index blob and rewrite the root attrs."""
    _rewrite_manifest(
        staged_path,
        plans=plans,
        dense_analysis_chunk=dense_analysis_chunk,
        dense_shard=dense_shard,
        now=datetime.now(UTC).isoformat(),
        source_release_id=str(_manifest_data(source)["release_id"]),
    )
    _rewrite_dense_index(staged_path, plans)
    reopened = open_group(staged_path / "data.zarr", "r+")
    _write_root_attrs(reopened, source_root, plans)


def _log_phase(label: str, started: float) -> None:
    """Print a phase's wall time, so a long conversion can be projected."""
    print(f"  {label}: {time.perf_counter() - started:.1f}s", flush=True)


__all__ = [
    "ConversionError",
    "ConversionVerificationError",
    "convert_dense_release",
    "verify_conversion",
]
