"""Convert a Store Release to format 0.2.0 (Zarr v3, sharded).

Format 0.1.0 is Zarr v2 with one chunk per file; 0.2.0 is Zarr v3 with the
sharding codec, so the unit a query reads (the *inner chunk*) is decoupled from
the unit stored as a file (the *shard*), and the Dense Analysis-axis inner chunk
can narrow without multiplying the file count (epic #240, ADR 0057).  The
conversion does **not** re-encode any value: every array is read as raw stored
codes and written as the same codes, so the derived release holds bit-identical
values under the new physical layout.

Design rules, all of them deliberately narrow:

* **Every layout whose arrays the seam can name a role for.**  Dense
  Observed-Only and Reference-Completed, Ragged Observed-Only and
  Reference-Completed, and Hybrid (its outer release, its nested Dense
  Component and its Ragged Overflow) are converted.  A 0.2.0 source is refused
  as already converted; any other format is refused as not a general migration.
  Nothing is guessed.
* **Every array is mapped to an `ArrayRole` by its path** (`role_for_array_path`,
  the seam's path -> role table).  An array -- or a group, empty ones included
  -- with no role fails the conversion; it is never copied with a default
  layout.
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
  result is a new release: one fresh `release_id` and `created_at` for every
  manifest, `store_id` kept, a `zarr_v3_conversion` provenance block per
  component, and `overview.html` regenerated for the layouts that carry one.

A Hybrid release is two Zarr trees and two manifests: the outer release and its
nested Dense Component (spec §16).  Both are converted, and a half-converted
Hybrid -- one manifest's arrays rewritten, the other's not -- is the failure the
format rule exists to catch, so the converter refuses to leave one behind and
`verify_conversion` checks both.

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

from opengwasdb.index.sqlite import connect, get_metadata, set_metadata
from opengwasdb.layouts.dense.overview import write_overview_html
from opengwasdb.model.analyses import read_analyses
from opengwasdb.store.arrays import (
    COMPRESSOR_RECORD,
    DENSE_CHUNK_SHAPE,
    DENSE_SHARD_SHAPE,
    SHARDED_COMPRESSOR_RECORD,
    TOP_HIT_SHARD_CHUNKS,
    ArrayRole,
    chunk_layout,
    create_array,
    is_recorded_group_path,
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
)
from opengwasdb.validation import validate_store

#: The layouts the converter understands (#248).  Every other value is refused
#: by manifest, before a byte is copied.
SUPPORTED_LAYOUTS = frozenset({"dense", "ragged", "hybrid"})

#: The `ArrayRole`s whose inner chunk narrows on the Analysis axis and whose
#: shard is the conversion's `(V_s, A_s)` parameter.  The manifest's recorded
#: `chunk_shape` describes these planes.
_DENSE_GRID_ROLES = frozenset({ArrayRole.DENSE_STATISTIC_PLANE, ArrayRole.DENSE_IMPUTED_MASK})

#: Root attributes that are *expected* to differ between a source `data.zarr`
#: and its converted copy: they describe the physical layout, which the
#: conversion deliberately changes (issue #245).
_REWRITTEN_ROOT_ATTRS = frozenset({"chunk_shape", "shard_shape", "compressor", "zarr_format"})

#: Verification block, in rows and columns.  One block per read keeps the
#: verifier's memory bounded on a 40 GB plane or a 3.09-billion-row sequence.
_VERIFY_ROWS = 50_000
_VERIFY_COLS = 1_024

#: numcodecs Blosc shuffle codes -> the v3 codec's spelling.
_SHUFFLE_NAMES = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}

#: The layouts whose release carries an `overview.html` to regenerate.  A Ragged
#: release never has one (spec §11), so the converter must not create one.
_OVERVIEW_LAYOUTS = frozenset({"dense", "hybrid"})


class ConversionError(Exception):
    """A refusal raised inside the staging block.

    An `Exception` subclass on purpose: `OpenGWASDBStore.staging` discards the
    staging directory on any `Exception` (issue #164).  A `SystemExit` would
    escape that cleanup and leave a half-converted release behind.
    """


class ConversionVerificationError(ConversionError):
    """The converted release is not bit-identical to its source."""


@dataclass(frozen=True)
class _Component:
    """One Zarr tree in a release, and the manifest that describes it.

    A standalone release has one; a Hybrid has two -- its outer release and its
    nested Dense Component (spec §16).  `kind` says which provenance block the
    component's recorded layout belongs in: `dense` for a Dense release or a
    nested Dense Component, `hybrid` for a Hybrid release's outer manifest,
    `ragged` for a standalone Ragged release (whose manifest records no Dense
    chunk at all).
    """

    zarr_rel: str
    manifest_rel: str
    kind: str


@dataclass(frozen=True)
class _ArrayPlan:
    """One source array, and the 0.2.0 layout it is converted into.

    Built in the parent from metadata only, then used both to create the
    destination array and to schedule its shard writes.  `codec` is the v3
    Blosc parameter triple, or `None` for an array the source stored
    uncompressed (the Z/EAF exception tables are deliberately uncompressed).
    """

    zarr_rel: str
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


def _manifest_paths(release: Path) -> list[Path]:
    """A release's known manifests, outermost first.

    A Hybrid release nests a Dense Component with a manifest of its own, and
    both declare `format_version`; the converter must see both before it starts.
    Deliberately not an `rglob`: that would walk every chunk file in a
    119,000-file `data.zarr` to find two files that sit at known paths.
    """
    paths = [release / "manifest.json"]
    nested = release / "dense" / "manifest.json"
    if nested.exists():
        paths.append(nested)
    return paths


def _components(layout: str) -> list[_Component]:
    """The Zarr trees and manifests a release of `layout` carries."""
    if layout == "hybrid":
        return [
            _Component("data.zarr", "manifest.json", "hybrid"),
            _Component("dense/data.zarr", "dense/manifest.json", "dense"),
        ]
    if layout == "ragged":
        return [_Component("data.zarr", "manifest.json", "ragged")]
    return [_Component("data.zarr", "manifest.json", "dense")]


def _refuse_unconvertible(source: Path) -> list[_Component]:
    """Fail against the *source* manifests, before any bytes are copied.

    The converter is a migration for one thing: a 0.1.0 release whose layout it
    fully understands.  A 0.2.0 manifest -- on any component -- means the
    release is already converted (or half-converted); any other format is not
    this tool's business.  A layout it does not know is refused by name, because
    a wrong conversion of an unknown layout would produce a plausible store.
    """
    manifests = _manifest_paths(source)
    if not manifests:
        raise ConversionError(f"{source}: no manifest.json; this is not a Store Release")
    for path in manifests:
        version = str(json.loads(path.read_text(encoding="utf-8")).get("format_version"))
        if version == SHARDED_FORMAT_VERSION:
            raise ConversionError(
                f"{path}: format_version is already {SHARDED_FORMAT_VERSION!r}; this "
                "release is already converted (issue #245)"
            )
        if version != CURRENT_FORMAT_VERSION:
            raise ConversionError(
                f"{path}: format_version is {version!r}, not "
                f"{CURRENT_FORMAT_VERSION!r}. The converter derives a "
                f"{SHARDED_FORMAT_VERSION} release from a {CURRENT_FORMAT_VERSION} one and "
                "is not a general migration tool; every other format is rebuilt (ADR 0041, "
                "spec §21.4)."
            )
    top = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    layout = str(top.get("primary_layout"))
    if layout not in SUPPORTED_LAYOUTS:
        raise ConversionError(
            f"{source}/manifest.json: primary_layout is {layout!r}; this converter accepts "
            f"{sorted(SUPPORTED_LAYOUTS)}. Add the layout's arrays to the seam's path table "
            "(issue #248)."
        )
    if layout == "hybrid" and not (source / "dense" / "manifest.json").exists():
        raise ConversionError(
            f"{source}: primary_layout is 'hybrid' but there is no nested Dense Component "
            "manifest at dense/manifest.json; a Hybrid release is two Store Releases and "
            "converting half of one would be refused one directory down (spec §16)."
        )
    return _components(layout)


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


def _group_component_chunk(group: Any, *, dense_analysis_chunk: int) -> int | None:
    """The variant-axis chunk a group's per-variant arrays must follow (spec §6).

    Derived from the group's own principal plane, with the *new* layout the
    conversion writes: a Dense grid's variant chunk (fixed at 1,000) for a group
    holding `z`/`se`/`eaf`, or a Ragged association sequence's chunk for the
    `ragged` group.  `None` for a group with neither (e.g. `top_hits`), whose
    arrays are never `PER_VARIANT`.
    """
    dense = _dense_group_chunk(group, dense_analysis_chunk=dense_analysis_chunk)
    if dense is not None:
        return dense
    return _sequence_group_chunk(group)


def _dense_group_chunk(group: Any, *, dense_analysis_chunk: int) -> int | None:
    """The variant chunk of a group holding a 2-D Dense grid plane, or `None`."""
    if "z" not in group:
        return None
    dense = group["z"]
    if getattr(dense, "ndim", 0) != 2:
        return None
    shape = tuple(int(size) for size in dense.shape)
    return chunk_layout(
        ArrayRole.DENSE_STATISTIC_PLANE,
        shape,
        hint=(DENSE_CHUNK_SHAPE[0], dense_analysis_chunk),
    )[0]


def _sequence_group_chunk(group: Any) -> int | None:
    """The chunk of a group's first 1-D Ragged association sequence, or `None`."""
    for name in ("variant_index", "z", "imputed", "eaf"):
        if name not in group:
            continue
        candidate = group[name]
        if getattr(candidate, "ndim", 0) == 1:
            shape = tuple(int(size) for size in candidate.shape)
            return chunk_layout(ArrayRole.ASSOCIATION_SEQUENCE, shape)[0]
    return None


def _component_chunks(source_root: Any, *, dense_analysis_chunk: int) -> dict[str, int | None]:
    """The per-variant component chunk for every group in one Zarr tree."""
    chunks: dict[str, int | None] = {
        "": _group_component_chunk(source_root, dense_analysis_chunk=dense_analysis_chunk)
    }
    for group_path in _group_paths(source_root):
        chunks[group_path] = _group_component_chunk(
            source_root[group_path], dense_analysis_chunk=dense_analysis_chunk
        )
    return chunks


def _plan_component(
    source_root: Any,
    component: _Component,
    *,
    dense_analysis_chunk: int,
    dense_shard: tuple[int, int],
    top_hit_shard_chunks: int = TOP_HIT_SHARD_CHUNKS,
) -> list[_ArrayPlan]:
    """Map every array in one Zarr tree to its role and 0.2.0 layout.

    The Dense plane's variant-axis inner chunk stays the fixed 1,000 rows the
    epic keeps; only the Analysis axis narrows.  The `PER_VARIANT` side arrays
    follow the variant chunk of the plane in their own group through the same
    policy, so they are never coarser than the plane they serve (issue #135,
    spec §6).

    `top_hit_shard_chunks` is #246's top-hit shard override, passed to the seam
    for `TOP_HIT_INDEX` alone; the default is the seam's own default.
    """
    _refuse_unknown_groups(source_root, component)
    component_chunks = _component_chunks(
        source_root, dense_analysis_chunk=dense_analysis_chunk
    )
    plans = [
        _plan_one_array(
            source_root,
            path,
            component=component,
            component_chunk=component_chunks[path.rpartition("/")[0]],
            dense_analysis_chunk=dense_analysis_chunk,
            dense_shard=dense_shard,
            top_hit_shard_chunks=top_hit_shard_chunks,
        )
        for path in _array_paths(source_root)
    ]
    if not plans:
        raise ConversionError(f"{component.zarr_rel} holds no arrays; nothing to convert")
    return plans


def _plan_arrays(
    source_root: Any,
    *,
    dense_analysis_chunk: int,
    dense_shard: tuple[int, int],
    top_hit_shard_chunks: int = TOP_HIT_SHARD_CHUNKS,
) -> list[_ArrayPlan]:
    """Plan one Dense `data.zarr` tree (#245's entry point; #246's plan test).

    The layout-aware converter plans per component through `_plan_component`;
    this is the single-Dense-tree form its own tests drive directly.
    """
    return _plan_component(
        source_root,
        _Component("data.zarr", "manifest.json", "dense"),
        dense_analysis_chunk=dense_analysis_chunk,
        dense_shard=dense_shard,
        top_hit_shard_chunks=top_hit_shard_chunks,
    )


def _refuse_unknown_groups(source_root: Any, component: _Component) -> None:
    """Fail on a `data.zarr` group the format does not define.

    Only arrays go through `role_for_array_path`; an empty unknown group has no
    array to name a role for, and would otherwise be recreated unnoticed.  The
    brief is explicit that an unmapped array **or group** fails conversion.
    """
    for group_path in _group_paths(source_root):
        if not is_recorded_group_path(group_path):
            raise ConversionError(
                f"{component.zarr_rel}/{group_path}: this group is not part of the format. "
                "A conversion never recreates an unknown group; known groups are "
                "top_hits, top_hits/<tier>, rho and ragged (issue #245, #248)."
            )


def _dense_shard_for(
    request: tuple[int, int], inner: tuple[int, ...]
) -> tuple[int, ...]:
    """The Dense shard hint, reconciled with the array's actual inner chunk.

    `--dense-shard` is a hint like the chunk shape, and an array's own size can
    clip the inner chunk below it: OGS-00004's Dense Component has nine Analyses,
    so the requested Analysis-axis inner chunk of 64 clips to 9, and a shard of
    1,024 is then not a whole multiple of it.  A shard MUST be one, so each axis
    is rounded down to the largest whole number of *actual* inner chunks that
    does not exceed the request (at least one).  With the standard shapes this is
    a no-op; the clip to the array that follows then gives the nine-Analysis
    component a shard of the whole axis.
    """
    return tuple(
        max(int(inner_axis), (int(want) // int(inner_axis)) * int(inner_axis))
        for want, inner_axis in zip(request, inner, strict=True)
    )


def _plan_one_array(
    source_root: Any,
    path: str,
    *,
    component: _Component,
    component_chunk: int | None,
    dense_analysis_chunk: int,
    dense_shard: tuple[int, int],
    top_hit_shard_chunks: int = TOP_HIT_SHARD_CHUNKS,
) -> _ArrayPlan:
    """One source array's role, inner chunk and shard; an unknown path refuses."""
    role = role_for_array_path(path)
    if role is None:
        raise ConversionError(
            f"{component.zarr_rel}/{path}: no ArrayRole is registered for this path. A "
            "conversion never guesses a layout; known arrays: the Dense planes and side "
            "arrays (z, se, eaf, eaf_baseline, eaf_reference, imputed, on_panel, "
            "se_coefficients, the z/se/eaf exception and overflow tables), the Ragged CSR "
            "group (ragged/z, se, eaf, variant_index, offsets, imputed, eaf_baseline, "
            "eaf_reference, the exception and overflow tables), top_hits/<tier>/* and "
            "rho/*. Add the role to the seam's path table (issue #248)."
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
        dense_shard=_dense_shard_for(dense_shard, inner) if is_grid else None,
        top_hit_shard_chunks=(
            top_hit_shard_chunks if role is ArrayRole.TOP_HIT_INDEX else None
        ),
    )
    return _ArrayPlan(
        zarr_rel=component.zarr_rel,
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
    """Create every group and array in one destination v3 tree.

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
    source_release: str, destination_release: str, plan: _ArrayPlan, starts: tuple[int, ...]
) -> None:
    """Read one source block and write exactly one destination shard."""
    source_array = _root(str(Path(source_release) / plan.zarr_rel), "r")[plan.path]
    destination_array = _root(str(Path(destination_release) / plan.zarr_rel), "r+")[plan.path]
    block = _slice_for(starts, plan.shape, plan.shard_shape)
    destination_array[block] = source_array[block]


def _write_shards(
    source: Path, destination: Path, plans: list[_ArrayPlan], *, workers: int
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
            _write_shard(str(source), str(destination), plan, starts)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_write_shard, str(source), str(destination), plan, starts)
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


def _dense_plane(plans: list[_ArrayPlan]) -> _ArrayPlan | None:
    """The Dense statistic plane a component records its layout from, or `None`."""
    for plan in plans:
        if plan.role is ArrayRole.DENSE_STATISTIC_PLANE:
            return plan
    return None


def _rewrite_manifest(
    staged_path: Path,
    component: _Component,
    *,
    plans: list[_ArrayPlan],
    dense_plane: _ArrayPlan | None,
    dense_analysis_chunk: int,
    dense_shard: tuple[int, int],
    top_hit_shard_chunks: int,
    release_id: str,
    now: str,
    source_release_id: str,
) -> None:
    """Restamp one manifest as part of a new 0.2.0 release and record the layout.

    Every manifest in the release gets the same fresh `release_id` and
    `created_at`, so a Hybrid's nested Dense Component still names the release
    that nests it.  The recorded layout goes where this component's manifest
    already keeps it: `provenance.dense` for a Dense release or a nested Dense
    Component (closing the gap #245's report named), `provenance.hybrid` for a
    Hybrid release's outer manifest.  A Ragged release records no Dense chunk.
    """
    path = staged_path / component.manifest_rel
    data = json.loads(path.read_text(encoding="utf-8"))
    data["release_id"] = release_id
    data["created_at"] = now
    data["format_version"] = SHARDED_FORMAT_VERSION
    provenance = {**data.get("provenance", {})}
    if component.kind in {"dense", "hybrid"} and dense_plane is not None:
        key = "dense" if component.kind == "dense" else "hybrid"
        block = dict(provenance.get(key, {}))
        block["chunk_shape"] = list(dense_plane.inner_chunk)
        block["shard_shape"] = list(dense_plane.shard_shape)
        block["compressor"] = SHARDED_COMPRESSOR_RECORD
        block["zarr_format"] = 3
        provenance[key] = block
    provenance["zarr_v3_conversion"] = {
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
        "top_hit_shard_chunks": top_hit_shard_chunks,
        "component": component.zarr_rel,
        "layouts": _recorded_layouts(plans),
        "note": (
            "Derived by scripts/convert_store_to_0_2_0.py: every array was re-written "
            "as Zarr v3 with the sharding codec, holding the same stored codes as the "
            "source. The source release was not modified."
        ),
    }
    data["provenance"] = provenance
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _installed_commit() -> str:
    """The installed opengwasdb commit, via the fingerprint #176 added.

    Imported here rather than at module scope: `resolve_manifest` pulls in the
    ancestry reference and the readers, and a conversion is not the only thing
    that imports this module.
    """
    from opengwasdb.build.resolve_manifest import _get_git_hash

    return _get_git_hash()


def _rewrite_dense_index(staged_path: Path, component: _Component, plane: _ArrayPlan) -> None:
    """Re-point a component's `index.sqlite` `dense` blob at the new arrays.

    Every Dense component -- a standalone release and a Hybrid's nested one --
    carries its own `index.sqlite` beside its manifest.  A Hybrid's *outer*
    index also holds a `dense` blob describing the nested component, so it is
    updated with the same plane even though this component is not Dense itself.
    """
    index_path = (staged_path / component.manifest_rel).parent / "index.sqlite"
    connection = connect(index_path)
    with connection:
        blob = get_metadata(connection, "dense", default={})
        if not isinstance(blob, dict):
            raise ConversionError(
                f"{component.manifest_rel}: 'dense' metadata is not an object"
            )
        blob = dict(blob)
        blob["chunk_shape"] = list(plane.inner_chunk)
        blob["shard_shape"] = list(plane.shard_shape)
        blob["compressor"] = SHARDED_COMPRESSOR_RECORD
        blob["zarr_format"] = 3
        set_metadata(connection, "dense", blob)


def _write_root_attrs(
    destination_root: Any, source_root: Any, plane: _ArrayPlan | None
) -> None:
    """Copy the source root attrs and rewrite the ones describing the layout.

    Only a root that actually holds Dense planes gets the Dense layout keys; a
    Ragged or Hybrid outer root has none, and adding them would be a recording
    that describes nothing.
    """
    attrs = dict(source_root.attrs)
    if plane is not None:
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
    """A message when the two attr sets differ beyond the rewritten keys.

    Key **membership** is compared before values: a source attribute whose value
    is `None` and a destination that does not carry the key are not the same
    thing, and `dict.get` makes them look equal.
    """
    for key in sorted(set(source_attrs) | set(destination_attrs)):
        if root and key in _REWRITTEN_ROOT_ATTRS:
            continue
        if key not in source_attrs:
            return (
                f"group attribute {key!r} is present only on the destination "
                f"(value {destination_attrs[key]!r})"
            )
        if key not in destination_attrs:
            return (
                f"group attribute {key!r} is missing from the destination "
                f"(source value {source_attrs[key]!r})"
            )
        if source_attrs[key] != destination_attrs[key]:
            return (
                f"group attribute {key!r} differs: source {source_attrs[key]!r} vs "
                f"destination {destination_attrs[key]!r}"
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


def _verify_component(source_root: Any, destination_root: Any) -> None:
    """Verify one Zarr tree, source against destination, bit for bit."""
    _require_same_paths(
        "array", set(_array_paths(source_root)), set(_array_paths(destination_root))
    )
    _require_same_paths(
        "group", set(_group_paths(source_root)), set(_group_paths(destination_root))
    )
    _verify_array_headers(source_root, destination_root)
    _verify_group_attrs(source_root, destination_root)
    if "z" in source_root:
        _verify_root_layout(destination_root)


def verify_conversion(source: str | Path, destination: str | Path) -> None:
    """Fail unless `destination` is a bit-exact 0.2.0 copy of `source`.

    Checks, for every array: the path set, shape, dtype and fill value; then the
    stored values, block by block, as raw bytes.  Every group's attributes must
    be equal, except the root's rewritten layout keys, which must instead name
    the destination's real inner chunk, shard and compressor.  Raises
    `ConversionVerificationError` on any difference.

    A Hybrid release has two Zarr trees; both are verified, so a half-converted
    Hybrid cannot pass.  Separately callable so a test can corrupt one
    destination shard and watch it fail, and so the CLI can run it before
    publishing.
    """
    for zarr_rel in _component_zarr_rels(Path(source)):
        _verify_component(
            open_group(Path(source) / zarr_rel, "r"),
            open_group(Path(destination) / zarr_rel, "r"),
        )


def _component_zarr_rels(release: Path) -> list[str]:
    """Every Zarr tree in a release, from the manifests it actually carries."""
    rels = ["data.zarr"]
    if (release / "dense" / "manifest.json").exists():
        rels.append("dense/data.zarr")
    return rels


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
        ):
            if source_value != destination_value:
                raise ConversionVerificationError(
                    f"data.zarr/{path}: {what} {destination_value!r} != source {source_value!r}"
                )
        if _fill_bytes(source_array) != _fill_bytes(destination_array):
            raise ConversionVerificationError(
                f"data.zarr/{path}: fill value {destination_array.fill_value!r} != source "
                f"{source_array.fill_value!r}"
            )
        if getattr(destination_array, "shards", None) is None:
            raise ConversionVerificationError(
                f"data.zarr/{path}: destination is not sharded; a 0.2.0 release stores "
                "every array with the sharding codec"
            )
        _verify_array_values(source_array, destination_array)


def _fill_bytes(array: Any) -> bytes | None:
    """An array's fill value as raw bytes in its own dtype, or `None`.

    `fill_value` is compared bitwise, not with `==`: a float fill may be NaN, and
    `NaN != NaN` would reject a faithful conversion of a valid store.  Narrowing
    to the array's dtype first keeps the comparison per-dtype, and `tobytes`
    keeps the NaN payload -- two different NaN bit patterns are a real
    difference, as they are for stored values.
    """
    if array.fill_value is None:
        return None
    return np.asarray(array.fill_value, dtype=array.dtype).tobytes()


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


def _reflink_copy(source: Path, destination: Path, zarr_rels: list[str]) -> None:
    """Copy every top-level entry of `source` except the Zarr trees.

    Reflink where the filesystem can, so the side files cost no space until one
    side is written.  Each `data.zarr` is deliberately left out -- top-level and
    a Hybrid's nested `dense/data.zarr` -- because the destination gets fresh v3
    trees, and copying 119,118 v2 chunk files into a release that never reads
    them is work for nothing.
    """
    destination.mkdir(parents=True, exist_ok=True)
    top_zarr = _top_level_zarr_rels(zarr_rels)
    for entry in sorted(source.iterdir()):
        if entry.name in top_zarr:
            continue
        nested = _nested_zarr_rels(zarr_rels, entry.name)
        if nested and entry.is_dir():
            _copy_directory_without(entry, destination / entry.name, _nested_children(nested))
        else:
            _copy_path(entry, destination / entry.name)


def _top_level_zarr_rels(zarr_rels: list[str]) -> set[str]:
    """The component Zarr roots that sit directly under the release."""
    return {rel for rel in zarr_rels if "/" not in rel}


def _nested_zarr_rels(zarr_rels: list[str], name: str) -> list[str]:
    """The component Zarr roots nested under the top-level entry `name`."""
    return [rel for rel in zarr_rels if rel.startswith(name + "/")]


def _nested_children(nested: list[str]) -> set[str]:
    """The entry names to omit when copying a directory that nests a Zarr root."""
    return {"/".join(rel.split("/")[1:]) for rel in nested}


def _copy_directory_without(source: Path, destination: Path, skip: set[str]) -> None:
    """Copy `source`'s entries into `destination`, omitting the named ones."""
    destination.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        if entry.name in skip:
            continue
        _copy_path(entry, destination / entry.name)


def _copy_path(source: Path, destination: Path) -> None:
    """Reflink-copy one path, falling back to a full copy where it cannot."""
    subprocess.run(
        ["cp", "-a", "--reflink=auto", str(source), str(destination)], check=True
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


def convert_release(
    source: str | Path,
    destination: str | Path,
    *,
    dense_analysis_chunk: int = 64,
    dense_shard: tuple[int, int] = DENSE_SHARD_SHAPE,
    top_hit_shard_chunks: int = TOP_HIT_SHARD_CHUNKS,
    workers: int = 1,
) -> Path:
    """Derive a 0.2.0 release at `destination` from a 0.1.0 one at `source`."""
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
    if int(top_hit_shard_chunks) < 1:
        raise ConversionError("--top-hit-shard-chunks must be at least 1")
    dense_shard = (int(dense_shard[0]), int(dense_shard[1]))
    components = _refuse_unconvertible(source)
    _stage_and_convert(
        source,
        destination,
        components,
        dense_analysis_chunk=int(dense_analysis_chunk),
        dense_shard=dense_shard,
        top_hit_shard_chunks=int(top_hit_shard_chunks),
        workers=int(workers),
    )
    print(f"Published {destination} as {SHARDED_FORMAT_VERSION}", flush=True)
    return destination


#: The #245 entry point's old name.  Kept so an existing caller does not break;
#: the function itself now accepts every layout #248 adds.
convert_dense_release = convert_release


def _stage_and_convert(
    source: Path,
    destination: Path,
    components: list[_Component],
    *,
    dense_analysis_chunk: int,
    dense_shard: tuple[int, int],
    top_hit_shard_chunks: int,
    workers: int,
) -> None:
    """Build, verify and validate the converted release in staging.

    Split out of `convert_release` so the refusal checks and this, the
    long-running half, are one function each.  Staging publishes by rename on
    clean exit and discards everything on any exception.
    """
    with OpenGWASDBStore.staging(destination) as staged:
        started = time.perf_counter()
        _reflink_copy(source, staged.path, [component.zarr_rel for component in components])
        _log_phase("copied the release envelope", started)
        started = time.perf_counter()
        plans_by_component, source_roots = _create_components(
            source,
            staged.path,
            components,
            dense_analysis_chunk=dense_analysis_chunk,
            dense_shard=dense_shard,
            top_hit_shard_chunks=top_hit_shard_chunks,
        )
        _log_phase("created the destination arrays", started)
        started = time.perf_counter()
        all_plans = [plan for plans in plans_by_component.values() for plan in plans]
        _write_shards(source, staged.path, all_plans, workers=workers)
        _log_phase(f"wrote every shard over {workers} worker(s)", started)
        started = time.perf_counter()
        _rewrite_staged_metadata(
            source,
            staged.path,
            components,
            plans_by_component,
            source_roots,
            dense_analysis_chunk,
            dense_shard,
            top_hit_shard_chunks,
            str(uuid.uuid4()),
            datetime.now(UTC).isoformat(),
        )
        _log_phase("rewrote the manifests, index blobs and root attrs", started)
        started = time.perf_counter()
        verify_conversion(source, staged.path)
        _log_phase("verified the conversion bit-exact", started)
        started = time.perf_counter()
        _refresh_overviews(source, staged.path, components)
        _require_valid_staged_release(staged.path, destination)
        _log_phase("regenerated overview.html and validated the release", started)


def _create_components(
    source: Path,
    staged_path: Path,
    components: list[_Component],
    *,
    dense_analysis_chunk: int,
    dense_shard: tuple[int, int],
    top_hit_shard_chunks: int,
) -> tuple[dict[str, list[_ArrayPlan]], dict[str, Any]]:
    """Create every destination Zarr tree, one component at a time.

    Returns the plans and the source roots, which the metadata rewrite and the
    verification need.  Each component's write handle is dropped when the loop
    moves on, so no handle survives into the forked shard writers.
    """
    plans_by_component: dict[str, list[_ArrayPlan]] = {}
    source_roots: dict[str, Any] = {}
    for component in components:
        source_root = open_group(source / component.zarr_rel, "r")
        source_roots[component.zarr_rel] = source_root
        plans = _plan_component(
            source_root,
            component,
            dense_analysis_chunk=dense_analysis_chunk,
            dense_shard=dense_shard,
            top_hit_shard_chunks=top_hit_shard_chunks,
        )
        plans_by_component[component.zarr_rel] = plans
        print(f"  {component.zarr_rel}: {len(plans)} arrays to convert", flush=True)
        destination_root = open_group_for_write(
            staged_path / component.zarr_rel, "w", zarr_format=3
        )
        _create_destination_arrays(destination_root, source_root, plans)
    return plans_by_component, source_roots


def _rewrite_staged_metadata(
    source: Path,
    staged_path: Path,
    components: list[_Component],
    plans_by_component: dict[str, list[_ArrayPlan]],
    source_roots: dict[str, Any],
    dense_analysis_chunk: int,
    dense_shard: tuple[int, int],
    top_hit_shard_chunks: int,
    release_id: str,
    now: str,
) -> None:
    """Restamp every manifest, re-point every index blob and rewrite root attrs."""
    dense_component = next(
        (component for component in components if component.kind == "dense"), None
    )
    dense_plane = (
        None
        if dense_component is None
        else _dense_plane(plans_by_component[dense_component.zarr_rel])
    )
    for component in components:
        plans = plans_by_component[component.zarr_rel]
        source_manifest = json.loads(
            (source / component.manifest_rel).read_text(encoding="utf-8")
        )
        _rewrite_manifest(
            staged_path,
            component,
            plans=plans,
            dense_plane=dense_plane,
            dense_analysis_chunk=dense_analysis_chunk,
            dense_shard=dense_shard,
            top_hit_shard_chunks=top_hit_shard_chunks,
            release_id=release_id,
            now=now,
            source_release_id=str(source_manifest["release_id"]),
        )
        plane = _dense_plane(plans)
        if plane is not None:
            _rewrite_dense_index(staged_path, component, plane)
        elif component.kind == "hybrid" and dense_plane is not None:
            # A Hybrid release's outer index also records the nested Dense
            # Component's layout; leaving it on the v2 chunk would be a stale
            # recording of arrays this conversion moved.
            _rewrite_dense_index(staged_path, component, dense_plane)
        reopened = open_group(staged_path / component.zarr_rel, "r+")
        _write_root_attrs(reopened, source_roots[component.zarr_rel], plane)


def _refresh_overviews(source: Path, staged_path: Path, components: list[_Component]) -> None:
    """Regenerate `overview.html` wherever the source release carried one.

    Its header embeds the release identity (ADR 0032), so a page left over from
    the copy would name the source release.  A Ragged release has no page
    (spec §11); a Hybrid has one at the top and another on its nested Dense
    Component.
    """
    for component in components:
        if component.kind not in _OVERVIEW_LAYOUTS:
            continue
        page = (staged_path / component.zarr_rel).parent / "overview.html"
        if not page.exists():
            continue
        write_overview_html(page.parent, read_analyses(page.parent / "analyses.tsv"))


def _log_phase(label: str, started: float) -> None:
    """Print a phase's wall time, so a long conversion can be projected."""
    print(f"  {label}: {time.perf_counter() - started:.1f}s", flush=True)


__all__ = [
    "ConversionError",
    "ConversionVerificationError",
    "convert_dense_release",
    "convert_release",
    "verify_conversion",
]
