"""Rules that hold a Dense grid's recorded layout to its arrays (spec §10a, ADR 0074).

A Dense release records its planes' chunk and shard shape, compressor and Zarr
format in three places: the manifest's provenance, the `index.sqlite` `dense`
blob and the `data.zarr` root attributes. A Hybrid records them for its Dense
Component in its outer manifest and index as well. None is derived from the
arrays, so each is a copy that can disagree with them, and a reader sizing its
reads from a disagreeing one decodes the wrong blocks (issue #245, #248).
"""

from __future__ import annotations

import numbers
import sqlite3
from collections.abc import Iterator
from typing import Any

from opengwasdb.index.sqlite import get_metadata
from opengwasdb.store.arrays import (
    COMPRESSOR_RECORD,
    SHARDED_COMPRESSOR_RECORD,
    ArrayRole,
    inner_chunk_of,
    shard_layout,
)
from opengwasdb.validation.rule import Check, Kind, Rule, Target
from opengwasdb.validation.rules.hybrid import COMPONENT, component, is_hybrid
from opengwasdb.validation.rules.release import REQUIRED


def recorded(target: Target) -> Iterator[str]:
    """A grid's three recordings describe its planes."""
    errors: list[str] = []
    store = target.store
    with store.index_connection() as connection:
        _validate_recorded_layout(store.manifest, connection, target.group, errors)
    yield from errors


def outer_recordings(target: Target) -> Iterator[str]:
    """A Hybrid's outer recordings describe its Dense Component's planes."""
    errors: list[str] = []
    _validate_hybrid_recordings(target.store, component(target).store, errors)
    yield from errors


def _recorded_from_block(
    label: str, block: Any
) -> tuple[str, Any, Any, Any, Any] | None:
    """One manifest provenance block as a recording, or `None` when it names no shape."""
    if isinstance(block, dict) and "chunk_shape" in block:
        return (
            label,
            block.get("chunk_shape"),
            block.get("shard_shape"),
            block.get("compressor"),
            block.get("zarr_format"),
        )
    return None


def _recorded_from_index(
    connection: sqlite3.Connection,
) -> tuple[str, Any, Any, Any, Any] | None:
    """The `index.sqlite` `dense` blob as a recording, or `None` when absent.

    An absent `dense` blob is not a disagreement -- none of the Dense
    Reference-Completion path writes one -- so only a present blob is judged.
    """
    blob = get_metadata(connection, "dense", default=None)
    if isinstance(blob, dict):
        return (
            "index.sqlite dense metadata",
            blob.get("chunk_shape"),
            blob.get("shard_shape"),
            blob.get("compressor"),
            blob.get("zarr_format"),
        )
    return None


def _recorded_layouts(
    manifest: Any,
    connection: sqlite3.Connection,
    root: Any,
    errors: list[str],
    *,
    require_all: bool,
) -> list[tuple[str, Any, Any, Any, Any]]:
    """The three places a Dense release records its chunk shape, as (label, ...).

    Returns `(label, chunk_shape, shard_shape, compressor, zarr_format)` for the
    manifest provenance, the `index.sqlite` `dense` blob and the `data.zarr` root
    attrs.  All three are written by the Dense builders and by the converter
    (#245) and none is derived from the arrays themselves, so they are the copies
    a disagreeing manifest hides behind.

    `require_all` is set for a 0.2.0 release, where spec §10a mandates all three:
    an absent `provenance.dense` or `dense` blob is then an error, not a skipped
    comparison (issue #248 review round 3).  A 0.1.0 release is only judged on
    the recordings it carries -- the completion path writes no `dense` blob.
    """
    recorded: list[tuple[str, Any, Any, Any, Any]] = []
    provenance = manifest.provenance if isinstance(manifest.provenance, dict) else {}
    block = None
    for key in ("dense", "hybrid"):
        block = _recorded_from_block(f"manifest.json provenance.{key}", provenance.get(key))
        if block is not None:
            recorded.append(block)
            break
    if block is None and require_all:
        errors.append(
            "manifest.json provenance.dense: no chunk_shape recorded, so the release "
            "does not describe its Dense planes' layout (spec §10a)"
        )
    index = _recorded_from_index(connection)
    if index is not None:
        recorded.append(index)
    elif require_all:
        errors.append(
            "index.sqlite dense metadata: no dense blob recorded, so the release does "
            "not describe its Dense planes' layout (spec §10a)"
        )
    attrs = dict(root.attrs)
    recorded.append(
        (
            "data.zarr root attrs",
            attrs.get("chunk_shape"),
            attrs.get("shard_shape"),
            attrs.get("compressor"),
            attrs.get("zarr_format"),
        )
    )
    return recorded


def _recorded_layout_mismatches(
    recorded: list[tuple[str, Any, Any, Any, Any]],
    planes: list[tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...] | None]],
    errors: list[str],
    *,
    zarr_label: str,
    require_compressor: bool,
) -> None:
    """Hold every recording against every present plane of one Zarr tree."""
    for name, shape, actual_inner, actual_shard in planes:
        for label, chunk_shape, shard_shape, _compressor, _zarr_format in recorded:
            plane_label = f"{label} ({zarr_label}/{name})"
            expected_inner = _recorded_chunk_errors(
                plane_label, chunk_shape, shape, actual_inner, errors
            )
            if expected_inner is not None:
                errors.extend(
                    _recorded_shard_errors(
                        plane_label, shard_shape, shape, expected_inner, actual_shard
                    )
                )
    errors.extend(
        _recorded_compressor_errors(
            recorded, planes[0][3], require=require_compressor
        )
    )
    errors.extend(_recorded_zarr_format_errors(recorded, planes[0][3]))


def _validate_recorded_layout(
    manifest: Any, connection: sqlite3.Connection, root: Any, errors: list[str]
) -> None:
    """Every recorded chunk and shard shape must describe the arrays present.

    A manifest that describes one shape over arrays of another is a silent
    failure class: a reader sizing its reads from the manifest would decode the
    wrong blocks.  Each of the three recordings is clipped to the plane's own
    dimensions exactly as the seam's role policy clips it (so a small store's
    build-wide hint is not a disagreement), then compared with the Dense
    plane's actual **inner** chunk and, for a sharded 0.2.0 release, its shard
    (issue #245).
    """
    if "z" not in root:
        return
    planes = _recorded_dense_planes(root)
    sharded = planes[0][3] is not None
    recorded = _recorded_layouts(manifest, connection, root, errors, require_all=sharded)
    _recorded_layout_mismatches(
        recorded,
        planes,
        errors,
        zarr_label="data.zarr",
        require_compressor=sharded,
    )


def _recorded_dense_planes(
    root: Any,
) -> list[tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...] | None]]:
    """Every present Dense statistic plane as (name, shape, inner chunk, shard).

    The recorded layout is one grid, but the rule must hold it against **every**
    plane that uses it: checking only `z` would let `se`, `eaf` or the imputed
    mask disagree with the manifest and still validate (issue #245 review).
    The imputed mask is present only in a Reference-Completed release.
    """
    planes: list[tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...] | None]] = []
    for name in ("z", "se", "eaf", "imputed"):
        if name not in root:
            continue
        plane = root[name]
        shape = tuple(int(size) for size in plane.shape)
        shard = (
            None
            if getattr(plane, "shards", None) is None
            else tuple(int(size) for size in plane.shards)
        )
        planes.append((name, shape, inner_chunk_of(plane), shard))
    return planes


def _recorded_chunk_errors(
    label: str,
    chunk_shape: Any,
    shape: tuple[int, ...],
    actual_inner: tuple[int, ...],
    errors: list[str],
) -> tuple[int, ...] | None:
    """One recording's chunk shape against the plane's inner chunk.

    Returns the clipped inner chunk for the shard check that follows, or `None`
    when the recording cannot describe this plane at all.
    """
    if chunk_shape is None:
        errors.append(
            f"{label}: no chunk_shape recorded, so the release does not describe its own "
            "Dense plane layout (issue #245)"
        )
        return None
    if len(chunk_shape) != len(shape):
        errors.append(
            f"{label} records chunk_shape {list(chunk_shape)} for a {len(shape)}-D plane "
            "(issue #245)"
        )
        return None
    expected_inner = tuple(
        min(int(size), dim) for size, dim in zip(chunk_shape, shape, strict=True)
    )
    if expected_inner != actual_inner:
        errors.append(
            f"{label} records chunk_shape {list(chunk_shape)}, which describes "
            f"{list(expected_inner)} for a {shape} plane, but data.zarr/z is chunked "
            f"{list(actual_inner)}; a manifest and arrays that disagree are a silent "
            "failure class (issue #245)"
        )
    return expected_inner


def _recorded_shard_errors(
    label: str,
    shard_shape: Any,
    shape: tuple[int, ...],
    expected_inner: tuple[int, ...],
    actual_shard: tuple[int, ...] | None,
) -> list[str]:
    """One recording's shard shape against the plane's shard, or nothing for v2."""
    if actual_shard is None:
        return []
    if shard_shape is None:
        return [
            f"{label}: data.zarr/z is sharded ({list(actual_shard)}) but no shard_shape "
            "is recorded (issue #245)"
        ]
    if len(shard_shape) != len(shape):
        return [
            f"{label} records shard_shape {list(shard_shape)} for a {len(shape)}-D Dense "
            "plane (issue #245)"
        ]
    recorded_shard = tuple(int(size) for size in shard_shape)
    try:
        expected_shard = shard_layout(
            ArrayRole.DENSE_STATISTIC_PLANE,
            shape,
            inner_chunk=expected_inner,
            dense_shard=(recorded_shard[0], recorded_shard[1]),
        )
    except ValueError as exc:
        return [
            f"{label} records shard_shape {list(shard_shape)}, which cannot describe a "
            f"{shape} plane chunked {list(expected_inner)}: {exc}"
        ]
    if expected_shard != actual_shard:
        return [
            f"{label} records shard_shape {list(shard_shape)}, which describes "
            f"{list(expected_shard)} for this plane, but data.zarr/z is sharded "
            f"{list(actual_shard)} (issue #245)"
        ]
    return []


def _recorded_compressor_errors(
    recorded: list[tuple[str, Any, Any, Any, Any]],
    actual_shard: tuple[int, ...] | None,
    *,
    require: bool,
) -> list[str]:
    """The three recordings' compressors agree, and name the format present.

    `require` is set where spec §10a mandates the compressor: the Hybrid outer
    recording always, and any sharded (0.2.0) recording.  A 0.1.0 nested Dense
    Component's `provenance.dense` predates the completion writing one, so a v2
    recording is only judged when it carries a compressor -- the builder gap is
    reported, not silently accepted or turned into a false failure.
    """
    expected = SHARDED_COMPRESSOR_RECORD if actual_shard is not None else COMPRESSOR_RECORD
    kind = "Zarr v3 sharded" if actual_shard is not None else "Zarr v2"
    return _compressor_consistency_errors(recorded) + _compressor_record_errors(
        recorded, expected, kind, require=require
    )


def _compressor_consistency_errors(
    recorded: list[tuple[str, Any, Any, Any, Any]],
) -> list[str]:
    """The recordings must not contradict each other about the compressor."""
    present = [entry[3] for entry in recorded if entry[3] is not None]
    if len(present) <= 1 or all(entry == present[0] for entry in present):
        return []
    return [
        "manifest provenance, index.sqlite dense metadata and data.zarr root attrs "
        f"record different compressors ({present}); they describe the same arrays and "
        "must agree (issue #245)"
    ]


def _compressor_record_errors(
    recorded: list[tuple[str, Any, Any, Any, Any]],
    expected: dict[str, Any],
    kind: str,
    *,
    require: bool,
) -> list[str]:
    """Every recording MUST name the format's one compressor when `require`."""
    errors: list[str] = []
    for label, _chunk, _shard, compressor, _zarr_format in recorded:
        if compressor is None:
            if require:
                errors.append(
                    f"{label}: no compressor recorded, so the release does not "
                    "describe the codec its arrays are stored with (spec §10a)"
                )
        elif compressor != expected:
            errors.append(
                f"{label} records compressor {compressor!r}, not the {kind} record "
                f"{expected!r} (issue #245)"
            )
    return errors


def _recorded_zarr_format_errors(
    recorded: list[tuple[str, Any, Any, Any, Any]], actual_shard: tuple[int, ...] | None
) -> list[str]:
    """A sharded (0.2.0) recording MUST name Zarr format 3; a v2 one must not claim otherwise.

    Spec §10a records `zarr_format` alongside the compressor.  A 0.1.0 release has
    no shard and does not record a format, but a 0.2.0 one must, and a recording
    that names the wrong format is the same silent failure class as a wrong chunk
    (issue #248 review round 2).

    The value must be the **integer** 2 or 3, compared without coercion: a JSON
    string `"3"`, a float `3.0`/`3.5` and `true` are all validation errors rather
    than being `int(...)`-ed into a passing value or raising (issue #248 review
    round 3).
    """
    errors: list[str] = []
    for label, _chunk, _shard, _compressor, zarr_format in recorded:
        value = _zarr_format_int(zarr_format)
        if actual_shard is not None:
            if value is None:
                errors.append(
                    f"{label}: zarr_format {zarr_format!r} is not the integer 3, which a "
                    "Zarr v3 sharded release MUST record (spec §10a)"
                )
            elif value != 3:
                errors.append(
                    f"{label} records zarr_format {zarr_format!r}, not 3; the arrays "
                    "are Zarr v3 sharded (spec §10a)"
                )
        elif zarr_format is not None and value != 2:
            errors.append(
                f"{label} records zarr_format {zarr_format!r}, not the integer 2; the "
                "arrays are Zarr v2 (spec §10a)"
            )
    return errors


def _zarr_format_int(value: Any) -> int | None:
    """`value` as a Zarr format integer, or `None` when it is not one.

    Only a true integer counts.  `bool` is excluded although it is an `int`
    subclass, and strings, floats and `None` are refused without coercion.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return None
    return int(value)


def _validate_hybrid_recordings(store: Any, dense_store: Any, errors: list[str]) -> None:
    """A Hybrid's outer layout recordings must agree with its nested component.

    Spec §10a: the outer `provenance.hybrid` and the outer `index.sqlite` `dense`
    blob describe the nested Dense Component's arrays and MUST agree with them,
    just as the component's own three recordings do.  `_validate_recorded_layout`
    never sees them -- it runs on the component's own root and returns when that
    root has no `z` -- so without this seam a Hybrid whose outer manifest was
    left on an older layout validates, and a reader sizing its reads from
    `provenance.hybrid` decodes the wrong blocks (issue #248 review round 1).
    """
    planes = _recorded_dense_planes(dense_store.arrays(mode="r"))
    if not planes:
        return
    _recorded_layout_mismatches(
        _hybrid_outer_recordings(store, errors),
        planes,
        errors,
        zarr_label="dense/data.zarr",
        require_compressor=True,
    )


def _hybrid_outer_recordings(
    store: Any, errors: list[str]
) -> list[tuple[str, Any, Any, Any, Any]]:
    """The outer `provenance.hybrid` and index blob, as recordings.

    The outer manifest MUST carry `chunk_shape` (spec §10a); a missing one is an
    error rather than a silently skipped comparison.  The outer `dense` blob is
    judged only when present, as for a standalone release.
    """
    recorded: list[tuple[str, Any, Any, Any, Any]] = []
    provenance = store.manifest.provenance if isinstance(store.manifest.provenance, dict) else {}
    block = _recorded_from_block(
        "manifest.json provenance.hybrid", provenance.get("hybrid")
    )
    if block is None:
        errors.append(
            "manifest.json provenance.hybrid: no chunk_shape recorded, so the release "
            "does not describe its Dense Component's layout (spec §10a, issue #248)"
        )
    else:
        recorded.append(block)
    connection = store.index_connection()
    try:
        index = _recorded_from_index(connection)
    finally:
        connection.close()
    if index is not None:
        recorded.append(index)
    return recorded


#: A grid's recordings, and a Hybrid's outer ones for its Dense Component.
RULES: tuple[Rule, ...] = (
    Check(name="layout.recorded", kinds=Kind.GRID, after=(REQUIRED,), run=recorded),
    Check(
        name="hybrid.recordings",
        kinds=Kind.RELEASE,
        when=is_hybrid,
        after=(REQUIRED, COMPONENT),
        run=outer_recordings,
    ),
)
