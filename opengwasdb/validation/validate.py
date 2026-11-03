"""Store validation."""

from __future__ import annotations

import numbers
import sqlite3
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.associations import Cells, DenseGridReader, Tally
from opengwasdb.associations import present as present_cells
from opengwasdb.completion.schema import COMPLETION_QUALITY_COLUMNS
from opengwasdb.encoding import (
    RaggedSePlane,
    StoreCodec,
    StoreEncoding,
)
from opengwasdb.index.sqlite import get_metadata
from opengwasdb.layouts.dense.indexed_subsets import validate_indexed_subsets
from opengwasdb.layouts.hybrid.layout import dense_component_path, dense_to_shared_path
from opengwasdb.model.analyses import (
    read_analyses,
)
from opengwasdb.model.enums import (
    CompletionState,
    PrimaryStorageLayout,
)
from opengwasdb.store.arrays import (
    COMPRESSOR_RECORD,
    SHARDED_COMPRESSOR_RECORD,
    ArrayRole,
    array_length,
    inner_chunk_of,
    open_group,
    shard_layout,
)
from opengwasdb.store.open import (
    OpenGWASDBStore,
    UnsupportedFormatVersion,
    open_store,
)
from opengwasdb.top_hits import threshold_key
from opengwasdb.validation.rule import (
    Check,
    Finding,
    Kind,
    Rule,
    Severity,
    Target,
    ValidationOptions,
    ValidationResult,
    check_registry,
)
from opengwasdb.validation.rules import analyses as analyses_rules
from opengwasdb.validation.rules import associations as association_rules
from opengwasdb.validation.rules import by_variant as by_variant_rules
from opengwasdb.validation.rules import completion as completion_rules
from opengwasdb.validation.rules import csr as csr_rules
from opengwasdb.validation.rules import encoding as encoding_rules
from opengwasdb.validation.rules import hybrid as hybrid_rules
from opengwasdb.validation.rules import release as release_rules
from opengwasdb.validation.rules import sample_size as sample_size_rules
from opengwasdb.validation.rules import sequence as sequence_rules
from opengwasdb.validation.rules import side_tables as side_table_rules
from opengwasdb.validation.rules import top_hits as top_hit_rules
from opengwasdb.validation.rules import variants as variant_rules
from opengwasdb.validation.rules.associations import DECODABLE, grid_counts
from opengwasdb.validation.rules.csr import ARRAYS, LENGTHS
from opengwasdb.validation.rules.encoding import PLAN
from opengwasdb.validation.rules.hybrid import DENSE_TO_SHARED
from opengwasdb.validation.rules.release import REQUIRED
from opengwasdb.validation.rules.variants import representative_variant_indices
from opengwasdb.validation.run import evaluate, selected
from opengwasdb.validation.scans import sequence_offsets
from opengwasdb.variants import (
    VariantAxis,
    variant_table_path,
)


def validate_store(
    path: str | Path,
    source: str | Path | Sequence[str | Path] | None = None,
    *,
    fidelity_samples: int = 2000,
    fidelity_seed: int = 0,
    source_assembly: str | None = None,
    chain_file: str | Path | None = None,
    rules: Collection[str] | None = None,
) -> ValidationResult:
    """Validate a Store Release directory by running every Validation Rule that applies.

    Internal (structural) validation is always run.  When ``source`` is given —
    the original dataset the store was built from — a **source-fidelity** check
    is also run for dense stores: a random sample of source associations is
    compared against the store, orienting through the stored source-ALID
    provenance (or, for lifted stores, via liftover).

    ``rules`` runs only the named rules and the rules they run after; a name no
    rule has is refused before anything is read.
    """
    if rules is not None:
        selected(RULES, rules)
    options = ValidationOptions(
        source, fidelity_samples, fidelity_seed, source_assembly, chain_file
    )
    errors: list[str] = []
    store = _load_manifest(Path(path), errors)
    if store is None:
        return ValidationResult(
            findings=tuple(Finding(MANIFEST_RULE, Severity.ERROR, e, ".") for e in errors)
        )
    return evaluate(RULES, release_targets(store, options), rules)


def release_targets(store: OpenGWASDBStore, options: ValidationOptions) -> list[Target]:
    """The targets one Store Release is judged as.

    A Dense release is a release and its grid; a Ragged one, a release and its
    association sequence. A Hybrid release is its outer release, its Ragged
    Overflow, and its Dense Component, which is a nested Store Release (spec §16)
    judged as one, by the same rules, when it opens as a Dense release; when it
    does not, the Hybrid rules say why and nothing is judged against it.
    """
    release = Target(Kind.RELEASE, ".", store, options=options)
    layout = store.manifest.primary_layout
    if layout is PrimaryStorageLayout.DENSE:
        return [release, Target(Kind.GRID, "data.zarr", store, parent=release, options=options)]
    sequence = Target(Kind.CSR, "data.zarr/ragged", store, parent=release, options=options)
    if layout is PrimaryStorageLayout.RAGGED:
        return [release, sequence]
    targets = [release, sequence]
    component = _load_manifest(dense_component_path(store.path), [])
    if component is not None and component.manifest.primary_layout is PrimaryStorageLayout.DENSE:
        nested = Target(Kind.RELEASE, "dense", component, parent=release, options=options)
        grid = Target(Kind.GRID, "dense/data.zarr", component, parent=nested, options=options)
        targets += [nested, grid]
    return targets


# --- the orchestrators rules replace, each run whole until its checks move out ---

#: The rule a release that cannot be opened fails: nothing else can be targeted.
MANIFEST_RULE = "manifest.loads"


@dataclass(frozen=True, kw_only=True)
class _Legacy(Rule):
    """A per-layout orchestrator from before rules had names, run as one rule.

    Each is deleted as the checks it runs move into rules of their own.
    """

    run: Callable[[Target, list[str], list[str]], None]

    def findings(self, target: Target) -> Iterator[Finding]:
        errors: list[str] = []
        warnings: list[str] = []
        try:
            self.run(target, errors, warnings)
        finally:
            # Yielded on the way out of an exception too: what the orchestrator
            # found before it failed is reported ahead of the failure.
            yield from (self.finding(target, m, Severity.ERROR) for m in errors)
            yield from (self.finding(target, m, Severity.WARNING) for m in warnings)


def _is(layout: PrimaryStorageLayout) -> Callable[[Target], bool]:
    return lambda target: target.layout is layout


def _not_dense(target: Target) -> bool:
    return target.layout is not PrimaryStorageLayout.DENSE


def _source_given(target: Target) -> bool:
    return target.options.source is not None


def _legacy_ragged_completion(target: Target, errors: list[str], _warnings: list[str]) -> None:
    store = target.store
    offsets = sequence_offsets(target)
    _validate_ragged_completion(
        store.data_path / "ragged", store, int(offsets[-1]), len(offsets) - 1, errors
    )


def _legacy_hybrid(target: Target, errors: list[str], _warnings: list[str]) -> None:
    _validate_hybrid_store(target.store, errors)


def _legacy_dense_completion(target: Target, errors: list[str], _warnings: list[str]) -> None:
    _dense_completion(target, errors)


def _legacy_rho(target: Target, errors: list[str], _warnings: list[str]) -> None:
    _dense_rho(target, errors)


def _legacy_dense_layout(target: Target, errors: list[str], _warnings: list[str]) -> None:
    _dense_layout(target, errors)


def _legacy_overflow(target: Target, errors: list[str], _warnings: list[str]) -> None:
    _overflow(target, errors)


def _completed(target: Target) -> bool:
    return target.store.manifest.completion_state is CompletionState.REFERENCE_COMPLETED


def _top_level_dense_grid(target: Target) -> bool:
    return target.layout is PrimaryStorageLayout.DENSE and not target.is_component


def _fidelity_unsupported(_target: Target) -> list[str]:
    return ["source-fidelity check is only supported for dense stores"]


def _legacy_fidelity(target: Target) -> list[str]:
    errors: list[str] = []
    options = target.options
    assert options.source is not None
    _validate_source_fidelity(
        target.store,
        options.source,
        errors,
        n_samples=options.fidelity_samples,
        seed=options.fidelity_seed,
        source_assembly=options.source_assembly,
        chain_file=options.chain_file,
    )
    return errors


_DENSE_COMPLETION = "legacy.dense_completion"
_RAGGED_COMPLETION = "legacy.ragged_completion"

#: Every rule about a sequence's own arrays and values: the by_variant and Top-Hit
#: cross-checks read the sequence, and would "confirm" one these found broken.
_SEQUENCE_JUDGED: tuple[str, ...] = (
    *PLAN,
    *(
        rule.name
        for rule in (
            *csr_rules.RULES,
            *sample_size_rules.CHECKS,
            *sequence_rules.RULES,
            *association_rules.PRESENCE_RULES,
            *association_rules.EAF_RULES,
            *side_table_rules.RULES,
            *sample_size_rules.RULES,
        )
        if rule.severity is Severity.ERROR
    ),
)

#: The rules before source fidelity, in order: it samples what a Store holds, so it
#: runs only on a release none of them found wrong.
_BEFORE_FIDELITY: tuple[Rule, ...] = (
    Check(name=MANIFEST_RULE, kinds=Kind.RELEASE, run=lambda _target: ()),
    *release_rules.RULES,
    *variant_rules.RULES,
    *analyses_rules.TABLE_RULES,
    *csr_rules.RULES,
    *hybrid_rules.RULES,
    *encoding_rules.RULES,
    *sample_size_rules.CHECKS,
    *by_variant_rules.CHECKS,
    *association_rules.CHECKS,
    *completion_rules.CHECKS,
    *sequence_rules.RULES,
    by_variant_rules.COUNTS,
    *association_rules.PRESENCE_RULES,
    *completion_rules.RULES,
    *association_rules.EAF_RULES,
    *side_table_rules.RULES,
    *sample_size_rules.RULES,
    *top_hit_rules.STRUCTURE_RULES,
    *top_hit_rules.VALUE_RULES,
    by_variant_rules.SAMPLE,
    _Legacy(
        name=_RAGGED_COMPLETION,
        kinds=Kind.CSR,
        when=lambda target: _is(PrimaryStorageLayout.RAGGED)(target) and _completed(target),
        after=(ARRAYS, LENGTHS, *PLAN, DECODABLE),
        run=_legacy_ragged_completion,
    ),
    _Legacy(
        name=_DENSE_COMPLETION,
        kinds=Kind.GRID,
        when=_completed,
        after=(REQUIRED,),
        run=_legacy_dense_completion,
    ),
    _Legacy(name="legacy.rho", kinds=Kind.GRID, after=(REQUIRED,), run=_legacy_rho),
    _Legacy(
        name="legacy.dense_layout", kinds=Kind.GRID, after=(REQUIRED,), run=_legacy_dense_layout
    ),
    _Legacy(
        name="legacy.overflow",
        kinds=Kind.CSR,
        when=_is(PrimaryStorageLayout.HYBRID),
        after=(*_SEQUENCE_JUDGED, DENSE_TO_SHARED),
        run=_legacy_overflow,
    ),
    _Legacy(
        name="legacy.hybrid",
        kinds=Kind.RELEASE,
        when=_is(PrimaryStorageLayout.HYBRID),
        after=(REQUIRED,),
        run=_legacy_hybrid,
    ),
    *analyses_rules.ORIENTATION_RULES,
)

RULES: tuple[Rule, ...] = check_registry(
    [
        *_BEFORE_FIDELITY,
        Check(
            name="legacy.fidelity_unsupported",
            kinds=Kind.RELEASE,
            when=lambda target: _source_given(target) and _not_dense(target),
            run=_fidelity_unsupported,
        ),
        Check(
            name="legacy.fidelity",
            kinds=Kind.GRID,
            when=lambda target: _source_given(target) and _top_level_dense_grid(target),
            after=tuple(r.name for r in _BEFORE_FIDELITY if r.severity is Severity.ERROR),
            run=_legacy_fidelity,
        ),
    ]
)


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


def _dense_completion(target: Target, errors: list[str]) -> None:
    """A Reference-Completed grid's completion metadata."""
    store = target.store
    with store.index_connection() as connection:
        _validate_completion_metadata(
            connection, store.analyses_path, grid_counts(target)[1], errors
        )


def _dense_rho(target: Target, errors: list[str]) -> None:
    """The optional Rho Matrix against `analyses.tsv`."""
    _validate_rho(target.group, grid_counts(target)[1], errors)


def _dense_layout(target: Target, errors: list[str]) -> None:
    """The recorded layout against the planes, and any Indexed Variant Subsets."""
    store = target.store
    n_variants, n_analyses = grid_counts(target)
    # After the shape seam, so a plane that does not span the axis is named as
    # such rather than only as a layout disagreement (#245).
    with store.index_connection() as connection:
        _validate_recorded_layout(store.manifest, connection, target.group, errors)
    # The optional indexed-subset namespace (ADR 0061, #264).  A release without it
    # stays valid and unchanged; a release with it must hold only published,
    # self-consistent, value-equal subsets.
    validate_indexed_subsets(
        store.path, target.group, store.manifest, n_variants, n_analyses, errors
    )


def _overflow(target: Target, errors: list[str]) -> None:
    """The Dense Component's coverage of the shared variant table."""
    store = target.store
    dense_to_shared = np.load(dense_to_shared_path(store.path))
    _validate_dense_coverage(
        store.path, dense_component_path(store.path), dense_to_shared, len(dense_to_shared), errors
    )


def _validate_completion_metadata(
    connection: sqlite3.Connection, analyses_path: Path, n_analyses: int, errors: list[str]
) -> None:
    """A Reference-Completed Dense store's completion metadata beside its arrays.

    The `completion_quality` table and the `completion_n_missing_total` column; the
    mask and panel flags are `completion.mask`, and what they hold is judged by
    the completion rules over the band scan.
    """
    _validate_completion_quality_table(connection, n_analyses, errors)
    analyses_cols = set(read_analyses(analyses_path).fieldnames)
    if "completion_n_missing_total" not in analyses_cols:
        errors.append(
            "analyses.tsv is missing completion_n_missing_total for a reference-completed store"
        )


def _validate_completion_quality_table(
    connection: sqlite3.Connection, n_analyses: int, errors: list[str]
) -> None:
    """Seam: the completion_quality table exists, is complete, and is in range.

    ``completion_quality.analysis_index`` is a positional reference into
    analyses.tsv, not a SQL foreign key (ADR 0030 consequences) -- this is the
    validator rule that decision says must stand in for the database engine's
    constraint, and it is shared by Dense and Ragged Reference-Completed
    stores so a quality row can never describe an Analysis that does not exist.
    """
    tables = {
        r[0]
        for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "completion_quality" not in tables:
        errors.append(
            "index.sqlite is missing the completion_quality table for a reference-completed store"
        )
        return
    cols = {
        r[1] for r in connection.execute("PRAGMA table_info(completion_quality)").fetchall()
    }
    required = COMPLETION_QUALITY_COLUMNS
    missing_cols = required - cols
    if missing_cols:
        errors.append(
            f"completion_quality table is missing columns: {', '.join(sorted(missing_cols))}"
        )
        return
    out_of_range = connection.execute(
        "SELECT COUNT(*) FROM completion_quality "
        "WHERE analysis_index < 0 OR analysis_index >= ?",
        (n_analyses,),
    ).fetchone()[0]
    if out_of_range:
        errors.append(
            f"completion_quality has {out_of_range} row(s) with analysis_index "
            f"outside analyses.tsv's range [0, {n_analyses})"
        )


def _decoded_csr_se(root: Any, encoding: StoreEncoding, n_assoc: int, label: str) -> Any:
    """Decoded CSR SE, or the reason it cannot be decoded.

    A residual plane needs its coefficients, its side table and a decodable EAF
    for every predicted cell. When one of those is wrong the codec raises, and
    validation must report that as a finding about the release rather than
    letting it escape as a traceback from `validate_store`.
    """
    try:
        return RaggedSePlane.open(root, encoding).slice(0, n_assoc), None
    except ValueError as exc:
        return None, f"{label} cannot be decoded under the declared plan: {exc}"


def _validate_ragged_imputed_mask(
    root: Any, n_assoc: int, errors: list[str]
) -> np.ndarray | None:
    """Seam: the loaded Ragged imputed mask, or ``None`` when it cannot be used.

    A missing array or a length that disagrees with the CSR offsets stops the
    completion seam; a mask holding values other than 0/1 records its error but
    still comes back so the missingness cross-checks can report what the mask
    declares.
    """
    if "imputed" not in root:
        errors.append("reference-completed store is missing data.zarr/ragged/imputed")
        return None
    imp: np.ndarray = root["imputed"][:]
    if len(imp) != n_assoc:
        errors.append(
            f"data.zarr/ragged/imputed has {len(imp)} entries but offsets imply {n_assoc}"
        )
        return None
    if not np.all((imp == 0) | (imp == 1)):
        errors.append("data.zarr/ragged/imputed contains values other than 0 and 1")
    return imp


def _validate_ragged_imputed_missingness(
    root: Any, encoding: StoreEncoding, imp: np.ndarray, n_assoc: int, errors: list[str]
) -> None:
    """Seam: what an imputed=1 row declares about the row's z and se.

    Imputed rows carry both z and se (spec §15, ADR 0070). "Present" is the
    complement of the plane's own missing marker, not ``isfinite`` of a float:
    an integer plane holds no NaN to test for, and an infinite SE is a value.
    Whether z and se agree with each other is checked on every Ragged
    component, completed or not (`associations.markers_paired`).
    """
    codec = StoreCodec(encoding)
    z_missing = codec.missing_mask(root["z"][:])
    se_vals, se_error = _decoded_csr_se(root, encoding, n_assoc, "data.zarr/ragged/se")
    if se_error is not None:
        errors.append(se_error)
        return
    tally = Tally.of(z_missing, np.isnan(se_vals), imp)
    if tally.imputed_z_missing:
        errors.append("imputed=1 rows have missing z-scores")
    if tally.imputed_se_missing:
        errors.append("imputed=1 rows have NaN se values")
    if tally.imputed_z_missing:
        errors.append("missing z rows have imputed=1 (inconsistent)")


def _validate_ragged_imputed(
    root: Any, encoding: StoreEncoding, n_assoc: int, errors: list[str]
) -> bool:
    """Seam: the Ragged imputed mask and the missingness it declares.

    Returns False when the mask cannot be used, stopping the pipeline.
    """
    imp = _validate_ragged_imputed_mask(root, n_assoc, errors)
    if imp is None:
        return False
    _validate_ragged_imputed_missingness(root, encoding, imp, n_assoc, errors)
    return True


def _validate_ragged_completion(
    ragged_path: Path,
    store: OpenGWASDBStore,
    n_assoc: int,
    n_analyses_csr: int,
    errors: list[str],
) -> None:
    """Validate the imputed mask and completion_quality table in a Reference-Completed store."""
    root = open_group(ragged_path)
    if not _validate_ragged_imputed(root, store.manifest.encoding, n_assoc, errors):
        return
    with store.index_connection() as conn:
        _validate_completion_quality_table(conn, n_analyses_csr, errors)


def _validate_hybrid_store(store: OpenGWASDBStore, errors: list[str]) -> None:
    """A Hybrid release's Dense Component opens as Dense, under the release's plan,
    and the outer recordings describe it (ADR 0026 / issue 059).

    The component is judged as a Dense release of its own, and the Overflow as a
    sequence of its own.
    """
    dense_store = _hybrid_dense_component(store, errors)
    if dense_store is None:
        return
    # The outer recordings must describe the component, which is judged as a Dense
    # release of its own.
    _validate_hybrid_recordings(store, dense_store, errors)


def _n_variants(store_path: Path) -> int:
    """The rows of a release's Store Variant Table, by its offset sidecar."""
    axis = VariantAxis(store_path)
    try:
        return axis.n_variants
    finally:
        axis.close()


def _hybrid_dense_component(
    store: OpenGWASDBStore, errors: list[str]
) -> OpenGWASDBStore | None:
    """The Dense Component, opened, Dense, and under the release's plan; else None."""
    dense_store = _load_manifest(dense_component_path(store.path), errors)
    if dense_store is None:
        return None
    if dense_store.manifest.primary_layout is not PrimaryStorageLayout.DENSE:
        errors.append(
            f"Dense Component manifest must be primary_layout=dense "
            f"(got {dense_store.manifest.primary_layout.value})"
        )
        return None
    # One plan for both components (issue #119): they partition one Analysis's
    # associations, so a result assembled from both is only coherent if they were
    # encoded the same way, and nothing but this check couples the two manifests.
    encoding_error = _component_encoding_error(dense_store, store.manifest.encoding)
    if encoding_error is not None:
        errors.append(encoding_error)
        return None
    return dense_store


def _component_encoding_error(
    dense_store: OpenGWASDBStore, release: StoreEncoding
) -> str | None:
    """Why a Hybrid release's two components disagree about their statistic encoding.

    They partition one Analysis's associations, so a result assembled from both is
    only coherent if they were written under one plan -- and each component carries
    its own manifest, so nothing but this check couples them. `eaf_reference` is the
    one field that legitimately differs: it records a physical fact about the
    component that declares it, and only the Dense Component has imputed cells for a
    reference frequency to describe (ADR 0037 §4). Compared with it normalised away,
    so a real disagreement about the *encoding* still fails.
    """
    encoding = dense_store.manifest.encoding
    if encoding.with_eaf_reference(False) == release:
        return None
    return (
        "the Dense Component and the Hybrid release declare different statistic "
        f"encodings ({encoding.to_manifest()} vs {release.to_manifest()}); both "
        "components of a Hybrid release are written under one plan"
    )


def _validate_dense_coverage(
    store_path: Path,
    dense_dir: Path,
    dense_to_shared: np.ndarray,
    n_panel: int,
    errors: list[str],
) -> None:
    """The shared row each dense row maps to must carry the Dense Component's ALID."""
    dense_axis = VariantAxis(dense_dir)
    shared_axis = VariantAxis(store_path)
    try:
        for r in representative_variant_indices(n_panel):
            dense_rec = dense_axis.by_index(r)
            shared_rec = shared_axis.by_index(int(dense_to_shared[r]))
            if dense_rec is None or shared_rec is None or dense_rec.alid != shared_rec.alid:
                errors.append(
                    "dense_to_shared coverage violated: Dense Component row "
                    f"{r} does not match its shared variant table row"
                )
                break
    finally:
        dense_axis.close()
        shared_axis.close()


def _load_manifest(store_path: Path, errors: list[str]) -> OpenGWASDBStore | None:
    try:
        return open_store(store_path)
    except UnsupportedFormatVersion as exc:
        errors.append(str(exc))
    except KeyError as exc:
        errors.append(f"manifest missing required field: {exc.args[0]}")
    except ValueError as exc:
        errors.append(f"manifest has invalid enum value: {exc}")
    except FileNotFoundError:
        errors.append("missing manifest.json")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"manifest is malformed: {exc}")
    return None




_RHO_REQUIRED_ATTRS = (
    "z_thresh",
    "min_nulls",
    "grid_n",
    "window_bp",
    "n_variants_used",
    "observed_only",
    "method",
    "n_analyses",
)


def _validate_rho(root: Any, n_analyses: int, errors: list[str]) -> None:
    """Validate the optional ``data.zarr/rho`` group when present (PRD
    "Validation", issue 049). A store without it stays valid unchanged --
    Rho is opt-in, add-in-place metadata (ADR 0025), not association data.
    Structural/consistency only: this does not re-run the estimator.
    """
    if "rho" not in root:
        return
    group = root["rho"]

    missing_attrs = [a for a in _RHO_REQUIRED_ATTRS if a not in group.attrs]
    if missing_attrs:
        errors.append(f"rho group is missing provenance attrs: {', '.join(missing_attrs)}")
        return
    for name in ("rho", "n_null", "variant_index"):
        if name not in group:
            errors.append(f"rho group is missing data.zarr/rho/{name}")
    if errors:
        return

    attr_n_analyses = int(group.attrs["n_analyses"])
    if attr_n_analyses != n_analyses:
        errors.append(
            f"rho group n_analyses attr {attr_n_analyses} does not match "
            f"analyses.tsv's {n_analyses}"
        )
        return

    expected_len = n_analyses * (n_analyses - 1) // 2
    rho_vals = np.asarray(group["rho"][:], dtype=np.float32)
    n_null_vals = np.asarray(group["n_null"][:], dtype=np.int64)
    if len(rho_vals) != expected_len:
        errors.append(f"rho array has {len(rho_vals)} entries but expected {expected_len}")
    if len(n_null_vals) != expected_len:
        errors.append(f"n_null array has {len(n_null_vals)} entries but expected {expected_len}")
    if errors:
        return

    min_nulls = int(group.attrs["min_nulls"])
    finite = np.isfinite(rho_vals)
    should_be_finite = n_null_vals >= min_nulls
    if not np.array_equal(finite, should_be_finite):
        errors.append(f"rho finiteness is inconsistent with n_null support (min_nulls={min_nulls})")
    if finite.any():
        vals = rho_vals[finite]
        if np.any(vals < -1.0) or np.any(vals > 1.0):
            errors.append("rho contains values outside [-1, 1]")

    n_variants_used = int(group.attrs["n_variants_used"])
    variant_index = group["variant_index"]
    if array_length(variant_index) != n_variants_used:
        errors.append(
            f"rho variant_index has {array_length(variant_index)} entries but "
            f"n_variants_used attr says {n_variants_used}"
        )


# ── Source-fidelity check ────────────────────────────────────────────────────
# Confirm the store faithfully persisted the *values* of the original dataset —
# the one class of bug internal validation structurally cannot catch (a store
# that is self-consistent but systematically wrong: sign flip, allele
# orientation, z/se swap, wrong analysis column, liftover mis-map, quantisation
# corruption). We reuse the build's own readers, so this is a round-trip
# persistence check, not an independent oracle of the normalisation maths.

_FIDELITY_MAX_FILES = 25
_FIDELITY_SCAN_PER_FILE = 100_000
# A sampled cell is compared against the source value put through the store's
# own codec -- the same quantisation the store applied -- with a tolerance
# loose enough for that rounding but far tighter than any sign/scale/column bug.
_FIDELITY_RTOL = 1e-2
_FIDELITY_ATOL = 1e-2
# One source association in canonical form as the build readers yield it:
# (analysis_id, chrom, pos, a1, a2, z, se) -- and one sampled-and-matched store
# cell carrying its source z/se plus the source ALID for diagnostics.
_FidelitySample = tuple[str, str, int, str, str, float, float]
_FidelityCell = tuple[int, int, float, float, str]


def _normalise_assembly(name: str) -> str:
    n = str(name).strip().lower()
    if n in {"grch37", "hg19", "b37", "37"}:
        return "hg19"
    if n in {"grch38", "hg38", "b38", "38"}:
        return "hg38"
    return n


def _is_vcf_path(path: Path) -> bool:
    s = str(path).lower()
    return s.endswith((".vcf", ".vcf.gz", ".vcf.bgz", ".bcf"))


def _source_units(
    source: str | Path | Sequence[str | Path], errors: list[str]
) -> list[tuple[str | None, Path, bool]]:
    """Resolve ``source`` into (analysis_id_or_None, path, is_vcf) units.

    A single TSV whose header carries ``trait_id``/``file_path`` is treated as a
    build manifest (one analysis per file, analysis_id from ``trait_id``);
    otherwise each path is a self-describing source file (analysis_id per row).
    """
    import csv

    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            errors.append(f"source path does not exist: {p}")
            return []
        header = None
        if not _is_vcf_path(p) and not str(p).lower().endswith(".gz"):
            try:
                with open(p, encoding="utf-8") as fh:
                    header = fh.readline()
            except (OSError, UnicodeDecodeError):
                header = None
        if header is not None and "file_path" in header and "trait_id" in header:
            units: list[tuple[str | None, Path, bool]] = []
            with open(p, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    fp = Path(row["file_path"])
                    units.append((row["trait_id"], fp, _is_vcf_path(fp)))
            return units
        return [(None, p, _is_vcf_path(p))]
    return [(None, Path(item), _is_vcf_path(Path(item))) for item in source]


def _stream_unit(
    analysis_id: str | None, path: Path, is_vcf: bool
) -> Iterator[tuple[str, str, int, str, str, float, float]]:
    """Yield canonical (analysis_id, chrom, pos, a1, a2, z, se) from one source."""
    if is_vcf:
        from opengwasdb.build.vcf_source import stream_vcf_associations

        if analysis_id is None:
            return  # a VCF carries no analysis_id; it needs a manifest trait_id
        # `_eaf` is unused here: this check compares stored z/se against the
        # source, and EAF plays no part in that (ADR 0036).
        for chrom, pos, ref, alt, z, se, _eaf in stream_vcf_associations(path):
            a1, a2 = sorted((ref, alt))
            yield (analysis_id, chrom, int(pos), a1, a2, float(z), float(se))
    else:
        from opengwasdb.build.source import read_normalised_associations

        for assoc in read_normalised_associations(path):
            v = assoc.variant
            a1, a2 = sorted((v.effect_allele, v.other_allele))
            yield (
                assoc.analysis_id,
                v.chromosome,
                int(v.position),
                a1,
                a2,
                float(assoc.z),
                float(assoc.se),
            )


def _sample_sources(
    units: list[tuple[str | None, Path, bool]], n_samples: int, rng: np.random.Generator
) -> list[tuple[str, str, int, str, str, float, float]]:
    """Reservoir-sample up to ``n_samples`` associations across a random subset
    of source files (scan-capped per file so cost stays bounded)."""
    if not units:
        return []
    order = rng.permutation(len(units))[:_FIDELITY_MAX_FILES]
    reservoir: list[tuple[str, str, int, str, str, float, float]] = []
    seen = 0
    for u in order:
        analysis_id, path, is_vcf = units[int(u)]
        scanned = 0
        for item in _stream_unit(analysis_id, path, is_vcf):
            seen += 1
            scanned += 1
            if len(reservoir) < n_samples:
                reservoir.append(item)
            else:
                j = int(rng.integers(0, seen))
                if j < n_samples:
                    reservoir[j] = item
            if scanned >= _FIDELITY_SCAN_PER_FILE:
                break
    return reservoir


def _resolve_via_origin(store_path: Path, wanted: set[str]) -> dict[str, int] | None:
    """Map wanted source ALIDs → variant_index via the stored source_alid column
    in one streamed pass. Returns None if the store predates that column."""
    import pysam

    out: dict[str, int] = {}
    remaining = set(wanted)
    with pysam.BGZFile(str(variant_table_path(store_path)), "r") as handle:  # type: ignore[call-arg]
        for raw in handle:
            line = raw.decode("utf-8")
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                return None  # no source_alid column → caller falls back to liftover
            src = fields[7]
            if src in remaining:
                out[src] = int(fields[2])
                remaining.discard(src)
                if not remaining:
                    break
    return out


def _draw_fidelity_samples(
    source: str | Path | Sequence[str | Path],
    errors: list[str],
    *,
    n_samples: int,
    seed: int,
) -> list[_FidelitySample] | None:
    """Sampling phase: deterministically reservoir-sample source associations.

    ``source`` is resolved into per-file units, then up to ``n_samples``
    associations are drawn with a generator seeded by ``seed``, so repeated
    validation of the same store reproduces the same sample. Returns None --
    having already recorded why in ``errors`` -- when the source is unusable
    (missing path) or no associations could be read from it.
    """
    units = _source_units(source, errors)
    if errors or not units:
        return None
    samples = _sample_sources(units, n_samples, np.random.default_rng(seed))
    if not samples:
        errors.append("source-fidelity: no associations could be read from the source")
        return None
    return samples


def _lookup_variant_rows(store_path: Path, alid_to_identifier: dict[str, str]) -> dict[str, int]:
    """Resolve source ALIDs → variant_index through the Variant Index.

    ``alid_to_identifier`` maps each source ALID to the identifier to look up --
    its own canonical ALID for a same-assembly join, the lifted HG38 ALID for a
    cross-assembly one.
    """
    row_by_alid: dict[str, int] = {}
    va = VariantAxis(store_path)
    try:
        for src_alid, identifier in alid_to_identifier.items():
            rec = va.by_identifier(identifier)
            if rec is not None:
                row_by_alid[src_alid] = rec.variant_index
    finally:
        va.close()
    return row_by_alid


def _resolve_fidelity_rows(
    store_path: Path,
    samples: list[_FidelitySample],
    *,
    src_asm: str,
    store_asm: str,
    chain_file: str | Path | None,
) -> dict[str, int]:
    """Row-resolution phase: map each sampled source variant to a store row.

    Assembly-dependent: same-assembly samples join through the canonical ALID;
    cross-assembly samples first use the stored source-ALID provenance column,
    and fall back to liftover only when the store predates that column.
    """
    src_alids = {f"{c}:{p}:{a1}:{a2}" for (_a, c, p, a1, a2, _z, _s) in samples}
    if src_asm == store_asm:
        return _lookup_variant_rows(store_path, {a: a for a in src_alids})
    origin_map = _resolve_via_origin(store_path, src_alids)
    if origin_map is not None:
        return origin_map  # store carries provenance → no liftover needed
    from opengwasdb.build.liftover import build_liftover_lookup

    tuples = {(c, p, a1, a2) for (_a, c, p, a1, a2, _z, _s) in samples}
    lut = build_liftover_lookup(
        tuples,
        from_build=src_asm,
        to_build=store_asm,
        failure_threshold=1.0,
        chain_file=chain_file,
    )
    wanted = {
        f"{c}:{p}:{a1}:{a2}": hg38
        for c, p, a1, a2 in tuples
        if (hg38 := lut.get((c, p, a1, a2))) is not None
    }
    return _lookup_variant_rows(store_path, wanted)


def _match_fidelity_pairs(
    store_path: Path,
    samples: list[_FidelitySample],
    row_by_alid: dict[str, int],
    errors: list[str],
) -> list[_FidelityCell] | None:
    """Analysis-matching phase: join each sample onto a (row, column) cell.

    A sample whose Analysis the store does not hold, or whose variant row it
    does not carry, is counted rather than silently dropped; when nothing
    matches, one diagnostic names both counts so a wrong source and a wrong
    assembly stay distinguishable. Returns None after recording that error.
    """
    col_by_aid = {
        row["analysis_id"]: int(row["analysis_index"])
        for row in read_analyses(store_path / "analyses.tsv").rows
    }
    pairs: list[_FidelityCell] = []
    n_unmatched_variant = 0
    n_unmatched_analysis = 0
    for aid, c, p, a1, a2, z, se in samples:
        col = col_by_aid.get(aid)
        if col is None:
            n_unmatched_analysis += 1
            continue
        row = row_by_alid.get(f"{c}:{p}:{a1}:{a2}")
        if row is None:
            n_unmatched_variant += 1
            continue
        pairs.append((row, col, z, se, f"{c}:{p}:{a1}:{a2}"))
    if not pairs:
        errors.append(
            f"source-fidelity: none of {len(samples)} sampled source associations could be "
            f"matched to a store cell ({n_unmatched_variant} variants, {n_unmatched_analysis} "
            f"analyses unmatched) — check source_assembly / that this is the right source"
        )
        return None
    return pairs


def _fidelity_blocks(
    store: OpenGWASDBStore, pairs: list[_FidelityCell]
) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    """Read the matched cells' z/se as one bounded block plus index maps.

    Returns ``(z_block, se_block, row_index, col_index)`` where the index maps
    take each pair's variant/analysis index to its position in the block.
    """
    urows = sorted({pr[0] for pr in pairs})
    ucols = sorted({pr[1] for pr in pairs})
    ri = {r: i for i, r in enumerate(urows)}
    ci = {c: i for i, c in enumerate(ucols)}
    grid = DenseGridReader.open(
        store.arrays(mode="r"), store.manifest.encoding, source=f"Dense Store Release {store.path}"
    )
    # A cross region comes Analysis-major: one run of the rows per column.
    found = grid.inspect(Cells.cross(urows, ucols))
    shape = (len(ucols), len(urows))
    return found.z.reshape(shape).T, found.se.reshape(shape).T, ri, ci


def _compare_fidelity_cells(
    store: OpenGWASDBStore, pairs: list[_FidelityCell]
) -> tuple[int, list[str]]:
    """Comparison phase: compare each matched cell against its source value.

    The store's own codec quantises the source value before comparison, and a
    cell whose store value is missing is skipped, not compared: a store holding
    *no* finite value at any matched cell is a different failure (dropped
    associations), reported by the caller.
    """
    z_blk, se_blk, ri, ci = _fidelity_blocks(store, pairs)
    codec = StoreCodec(store.manifest.encoding)

    n_compared = 0
    mismatches: list[str] = []
    for row, col, z_src, se_src, alid in pairs:
        sz = z_blk[ri[row], ci[col]]
        sse = se_blk[ri[row], ci[col]]
        if not present_cells(np.array([sz]), np.array([sse]))[0]:
            continue
        n_compared += 1
        z_ref = float(codec.quantise_z(np.array([z_src]))[0])
        se_ref = float(np.float16(se_src))
        if not (
            np.isclose(sz, z_ref, rtol=_FIDELITY_RTOL, atol=_FIDELITY_ATOL)
            and np.isclose(sse, se_ref, rtol=_FIDELITY_RTOL, atol=_FIDELITY_ATOL)
        ):
            if len(mismatches) < 5:
                mismatches.append(
                    f"{alid}: store z={sz:.4f} se={sse:.4f} vs source z={z_src:.4f} se={se_src:.4f}"
                )
    return n_compared, mismatches


def _report_fidelity_disagreements(
    errors: list[str], pairs: list[_FidelityCell], n_compared: int, mismatches: list[str]
) -> None:
    """Reporting phase: turn the comparison outcome into a hard error.

    Disagreements name the store/source pair; matched cells where the store
    holds no finite value at all are reported as possible dropped associations.
    """
    if mismatches:
        errors.append(
            f"source-fidelity: {len(mismatches)}{'+' if len(mismatches) == 5 else ''} of "
            f"{n_compared} compared associations disagree with the source (e.g. {mismatches[0]})"
        )
    elif n_compared == 0:
        errors.append(
            f"source-fidelity: matched {len(pairs)} source associations but the store held no "
            f"finite value at any of those cells (possible dropped associations)"
        )


def _validate_source_fidelity(
    store: OpenGWASDBStore,
    source: str | Path | Sequence[str | Path],
    errors: list[str],
    *,
    n_samples: int,
    seed: int,
    source_assembly: str | None,
    chain_file: str | Path | None,
) -> None:
    """Cross-check the source's z/se against the store (issue #130).

    A thin orchestrator over five phases -- deterministic sampling, row
    resolution (assembly-dependent), Analysis matching, block comparison, and
    mismatch reporting -- each of which guards the read in front of it.
    """
    samples = _draw_fidelity_samples(source, errors, n_samples=n_samples, seed=seed)
    if samples is None:
        return
    store_asm = _normalise_assembly(store.manifest.reference_assembly)
    src_asm = _normalise_assembly(source_assembly) if source_assembly else store_asm
    row_by_alid = _resolve_fidelity_rows(
        store.path, samples, src_asm=src_asm, store_asm=store_asm, chain_file=chain_file
    )
    pairs = _match_fidelity_pairs(store.path, samples, row_by_alid, errors)
    if pairs is None:
        return
    n_compared, mismatches = _compare_fidelity_cells(store, pairs)
    _report_fidelity_disagreements(errors, pairs, n_compared, mismatches)

def default_top_hit_key(threshold: float = 5e-8) -> str:
    return threshold_key(threshold)
