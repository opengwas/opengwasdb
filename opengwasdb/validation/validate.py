"""Store validation."""

from __future__ import annotations

import numbers
import sqlite3
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from opengwasdb import top_hits
from opengwasdb.associations import Cells, DenseGridReader, State, Tally, non_canonical
from opengwasdb.associations import present as present_cells
from opengwasdb.completion.schema import COMPLETION_QUALITY_COLUMNS
from opengwasdb.encoding import (
    EAF_BASELINE,
    EAF_EXCEPTION,
    EAF_REFERENCE,
    SE_EXCEPTION,
    Z_OVERFLOW,
    DenseEafPlane,
    DenseSePlane,
    DenseZPlane,
    EafExceptionTable,
    RaggedEafPlane,
    RaggedSePlane,
    SeExceptionTable,
    StoreCodec,
    StoreEncoding,
    ZOverflowTable,
)
from opengwasdb.index.sqlite import get_metadata
from opengwasdb.layouts.dense.indexed_subsets import validate_indexed_subsets
from opengwasdb.layouts.hybrid.layout import dense_component_path, dense_to_shared_path
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.model.analyses import (
    read_analyses,
)
from opengwasdb.model.enums import (
    CompletionState,
    PrimaryStorageLayout,
)
from opengwasdb.stats import p_value_from_z
from opengwasdb.store.arrays import (
    COMPRESSOR_RECORD,
    SHARDED_COMPRESSOR_RECORD,
    ArrayRole,
    array_length,
    chunk_layout,
    inner_chunk_hint_for_path,
    inner_chunk_of,
    open_group,
    role_for_array_path,
    shard_layout,
)
from opengwasdb.store.open import (
    OpenGWASDBStore,
    UnsupportedFormatVersion,
    open_store,
)
from opengwasdb.top_hits import threshold_key, z_critical
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
from opengwasdb.validation.rules import csr as csr_rules
from opengwasdb.validation.rules import encoding as encoding_rules
from opengwasdb.validation.rules import release as release_rules
from opengwasdb.validation.rules import variants as variant_rules
from opengwasdb.validation.rules.csr import ARRAYS
from opengwasdb.validation.rules.encoding import PLAN
from opengwasdb.validation.rules.release import REQUIRED
from opengwasdb.validation.rules.variants import representative_variant_indices
from opengwasdb.validation.run import evaluate, selected
from opengwasdb.validation.sample_size import validate_ragged_n
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


def _legacy_infinite_se(target: Target) -> list[str]:
    warnings: list[str] = []
    _warn_about_infinite_ragged_se(target.store, warnings)
    return warnings


def _legacy_ragged(target: Target, errors: list[str], _warnings: list[str]) -> None:
    _validate_ragged_store(target.store, errors)


def _legacy_hybrid(target: Target, errors: list[str], _warnings: list[str]) -> None:
    _validate_hybrid_store(target.store, errors)


def _legacy_dense_completion(target: Target, errors: list[str], _warnings: list[str]) -> None:
    _dense_completion(target, errors)


def _legacy_dense(target: Target, errors: list[str], warnings: list[str]) -> None:
    _dense_values(target, errors, warnings)


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

#: The rules before source fidelity, in order: it samples what a Store holds, so it
#: runs only on a release none of them found wrong.
_BEFORE_FIDELITY: tuple[Rule, ...] = (
    Check(name=MANIFEST_RULE, kinds=Kind.RELEASE, run=lambda _target: ()),
    *release_rules.RULES,
    *variant_rules.RULES,
    *analyses_rules.TABLE_RULES,
    *csr_rules.RULES,
    *encoding_rules.RULES,
    Check(
        name="legacy.infinite_se",
        kinds=Kind.RELEASE,
        severity=Severity.WARNING,
        run=_legacy_infinite_se,
    ),
    _Legacy(
        name="legacy.ragged",
        kinds=Kind.CSR,
        when=_is(PrimaryStorageLayout.RAGGED),
        after=(ARRAYS, *PLAN),
        run=_legacy_ragged,
    ),
    _Legacy(
        name=_DENSE_COMPLETION,
        kinds=Kind.GRID,
        when=_completed,
        after=(REQUIRED,),
        run=_legacy_dense_completion,
    ),
    _Legacy(
        name="legacy.dense_values",
        kinds=Kind.GRID,
        after=(*PLAN, _DENSE_COMPLETION),
        run=_legacy_dense,
    ),
    _Legacy(
        name="legacy.dense_layout", kinds=Kind.GRID, after=(REQUIRED,), run=_legacy_dense_layout
    ),
    _Legacy(
        name="legacy.overflow",
        kinds=Kind.CSR,
        when=_is(PrimaryStorageLayout.HYBRID),
        after=(ARRAYS, *PLAN),
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


# --- an infinite se is a value: reported, never a failure (issues #337, #338) ---

#: Analyses one infinite-se warning names. The count says how much of the release is
#: affected; five ids say where to start reading it.
INFINITE_SE_ANALYSES_NAMED = 5

def _warn_about_infinite_ragged_se(store: OpenGWASDBStore, warnings: list[str]) -> None:
    """Warn about the infinite `se` cells of every Ragged component of this release.

    An SE above a `float16` plane's range is narrowed to `+inf` beside a finite Z
    (issue #338). Spec §15 makes that cell present, so the release is valid and the
    Observed-View Check compares it -- but `beta = z * se` reads as infinity, and
    nothing else in a Store says so (issue #337).

    A Dense plane is not read here: `_validate_dense_store` counts it in the band
    pass it already runs over that plane, which costs no second read. A Ragged
    Store Release's plane and a Hybrid release's Overflow have no such pass, so
    they are read windowed here -- and only when the plane is floating-point,
    because `+inf` is a value no integer plane can hold.
    """
    try:
        for group, label in _ragged_se_planes(store):
            _report_infinite_se(
                warnings, label, _csr_infinite_se_counts(group), store.analyses_path
            )
    except (OSError, ValueError, KeyError, IndexError):
        # A release whose arrays cannot be read is a failure its layout's own validator
        # reports in as many words; a warning is not the place to say so twice.
        return


def _ragged_se_planes(store: OpenGWASDBStore) -> Iterator[tuple[Any, str]]:
    """Each of a release's Ragged `se` planes, with the label a warning should name it by."""
    layout = store.manifest.primary_layout
    if layout in (PrimaryStorageLayout.HYBRID, PrimaryStorageLayout.RAGGED):
        yield store.arrays(mode="r")["ragged"], "data.zarr/ragged/se"


def _csr_infinite_se_counts(group: Any) -> np.ndarray:
    """Per Analysis, how many cells of a stored Ragged or Overflow `se` plane are infinite.

    Read in `_INDEX_WINDOW` cells, so the sequence is never resident, and not at all
    from an integer plane: `+inf` is a value only a floating-point plane can hold, so
    a residual `int8` plane has no cell for this to find, and skipping it costs
    nothing rather than a full read that cannot succeed.
    """
    offsets = np.asarray(group["offsets"][:], dtype=np.int64)
    counts = np.zeros(max(len(offsets) - 1, 0), dtype=np.int64)
    n_assoc = int(offsets[-1]) if len(offsets) else 0
    if not np.issubdtype(group["se"].dtype, np.floating):
        return counts
    for lo in range(0, n_assoc, _INDEX_WINDOW):
        window = np.asarray(group["se"][lo : min(lo + _INDEX_WINDOW, n_assoc)])
        hits = np.flatnonzero(np.isinf(window)) + lo
        if hits.size:
            # `offsets` bounds each Analysis's segment, so the last boundary at or before
            # a position names the Analysis holding it.
            np.add.at(counts, np.searchsorted(offsets, hits, side="right") - 1, 1)
    return counts


def _report_infinite_se(
    warnings: list[str],
    label: str,
    per_analysis: np.ndarray,
    analyses_path: Path | None,
) -> None:
    """Warn about one plane's infinite `se` cells, naming the count and first Analyses."""
    total = int(per_analysis.sum())
    if not total:
        return
    named = _first_infinite_analyses(per_analysis, analyses_path)
    message = (
        f"{total} cell(s) of {label} hold an infinite se: a source standard error above "
        "the plane's range is stored as +inf, so beta = z * se reads as infinity although "
        "the source value was finite (issue #338)"
    )
    if named:
        message += f". The first Analyses holding one: {', '.join(map(repr, named))}"
    warnings.append(message)


def _first_infinite_analyses(
    per_analysis: np.ndarray, analyses_path: Path | None
) -> list[str]:
    """The `analysis_id` of up to `INFINITE_SE_ANALYSES_NAMED` Analyses holding an infinite se."""
    holding = np.flatnonzero(per_analysis)[:INFINITE_SE_ANALYSES_NAMED].tolist()
    if not holding or analyses_path is None:
        return []
    ids = _analysis_ids(analyses_path)
    return [ids[i] for i in holding if i in ids]


def _analysis_ids(analyses_path: Path) -> dict[int, str]:
    """`analysis_index` -> `analysis_id`, for naming an Analysis a warning is about."""
    ids: dict[int, str] = {}
    for row in read_analyses(analyses_path).rows:
        try:
            ids[int(row.get("analysis_index", ""))] = str(row.get("analysis_id", ""))
        except ValueError:
            continue  # a malformed row is reported by the analyses.analysis_index rule
    return ids


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


def _dense_counts(target: Target) -> tuple[int, int]:
    """The grid's expected shape: the variant axis's rows by `analyses.tsv`'s."""
    store = target.store
    n_variants = target.remember("n_variants", lambda: _n_variants(store.path))
    n_analyses = target.remember("n_analyses", lambda: len(read_analyses(store.analyses_path).rows))
    return n_variants, n_analyses


def _dense_completion(target: Target, errors: list[str]) -> None:
    """A Reference-Completed grid's mask, panel flags and completion metadata."""
    store = target.store
    n_variants, n_analyses = _dense_counts(target)
    with store.index_connection() as connection:
        _validate_completion_metadata(
            target.group, connection, store.analyses_path, n_variants, n_analyses, errors
        )


def _dense_values(target: Target, errors: list[str], warnings: list[str]) -> None:
    """The streamed band pass, then the Top-Hit Indexes and rho against the planes.

    Runs after the plan and completion rules, so every plane it decodes is the
    shape and encoding they say.
    """
    store = target.store
    root = target.group
    manifest = store.manifest
    n_variants, n_analyses = _dense_counts(target)
    imputed_arr = on_panel = None
    if manifest.completion_state is CompletionState.REFERENCE_COMPLETED:
        imputed_arr = root["imputed"]
        on_panel = target.remember("on_panel", lambda: root["on_panel"][:])
    # A Hybrid release's Dense Component is a Dense release at `<store>/dense` (spec
    # §16), so a warning naming plain `data.zarr` would send its reader to the wrong one.
    label = "dense/data.zarr/se" if target.is_component else "data.zarr/se"
    _validate_dense_arrays(
        root, n_variants, n_analyses, errors, manifest.encoding,
        imputed_arr, on_panel, store.analyses_path, warnings, label,
    )
    if not errors:
        _validate_top_hits(root, errors, manifest.encoding)
    if not errors:
        _validate_rho(root, n_analyses, errors)


def _dense_layout(target: Target, errors: list[str]) -> None:
    """The recorded layout against the planes, and any Indexed Variant Subsets."""
    store = target.store
    n_variants, n_analyses = _dense_counts(target)
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
    """The Overflow's structure and values, then the cross-component invariants,
    then its Top-Hit Index -- each only once the one before it found nothing."""
    store = target.store
    root = target.group
    n_shared = _n_variants(store.path)
    n_assoc = _validate_overflow_structure(root, n_shared, errors)
    if n_assoc is None:
        return
    _validate_overflow_values(root, store.manifest.encoding, n_assoc, errors)
    if errors:
        return
    _validate_hybrid_invariants(store.path, dense_component_path(store.path), n_shared, errors)
    if errors:
        return
    data_root = store.arrays(mode="r")
    if top_hits.has_index(data_root):
        _validate_ragged_top_hits(store.path, data_root, errors)


def _validate_completion_metadata(
    root: Any,
    connection: sqlite3.Connection,
    analyses_path: Path,
    n_variants: int,
    n_analyses: int,
    errors: list[str],
) -> tuple[Any, np.ndarray | None]:
    """Validate the non-matrix completion metadata for a Reference-Completed
    Dense store and hand back the arrays the streamed band pass needs.

    The matrix-touching checks (imputed values are 0/1, imputed cells have
    finite z/se, off-panel rows are never imputed) are *not* done here — they
    are folded into ``_validate_dense_arrays``' row-band loop so the full
    z/se/imputed matrices are never resident (issue 045). This function only
    runs the cheap structural/metadata checks and returns the ``imputed`` zarr
    array (lazy) and the fully-loaded 1-D ``on_panel`` array (~n_variants bytes)
    for that loop. Returns ``(None, None)`` if the arrays are missing/malformed,
    which the caller treats as a hard error and skips the band pass.
    """
    imputed_arr, on_panel = _dense_completion_arrays(root, n_variants, n_analyses, errors)
    if errors:
        return None, None
    _validate_completion_quality_table(connection, n_analyses, errors)
    analyses_cols = set(read_analyses(analyses_path).fieldnames)
    if "completion_n_missing_total" not in analyses_cols:
        errors.append(
            "analyses.tsv is missing completion_n_missing_total for a reference-completed store"
        )

    # Only hand the arrays to the band pass if they are well-formed; a shape
    # mismatch above already recorded a hard error, so the caller skips the pass.
    if errors:
        return None, None
    return imputed_arr, on_panel


def _dense_completion_arrays(
    root: Any, n_variants: int, n_analyses: int, errors: list[str]
) -> tuple[Any, np.ndarray | None]:
    """Seam: the lazy ``imputed`` array and fully-loaded ``on_panel`` vector.

    ``imputed`` stays lazy and ``on_panel`` stays a 1-D array of ~n_variants
    bytes so the band pass that consumes them never holds the full matrices
    resident (issue 045). Missing or malformed arrays record their own errors;
    the caller returns ``(None, None)`` for them.
    """
    for name in ("imputed", "on_panel"):
        if name not in root:
            errors.append(f"reference-completed store is missing data.zarr/{name}")
    if errors:
        return None, None

    imputed_arr = root["imputed"]
    expected_shape = (n_variants, n_analyses)
    if tuple(imputed_arr.shape) != expected_shape:
        errors.append(
            f"data.zarr/imputed shape {tuple(imputed_arr.shape)} does not match {expected_shape}"
        )

    on_panel = root["on_panel"][:]
    if len(on_panel) != n_variants:
        errors.append(f"data.zarr/on_panel has {len(on_panel)} entries but expected {n_variants}")
    if not np.all((on_panel == 0) | (on_panel == 1)):
        errors.append("data.zarr/on_panel contains values other than 0 and 1")
    return imputed_arr, on_panel


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


def _match_csr_se_exceptions(
    root: Any, encoding: StoreEncoding, errors: list[str], label: str
) -> None:
    """Every `-127` in a CSR `se` plane has a table entry, and the table has no other.

    A residual plane and its side table are one artifact: a code marked as an
    exception with nothing behind it decodes to whatever the lookup returns,
    and an entry for a cell that is not marked is never read at all.
    """
    if not encoding.se.is_residual:
        return
    _overflow_positions_match(
        label,
        np.flatnonzero(np.asarray(root["se"][:]) == SE_EXCEPTION).astype(np.int64),
        SeExceptionTable.read(root),
        errors,
        what="se exception",
        table_name="se exception",
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


def _csr_parallel_length_errors(root: Any, n_assoc: int, errors: list[str]) -> None:
    """Record each parallel CSR array whose length disagrees with ``offsets``."""
    for name in ("variant_index", "z", "se"):
        if array_length(root[name]) != n_assoc:
            errors.append(
                f"data.zarr/ragged/{name} has {array_length(root[name])} entries "
                f"but offsets imply {n_assoc}"
            )


#: Cells one `variant_index` ordering window holds. The rule must never
#: materialise the array -- validation memory is already a problem (#254) -- so
#: it reads this many int32 at once (4 MB) and carries one previous value
#: across windows. Peak memory is the window, its comparison bool and the
#: per-Analysis offsets (8 bytes each), independent of the association count.
_ORDER_WINDOW = 1_000_000

#: The two ways one Analysis's `variant_index` slice can break its invariant.
_BreakKind = Literal["decrease", "repeat"]
_DECREASE: _BreakKind = "decrease"
_REPEAT: _BreakKind = "repeat"


@dataclass(frozen=True)
class _SegmentBreak:
    """The first row of one Analysis's segment that is not above its predecessor.

    One object rather than a bare offset and a separately-typed kind: the kind
    decides what the failure means, so it cannot be optional where the failure
    is not, and a caller that has a `_SegmentBreak` has both.
    """

    offset: int
    kind: _BreakKind


def _segment_first_break(
    variant_index: Any, start: int, end: int
) -> tuple[_SegmentBreak | None, int]:
    """First break in `[start, end)`, and the rows read to find it.

    Bounded: `_ORDER_WINDOW` cells at a time, the preceding cell carried across
    a window boundary. The comparison is `<=`, so a repeat is found at the same
    offset a decrease would be, and the first offending row is always the one
    reported (the kind is read off the pair, not from which test ran first).
    Returns `(break or None, entries read)` so a caller can report what was
    actually checked rather than what the offsets imply.
    """
    previous: int | None = None
    read = 0
    for lo in range(start, end, _ORDER_WINDOW):
        hi = min(lo + _ORDER_WINDOW, end)
        window = np.asarray(variant_index[lo:hi], dtype=np.int32)
        read += len(window)
        if previous is not None and int(window[0]) <= previous:
            kind: _BreakKind = _REPEAT if int(window[0]) == previous else _DECREASE
            return _SegmentBreak(lo, kind), read
        breaks = np.flatnonzero(window[1:] <= window[:-1])
        if len(breaks):
            failure = lo + int(breaks[0]) + 1
            repeated = int(window[failure - lo]) == int(window[failure - lo - 1])
            return _SegmentBreak(failure, _REPEAT if repeated else _DECREASE), read
        previous = int(window[-1])
    return None, read


def _segment_order_error(
    label: str, analysis: int, variant_index: Any, failure: _SegmentBreak
) -> str:
    """Name the Analysis whose segment broke the ordering invariant, and how.

    A repeat and a decrease are different defects and are reported differently:
    a repeat is two associations at one variant of one Analysis (a post-lift
    collision, or a duplicate source row), which no reader can resolve; a
    decrease is a segment out of genomic order, on which the variant-side
    binary search answers with a plausible, wrong row.
    """
    offset = failure.offset
    previous = int(variant_index[offset - 1])
    held = int(variant_index[offset])
    if failure.kind == _REPEAT:
        return (
            f"{label}/variant_index repeats variant {held} within Analysis "
            f"{analysis}'s segment: offsets {offset - 1} and {offset} both hold it, "
            f"so one Analysis carries two associations at one variant"
        )
    return (
        f"{label}/variant_index is not non-decreasing within Analysis "
        f"{analysis}'s segment: offset {offset} holds {held} after {previous}"
    )


def _segment_order_errors(
    root: Any, offsets: np.ndarray, n_assoc: int, errors: list[str], label: str
) -> int:
    """Require `variant_index` strictly increasing within every Analysis's segment.

    The variant-side binary search (`RaggedCSRReader.segment_positions`) is
    correct only on an ordered segment, and a repeated `variant_index` is two
    associations at one variant, which `(variant, Analysis)` resolution cannot
    represent; so the invariant is checked on every release, not only on the
    ones this build wrote. The read is bounded: `_ORDER_WINDOW` cells at a
    time, the preceding cell carried across a window boundary, the comparison
    reset at each Analysis boundary. Returns the number of entries actually
    read, so an evidence runner can report a counted total and not an
    offset-implied one.
    """
    if array_length(root["variant_index"]) != n_assoc:
        return 0  # the parallel-length rule reports this
    variant_index = root["variant_index"]
    checked = 0
    for analysis in range(len(offsets) - 1):
        start, end = int(offsets[analysis]), int(offsets[analysis + 1])
        if end <= start:
            continue
        failure, read = _segment_first_break(variant_index, start, end)
        checked += read
        if failure is not None:
            errors.append(_segment_order_error(label, analysis, variant_index, failure))
            return checked
    return checked


def _validate_ragged_csr_structure(root: Any, errors: list[str]) -> tuple[int, int]:
    """Seam: the CSR's parallel arrays agree in length, and each segment is ordered.

    Returns ``(n_assoc, n_analyses)``. A length disagreement against ``offsets``
    records an error but does not stop the pipeline -- the value seam reports
    what the arrays decode to either way.
    """
    offsets = np.asarray(root["offsets"][:], dtype=np.int64)
    n_assoc = int(offsets[-1])
    _csr_parallel_length_errors(root, n_assoc, errors)
    _segment_order_errors(root, offsets, n_assoc, errors, "data.zarr/ragged")
    return n_assoc, len(offsets) - 1


def _validate_ragged_csr_values(
    root: Any, encoding: StoreEncoding, n_assoc: int, errors: list[str]
) -> bool:
    """Seam: every value the CSR holds decodes to what its plane claims.

    Decoded, not raw: a fixed-point plane's bytes are codes, not values, and
    a residual `se` plane can only be checked against its side tables and
    coefficients once it decodes (issue #118). Returns False when `se` cannot
    be decoded -- the completion and Top-Hit seams read decoded se, so a
    store that fails here is not taken further.
    """
    raw_z = np.asarray(root["z"][:])
    if encoding.z.is_fixed_point:
        _overflow_positions_match(
            "data.zarr/ragged/z",
            np.flatnonzero(raw_z == Z_OVERFLOW).astype(np.int64),
            ZOverflowTable.read(root),
            errors,
        )
    se_vals, se_error = _decoded_csr_se(root, encoding, n_assoc, "data.zarr/ragged/se")
    if se_error is not None:
        errors.append(se_error)
        return False
    _match_csr_se_exceptions(root, encoding, errors, "data.zarr/ragged/se")
    if np.any(np.isfinite(se_vals) & (se_vals < 0)):
        errors.append("se contains negative finite values")
    errors.extend(_paired_marker_errors(raw_z, se_vals, encoding, ""))
    # `eaf` is optional (ADR 0036); when present it is a fourth parallel
    # CSR array and must line up with the other three.
    if "eaf" in root:
        if array_length(root["eaf"]) != n_assoc:
            errors.append(
                f"data.zarr/ragged/eaf has {array_length(root['eaf'])} entries "
                f"but offsets imply {n_assoc}"
            )
        else:
            _validate_ragged_eaf_values(root, encoding, n_assoc, errors)
    return True


def _validate_ragged_analyses(
    store: OpenGWASDBStore, n_analyses_csr: int, errors: list[str]
) -> None:
    """Seam: analyses.tsv and index.sqlite describe the Analyses the CSR does.

    `analyses.tsv` is the sole source of truth for Analytical Metadata (ADR
    0030, ADR 0034), so its row count must equal the number of Analyses the
    CSR's offsets bound, and a leftover SQLite `analyses` table is a failure
    rather than a harmless relic (ADR 0034, issue #72).
    """
    n_analyses_tsv = len(read_analyses(store.analyses_path).rows)
    if n_analyses_tsv != n_analyses_csr:
        errors.append(
            f"analyses.tsv has {n_analyses_tsv} rows but "
            f"zarr CSR offsets imply {n_analyses_csr} analyses"
        )


def _validate_ragged_downstream_seams(
    store: OpenGWASDBStore,
    ragged_path: Path,
    n_assoc: int,
    n_analyses_csr: int,
    errors: list[str],
) -> None:
    """Seam: Reference-Completion state, the variant index, then Top-Hit Indexes.

    Completion is validated whenever the release declares it, for its own
    sake. The variant-centric index and the Top-Hit Indexes are only
    cross-checked against the CSR once every earlier seam passed -- both read
    the CSR and would otherwise "confirm" a store the structural and value
    seams already found broken.
    """
    data_root = store.arrays(mode="r")
    if store.manifest.completion_state is CompletionState.REFERENCE_COMPLETED:
        _validate_ragged_completion(ragged_path, store, n_assoc, n_analyses_csr, errors)
    if not errors:
        axis = VariantAxis(store.path)
        try:
            n_variant_axis = axis.n_variants
        finally:
            axis.close()
        _validate_variant_index(
            store,
            ragged_path,
            n_assoc,
            errors,
            component_label="data.zarr/ragged",
            n_axis=n_variant_axis,
        )
    if not errors and top_hits.has_index(data_root):
        _validate_ragged_top_hits(store.path, data_root, errors)


#: The one source of the index group name, duplicated from the builder module
#: to keep validation importable without the build stack.
_BY_VARIANT_GROUP = "by_variant"

#: Array cells scanned in one window (issue #254).
_INDEX_WINDOW = 1_000_000


def _validate_variant_index_presence(
    store: OpenGWASDBStore, root: Any, errors: list[str], label: str
) -> Any | None:
    """Check the group and its manifest provenance block agree; return the group."""
    group = root[_BY_VARIANT_GROUP] if _BY_VARIANT_GROUP in root else None
    block = store.manifest.provenance.get("ragged", {}).get("by_variant")
    if group is None:
        if block is not None:
            errors.append(
                f"{label}: the manifest records provenance.ragged.by_variant but the release "
                f"carries no {_BY_VARIANT_GROUP} group; the index it describes is absent"
            )
        else:
            errors.append(
                f"{label}: carries no {_BY_VARIANT_GROUP} group; a Ragged component must "
                f"carry the variant-centric index (ADR 0060)"
            )
        return None
    if block is None:
        errors.append(
            f"{label}/{_BY_VARIANT_GROUP}: the release carries a variant index but its "
            "manifest records no provenance.ragged.by_variant block"
        )
    return group


def _validate_variant_index_structure(
    index: Any,
    n_axis: int,
    n_analyses: int,
    n_assoc: int,
    errors: list[str],
    label: str,
) -> np.ndarray | None:
    """Shape/ordering checks for the index, returning its offsets or `None`."""
    offsets = _validate_index_offsets(index, n_axis, n_assoc, errors, label)
    if offsets is None:
        return None
    if not _validate_index_analyses(index, n_analyses, n_assoc, errors, label):
        return None
    return offsets


def _validate_index_offsets(
    index: Any, n_axis: int, n_assoc: int, errors: list[str], label: str
) -> np.ndarray | None:
    """The offsets array must span `[0, n_assoc)` over `n_axis` variants."""
    offsets = np.asarray(index["offsets"][:], dtype=np.int64)
    if len(offsets) != n_axis + 1:
        errors.append(
            f"{label}/{_BY_VARIANT_GROUP}/offsets has {len(offsets)} entries but the variant "
            f"axis is {n_axis} (expected {n_axis + 1})"
        )
        return None
    if int(offsets[0]) != 0 or int(offsets[-1]) != n_assoc:
        errors.append(
            f"{label}/{_BY_VARIANT_GROUP}/offsets does not span [0, {n_assoc}) "
            f"(starts {int(offsets[0])}, ends {int(offsets[-1])})"
        )
        return None
    if np.any(np.diff(offsets) < 0):
        errors.append(f"{label}/{_BY_VARIANT_GROUP}/offsets is not non-decreasing")
        return None
    for name in ("z", "se", "analysis_index"):
        if array_length(index[name]) != n_assoc:
            errors.append(
                f"{label}/{_BY_VARIANT_GROUP}/{name} has {array_length(index[name])} "
                f"entries but the component has {n_assoc} associations"
            )
            return None
    return offsets


def _validate_index_analyses(
    index: Any, n_analyses: int, n_assoc: int, errors: list[str], label: str
) -> bool:
    """Every `analysis_index` must name a real Analysis, checked in windows.

    Windowed so a 3.09 B-row index does not read its 12.3 GB `analysis_index`
    twice as int64 (review round 1, finding 3).
    """
    if n_assoc == 0:
        return True
    lowest = np.iinfo(np.int64).max
    highest = np.iinfo(np.int64).min
    for lo in range(0, n_assoc, _INDEX_WINDOW):
        hi = min(lo + _INDEX_WINDOW, n_assoc)
        window = np.asarray(index["analysis_index"][lo:hi], dtype=np.int64)
        lowest = min(lowest, int(window.min()))
        highest = max(highest, int(window.max()))
    if lowest < 0 or highest >= n_analyses:
        errors.append(
            f"{label}/{_BY_VARIANT_GROUP}/analysis_index is outside the component's "
            f"{n_analyses} Analyses"
        )
        return False
    return True


def _validate_variant_index_members(
    index: Any, root: Any, errors: list[str], label: str
) -> None:
    """`eaf`, `imputed` and `n` must be present on exactly both sides or neither."""
    for name in ("eaf", "imputed", "n"):
        if (name in index) != (name in root):
            errors.append(
                f"{label}/{_BY_VARIANT_GROUP}/{name} is present on one side only; the "
                "component and its variant index must carry the same members"
            )


def _code_differs(a: Any, b: Any) -> bool:
    """True when stored codes a and b differ, treating NaN == NaN."""
    if a == b:
        return False
    try:
        return not (np.isnan(a) and np.isnan(b))
    except (TypeError, ValueError):
        return True


def _sample_non_empty_variants(by_offsets: np.ndarray, sample_variants: int) -> np.ndarray:
    """Deterministic selection of non-empty variant indices for sampled validation."""
    widths = np.diff(by_offsets)
    non_empty = np.where(widths > 0)[0]
    if len(non_empty) <= sample_variants:
        return non_empty
    rng = np.random.default_rng(0)
    chosen = rng.choice(non_empty, size=sample_variants - 2, replace=False)
    return np.unique(np.concatenate([[non_empty[0], non_empty[-1]], chosen]))


def _check_z_overflow(
    root: Any, index: Any, b_z: Any, tag: str, s_idx: int, src_idx: int
) -> str | None:
    if b_z != Z_OVERFLOW:
        return None
    idx_v = ZOverflowTable.open(index).lookup(np.array([s_idx]))
    src_v = ZOverflowTable.open(root).lookup(np.array([src_idx]))
    return f"{tag} z overflow value != Analysis-sorted table" if idx_v != src_v else None


def _check_eaf_exception(
    root: Any, index: Any, b_eaf: Any, tag: str, s_idx: int, src_idx: int
) -> str | None:
    if b_eaf != EAF_EXCEPTION:
        return None
    idx_v = EafExceptionTable.open(index).lookup(np.array([s_idx]))
    src_v = EafExceptionTable.open(root).lookup(np.array([src_idx]))
    return f"{tag} eaf exception value != Analysis-sorted table" if idx_v != src_v else None


def _check_se_exception(
    root: Any, index: Any, b_se: Any, tag: str, s_idx: int, src_idx: int
) -> str | None:
    if b_se != SE_EXCEPTION:
        return None
    idx_v = SeExceptionTable.open(index).lookup(np.array([s_idx]))
    src_v = SeExceptionTable.open(root).lookup(np.array([src_idx]))
    return f"{tag} se exception value != Analysis-sorted table" if idx_v != src_v else None


def _check_cell_exceptions(
    root: Any,
    index: Any,
    encoding: StoreEncoding,
    b_z: Any,
    b_se: Any,
    b_eaf: Any,
    tag: str,
    s_idx: int,
    src_idx: int,
) -> str | None:
    """Validate exception-table values at an exception cell; returns an error string or None."""
    if encoding.z.is_fixed_point:
        err = _check_z_overflow(root, index, b_z, tag, s_idx, src_idx)
        if err is not None:
            return err
    if encoding.eaf.is_residual and b_eaf is not None:
        err = _check_eaf_exception(root, index, b_eaf, tag, s_idx, src_idx)
        if err is not None:
            return err
    if encoding.se.is_residual:
        return _check_se_exception(root, index, b_se, tag, s_idx, src_idx)
    return None


def _check_cell_codes(
    root: Any,
    b_z: Any,
    b_se: Any,
    b_eaf: Any,
    b_imp: Any,
    tag: str,
    src_idx: int,
    b_n: Any = None,
) -> str | None:
    """Compare raw stored codes for equality, returning an error string on mismatch."""
    if _code_differs(b_z, root["z"][src_idx]):
        return f"{tag} z code {b_z!r} != source {root['z'][src_idx]!r}"
    if _code_differs(b_se, root["se"][src_idx]):
        return f"{tag} se code {b_se!r} != source {root['se'][src_idx]!r}"
    for name, code in (("eaf", b_eaf), ("n", b_n)):
        if code is not None and _code_differs(code, root[name][src_idx]):
            return f"{tag} {name} code {code!r} != source {root[name][src_idx]!r}"
    if b_imp is not None and b_imp != root["imputed"][src_idx]:
        return f"{tag} imputed {b_imp!r} != source {root['imputed'][src_idx]!r}"
    return None


def _find_source_position(
    src_vi: np.ndarray, v: int, a: int, label: str
) -> tuple[int, str | None]:
    pos = int(np.searchsorted(src_vi, v))
    if pos >= len(src_vi) or src_vi[pos] != v:
        return -1, (
            f"{label}/{_BY_VARIANT_GROUP}: variant {v} in Analysis {a} is missing "
            "from the Analysis-sorted planes"
        )
    return pos, None


def _validate_one_cell(
    root: Any,
    index: Any,
    offsets: np.ndarray,
    encoding: StoreEncoding,
    label: str,
    v: int,
    a: int,
    codes: tuple[Any, Any, Any, Any, Any],
    ordinal: int,
) -> str | None:
    cs, ce = int(offsets[a]), int(offsets[a + 1])
    src_vi = np.asarray(root["variant_index"][cs:ce], dtype=np.int32)
    pos, err = _find_source_position(src_vi, v, a, label)
    if err is not None:
        return err
    src_pos = cs + pos
    tag = f"{label}/{_BY_VARIANT_GROUP}: (variant={v}, analysis={a})"
    b_z, b_se, b_eaf, b_imp, b_n = codes
    err = _check_cell_codes(root, b_z, b_se, b_eaf, b_imp, tag, src_pos, b_n)
    if err is not None:
        return err
    return _check_cell_exceptions(root, index, encoding, b_z, b_se, b_eaf, tag, ordinal, src_pos)


def _load_optional_columns(
    root: Any, index: Any, s: int, e: int
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """`eaf`, `imputed` and `n` for index rows `[s, e)`, each `None` unless both sides hold it."""

    def column(name: str) -> np.ndarray | None:
        return np.asarray(index[name][s:e]) if name in index and name in root else None

    return column("eaf"), column("imputed"), column("n")


def _cell_codes_tuple(
    b_z: np.ndarray,
    b_se: np.ndarray,
    b_eaf: np.ndarray | None,
    b_imp: np.ndarray | None,
    i: int,
    b_n: np.ndarray | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    eaf_val = b_eaf[i] if b_eaf is not None else None
    imp_val = b_imp[i] if b_imp is not None else None
    n_val = b_n[i] if b_n is not None else None
    return (b_z[i], b_se[i], eaf_val, imp_val, n_val)


def _validate_variant_cells(
    root: Any,
    index: Any,
    offsets: np.ndarray,
    by_offsets: np.ndarray,
    encoding: StoreEncoding,
    label: str,
    v: int,
) -> str | None:
    """Compare all index cells for one variant against the source planes."""
    s, e = int(by_offsets[v]), int(by_offsets[v + 1])
    b_analyses = np.asarray(index["analysis_index"][s:e], dtype=np.int32)
    b_z, b_se = np.asarray(index["z"][s:e]), np.asarray(index["se"][s:e])
    b_eaf, b_imp, b_n = _load_optional_columns(root, index, s, e)
    for i, a_val in enumerate(b_analyses):
        codes = _cell_codes_tuple(b_z, b_se, b_eaf, b_imp, i, b_n)
        err = _validate_one_cell(
            root, index, offsets, encoding, label, v, int(a_val), codes, s + i
        )
        if err is not None:
            return err
    return None


def _validate_variant_index_sample(
    root: Any,
    index: Any,
    offsets: np.ndarray,
    by_offsets: np.ndarray,
    encoding: StoreEncoding,
    n_assoc: int,
    errors: list[str],
    label: str,
    *,
    sample_variants: int = 100,
) -> None:
    """Sampled cell comparison: cells in `by_variant` match the Analysis-sorted planes."""
    if n_assoc == 0:
        return
    sample = _sample_non_empty_variants(by_offsets, sample_variants)
    for v in sample:
        err = _validate_variant_cells(root, index, offsets, by_offsets, encoding, label, int(v))
        if err is not None:
            errors.append(err)
            return


def _validate_variant_index(
    store: OpenGWASDBStore,
    ragged_path: Path,
    n_assoc: int,
    errors: list[str],
    *,
    component_label: str,
    n_axis: int,
) -> None:
    """Seam: the variant-centric index agrees with the Analysis-sorted planes.

    Presence, shape, per-variant counts, within-variant ordering, and a
    sampled cell comparison against the Analysis-sorted side.
    """
    root = open_group(ragged_path)
    index = _validate_variant_index_presence(store, root, errors, component_label)
    if index is None:
        return
    offsets = np.asarray(root["offsets"][:], dtype=np.int64)
    by_offsets = _validate_variant_index_structure(
        index, n_axis, len(offsets) - 1, n_assoc, errors, component_label
    )
    if by_offsets is None:
        return
    _validate_variant_index_block(store, n_axis, n_assoc, errors, component_label)
    _validate_variant_index_members(index, root, errors, component_label)
    _validate_variant_index_layout(index, errors, component_label)
    _validate_variant_index_counts(root, by_offsets, n_axis, n_assoc, errors, component_label)
    _validate_variant_index_ordering(index, by_offsets, n_assoc, errors, component_label)
    if errors:
        return
    _validate_variant_index_sample(
        root, index, offsets, by_offsets, store.manifest.encoding, n_assoc, errors, component_label
    )


def _validate_variant_index_block(
    store: OpenGWASDBStore,
    n_axis: int,
    n_assoc: int,
    errors: list[str],
    label: str,
) -> None:
    """The recorded provenance block must describe the index that is present."""
    block = store.manifest.provenance.get("ragged", {}).get("by_variant")
    if block is None:
        return
    if int(block.get("n_axis", -1)) != n_axis or int(block.get("n_rows", -1)) != n_assoc:
        errors.append(
            f"{label}: provenance.ragged.by_variant records "
            f"n_axis={block.get('n_axis')!r} n_rows={block.get('n_rows')!r} but the index "
            f"holds {n_axis} variants over {n_assoc} rows"
        )


def _validate_variant_index_layout(index: Any, errors: list[str], label: str) -> None:
    """Every `by_variant/` array must carry the role's decided inner chunk and shard.

    ADR 0060: the index adds no role and no shard policy, and the offsets array's
    1,000-element inner-chunk hint is part of its recorded layout.  A
    conversion that reproduced a different chunk from a build would be a layout
    divergence a reader cannot see, so it is checked here.
    """
    for name in sorted(index.array_keys()):
        _validate_one_index_array(index, name, errors, label)


def _validate_one_index_array(
    index: Any, name: str, errors: list[str], label: str
) -> None:
    """One `by_variant/` array's inner chunk and shard against its role policy."""
    path = f"ragged/by_variant/{name}"
    role = role_for_array_path(path)
    if role is None:
        errors.append(f"{label}/{path}: no ArrayRole is registered for this path")
        return
    array = index[name]
    shape = tuple(int(size) for size in array.shape)
    expected = chunk_layout(role, shape, hint=inner_chunk_hint_for_path(path))
    if tuple(inner_chunk_of(array)) != expected:
        errors.append(
            f"{label}/{path}: inner chunk {tuple(inner_chunk_of(array))} != the "
            f"role policy's {expected}"
        )
    expected_shard = shard_layout(role, shape, inner_chunk=expected)
    if array.shards is not None and tuple(int(s) for s in array.shards) != expected_shard:
        errors.append(
            f"{label}/{path}: shard {tuple(int(s) for s in array.shards)} != the "
            f"role policy's {expected_shard}"
        )


def _validate_variant_index_counts(
    root: Any,
    by_offsets: np.ndarray,
    n_axis: int,
    n_assoc: int,
    errors: list[str],
    label: str,
) -> None:
    """Every variant must hold exactly as many index rows as the source.

    The histogram is accumulated with `np.add.at` over each window's unique
    values, not `np.bincount(minlength=n_axis)`: the latter allocates 1.31 GB
    per window (3,085 windows at OGS-00011) where the former allocates one
    window's uniques (review round 1, finding 3).
    """
    counts = np.zeros(n_axis, dtype=np.int64)
    for lo in range(0, n_assoc, _INDEX_WINDOW):
        hi = min(lo + _INDEX_WINDOW, n_assoc)
        window = np.asarray(root["variant_index"][lo:hi], dtype=np.int64)
        values, multiplicity = np.unique(window, return_counts=True)
        np.add.at(counts, values, multiplicity)
    if not np.array_equal(np.diff(by_offsets), counts):
        differing = int(np.argmax(np.diff(by_offsets) != counts))
        errors.append(
            f"{label}/{_BY_VARIANT_GROUP}/offsets: variant {differing} holds "
            f"{int(by_offsets[differing + 1] - by_offsets[differing])} rows but the "
            f"Analysis-sorted planes hold {int(counts[differing])}"
        )


def _validate_variant_index_ordering(
    index: Any,
    by_offsets: np.ndarray,
    n_assoc: int,
    errors: list[str],
    label: str,
) -> None:
    """Within each variant, the index's analyses must be non-decreasing.

    Bounded by the same window the digest uses; the comparison resets at each
    variant boundary carried by `by_offsets` and the window's array is read one
    window at a time (review round 1, finding 3).
    """
    for lo in range(0, n_assoc, _INDEX_WINDOW):
        hi = min(lo + _INDEX_WINDOW, n_assoc)
        window = np.asarray(index["analysis_index"][lo:hi], dtype=np.int64)
        if len(window) < 2:
            continue
        variant = np.searchsorted(by_offsets, np.arange(lo, hi, dtype=np.int64), side="right") - 1
        same_variant = variant[1:] == variant[:-1]
        if np.any(same_variant & (window[1:] < window[:-1])):
            errors.append(
                f"{label}/{_BY_VARIANT_GROUP}/analysis_index is not non-decreasing within "
                "a variant's block"
            )
            return


def _validate_ragged_store(store: OpenGWASDBStore, errors: list[str]) -> None:
    """Validate a Ragged Store Release (spec §11, §15, §17) in fail-safe seams.

    ``_validate_ragged_store`` is a thin orchestrator over the seams that follow
    the envelope, the CSR's arrays and its plan, which are rules of their own:
    the CSR structure, the encoded values, the Analysis metadata, then
    completion and the Top-Hit Indexes (issue #130). Each seam
    stops the pipeline when a later seam would read what it guards, so a
    malformed release is reported where it first goes wrong rather than as a
    crash in a later read. An exception is reported by the rule that runs this,
    after whatever was found before it.
    """
    ragged_path = store.data_path / "ragged"
    root = open_group(ragged_path)
    n_assoc, n_analyses_csr = _validate_ragged_csr_structure(root, errors)
    if not _validate_ragged_csr_values(root, store.manifest.encoding, n_assoc, errors):
        return
    _validate_ragged_analyses(store, n_analyses_csr, errors)
    validate_ragged_n(store, root, store.manifest.encoding, errors)
    _validate_ragged_downstream_seams(store, ragged_path, n_assoc, n_analyses_csr, errors)


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
    component, completed or not (`_paired_marker_errors`).
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


def _paired_marker_errors(
    raw_z: np.ndarray, se_vals: np.ndarray, encoding: StoreEncoding, label: str
) -> list[str]:
    """A Ragged component's z and se markers, which must agree row for row and be
    canonical (spec §15).

    Applied to every association sequence -- an Observed-Only Ragged release
    and a Hybrid's Ragged Overflow as well as a completed one. An infinite SE
    is present, so a missing z beside one fails.
    """
    z_missing = StoreCodec(encoding).missing_mask(raw_z)
    se_missing = np.isnan(se_vals)
    odd = non_canonical(raw_z) | non_canonical(se_vals, se_missing)
    tally = Tally.of(z_missing, se_missing, odd_marker=odd)
    found = []
    if tally.z_only_missing:
        found.append(f"{label}missing z rows have a present se value (inconsistent missingness)")
    if tally.se_only_missing:
        found.append(f"{label}z and se missingness is inconsistent")
    if tally.count(State.NON_CANONICAL):
        found.append(_non_canonical_message(label))
    return found


def _non_canonical_message(label: str) -> str:
    return f"{label}z or se holds a non-canonical missing marker (a NaN other than np.nan)"


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


_TOP_HIT_SAMPLE_SIZE = 1000  # cross-validate this many sampled entries against the CSR


def _sample_csr_positions(
    key: str,
    sample: np.ndarray,
    vis: np.ndarray,
    ais: np.ndarray,
    zs: np.ndarray,
    threshold: float,
    offsets: np.ndarray,
    vi_all: np.ndarray,
    z_all: np.ndarray,
    errors: list[str],
) -> list[int]:
    """Locate each sampled hit in the CSR and check what it claims about it.

    Returns the CSR positions found, which is short of `sample` exactly when a
    check failed -- the caller uses that to skip the frequency comparison
    rather than compare against positions it never resolved.
    """
    positions: list[int] = []
    for idx in sample.tolist():
        ai, vi = int(ais[idx]), int(vis[idx])
        start, end = int(offsets[ai]), int(offsets[ai + 1])
        pos = start + int(np.searchsorted(vi_all[start:end], vi))
        if pos >= end or int(vi_all[pos]) != vi:
            errors.append(f"top-hit index {key} references missing association")
            return positions
        stored_z = float(z_all[pos])
        if not np.isclose(stored_z, float(zs[idx]), rtol=1e-3, atol=1e-3):
            errors.append(f"top-hit index {key} z value inconsistent with CSR")
            return positions
        if p_value_from_z(stored_z) > threshold:
            errors.append(f"top-hit index {key} contains association above threshold")
            return positions
        positions.append(pos)
    return positions


def _missing_top_hit_arrays(key: str, group: Any, errors: list[str]) -> bool:
    """Record what a tier lacks; True when it lacks anything."""
    missing = top_hits.missing_arrays(group)
    if missing:
        errors.append(f"top-hit index {key} is missing {', '.join(missing)}")
    return bool(missing)


def _validate_ragged_top_hits(
    store_path: Path,
    data_root: Any,
    errors: list[str],
) -> None:
    """Validate the top-hit index groups in a ragged zarr store.

    Structural checks (lengths, sort order) are exhaustive.
    CSR cross-validation is sampled to keep validation O(1) for large stores.
    """
    top = top_hits.index_group(data_root)
    csr = RaggedCSRReader(store_path)
    offsets = csr.offsets
    vi_all = csr.variant_index(0, csr.sequence_length)
    z_all = csr.z_all()
    n_analyses = len(offsets) - 1

    for key in top:
        group = top[key]
        if _missing_top_hit_arrays(key, group, errors):
            continue
        threshold = top_hits.tier_threshold(key, group, missing=0.0)
        arrays = top_hits.load_tier_arrays(group)
        problem = top_hits.structural_problem(
            arrays, group, n_analyses, expects_imputed=csr.has_imputed
        )
        if problem is not None:
            errors.append(f"top-hit index {key} {problem}")
            continue
        vis = arrays["rows"].astype(np.int32)
        ais = arrays["cols"].astype(np.int32)
        zs = arrays["z_values"]
        ses = arrays["se_values"]
        abs_zs = arrays["abs_z"]
        eaf = arrays["eaf_values"]

        # Exhaustive: all abs_z must match |z|
        if not np.allclose(abs_zs, np.abs(zs), rtol=1e-3, atol=1e-3):
            errors.append(f"top-hit index {key} abs_z inconsistent with z")

        # Sampled cross-validation against the CSR
        n = len(vis)
        if n == 0:
            continue
        rng = np.random.default_rng(0)
        sample = rng.choice(n, size=min(_TOP_HIT_SAMPLE_SIZE, n), replace=False)
        sample_positions = _sample_csr_positions(
            key, sample, vis, ais, zs, threshold, offsets, vi_all, z_all, errors
        )
        if len(sample_positions) == len(sample):
            expected_se = csr.se_at(np.asarray(sample_positions, dtype=np.int64))
            # Same tolerance, same reason, as the Dense band check above.
            if not np.allclose(ses[sample], expected_se, rtol=1e-6, atol=0.0, equal_nan=True):
                errors.append(f"top-hit index {key} se value inconsistent with decoded CSR se")
        if eaf is not None and len(sample_positions) == len(sample):
            expected_eaf = csr.eaf_at(np.asarray(sample_positions, dtype=np.int64))
            if not np.allclose(eaf[sample], expected_eaf, equal_nan=True):
                errors.append(f"top-hit index {key} eaf value inconsistent with CSR")


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


def _validate_overflow_structure(root: Any, n_shared: int, errors: list[str]) -> int | None:
    """Seam: the Overflow CSR's shape, shared-index bounds, and observed-only status.

    Returns the association count when the structural guards pass; ``None``
    stops the pipeline. ``variant_index`` bounds are checked against the
    shared variant table the Overflow's indices address.
    """
    before = len(errors)
    n_assoc = int(root["offsets"][:][-1])
    _csr_parallel_length_errors(root, n_assoc, errors)
    _segment_order_errors(
        root, np.asarray(root["offsets"][:], dtype=np.int64), n_assoc, errors,
        "data.zarr/ragged",
    )
    if "imputed" in root:
        errors.append(
            "Ragged Overflow has an imputed array — the overflow is off-panel and "
            "must never be imputed (always observed)"
        )
    if len(errors) > before:
        return None
    vi = root["variant_index"][:]
    if n_assoc and (int(vi.min()) < 0 or int(vi.max()) >= n_shared):
        errors.append(
            f"Ragged Overflow variant_index falls outside the shared variant table [0, {n_shared})"
        )
    return n_assoc


def _validate_overflow_values(
    root: Any, encoding: StoreEncoding, n_assoc: int, errors: list[str]
) -> None:
    """Seam: what the Overflow's encoded values decode to.

    Decoded, not raw: a fixed-point `z`'s bytes are codes, and a residual `se`
    plane can only be checked against its side tables and coefficients once
    it decodes (issue #118). Runs even when the shared-index bounds check
    above failed, so every defect reports rather than the first one masking
    the rest.
    """
    se_vals, se_error = _decoded_csr_se(root, encoding, n_assoc, "Ragged Overflow se")
    if se_error is not None:
        errors.append(se_error)
        return
    _match_csr_se_exceptions(root, encoding, errors, "Ragged Overflow se")
    if np.any(np.isfinite(se_vals) & (se_vals < 0)):
        errors.append("Ragged Overflow se contains negative finite values")
    raw_z = np.asarray(root["z"][:])
    errors.extend(
        _paired_marker_errors(raw_z, se_vals, encoding, "Ragged Overflow ")
    )
    if encoding.z.is_fixed_point:
        _overflow_positions_match(
            "Ragged Overflow z",
            np.flatnonzero(raw_z == Z_OVERFLOW).astype(np.int64),
            ZOverflowTable.read(root),
            errors,
        )


def _validate_hybrid_invariants(
    store_path: Path, dense_dir: Path, n_shared: int, errors: list[str]
) -> None:
    """Coverage, disjoint-partition and variant-index invariants over the shared space.

    The Overflow's variant-centric index (ADR 0060) is checked last, after the
    partition invariant, so a corruption that both breaks disjointness and
    stales the index reports the disjointness it broke rather than only the
    index's disagreement.
    """
    dense_to_shared = np.load(dense_to_shared_path(store_path))
    n_panel = len(dense_to_shared)

    # dense_to_shared must be a strictly-increasing set of valid shared rows.
    if n_panel and (int(dense_to_shared.min()) < 0 or int(dense_to_shared.max()) >= n_shared):
        errors.append("dense_to_shared references rows outside the shared variant table")
        return
    if n_panel != len(np.unique(dense_to_shared)):
        errors.append("dense_to_shared maps two Dense Component rows to one shared row")
        return

    _validate_overflow_disjoint(store_path, n_shared, dense_to_shared, errors)
    _validate_dense_coverage(store_path, dense_dir, dense_to_shared, n_panel, errors)
    _validate_overflow_index(store_path, n_shared, errors)


def _validate_overflow_disjoint(
    store_path: Path, n_shared: int, dense_to_shared: np.ndarray, errors: list[str]
) -> None:
    """No overflow variant may also be a Dense Component (on-panel) row."""
    ragged_root = open_group(store_path / "data.zarr" / "ragged")
    overflow_vi = np.unique(ragged_root["variant_index"][:])
    on_panel = np.zeros(n_shared, dtype=bool)
    on_panel[dense_to_shared] = True
    if len(overflow_vi) and np.any(on_panel[overflow_vi]):
        errors.append(
            "disjoint-partition violated: an overflow association references an "
            "on-panel (Dense Component) variant"
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


def _validate_overflow_index(store_path: Path, n_shared: int, errors: list[str]) -> None:
    """Validate the Overflow's variant-centric index (ADR 0060)."""
    ragged_path = store_path / "data.zarr" / "ragged"
    if errors:
        return
    root = open_group(ragged_path)
    offsets = np.asarray(root["offsets"][:], dtype=np.int64)
    _validate_variant_index(
        open_store(store_path),
        ragged_path,
        int(offsets[-1]),
        errors,
        component_label="data.zarr/ragged",
        n_axis=n_shared,
    )


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


# Dense validation decodes several raw, float32/float64 and boolean views of a
# cell at once. OGS-00009/10 profiling measured at most 59 bytes per cell in
# the Dense pass and 47 in the Top-Hit pass; the 96-byte allowance leaves 63%
# headroom inside this one-GiB decoded working-set budget. Top-Hit Index arrays
# retain their compact contract dtypes separately. The row cap preserves the
# old call size for a narrow store, while the byte budget makes a wide Store's
# band shrink instead of scaling its memory with the number of Analyses
# (issue #254).
_VALIDATE_BAND_ROWS = 250_000
_DENSE_VALIDATE_WORKING_BYTES = 1024**3
_DENSE_VALIDATE_BYTES_PER_CELL = 96


def _dense_validation_band_rows(n_analyses: int, row_chunk: int) -> int:
    """Rows whose decoded working set fits the Dense validation byte budget.

    Whole inner chunks avoid decoding a physical chunk into two consecutive
    bands. A single row is the irreducible minimum when an exceptionally wide
    store exceeds the budget by itself.
    """
    if n_analyses <= 0:
        return _VALIDATE_BAND_ROWS
    budget_rows = max(
        1,
        _DENSE_VALIDATE_WORKING_BYTES
        // (_DENSE_VALIDATE_BYTES_PER_CELL * n_analyses),
    )
    rows = min(_VALIDATE_BAND_ROWS, budget_rows)
    if row_chunk > 0 and rows >= row_chunk:
        rows = max(row_chunk, rows // row_chunk * row_chunk)
    return rows


def _dense_validation_bands(
    n_variants: int, n_analyses: int, row_chunk: int
) -> Iterator[tuple[int, int]]:
    """Every Dense row band, exhaustively, under the decoded-byte budget."""
    rows = _dense_validation_band_rows(n_analyses, row_chunk)
    for r0 in range(0, n_variants, rows):
        yield r0, min(r0 + rows, n_variants)


def _dense_array_validation_bands(
    array: Any, n_variants: int, n_analyses: int
) -> Iterator[tuple[int, int]]:
    """Byte-bounded bands aligned to a Dense plane's physical read unit."""
    return _dense_validation_bands(n_variants, n_analyses, inner_chunk_of(array)[0])


def _validate_ragged_eaf_values(
    root: Any, encoding: StoreEncoding, n_assoc: int, errors: list[str]
) -> None:
    """Decoded frequencies are frequencies, and every exception cell is held.

    Decoded, not raw: an `int8` residual plane's bytes are codes, and checking
    those against `[0, 1]` would pass every store while saying nothing about
    what a query returns.
    """
    if encoding.eaf.is_residual:
        before = len(errors)
        _overflow_positions_match(
            "data.zarr/ragged/eaf",
            np.flatnonzero(np.asarray(root["eaf"][:]) == EAF_EXCEPTION).astype(np.int64),
            EafExceptionTable.read(root),
            errors,
            what="eaf exception",
            table_name="eaf exception",
        )
        if len(errors) > before:
            # The plane cannot be decoded at all while its table disagrees with
            # it, and saying so twice adds nothing.
            return
    plane = RaggedEafPlane.open(
        root, encoding, imputed=root["imputed"] if "imputed" in root else None
    )
    values = plane.slice(0, n_assoc)
    finite = np.isfinite(values)
    if np.any(finite & ((values < 0.0) | (values > 1.0))):
        errors.append("data.zarr/ragged/eaf contains finite values outside [0, 1]")


def _overflow_positions_match(
    label: str,
    observed: np.ndarray,
    table: ZOverflowTable | EafExceptionTable | SeExceptionTable,
    errors: list[str],
    *,
    what: str = "out-of-range z",
    table_name: str = "overflow",
) -> None:
    """Every side-table cell has an exact value, and nothing else does.

    A plane that says "look in the table" for a cell the table does not
    describe has lost that association outright. For `z` it would be the
    *largest* |z| in the store -- exactly the value nothing may be quietly
    wrong about -- and for `eaf` it is the frequency furthest from its
    baseline, which is the rare variant a user is filtering on.
    """
    if np.array_equal(np.sort(observed), table.index):
        return
    orphan = np.setdiff1d(observed, table.index)
    stray = np.setdiff1d(table.index, observed)
    if len(orphan):
        errors.append(
            f"{label}: {len(orphan)} {what} cell(s) have no entry in the {table_name} "
            f"table (first at flat position {int(orphan[0])})"
        )
    if len(stray):
        errors.append(
            f"{label}: {table_name} table describes {len(stray)} cell(s) that are not marked "
            f"{what} (first at flat position {int(stray[0])})"
        )


def _imputed_declaration_matches_arrays(
    analyses_path: Path,
    imputed_per_analysis: np.ndarray,
    errors: list[str],
) -> None:
    """Each Analysis holds imputed cells if and only if it says it does.

    Categorical, not a count comparison: `completion_n_imputed_total` is rolled
    up from per-block quality and the arrays hold what was actually written, so
    the two need not agree to the cell. What they may not do is disagree about
    *whether there are any* -- that is the shape the ancestry-match filter's
    defect took, an Analysis carried through observed-only while its metadata
    claimed thousands of imputed cells, and `eaf_scope` derived from the claim
    (ADR 0028, ADR 0037 §4). #106's defect was the same disagreement one
    granularity up.
    """
    for row in read_analyses(analyses_path).rows:
        try:
            index = int(row.get("analysis_index", ""))
        except ValueError:
            continue  # reported by the analyses rules
        if not 0 <= index < len(imputed_per_analysis):
            continue
        try:
            declared = int(row.get("completion_n_imputed_total", "") or 0)
        except ValueError:
            continue  # reported by the analyses rules
        actual = int(imputed_per_analysis[index])
        analysis_id = row.get("analysis_id", "")
        if declared and not actual:
            errors.append(
                f"analysis {analysis_id!r} declares completion_n_imputed_total={declared} "
                "but has no imputed=1 cell in data.zarr/imputed; the metadata describes "
                "cells the release does not contain"
            )
        elif actual and not declared:
            errors.append(
                f"analysis {analysis_id!r} has {actual} imputed=1 cell(s) in "
                "data.zarr/imputed but declares completion_n_imputed_total="
                f"{row.get('completion_n_imputed_total', '')!r}; the release holds "
                "imputed cells its metadata does not account for"
            )


def _validate_dense_arrays(
    root: Any, n_variants: int, n_analyses: int, errors: list[str],
    encoding: StoreEncoding, imputed_arr: Any = None,
    on_panel: np.ndarray | None = None, analyses_path: Path | None = None,
    warnings: list[str] | None = None, label: str = "data.zarr/se",
) -> None:
    """Stream the dense z/se (and, for a completed store, imputed) matrices in
    row-bands and run every matrix-touching content check in one pass.

    ``imputed_arr``/``on_panel`` are supplied only for Reference-Completed
    stores (from ``_validate_completion_metadata``); when present, the imputed
    checks that used to load the whole matrix are folded into this same band
    loop so peak memory stays at one band rather than the full matrices
    (issue 045). ``analyses_path`` lets the same pass answer the per-Analysis
    question -- does each Analysis hold the imputed cells it declares? -- for
    the cost of one ``int64`` per analysis, since the loop already reads
    ``imputed``.
    """
    planes = _dense_required_planes(root, errors)
    if planes is None:
        return
    z_arr, se_arr = planes
    eaf_arr = _dense_plane_shape_errors(root, z_arr, se_arr, n_variants, n_analyses, errors)
    _dense_eaf_side_lengths(root, n_variants, errors)
    if errors:
        return
    # Opened when there is a plane to decode *or* a panel frequency to
    # substitute: both are frequencies a query returns, so both are checked.
    eaf_plane = (
        DenseEafPlane.open(root, encoding)
        if eaf_arr is not None or encoding.eaf.reference
        else None
    )
    try:
        se_plane = DenseSePlane.open(root, encoding)
    except ValueError as exc:
        errors.append(str(exc))
        return

    # Stream in row-bands. Missingness is read through the plane's declared
    # missing *marker* (spec §15) -- NaN for a float plane, the reserved
    # sentinel for an integer one -- so nothing here decodes a value it does
    # not need, and no band is upcast.
    codec = StoreCodec(encoding)
    findings = _DenseBandState(n_analyses)
    for r0, r1 in _dense_array_validation_bands(z_arr, n_variants, n_analyses):
        if not _validate_dense_band(
            z_arr,
            se_arr,
            eaf_arr,
            se_plane,
            codec,
            encoding,
            eaf_plane,
            imputed_arr,
            on_panel,
            findings,
            n_analyses,
            r0,
            r1,
            errors,
        ):
            return
    _report_dense_band_flags(errors, findings, warnings, analyses_path, label)
    if imputed_arr is not None and analyses_path is not None:
        _imputed_declaration_matches_arrays(analyses_path, findings.imputed_per_analysis, errors)
    _report_dense_eaf_range(errors, findings)
    _report_dense_z_overflow(root, encoding, findings, errors)
    _report_dense_eaf_exceptions(root, encoding, eaf_plane, findings, errors)
    _match_dense_se_exceptions(root, encoding, findings.se_exception_positions, errors)


def _dense_required_planes(root: Any, errors: list[str]) -> tuple[Any, Any] | None:
    """The `z` and `se` planes exist, or every missing one is named and the pass stops.

    The first fail-safe guard: no check downstream means anything for a release
    that has no plane to check, and naming *both* missing planes rather than the
    first saves a repair-and-revalidate round trip.
    """
    for name in ("z", "se"):
        if name not in root:
            errors.append(f"missing data.zarr/{name}")
    if errors:
        return None
    return root["z"], root["se"]


def _dense_plane_shape_errors(
    root: Any,
    z_arr: Any,
    se_arr: Any,
    n_variants: int,
    n_analyses: int,
    errors: list[str],
) -> Any | None:
    """`z`/`se`/`eaf` all span exactly the (n_variants, n_analyses) matrix.

    `eaf` is optional (ADR 0036) -- absent on every store built before it, and
    on any build whose sources report no frequency -- but when present it is a
    third parallel plane and must have the same shape as z/se. A mis-shaped one
    is dropped here so no later seam decodes a plane that does not line up with
    the variant/analysis axes it would be read against; its shape error is
    already on the list, and the caller stops.
    """
    expected_shape = (n_variants, n_analyses)
    if tuple(z_arr.shape) != expected_shape:
        errors.append(f"z shape {tuple(z_arr.shape)} does not match {expected_shape}")
    if tuple(se_arr.shape) != expected_shape:
        errors.append(f"se shape {tuple(se_arr.shape)} does not match {expected_shape}")
    eaf_arr = root["eaf"] if "eaf" in root else None
    if eaf_arr is not None and tuple(eaf_arr.shape) != expected_shape:
        errors.append(f"eaf shape {tuple(eaf_arr.shape)} does not match {expected_shape}")
        eaf_arr = None
    return eaf_arr


def _dense_eaf_side_lengths(root: Any, n_variants: int, errors: list[str]) -> None:
    """The per-variant eaf side arrays are the length of the variant axis.

    Checked whether or not there is an `eaf` plane: a completed release whose
    Analyses reported no frequency carries `eaf_reference` and nothing else,
    and a mis-sized one there would hand every imputed cell the frequency of
    some other variant (issue #113).
    """
    if EAF_BASELINE in root and array_length(root[EAF_BASELINE]) != n_variants:
        errors.append(
            f"{EAF_BASELINE} has {array_length(root[EAF_BASELINE])} entries but the variant axis "
            f"has {n_variants}"
        )
    if EAF_REFERENCE in root and array_length(root[EAF_REFERENCE]) != n_variants:
        errors.append(
            f"{EAF_REFERENCE} has {array_length(root[EAF_REFERENCE])} entries but the variant axis "
            f"has {n_variants}"
        )


@dataclass
class _DenseBandState:
    """What one streamed band pass has found so far (issue 045).

    The state seam the row-band loop feeds: one field per independent flag or
    counter, and one list per side-table whose positions the report seams
    reconcile. Each ``_latch_*`` helper below owns exactly one field or one list,
    so no check sets another check's flag as a side effect.
    """

    n_analyses: int
    imputed_per_analysis: np.ndarray = field(init=False)
    infinite_se_per_analysis: np.ndarray = field(init=False)
    overflow_positions: list[np.ndarray] = field(default_factory=list)
    eaf_exception_positions: list[np.ndarray] = field(default_factory=list)
    se_exception_positions: list[np.ndarray] = field(default_factory=list)
    eaf_undecodable: bool = False
    neg_se: bool = False
    missingness: bool = False
    non_canonical: bool = False
    imp_not_binary: bool = False
    imp_nan_z: bool = False
    imp_nan_se: bool = False
    off_panel_imputed: bool = False
    eaf_out_of_range: bool = False

    def __post_init__(self) -> None:
        self.imputed_per_analysis = np.zeros(self.n_analyses, dtype=np.int64)
        self.infinite_se_per_analysis = np.zeros(self.n_analyses, dtype=np.int64)


def _validate_dense_band(
    z_arr: Any,
    se_arr: Any,
    eaf_arr: Any | None,
    se_plane: DenseSePlane,
    codec: StoreCodec,
    encoding: StoreEncoding,
    eaf_plane: DenseEafPlane | None,
    imputed_arr: Any | None,
    on_panel: np.ndarray | None,
    findings: _DenseBandState,
    n_analyses: int,
    r0: int,
    r1: int,
    errors: list[str],
) -> bool:
    """One row-band of the streamed pass; False after reporting a decode failure.

    Each band is sliced once per plane and handed to the flag and side-table
    seams below. A band whose `se` cannot be decoded is a hard stop -- the
    caller skips every report seam -- because a plane that cannot be decoded is
    not a release to keep reading.
    """
    # Residual SE decodes against physical EAF. Read that expensive plane once
    # and share it with the independent frequency-range check instead of
    # decoding it twice for every band (issue #254).
    eaf_band = _latch_dense_eaf_range(findings, eaf_plane, r0, r1)
    try:
        se_band = se_plane.band(r0, r1, eaf=eaf_band)
    except ValueError as exc:
        errors.append(f"cannot decode se: {exc}")
        return False
    z_band = z_arr[r0:r1]
    z_missing = codec.missing_mask(z_band)
    if encoding.z.is_fixed_point:
        findings.overflow_positions.append(
            _band_marked_positions(z_band, Z_OVERFLOW, r0, n_analyses)
        )
    if encoding.se.is_residual:
        findings.se_exception_positions.append(
            _band_marked_positions(se_arr[r0:r1], SE_EXCEPTION, r0, n_analyses)
        )
    _latch_negative_se(findings, se_band)
    # Only a floating-point plane can hold an infinity; a residual `int8` plane's decoded
    # values are finite by construction, so there is nothing there to count.
    if not encoding.se.is_residual:
        _latch_infinite_se(findings, se_band)
    if eaf_arr is not None and encoding.eaf.is_residual:
        findings.eaf_exception_positions.append(
            _band_marked_positions(eaf_arr[r0:r1], EAF_EXCEPTION, r0, n_analyses)
        )
    imp_raw = None if imputed_arr is None else np.asarray(imputed_arr[r0:r1])
    _latch_dense_states(findings, z_missing, se_band, imp_raw, z_band)
    imp_mask = _count_dense_imputed(findings, imp_raw)
    _latch_dense_off_panel_imputed(findings, on_panel, imp_mask, r0, r1)
    return True


def _band_marked_positions(raw: Any, marker: int, r0: int, n_analyses: int) -> np.ndarray:
    """Flat positions of the cells equal to ``marker`` in one Dense row band."""
    return np.flatnonzero(np.asarray(raw) == marker).astype(np.int64) + r0 * n_analyses


def _flatten_band_positions(bands: list[np.ndarray]) -> np.ndarray:
    """Band position lists concatenated in band order, or an empty int64 array."""
    return np.concatenate(bands) if bands else np.empty(0, dtype=np.int64)


def _latch_negative_se(findings: _DenseBandState, se_band: np.ndarray) -> None:
    """Any band with a finite negative se raises the flag once; later bands skip it."""
    if findings.neg_se:
        return
    findings.neg_se = bool(np.any(np.isfinite(se_band) & (se_band < 0)))


def _latch_infinite_se(findings: _DenseBandState, se_band: np.ndarray) -> None:
    """Count each Analysis's infinite se cells: a value, reported, never a failure.

    A source SE above `float16`'s maximum is narrowed to `+inf` beside a finite Z
    (issue #338). Spec §15 makes that cell present, so the release is valid -- but
    `beta = z * se` reads as infinity, and nothing else in a Store says so, so it
    is counted from the band the pass has already decoded and reported as a
    warning (issue #337). No plane is read a second time for this: the cost is one
    `isinf` pass over a band that is already resident, about 0.8 ns a cell.
    """
    findings.infinite_se_per_analysis += np.isinf(se_band).sum(axis=0, dtype=np.int64)


def _latch_dense_states(
    findings: _DenseBandState,
    z_missing: np.ndarray,
    se_band: np.ndarray,
    imputed: np.ndarray | None,
    z_raw: np.ndarray | None = None,
) -> None:
    """The band's cells under the presence rule (spec §15, ADR 0070).

    Z and SE must be missing in exactly the same cells, each marker canonical;
    the mask holds only 0 and 1; an imputed cell's Z and SE are both present. An
    infinite SE is present, so an imputed cell holding one passes here and is
    warned about by `_latch_infinite_se`.
    """
    se_missing = np.isnan(se_band)
    odd = non_canonical(se_band, se_missing)
    if z_raw is not None:
        odd |= non_canonical(z_raw)
    tally = Tally.of(z_missing, se_missing, imputed, odd)
    findings.missingness |= tally.count(State.PARTIAL) > 0
    findings.non_canonical |= tally.count(State.NON_CANONICAL) > 0
    findings.imp_not_binary |= tally.flags_not_binary > 0
    findings.imp_nan_z |= tally.imputed_z_missing > 0
    findings.imp_nan_se |= tally.imputed_se_missing > 0


def _latch_dense_eaf_range(
    findings: _DenseBandState, eaf_plane: DenseEafPlane | None, r0: int, r1: int
) -> np.ndarray | None:
    """Decode once, latch an EAF finding, and return the block for SE reuse.

    Decoded, not raw: an `int8` residual plane's bytes are codes, and checking
    those for "in [0, 1]" would pass every store while saying nothing about
    the frequencies a query returns. A decode that cannot resolve an exception
    cell is not reported from here: the position check says exactly which cell
    and why, so a band that fails to decode only sets the flag that stops
    further decode attempts.
    """
    if eaf_plane is None or findings.eaf_out_of_range or findings.eaf_undecodable:
        return None
    try:
        eaf = eaf_plane.band(r0, r1)
    except ValueError:
        findings.eaf_undecodable = True
        return None
    finite = np.isfinite(eaf)
    if np.any(finite & ((eaf < 0.0) | (eaf > 1.0))):
        findings.eaf_out_of_range = True
    return eaf


def _count_dense_imputed(
    findings: _DenseBandState, imputed: np.ndarray | None
) -> np.ndarray | None:
    """Per band: the imputed cells, counted per Analysis, as a mask.

    The counts keep accumulating whatever else the band held: they feed
    ``_imputed_declaration_matches_arrays``, and every imputed cell has to be
    counted.
    """
    if imputed is None:
        return None
    imp_mask: np.ndarray = imputed == 1
    findings.imputed_per_analysis += imp_mask.sum(axis=0, dtype=np.int64)
    return imp_mask


def _latch_dense_off_panel_imputed(
    findings: _DenseBandState,
    on_panel: np.ndarray | None,
    imp_mask: np.ndarray | None,
    r0: int,
    r1: int,
) -> None:
    """Off-panel rows can never be imputed (no LD structure)."""
    if imp_mask is None or findings.off_panel_imputed:
        return
    if on_panel is None:
        raise ValueError("imputed validation requires the on_panel array")
    off_band = on_panel[r0:r1] == 0
    if off_band.any() and np.any(imp_mask[off_band]):
        findings.off_panel_imputed = True


def _report_dense_band_flags(
    errors: list[str],
    findings: _DenseBandState,
    warnings: list[str] | None,
    analyses_path: Path | None,
    label: str,
) -> None:
    """The independent value flags the band latches set, in a fixed order.

    The one finding a release may keep -- an infinite `se` cell (issues #337, #338) --
    is a warning rather than an error, appended here so the band pass reports it
    without a second read of the plane.
    """
    if warnings is not None:
        _report_infinite_se(
            warnings, label, findings.infinite_se_per_analysis, analyses_path
        )
    if findings.neg_se:
        errors.append("se contains negative finite values")
    if findings.missingness:
        errors.append("z and se missingness is inconsistent")
    if findings.non_canonical:
        errors.append(_non_canonical_message(""))
    _report_dense_imputed_flags(errors, findings)


def _report_dense_imputed_flags(errors: list[str], findings: _DenseBandState) -> None:
    """The band flags about the imputed mask, in the reporter's fixed order."""
    if findings.imp_not_binary:
        errors.append("data.zarr/imputed contains values other than 0 and 1")
    if findings.imp_nan_z:
        errors.append("imputed=1 cells have missing z-scores")
    if findings.imp_nan_se:
        errors.append("imputed=1 cells have NaN se values")
    if findings.off_panel_imputed:
        errors.append(
            "off-panel (on_panel=0) rows have imputed=1 cells — off-panel is never imputable"
        )


def _report_dense_eaf_range(errors: list[str], findings: _DenseBandState) -> None:
    """Decoded frequencies are frequencies: none may fall outside [0, 1]."""
    if findings.eaf_out_of_range:
        errors.append("data.zarr/eaf contains finite values outside [0, 1]")


def _report_dense_z_overflow(
    root: Any, encoding: StoreEncoding, findings: _DenseBandState, errors: list[str]
) -> None:
    """The z overflow table describes exactly the Z_OVERFLOW cells (issue #114)."""
    if not encoding.z.is_fixed_point:
        return
    _overflow_positions_match(
        "data.zarr/z",
        _flatten_band_positions(findings.overflow_positions),
        ZOverflowTable.read(root),
        errors,
    )


def _report_dense_eaf_exceptions(
    root: Any,
    encoding: StoreEncoding,
    eaf_plane: DenseEafPlane | None,
    findings: _DenseBandState,
    errors: list[str],
) -> None:
    """Every eaf exception cell has an exact table entry, and no entry is stray.

    An undecodable band is only reported once the positions have been
    reconciled: a table that disagrees with the plane is the *reason* the plane
    cannot be decoded, and saying "cannot decode" before naming that reason
    would hide it.
    """
    if eaf_plane is None or not encoding.eaf.is_residual:
        return
    before = len(errors)
    _overflow_positions_match(
        "data.zarr/eaf",
        _flatten_band_positions(findings.eaf_exception_positions),
        EafExceptionTable.read(root),
        errors,
        what="eaf exception",
        table_name="eaf exception",
    )
    if findings.eaf_undecodable and len(errors) == before:
        errors.append("data.zarr/eaf cannot be decoded under the declared plan")


def _match_dense_se_exceptions(
    root: Any, encoding: StoreEncoding, bands: list[np.ndarray], errors: list[str]
) -> None:
    """Every `-127` in a Dense `se` plane has a table entry, and no entry is stray."""
    if not encoding.se.is_residual:
        return
    _overflow_positions_match(
        "data.zarr/se",
        _flatten_band_positions(bands),
        SeExceptionTable.read(root),
        errors,
        what="se exception",
        table_name="se exception",
    )


def _read_top_hit_arrays(key: str, group: Any, errors: list[str]) -> dict[str, Any] | None:
    """One tier's index arrays (`top_hits.load_tier_arrays`), or None having
    recorded what the tier lacks."""
    if _missing_top_hit_arrays(key, group, errors):
        return None
    return top_hits.load_tier_arrays(group)


def _top_hit_group_entry(
    key: str,
    group: Any,
    n_analyses: int,
    imputed_arr: Any,
    errors: list[str],
) -> dict[str, Any] | None:
    """Every check on one tier that does not need the matrix, then its entry.

    Returns None -- having recorded why -- when the tier is malformed enough
    that streaming the matrix against it would be meaningless.
    """
    arrays = _read_top_hit_arrays(key, group, errors)
    if arrays is None:
        return None
    problem = top_hits.structural_problem(arrays, group, n_analyses, imputed_arr is not None)
    if problem is not None:
        errors.append(f"top-hit index {key} {problem}")
        return None
    threshold = top_hits.tier_threshold(key, group)
    return {
        "z_crit": z_critical(threshold),
        "rows": arrays["rows"],
        "cols": arrays["cols"],
        "z_values": arrays["z_values"],
        "se_values": arrays["se_values"],
        "imputed_values": arrays["imputed_values"],
        "eaf_values": arrays["eaf_values"],
        "n": len(arrays["rows"]),
        "pass_count": 0,
        "consistent": True,
        "se_consistent": True,
        "imputed_consistent": True,
        "eaf_consistent": True,
    }


def _band_z_consistent(g: dict[str, Any], gathered: np.ndarray, in_band: np.ndarray) -> bool:
    """Whether the matrix agrees with what the tier recorded for these cells."""
    return bool(
        np.all(np.isfinite(gathered))
        and np.all(np.isclose(gathered, g["z_values"][in_band], rtol=1e-3, atol=1e-3))
        and np.all(np.abs(gathered) >= g["z_crit"])
    )


def _mark_band_mismatch(
    g: dict[str, Any],
    flag: str,
    gathered: np.ndarray,
    indexed: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Clear `flag` when the index disagrees with the plane it was built from.

    The tolerances are numpy's defaults unless a caller widens them: `eaf` is
    written from the decoded plane and must match it, while `se` is compared
    across a residual round trip and is allowed its quantisation step.
    """
    if not np.allclose(gathered, indexed, rtol=rtol, atol=atol, equal_nan=True):
        g[flag] = False


def _check_top_hit_band(
    g: dict[str, Any],
    r0: int,
    r1: int,
    z_band: np.ndarray,
    se_band: np.ndarray,
    abs_band: np.ndarray,
    imputed_arr: Any,
    eaf_band: Callable[[], np.ndarray],
) -> None:
    """Verify one tier's cells that fall in this row band, and count its passes."""
    g["pass_count"] += int(np.count_nonzero(abs_band >= g["z_crit"]))
    in_band = (g["rows"] >= r0) & (g["rows"] < r1)
    if not np.any(in_band):
        return
    br = g["rows"][in_band] - r0
    bc = g["cols"][in_band]
    if not _band_z_consistent(g, z_band[br, bc], in_band):
        g["consistent"] = False
    # rtol=1e-6, atol=0: the index is *written* from the decoded plane, so the
    # two are bit-identical in a sound store and anything looser hides the
    # defect this checks for. A stale index left over from a format migration
    # differs by the coding's half step -- 0.197% at range 0.5 -- which against
    # a typical SE of 0.05 is 1e-4, and an atol of 1e-3 waves it straight
    # through. Measured on the FinnGen R13 pilot, where exactly that happened.
    _mark_band_mismatch(
        g, "se_consistent", se_band[br, bc], g["se_values"][in_band], rtol=1e-6, atol=0.0
    )
    if g["imputed_values"] is not None and imputed_arr is not None:
        gathered = imputed_arr[r0:r1][br, bc].astype(np.uint8)
        if not np.array_equal(gathered, g["imputed_values"][in_band]):
            g["imputed_consistent"] = False
    if g["eaf_values"] is not None:
        _mark_band_mismatch(g, "eaf_consistent", eaf_band()[br, bc], g["eaf_values"][in_band])


def _report_top_hit_group(key: str, g: dict[str, Any], errors: list[str]) -> None:
    """The first thing wrong with a tier, once the streamed pass has finished."""
    if not g["consistent"]:
        errors.append(f"top-hit index {key} contains z value inconsistent with z array")
    elif not g["se_consistent"]:
        errors.append(f"top-hit index {key} contains se value inconsistent with decoded se plane")
    elif not g["imputed_consistent"]:
        errors.append(f"top-hit index {key} contains imputed value inconsistent with imputed array")
    elif not g["eaf_consistent"]:
        errors.append(f"top-hit index {key} contains eaf value inconsistent with eaf plane")
    elif g["n"] != g["pass_count"]:
        errors.append(f"top-hit index {key} does not match stored z values")


def _decode_top_hit_band(
    z_plane: DenseZPlane,
    se_plane: DenseSePlane,
    eaf_plane: DenseEafPlane,
    encoding: StoreEncoding,
    r0: int,
    r1: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Decode one Top-Hit validation band, sharing EAF with residual SE."""
    z_band = z_plane.band(r0, r1)
    shared_eaf = eaf_plane.band(r0, r1) if encoding.se.is_residual else None
    se_band = se_plane.band(r0, r1, eaf=shared_eaf)
    cached_eaf = [] if shared_eaf is None else [shared_eaf]
    return z_band, se_band, np.abs(z_band), cached_eaf


def _validate_top_hits(root: Any, errors: list[str], encoding: StoreEncoding) -> None:
    if not top_hits.has_index(root):
        return
    top = top_hits.index_group(root)
    z_plane = DenseZPlane.open(root, encoding)
    se_plane = DenseSePlane.open(root, encoding)
    z_arr = z_plane.array
    imputed_arr = root["imputed"] if "imputed" in root else None
    eaf_plane = DenseEafPlane.open(root, encoding)
    n_variants, n_analyses = int(z_arr.shape[0]), int(z_arr.shape[1])

    # Load each threshold group's (comparatively small) index arrays and run the
    # checks that don't need the matrix: array-length consistency, descending-|z|
    # ranking, and cell uniqueness. Keep the survivors for the streamed matrix pass.
    groups: dict[str, dict[str, Any]] = {}
    for key in top:
        entry = _top_hit_group_entry(key, top[key], n_analyses, imputed_arr, errors)
        if entry is not None:
            groups[key] = entry
    if not groups:
        return

    # Single streamed pass over the matrix in row-bands: per band, count cells that
    # pass each threshold (completeness) and verify every index cell in that band
    # (matrix z finite, matches the stored index z, and itself clears the cutoff).
    for r0, r1 in _dense_array_validation_bands(z_arr, n_variants, n_analyses):
        z_band, se_band, abs_band, cached = _decode_top_hit_band(
            z_plane, se_plane, eaf_plane, encoding, r0, r1
        )
        # Decoding the frequency plane is the most expensive read in this pass,
        # and every tier gathers from the same band: decode it at most once.

        def eaf_band(r0: int = r0, r1: int = r1, cached: list[np.ndarray] = cached) -> np.ndarray:
            if not cached:
                cached.append(eaf_plane.band(r0, r1))
            return cached[0]

        for g in groups.values():
            _check_top_hit_band(g, r0, r1, z_band, se_band, abs_band, imputed_arr, eaf_band)

    for key, g in groups.items():
        _report_top_hit_group(key, g, errors)


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
