"""Indexed Variant Subsets for Observed-Only Dense stores (ADR 0053, issue #264).

An **Indexed Variant Subset** is a named, optional, rebuildable query index over
a caller-supplied set of canonical ALIDs.  It holds every Analysis's full
available statistics -- Z, SE and EAF with each plane's exception/overflow
dependencies -- laid out Analysis-major so a single-Analysis read is a
sequential run of chunks.  It is *derived, non-authoritative* data: deleting it
changes no association and no metadata, and it does not move `format_version`.

This module owns the whole physical contract, so no caller or test has to know
the Zarr layout:

* the writer (`build_indexed_subset`) and its atomic publication;
* the read seam (`open_indexed_subset`, `IndexedSubset.decode_analysis`);
* the standalone validation rules (`validate_indexed_subsets`).

Three decisions are load-bearing and worth stating where they are implemented:

1. **The index copies the primary planes' stored codes, it does not re-encode
   decoded values.**  A decoded `int8_residual` EAF is the result of an `expit`,
   and re-encoding it through `logit` can land on a neighbouring code at a step
   boundary -- a value that is plausible, wrong, and exactly the class of defect
   ADR 0053 asks validation to catch.  Copying the codes and remapping the
   side-table positions makes the decoded index cell *identical* to the primary
   cell by construction.
2. **The EAF baseline is the primary baseline subset to the subset's variants.**
   A freshly computed subset baseline would decode a single-Analysis variant to
   its own value, not the release's.  The stored baseline here is the release's,
   so a copied residual code still means what it meant.
3. **Analysis-major planes, one Analysis per shard.**  A query for one Analysis
   reads that Analysis's shards and no other's; the build pays for it by
   rewriting shards band by band.

Query integration (#265) lives in `opengwasdb.query.facade`, which reads this
module's seam and applies the ordinary selected-Analysis result contract to it;
Reference-Completed support is issue #266 and is not implemented here.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from opengwasdb.build.liftover import normalise_build
from opengwasdb.encoding import (
    EAF_BASELINE,
    EAF_EXCEPTION_INDEX,
    EAF_EXCEPTION_VALUE,
    SE_MISSING,
    Z_OVERFLOW_INDEX,
    Z_OVERFLOW_VALUE,
    DenseEafPlane,
    DenseSePlane,
    DenseZPlane,
    EafExceptionTable,
    SeExceptionTable,
    StoreCodec,
    StoreEncoding,
    UnsupportedEncoding,
    ZOverflowTable,
    positions_flat,
    positions_rows_cols,
)
from opengwasdb.encoding.planes import SE_COEFFICIENTS
from opengwasdb.model.enums import CompletionState, PrimaryStorageLayout
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store.arrays import ArrayRole, array_length
from opengwasdb.store.open import (
    DestinationExistsError,
    destination_lock,
    directory_lock,
    open_store,
    staged_named_group,
    zarr_format_for_version,
)
from opengwasdb.variants import VariantAxis, parse_canonical_alid

log = logging.getLogger(__name__)

#: The group under `data.zarr` that holds every Indexed Variant Subset.
INDEXED_SUBSETS_GROUP = "indexed_subsets"

#: Target cell count per build/validate band.  Bands bound peak memory at the
#: band's raw codes plus its decoded floats and fixed overhead, independent of
#: the subset's total size; the CLI exposes it as `--band-cells`.
DEFAULT_BAND_CELLS = 4_000_000

#: The one attribute schema this module writes and reads.  A future layout
#: change moves it rather than adding a field a reader would have to guess at.
INDEXED_SUBSET_SCHEMA_VERSION = 1

#: The statistic profile every subset carries.  There is deliberately no Z-only
#: mode (ADR 0053), so this is a constant rather than a choice.
FULL_STATISTIC_PROFILE = "full_statistic"

#: Subset-name grammar: a leading alphanumeric, then letters/digits/`.`/`_`/`-`.
#: A leading `.` cannot occur, which is what stops a name colliding with the
#: `.{name}.tmp.*` staging groups this module creates inside the namespace.
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

#: Attribute keys.  Named constants so the writer, the reader and the validator
#: cannot disagree about a key's spelling -- the failure mode of a mistyped
#: attribute is a group that reads as if it never recorded the fact.
ATTR_SCHEMA = "indexed_subset_schema"
ATTR_NAME = "indexed_subset_name"
ATTR_PROFILE = "statistic_profile"
ATTR_ENCODING = "encoding"
ATTR_REFERENCE_ASSEMBLY = "reference_assembly"
ATTR_SOURCE_STORE_ID = "source_store_id"
ATTR_SOURCE_RELEASE_ID = "source_release_id"
ATTR_SOURCE_FORMAT_VERSION = "source_format_version"
ATTR_N_ANALYSES = "n_analyses"
ATTR_N_SUBSET_VARIANTS = "n_subset_variants"
ATTR_INPUT_SHA256 = "input_sha256"
ATTR_REQUESTED_COUNT = "requested_count"
ATTR_RESOLVED_COUNT = "resolved_count"
ATTR_ABSENT_COUNT = "absent_count"
ATTR_BUILDER_VERSION = "builder_version"
ATTR_CREATED_AT = "created_at"
ATTR_ORDER = "order"

#: Every attribute a published group must carry, in the order a message names
#: them.  The value checks below read them once this set is satisfied.
_REQUIRED_ATTRS: tuple[str, ...] = (
    ATTR_SCHEMA,
    ATTR_NAME,
    ATTR_PROFILE,
    ATTR_ENCODING,
    ATTR_REFERENCE_ASSEMBLY,
    ATTR_SOURCE_STORE_ID,
    ATTR_SOURCE_RELEASE_ID,
    ATTR_SOURCE_FORMAT_VERSION,
    ATTR_N_ANALYSES,
    ATTR_N_SUBSET_VARIANTS,
    ATTR_INPUT_SHA256,
    ATTR_REQUESTED_COUNT,
    ATTR_RESOLVED_COUNT,
    ATTR_ABSENT_COUNT,
    ATTR_BUILDER_VERSION,
    ATTR_CREATED_AT,
    ATTR_ORDER,
)

#: The `order` attribute value: the Analysis axis first, then the subset's
#: Store-order variant slot.  It is what a side table's flat position means.
_SUBSET_ORDER = "analysis,variant"

#: The SE exception table's array names.  Naming them once keeps the writer, the
#: reader and the validator from spelling one of them differently.
_SE_EXCEPTION_INDEX = "se_exception_index"
_SE_EXCEPTION_VALUE = "se_exception_value"

#: Zarr's own metadata filenames.  They are a group's own record of itself, not
#: namespace entries, so the namespace walk skips them (a v3 group's
#: `zarr.json`, a v2 group's `.zgroup`/`.zattrs`/`.zarray`/`.zmetadata`).
_ZARR_METADATA_NAMES = frozenset({"zarr.json", ".zgroup", ".zattrs", ".zarray", ".zmetadata"})


class IndexedSubsetError(Exception):
    """Base class for an Indexed Variant Subset that cannot be built or read."""


class IndexedSubsetNameError(IndexedSubsetError, ValueError):
    """A subset name is not in the accepted grammar."""


class VariantListError(IndexedSubsetError, ValueError):
    """A variant list is malformed, duplicated, or matches nothing."""


class IndexedSubsetAssemblyError(IndexedSubsetError, ValueError):
    """The requested Reference Assembly is not the Store Release's."""


class IndexedSubsetExistsError(IndexedSubsetError, FileExistsError):
    """A published subset already occupies the destination name."""


class IndexedSubsetStaleError(IndexedSubsetError):
    """A published subset does not belong to the release it sits in.

    Raised by the read seam before any decoded value is handed back: a stale
    index would return plausible associations for the wrong release, which is
    the failure class this project exists to refuse (#264 review).
    """


class IndexedSubsetLayoutError(IndexedSubsetError):
    """An Indexed Variant Subset was requested on a layout that has none.

    Indexed Variant Subsets are Observed-Only Dense only (ADR 0053, #264): a
    Ragged release is already a direct per-Analysis CSR and a Hybrid one has no
    rule for unifying its two components.  A selector naming a subset on either
    is a caller error, not a reason to answer from the ordinary path (#265).
    """


class IndexedSubsetValidationError(IndexedSubsetError):
    """A staged subset failed validation and must not be published."""


@dataclass(frozen=True)
class VariantList:
    """A parsed canonical-ALID list and the checksum of the bytes it came from."""

    path: Path
    sha256: str
    alids: tuple[str, ...]

    @property
    def requested(self) -> int:
        return len(self.alids)


@dataclass(frozen=True)
class ResolvedVariantList:
    """A variant list resolved against one Store Release's Variant Index.

    ``variant_index`` is sorted and unique in Store order regardless of the
    input order.  ``absent`` counts requested ALIDs the store does not carry;
    they are never treated as matches.
    """

    variant_index: np.ndarray
    requested: int
    resolved: int
    absent: int


@dataclass(frozen=True)
class IndexedSubsetAnalysis:
    """One Analysis's decoded subset column, in Store Variant Index order."""

    variant_index: np.ndarray
    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray | None


@dataclass(frozen=True)
class IndexedSubset:
    """Read-only metadata and decode seam for one published Indexed Variant Subset.

    The physical Zarr group is private: callers read `variant_index`, the
    recorded provenance, and decoded columns through this object.
    """

    store_path: Path
    name: str
    path: Path
    variant_index: np.ndarray
    n_analyses: int
    n_subset_variants: int
    encoding: StoreEncoding
    reference_assembly: str
    source_store_id: str
    source_release_id: str
    source_format_version: str
    input_sha256: str
    requested_count: int
    resolved_count: int
    absent_count: int
    builder_version: str
    created_at: str
    has_eaf: bool
    n_z_overflow: int
    n_eaf_exceptions: int
    n_se_exceptions: int
    _group: Any = field(repr=False, compare=False)

    def decode_analysis(self, analysis_index: int) -> IndexedSubsetAnalysis:
        """Decode one Analysis's Z, SE and (when present) EAF columns.

        This is the read seam #265 builds the query fast path on: the returned
        values are what the primary planes decode to at the same variants.
        """
        if not 0 <= analysis_index < self.n_analyses:
            raise IndexedSubsetError(
                f"store {self.store_path}: Indexed Variant Subset {self.name!r}: "
                f"Analysis index {analysis_index} is outside [0, {self.n_analyses})"
            )
        return _decode_analysis_column(
            self._group,
            self.encoding,
            self.n_subset_variants,
            analysis_index,
            self.variant_index,
        )


@dataclass(frozen=True)
class IndexedSubsetBuild:
    """What one successful `build_indexed_subset` published."""

    name: str
    path: Path
    n_analyses: int
    n_subset_variants: int
    requested_count: int
    resolved_count: int
    absent_count: int
    input_sha256: str


@dataclass(frozen=True)
class _SubsetPlan:
    """Everything a prepared build needs to stage and publish its group."""

    store_path: Path
    name: str
    manifest: StoreManifest
    variant_list: VariantList
    resolved: ResolvedVariantList
    root: Any
    n_analyses: int
    n_variants: int
    fmt: int
    attrs: dict[str, Any]


# ── Input contract ──────────────────────────────────────────────────────────


def parse_indexed_subset_name(name: str) -> str:
    """Validate and return a subset name, or fail naming the grammar.

    A leading `.` is refused so a subset name can never collide with the
    ``.{name}.tmp.*`` staging groups or the ``.{name}.old`` replacement
    directory this module creates beside the published group.
    """
    if not _NAME_RE.match(name):
        raise IndexedSubsetNameError(
            f"indexed subset name {name!r} is not valid: it must start with a letter or "
            "digit and contain only letters, digits, '.', '_' and '-'; a leading '.' is "
            "reserved for staging and replacement directories"
        )
    return name


def is_valid_indexed_subset_name(name: str) -> bool:
    """Whether `name` could be a published subset name (no raise)."""
    return bool(_NAME_RE.match(name))


def _is_staging_name(name: str) -> bool:
    """Whether a namespace entry is a staging/replacement directory.

    A staging group is `.{name}.tmp.{token}` and a mid-swap replacement is
    `.{name}.old`; both start with a dot and neither is a published index.
    """
    return name.startswith(".")


def read_variant_list(path: str | Path) -> VariantList:
    """Parse one canonical ALID per nonblank line, rejecting malformed or repeated.

    The rejected line and its number are named, because "malformed ALID" without
    a line is not something a caller can act on.
    """
    list_path = Path(path)
    raw = list_path.read_bytes()
    alids: list[str] = []
    seen: set[str] = set()
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        candidate = line.strip()
        if not candidate:
            continue
        canonical = _canonical_alid_or_none(candidate)
        if canonical is None:
            raise VariantListError(
                f"{list_path}:{number}: {candidate!r} is not a canonical ALID"
            )
        if canonical in seen:
            raise VariantListError(f"{list_path}:{number}: duplicate ALID {canonical!r}")
        seen.add(canonical)
        alids.append(canonical)
    return VariantList(
        path=list_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        alids=tuple(alids),
    )


def _canonical_alid_or_none(candidate: str) -> str | None:
    """`candidate` in canonical form, or None when it is not a canonical ALID."""
    parsed = parse_canonical_alid(candidate)
    if parsed is None:
        return None
    effect, other = sorted((parsed.effect_allele, parsed.other_allele))
    canonical = f"{parsed.chromosome}:{parsed.position}:{effect}:{other}"
    return canonical if canonical == candidate else None


def resolve_variant_list(
    store_path: str | Path, alids: tuple[str, ...] | list[str]
) -> ResolvedVariantList:
    """Resolve canonical ALIDs to sorted, unique Store Variant Indices.

    Absent ALIDs are dropped and counted, never substituted.  The result is in
    Store order, because the Variant Index is the store's axis and the index's
    arrays are written against the store's order.
    """
    axis = VariantAxis(Path(store_path))
    try:
        resolved = axis.indices_by_identifiers(list(alids))
    finally:
        axis.close()
    unique = np.unique(np.asarray(resolved, dtype=np.int64))
    return ResolvedVariantList(
        variant_index=unique.astype(np.int64),
        requested=len(alids),
        resolved=int(len(unique)),
        absent=int(len(alids) - len(unique)),
    )


# ── Build ───────────────────────────────────────────────────────────────────


def _check_assembly(manifest: StoreManifest, requested: str) -> str:
    declared = normalise_build(manifest.reference_assembly)
    wanted = normalise_build(requested)
    if declared != wanted:
        raise IndexedSubsetAssemblyError(
            f"requested Reference Assembly {requested!r} ({wanted}) does not match "
            f"this Store Release's {manifest.reference_assembly!r} ({declared}); an "
            "Indexed Variant Subset is only meaningful against the assembly its "
            "Variant Index was built on"
        )
    return wanted


def _builder_version() -> str:
    """The installed package version, or the constant this build ships with."""
    from importlib.metadata import PackageNotFoundError, version

    from opengwasdb.build.resolve_manifest import OPENGWASDB_PACKAGE_VERSION

    try:
        return version("opengwasdb")
    except PackageNotFoundError:
        return OPENGWASDB_PACKAGE_VERSION


def _band_bounds(n_subset: int, n_analyses: int, band_cells: int) -> list[tuple[int, int]]:
    """Subset-variant bands whose cell count is at most `band_cells`.

    A zero-variant subset has no bands; returning an empty list rather than a
    zero-stride ``range`` keeps a caller from crashing on an index that
    validation separately rejects as empty.
    """
    if n_subset <= 0:
        return []
    per_band = max(1, int(band_cells) // max(1, n_analyses))
    per_band = min(per_band, n_subset)
    return [
        (start, min(start + per_band, n_subset))
        for start in range(0, n_subset, per_band)
    ]


def _remap_side_table(
    source_index: np.ndarray,
    source_value: np.ndarray,
    subset: np.ndarray,
    n_analyses: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-key a primary side table into the Analysis-major subset layout.

    A primary cell's flat position is ``variant * n_analyses + analysis``; the
    subset's is ``analysis * n_subset + slot``.  Positions whose variant is not
    in the subset are dropped -- they are not cells this index holds.
    """
    index = np.asarray(source_index, dtype=np.int64)
    value = np.asarray(source_value, dtype=np.float32)
    if len(index) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    source_variants = index // n_analyses
    source_analyses = index % n_analyses
    n_subset = len(subset)
    slots = np.searchsorted(subset, source_variants)
    in_bounds = slots < n_subset
    hit = np.zeros(len(index), dtype=bool)
    hit[in_bounds] = subset[slots[in_bounds]] == source_variants[in_bounds]
    new_index = source_analyses[hit] * n_subset + slots[hit]
    new_value = value[hit]
    order = np.argsort(new_index, kind="stable")
    return new_index[order], new_value[order]


def _read_side(root: Any, name: str) -> np.ndarray:
    return np.asarray(root[name][:], dtype=np.int64 if name.endswith("index") else np.float32)


def _create_subset_arrays(
    group: Any, encoding: StoreEncoding, codec: StoreCodec, n_analyses: int, n_subset: int
) -> tuple[Any, Any, Any | None]:
    """Create the Analysis-major statistic planes, filled with each plane's marker."""
    comp = store_arrays.compressor()
    z = store_arrays.create_array(
        group,
        "z",
        ArrayRole.INDEXED_SUBSET_PLANE,
        shape=(n_analyses, n_subset),
        dtype=encoding.z.dtype,
        fill_value=codec.z_fill_value,
        compressor=comp,
    )
    se = store_arrays.create_array(
        group,
        "se",
        ArrayRole.INDEXED_SUBSET_PLANE,
        shape=(n_analyses, n_subset),
        dtype=encoding.se.dtype,
        fill_value=codec.se_fill_value,
        compressor=comp,
    )
    eaf = None
    if not encoding.eaf.is_absent:
        eaf = store_arrays.create_array(
            group,
            "eaf",
            ArrayRole.INDEXED_SUBSET_PLANE,
            shape=(n_analyses, n_subset),
            dtype=encoding.eaf.dtype,
            fill_value=codec.eaf_fill_value,
            compressor=comp,
        )
    return z, se, eaf


def _copy_plane_bands(
    z_target: Any,
    se_target: Any,
    eaf_target: Any | None,
    root: Any,
    subset: np.ndarray,
    n_analyses: int,
    band_cells: int,
) -> None:
    """Copy the primary planes' codes into the Analysis-major index, by band."""
    for start, stop in _band_bounds(len(subset), n_analyses, band_cells):
        rows = subset[start:stop]
        z_target[:, start:stop] = np.asarray(root["z"].oindex[rows, :]).T
        se_target[:, start:stop] = np.asarray(root["se"].oindex[rows, :]).T
        if eaf_target is not None:
            eaf_target[:, start:stop] = np.asarray(root["eaf"].oindex[rows, :]).T


def _remap_and_write(
    group: Any,
    root: Any,
    table_type: type[Any],
    index_name: str,
    value_name: str,
    subset: np.ndarray,
    n_analyses: int,
) -> None:
    """Re-key one primary side table and write it beside the indexed planes."""
    index, value = _remap_side_table(
        _read_side(root, index_name), _read_side(root, value_name), subset, n_analyses
    )
    table_type(index=index, value=value).write(group, compressor=store_arrays.compressor())


def _write_z_side_tables(
    group: Any, root: Any, encoding: StoreEncoding, subset: np.ndarray, n_analyses: int
) -> None:
    if not encoding.z.is_fixed_point:
        return
    _remap_and_write(
        group, root, ZOverflowTable, Z_OVERFLOW_INDEX, Z_OVERFLOW_VALUE, subset, n_analyses
    )


def _write_eaf_side_tables(
    group: Any, root: Any, encoding: StoreEncoding, subset: np.ndarray, n_analyses: int
) -> None:
    if not encoding.eaf.is_residual:
        return
    baseline = np.asarray(root[EAF_BASELINE].oindex[subset], dtype=np.float32)
    store_arrays.create_array(
        group,
        EAF_BASELINE,
        ArrayRole.PER_VARIANT,
        data=baseline,
        dtype="float32",
        compressor=store_arrays.compressor(),
        hint=store_arrays.INDEXED_SUBSET_CHUNK,
    )
    _remap_and_write(
        group,
        root,
        EafExceptionTable,
        EAF_EXCEPTION_INDEX,
        EAF_EXCEPTION_VALUE,
        subset,
        n_analyses,
    )


def _write_se_side_tables(
    group: Any, root: Any, encoding: StoreEncoding, subset: np.ndarray, n_analyses: int
) -> None:
    if not encoding.se.is_residual:
        return
    store_arrays.create_array(
        group,
        SE_COEFFICIENTS,
        ArrayRole.SE_COEFFICIENTS,
        data=np.asarray(root[SE_COEFFICIENTS][:], dtype=np.float32),
        dtype="float32",
        compressor=store_arrays.compressor(),
    )
    _remap_and_write(
        group,
        root,
        SeExceptionTable,
        _SE_EXCEPTION_INDEX,
        _SE_EXCEPTION_VALUE,
        subset,
        n_analyses,
    )


def write_indexed_subset_group(
    group: Any,
    root: Any,
    encoding: StoreEncoding,
    subset: np.ndarray,
    n_analyses: int,
    attrs: dict[str, Any],
    band_cells: int,
) -> None:
    """Write one complete Indexed Variant Subset group from the primary planes.

    Every array a published subset holds is created here; validation compares
    what this wrote against the primaries, and a caller never assembles the
    group by hand.  The planes are copied band by band, so peak memory is the
    configured band rather than the subset matrix.
    """
    subset = np.asarray(subset, dtype=np.int64)
    n_subset = len(subset)
    codec = StoreCodec(encoding)
    store_arrays.create_array(
        group,
        "variant_index",
        ArrayRole.INDEXED_SUBSET_VARIANT_INDEX,
        data=subset.astype("int32"),
        dtype="int32",
        compressor=store_arrays.compressor(),
    )
    z_target, se_target, eaf_target = _create_subset_arrays(
        group, encoding, codec, n_analyses, n_subset
    )
    _copy_plane_bands(z_target, se_target, eaf_target, root, subset, n_analyses, band_cells)
    _write_z_side_tables(group, root, encoding, subset, n_analyses)
    _write_eaf_side_tables(group, root, encoding, subset, n_analyses)
    _write_se_side_tables(group, root, encoding, subset, n_analyses)
    group.attrs.update(attrs)


def _check_layout(manifest: StoreManifest) -> None:
    """Refuse anything but an Observed-Only Dense Store Release."""
    if manifest.primary_layout is not PrimaryStorageLayout.DENSE:
        raise IndexedSubsetError(
            "Indexed Variant Subsets are Dense-only (ADR 0053); this release's "
            f"primary_layout is {manifest.primary_layout.value!r}"
        )
    if manifest.completion_state is not CompletionState.OBSERVED_ONLY:
        raise IndexedSubsetError(
            "Indexed Variant Subsets are Observed-Only for now; Reference-Completed "
            "support is issue #266. This release is "
            f"{manifest.completion_state.value!r}"
        )
    if manifest.encoding.eaf.reference:
        raise IndexedSubsetError(
            "this Observed-Only release declares reference EAF, which an Observed-Only "
            "index does not carry; refusing rather than writing an index that cannot "
            "reproduce the release's ordinary result"
        )


def _build_attributes(
    name: str,
    manifest: StoreManifest,
    variant_list: VariantList,
    resolved: ResolvedVariantList,
    n_analyses: int,
    n_subset: int,
) -> dict[str, Any]:
    return {
        ATTR_SCHEMA: INDEXED_SUBSET_SCHEMA_VERSION,
        ATTR_NAME: name,
        ATTR_PROFILE: FULL_STATISTIC_PROFILE,
        ATTR_ENCODING: manifest.encoding.to_manifest(),
        ATTR_REFERENCE_ASSEMBLY: normalise_build(manifest.reference_assembly),
        ATTR_SOURCE_STORE_ID: manifest.store_id,
        ATTR_SOURCE_RELEASE_ID: manifest.release_id,
        ATTR_SOURCE_FORMAT_VERSION: manifest.format_version,
        ATTR_N_ANALYSES: int(n_analyses),
        ATTR_N_SUBSET_VARIANTS: int(n_subset),
        ATTR_INPUT_SHA256: variant_list.sha256,
        ATTR_REQUESTED_COUNT: int(resolved.requested),
        ATTR_RESOLVED_COUNT: int(resolved.resolved),
        ATTR_ABSENT_COUNT: int(resolved.absent),
        ATTR_BUILDER_VERSION: _builder_version(),
        ATTR_CREATED_AT: dt.datetime.now(dt.UTC).isoformat(),
        ATTR_ORDER: _SUBSET_ORDER,
    }


def _resolve_inputs(
    store_path: Path, variant_list_path: str | Path, reference_assembly: str
) -> tuple[StoreManifest, VariantList, ResolvedVariantList]:
    """Open the release, check the layout and assembly, and resolve the list."""
    manifest = open_store(store_path).manifest
    _check_layout(manifest)
    _check_assembly(manifest, reference_assembly)
    variant_list = read_variant_list(variant_list_path)
    resolved = resolve_variant_list(store_path, variant_list.alids)
    if resolved.resolved == 0:
        raise VariantListError(
            f"variant list {variant_list.path} names no Store variants; an Indexed "
            "Variant Subset with no variants is not a valid index"
        )
    if resolved.absent:
        log.warning(
            "%d of %d requested ALIDs are absent from this Store Release and were not "
            "indexed",
            resolved.absent,
            resolved.requested,
        )
    return manifest, variant_list, resolved


def _prepare_subset(
    store_path: str | Path,
    subset_name: str,
    variant_list_path: str | Path,
    reference_assembly: str,
    band_cells: int,
) -> _SubsetPlan:
    """Validate the whole input contract and open the release, before any write."""
    store_path = Path(store_path)
    name = parse_indexed_subset_name(subset_name)
    if band_cells <= 0:
        raise IndexedSubsetError(f"band_cells must be positive, got {band_cells}")
    manifest, variant_list, resolved = _resolve_inputs(
        store_path, variant_list_path, reference_assembly
    )
    root = store_arrays.open_group(store_path / "data.zarr")
    n_analyses = int(root["z"].shape[1])
    n_variants = int(root["z"].shape[0])
    attrs = _build_attributes(
        name, manifest, variant_list, resolved, n_analyses, len(resolved.variant_index)
    )
    return _SubsetPlan(
        store_path=store_path,
        name=name,
        manifest=manifest,
        variant_list=variant_list,
        resolved=resolved,
        root=root,
        n_analyses=n_analyses,
        n_variants=n_variants,
        fmt=zarr_format_for_version(manifest.format_version, source=f"release at {store_path}"),
        attrs=attrs,
    )


def _ensure_indexed_subset_namespace(store_path: Path, fmt: int) -> None:
    """Create the optional namespace group, serialised against other creators.

    Namespace creation is a write to ``data.zarr``, so it must not race: two
    first-time builds for different names would otherwise both observe the
    namespace absent and both create it (#264 review).  The lock is the
    ``data.zarr`` inode -- a directory every creator shares -- and is released
    before the staged build takes the namespace's own publication lock, so the
    two locks are never nested.
    """
    data_path = store_path / "data.zarr"
    try:
        with directory_lock(data_path):
            root = store_arrays.open_group_for_write(data_path, "a", zarr_format=fmt)
            store_arrays.require_group(root, INDEXED_SUBSETS_GROUP)
    except IndexedSubsetError:
        raise
    except Exception as exc:
        raise IndexedSubsetError(
            f"could not create the indexed-subset namespace in {data_path}: {exc}"
        ) from exc


def _publish_subset(plan: _SubsetPlan, overwrite: bool, band_cells: int) -> IndexedSubsetBuild:
    """Stage, validate and atomically publish one prepared subset."""
    dest = plan.store_path / "data.zarr" / INDEXED_SUBSETS_GROUP / plan.name
    _ensure_indexed_subset_namespace(plan.store_path, plan.fmt)
    try:
        with staged_named_group(dest, overwrite=overwrite) as work:
            staged = store_arrays.open_group_for_write(
                work, "w", zarr_format=plan.fmt
            )
            write_indexed_subset_group(
                staged,
                plan.root,
                plan.manifest.encoding,
                plan.resolved.variant_index,
                plan.n_analyses,
                plan.attrs,
                band_cells,
            )
            _validate_staged(plan, work)
    except DestinationExistsError as exc:
        raise IndexedSubsetExistsError(str(exc)) from exc
    return IndexedSubsetBuild(
        name=plan.name,
        path=dest,
        n_analyses=plan.n_analyses,
        n_subset_variants=int(len(plan.resolved.variant_index)),
        requested_count=plan.resolved.requested,
        resolved_count=plan.resolved.resolved,
        absent_count=plan.resolved.absent,
        input_sha256=plan.variant_list.sha256,
    )


def build_indexed_subset(
    store_path: str | Path,
    subset_name: str,
    variant_list_path: str | Path,
    *,
    reference_assembly: str,
    overwrite: bool = False,
    band_cells: int = DEFAULT_BAND_CELLS,
) -> IndexedSubsetBuild:
    """Build, validate and atomically publish one Indexed Variant Subset.

    The input contract (canonical ALIDs, an explicit matching Reference
    Assembly, a nonempty match) is enforced before any group is created; the
    build then stages a unique sibling, validates it against the primary planes,
    and publishes it under the namespace's advisory lock (ADR 0043).
    """
    plan = _prepare_subset(
        store_path, subset_name, variant_list_path, reference_assembly, band_cells
    )
    return _publish_subset(plan, overwrite=overwrite, band_cells=band_cells)


def _validate_staged(plan: _SubsetPlan, work: Path) -> None:
    """Validate the freshly written group before it may be published."""
    staged = store_arrays.open_group(work)
    errors: list[str] = []
    context = _SubsetValidation(
        plan.store_path, plan.root, plan.manifest, plan.n_variants, plan.n_analyses
    )
    _validate_one_indexed_subset(context, plan.name, staged, errors)
    if errors:
        raise IndexedSubsetValidationError(
            f"staged Indexed Variant Subset {plan.name!r} failed validation and was not "
            "published: " + "; ".join(errors)
        )


def remove_indexed_subset(store_path: str | Path, subset_name: str) -> bool:
    """Delete one published subset group, returning whether it existed.

    Removing a derived index changes no authoritative data; a release with none
    is valid (ADR 0053).  The removal takes the namespace's publication lock
    and first renames the group aside, so it can never interleave with a commit
    of the same name and a reader never sees a half-deleted group: the published
    name disappears atomically and the bytes are then reclaimed.  The rename
    bypasses zarr's own delete guard, so a group under consolidated metadata is
    refused before it -- the record would keep listing the removed group and
    the next open would read it.
    """
    name = parse_indexed_subset_name(subset_name)
    store_path = Path(store_path)
    namespace_dir = store_path / "data.zarr" / INDEXED_SUBSETS_GROUP
    if not namespace_dir.is_dir():
        return False
    dest = namespace_dir / name
    with destination_lock(dest):
        if not dest.is_dir():
            return False
        # A raw rename bypasses zarr's own delete guard, so it repeats the
        # consolidated-metadata refusal here: a record would keep listing the
        # removed group and the next open would read it (#264 review).
        store_arrays.refuse_under_consolidated_metadata(
            dest, f"deleting indexed subset {name!r} in", wiped=True
        )
        doomed = namespace_dir / f".{name}.old.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        os.replace(dest, doomed)
    shutil.rmtree(doomed, ignore_errors=True)
    return True


# ── Read seam ───────────────────────────────────────────────────────────────


def list_indexed_subsets(store_path: str | Path) -> tuple[str, ...]:
    """The published subset names under one release, sorted.

    Read from the filesystem so a stray entry that is not a group is visible to
    validation rather than silently absent from the listing.
    """
    namespace = Path(store_path) / "data.zarr" / INDEXED_SUBSETS_GROUP
    if not namespace.is_dir():
        return ()
    return tuple(
        sorted(
            entry.name
            for entry in namespace.iterdir()
            if entry.is_dir() and is_valid_indexed_subset_name(entry.name)
        )
    )


def _missing_required_attrs(attrs: dict[str, Any]) -> list[str]:
    return [key for key in _REQUIRED_ATTRS if key not in attrs]


def _require_attr_int(name: str, attrs: dict[str, Any], key: str) -> int:
    """One attribute's integer value, or a loud failure naming the subset.

    A malformed `indexed_subset`/`n_analyses`/count attr must not escape as a
    bare `ValueError`: the query path and the CLI promise a subset-naming
    error, and a traceback naming neither Store nor subset breaks that promise.
    """
    value = attrs[key]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} records a non-integer {key}: {value!r}"
        )
    return int(value)


def _parse_subset_encoding(
    name: str, attrs: dict[str, Any], errors: list[str]
) -> StoreEncoding | None:
    """The group's declared encoding, or None with a finding appended.

    Any way the encoding block can be malformed -- an unsupported kind, a
    missing key, a non-mapping value, an unparseable number -- is normalised
    into the same subset-naming failure.  The read seam and validation share
    this, so a reader and the validator cannot disagree about whether a block
    is readable (#265 review).
    """
    try:
        return StoreEncoding.from_manifest(attrs[ATTR_ENCODING])
    except UnsupportedEncoding as exc:
        errors.append(
            f"indexed subset {name!r} declares an encoding this build cannot read: {exc}"
        )
        return None
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(
            f"indexed subset {name!r} declares malformed encoding metadata "
            f"({type(exc).__name__}: {exc})"
        )
        return None


def _refuse_stale_subset(
    name: str, attrs: dict[str, Any], encoding: StoreEncoding, manifest: StoreManifest
) -> None:
    """Raise when a published subset does not belong to `manifest`'s release."""
    mismatches = _source_identity_mismatches(name, attrs, manifest)
    if encoding != manifest.encoding:
        mismatches.append(
            f"indexed subset {name!r} was built with an encoding that is not the "
            "release's; its values cannot be decoded against the authoritative planes"
        )
    if mismatches:
        raise IndexedSubsetStaleError("; ".join(mismatches))


def _side_table_length(group: Any, name: str) -> int:
    return array_length(group[name]) if name in group else 0


def _require_expected_arrays(name: str, group: Any, encoding: StoreEncoding) -> None:
    """The group carries exactly the arrays its encoding defines."""
    expected = _expected_arrays(encoding)
    missing = sorted(key for key in expected if key not in group)
    if missing:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} is missing arrays: {', '.join(missing)}"
        )
    _require_only_expected_members(name, group, expected)


def _require_only_expected_members(
    name: str, group: Any, expected: frozenset[str]
) -> None:
    """Every member is an expected array, not a group or an unknown array."""
    members = list(group.keys())
    non_arrays = sorted(key for key in members if not isinstance(group[key], zarr.Array))
    if non_arrays:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} carries unexpected groups: {', '.join(non_arrays)}"
        )
    unknown = sorted(key for key in members if key not in expected)
    if unknown:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} carries unexpected arrays: {', '.join(unknown)}"
        )


def _require_declared_metadata(name: str, attrs: dict[str, Any]) -> None:
    """The group's own name, schema, profile and axis order, before decoding.

    These mirror validation's `_read_declared_encoding` / `_check_identity_attrs`
    / `_check_variant_index_attrs`, and running them on the read path is what
    stops a re-labelled or transposed group from decoding under the wrong
    contract when its shapes happen to fit (a square `n_analyses ==
    n_subset_variants` group is exactly the case the `order` check is for).
    Each is one attribute read, so the seam stays cheap.
    """
    if _require_attr_int(name, attrs, ATTR_SCHEMA) != INDEXED_SUBSET_SCHEMA_VERSION:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} declares schema {attrs[ATTR_SCHEMA]!r}, not "
            f"{INDEXED_SUBSET_SCHEMA_VERSION}"
        )
    if str(attrs[ATTR_NAME]) != name:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} records its name as {attrs[ATTR_NAME]!r}"
        )
    if str(attrs[ATTR_PROFILE]) != FULL_STATISTIC_PROFILE:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} declares statistic_profile "
            f"{attrs[ATTR_PROFILE]!r}, not {FULL_STATISTIC_PROFILE!r}"
        )
    if str(attrs[ATTR_ORDER]) != _SUBSET_ORDER:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} declares order {attrs[ATTR_ORDER]!r}, not "
            f"{_SUBSET_ORDER!r}; its planes may be transposed"
        )


def _require_declared_counts(name: str, attrs: dict[str, Any], n_analyses: int) -> None:
    """The recorded Analysis count matches the release, and the counts add up.

    Mirrors validation's `_check_identity_attrs`.  Cheap (four attributes) and
    validation-visible, so the read path runs it rather than letting a
    wrong-width index decode under the release's shape by coincidence.
    """
    if _require_attr_int(name, attrs, ATTR_N_ANALYSES) != n_analyses:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} declares {attrs[ATTR_N_ANALYSES]} Analyses "
            f"but this release has {n_analyses}"
        )
    requested = _require_attr_int(name, attrs, ATTR_REQUESTED_COUNT)
    resolved = _require_attr_int(name, attrs, ATTR_RESOLVED_COUNT)
    absent = _require_attr_int(name, attrs, ATTR_ABSENT_COUNT)
    if requested != resolved + absent:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} records requested={requested}, "
            f"resolved={resolved}, absent={absent}, which do not add up"
        )


def _require_variant_axis(
    name: str, group: Any, attrs: dict[str, Any], n_variants: int
) -> np.ndarray:
    """The subset's Variant Indices are non-empty, sorted, unique and in bounds."""
    n_subset = _require_attr_int(name, attrs, ATTR_N_SUBSET_VARIANTS)
    variant_index = np.asarray(group["variant_index"][:], dtype=np.int64)
    if len(variant_index) != n_subset:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} variant_index holds {len(variant_index)} "
            f"variants but declares {n_subset}"
        )
    if n_subset == 0:
        raise IndexedSubsetError(f"Indexed Variant Subset {name!r} contains no variants")
    if n_subset > 1 and np.any(variant_index[1:] <= variant_index[:-1]):
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} variant_index is not sorted ascending and unique"
        )
    if int(variant_index.min()) < 0 or int(variant_index.max()) >= n_variants:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} variant_index is out of bounds "
            f"[0, {n_variants})"
        )
    return variant_index


def _require_plane_shapes(
    name: str, group: Any, encoding: StoreEncoding, attrs: dict[str, Any], n_subset: int
) -> None:
    """Each statistic plane spans ``(n_analyses, n_subset_variants)`` in its declared dtype."""
    expected_shape = (_require_attr_int(name, attrs, ATTR_N_ANALYSES), n_subset)
    encodings: dict[str, Any] = {"z": encoding.z, "se": encoding.se, "eaf": encoding.eaf}
    for plane, declared in encodings.items():
        if plane not in group:
            continue
        actual = tuple(int(size) for size in group[plane].shape)
        if actual != expected_shape:
            raise IndexedSubsetError(
                f"Indexed Variant Subset {name!r} {plane} shape {actual} does not match "
                f"{expected_shape}"
            )
        if str(group[plane].dtype) != declared.dtype:
            raise IndexedSubsetError(
                f"Indexed Variant Subset {name!r} {plane} has dtype {group[plane].dtype} "
                f"but the declared encoding is {declared.kind} ({declared.dtype})"
            )


def _require_decodable_subset(
    name: str,
    group: Any,
    encoding: StoreEncoding,
    attrs: dict[str, Any],
    n_variants: int,
    n_analyses: int,
) -> np.ndarray:
    """Reject a published group that cannot be decoded safely, and return its axis.

    The read seam is what #265's query path trusts; a group whose arrays,
    dtype, shapes or declared axis order disagree with its own attributes would
    decode to a plausible, wrong association rather than raise
    (CONTRIBUTING.md, "a wrong answer that looks like a right answer").
    Validation checks all of this too, but a query must not depend on someone
    having run `validate` first.  The checks here are the cheap, structural,
    validation-visible ones; deliberately not among them is a full decoded-value
    or side-table comparison against the primary planes, which is what makes
    `validate` expensive and is not a per-query obligation.  An in-range shift
    of a Variant Index is likewise not detectable here: identity and bounds hold
    and nothing cheap distinguishes it from the real axis.
    """
    _require_declared_metadata(name, attrs)
    _require_declared_counts(name, attrs, n_analyses)
    _require_expected_arrays(name, group, encoding)
    variant_index = _require_variant_axis(name, group, attrs, n_variants)
    _require_plane_shapes(name, group, encoding, attrs, len(variant_index))
    return variant_index


def open_indexed_subset(store_path: str | Path, subset_name: str) -> IndexedSubset:
    """Open a published subset's metadata and decode seam, or fail loudly.

    The seam refuses a subset whose recorded source identity no longer matches
    the release it sits in, before returning any object a caller could decode
    through: a stale index would return plausible associations for the wrong
    release, and #265's query path must not be the first place that is noticed.

    Every failure names the Store and the subset: the caller asked for one
    release's index, and "this release has no subset" without the release is
    not actionable (issue #265 review).
    """
    store_path = Path(store_path)
    try:
        return _open_indexed_subset(store_path, subset_name)
    except IndexedSubsetError as exc:
        scoped = _store_scoped(store_path, exc)
        if scoped is exc:
            raise
        raise scoped from exc


def _store_scoped(store_path: Path, exc: IndexedSubsetError) -> IndexedSubsetError:
    """Prefix a subset failure with the Store it belongs to, at most once."""
    prefix = f"store {store_path}: "
    if str(exc).startswith(prefix):
        return exc
    return type(exc)(f"{prefix}{exc}")


def _open_subset_group(path: Path, name: str) -> tuple[Any, dict[str, Any]]:
    """Open a published subset's Zarr group, normalising a corrupt store entry.

    `path` is a directory the namespace layout selected, but it may be a plain
    directory, a Zarr array, or a group whose metadata is unreadable.  Those are
    corrupt-index failures the caller must see as an IndexedSubsetError, not as
    a zarr/json exception leaking through the query facade (#265 review).  Only
    the open and metadata read sit inside this boundary; a programming error in
    this module is deliberately not caught.
    """
    try:
        group = store_arrays.open_group(path)
        attrs = dict(group.attrs)
    except (zarr.errors.BaseZarrError, json.JSONDecodeError) as exc:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} at {path} is not a readable Zarr "
            f"group: {exc}"
        ) from exc
    return group, attrs


def _open_indexed_subset(store_path: Path, subset_name: str) -> IndexedSubset:
    name = parse_indexed_subset_name(subset_name)
    path = store_path / "data.zarr" / INDEXED_SUBSETS_GROUP / name
    if not path.is_dir():
        raise IndexedSubsetError(f"this release has no Indexed Variant Subset {name!r}")
    group, attrs = _open_subset_group(path, name)
    missing = _missing_required_attrs(attrs)
    if missing:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} is missing attributes: {', '.join(missing)}"
        )
    errors: list[str] = []
    encoding = _parse_subset_encoding(name, attrs, errors)
    if encoding is None:
        raise IndexedSubsetError(errors[0])
    release = open_store(store_path)
    _refuse_stale_subset(name, attrs, encoding, release.manifest)
    # Bound the subset against the release's own axis: an out-of-range Variant
    # Index would name a variant the Store does not have, and `resolve_rows`
    # would index the Store Variant Table out of bounds.  The Analysis width is
    # read from the same primary plane so a wrong-width index is refused too.
    primary_z = release.arrays(mode="r")["z"]
    variant_index = _require_decodable_subset(
        name,
        group,
        encoding,
        attrs,
        int(primary_z.shape[0]),
        int(primary_z.shape[1]),
    )
    return IndexedSubset(
        store_path=store_path,
        name=name,
        path=path,
        variant_index=variant_index,
        n_analyses=_require_attr_int(name, attrs, ATTR_N_ANALYSES),
        n_subset_variants=_require_attr_int(name, attrs, ATTR_N_SUBSET_VARIANTS),
        encoding=encoding,
        reference_assembly=str(attrs[ATTR_REFERENCE_ASSEMBLY]),
        source_store_id=str(attrs[ATTR_SOURCE_STORE_ID]),
        source_release_id=str(attrs[ATTR_SOURCE_RELEASE_ID]),
        source_format_version=str(attrs[ATTR_SOURCE_FORMAT_VERSION]),
        input_sha256=str(attrs[ATTR_INPUT_SHA256]),
        requested_count=_require_attr_int(name, attrs, ATTR_REQUESTED_COUNT),
        resolved_count=_require_attr_int(name, attrs, ATTR_RESOLVED_COUNT),
        absent_count=_require_attr_int(name, attrs, ATTR_ABSENT_COUNT),
        builder_version=str(attrs[ATTR_BUILDER_VERSION]),
        created_at=str(attrs[ATTR_CREATED_AT]),
        has_eaf="eaf" in group,
        n_z_overflow=_side_table_length(group, Z_OVERFLOW_INDEX),
        n_eaf_exceptions=_side_table_length(group, EAF_EXCEPTION_INDEX),
        n_se_exceptions=_side_table_length(group, _SE_EXCEPTION_INDEX),
        _group=group,
    )


def _codec(group: Any, encoding: StoreEncoding) -> StoreCodec:
    return StoreCodec(
        encoding,
        z_overflow=ZOverflowTable.read(group),
        eaf_exceptions=EafExceptionTable.read(group),
        se_exceptions=SeExceptionTable.read(group),
    )


def _decode_analysis_column(
    group: Any,
    encoding: StoreEncoding,
    n_subset: int,
    analysis_index: int,
    variant_index: np.ndarray,
) -> IndexedSubsetAnalysis:
    """Decode one Analysis's column from `group`.

    `variant_index` is passed in rather than re-read from the group: the read
    seam already holds it (`open_indexed_subset` validated it), so a query must
    not pay for a second full read of the subset's axis chunk (#265 review).
    """
    codec = _codec(group, encoding)
    start = analysis_index * n_subset
    positions = positions_flat(start)
    z = codec.decode_z(np.asarray(group["z"][analysis_index, :]), positions=positions)
    eaf = None
    if "eaf" in group:
        baseline = (
            np.asarray(group[EAF_BASELINE][:], dtype=np.float32)
            if EAF_BASELINE in group
            else None
        )
        eaf = codec.decode_eaf(
            np.asarray(group["eaf"][analysis_index, :]),
            baseline=baseline,
            positions=positions,
        )
    se_raw = np.asarray(group["se"][analysis_index, :])
    se = _decode_se_column(codec, encoding, se_raw, eaf, group, n_subset, analysis_index, positions)
    return IndexedSubsetAnalysis(
        variant_index=variant_index,
        z=z,
        se=se,
        eaf=eaf,
    )


def _decode_se_column(
    codec: StoreCodec,
    encoding: StoreEncoding,
    se_raw: np.ndarray,
    eaf: np.ndarray | None,
    group: Any,
    n_subset: int,
    analysis_index: int,
    positions: Any,
) -> np.ndarray:
    if not encoding.se.is_residual:
        return codec.decode_se(
            se_raw,
            eaf=np.empty(0, dtype=np.float32),
            analysis_index=np.empty(0, dtype=np.int64),
            coefficients=np.empty((0, 2), dtype=np.float32),
            positions=positions,
        )
    return codec.decode_se(
        se_raw,
        eaf=eaf if eaf is not None else np.empty(n_subset, dtype=np.float32),
        analysis_index=np.full(n_subset, analysis_index, dtype=np.int64),
        coefficients=np.asarray(group[SE_COEFFICIENTS][:], dtype=np.float32),
        positions=positions,
    )


# ── Validation seam ─────────────────────────────────────────────────────────


def _expected_arrays(encoding: StoreEncoding) -> frozenset[str]:
    """Every Zarr array a subset under `encoding` must carry, and no others."""
    names = {"variant_index", "z", "se"}
    if not encoding.eaf.is_absent:
        names.add("eaf")
    if encoding.z.is_fixed_point:
        names |= {Z_OVERFLOW_INDEX, Z_OVERFLOW_VALUE}
    if encoding.se.is_residual:
        names |= {_SE_EXCEPTION_INDEX, _SE_EXCEPTION_VALUE, SE_COEFFICIENTS}
    if encoding.eaf.is_residual:
        names |= {EAF_BASELINE, EAF_EXCEPTION_INDEX, EAF_EXCEPTION_VALUE}
    return frozenset(names)


@dataclass(frozen=True)
class _SubsetValidation:
    """The release-level facts every per-subset validation rule needs."""

    store_path: Path
    root: Any
    manifest: StoreManifest
    n_variants: int
    n_analyses: int

    @property
    def namespace_dir(self) -> Path:
        return self.store_path / "data.zarr" / INDEXED_SUBSETS_GROUP


def validate_indexed_subsets(
    store_path: str | Path,
    root: Any,
    manifest: StoreManifest,
    n_variants: int,
    n_analyses: int,
    errors: list[str],
) -> None:
    """Validate the optional `data.zarr/indexed_subsets` namespace, if present.

    An Observed-Only Dense release with no namespace remains valid and
    unchanged.  When the namespace is present every entry must be a published
    subset: a staging/replacement directory, an unknown name, a file or a group
    whose decoded values disagree with the primary planes all fail rather than
    being ignored.
    """
    context = _SubsetValidation(Path(store_path), root, manifest, n_variants, n_analyses)
    if INDEXED_SUBSETS_GROUP not in root:
        _report_absent_namespace(context.namespace_dir, errors)
        return
    namespace = root[INDEXED_SUBSETS_GROUP]
    if not isinstance(namespace, zarr.Group):
        errors.append(f"data.zarr/{INDEXED_SUBSETS_GROUP} is not a group")
        return
    for entry in _namespace_entries(context.namespace_dir):
        _validate_namespace_entry(context, namespace, entry, errors)


def _report_absent_namespace(namespace_dir: Path, errors: list[str]) -> None:
    """A directory with no Zarr group behind it is a malformed namespace."""
    if namespace_dir.exists():
        errors.append(
            f"{namespace_dir} exists but is not a Zarr group; the indexed-subset "
            "namespace is malformed"
        )


def _namespace_entries(namespace_dir: Path) -> list[str]:
    """Every entry in the namespace directory, excluding Zarr's own metadata."""
    if not namespace_dir.is_dir():
        return []
    return sorted(
        entry.name for entry in namespace_dir.iterdir() if entry.name not in _ZARR_METADATA_NAMES
    )


def _validate_namespace_entry(
    context: _SubsetValidation, namespace: Any, entry: str, errors: list[str]
) -> None:
    """Judge one namespace entry and validate it when it is a published group."""
    if _is_staging_name(entry):
        errors.append(
            f"temporary indexed-subset entry {entry!r} is not a published index; a "
            "staging or replacement directory left by an interrupted build must be removed"
        )
        return
    if not is_valid_indexed_subset_name(entry):
        errors.append(
            f"unexpected entry {entry!r} in the indexed-subset namespace; a subset name "
            "must start with a letter or digit and contain only letters, digits, '.', '_' "
            "and '-'"
        )
        return
    member = _namespace_member(namespace, entry, errors)
    if member is None:
        return
    if not isinstance(member, zarr.Group):
        errors.append(f"indexed-subset entry {entry!r} is not a group")
        return
    _validate_one_indexed_subset(context, entry, member, errors)


def _namespace_member(namespace: Any, entry: str, errors: list[str]) -> Any:
    """Open one namespace member, or report that it is not a Zarr group."""
    try:
        return namespace[entry]
    except Exception:
        errors.append(
            f"indexed-subset entry {entry!r} is not a Zarr group; the namespace holds "
            "only published subset groups"
        )
        return None


def _array_keys(group: Any) -> list[str]:
    """The array members of a subset group."""
    return [key for key in group.keys() if isinstance(group[key], zarr.Array)]


def _subset_preconditions_ok(
    name: str, group: Any, attrs: dict[str, Any], errors: list[str]
) -> bool:
    """The group is non-empty and records the attributes every rule reads."""
    if not _array_keys(group):
        errors.append(
            f"indexed subset {name!r} is an explicit empty group with no arrays; a "
            "published subset must carry at least its variant_index and statistic planes"
        )
        return False
    missing = _missing_required_attrs(attrs)
    if missing:
        errors.append(
            f"indexed subset {name!r} is missing attributes: {', '.join(missing)}"
        )
        return False
    return True


def _validate_one_indexed_subset(
    context: _SubsetValidation, name: str, group: Any, errors: list[str]
) -> None:
    """Every rule one published Indexed Variant Subset must satisfy."""
    attrs = dict(group.attrs)
    if not _subset_preconditions_ok(name, group, attrs, errors):
        return
    declared = _read_declared_encoding(name, attrs, context.manifest, errors)
    if declared is None:
        return
    _check_identity_attrs(name, attrs, context.manifest, context.n_analyses, errors)
    if errors:
        return
    if not _check_subset_arrays(name, group, declared, errors):
        return
    n_subset = _check_variant_index(name, group, attrs, context.n_variants, errors)
    if n_subset is None:
        return
    if not _check_plane_shapes(name, group, declared, context.n_analyses, n_subset, errors):
        return
    _validate_subset_side_tables(name, group, declared, n_subset, context.n_analyses, errors)
    if errors:
        return
    subset = np.asarray(group["variant_index"][:], dtype=np.int64)
    _compare_indexed_values(name, context.root, group, declared, subset, context.n_analyses, errors)


def _read_declared_encoding(
    name: str, attrs: dict[str, Any], manifest: StoreManifest, errors: list[str]
) -> StoreEncoding | None:
    """The schema, profile and encoding a group declares, or None with findings."""
    if int(attrs[ATTR_SCHEMA]) != INDEXED_SUBSET_SCHEMA_VERSION:
        errors.append(
            f"indexed subset {name!r} declares schema {attrs[ATTR_SCHEMA]!r}, not "
            f"{INDEXED_SUBSET_SCHEMA_VERSION}"
        )
        return None
    if str(attrs[ATTR_PROFILE]) != FULL_STATISTIC_PROFILE:
        errors.append(
            f"indexed subset {name!r} declares statistic_profile "
            f"{attrs[ATTR_PROFILE]!r}, not {FULL_STATISTIC_PROFILE!r}"
        )
    declared = _parse_subset_encoding(name, attrs, errors)
    if declared is None:
        return None
    if declared != manifest.encoding:
        errors.append(
            f"indexed subset {name!r} was built with an encoding that is not the "
            "release's; the index cannot be decoded against the authoritative planes"
        )
        return None
    return declared


def _source_identity_mismatches(
    name: str, attrs: dict[str, Any], manifest: StoreManifest
) -> list[str]:
    """Recorded source identity that no longer matches the release it sits in.

    Shared by validation (which reports every mismatch) and the read seam
    (which refuses to hand out a stale index at all), so a reader and the
    validator cannot disagree about what stale means (#264 review).
    """
    mismatches: list[str] = []
    if str(attrs[ATTR_SOURCE_RELEASE_ID]) != manifest.release_id:
        mismatches.append(
            f"indexed subset {name!r} was built from release "
            f"{attrs[ATTR_SOURCE_RELEASE_ID]!r}, but this release is "
            f"{manifest.release_id!r}; the index is stale"
        )
    if str(attrs[ATTR_SOURCE_STORE_ID]) != manifest.store_id:
        mismatches.append(
            f"indexed subset {name!r} was built from store "
            f"{attrs[ATTR_SOURCE_STORE_ID]!r}, but this release is {manifest.store_id!r}"
        )
    if str(attrs[ATTR_SOURCE_FORMAT_VERSION]) != manifest.format_version:
        mismatches.append(
            f"indexed subset {name!r} records source_format_version "
            f"{attrs[ATTR_SOURCE_FORMAT_VERSION]!r}, but this release is "
            f"{manifest.format_version!r}"
        )
    if str(attrs[ATTR_REFERENCE_ASSEMBLY]) != normalise_build(manifest.reference_assembly):
        mismatches.append(
            f"indexed subset {name!r} records reference_assembly "
            f"{attrs[ATTR_REFERENCE_ASSEMBLY]!r}, but this release is "
            f"{manifest.reference_assembly!r}"
        )
    return mismatches


def _check_identity_attrs(
    name: str,
    attrs: dict[str, Any],
    manifest: StoreManifest,
    n_analyses: int,
    errors: list[str],
) -> None:
    """The recorded name, source identity, Analysis count and counts add up."""
    if str(attrs[ATTR_NAME]) != name:
        errors.append(f"indexed subset {name!r} records its name as {attrs[ATTR_NAME]!r}")
    errors.extend(_source_identity_mismatches(name, attrs, manifest))
    if int(attrs[ATTR_N_ANALYSES]) != n_analyses:
        errors.append(
            f"indexed subset {name!r} declares {attrs[ATTR_N_ANALYSES]} Analyses but "
            f"analyses.tsv has {n_analyses}"
        )
    requested = int(attrs[ATTR_REQUESTED_COUNT])
    resolved = int(attrs[ATTR_RESOLVED_COUNT])
    absent = int(attrs[ATTR_ABSENT_COUNT])
    if requested != resolved + absent:
        errors.append(
            f"indexed subset {name!r} records requested={requested}, resolved={resolved}, "
            f"absent={absent}, which do not add up"
        )


def _non_array_keys(group: Any, keys: list[str]) -> list[str]:
    """Members that are groups, not arrays -- never a valid index array."""
    return sorted(key for key in keys if not isinstance(group[key], zarr.Array))


def _unknown_array_keys(group: Any, keys: list[str], expected: frozenset[str]) -> list[str]:
    """Array members this format does not define."""
    return sorted(
        key for key in keys if isinstance(group[key], zarr.Array) and key not in expected
    )


def _check_subset_arrays(
    name: str, group: Any, encoding: StoreEncoding, errors: list[str]
) -> bool:
    """The group's array members are exactly the ones its encoding defines."""
    expected = _expected_arrays(encoding)
    keys = list(group.keys())
    non_arrays = _non_array_keys(group, keys)
    unknown = _unknown_array_keys(group, keys, expected)
    missing = sorted(key for key in expected if key not in group)
    if non_arrays:
        errors.append(
            f"indexed subset {name!r} carries unexpected groups: {', '.join(non_arrays)}"
        )
    if unknown:
        errors.append(
            f"indexed subset {name!r} carries unexpected arrays: {', '.join(unknown)}"
        )
    if missing:
        errors.append(
            f"indexed subset {name!r} is missing required arrays: {', '.join(missing)}"
        )
    return not (non_arrays or unknown or missing)


def _check_variant_index_attrs(
    name: str, attrs: dict[str, Any], n_subset: int, errors: list[str]
) -> None:
    """The recorded subset-variant count and axis order match the array."""
    if int(attrs[ATTR_N_SUBSET_VARIANTS]) != n_subset:
        errors.append(
            f"indexed subset {name!r} declares {attrs[ATTR_N_SUBSET_VARIANTS]} subset "
            f"variants but its variant_index holds {n_subset}"
        )
    if str(attrs[ATTR_ORDER]) != _SUBSET_ORDER:
        errors.append(
            f"indexed subset {name!r} declares order {attrs[ATTR_ORDER]!r}, not "
            f"{_SUBSET_ORDER!r}"
        )


def _check_variant_index_values(
    name: str, variant_index: np.ndarray, n_variants: int, errors: list[str]
) -> None:
    """The Variant Indices are sorted ascending, unique and in bounds."""
    if len(variant_index) and np.any(variant_index[1:] <= variant_index[:-1]):
        errors.append(
            f"indexed subset {name!r} variant_index is not sorted ascending and unique"
        )
    if len(variant_index) and (
        int(variant_index.min()) < 0 or int(variant_index.max()) >= n_variants
    ):
        errors.append(
            f"indexed subset {name!r} variant_index is out of bounds [0, {n_variants})"
        )


def _check_variant_index(
    name: str, group: Any, attrs: dict[str, Any], n_variants: int, errors: list[str]
) -> int | None:
    """The subset's Variant Indices are sorted, unique and in bounds."""
    variant_index = np.asarray(group["variant_index"][:], dtype=np.int64)
    n_subset = len(variant_index)
    if n_subset == 0:
        errors.append(
            f"indexed subset {name!r} contains no variants; an Indexed Variant Subset "
            "must cover at least one Store variant"
        )
        return None
    _check_variant_index_attrs(name, attrs, n_subset, errors)
    _check_variant_index_values(name, variant_index, n_variants, errors)
    return None if errors else n_subset


def _check_plane_shapes(
    name: str,
    group: Any,
    encoding: StoreEncoding,
    n_analyses: int,
    n_subset: int,
    errors: list[str],
) -> bool:
    """Each plane spans ``(n_analyses, n_subset)`` in its declared dtype."""
    expected_shape = (n_analyses, n_subset)
    encodings: dict[str, Any] = {
        "z": encoding.z,
        "se": encoding.se,
        "eaf": encoding.eaf,
    }
    for plane, declared in encodings.items():
        if plane not in group:
            continue
        actual = tuple(int(size) for size in group[plane].shape)
        if actual != expected_shape:
            errors.append(
                f"indexed subset {name!r} {plane} shape {actual} does not match "
                f"{expected_shape}"
            )
        if str(group[plane].dtype) != declared.dtype:
            errors.append(
                f"indexed subset {name!r} {plane} has dtype {group[plane].dtype} but the "
                f"declared encoding is {declared.kind} ({declared.dtype})"
            )
    return not errors


def _validate_subset_side_tables(
    name: str,
    group: Any,
    encoding: StoreEncoding,
    n_subset: int,
    n_analyses: int,
    errors: list[str],
) -> None:
    n_cells = n_analyses * n_subset
    if encoding.z.is_fixed_point:
        _check_table(name, "z overflow", ZOverflowTable.read(group), n_cells, errors)
    if encoding.se.is_residual:
        coefficients = np.asarray(group[SE_COEFFICIENTS][:], dtype=np.float32)
        if coefficients.shape != (n_analyses, 2):
            errors.append(
                f"indexed subset {name!r} se_coefficients shape {coefficients.shape} is "
                f"not {(n_analyses, 2)}"
            )
        _check_table(name, "se exception", SeExceptionTable.read(group), n_cells, errors)
    if encoding.eaf.is_residual:
        baseline = np.asarray(group[EAF_BASELINE][:], dtype=np.float32)
        if len(baseline) != n_subset:
            errors.append(
                f"indexed subset {name!r} eaf_baseline has {len(baseline)} entries but "
                f"the subset has {n_subset} variants"
            )
        _check_table(name, "eaf exception", EafExceptionTable.read(group), n_cells, errors)


def _check_table(name: str, what: str, table: Any, n_cells: int, errors: list[str]) -> None:
    if len(table.index) != len(table.value):
        errors.append(
            f"indexed subset {name!r} {what} table has {len(table.index)} positions but "
            f"{len(table.value)} values"
        )
        return
    if len(table.index) and (table.index.min() < 0 or table.index.max() >= n_cells):
        errors.append(
            f"indexed subset {name!r} {what} table positions are outside [0, {n_cells})"
        )
    if len(table.index) > 1 and np.any(table.index[1:] <= table.index[:-1]):
        errors.append(f"indexed subset {name!r} {what} table is not sorted and unique")


@dataclass
class _DecodeContext:
    """The primary and indexed decoders one validation pass reuses."""

    codec: StoreCodec
    z_plane: DenseZPlane
    se_plane: DenseSePlane
    eaf_plane: DenseEafPlane | None
    coefficients: np.ndarray
    baseline: np.ndarray | None


def _open_decode_context(
    name: str, group: Any, root: Any, encoding: StoreEncoding, errors: list[str]
) -> _DecodeContext | None:
    """Open every decoder, or report that the index cannot be decoded at all."""
    try:
        codec = _codec(group, encoding)
        z_plane = DenseZPlane.open(root, encoding)
        se_plane = DenseSePlane.open(root, encoding)
        eaf_plane = DenseEafPlane.open(root, encoding) if "eaf" in group else None
    except Exception as exc:
        errors.append(f"indexed subset {name!r} cannot be decoded: {exc}")
        return None
    coefficients = (
        np.asarray(group[SE_COEFFICIENTS][:], dtype=np.float32)
        if encoding.se.is_residual
        else np.empty((0, 2), dtype=np.float32)
    )
    baseline = (
        np.asarray(group[EAF_BASELINE][:], dtype=np.float32)
        if encoding.eaf.is_residual
        else None
    )
    return _DecodeContext(codec, z_plane, se_plane, eaf_plane, coefficients, baseline)


def _compare_indexed_values(
    name: str,
    root: Any,
    group: Any,
    encoding: StoreEncoding,
    subset: np.ndarray,
    n_analyses: int,
    errors: list[str],
) -> None:
    """Stream decoded index bands and compare them to the primary planes.

    Every decoded indexed Z, SE and EAF must equal its authoritative
    primary-plane cell; missingness must agree within the index before the
    values are compared.
    """
    context = _open_decode_context(name, group, root, encoding, errors)
    if context is None:
        return
    for start, stop in _band_bounds(len(subset), n_analyses, DEFAULT_BAND_CELLS):
        message = _compare_indexed_band(
            name, context, group, encoding, subset, start, stop, n_analyses
        )
        if message is not None:
            errors.append(message)
            return


def _compare_indexed_band(
    name: str,
    context: _DecodeContext,
    group: Any,
    encoding: StoreEncoding,
    subset: np.ndarray,
    start: int,
    stop: int,
    n_analyses: int,
) -> str | None:
    """One band's decoded equality checks; the first failure's message, or None."""
    n_subset = len(subset)
    rows = subset[start:stop]
    positions = positions_rows_cols(np.arange(n_analyses), np.arange(start, stop), n_subset)
    index_z_raw = np.asarray(group["z"][:, start:stop])
    index_se_raw = np.asarray(group["se"][:, start:stop])
    primary_eaf = (
        context.eaf_plane.read_rows(rows).values if context.eaf_plane is not None else None
    )
    primary_z = context.z_plane.rows(rows)
    primary_se = context.se_plane.rows(rows, eaf=primary_eaf)
    index_eaf, eaf_error = _decode_index_eaf(
        name, context, group, encoding, start, stop, n_analyses
    )
    if eaf_error is not None:
        return eaf_error
    return (
        _missingness_error(name, context, index_z_raw, index_se_raw, encoding)
        or _z_band_error(name, context, index_z_raw, primary_z, positions)
        or _eaf_band_error(name, index_eaf, primary_eaf)
        or _se_band_error(
            name,
            context,
            index_se_raw,
            index_eaf,
            primary_se,
            encoding,
            n_analyses,
            positions,
        )
    )


def _decode_index_eaf(
    name: str,
    context: _DecodeContext,
    group: Any,
    encoding: StoreEncoding,
    start: int,
    stop: int,
    n_analyses: int,
) -> tuple[np.ndarray | None, str | None]:
    """Decode one band's indexed EAF, with an error string when it cannot be."""
    if context.eaf_plane is None:
        return None, None
    n_subset = int(group["z"].shape[1])
    baseline_cells = (
        None
        if context.baseline is None
        else context.baseline[start:stop][None, :].repeat(n_analyses, axis=0)
    )
    positions = positions_rows_cols(np.arange(n_analyses), np.arange(start, stop), n_subset)
    try:
        decoded = context.codec.decode_eaf(
            np.asarray(group["eaf"][:, start:stop]),
            baseline=baseline_cells,
            positions=positions,
        )
    except Exception as exc:
        return None, f"indexed subset {name!r} eaf cannot be decoded: {exc}"
    return decoded, None


def _missingness_error(
    name: str,
    context: _DecodeContext,
    index_z_raw: np.ndarray,
    index_se_raw: np.ndarray,
    encoding: StoreEncoding,
) -> str | None:
    """Whether the index's own Z and SE missing markers agree, cell for cell."""
    se_missing = (
        index_se_raw == SE_MISSING
        if encoding.se.is_residual
        else np.isnan(index_se_raw)
    )
    if not np.array_equal(context.codec.missing_mask(index_z_raw), se_missing):
        return f"indexed subset {name!r} Z/SE missingness disagrees within the index"
    return None


def _z_band_error(
    name: str,
    context: _DecodeContext,
    index_z_raw: np.ndarray,
    primary_z: np.ndarray,
    positions: Any,
) -> str | None:
    try:
        index_z = context.codec.decode_z(index_z_raw, positions=positions)
    except Exception as exc:
        return f"indexed subset {name!r} z cannot be decoded: {exc}"
    if not np.array_equal(index_z.T, primary_z, equal_nan=True):
        return (
            f"indexed subset {name!r} z values differ from the primary plane; the index "
            "is stale or corrupt"
        )
    return None


def _eaf_band_error(
    name: str, index_eaf: np.ndarray | None, primary_eaf: np.ndarray | None
) -> str | None:
    if index_eaf is None:
        return None
    if primary_eaf is None or not np.array_equal(index_eaf.T, primary_eaf, equal_nan=True):
        return (
            f"indexed subset {name!r} eaf values differ from the primary plane; the index "
            "is stale or corrupt"
        )
    return None


def _se_band_error(
    name: str,
    context: _DecodeContext,
    index_se_raw: np.ndarray,
    index_eaf: np.ndarray | None,
    primary_se: np.ndarray,
    encoding: StoreEncoding,
    n_analyses: int,
    positions: Any,
) -> str | None:
    if encoding.se.is_residual:
        if index_eaf is None:
            return f"indexed subset {name!r} se cannot be decoded without its eaf plane"
        index_se = context.codec.decode_se(
            index_se_raw,
            eaf=index_eaf,
            analysis_index=np.broadcast_to(
                np.arange(n_analyses)[:, None], index_se_raw.shape
            ),
            coefficients=context.coefficients,
            positions=positions,
        )
    else:
        index_se = context.codec.decode_se(
            index_se_raw,
            eaf=np.empty(0, dtype=np.float32),
            analysis_index=np.empty(0, dtype=np.int64),
            coefficients=np.empty((0, 2), dtype=np.float32),
            positions=positions,
        )
    if not np.array_equal(index_se.T, primary_se, equal_nan=True):
        return (
            f"indexed subset {name!r} se values differ from the primary plane or cannot "
            "be decoded; the index is stale or corrupt"
        )
    return None
