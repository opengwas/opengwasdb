"""Indexed Variant Subsets for Dense stores (ADR 0061, issue #264).

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
   ADR 0061 asks validation to catch.  Copying the codes and remapping the
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
module's seam and applies the ordinary selected-Analysis result contract to it.

A Reference-Completed release (#266) adds two dependencies the Observed-Only
index does not need, and neither may be inferred from another Analysis:

4. **The imputed mask is copied per cell, Analysis-major.**  Association Status
   is per cell, not per Analysis: one Analysis completed against the panel can
   be imputed at a variant where another was never completed, and an observed
   cell keeps `observed` even inside a completed Analysis.  A reader that
   inferred imputation from an Analysis-level flag would label an observed cell
   imputed, or an imputed cell observed (#266).
5. **The subset's `eaf_reference` is the primary per-variant panel frequency
   restricted to the subset's variants.**  Decoding an imputed cell substitutes
   it through the same codec the primary plane uses, so an imputed cell reads
   the panel's value and an observed cell whose source reported none stays NaN
   -- never the panel's (ADR 0037 §4).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import os
import re
import shutil
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import zarr

from opengwasdb.build.liftover import normalise_build
from opengwasdb.encoding import (
    EAF_BASELINE,
    EAF_EXCEPTION_INDEX,
    EAF_EXCEPTION_VALUE,
    EAF_REFERENCE,
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
    positions_at,
    positions_flat,
    positions_rows_cols,
)
from opengwasdb.encoding.planes import SE_COEFFICIENTS
from opengwasdb.layouts.hybrid.layout import dense_component_path
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

#: Target band cell count for the build **writer**.  The variant band is the
#: index's physical write unit (a v2 chunk or a v3 shard) and this budget sizes
#: the Analysis tile within it.  The default is exactly one v2 output chunk times
#: one representative primary Analysis read unit (65,536 x 1,000 = 65,536,000
#: cells), so a v2 build decodes each primary file once rather than once per
#: tile.  A smaller budget is honoured: the writer re-reads the source instead
#: of materialising more than ``max(band_cells, write_unit)`` cells.  This is a
#: writer-source-block input, not an RSS acceptance target.  The CLI exposes it
#: as `--band-cells`.
DEFAULT_BAND_CELLS = 65_536_000

#: Target band cell count for the standalone **exhaustive** validation pass.
#: It decodes and compares float32 planes, so its band is kept at the historical
#: 4,000,000 cells and is deliberately independent of the writer default: a
#: writer that reads 1,000-Analysis source blocks must not make the validation
#: pass hold 65.5M decoded cells.  The build's own publication gate is sampled
#: and does not use this.
EXHAUSTIVE_VALIDATION_BAND_CELLS = 4_000_000

#: The build's publication gate validates the structural invariants exhaustively
#: and a bounded deterministic sample of decoded cells.  Standalone `validate`
#: uses the exhaustive mode below, which decodes every cell (#270).
VALIDATION_MODE_SAMPLED = "sampled"
VALIDATION_MODE_EXHAUSTIVE = "exhaustive"

#: The build-time decoded sample's fixed budget.  It is small and independent of
#: the matrix: a safety net against systemic, ordering and output-geometry
#: corruption, not a substitute for the exhaustive standalone audit (#270).
SAMPLE_ANALYSES = 16
SAMPLE_RANDOM_PER_ANALYSIS = 8
#: Physical output write-unit boundaries (a v2 chunk or a v3 shard) and a
#: separate, smaller inner-chunk spread, so a v3 index whose whole subset is one
#: shard still samples its inner geometry.
SAMPLE_SHARD_ANCHORS = 8
SAMPLE_INNER_ANCHORS = 4
SAMPLE_GLOBAL_RANDOM = 32

#: The one attribute schema this module writes and reads.  A future layout
#: change moves it rather than adding a field a reader would have to guess at.
INDEXED_SUBSET_SCHEMA_VERSION = 1

#: The statistic profile every subset carries.  There is deliberately no Z-only
#: mode (ADR 0061), so this is a constant rather than a choice.
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

#: The Association Status mask's array name.  It matches the primary Dense
#: grid's `imputed` array so the two decoders read the same vocabulary.
_IMPUTED_MASK = "imputed"

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

    Indexed Variant Subsets are Dense-only (ADR 0061, #264): a Ragged release is
    already a direct per-Analysis CSR and a Hybrid one has no rule for unifying
    its two components.  A selector naming a subset on either is a caller error,
    not a reason to answer from the ordinary path (#265).
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
    """One Analysis's decoded subset column, in Store Variant Index order.

    ``imputed`` is the per-cell Association Status mask, or ``None`` on an
    Observed-Only index that carries none.  ``eaf`` is ``None`` only when the
    release holds neither an `eaf` plane nor reference EAF; a completed release
    carrying reference EAF yields an array even with no `eaf` plane, because an
    imputed cell there has the panel's frequency (issue #113).
    """

    variant_index: np.ndarray
    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray | None
    imputed: np.ndarray | None


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
    has_imputed: bool
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
            self.store_path,
            self.name,
            self._group,
            self.encoding,
            self.n_subset_variants,
            analysis_index,
            self.variant_index,
        )


@dataclass(frozen=True)
class IndexedSubsetBuild:
    """What one successful `build_indexed_subset` published.

    ``write_seconds`` and ``validate_seconds`` split the staged publish into the
    construction pass and the publication gate (#270).  ``validation_mode`` is
    the mode the gate used (``sampled`` for a build), and the sample fields name
    the deterministic decoded sample so a reader can reproduce it.
    """

    name: str
    path: Path
    n_analyses: int
    n_subset_variants: int
    requested_count: int
    resolved_count: int
    absent_count: int
    input_sha256: str
    write_seconds: float = 0.0
    validate_seconds: float = 0.0
    validation_mode: str = VALIDATION_MODE_EXHAUSTIVE
    sample_strategy: str = ""
    sample_seed: int = 0
    sample_cells: int = 0
    band_cells_effective: int = 0
    physical_write_unit: int = 0
    max_materialized_cells: int = 0
    max_materialized_bytes: int = 0
    materialized_plane_cells: Mapping[str, int] = field(default_factory=dict)
    materialized_plane_bytes: Mapping[str, int] = field(default_factory=dict)
    materialized_plane_dtypes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _WriteStats:
    """What one chunk-aligned write materialised, for the declared invariant.

    ``band_cells_effective`` is ``max(band_cells, write_unit)`` -- the bound the
    writer enforces -- and ``max_materialized_cells``/``_bytes`` are the largest
    source block it actually held, so a build records both the invariant and the
    measurement.  The per-plane cells, derived bytes and dtype are recorded so a
    reader can recompute each buffer from the other two (#270 review).
    """

    physical_write_unit: int
    band_cells_effective: int
    max_materialized_cells: int
    max_materialized_bytes: int
    materialized_plane_cells: Mapping[str, int]
    materialized_plane_bytes: Mapping[str, int]
    materialized_plane_dtypes: Mapping[str, str]


@dataclass(frozen=True)
class _SamplePlan:
    """One build's deterministic decoded sample and the facts that produced it."""

    analyses: np.ndarray
    slots: np.ndarray
    seed: int
    strategy: str

    @property
    def cells(self) -> int:
        return int(len(self.analyses))


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
    completed: bool
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


def indexed_subset_store_path(store_path: str | Path) -> Path:
    """The directory that owns `store_path`'s Indexed Variant Subsets.

    A Dense Store Release owns its own ``data.zarr/indexed_subsets``.  A Hybrid
    release stores them inside its self-contained Dense Component at
    ``<root>/dense`` (issue #271): the shared/overflow axis has no association
    planes of its own to index, and the nested component is already a valid
    Dense release the unchanged builder and validator operate on.  Ragged owns
    none.  An unreadable manifest is returned unchanged, so a caller still sees
    the missing-subset or unreadable-group failure it would have seen before
    this resolver rather than a manifest error.
    """
    path = Path(store_path)
    try:
        manifest = StoreManifest.load(path)
    except (OSError, ValueError, KeyError, TypeError):
        return path
    if manifest.primary_layout is PrimaryStorageLayout.HYBRID:
        return dense_component_path(path)
    return path


def indexed_subset_namespace_path(store_path: str | Path) -> Path:
    """The `indexed_subsets` directory for `store_path`, resolving a Hybrid root.

    The one place the physical namespace is located, so discovery, removal,
    overview and reading cannot disagree about where a Hybrid release keeps
    its index (issue #271).
    """
    return indexed_subset_store_path(store_path) / "data.zarr" / INDEXED_SUBSETS_GROUP


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


def _physical_write_unit(array: Any) -> int:
    """The variant-axis width at which one write covers one whole store file.

    A Zarr v3 array is sharded: one compressed file per shard, so a write that
    covers part of a shard is a read-modify-write of the whole shard (the
    partial-shard defect #249 fixed for Ragged).  A Zarr v2 array has one file
    per inner chunk.  The output band is therefore the shard where there is
    one, and the inner chunk otherwise; an index plane written on its inner
    chunk in a v3 store would re-encode every shard once per inner chunk.
    """
    shards = getattr(array, "shards", None)
    return max(1, int(shards[1])) if shards is not None else max(1, int(array.chunks[1]))


def _source_read_unit(array: Any) -> int:
    """The primary plane's Analysis-axis read unit, for source-aligned tiling.

    A tile that is a whole multiple of this decodes each primary chunk or shard
    once; a smaller tile re-decodes the same primary file for every tile.
    Reading is far cheaper than re-encoding a written chunk, but a band that
    ignored the source geometry would decode the source tens of times more
    than it must.  It is the same per-file geometry as the output write unit.
    """
    return _physical_write_unit(array)


def _output_write_bands(n_subset: int, write_unit: int) -> list[tuple[int, int]]:
    """Subset-variant bands aligned to the index's physical write unit.

    Each band is a whole store file (a v2 chunk or a v3 shard) along the variant
    axis, so the writer encodes every output file exactly once (#270).  A
    zero-variant subset has no bands; returning an empty list rather than a
    zero-stride ``range`` keeps a caller from crashing on an index that
    validation separately rejects as empty.
    """
    if n_subset <= 0:
        return []
    unit = max(1, int(write_unit))
    return [(start, min(start + unit, n_subset)) for start in range(0, n_subset, unit)]


def _analysis_tiles(
    n_analyses: int, write_unit: int, band_cells: int, read_unit: int
) -> list[tuple[int, int]]:
    """Analysis tiles that bound a write band's memory.

    A band of ``write_unit`` variants times a tile of Analyses must stay within
    ``band_cells``.  The tile is the largest whole multiple of the primary's
    Analysis-axis read unit that fits, but it never grows past the budget to
    reach that unit: when the budget is below one read unit the writer decodes
    the primary once per tile rather than materialising ``write_unit *
    read_unit`` cells.  The true bound is therefore ``max(band_cells,
    write_unit)`` -- one physical output unit when the budget is smaller than
    it -- independent of the subset's length.
    """
    if n_analyses <= 0:
        return []
    write_unit = max(1, int(write_unit))
    budget = max(1, int(band_cells) // write_unit)
    unit = max(1, min(int(read_unit), n_analyses))
    tile = (budget // unit) * unit if budget >= unit else budget
    tile = max(1, min(tile, n_analyses))
    return [(start, min(start + tile, n_analyses)) for start in range(0, n_analyses, tile)]


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
    group: Any,
    encoding: StoreEncoding,
    codec: StoreCodec,
    n_analyses: int,
    n_subset: int,
    *,
    completed: bool,
) -> tuple[Any, Any, Any | None, Any | None]:
    """Create the Analysis-major statistic planes, filled with each plane's marker.

    A Reference-Completed release additionally gains the `imputed` mask, which
    is per cell: Association Status is not an Analysis-level fact (#266).
    """
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
    imputed = None
    if completed:
        imputed = store_arrays.create_array(
            group,
            _IMPUTED_MASK,
            ArrayRole.INDEXED_SUBSET_PLANE,
            shape=(n_analyses, n_subset),
            dtype="uint8",
            fill_value=0,
            compressor=comp,
        )
    return z, se, eaf, imputed


def _copy_one_plane(
    target: Any, source: Any, subset: np.ndarray, n_analyses: int, band_cells: int
) -> tuple[int, str]:
    """Copy one primary plane's codes into the index, one output file at a time.

    Returns the measured maximum source cells held and the source dtype name, so
    the caller can derive the raw buffer bytes.  Raises if the budget is somehow
    exceeded, so the invariant cannot drift silently.
    """
    write_unit = _physical_write_unit(target)
    read_unit = _source_read_unit(source)
    bound = max(int(band_cells), write_unit)
    max_materialized = 0
    for start, stop in _output_write_bands(len(subset), write_unit):
        rows = subset[start:stop]
        for a0, a1 in _analysis_tiles(n_analyses, write_unit, band_cells, read_unit):
            cells = int(len(rows)) * (a1 - a0)
            if cells > bound:
                raise IndexedSubsetError(
                    f"Indexed Variant Subset writer materialised {cells} source cells in "
                    f"one band, above the declared bound of {bound} (band_cells="
                    f"{band_cells}, physical write unit={write_unit})"
                )
            max_materialized = max(max_materialized, cells)
            target[a0:a1, start:stop] = np.asarray(source.oindex[rows, a0:a1]).T
    return max_materialized, str(np.dtype(source.dtype))


def _copy_plane_bands(
    z_target: Any,
    se_target: Any,
    eaf_target: Any | None,
    imputed_target: Any | None,
    root: Any,
    subset: np.ndarray,
    n_analyses: int,
    band_cells: int,
) -> tuple[int, list[tuple[str, int, str]]]:
    """Copy every plane and return the largest block and each (plane, cells, dtype)."""
    planes: dict[str, tuple[Any, Any]] = {
        "z": (z_target, root["z"]),
        "se": (se_target, root["se"]),
    }
    if eaf_target is not None:
        planes["eaf"] = (eaf_target, root["eaf"])
    if imputed_target is not None:
        planes["imputed"] = (imputed_target, root[_IMPUTED_MASK])
    max_cells = 0
    buffers: list[tuple[str, int, str]] = []
    for name, (target, source) in planes.items():
        cells, dtype = _copy_one_plane(
            target, source, subset, n_analyses, band_cells
        )
        max_cells = max(max_cells, cells)
        buffers.append((name, cells, dtype))
    return max_cells, buffers


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
    if encoding.eaf.reference:
        # The panel frequency is per variant and identical for every Analysis
        # imputed there, so the subset's copy is the primary's restricted to the
        # subset's variants -- not one recomputed from the subset (#266).
        store_arrays.create_array(
            group,
            EAF_REFERENCE,
            ArrayRole.PER_VARIANT,
            data=np.asarray(root[EAF_REFERENCE].oindex[subset], dtype=np.float32),
            dtype="float32",
            compressor=store_arrays.compressor(),
            hint=store_arrays.INDEXED_SUBSET_CHUNK,
        )
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
    *,
    completed: bool,
) -> _WriteStats:
    """Write one complete Indexed Variant Subset group from the primary planes.

    Every array a published subset holds is created here; validation compares
    what this wrote against the primaries, and a caller never assembles the
    group by hand.  The planes are copied one output store file at a time,
    plane by plane, so each compressed chunk or shard is encoded once and peak
    memory is one band of one plane rather than the subset matrix (#270).

    Returns the invariant and the measured maximum source block for the build
    result: ``max_materialized <= max(band_cells, write_unit)``.

    ``completed`` is the release's Completion State, passed in rather than
    re-derived from the presence of a primary array: a Reference-Completed
    index must carry the imputed mask, and an Observed-Only one must not.
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
    z_target, se_target, eaf_target, imputed_target = _create_subset_arrays(
        group, encoding, codec, n_analyses, n_subset, completed=completed
    )
    max_materialized, buffers = _copy_plane_bands(
        z_target,
        se_target,
        eaf_target,
        imputed_target,
        root,
        subset,
        n_analyses,
        band_cells,
    )
    _write_z_side_tables(group, root, encoding, subset, n_analyses)
    _write_eaf_side_tables(group, root, encoding, subset, n_analyses)
    _write_se_side_tables(group, root, encoding, subset, n_analyses)
    group.attrs.update(attrs)
    return _write_stats(z_target, band_cells, max_materialized, buffers)


def _write_stats(
    z_target: Any,
    band_cells: int,
    max_materialized: int,
    buffers: list[tuple[str, int, str]],
) -> _WriteStats:
    """The declared band invariant and the measured per-plane raw buffers."""
    write_unit = _physical_write_unit(z_target)
    cells = {name: count for name, count, _dtype in buffers}
    dtypes = {name: dtype for name, _count, dtype in buffers}
    byte_sizes = {
        name: count * np.dtype(dtype).itemsize for name, count, dtype in buffers
    }
    return _WriteStats(
        physical_write_unit=write_unit,
        band_cells_effective=max(int(band_cells), write_unit),
        max_materialized_cells=max_materialized,
        max_materialized_bytes=max(byte_sizes.values(), default=0),
        materialized_plane_cells=MappingProxyType(cells),
        materialized_plane_bytes=MappingProxyType(byte_sizes),
        materialized_plane_dtypes=MappingProxyType(dtypes),
    )


def _check_layout(manifest: StoreManifest, root: Any) -> None:
    """Refuse anything but a Dense Store Release an index can reproduce.

    Observed-Only and Reference-Completed Dense are both supported (#264,
    #266).  Ragged and Hybrid remain explicit failures (ADR 0061).  A
    Reference-Completed release must carry its per-cell `imputed` mask: without
    it the index cannot reproduce Association Status, so it is refused here
    rather than written incompletely.
    """
    if manifest.primary_layout is not PrimaryStorageLayout.DENSE:
        raise IndexedSubsetError(
            "Indexed Variant Subsets are Dense-only (ADR 0061); this release's "
            f"primary_layout is {manifest.primary_layout.value!r}"
        )
    if manifest.completion_state is CompletionState.OBSERVED_ONLY:
        if manifest.encoding.eaf.reference:
            raise IndexedSubsetError(
                "this Observed-Only release declares reference EAF, which an "
                "Observed-Only index does not carry; refusing rather than writing an "
                "index that cannot reproduce the release's ordinary result"
            )
        return
    if _IMPUTED_MASK not in root:
        raise IndexedSubsetError(
            "this Reference-Completed release carries no per-cell imputed mask; an "
            "Indexed Variant Subset cannot reproduce Association Status without it"
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


def _resolve_build_target(store_path: Path) -> tuple[Path, StoreManifest, Path | None]:
    """The release the index physically belongs to, and its Hybrid root if any.

    A Dense release is its own target.  A Hybrid release's target is its
    self-contained Dense Component (``<root>/dense``), returned with the root so
    the variant-list check can distinguish an overflow-only ALID from one truly
    absent from the shared union table (issue #271).  A Ragged release is
    returned unchanged and refused by `_check_layout` with the existing
    Dense-only message.
    """
    release = open_store(store_path)
    if release.manifest.primary_layout is PrimaryStorageLayout.HYBRID:
        dense_path = dense_component_path(store_path)
        try:
            dense_manifest = open_store(dense_path).manifest
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise IndexedSubsetError(
                f"Hybrid release {store_path} has no readable Dense Component at "
                f"{dense_path}: {exc}"
            ) from exc
        return dense_path, dense_manifest, store_path
    return store_path, release.manifest, None


def _matched_alids(axis: VariantAxis, variant_index: np.ndarray) -> set[str]:
    """The canonical ALIDs of `variant_index` in `axis`, without a table scan.

    `identity_by_indices` reads the mmap'd ALID index when the store carries
    one; a store built before issue 029 has none, so per-row `by_indices` is
    the fallback.
    """
    if len(variant_index) == 0:
        return set()
    identity = axis.identity_by_indices(np.asarray(variant_index, dtype=np.int64))
    if identity is not None:
        return {str(alid) for alid in identity["alid"]}
    return {record.alid for record in axis.by_indices(variant_index).values()}


def _off_panel_candidates(
    dense_path: Path, alids: tuple[str, ...], dense_resolved: ResolvedVariantList
) -> list[str]:
    """Requested ALIDs the Dense Component does not carry.

    A count mismatch between the resolved Variant Indices and the ALIDs read
    back from them means the two axes disagree about what a row is; refusing to
    guess keeps an on-panel ALID from being mislabelled overflow-only.
    """
    if dense_resolved.resolved == 0:
        return list(alids)
    dense_axis = VariantAxis(dense_path)
    try:
        dense_alids = _matched_alids(dense_axis, dense_resolved.variant_index)
    finally:
        dense_axis.close()
    if len(dense_alids) != int(dense_resolved.resolved):
        raise IndexedSubsetError(
            f"internal error resolving the variant list against Dense Component "
            f"{dense_path}: {dense_resolved.resolved} Variant Indices resolved but "
            f"{len(dense_alids)} distinct ALIDs were read back; refusing to guess "
            "which requested ALIDs are on-panel"
        )
    return [alid for alid in alids if alid not in dense_alids]


def _overflow_only_alids(hybrid_root: Path, candidates: list[str]) -> list[str]:
    """The candidates present in the shared union table -- off-panel, not absent.

    One batched pass short-circuits the common all-absent case; only when
    something is present does the per-ALID pass pay for names to report.
    """
    shared_axis = VariantAxis(hybrid_root)
    try:
        if len(shared_axis.indices_by_identifiers(candidates)) == 0:
            return []
        return [alid for alid in candidates if shared_axis.by_identifier(alid) is not None]
    finally:
        shared_axis.close()


def _refuse_overflow_only(
    hybrid_root: Path,
    dense_path: Path,
    alids: tuple[str, ...],
    dense_resolved: ResolvedVariantList,
) -> None:
    """Refuse requested ALIDs that exist only on a Hybrid release's overflow axis.

    An Indexed Variant Subset built from a Hybrid root indexes its Dense
    Component only (ADR 0061, issue #271).  An off-panel ALID is present in the
    shared union table but has no Dense row, so resolving against the Dense
    Component alone would count it as "absent from this Store Release" and drop
    it silently -- the plausible-wrong-answer class this project refuses.

    The shared union table makes the distinction observable, so the existing
    absent handling is preserved only for an ALID absent from *both* axes; one
    present in the shared table but not the Dense Component fails the build and
    is named.
    """
    candidates = _off_panel_candidates(dense_path, alids, dense_resolved)
    if not candidates:
        return
    overflow = _overflow_only_alids(hybrid_root, candidates)
    if not overflow:
        return
    preview = ", ".join(repr(alid) for alid in overflow[:5])
    suffix = f" (and {len(overflow) - 5} more)" if len(overflow) > 5 else ""
    raise IndexedSubsetError(
        f"{len(overflow)} requested canonical ALID(s) belong only to the shared/overflow "
        f"axis of Hybrid release {hybrid_root} and cannot be indexed by its Dense "
        f"Component's Indexed Variant Subset: {preview}{suffix}. An Indexed Variant "
        "Subset indexes the Dense Component's on-panel variants only; remove these "
        "ALIDs or query them through the ordinary hybrid association path"
    )


def _resolve_inputs(
    target_path: Path,
    manifest: StoreManifest,
    variant_list_path: str | Path,
    reference_assembly: str,
    hybrid_root: Path | None,
) -> tuple[VariantList, ResolvedVariantList]:
    """Check the assembly and resolve the list against the owning release.

    On a Hybrid root the list is resolved against the nested Dense Component and
    any requested ALID that belongs only to the shared/overflow axis is refused
    (issue #271); an ALID absent from the shared union table is still counted and
    recorded as absent, because the shared table makes that distinction
    observable.
    """
    _check_assembly(manifest, reference_assembly)
    variant_list = read_variant_list(variant_list_path)
    resolved = resolve_variant_list(target_path, variant_list.alids)
    if hybrid_root is not None:
        _refuse_overflow_only(hybrid_root, target_path, variant_list.alids, resolved)
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
    return variant_list, resolved


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
    target_path, manifest, hybrid_root = _resolve_build_target(store_path)
    variant_list, resolved = _resolve_inputs(
        target_path, manifest, variant_list_path, reference_assembly, hybrid_root
    )
    root = store_arrays.open_group(target_path / "data.zarr")
    _check_layout(manifest, root)
    n_analyses = int(root["z"].shape[1])
    n_variants = int(root["z"].shape[0])
    attrs = _build_attributes(
        name, manifest, variant_list, resolved, n_analyses, len(resolved.variant_index)
    )
    return _SubsetPlan(
        store_path=target_path,
        name=name,
        manifest=manifest,
        variant_list=variant_list,
        resolved=resolved,
        root=root,
        n_analyses=n_analyses,
        n_variants=n_variants,
        fmt=zarr_format_for_version(manifest.format_version, source=f"release at {target_path}"),
        completed=manifest.completion_state is CompletionState.REFERENCE_COMPLETED,
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
    write_seconds = 0.0
    validate_seconds = 0.0
    write_stats = _WriteStats(
        0,
        max(int(band_cells), 1),
        0,
        0,
        MappingProxyType({}),
        MappingProxyType({}),
        MappingProxyType({}),
    )
    try:
        with staged_named_group(dest, overwrite=overwrite) as work:
            staged = store_arrays.open_group_for_write(
                work, "w", zarr_format=plan.fmt
            )
            started = time.perf_counter()
            write_stats = write_indexed_subset_group(
                staged,
                plan.root,
                plan.manifest.encoding,
                plan.resolved.variant_index,
                plan.n_analyses,
                plan.attrs,
                band_cells,
                completed=plan.completed,
            )
            write_seconds = time.perf_counter() - started
            started = time.perf_counter()
            sample = _validate_staged(plan, work)
            validate_seconds = time.perf_counter() - started
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
        write_seconds=write_seconds,
        validate_seconds=validate_seconds,
        validation_mode=VALIDATION_MODE_SAMPLED,
        sample_strategy="" if sample is None else sample.strategy,
        sample_seed=0 if sample is None else sample.seed,
        sample_cells=0 if sample is None else sample.cells,
        band_cells_effective=write_stats.band_cells_effective,
        physical_write_unit=write_stats.physical_write_unit,
        max_materialized_cells=write_stats.max_materialized_cells,
        max_materialized_bytes=write_stats.max_materialized_bytes,
        materialized_plane_cells=dict(write_stats.materialized_plane_cells),
        materialized_plane_bytes=dict(write_stats.materialized_plane_bytes),
        materialized_plane_dtypes=dict(write_stats.materialized_plane_dtypes),
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

    `store_path` is a Dense release or a Hybrid release. For a Hybrid release
    the target is its self-contained Dense Component, so the index is stored at
    `<root>/dense/data.zarr/indexed_subsets/<name>` and covers only that
    component's on-panel variants; a requested ALID that exists only on the
    shared/overflow axis is refused (issue #271).

    The input contract (canonical ALIDs, an explicit matching Reference
    Assembly, a nonempty match) is enforced before any group is created; the
    build then stages a unique sibling, validates it against the primary planes,
    and publishes it under the namespace's advisory lock (ADR 0043).
    """
    plan = _prepare_subset(
        store_path, subset_name, variant_list_path, reference_assembly, band_cells
    )
    return _publish_subset(plan, overwrite=overwrite, band_cells=band_cells)


def _validate_staged(plan: _SubsetPlan, work: Path) -> _SamplePlan | None:
    """Validate the freshly written group before it may be published.

    The publication gate is **sampled**: the structural invariants are checked
    exhaustively and a bounded deterministic sample of decoded cells is
    compared.  The returned plan names that sample so the build result records
    it.  Standalone ``validate`` stays exhaustive (#270).
    """
    staged = store_arrays.open_group(work)
    errors: list[str] = []
    context = _SubsetValidation(
        plan.store_path,
        plan.root,
        plan.manifest,
        plan.n_variants,
        plan.n_analyses,
        plan.fmt,
    )
    sample = _validate_one_indexed_subset(
        context, plan.name, staged, errors, exhaustive=False
    )
    if errors:
        raise IndexedSubsetValidationError(
            f"staged Indexed Variant Subset {plan.name!r} failed validation and was not "
            "published: " + "; ".join(errors)
        )
    return sample


def remove_indexed_subset(store_path: str | Path, subset_name: str) -> bool:
    """Delete one published subset group, returning whether it existed.

    `store_path` may be a Dense release or a Hybrid root; a Hybrid root resolves
    to its nested Dense Component (issue #271).  Removing a derived index changes
    no authoritative data; a release with none
    is valid (ADR 0061).  The removal takes the namespace's publication lock
    and first renames the group aside, so it can never interleave with a commit
    of the same name and a reader never sees a half-deleted group: the published
    name disappears atomically and the bytes are then reclaimed.  The rename
    bypasses zarr's own delete guard, so a group under consolidated metadata is
    refused before it -- the record would keep listing the removed group and
    the next open would read it.
    """
    name = parse_indexed_subset_name(subset_name)
    store_path = Path(store_path)
    # A Hybrid root keeps its subsets under the nested Dense Component; resolve
    # once here so removal names the release root and not `<root>/dense` (#271).
    namespace_dir = indexed_subset_namespace_path(store_path)
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

    `store_path` may be a Dense release or a Hybrid root; a Hybrid root resolves
    to its nested Dense Component (issue #271).  Read from the filesystem so a
    stray entry that is not a group is visible to validation rather than
    silently absent from the listing.
    """
    namespace = indexed_subset_namespace_path(Path(store_path))
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


def _require_expected_arrays(
    name: str, group: Any, encoding: StoreEncoding, *, completed: bool
) -> None:
    """The group carries exactly the arrays its encoding and completion state define."""
    expected = _expected_arrays(encoding, completed=completed)
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
    """The subset's Variant Indices are int32, non-empty, sorted, unique and in bounds."""
    n_subset = _require_attr_int(name, attrs, ATTR_N_SUBSET_VARIANTS)
    dtype = str(group["variant_index"].dtype)
    if dtype != "int32":
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} variant_index has dtype {dtype}, not "
            "int32; a widened or floating-point axis could silently truncate or "
            "round a Store Variant Index"
        )
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
    name: str,
    group: Any,
    encoding: StoreEncoding,
    attrs: dict[str, Any],
    n_subset: int,
    *,
    completed: bool,
) -> None:
    """Each statistic plane spans ``(n_analyses, n_subset_variants)`` in its declared dtype."""
    expected_shape = (_require_attr_int(name, attrs, ATTR_N_ANALYSES), n_subset)
    encodings: dict[str, Any] = {"z": encoding.z, "se": encoding.se, "eaf": encoding.eaf}
    for plane, declared in encodings.items():
        _require_one_plane_shape(name, group, plane, declared, expected_shape)
    _require_status_dependency_shapes(
        name, group, encoding, expected_shape, n_subset, completed=completed
    )


def _plane_shape_findings(
    group: Any, plane: str, declared: Any, expected_shape: tuple[int, int]
) -> list[str]:
    """One plane's shape/dtype findings, unprefixed so both callers share them."""
    actual = tuple(int(size) for size in group[plane].shape)
    findings: list[str] = []
    if actual != expected_shape:
        findings.append(f"{plane} shape {actual} does not match {expected_shape}")
    if str(group[plane].dtype) != declared.dtype:
        findings.append(
            f"{plane} has dtype {group[plane].dtype} but the declared encoding is "
            f"{declared.kind} ({declared.dtype})"
        )
    return findings


def _require_one_plane_shape(
    name: str, group: Any, plane: str, declared: Any, expected_shape: tuple[int, int]
) -> None:
    """One statistic plane's shape and dtype, when the index carries it."""
    if plane not in group:
        return
    findings = _plane_shape_findings(group, plane, declared, expected_shape)
    if findings:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} " + "; ".join(findings)
        )


def _require_status_dependency_shapes(
    name: str,
    group: Any,
    encoding: StoreEncoding,
    expected_shape: tuple[int, int],
    n_subset: int,
    *,
    completed: bool,
) -> None:
    """The status mask and reference EAF the release's Completion State implies.

    The mask's dtype is checked here because it is read from the array's
    metadata: a float mask would be coerced to uint8 by a decoder and lose the
    very values that should have refused it (#266 review).  The mask's *domain*
    is checked per decoded column in `_read_imputed_mask_column`, and its
    content against the primary release by validation.
    """
    if completed:
        actual = tuple(int(size) for size in group[_IMPUTED_MASK].shape)
        if actual != expected_shape:
            raise IndexedSubsetError(
                f"Indexed Variant Subset {name!r} imputed shape {actual} does not match "
                f"{expected_shape}"
            )
        dtype = str(group[_IMPUTED_MASK].dtype)
        if dtype != "uint8":
            raise IndexedSubsetError(
                f"Indexed Variant Subset {name!r} imputed has dtype {dtype}, not uint8"
            )
    if encoding.eaf.reference and array_length(group[EAF_REFERENCE]) != n_subset:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} eaf_reference has "
            f"{array_length(group[EAF_REFERENCE])} entries but the subset has "
            f"{n_subset} variants"
        )


def _require_decodable_subset(
    name: str,
    group: Any,
    encoding: StoreEncoding,
    attrs: dict[str, Any],
    n_variants: int,
    n_analyses: int,
    zarr_format: int,
    *,
    completed: bool,
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

    The physical layout is checked too (#263 review): an inner chunk or shard
    that spans Analyses, or a side array off the format's layout, would answer a
    one-Analysis read by decoding another Analysis's cells.
    """
    _require_declared_metadata(name, attrs)
    _require_declared_counts(name, attrs, n_analyses)
    _require_expected_arrays(name, group, encoding, completed=completed)
    variant_index = _require_variant_axis(name, group, attrs, n_variants)
    _require_plane_shapes(
        name, group, encoding, attrs, len(variant_index), completed=completed
    )
    _require_array_layouts(name, group, zarr_format)
    return variant_index


def open_indexed_subset(store_path: str | Path, subset_name: str) -> IndexedSubset:
    """Open a published subset's metadata and decode seam, or fail loudly.

    `store_path` may be a Dense release or a Hybrid root; a Hybrid root resolves
    to its nested Dense Component, and every failure names the path the caller
    passed (issue #271).  The seam refuses a subset whose recorded source identity no longer matches
    the release it sits in, before returning any object a caller could decode
    through: a stale index would return plausible associations for the wrong
    release, and #265's query path must not be the first place that is noticed.

    Every failure names the Store and the subset: the caller asked for one
    release's index, and "this release has no subset" without the release is
    not actionable (issue #265 review).
    """
    requested = Path(store_path)
    # The index physically belongs to a Dense release or, for a Hybrid root, to
    # its nested Dense Component; failures name the path the caller passed so an
    # operator never has to know the nested path (issue #271).
    target = indexed_subset_store_path(requested)
    try:
        return _open_indexed_subset(target, subset_name)
    except IndexedSubsetError as exc:
        scoped = _store_scoped(requested, exc)
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
    directory, a Zarr array, or a group whose metadata is unreadable or invalid
    (bad JSON, an unsupported `zarr_format`, a non-object document).  Those are
    corrupt-index failures the caller must see as an IndexedSubsetError, not as
    a zarr/json/ValueError exception leaking through the query facade (#265
    review).  The catch is broad on purpose, matching validation's
    `_namespace_member`: the try body is only zarr's own open and attrs read, so
    there is no module logic whose bug it could hide.
    """
    try:
        group = store_arrays.open_group(path)
        attrs = dict(group.attrs)
    except Exception as exc:  # broad on purpose; see docstring, mirrors validation
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
    completed = release.manifest.completion_state is CompletionState.REFERENCE_COMPLETED
    variant_index = _require_decodable_subset(
        name,
        group,
        encoding,
        attrs,
        int(primary_z.shape[0]),
        int(primary_z.shape[1]),
        zarr_format_for_version(
            release.manifest.format_version, source=f"release at {store_path}"
        ),
        completed=completed,
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
        has_imputed=_IMPUTED_MASK in group,
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
    store_path: Path,
    name: str,
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

    The imputed mask and reference EAF are read from the index's own arrays and
    handed to the same codec the primary planes use (#266): an imputed cell
    reads the panel's frequency and an observed cell whose source reported none
    stays NaN, exactly as on the primary plane.  The mask column is validated
    before the codec sees it, so a float or out-of-domain value cannot be read
    as imputed by the frequency substitution and as observed by Association
    Status at the same time.
    """
    codec = _codec(group, encoding)
    start = analysis_index * n_subset
    positions = positions_flat(start)
    imputed = _read_imputed_mask_column(store_path, name, group, analysis_index)
    z = codec.decode_z(np.asarray(group["z"][analysis_index, :]), positions=positions)
    eaf = _decode_eaf_column(codec, group, n_subset, analysis_index, positions, imputed)
    se_raw = np.asarray(group["se"][analysis_index, :])
    se = _decode_se_column(codec, encoding, se_raw, eaf, group, n_subset, analysis_index, positions)
    return IndexedSubsetAnalysis(
        variant_index=variant_index,
        z=z,
        se=se,
        eaf=eaf,
        imputed=imputed,
    )


def _read_imputed_mask_column(
    store_path: Path, name: str, group: Any, analysis_index: int
) -> np.ndarray | None:
    """One Analysis's status mask, or None on an Observed-Only index.

    The mask is read without coercing its dtype: a float mask whose 0.5 cells
    are silently narrowed to 0 would be observed to Association Status and
    imputed to the frequency substitution, which is the plausible-wrong answer
    this refuses (#266 review).  The check is bounded to the column being
    decoded; `validate` compares the whole mask's content against the primary
    release.
    """
    if _IMPUTED_MASK not in group:
        return None
    raw = np.asarray(group[_IMPUTED_MASK][analysis_index, :])
    if str(raw.dtype) != "uint8":
        raise IndexedSubsetError(
            f"store {store_path}: Indexed Variant Subset {name!r} imputed mask has "
            f"dtype {raw.dtype}, not uint8"
        )
    invalid = (raw != 0) & (raw != 1)
    if np.any(invalid):
        examples = np.unique(raw[invalid]).tolist()[:3]
        raise IndexedSubsetError(
            f"store {store_path}: Indexed Variant Subset {name!r} imputed mask contains "
            f"values other than 0 and 1 (e.g. {examples}); its cells cannot be classified "
            "as observed or imputed"
        )
    return raw


def _index_reference(group: Any) -> np.ndarray | None:
    """The subset's per-variant reference frequency, or None when it carries none."""
    if EAF_REFERENCE not in group:
        return None
    return np.asarray(group[EAF_REFERENCE][:], dtype=np.float32)


def _decode_eaf_column(
    codec: StoreCodec,
    group: Any,
    n_subset: int,
    analysis_index: int,
    positions: Any,
    imputed: np.ndarray | None,
) -> np.ndarray | None:
    """One Analysis's decoded frequencies, with the panel substitution applied.

    A completed release may carry reference EAF with no `eaf` plane at all
    (issue #113); then every observed cell reads NaN and every imputed cell
    reads the panel's value.  Returning None is reserved for a release that
    holds no frequency anywhere.
    """
    reference = _index_reference(group)
    if "eaf" in group:
        baseline = (
            np.asarray(group[EAF_BASELINE][:], dtype=np.float32)
            if EAF_BASELINE in group
            else None
        )
        return codec.decode_eaf(
            np.asarray(group["eaf"][analysis_index, :]),
            baseline=baseline,
            positions=positions,
            imputed=imputed,
            reference=reference,
        )
    if reference is None:
        return None
    blank = np.full(n_subset, np.nan, dtype=np.float32)
    return codec.decode_eaf(blank, imputed=imputed, reference=reference)


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


def _expected_arrays(encoding: StoreEncoding, *, completed: bool) -> frozenset[str]:
    """Every Zarr array a subset under `encoding` must carry, and no others."""
    names = {"variant_index", "z", "se"}
    if not encoding.eaf.is_absent:
        names.add("eaf")
    if completed:
        names.add(_IMPUTED_MASK)
    if encoding.eaf.reference:
        names.add(EAF_REFERENCE)
    if encoding.z.is_fixed_point:
        names |= {Z_OVERFLOW_INDEX, Z_OVERFLOW_VALUE}
    if encoding.se.is_residual:
        names |= {_SE_EXCEPTION_INDEX, _SE_EXCEPTION_VALUE, SE_COEFFICIENTS}
    if encoding.eaf.is_residual:
        names |= {EAF_BASELINE, EAF_EXCEPTION_INDEX, EAF_EXCEPTION_VALUE}
    return frozenset(names)


#: The physical layout of every array a published subset carries: its
#: `ArrayRole` and, for the `PER_VARIANT` side arrays, the override hint the
#: writer passes.  The reader and the validator derive the expected inner chunk
#: and shard from this through the store-array seam's authoritative policies, so
#: neither trusts a recorded attribute nor reproduces a chunk sum by hand.  A
#: layout is part of the format, not a free choice: a plane chunk or shard that
#: spans Analyses defeats the one-Analysis narrow read the index exists for, and
#: an off-format side array is the same class of silent divergence (#263 review).
_INDEXED_SUBSET_ARRAY_LAYOUTS: Mapping[str, tuple[ArrayRole, int | None]] = (
    MappingProxyType(
        {
            "variant_index": (ArrayRole.INDEXED_SUBSET_VARIANT_INDEX, None),
            "z": (ArrayRole.INDEXED_SUBSET_PLANE, None),
            "se": (ArrayRole.INDEXED_SUBSET_PLANE, None),
            "eaf": (ArrayRole.INDEXED_SUBSET_PLANE, None),
            _IMPUTED_MASK: (ArrayRole.INDEXED_SUBSET_PLANE, None),
            EAF_REFERENCE: (ArrayRole.PER_VARIANT, store_arrays.INDEXED_SUBSET_CHUNK),
            EAF_BASELINE: (ArrayRole.PER_VARIANT, store_arrays.INDEXED_SUBSET_CHUNK),
            Z_OVERFLOW_INDEX: (ArrayRole.EXCEPTION_TABLE, None),
            Z_OVERFLOW_VALUE: (ArrayRole.EXCEPTION_TABLE, None),
            _SE_EXCEPTION_INDEX: (ArrayRole.EXCEPTION_TABLE, None),
            _SE_EXCEPTION_VALUE: (ArrayRole.EXCEPTION_TABLE, None),
            EAF_EXCEPTION_INDEX: (ArrayRole.EXCEPTION_TABLE, None),
            EAF_EXCEPTION_VALUE: (ArrayRole.EXCEPTION_TABLE, None),
            SE_COEFFICIENTS: (ArrayRole.SE_COEFFICIENTS, None),
        }
    )
)


def _inner_chunk_finding(
    name: str,
    role: ArrayRole,
    actual: tuple[int, ...],
    expected: tuple[int, ...],
) -> str:
    """The message for one array whose inner chunk is not the format's.

    A 2-D Analysis-major plane is read one Analysis row at a time, so its chunk
    has that specific reason; every other array's chunk is fixed by its role so
    it tiles the format's read unit.  The message says which, rather than
    claiming a one-Analysis-row rule for a 1-D side array, table or coefficient
    array.
    """
    if role is ArrayRole.INDEXED_SUBSET_PLANE:
        detail = (
            "an Indexed Variant Subset plane must be one Analysis row per inner chunk"
        )
    else:
        detail = "the inner chunk is fixed by this array's role in the format"
    return (
        f"{name} inner chunk {actual} does not match the format's {expected}; {detail}"
    )


def _shard_findings(
    name: str,
    array: Any,
    role: ArrayRole,
    shape: tuple[int, ...],
    expected_chunk: tuple[int, ...],
    zarr_format: int,
) -> list[str]:
    """One array's shard findings on a v3 release, or its unsharded v2 ones."""
    actual_shard = getattr(array, "shards", None)
    if zarr_format != 3:
        if actual_shard is None:
            return []
        return [f"{name} is sharded but this release's Zarr format is v2"]
    if actual_shard is None:
        return [
            f"{name} is not sharded; format 0.2.0 stores every array with the "
            "sharding codec"
        ]
    expected_shard = store_arrays.shard_layout(role, shape, inner_chunk=expected_chunk)
    actual = tuple(int(size) for size in actual_shard)
    if actual == expected_shard:
        return []
    detail = (
        "a shard must not span Analyses"
        if role is ArrayRole.INDEXED_SUBSET_PLANE
        else "the shard is fixed by this array's role in the format"
    )
    return [
        f"{name} shard {actual} does not match the format's {expected_shard}; {detail}"
    ]


def _array_layout_findings(group: Any, zarr_format: int) -> list[str]:
    """Every array whose inner chunk or shard is not the format's, unprefixed.

    The expected layout is derived from each array's role and shape through
    `chunk_layout`/`shard_layout`, the one table the writer and the converter
    both decide layouts from, so no attribute needs to record it.  On a 0.2.0
    (Zarr v3) release every array must carry a shard equal to the policy's;
    on a 0.1.0 (Zarr v2) release none may (sharding is 0.2.0's).
    """
    findings: list[str] = []
    for name, (role, hint) in _INDEXED_SUBSET_ARRAY_LAYOUTS.items():
        if name not in group:
            continue
        array = group[name]
        shape = tuple(int(size) for size in array.shape)
        expected_chunk = store_arrays.chunk_layout(role, shape, hint=hint)
        actual_chunk = tuple(int(size) for size in array.chunks)
        if actual_chunk != expected_chunk:
            findings.append(
                _inner_chunk_finding(name, role, actual_chunk, expected_chunk)
            )
            continue
        findings.extend(
            _shard_findings(name, array, role, shape, expected_chunk, zarr_format)
        )
    return findings


def _require_array_layouts(name: str, group: Any, zarr_format: int) -> None:
    """Refuse a subset whose arrays are not on the format's physical layout."""
    findings = _array_layout_findings(group, zarr_format)
    if findings:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} " + "; ".join(findings)
        )


def _check_array_layouts(
    name: str, group: Any, zarr_format: int, errors: list[str]
) -> None:
    """Append every off-format inner chunk or shard of one subset."""
    for finding in _array_layout_findings(group, zarr_format):
        errors.append(f"indexed subset {name!r} {finding}")


@dataclass(frozen=True)
class _SubsetValidation:
    """The release-level facts every per-subset validation rule needs."""

    store_path: Path
    root: Any
    manifest: StoreManifest
    n_variants: int
    n_analyses: int
    zarr_format: int

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
    *,
    exhaustive: bool = True,
) -> None:
    """Validate the optional `data.zarr/indexed_subsets` namespace, if present.

    A release with no `indexed_subsets` namespace remains valid and unchanged.
    When the namespace is present every entry must be a published
    subset: a staging/replacement directory, an unknown name, a file or a group
    whose decoded values disagree with the primary planes all fail rather than
    being ignored.

    ``exhaustive`` is the default and is what standalone `validate` uses: every
    indexed cell is decoded and compared.  The build's publication gate calls
    this family of rules with ``exhaustive=False``, which keeps every structural
    invariant exhaustive and compares a bounded deterministic decoded sample
    (#270).
    """
    context = _SubsetValidation(
        Path(store_path),
        root,
        manifest,
        n_variants,
        n_analyses,
        zarr_format_for_version(
            manifest.format_version, source=f"release at {store_path}"
        ),
    )
    if INDEXED_SUBSETS_GROUP not in root:
        _report_absent_namespace(context.namespace_dir, errors)
        return
    namespace = root[INDEXED_SUBSETS_GROUP]
    if not isinstance(namespace, zarr.Group):
        errors.append(f"data.zarr/{INDEXED_SUBSETS_GROUP} is not a group")
        return
    for entry in _namespace_entries(context.namespace_dir):
        _validate_namespace_entry(context, namespace, entry, errors, exhaustive=exhaustive)


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
    context: _SubsetValidation,
    namespace: Any,
    entry: str,
    errors: list[str],
    *,
    exhaustive: bool = True,
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
    _validate_one_indexed_subset(context, entry, member, errors, exhaustive=exhaustive)


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
    context: _SubsetValidation,
    name: str,
    group: Any,
    errors: list[str],
    *,
    exhaustive: bool,
) -> _SamplePlan | None:
    """Every rule one published Indexed Variant Subset must satisfy.

    The structural invariants are always exhaustive.  Decoded Z/SE/EAF and
    Association Status are compared for every cell when ``exhaustive`` is set
    (standalone `validate`) and over a bounded deterministic sample otherwise
    (the build's publication gate).  The sample plan is returned in the sampled
    case so the build result can record it (#270).
    """
    attrs = dict(group.attrs)
    prepared = _validate_subset_structure(context, name, group, attrs, errors)
    if prepared is None:
        return None
    declared, subset = prepared
    # Cheap, high-value invariants that stay exhaustive even on the sampled
    # build path: the side tables are small, so their remapping and the
    # per-variant baselines are compared in full rather than sampled.
    _check_side_tables_against_primary(
        name, group, context.root, declared, subset, context.n_analyses, errors
    )
    _eaf_reference_error(name, group, context.root, subset, declared, errors)
    if errors:
        return None
    if exhaustive:
        _compare_indexed_values(
            name, context.root, group, declared, subset, context.n_analyses, errors
        )
        return None
    sample = _validation_sample(
        attrs,
        context.n_analyses,
        len(subset),
        _physical_write_unit(group["z"]),
        int(group["z"].chunks[1]),
    )
    _compare_indexed_sample(name, context.root, group, declared, subset, sample, errors)
    return sample


def _validate_subset_structure(
    context: _SubsetValidation,
    name: str,
    group: Any,
    attrs: dict[str, Any],
    errors: list[str],
) -> tuple[StoreEncoding, np.ndarray] | None:
    """The exhaustive structural rules, and the resolved Variant Index.

    Returns the declared encoding and the sorted Variant Index, or None when a
    finding already rules the group out.
    """
    if not _subset_preconditions_ok(name, group, attrs, errors):
        return None
    declared = _read_declared_encoding(name, attrs, context.manifest, errors)
    if declared is None:
        return None
    completed = context.manifest.completion_state is CompletionState.REFERENCE_COMPLETED
    _check_identity_attrs(name, attrs, context.manifest, context.n_analyses, errors)
    if errors:
        return None
    if not _check_subset_arrays(name, group, declared, completed, errors):
        return None
    n_subset = _check_variant_index(name, group, attrs, context.n_variants, errors)
    if n_subset is None:
        return None
    if not _check_plane_shapes(
        name, group, declared, context.n_analyses, n_subset, completed, errors
    ):
        return None
    _check_array_layouts(name, group, context.zarr_format, errors)
    _validate_subset_side_tables(name, group, declared, n_subset, context.n_analyses, errors)
    if errors:
        return None
    subset = np.asarray(group["variant_index"][:], dtype=np.int64)
    return declared, subset


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
    name: str, group: Any, encoding: StoreEncoding, completed: bool, errors: list[str]
) -> bool:
    """The group's array members are exactly the ones its encoding defines."""
    expected = _expected_arrays(encoding, completed=completed)
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
    """The subset's Variant Indices are int32, sorted, unique and in bounds."""
    dtype = str(group["variant_index"].dtype)
    if dtype != "int32":
        errors.append(
            f"indexed subset {name!r} variant_index has dtype {dtype}, not int32"
        )
        return None
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
    completed: bool,
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
        _check_one_plane_shape(name, group, plane, declared, expected_shape, errors)
    if completed:
        _check_imputed_shape(name, group, expected_shape, errors)
    return not errors


def _check_one_plane_shape(
    name: str,
    group: Any,
    plane: str,
    declared: Any,
    expected_shape: tuple[int, int],
    errors: list[str],
) -> None:
    """One statistic plane's shape and dtype, when the index carries it."""
    if plane not in group:
        return
    for finding in _plane_shape_findings(group, plane, declared, expected_shape):
        errors.append(f"indexed subset {name!r} {finding}")


def _check_imputed_shape(
    name: str, group: Any, expected_shape: tuple[int, int], errors: list[str]
) -> None:
    """The Reference-Completed index's status mask has the plane shape and dtype."""
    actual = tuple(int(size) for size in group[_IMPUTED_MASK].shape)
    if actual != expected_shape:
        errors.append(
            f"indexed subset {name!r} imputed shape {actual} does not match {expected_shape}"
        )
    if str(group[_IMPUTED_MASK].dtype) != "uint8":
        errors.append(
            f"indexed subset {name!r} imputed has dtype {group[_IMPUTED_MASK].dtype}, not uint8"
        )


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
    if encoding.eaf.reference:
        reference = np.asarray(group[EAF_REFERENCE][:], dtype=np.float32)
        if len(reference) != n_subset:
            errors.append(
                f"indexed subset {name!r} eaf_reference has {len(reference)} entries but "
                f"the subset has {n_subset} variants"
            )


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


def _check_side_tables_against_primary(
    name: str,
    group: Any,
    root: Any,
    encoding: StoreEncoding,
    subset: np.ndarray,
    n_analyses: int,
    errors: list[str],
) -> None:
    """The index's small side tables are exactly the primary's, remapped.

    Exhaustive even on the sampled build path: the tables are O(exception
    count) or O(n_subset), not O(n_analyses * n_subset), and a wrong remap
    would move a decoded cell plausibly and wrongly (#270).  The full decoded
    matrix comparison is what sampling replaces; this is not.
    """
    if encoding.z.is_fixed_point:
        _check_remapped_table(
            name, "z overflow", group, root,
            Z_OVERFLOW_INDEX, Z_OVERFLOW_VALUE, subset, n_analyses, errors,
        )
    if encoding.se.is_residual:
        expected_coefficients = np.asarray(root[SE_COEFFICIENTS][:], dtype=np.float32)
        actual_coefficients = np.asarray(group[SE_COEFFICIENTS][:], dtype=np.float32)
        if not np.array_equal(actual_coefficients, expected_coefficients):
            errors.append(
                f"indexed subset {name!r} se_coefficients disagree with the primary release"
            )
        _check_remapped_table(
            name, "se exception", group, root,
            _SE_EXCEPTION_INDEX, _SE_EXCEPTION_VALUE, subset, n_analyses, errors,
        )
    if encoding.eaf.is_residual:
        expected_baseline = np.asarray(root[EAF_BASELINE].oindex[subset], dtype=np.float32)
        actual_baseline = np.asarray(group[EAF_BASELINE][:], dtype=np.float32)
        if not np.array_equal(actual_baseline, expected_baseline, equal_nan=True):
            errors.append(
                f"indexed subset {name!r} eaf_baseline disagrees with the primary release"
            )
        _check_remapped_table(
            name, "eaf exception", group, root,
            EAF_EXCEPTION_INDEX, EAF_EXCEPTION_VALUE, subset, n_analyses, errors,
        )


def _check_remapped_table(
    name: str,
    what: str,
    group: Any,
    root: Any,
    index_name: str,
    value_name: str,
    subset: np.ndarray,
    n_analyses: int,
    errors: list[str],
) -> None:
    """One exception/overflow table must equal the primary's remapped table."""
    expected_index, expected_value = _remap_side_table(
        _read_side(root, index_name), _read_side(root, value_name), subset, n_analyses
    )
    if not (
        np.array_equal(_read_side(group, index_name), expected_index)
        and np.array_equal(
            _read_side(group, value_name), expected_value, equal_nan=True
        )
    ):
        errors.append(
            f"indexed subset {name!r} {what} table is not the primary release's remapped "
            "table; the index is stale or corrupt"
        )


@dataclass
class _DecodeContext:
    """The primary and indexed decoders one validation pass reuses."""

    codec: StoreCodec
    z_plane: DenseZPlane
    se_plane: DenseSePlane
    eaf_plane: DenseEafPlane
    coefficients: np.ndarray
    baseline: np.ndarray | None
    primary_imputed: Any | None


def _open_decode_context(
    name: str, group: Any, root: Any, encoding: StoreEncoding, errors: list[str]
) -> _DecodeContext | None:
    """Open every decoder, or report that the index cannot be decoded at all."""
    try:
        codec = _codec(group, encoding)
        z_plane = DenseZPlane.open(root, encoding)
        se_plane = DenseSePlane.open(root, encoding)
        # Opened even when the index carries no `eaf` array: a release whose
        # source reported no frequency still has imputed cells with the panel's
        # (issue #113), and those are exactly the values a naive skip would drop.
        eaf_plane = DenseEafPlane.open(root, encoding)
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
    primary_imputed = root[_IMPUTED_MASK] if _IMPUTED_MASK in root else None
    return _DecodeContext(
        codec, z_plane, se_plane, eaf_plane, coefficients, baseline, primary_imputed
    )


def _validation_sample(
    attrs: Mapping[str, Any],
    n_analyses: int,
    n_subset: int,
    write_unit: int,
    inner_chunk: int,
) -> _SamplePlan:
    """A deterministic, bounded, stratified sample of index cells to decode.

    The sample covers the four corners; the slot before, at and last-of each
    anchored **physical write-unit** boundary (a v2 chunk or a v3 shard) and,
    separately, a smaller spread of **inner-chunk** boundaries; a spread of
    Analyses including the first and last; and input-checksum-seeded cells.  The
    two boundary sets are kept separate so a v3 index whose whole subset is one
    shard still samples its inner geometry.  It is deliberately bounded and
    independent of the matrix, so it catches the systemic, ordering and
    output-geometry corruption a sample can catch and may miss an isolated cell;
    the exhaustive standalone `validate` audit is what rules that out (#270).
    """
    seed = int.from_bytes(
        hashlib.sha256(str(attrs[ATTR_INPUT_SHA256]).encode("utf-8")).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    analysis_axis = _sample_analysis_axis(n_analyses)
    shard_starts = _sample_anchor_starts(n_subset, write_unit, SAMPLE_SHARD_ANCHORS)
    # The inner spread stays inside one physical write unit, so at least some of
    # its boundaries are distinct from the write-unit boundaries (a v3 shard is
    # many inner chunks).
    inner_span = min(int(n_subset), max(1, int(write_unit)))
    inner_starts = _sample_anchor_starts(inner_span, inner_chunk, SAMPLE_INNER_ANCHORS)
    cells = _sample_cell_list(
        n_analyses,
        n_subset,
        analysis_axis,
        shard_starts,
        int(write_unit),
        inner_starts,
        int(inner_chunk),
        rng,
    )
    sample_analyses, sample_slots = _dedupe_cells(cells)
    strategy = _sample_strategy(analysis_axis, shard_starts, inner_starts)
    return _SamplePlan(
        analyses=sample_analyses, slots=sample_slots, seed=seed, strategy=strategy
    )


def _sample_strategy(
    analysis_axis: np.ndarray, shard_starts: np.ndarray, inner_starts: np.ndarray
) -> str:
    """The human-readable sample strategy, with correct singular/plural."""
    analysis_label = _counted(
        len(analysis_axis), "evenly spaced Analysis", "evenly spaced Analyses"
    )
    shard_label = _counted(
        len(shard_starts), "physical write-unit boundary", "physical write-unit boundaries"
    )
    inner_label = _counted(
        len(inner_starts), "inner-chunk boundary", "inner-chunk boundaries"
    )
    return (
        "corners + "
        f"{analysis_label} + "
        f"{shard_label} + "
        f"{inner_label} + "
        f"{SAMPLE_RANDOM_PER_ANALYSIS} checksum-seeded cells per Analysis + "
        f"{SAMPLE_GLOBAL_RANDOM} checksum-seeded cells across Analyses"
    )


def _counted(count: int, singular: str, plural: str) -> str:
    """`<count> <singular|plural>`, so a one-boundary sample reads correctly."""
    return f"{count} {singular if count == 1 else plural}"


def _sample_analysis_axis(n_analyses: int) -> np.ndarray:
    """A bounded spread of Analysis indices, including the first and last."""
    return np.unique(
        np.linspace(
            0,
            max(n_analyses - 1, 0),
            num=min(SAMPLE_ANALYSES, max(n_analyses, 1)),
            dtype=np.int64,
        )
    )


def _sample_anchor_starts(n_subset: int, unit: int, cap: int) -> np.ndarray:
    """Up to `cap` evenly spread multiples of `unit` below `n_subset`."""
    unit = max(1, int(unit))
    starts = np.arange(0, max(n_subset, 1), unit, dtype=np.int64)
    if len(starts) > cap:
        picks = np.unique(np.linspace(0, len(starts) - 1, num=cap, dtype=np.int64))
        starts = starts[picks]
    return starts


def _add_boundary_cells(
    add: Any, analysis: int, n_subset: int, start: int, unit: int
) -> None:
    """The slot before a boundary, the first slot on it and the unit's last slot."""
    add(analysis, int(start) - 1)
    add(analysis, int(start))
    add(analysis, min(int(start) + int(unit), n_subset) - 1)


def _sample_cell_list(
    n_analyses: int,
    n_subset: int,
    analysis_axis: np.ndarray,
    shard_starts: np.ndarray,
    shard_unit: int,
    inner_starts: np.ndarray,
    inner_unit: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """The bounded, stratified cell list before deduplication."""
    cells: list[tuple[int, int]] = []

    def add(analysis: int, slot: int) -> None:
        if 0 <= analysis < n_analyses and 0 <= slot < n_subset:
            cells.append((int(analysis), int(slot)))

    # First/last cells on both axes.
    for analysis in (0, n_analyses - 1):
        for slot in (0, n_subset - 1):
            add(analysis, slot)
    for analysis in analysis_axis:
        _add_analysis_cells(
            add,
            int(analysis),
            n_subset,
            shard_starts,
            shard_unit,
            inner_starts,
            inner_unit,
            rng,
        )
    # Seeded cells over arbitrary Analyses, independent of the axis spread.
    random_analyses = rng.integers(0, n_analyses, size=SAMPLE_GLOBAL_RANDOM)
    random_slots = rng.integers(0, n_subset, size=SAMPLE_GLOBAL_RANDOM)
    for analysis, slot in zip(random_analyses, random_slots, strict=True):
        add(int(analysis), int(slot))
    return cells


def _add_analysis_cells(
    add: Any,
    analysis: int,
    n_subset: int,
    shard_starts: np.ndarray,
    shard_unit: int,
    inner_starts: np.ndarray,
    inner_unit: int,
    rng: np.random.Generator,
) -> None:
    """The endpoints, boundary cells and seeded slots of one Analysis."""
    for slot in (0, n_subset - 1):
        add(analysis, slot)
    for start in shard_starts:
        _add_boundary_cells(add, analysis, n_subset, int(start), shard_unit)
    for start in inner_starts:
        _add_boundary_cells(add, analysis, n_subset, int(start), inner_unit)
    for slot in rng.integers(0, n_subset, size=SAMPLE_RANDOM_PER_ANALYSIS):
        add(analysis, int(slot))


def _dedupe_cells(cells: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate cells, preserving the deterministic order."""
    seen: set[tuple[int, int]] = set()
    analyses: list[int] = []
    slots: list[int] = []
    for analysis, slot in cells:
        if (analysis, slot) not in seen:
            seen.add((analysis, slot))
            analyses.append(analysis)
            slots.append(slot)
    return np.asarray(analyses, dtype=np.int64), np.asarray(slots, dtype=np.int64)


def _compare_indexed_values(
    name: str,
    root: Any,
    group: Any,
    encoding: StoreEncoding,
    subset: np.ndarray,
    n_analyses: int,
    errors: list[str],
) -> None:
    """Stream decoded index bands and compare every one to the primary planes.

    Standalone ``validate`` only (#270): every decoded indexed Z, SE and EAF
    must equal its authoritative primary-plane cell; the imputed mask and
    reference EAF must equal the release's own; and missingness must agree
    within the index before the values are compared.  The build's publication
    gate uses the bounded sample in `_compare_indexed_sample` instead.
    """
    context = _open_decode_context(name, group, root, encoding, errors)
    if context is None:
        return
    # Read the index in its own physical write units and tile the Analysis axis
    # so each compressed output chunk (v2) or shard (v3) is decoded exactly
    # once, while peak memory is one band of one plane.  The Analysis tile is
    # the budget-derived one rather than the primary's read unit: validation
    # decodes to float32, so a source-aligned 65,536-variant band held ~1.9 GB
    # on the 20k probe (measured) and was no faster -- the decode and compare,
    # not the primary re-read, is the work.
    write_unit = _physical_write_unit(group["z"])
    for start, stop in _output_write_bands(len(subset), write_unit):
        for a0, a1 in _analysis_tiles(
            n_analyses, write_unit, EXHAUSTIVE_VALIDATION_BAND_CELLS, 1
        ):
            message = _compare_indexed_band(
                name, context, group, encoding, subset, start, stop, a0, a1
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
    a0: int,
    a1: int,
) -> str | None:
    """One band's decoded equality checks; the first failure's message, or None."""
    n_subset = len(subset)
    rows = subset[start:stop]
    cols = np.arange(a0, a1)
    positions = positions_rows_cols(cols, np.arange(start, stop), n_subset)
    index_z_raw = np.asarray(group["z"][a0:a1, start:stop])
    index_se_raw = np.asarray(group["se"][a0:a1, start:stop])
    primary_eaf = (
        context.eaf_plane.read_block(rows, cols).values
        if context.eaf_plane.can_report_frequencies
        else None
    )
    primary_z = context.z_plane.block(rows, cols)
    primary_se = context.se_plane.block(rows, cols, eaf=primary_eaf)
    index_eaf, eaf_error = _decode_index_eaf(
        name, context, group, encoding, start, stop, a0, a1
    )
    if eaf_error is not None:
        return eaf_error
    def imputed_check() -> str | None:
        return _imputed_band_error(
            name, context, group, index_z_raw, index_se_raw, encoding, rows, start, stop, a0, a1
        )

    cells = (index_z_raw, index_se_raw, index_eaf, primary_z, primary_eaf, primary_se, positions)
    return _compare_decoded_cells(name, context, encoding, cells, cols, imputed_check)


def _compare_decoded_cells(
    name: str,
    context: _DecodeContext,
    encoding: StoreEncoding,
    cells: tuple[Any, ...],
    cols: np.ndarray,
    imputed_check: Any,
) -> str | None:
    """The decoded equality checks shared by an exhaustive band and the sample."""
    index_z_raw, index_se_raw, index_eaf, primary_z, primary_eaf, primary_se, positions = cells
    return (
        _missingness_error(name, context, index_z_raw, index_se_raw, encoding)
        or imputed_check()
        or _z_band_error(name, context, index_z_raw, primary_z, positions)
        or _eaf_band_error(name, index_eaf, primary_eaf)
        or _se_band_error(
            name,
            context,
            index_se_raw,
            index_eaf,
            primary_se,
            encoding,
            cols,
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
    a0: int,
    a1: int,
) -> tuple[np.ndarray | None, str | None]:
    """Decode one band's indexed EAF, with an error string when it cannot be.

    The index's own `imputed` mask and `eaf_reference` are used, so a corrupt
    reference or status that changes a decoded frequency is caught by the
    comparison against the primary rather than papered over with the primary's
    own values.
    """
    has_eaf = "eaf" in group
    if not has_eaf and not encoding.eaf.reference:
        return None, None
    n_subset = int(group["z"].shape[1])
    tile = a1 - a0
    positions = positions_rows_cols(np.arange(a0, a1), np.arange(start, stop), n_subset)
    if has_eaf:
        codes = np.asarray(group["eaf"][a0:a1, start:stop])
        baseline_cells = (
            None
            if context.baseline is None
            else context.baseline[start:stop][None, :].repeat(tile, axis=0)
        )
    else:
        codes = np.full((tile, stop - start), np.nan, dtype=np.float32)
        baseline_cells = None
    reference_values = (
        np.asarray(group[EAF_REFERENCE][start:stop], dtype=np.float32)
        if EAF_REFERENCE in group
        else None
    )
    return _decode_index_eaf_codes(
        name, context, codes, baseline_cells, reference_values,
        _index_imputed_band(group, start, stop, a0, a1), positions,
    )


def _decode_index_eaf_codes(
    name: str,
    context: _DecodeContext,
    codes: np.ndarray,
    baseline: np.ndarray | None,
    reference_values: np.ndarray | None,
    imputed: np.ndarray | None,
    positions: Any,
) -> tuple[np.ndarray | None, str | None]:
    """Decode index EAF codes through the index's own codec, or report why not.

    ``reference_values`` is the per-variant panel frequency (1-D) or None; it is
    broadcast to the codes' shape so a band and a point sample share this path.
    """
    reference = (
        None if reference_values is None else np.broadcast_to(reference_values, codes.shape)
    )
    try:
        decoded = context.codec.decode_eaf(
            codes,
            baseline=baseline,
            positions=positions,
            imputed=imputed,
            reference=reference,
        )
    except Exception as exc:
        return None, f"indexed subset {name!r} eaf cannot be decoded: {exc}"
    return decoded, None


def _index_imputed_band(
    group: Any, start: int, stop: int, a0: int, a1: int
) -> np.ndarray | None:
    """The index's own `imputed` band, or None on an Observed-Only index."""
    if _IMPUTED_MASK not in group:
        return None
    return np.asarray(group[_IMPUTED_MASK][a0:a1, start:stop], dtype=np.uint8)


def _imputed_band_error(
    name: str,
    context: _DecodeContext,
    group: Any,
    index_z_raw: np.ndarray,
    index_se_raw: np.ndarray,
    encoding: StoreEncoding,
    rows: np.ndarray,
    start: int,
    stop: int,
    a0: int,
    a1: int,
) -> str | None:
    """Whether the index's Association Status mask is well-formed and authoritative."""
    index_imputed = _index_imputed_band(group, start, stop, a0, a1)
    if index_imputed is None:
        return None
    primary_cells = (
        None
        if context.primary_imputed is None
        else np.asarray(context.primary_imputed.oindex[rows, a0:a1]).T
    )
    return _imputed_mask_message(
        name, context, index_z_raw, index_se_raw, encoding, index_imputed, primary_cells
    )


def _imputed_mask_message(
    name: str,
    context: _DecodeContext,
    index_z_raw: np.ndarray,
    index_se_raw: np.ndarray,
    encoding: StoreEncoding,
    index_imputed: np.ndarray,
    primary_cells: np.ndarray | None,
) -> str | None:
    """The well-formedness and authority checks shared by the band and sample paths.

    Three silent-wrong-answer classes in one rule: an imputed cell that is
    missing is an invalid state (spec §15), an imputed mask that does not match
    the primary labels observed cells imputed and vice versa, and a non-0/1
    value is not a mask at all.  Each is refused rather than decoded.
    """
    if not np.all((index_imputed == 0) | (index_imputed == 1)):
        return f"indexed subset {name!r} imputed mask contains values other than 0 and 1"
    se_missing = (
        index_se_raw == SE_MISSING if encoding.se.is_residual else np.isnan(index_se_raw)
    )
    missing = context.codec.missing_mask(index_z_raw) | se_missing
    if np.any(index_imputed.astype(bool) & missing):
        return f"indexed subset {name!r} marks a missing Z/SE cell imputed"
    if primary_cells is None:
        return f"indexed subset {name!r} carries an imputed mask but the release has none"
    if not np.array_equal(index_imputed, primary_cells):
        return (
            f"indexed subset {name!r} imputed mask disagrees with the primary release; "
            "the index is stale or corrupt"
        )
    return None


def _eaf_reference_error(
    name: str,
    group: Any,
    root: Any,
    subset: np.ndarray,
    encoding: StoreEncoding,
    errors: list[str],
) -> None:
    """The subset's reference EAF must be the release's restricted to its variants."""
    if not encoding.eaf.reference:
        return
    index_reference = np.asarray(group[EAF_REFERENCE][:], dtype=np.float32)
    primary = np.asarray(root[EAF_REFERENCE].oindex[subset], dtype=np.float32)
    if not np.array_equal(index_reference, primary, equal_nan=True):
        errors.append(
            f"indexed subset {name!r} eaf_reference disagrees with the primary release; "
            "the index is stale or corrupt"
        )


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
    cols: np.ndarray,
    positions: Any,
) -> str | None:
    if encoding.se.is_residual:
        if index_eaf is None:
            return f"indexed subset {name!r} se cannot be decoded without its eaf plane"
        analysis_index = np.asarray(cols, dtype=np.int64)
        if index_se_raw.ndim == 2:
            # A band is (Analysis, slot); the sample is a flat run of cells.
            analysis_index = np.broadcast_to(analysis_index[:, None], index_se_raw.shape)
        index_se = context.codec.decode_se(
            index_se_raw,
            eaf=index_eaf,
            analysis_index=analysis_index,
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


# ── Sampled publication gate (#270) ─────────────────────────────────────────


def _points(array: Any, analyses: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """The pointwise cells `(analyses[i], slots[i])` of an index array."""
    return np.asarray(array.vindex[analyses, slots])


def _index_imputed_points(
    group: Any, analyses: np.ndarray, slots: np.ndarray
) -> np.ndarray | None:
    """The index's own `imputed` cells for the sample, or None without one."""
    if _IMPUTED_MASK not in group:
        return None
    return np.asarray(group[_IMPUTED_MASK].vindex[analyses, slots], dtype=np.uint8)


def _imputed_points_error(
    name: str,
    context: _DecodeContext,
    group: Any,
    variants: np.ndarray,
    sample: _SamplePlan,
    encoding: StoreEncoding,
    index_z_raw: np.ndarray,
    index_se_raw: np.ndarray,
) -> str | None:
    """The sample's Association Status mask, well-formed and authoritative."""
    index_imputed = _index_imputed_points(group, sample.analyses, sample.slots)
    if index_imputed is None:
        return None
    primary_cells = (
        None
        if context.primary_imputed is None
        else np.asarray(
            context.primary_imputed.vindex[variants, sample.analyses], dtype=np.uint8
        )
    )
    return _imputed_mask_message(
        name, context, index_z_raw, index_se_raw, encoding, index_imputed, primary_cells
    )


def _decode_index_eaf_points(
    name: str,
    context: _DecodeContext,
    group: Any,
    encoding: StoreEncoding,
    sample: _SamplePlan,
    positions: Any,
) -> tuple[np.ndarray | None, str | None]:
    """Decode the sample's indexed EAF from the index's own tables."""
    has_eaf = "eaf" in group
    if not has_eaf and not encoding.eaf.reference:
        return None, None
    if has_eaf:
        codes = _points(group["eaf"], sample.analyses, sample.slots)
        baseline = (
            None
            if context.baseline is None
            else np.asarray(context.baseline[sample.slots], dtype=np.float32)
        )
    else:
        codes = np.full(sample.cells, np.nan, dtype=np.float32)
        baseline = None
    reference_values = (
        np.asarray(group[EAF_REFERENCE][sample.slots], dtype=np.float32)
        if EAF_REFERENCE in group
        else None
    )
    return _decode_index_eaf_codes(
        name, context, codes, baseline, reference_values,
        _index_imputed_points(group, sample.analyses, sample.slots), positions,
    )


def _compare_indexed_sample(
    name: str,
    root: Any,
    group: Any,
    encoding: StoreEncoding,
    subset: np.ndarray,
    sample: _SamplePlan,
    errors: list[str],
) -> None:
    """Decode and compare a bounded deterministic sample of index cells.

    The build's publication gate (#270): the structural invariants and the
    small side tables are checked exhaustively elsewhere, and here the decoded
    Z/SE/EAF and Association Status of the sample's cells are compared against
    the primary planes.  The sample covers first/last cells, physical output
    boundaries, a spread of Analyses and checksum-seeded cells; an isolated
    unsampled corruption can pass this gate by design and is left to the
    exhaustive standalone audit.
    """
    context = _open_decode_context(name, group, root, encoding, errors)
    if context is None:
        return
    if sample.cells == 0:
        errors.append(f"indexed subset {name!r} validation sample is empty")
        return
    view = _sample_decoded_view(name, context, group, encoding, subset, sample, errors)
    if view is None:
        return
    cells, cols, imputed_check = view
    message = _compare_decoded_cells(name, context, encoding, cells, cols, imputed_check)
    if message is not None:
        errors.append(message + " (sampled cell)")


def _sample_decoded_view(
    name: str,
    context: _DecodeContext,
    group: Any,
    encoding: StoreEncoding,
    subset: np.ndarray,
    sample: _SamplePlan,
    errors: list[str],
) -> tuple[tuple[Any, ...], np.ndarray, Any] | None:
    """Gather the sample's index and primary decoded cells for comparison.

    Returns the ``(cells, cols, imputed_check)`` triple `_compare_decoded_cells`
    needs, or None after reporting a decode failure.
    """
    variants = subset[sample.slots]
    n_subset = len(subset)
    positions = positions_at(sample.analyses * n_subset + sample.slots)
    index_z_raw = _points(group["z"], sample.analyses, sample.slots)
    index_se_raw = _points(group["se"], sample.analyses, sample.slots)
    primary_eaf = (
        context.eaf_plane.read_points(variants, sample.analyses).values
        if context.eaf_plane.can_report_frequencies
        else None
    )
    index_eaf, eaf_error = _decode_index_eaf_points(
        name, context, group, encoding, sample, positions
    )
    if eaf_error is not None:
        errors.append(eaf_error)
        return None
    primary_z = context.z_plane.points(variants, sample.analyses)
    primary_se = context.se_plane.points(variants, sample.analyses, eaf=primary_eaf)
    def imputed_check() -> str | None:
        return _imputed_points_error(
            name, context, group, variants, sample, encoding, index_z_raw, index_se_raw
        )

    cells = (index_z_raw, index_se_raw, index_eaf, primary_z, primary_eaf, primary_se, positions)
    return cells, sample.analyses, imputed_check
