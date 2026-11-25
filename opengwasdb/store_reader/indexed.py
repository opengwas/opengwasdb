"""Reading an Indexed Variant Subset (ADR 0061, issue #264): the read seam.

An **Indexed Variant Subset** is a named, optional, rebuildable query index over
a caller-supplied set of canonical ALIDs, holding every Analysis's full
statistics laid out Analysis-major. This module is its read seam --
`open_indexed_subset` and `IndexedSubset` -- and the structural, layout and
staleness checks an open makes before anything is decoded, so a query never
depends on someone having run `validate` first. Building, publishing and the
standalone validation rules are `opengwasdb.layouts.dense.indexed_subsets`',
which imports this module (#464); its docstring states the decisions the
format rests on.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import zarr

from opengwasdb.store_reader import arrays as store_arrays
from opengwasdb.store_reader.arrays import array_length
from opengwasdb.store_reader.assembly import normalise_build
from opengwasdb.store_reader.encoding.codec import (
    EAF_BASELINE,
    EAF_EXCEPTION_INDEX,
    EAF_EXCEPTION_VALUE,
    EAF_REFERENCE,
    SE_COEFFICIENTS,
    Z_OVERFLOW_INDEX,
    Z_OVERFLOW_VALUE,
    EafExceptionTable,
    SeExceptionTable,
    StoreDecoder,
    ZOverflowTable,
    positions_flat,
)
from opengwasdb.store_reader.encoding.plan import StoreEncoding, UnsupportedEncoding
from opengwasdb.store_reader.enums import CompletionState, PrimaryStorageLayout
from opengwasdb.store_reader.format import zarr_format_for_version
from opengwasdb.store_reader.hybrid import dense_component_path
from opengwasdb.store_reader.layout import (
    INDEXED_SUBSET_CHUNK,
    ArrayRole,
    chunk_layout,
    shard_layout,
)
from opengwasdb.store_reader.manifest import StoreManifest
from opengwasdb.store_reader.release import open_release
from opengwasdb.store_reader.stored import eaf_baseline, se_coefficients

#: The group under `data.zarr` that holds every Indexed Variant Subset.
INDEXED_SUBSETS_GROUP = "indexed_subsets"

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

#: The Overflow-only record (#324).  Deliberately outside `_REQUIRED_ATTRS`: a
#: group built before the option existed records neither key, and its true count
#: is 0 because that builder could only refuse such an ALID.  `_overflow_only_record`
#: reads that case, and the counts-sum rule still refuses a group whose recorded
#: skip was removed from the sum.
ATTR_OVERFLOW_ONLY_COUNT = "overflow_only_count"

ATTR_OVERFLOW_ONLY_SAMPLE = "overflow_only_sample"

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


class IndexedSubsetError(Exception):
    """Base class for an Indexed Variant Subset that cannot be built or read."""


class IndexedSubsetNameError(IndexedSubsetError, ValueError):
    """A subset name is not in the accepted grammar."""


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

#: How many of the skipped Overflow-only ALIDs a build records and reports.  A
#: deterministic prefix in variant-list order, so a reader can see which requests
#: the index could not answer without carrying the whole list (#324).
OVERFLOW_ONLY_SAMPLE_SIZE = 20


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
    n_overflow_only: int
    overflow_only_sample: tuple[str, ...]
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


def is_valid_indexed_subset_name(name: str) -> bool:
    """Whether `name` could be a published subset name (no raise)."""
    return bool(_NAME_RE.match(name))


def indexed_subset_namespace_path(store_path: str | Path) -> Path:
    """The `indexed_subsets` directory for `store_path`, resolving a Hybrid root.

    The one place the physical namespace is located, so discovery, removal,
    overview and reading cannot disagree about where a Hybrid release keeps
    its index (issue #271).
    """
    return indexed_subset_store_path(store_path) / "data.zarr" / INDEXED_SUBSETS_GROUP


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


def _overflow_only_record(
    name: str, attrs: dict[str, Any]
) -> tuple[int, tuple[str, ...], list[str]]:
    """The recorded Overflow-only count and sample, with findings for a bad record.

    Every group this writer publishes records both attributes (#324).  A group
    recording neither was built before the ``--overflow-only`` option existed,
    when an ALID only the Overflow carries could do nothing but fail the build:
    its count is zero by construction, not an unknown read as zero.  The
    requested/resolved/absent sum still has to hold, so a group whose recorded
    count was removed or forged fails validation rather than hiding a skip.
    """
    has_count = ATTR_OVERFLOW_ONLY_COUNT in attrs
    has_sample = ATTR_OVERFLOW_ONLY_SAMPLE in attrs
    if not has_count and not has_sample:
        return 0, (), []
    if has_count != has_sample:
        missing = ATTR_OVERFLOW_ONLY_SAMPLE if has_count else ATTR_OVERFLOW_ONLY_COUNT
        return 0, (), [f"indexed subset {name!r} records no {missing}"]
    count, errors = _recorded_overflow_count(name, attrs)
    sample, sample_errors = _recorded_overflow_sample(name, attrs, count)
    return count, sample, [*errors, *sample_errors]


def _recorded_overflow_count(name: str, attrs: dict[str, Any]) -> tuple[int, list[str]]:
    """The recorded count, or the reason it cannot be one."""
    raw = attrs[ATTR_OVERFLOW_ONLY_COUNT]
    if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
        return 0, [
            f"indexed subset {name!r} records a non-integer "
            f"{ATTR_OVERFLOW_ONLY_COUNT}: {raw!r}"
        ]
    count = int(raw)
    if count < 0:
        return count, [
            f"indexed subset {name!r} records a negative {ATTR_OVERFLOW_ONLY_COUNT}: {count}"
        ]
    return count, []


def _recorded_overflow_sample(
    name: str, attrs: dict[str, Any], count: int
) -> tuple[tuple[str, ...], list[str]]:
    """The recorded sample, or the reason it is not the deterministic prefix."""
    raw = attrs[ATTR_OVERFLOW_ONLY_SAMPLE]
    if not isinstance(raw, (list, tuple)) or any(
        not isinstance(alid, str) for alid in raw
    ):
        return (), [
            f"indexed subset {name!r} records a malformed "
            f"{ATTR_OVERFLOW_ONLY_SAMPLE}: {raw!r}"
        ]
    sample = tuple(raw)
    if len(sample) != min(max(count, 0), OVERFLOW_ONLY_SAMPLE_SIZE):
        return sample, [
            f"indexed subset {name!r} records {len(sample)} "
            f"{ATTR_OVERFLOW_ONLY_SAMPLE} entries for {count} skipped ALIDs; the "
            f"sample is the first {OVERFLOW_ONLY_SAMPLE_SIZE} in list order"
        ]
    return sample, []


def _require_overflow_only_record(
    name: str, attrs: dict[str, Any]
) -> tuple[int, tuple[str, ...]]:
    """The recorded Overflow-only count and sample, or a subset-naming failure."""
    count, sample, errors = _overflow_only_record(name, attrs)
    if errors:
        raise IndexedSubsetError(errors[0])
    return count, sample


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
    mismatches = stale_subset_findings(name, attrs, encoding, manifest)
    if mismatches:
        raise IndexedSubsetStaleError("; ".join(mismatches))


def stale_subset_findings(
    name: str, attrs: dict[str, Any], encoding: StoreEncoding, manifest: StoreManifest
) -> list[str]:
    """Every way a published subset is stale for `manifest`'s release: the one
    definition the reader, `subsets.indexed` and the rewrite refuse by. Its recorded
    source identity (`_source_identity_mismatches`) and its declared `encoding` must
    both be the release's."""
    findings = _source_identity_mismatches(name, attrs, manifest)
    mismatch = encoding_mismatch(name, encoding, manifest)
    if mismatch is not None:
        findings.append(mismatch)
    return findings


def encoding_mismatch(
    name: str, encoding: StoreEncoding, manifest: StoreManifest
) -> str | None:
    """The finding for a subset built under another encoding than the release's."""
    if encoding == manifest.encoding:
        return None
    return (
        f"indexed subset {name!r} was built with an encoding that is not the "
        "release's; its values cannot be decoded against the authoritative planes"
    )


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

    Mirrors validation's `_check_identity_attrs`.  Cheap (a handful of
    attributes) and validation-visible, so the read path runs it rather than
    letting a wrong-width index decode under the release's shape by coincidence.
    The Overflow-only count is part of the sum: a requested ALID is either
    indexed, absent, or carried only by the Overflow (issue #324).
    """
    if _require_attr_int(name, attrs, ATTR_N_ANALYSES) != n_analyses:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} declares {attrs[ATTR_N_ANALYSES]} Analyses "
            f"but this release has {n_analyses}"
        )
    requested = _require_attr_int(name, attrs, ATTR_REQUESTED_COUNT)
    resolved = _require_attr_int(name, attrs, ATTR_RESOLVED_COUNT)
    absent = _require_attr_int(name, attrs, ATTR_ABSENT_COUNT)
    overflow_only, _sample, errors = _overflow_only_record(name, attrs)
    if errors:
        raise IndexedSubsetError(errors[0])
    if requested != resolved + absent + overflow_only:
        raise IndexedSubsetError(
            f"Indexed Variant Subset {name!r} records requested={requested}, "
            f"resolved={resolved}, absent={absent}, overflow_only={overflow_only}, "
            "which do not add up"
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


def rescope_indexed_subset_error(
    caller_path: str | Path, physical_path: str | Path, exc: IndexedSubsetError
) -> IndexedSubsetError:
    """Re-point a subset failure from its physical Store to the caller's path.

    A Hybrid root's Indexed Variant Subset lives inside its nested Dense
    Component, so a query entered through that component scopes the failure to
    ``<root>/dense``.  The operator named the Hybrid root and must see that
    path, not the internal component (issue #271 review).  Any leading
    ``store <physical>: `` prefix is **replaced** rather than prepended, so the
    message is never double-prefixed, and an existing ``store <caller>: ``
    prefix is returned untouched.  ``type(exc)`` preserves the subclass, so a
    stale index stays an ``IndexedSubsetStaleError``.
    """
    text = str(exc)
    caller_prefix = f"store {Path(caller_path)}: "
    if text.startswith(caller_prefix):
        return exc
    physical_prefix = f"store {Path(physical_path)}: "
    if text.startswith(physical_prefix):
        text = text[len(physical_prefix):]
    return type(exc)(f"{caller_prefix}{text}")


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


def _require_release_bound_axis(
    store_path: Path,
    name: str,
    group: Any,
    attrs: dict[str, Any],
    encoding: StoreEncoding,
) -> np.ndarray:
    """Refuse a stale subset and bound its Variant Index against the release's axis."""
    release = open_release(store_path)
    _refuse_stale_subset(name, attrs, encoding, release.manifest)
    # Bound the subset against the release's own axis: an out-of-range Variant
    # Index would name a variant the Store does not have, and `resolve_rows`
    # would index the Store Variant Table out of bounds.  The Analysis width is
    # read from the same primary plane so a wrong-width index is refused too.
    primary_z = release.arrays()["z"]
    completed = release.manifest.completion_state is CompletionState.REFERENCE_COMPLETED
    return _require_decodable_subset(
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
    variant_index = _require_release_bound_axis(store_path, name, group, attrs, encoding)
    overflow_only, overflow_only_sample = _require_overflow_only_record(name, attrs)
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
        n_overflow_only=overflow_only,
        overflow_only_sample=overflow_only_sample,
        builder_version=str(attrs[ATTR_BUILDER_VERSION]),
        created_at=str(attrs[ATTR_CREATED_AT]),
        has_eaf="eaf" in group,
        has_imputed=_IMPUTED_MASK in group,
        n_z_overflow=_side_table_length(group, Z_OVERFLOW_INDEX),
        n_eaf_exceptions=_side_table_length(group, EAF_EXCEPTION_INDEX),
        n_se_exceptions=_side_table_length(group, _SE_EXCEPTION_INDEX),
        _group=group,
    )


def _codec(group: Any, encoding: StoreEncoding) -> StoreDecoder:
    return StoreDecoder(
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
    codec: StoreDecoder,
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
        return codec.decode_eaf(
            np.asarray(group["eaf"][analysis_index, :]),
            baseline=eaf_baseline(group),
            positions=positions,
            imputed=imputed,
            reference=reference,
        )
    if reference is None:
        return None
    blank = np.full(n_subset, np.nan, dtype=np.float32)
    return codec.decode_eaf(blank, imputed=imputed, reference=reference)


def _decode_se_column(
    codec: StoreDecoder,
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
        coefficients=_required_coefficients(group),
        positions=positions,
    )


def _required_coefficients(group: Any) -> np.ndarray:
    """A residual subset's `se_coefficients`; `_expected_arrays` requires the array."""
    coefficients = se_coefficients(group)
    if coefficients is None:
        raise KeyError(SE_COEFFICIENTS)
    return coefficients


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
            EAF_REFERENCE: (ArrayRole.PER_VARIANT, INDEXED_SUBSET_CHUNK),
            EAF_BASELINE: (ArrayRole.PER_VARIANT, INDEXED_SUBSET_CHUNK),
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
    expected_shard = shard_layout(role, shape, inner_chunk=expected_chunk)
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
        expected_chunk = chunk_layout(role, shape, hint=hint)
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
