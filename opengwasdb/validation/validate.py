"""Store validation."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from opengwasdb.completion.schema import COMPLETION_QUALITY_COLUMNS
from opengwasdb.encoding import (
    EAF_BASELINE,
    EAF_EXCEPTION,
    EAF_EXCEPTION_INDEX,
    EAF_EXCEPTION_VALUE,
    EAF_REFERENCE,
    Z_OVERFLOW,
    Z_OVERFLOW_INDEX,
    Z_OVERFLOW_VALUE,
    DenseEafPlane,
    DenseZPlane,
    EafExceptionTable,
    RaggedEafPlane,
    StoreCodec,
    StoreEncoding,
    ZOverflowTable,
)
from opengwasdb.layouts.dense.top_hits import threshold_key, z_critical
from opengwasdb.layouts.hybrid.layout import dense_component_path, dense_to_shared_path
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.model.analyses import (
    RETIRED_ANALYSIS_COLUMNS,
    TOP_HIT_COUNT_COLUMNS,
    read_analyses,
)
from opengwasdb.model.enums import (
    AncestryAssignmentMethod,
    CompletionState,
    EafOrientationOutcome,
    EafScope,
    OriginalSdMethod,
    PrimaryStorageLayout,
    StoredEffectScale,
)
from opengwasdb.stats import p_value_from_z
from opengwasdb.store.open import (
    DENSE_ENVELOPE,
    HYBRID_DENSE_COMPONENT_ENVELOPE,
    HYBRID_ENVELOPE,
    RAGGED_ENVELOPE,
    OpenGWASDBStore,
    UnsupportedFormatVersion,
    open_store,
)
from opengwasdb.variants import (
    VariantAxis,
    VariantRecord,
    is_indexable_rsid,
    variant_alid_bytes_path,
    variant_alid_rows_path,
    variant_offsets_path,
    variant_tabix_path,
    variant_table_path,
)


@dataclass(frozen=True)
class ValidationResult:
    """Validation outcome with actionable error strings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_store(
    path: str | Path,
    source: str | Path | Sequence[str | Path] | None = None,
    *,
    fidelity_samples: int = 2000,
    fidelity_seed: int = 0,
    source_assembly: str | None = None,
    chain_file: str | Path | None = None,
) -> ValidationResult:
    """Validate a v0.1 Store Release directory.

    Internal (structural) validation is always run. When ``source`` is given —
    the original dataset the store was built from (a manifest TSV with
    ``trait_id``/``file_path`` columns, or one/many self-describing source
    files) — a **source-fidelity** check is also run for dense stores: it draws
    a random sample of source associations and confirms the store holds the same
    z/se, orienting the join through the stored source-ALID provenance (or, for
    lifted stores that predate that column, via liftover). ``source_assembly``
    names the source coordinate build (default: the store's own assembly, i.e.
    no liftover); ``chain_file`` overrides the liftover chain for the fallback.
    """

    store_path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    store = _load_manifest(store_path, errors)
    if store is None:
        return ValidationResult(errors=errors)
    manifest = store.manifest

    if manifest.primary_layout is PrimaryStorageLayout.RAGGED:
        _validate_ragged_store(store, errors)
        _validate_eaf_orientation(store, errors, warnings)
        if source is not None:
            errors.append("source-fidelity check is only supported for dense stores")
        return ValidationResult(errors=errors, warnings=warnings)

    if manifest.primary_layout is PrimaryStorageLayout.HYBRID:
        _validate_hybrid_store(store, errors)
        _validate_eaf_orientation(store, errors, warnings)
        if source is not None:
            errors.append("source-fidelity check is only supported for dense stores")
        return ValidationResult(errors=errors, warnings=warnings)

    _validate_dense_store(store, errors)
    _validate_eaf_orientation(store, errors, warnings)
    if source is not None and not errors:
        _validate_source_fidelity(
            store, source, errors,
            n_samples=fidelity_samples, seed=fidelity_seed,
            source_assembly=source_assembly, chain_file=chain_file,
        )
    return ValidationResult(errors=errors, warnings=warnings)


def _validate_closed_envelope(
    store_path: Path, allowed: frozenset[str], errors: list[str]
) -> None:
    """store-format spec §1/§20 (issue #80): a Store Release's top-level
    directory is closed -- no file or directory beyond what its layout
    legitimately produces. Every other check here is a *presence* check
    ("required entry X exists"); this is the one *closure* check, catching
    drift like the long-unnoticed Ragged `traits.tsv.gz` side-file (retired
    in issue #69) that no presence check could ever have flagged."""
    for name in sorted(p.name for p in store_path.iterdir() if p.name not in allowed):
        errors.append(
            f"unexpected store entry {name!r} in {store_path} is not part of the "
            "documented store envelope for this layout (store-format spec §1)"
        )


def _validate_dense_store(
    store: OpenGWASDBStore, errors: list[str], *, envelope: frozenset[str] = DENSE_ENVELOPE
) -> ValidationResult:
    store_path = store.path
    manifest = store.manifest
    index_path = store.index_path
    data_path = store.data_path
    analyses_path = store.analyses_path
    variants_path = variant_table_path(store_path)
    tabix_path = variant_tabix_path(store_path)
    offsets_path = variant_offsets_path(store_path)
    if not index_path.exists():
        errors.append("missing index.sqlite")
    if not data_path.exists():
        errors.append("missing data.zarr")
    if not analyses_path.exists():
        errors.append("missing analyses.tsv")
    if not variants_path.exists():
        errors.append("missing variants.tsv.gz")
    if not tabix_path.exists():
        errors.append("missing variants.tsv.gz.tbi")
    if not offsets_path.exists():
        errors.append("missing variant_offsets.npy")
    alid_bytes_path = variant_alid_bytes_path(store_path)
    alid_rows_path = variant_alid_rows_path(store_path)
    if not alid_bytes_path.exists():
        errors.append(
            "missing variant_alid_bytes.npy — rebuild the store to generate the ALID search index"
        )
    if not alid_rows_path.exists():
        errors.append(
            "missing variant_alid_rows.npy — rebuild the store to generate the ALID search index"
        )
    _validate_closed_envelope(store_path, envelope, errors)
    if errors:
        return ValidationResult(errors=errors)

    try:
        with store.index_connection() as connection:
            variant_axis = VariantAxis(store_path, connection)
            try:
                n_variants = _validate_variant_axis(variant_axis, errors)
                _validate_sqlite(connection, n_variants, errors)
            finally:
                variant_axis.close()
            n_analyses = _validate_analyses_tsv(analyses_path, errors)
            root = store.arrays(mode="r")
            imputed_arr = None
            on_panel_arr = None
            if manifest.completion_state is CompletionState.REFERENCE_COMPLETED:
                imputed_arr, on_panel_arr = _validate_completion_metadata(
                    root, connection, analyses_path, n_variants, n_analyses, errors
                )
            _validate_encoding_plan(root, manifest.encoding, errors, label="data.zarr")
            if not errors:
                _validate_dense_arrays(
                    root, n_variants, n_analyses, errors, manifest.encoding,
                    imputed_arr, on_panel_arr,
                )
            if not errors:
                _validate_top_hits(root, errors, manifest.encoding)
            if not errors:
                _validate_rho(root, n_analyses, errors)
    except Exception as exc:  # noqa: BLE001 - validators should report actionable failures
        errors.append(f"validation failed: {exc}")
    return ValidationResult(errors=errors)


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

    tables = {r[0] for r in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "completion_quality" not in tables:
        errors.append(
            "index.sqlite is missing the completion_quality table for a reference-completed store"
        )
    else:
        cols = {
            r[1] for r in connection.execute("PRAGMA table_info(completion_quality)").fetchall()
        }
        required = COMPLETION_QUALITY_COLUMNS
        missing_cols = required - cols
        if missing_cols:
            errors.append(
                f"completion_quality table is missing columns: {', '.join(sorted(missing_cols))}"
            )
        else:
            # completion_quality.analysis_index is a positional reference into
            # analyses.tsv, not a SQL foreign key (ADR 0030 consequences) --
            # this is the validator rule that decision says must stand in for
            # the database engine's constraint.
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


def _validate_ragged_store(store: OpenGWASDBStore, errors: list[str]) -> ValidationResult:
    store_path = store.path
    manifest = store.manifest
    index_path = store.index_path
    data_path = store.data_path
    ragged_path = data_path / "ragged"
    analyses_path = store.analyses_path

    for label, p in [
        ("index.sqlite", index_path),
        ("data.zarr", data_path),
        ("data.zarr/ragged", ragged_path),
        ("variants.tsv.gz", variant_table_path(store_path)),
        ("variants.tsv.gz.tbi", variant_tabix_path(store_path)),
        ("variant_alid_bytes.npy", variant_alid_bytes_path(store_path)),
        ("variant_alid_rows.npy", variant_alid_rows_path(store_path)),
        ("analyses.tsv", analyses_path),
    ]:
        if not p.exists():
            errors.append(f"missing {label}")
    _validate_closed_envelope(store_path, RAGGED_ENVELOPE, errors)
    if errors:
        return ValidationResult(errors=errors)

    try:
        root = zarr.open_group(str(ragged_path), mode="r")
        for name in ("offsets", "variant_index", "z", "se"):
            if name not in root:
                errors.append(f"missing data.zarr/ragged/{name}")
        if errors:
            return ValidationResult(errors=errors)

        _validate_encoding_plan(root, manifest.encoding, errors, label="data.zarr/ragged")
        if errors:
            return ValidationResult(errors=errors)
        offsets = root["offsets"][:]
        n_assoc = int(offsets[-1])
        for name in ("variant_index", "z", "se"):
            if len(root[name]) != n_assoc:
                errors.append(
                    f"data.zarr/ragged/{name} has {len(root[name])} entries "
                    f"but offsets imply {n_assoc}"
                )
        if manifest.encoding.z.is_fixed_point:
            raw_z = np.asarray(root["z"][:])
            _overflow_positions_match(
                "data.zarr/ragged/z",
                np.flatnonzero(raw_z == Z_OVERFLOW).astype(np.int64),
                ZOverflowTable.read(root),
                errors,
            )
        se_vals = root["se"][:].astype("float32")
        if np.any(np.isfinite(se_vals) & (se_vals < 0)):
            errors.append("se contains negative finite values")
        # `eaf` is optional (ADR 0036); when present it is a fourth parallel
        # CSR array and must line up with the other three.
        if "eaf" in root:
            if len(root["eaf"]) != n_assoc:
                errors.append(
                    f"data.zarr/ragged/eaf has {len(root['eaf'])} entries "
                    f"but offsets imply {n_assoc}"
                )
            else:
                _validate_ragged_eaf_values(root, manifest.encoding, n_assoc, errors)

        n_analyses_csr = len(offsets) - 1
        n_analyses_tsv = _validate_analyses_tsv(analyses_path, errors)
        if n_analyses_tsv != n_analyses_csr:
            errors.append(
                f"analyses.tsv has {n_analyses_tsv} rows but "
                f"zarr CSR offsets imply {n_analyses_csr} analyses"
            )

        with store.index_connection() as conn:
            _reject_stray_analyses_table(conn, errors)

        data_root = store.arrays(mode="r")

        # Reference-completed stores: validate imputed array and quality table.
        if manifest.completion_state is CompletionState.REFERENCE_COMPLETED:
            _validate_ragged_completion(ragged_path, store, n_assoc, errors)

        if not errors and "top_hits" in data_root:
            _validate_ragged_top_hits(store_path, data_root, errors)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"validation failed: {exc}")
    return ValidationResult(errors=errors)


def _validate_ragged_completion(
    ragged_path: Path,
    store: OpenGWASDBStore,
    n_assoc: int,
    errors: list[str],
) -> None:
    """Validate the imputed mask and completion_quality table in a Reference-Completed store."""
    root = zarr.open_group(str(ragged_path), mode="r")
    if "imputed" not in root:
        errors.append("reference-completed store is missing data.zarr/ragged/imputed")
        return

    imp = root["imputed"][:]
    if len(imp) != n_assoc:
        errors.append(
            f"data.zarr/ragged/imputed has {len(imp)} entries but offsets imply {n_assoc}"
        )
        return

    # imputed values must be 0 or 1
    if not np.all((imp == 0) | (imp == 1)):
        errors.append("data.zarr/ragged/imputed contains values other than 0 and 1")

    # Where imputed=1: z and se must both be present. "Present" is the
    # complement of the plane's own missing marker (spec §15), not `isfinite`
    # of a float: an integer plane holds no NaN to test for.
    codec = StoreCodec(store.manifest.encoding)
    z_missing = codec.missing_mask(root["z"][:])
    se_vals = root["se"][:].astype("float32")
    imp_mask = imp == 1
    if imp_mask.any():
        if np.any(z_missing[imp_mask]):
            errors.append("imputed=1 rows have missing z-scores")
        if not np.all(np.isfinite(se_vals[imp_mask])):
            errors.append("imputed=1 rows have NaN se values")

    # Where z is missing: se must also be missing and imputed must be 0
    if z_missing.any():
        if not np.all(~np.isfinite(se_vals[z_missing])):
            errors.append("missing z rows have finite se values (inconsistent missingness)")
        if np.any(imp[z_missing] == 1):
            errors.append("missing z rows have imputed=1 (inconsistent)")

    with store.index_connection() as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "completion_quality" not in tables:
            errors.append(
                "index.sqlite is missing the completion_quality table "
                "for a reference-completed store"
            )
        else:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(completion_quality)").fetchall()}
            required = COMPLETION_QUALITY_COLUMNS
            missing_cols = required - cols
            if missing_cols:
                errors.append(
                    "completion_quality table is missing columns: "
                    f"{', '.join(sorted(missing_cols))}"
                )


_TOP_HIT_SAMPLE_SIZE = 1000  # cross-validate this many sampled entries against the CSR


def _validate_ragged_top_hits(
    store_path: Path,
    data_root: Any,
    errors: list[str],
) -> None:
    """Validate the top-hit index groups in a ragged zarr store.

    Structural checks (lengths, sort order) are exhaustive.
    CSR cross-validation is sampled to keep validation O(1) for large stores.
    """
    top = data_root["top_hits"]
    csr = RaggedCSRReader(store_path)
    offsets = csr._offsets[:]
    vi_all = csr._variant_index[:].astype(np.int32)
    z_all = csr.z_all()

    for key in top:
        group = top[key]
        required = {
            "analysis_offsets", "variant_index", "analysis_index", "abs_z",
            "z", "se", "p_value",
        }
        missing = sorted(required.difference(group.keys()))
        if missing:
            errors.append(f"top-hit index {key} is missing {', '.join(missing)}")
            continue
        threshold = float(group.attrs.get("threshold", 0))
        vis = group["variant_index"][:].astype(np.int32)
        ais = group["analysis_index"][:].astype(np.int32)
        zs = group["z"][:].astype(np.float32)
        abs_zs = group["abs_z"][:].astype(np.float32)
        analysis_offsets = group["analysis_offsets"][:].astype(np.int64)
        imputed = group["imputed"][:].astype(np.uint8) if "imputed" in group else None

        if not (
            len(vis) == len(ais) == len(zs) == len(abs_zs)
            == len(group["se"]) == len(group["p_value"])
            and (imputed is None or len(imputed) == len(vis))
        ):
            errors.append(f"top-hit index {key} has inconsistent array lengths")
            continue

        n_analyses = len(offsets) - 1
        if (
            len(analysis_offsets) != n_analyses + 1
            or analysis_offsets[0] != 0
            or np.any(analysis_offsets[:-1] > analysis_offsets[1:])
            or analysis_offsets[-1] != len(vis)
        ):
            errors.append(f"top-hit index {key} has invalid analysis offsets")
            continue
        ordered = True
        for analysis_index in range(n_analyses):
            start = int(analysis_offsets[analysis_index])
            end = int(analysis_offsets[analysis_index + 1])
            if np.any(ais[start:end] != analysis_index):
                ordered = False
                break
            if end - start > 1 and np.any(vis[start : end - 1] >= vis[start + 1 : end]):
                ordered = False
                break
        if not ordered:
            errors.append(f"top-hit index {key} has incorrect or non-genomic analysis slices")
            continue
        if "imputed" in csr._root and imputed is None:
            errors.append(f"top-hit index {key} is missing imputed completion status")
            continue
        if imputed is not None and not np.all((imputed == 0) | (imputed == 1)):
            errors.append(f"top-hit index {key} has invalid imputed completion status")
            continue

        # Exhaustive: all abs_z must match |z|
        if not np.allclose(abs_zs, np.abs(zs), rtol=1e-3, atol=1e-3):
            errors.append(f"top-hit index {key} abs_z inconsistent with z")

        # Sampled cross-validation against the CSR
        n = len(vis)
        if n == 0:
            continue
        rng = np.random.default_rng(0)
        sample = rng.choice(n, size=min(_TOP_HIT_SAMPLE_SIZE, n), replace=False)
        for idx in sample.tolist():
            ai = int(ais[idx])
            vi = int(vis[idx])
            start = int(offsets[ai])
            end = int(offsets[ai + 1])
            pos = start + int(np.searchsorted(vi_all[start:end], vi))
            if pos >= end or int(vi_all[pos]) != vi:
                errors.append(f"top-hit index {key} references missing association")
                break
            stored_z = float(z_all[pos])
            if not np.isclose(stored_z, float(zs[idx]), rtol=1e-3, atol=1e-3):
                errors.append(f"top-hit index {key} z value inconsistent with CSR")
                break
            if p_value_from_z(stored_z) > threshold:
                errors.append(f"top-hit index {key} contains association above threshold")
                break


def _validate_hybrid_store(store: OpenGWASDBStore, errors: list[str]) -> ValidationResult:
    """Validate a Hybrid store (ADR 0026 / issue 059).

    Reuses the dense component validator on ``<store>/dense`` and adds only the
    hybrid-specific invariants: the shared union table covers both components, the
    on-panel/off-panel partition is disjoint, and the overflow is observed-only.
    """
    from opengwasdb.model.enums import AssociationCoverage

    store_path = store.path
    manifest = store.manifest
    if manifest.association_coverage is not AssociationCoverage.FULL:
        errors.append(
            "hybrid store must have association_coverage=full "
            f"(got {manifest.association_coverage.value})"
        )

    dense_dir = dense_component_path(store_path)
    ragged_path = store_path / "data.zarr" / "ragged"
    for label, p in [
        ("dense/ (Dense Component)", dense_dir),
        ("dense/manifest.json", dense_dir / "manifest.json"),
        ("dense/dense_to_shared.npy", dense_to_shared_path(store_path)),
        ("index.sqlite", store_path / "index.sqlite"),
        ("variants.tsv.gz", variant_table_path(store_path)),
        ("variants.tsv.gz.tbi", variant_tabix_path(store_path)),
        ("variant_alid_bytes.npy", variant_alid_bytes_path(store_path)),
        ("variant_alid_rows.npy", variant_alid_rows_path(store_path)),
        ("data.zarr/ragged (Ragged Overflow)", ragged_path),
    ]:
        if not p.exists():
            errors.append(f"missing {label}")
    _validate_closed_envelope(store_path, HYBRID_ENVELOPE, errors)
    if errors:
        return ValidationResult(errors=errors)

    # 1. Dense Component — reuse the dense validator unchanged, but with the
    #    Dense Component's own envelope (it additionally carries
    #    dense_to_shared.npy, which a standalone Dense release never does).
    dense_store = _load_manifest(dense_dir, errors)
    if dense_store is None:
        return ValidationResult(errors=errors)
    if dense_store.manifest.primary_layout is not PrimaryStorageLayout.DENSE:
        errors.append(
            f"Dense Component manifest must be primary_layout=dense "
            f"(got {dense_store.manifest.primary_layout.value})"
        )
        return ValidationResult(errors=errors)
    # One plan for both components (issue #119). They partition one Analysis's
    # associations, so a result assembled from both is only coherent if they
    # were encoded the same way -- and each component carries its own manifest,
    # so nothing but this check couples them.
    #
    # `eaf_reference` is the one field that legitimately differs: it records a
    # physical fact about the component that declares it, and only the Dense
    # Component has imputed cells for a reference frequency to describe
    # (ADR 0037 §4). Compared with it normalised away, so a real disagreement
    # about the *encoding* still fails.
    if dense_store.manifest.encoding.with_eaf_reference(False) != manifest.encoding:
        errors.append(
            "the Dense Component and the Hybrid release declare different statistic "
            f"encodings ({dense_store.manifest.encoding.to_manifest()} vs "
            f"{manifest.encoding.to_manifest()}); both components of a Hybrid release "
            "are written under one plan"
        )
        return ValidationResult(errors=errors)
    before = len(errors)
    _validate_dense_store(dense_store, errors, envelope=HYBRID_DENSE_COMPONENT_ENVELOPE)
    if len(errors) > before:
        return ValidationResult(errors=errors)

    # 2. Shared union table structural checks.
    try:
        with store.index_connection() as connection:
            variant_axis = VariantAxis(store_path, connection)
            try:
                n_shared = _validate_variant_axis(variant_axis, errors)
                _validate_sqlite(connection, n_shared, errors)
            finally:
                variant_axis.close()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"hybrid shared-table validation failed: {exc}")
        return ValidationResult(errors=errors)
    if errors:
        return ValidationResult(errors=errors)

    # 3. Ragged Overflow structural checks + observed-only invariant.
    _validate_overflow(store_path, ragged_path, n_shared, errors, manifest.encoding)
    if errors:
        return ValidationResult(errors=errors)
    # 4. Hybrid cross-component invariants (coverage + disjoint partition).
    _validate_hybrid_invariants(store_path, dense_dir, n_shared, errors)
    if errors:
        return ValidationResult(errors=errors)

    # 5. Overflow top-hit addressing and CSR consistency.
    data_root = store.arrays(mode="r")
    if "top_hits" in data_root:
        _validate_ragged_top_hits(store_path, data_root, errors)
    return ValidationResult(errors=errors)


def _validate_overflow(
    store_path: Path,
    ragged_path: Path,
    n_shared: int,
    errors: list[str],
    encoding: StoreEncoding,
) -> None:
    """Validate the Ragged Overflow CSR: array lengths, se sign, shared-index
    bounds, and that it is observed-only (never imputed, even after completion)."""
    try:
        root = zarr.open_group(str(ragged_path), mode="r")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cannot open Ragged Overflow CSR: {exc}")
        return
    for name in ("offsets", "variant_index", "z", "se"):
        if name not in root:
            errors.append(f"missing data.zarr/ragged/{name}")
    if errors:
        return
    # Both Hybrid components are written under one plan (issue #119): the
    # Dense Component and the Ragged Overflow partition one Analysis's
    # associations, so a shared result contract needs a shared encoding.
    _validate_encoding_plan(root, encoding, errors, label="data.zarr/ragged")
    if errors:
        return
    offsets = root["offsets"][:]
    n_assoc = int(offsets[-1])
    for name in ("variant_index", "z", "se"):
        if len(root[name]) != n_assoc:
            errors.append(
                f"data.zarr/ragged/{name} has {len(root[name])} entries but "
                f"offsets imply {n_assoc}"
            )
    if "imputed" in root:
        errors.append(
            "Ragged Overflow has an imputed array — the overflow is off-panel and "
            "must never be imputed (always observed)"
        )
    if errors:
        return
    vi = root["variant_index"][:]
    if n_assoc and (int(vi.min()) < 0 or int(vi.max()) >= n_shared):
        errors.append(
            "Ragged Overflow variant_index falls outside the shared variant table "
            f"[0, {n_shared})"
        )
    se_vals = root["se"][:].astype("float32")
    if np.any(np.isfinite(se_vals) & (se_vals < 0)):
        errors.append("Ragged Overflow se contains negative finite values")
    if encoding.z.is_fixed_point:
        raw_z = np.asarray(root["z"][:])
        _overflow_positions_match(
            "Ragged Overflow z",
            np.flatnonzero(raw_z == Z_OVERFLOW).astype(np.int64),
            ZOverflowTable.read(root),
            errors,
        )


def _validate_hybrid_invariants(
    store_path: Path, dense_dir: Path, n_shared: int, errors: list[str]
) -> None:
    """Coverage + disjoint-partition invariants over the shared variant index space."""
    dense_to_shared = np.load(dense_to_shared_path(store_path))
    n_panel = len(dense_to_shared)

    # dense_to_shared must be a strictly-increasing set of valid shared rows.
    if n_panel and (int(dense_to_shared.min()) < 0 or int(dense_to_shared.max()) >= n_shared):
        errors.append("dense_to_shared references rows outside the shared variant table")
        return
    if n_panel != len(np.unique(dense_to_shared)):
        errors.append("dense_to_shared maps two Dense Component rows to one shared row")
        return

    # Disjoint partition: no overflow variant is also a Dense Component (on-panel) row.
    ragged_root = zarr.open_group(str(store_path / "data.zarr" / "ragged"), mode="r")
    overflow_vi = np.unique(ragged_root["variant_index"][:])
    on_panel = np.zeros(n_shared, dtype=bool)
    on_panel[dense_to_shared] = True
    if len(overflow_vi) and np.any(on_panel[overflow_vi]):
        errors.append(
            "disjoint-partition violated: an overflow association references an "
            "on-panel (Dense Component) variant"
        )

    # Coverage: the shared table row each dense row maps to must carry the same
    # ALID as the Dense Component's own variant (sampled for large stores).
    dense_axis = VariantAxis(dense_dir)
    shared_axis = VariantAxis(store_path)
    try:
        for r in _representative_variant_indices(n_panel):
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


def _validate_sqlite(
    connection: sqlite3.Connection,
    n_variants: int,
    errors: list[str],
) -> None:
    duplicates = connection.execute(
        """
        SELECT alid, COUNT(*) AS n
        FROM variants
        GROUP BY alid
        HAVING n > 1
        """
    ).fetchall()
    if duplicates:
        errors.append("duplicate canonical variants in variants table")
    alias_rows = connection.execute("SELECT alias, variant_index FROM variant_aliases").fetchall()
    for row in alias_rows:
        variant_index = int(row["variant_index"])
        if variant_index < 0 or variant_index >= n_variants:
            errors.append(f"alias {row['alias']!r} points to missing variant {variant_index}")
    _reject_stray_analyses_table(connection, errors)


def _reject_stray_analyses_table(connection: sqlite3.Connection, errors: list[str]) -> None:
    """Every layout shares this rule (ADR 0034, issue #72): analyses.tsv is
    the sole source of truth for Analytical Metadata, so a leftover SQL
    ``analyses`` table -- Dense/Hybrid retired theirs in issue #68, Ragged in
    issue #69 -- is a validation failure, not a harmless relic."""
    tables = {
        r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if "analyses" in tables:
        errors.append(
            "index.sqlite still contains an analyses table -- analyses.tsv is the sole "
            "source of truth for Analytical Metadata (ADR 0030, issue #22)"
        )


def _validate_eaf_orientation(
    store: OpenGWASDBStore, errors: list[str], warnings: list[str]
) -> None:
    """Every Analysis storing EAF must carry orientation evidence (issue #115).

    Checks the *recorded* evidence, not the reference panel: a Store Release is
    meant to be interpretable on its own, long after the panel it was built
    against has moved or been superseded. `opengwasdb audit-eaf-orientation`
    re-runs the correlation when a panel is supplied.

    A blank `eaf_orientation` on an Analysis whose `eaf_scope` is `association`
    is an error, not a shrug: it means the store was built before the check
    existed, and its frequency column has never been confirmed to describe the
    allele the store says it does -- which is exactly the state
    `gwas-catalog-eur-hybrid` was in. `unverified` is a warning: honestly
    recorded, still not evidence of correctness.
    """
    analyses_path = store.analyses_path
    if not analyses_path.exists():
        return
    try:
        table = read_analyses(analyses_path)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        errors.append(f"cannot read analyses.tsv for the EAF orientation check: {exc}")
        return
    # `eaf_scope` is per Analysis and the encoding plan is per store: two
    # granularities of the same fact, so they must agree. Their *disagreement*
    # is what got through review on #106 -- `analyses.tsv` declared
    # `eaf_scope=association` for a store that had no `eaf` array at all.
    declares_association = [
        str(row.get("analysis_id", ""))
        for row in table.rows
        if row.get("eaf_scope", "") == EafScope.ASSOCIATION.value
    ]
    if store.manifest.encoding.eaf.is_absent and declares_association:
        errors.append(
            f"{len(declares_association)} analysis/analyses declare "
            f"eaf_scope=association (first {declares_association[0]!r}) but the manifest's "
            "encoding declares no eaf plane; the store's metadata and its arrays disagree "
            "about whether it holds frequencies (ADR 0036, ADR 0037)"
        )
    declared_eaf = store.manifest.encoding.eaf
    if declared_eaf.kind in ("float32", "int8_residual") and not declares_association:
        errors.append(
            "the manifest declares an eaf plane but no Analysis declares "
            "eaf_scope=association; a release stores frequencies for the Analyses that "
            "say they have them, or for none"
        )

    recorded = store.manifest.provenance.get("eaf_orientation") or {}
    by_id = {
        str(entry.get("analysis_id", "")): entry for entry in recorded.get("analyses", [])
    }
    for row in table.rows:
        analysis_id = row.get("analysis_id", "")
        if row.get("eaf_scope", "") != EafScope.ASSOCIATION.value:
            continue
        outcome = row.get("eaf_orientation", "")
        if not outcome:
            errors.append(
                f"analysis {analysis_id!r} stores per-association EAF but carries no "
                "eaf_orientation evidence -- its frequencies have never been checked "
                "against a reference, so a column reported against the other allele "
                "would be indistinguishable from a correct one (issue #115). Rebuild "
                "the store, or audit it with a reference panel"
            )
            continue
        if outcome == EafOrientationOutcome.FAILED.value:
            errors.append(
                f"analysis {analysis_id!r} recorded eaf_orientation=failed "
                f"(r = {row.get('eaf_orientation_r', '?')}): its stored EAF is reported "
                "against the other allele and must not be read as a frequency of the "
                "stored effect allele"
            )
        elif outcome == EafOrientationOutcome.UNVERIFIED.value:
            entry = by_id.get(analysis_id, {})
            warnings.append(
                f"analysis {analysis_id!r} stores EAF whose orientation is unverified"
                + (f": {entry['note']}" if entry.get("note") else "")
            )
        entry = by_id.get(analysis_id)
        if entry is None:
            errors.append(
                f"analysis {analysis_id!r} records eaf_orientation={outcome!r} in "
                "analyses.tsv, but manifest.json's provenance has no orientation "
                "evidence for it -- the two must agree on what was checked"
            )
        elif str(entry.get("outcome", "")) != outcome:
            errors.append(
                f"analysis {analysis_id!r} records eaf_orientation={outcome!r} in "
                f"analyses.tsv but {entry.get('outcome')!r} in manifest.json provenance"
            )


def _validate_analyses_tsv(analyses_path: Path, errors: list[str]) -> int:
    """Validate `analyses.tsv` (store-format spec §20, ADR 0030) and return its
    row count -- the `n_analyses` the rest of dense validation cross-checks
    zarr array shapes against.

    Deliberately narrower than `opengwasdb.model.analyses.validate_analyses`
    (issue #16): that function's `REQUIRED_COLUMNS` enforces non-blank
    `sample_size_kind`/`sample_size_scope`/`sample_size`/`original_effect_scale`
    per row, which is the contract a *registry* manifest must satisfy before a
    build even runs. No current build path here has a source for those fields
    (issue #22 scope note), so a built store leaves them honestly blank; a
    store-level validator that required them would fail every store this
    package can currently build. This function instead checks only what a
    *built* store can actually promise: structural integrity (one row per
    `analysis_index`, no duplicate `analysis_id`) and vocabulary conformance
    for whichever controlled-vocabulary columns are present.
    """
    table = read_analyses(analyses_path)
    n = len(table.rows)
    retired_present = [c for c in RETIRED_ANALYSIS_COLUMNS if c in table.fieldnames]
    if retired_present:
        errors.append(
            "analyses.tsv carries retired column(s) "
            f"{', '.join(retired_present)} -- a Store Release built against a "
            "prior column list is not valid against the current schema "
            "(store-format spec §7a, ADR 0034/0035)"
        )
    seen_index: set[int] = set()
    seen_id: set[str] = set()
    for row in table.rows:
        raw_index = row.get("analysis_index", "")
        try:
            index = int(raw_index)
        except ValueError:
            errors.append(f"analyses.tsv row has a non-integer analysis_index {raw_index!r}")
            continue
        if index in seen_index:
            errors.append(f"analyses.tsv has more than one row for analysis_index {index}")
        seen_index.add(index)
        analysis_id = row.get("analysis_id", "")
        if analysis_id in seen_id:
            errors.append(f"analyses.tsv has more than one row for analysis_id {analysis_id!r}")
        seen_id.add(analysis_id)
    if seen_index and seen_index != set(range(n)):
        errors.append(f"analyses.tsv analysis_index values are not exactly the range 0..{n - 1}")
    for row in table.rows:
        for column, vocabulary in (
            ("stored_effect_scale", StoredEffectScale),
            ("original_sd_method", OriginalSdMethod),
            ("ancestry_assignment_method", AncestryAssignmentMethod),
            ("eaf_orientation", EafOrientationOutcome),
        ):
            value = row.get(column, "")
            if not value:
                continue
            try:
                vocabulary(value)
            except ValueError:
                errors.append(
                    f"analyses.tsv analysis {row.get('analysis_id')!r} has invalid "
                    f"{column} {value!r}"
                )
    # Top-Hit Counts (ADR 0032) are computed by every build path, unlike the
    # completion_* rollups above (Reference-Completed releases only) -- so,
    # unlike those, their presence is checked unconditionally here.
    missing_hit_columns = [c for c in TOP_HIT_COUNT_COLUMNS if c not in table.fieldnames]
    if missing_hit_columns:
        errors.append(
            f"analyses.tsv is missing Top-Hit Count column(s): {', '.join(missing_hit_columns)}"
        )
    else:
        for row in table.rows:
            for column in TOP_HIT_COUNT_COLUMNS:
                value = row.get(column, "")
                if not value.isdigit():
                    errors.append(
                        f"analyses.tsv analysis {row.get('analysis_id')!r} has invalid "
                        f"{column} {value!r} (must be a non-negative integer)"
                    )
    return n


def _validate_rsid_index(
    variant_axis: VariantAxis, records: list[VariantRecord], errors: list[str]
) -> None:
    """Every rsid in `variants.tsv.gz` must be resolvable (issue #109).

    The failure this exists to catch is silent by construction: a store whose
    variant table is full of rsids but whose index is empty answers every rsid
    lookup with "nothing here", which reads exactly like a real negative. That
    was the state of every Store Release built before #109, and it was also
    reintroduced once afterwards, by a completion path that rewrote the
    variant table and dropped the rsids on the way through.

    A store with no rsids at all is fine -- not every source names variants --
    so this compares the index against the table rather than demanding either
    be non-empty. A *missing* index is likewise only an error when there are
    rsids to index: stores built before #109 have none of these sidecars, and
    an rsid-less one of those is not wrong, merely older. That is why the
    index is not in any layout's required-entry list, only its envelope.
    """
    # Counted through the same predicate `_write_rsid_index` indexes by, so a
    # correctly-built store cannot fail its own validator over an identifier
    # that was deliberately left out.
    named = sum(1 for record in records if is_indexable_rsid(record.rsid))
    indexed = variant_axis._rsid_bytes
    if indexed is None:
        if named:
            errors.append(
                f"variants.tsv.gz names {named} rsid(s) but the store has no rsid search "
                "index — rebuild the store, or every rsid lookup against it returns nothing"
            )
        return
    if len(indexed) != named:
        errors.append(
            f"variant_rsid_bytes.npy has {len(indexed)} entries but variants.tsv.gz "
            f"names {named} rsid(s)"
        )


def _validate_variant_axis(variant_axis: VariantAxis, errors: list[str]) -> int:
    records = variant_axis.all()
    if variant_axis.n_variants != len(records):
        errors.append(
            f"variant_offsets.npy has {variant_axis.n_variants} rows but "
            f"variants.tsv.gz has {len(records)} rows"
        )
    if variant_axis._alid_bytes is not None and len(variant_axis._alid_bytes) != len(records):
        errors.append(
            f"variant_alid_bytes.npy has {len(variant_axis._alid_bytes)} entries but "
            f"variants.tsv.gz has {len(records)} rows"
        )
    _validate_rsid_index(variant_axis, records, errors)
    seen_alids: set[str] = set()
    for expected_index, record in enumerate(records):
        if record.variant_index != expected_index:
            errors.append(
                f"variant table row {expected_index} has variant_index {record.variant_index}"
            )
            break
        if record.alid in seen_alids:
            errors.append("duplicate canonical variants in variant table")
            break
        seen_alids.add(record.alid)
    for expected_index in _representative_variant_indices(len(records)):
        record = records[expected_index]
        try:
            offset_record = variant_axis.by_index(expected_index)
        except ValueError as exc:
            errors.append(str(exc))
            break
        if offset_record != record:
            errors.append(f"variant offset for row {expected_index} points to a different row")
            break
        fetched = variant_axis.range(record.chromosome, record.position, record.position)
        if record not in fetched:
            errors.append(f"tabix index cannot fetch variant {record.alid}")
            break
    return len(records)


def _representative_variant_indices(n_variants: int) -> list[int]:
    if n_variants <= 0:
        return []
    if n_variants <= 1000:
        return list(range(n_variants))
    anchors = {0, n_variants // 2, n_variants - 1}
    step = max(1, n_variants // 997)
    anchors.update(range(0, n_variants, step))
    return sorted(index for index in anchors if 0 <= index < n_variants)


# Row-band height for streaming the dense matrix during validation, so no check
# ever materialises the full (n_variants × n_analyses) array (issue 045).
_VALIDATE_BAND_ROWS = 250_000


def _validate_eaf_plan(
    group: Any, encoding: StoreEncoding, errors: list[str], *, label: str
) -> None:
    """The `eaf` plane, its baseline, its exception table and its reference.

    They are one artifact in up to four arrays, and a component holding some of
    them and not others cannot be decoded: an `int8` residual plane without its
    baseline is meaningless, and an exception cell with no table entry has lost
    its value entirely. The plan says which should be there; this is where that
    stops being a claim.
    """
    declared = encoding.eaf
    present = "eaf" in group
    if declared.is_absent and present:
        errors.append(
            f"{label} carries an eaf plane but the manifest declares eaf absent"
        )
    if not present and declared.kind in ("float32", "int8_residual"):
        errors.append(
            f"{label} declares an eaf encoding of {declared.kind!r} but carries no eaf plane"
        )

    has_baseline = EAF_BASELINE in group
    has_table = EAF_EXCEPTION_INDEX in group or EAF_EXCEPTION_VALUE in group
    if not declared.is_residual:
        if has_baseline:
            errors.append(
                f"{label} carries {EAF_BASELINE} but declares no residual-coded eaf plane"
            )
        if has_table:
            errors.append(
                f"{label} carries an eaf exception table but declares no residual-coded "
                "eaf plane"
            )
    else:
        if not has_baseline:
            errors.append(
                f"{label} declares a residual-coded eaf plane but is missing {EAF_BASELINE}, "
                "without which the plane cannot be decoded"
            )
        elif not _baseline_in_unit_interval(group[EAF_BASELINE]):
            errors.append(f"{label}/{EAF_BASELINE} holds a finite value outside (0, 1)")
        if EAF_EXCEPTION_INDEX not in group or EAF_EXCEPTION_VALUE not in group:
            errors.append(
                f"{label} declares a residual-coded eaf plane but is missing "
                f"{EAF_EXCEPTION_INDEX}/{EAF_EXCEPTION_VALUE}, which hold the exact values "
                "of any cell the residual coding cannot express"
            )
        else:
            table = EafExceptionTable.read(group)
            if len(table.index) != len(table.value):
                errors.append(
                    f"{label} eaf exception table has {len(table.index)} positions but "
                    f"{len(table.value)} values"
                )
            elif len(table.index) > 1 and np.any(table.index[1:] <= table.index[:-1]):
                errors.append(
                    f"{label} eaf exception table is not sorted by position, or repeats one"
                )
            elif len(table.value) and not np.all(
                np.isfinite(table.value) & (table.value >= 0.0) & (table.value <= 1.0)
            ):
                errors.append(
                    f"{label} eaf exception table holds a value that is not a frequency "
                    "in [0, 1]"
                )

    has_reference = EAF_REFERENCE in group
    if declared.reference and not has_reference:
        errors.append(
            f"{label} declares reference EAF for its imputed cells but carries no "
            f"{EAF_REFERENCE} array"
        )
    if declared.reference and "imputed" not in group:
        errors.append(
            f"{label} declares reference EAF but carries no imputed mask to say which "
            "cells it describes (spec §9, §15)"
        )
    if has_reference and not declared.reference:
        errors.append(
            f"{label} carries {EAF_REFERENCE} but the manifest does not declare it; a "
            "reader following the plan would never read those frequencies"
        )


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


def _baseline_in_unit_interval(array: Any) -> bool:
    values = np.asarray(array[:], dtype=np.float64)
    finite = np.isfinite(values)
    return not bool(np.any(finite & ((values <= 0.0) | (values >= 1.0))))


def _validate_encoding_plan(
    group: Any, encoding: StoreEncoding, errors: list[str], *, label: str
) -> None:
    """Check the arrays against the plan the manifest declares (issue #119).

    The plan is authoritative: a reader decodes what the manifest says, so a
    store whose `z` plane is not in its declared encoding does not read as
    "the other encoding" -- it reads as plausible, wrong z-scores. That is the
    disagreement between declared metadata and actual arrays that got through
    review once already (#106), and it is why the check exists at all.
    """
    for name, declared_plane in (
        ("z", encoding.z), ("se", encoding.se), ("eaf", encoding.eaf)
    ):
        if name not in group:
            continue
        actual = str(group[name].dtype)
        if actual != declared_plane.dtype:
            errors.append(
                f"{label}/{name} has dtype {actual} but the manifest declares "
                f"{declared_plane.kind} ({declared_plane.dtype})"
            )
    _validate_eaf_plan(group, encoding, errors, label=label)
    if errors or "z" not in group:
        return
    declared = encoding.z
    has_table = Z_OVERFLOW_INDEX in group or Z_OVERFLOW_VALUE in group
    if not declared.is_fixed_point:
        if has_table:
            errors.append(
                f"{label} carries a z overflow table but declares no fixed-point z encoding"
            )
        return
    if Z_OVERFLOW_INDEX not in group or Z_OVERFLOW_VALUE not in group:
        errors.append(
            f"{label} declares fixed-point z but is missing "
            f"{Z_OVERFLOW_INDEX}/{Z_OVERFLOW_VALUE}, which hold the exact values of "
            "any cell outside the representable range"
        )
        return
    table = ZOverflowTable.read(group)
    if len(table.index) != len(table.value):
        errors.append(
            f"{label} z overflow table has {len(table.index)} positions but "
            f"{len(table.value)} values"
        )
    elif len(table.index) > 1 and np.any(table.index[1:] <= table.index[:-1]):
        errors.append(f"{label} z overflow table is not sorted by position, or repeats one")
    elif len(table.index) and not np.all(np.isfinite(table.value)):
        errors.append(f"{label} z overflow table contains non-finite values")


def _overflow_positions_match(
    label: str,
    observed: np.ndarray,
    table: ZOverflowTable | EafExceptionTable,
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


def _validate_dense_arrays(
    root: Any,
    n_variants: int,
    n_analyses: int,
    errors: list[str],
    encoding: StoreEncoding,
    imputed_arr: Any = None,
    on_panel: np.ndarray | None = None,
) -> None:
    """Stream the dense z/se (and, for a completed store, imputed) matrices in
    row-bands and run every matrix-touching content check in one pass.

    ``imputed_arr``/``on_panel`` are supplied only for Reference-Completed
    stores (from ``_validate_completion_metadata``); when present, the imputed
    checks that used to load the whole matrix are folded into this same band
    loop so peak memory stays at one band rather than the full matrices
    (issue 045).
    """
    for name in ("z", "se"):
        if name not in root:
            errors.append(f"missing data.zarr/{name}")
    if errors:
        return
    z_arr = root["z"]
    se_arr = root["se"]
    expected_shape = (n_variants, n_analyses)
    if tuple(z_arr.shape) != expected_shape:
        errors.append(f"z shape {tuple(z_arr.shape)} does not match {expected_shape}")
    if tuple(se_arr.shape) != expected_shape:
        errors.append(f"se shape {tuple(se_arr.shape)} does not match {expected_shape}")
    # `eaf` is optional (ADR 0036) -- absent on every store built before it, and
    # on any build whose sources report no frequency -- but when present it is a
    # third parallel plane and must have the same shape as z/se.
    eaf_arr = root["eaf"] if "eaf" in root else None
    if eaf_arr is not None and tuple(eaf_arr.shape) != expected_shape:
        errors.append(f"eaf shape {tuple(eaf_arr.shape)} does not match {expected_shape}")
        eaf_arr = None
    if eaf_arr is not None and EAF_BASELINE in root and len(root[EAF_BASELINE]) != n_variants:
        errors.append(
            f"{EAF_BASELINE} has {len(root[EAF_BASELINE])} entries but the variant axis "
            f"has {n_variants}"
        )
    if eaf_arr is not None and EAF_REFERENCE in root and len(root[EAF_REFERENCE]) != n_variants:
        errors.append(
            f"{EAF_REFERENCE} has {len(root[EAF_REFERENCE])} entries but the variant axis "
            f"has {n_variants}"
        )
    if errors:
        return
    eaf_plane = DenseEafPlane.open(root, encoding) if eaf_arr is not None else None

    # Stream in row-bands. Missingness is read through the plane's declared
    # missing *marker* (spec §15) -- NaN for a float plane, the reserved
    # sentinel for an integer one -- so nothing here decodes a value it does
    # not need, and no band is upcast.
    codec = StoreCodec(encoding)
    fixed_point = encoding.z.is_fixed_point
    overflow_positions: list[np.ndarray] = []
    eaf_exception_positions: list[np.ndarray] = []
    eaf_undecodable = False
    neg_se = False
    missingness = False
    imp_not_binary = False
    imp_nan_z = False
    imp_nan_se = False
    off_panel_imputed = False
    eaf_out_of_range = False
    for r0 in range(0, n_variants, _VALIDATE_BAND_ROWS):
        r1 = min(r0 + _VALIDATE_BAND_ROWS, n_variants)
        z = z_arr[r0:r1]
        se = se_arr[r0:r1]
        z_missing = codec.missing_mask(z)
        if fixed_point:
            overflow_positions.append(
                np.flatnonzero(np.asarray(z) == Z_OVERFLOW).astype(np.int64) + r0 * n_analyses
            )
        if not neg_se and np.any(np.isfinite(se) & (se < 0)):
            neg_se = True
        if eaf_plane is not None:
            # Decoded, not raw: an `int8` residual plane's bytes are codes, and
            # checking those for "in [0, 1]" would pass every store while
            # saying nothing about the frequencies a query returns. A decode
            # that cannot resolve an exception cell is not reported from here:
            # the position check below says exactly which cell and why.
            if not eaf_out_of_range and not eaf_undecodable:
                try:
                    eaf = eaf_plane.band(r0, r1)
                except ValueError:
                    eaf_undecodable = True
                else:
                    finite = np.isfinite(eaf)
                    if np.any(finite & ((eaf < 0.0) | (eaf > 1.0))):
                        eaf_out_of_range = True
            if encoding.eaf.is_residual and eaf_arr is not None:
                eaf_exception_positions.append(
                    np.flatnonzero(np.asarray(eaf_arr[r0:r1]) == EAF_EXCEPTION).astype(np.int64)
                    + r0 * n_analyses
                )
        if not missingness and np.any(z_missing != np.isnan(se)):
            missingness = True
        if imputed_arr is not None:
            imp = imputed_arr[r0:r1]
            if not imp_not_binary and not np.all((imp == 0) | (imp == 1)):
                imp_not_binary = True
            imp_mask = imp == 1
            if imp_mask.any():
                if not imp_nan_z and np.any(z_missing[imp_mask]):
                    imp_nan_z = True
                if not imp_nan_se and not np.all(np.isfinite(se[imp_mask])):
                    imp_nan_se = True
                # Off-panel rows can never be imputed (no LD structure).
                if not off_panel_imputed:
                    off_band = on_panel[r0:r1] == 0
                    if off_band.any() and np.any(imp[off_band] == 1):
                        off_panel_imputed = True
    if neg_se:
        errors.append("se contains negative finite values")
    if missingness:
        errors.append("z and se missingness is inconsistent")
    if imp_not_binary:
        errors.append("data.zarr/imputed contains values other than 0 and 1")
    if imp_nan_z:
        errors.append("imputed=1 cells have missing z-scores")
    if imp_nan_se:
        errors.append("imputed=1 cells have NaN se values")
    if off_panel_imputed:
        errors.append(
            "off-panel (on_panel=0) rows have imputed=1 cells — off-panel is never imputable"
        )
    if eaf_out_of_range:
        errors.append("data.zarr/eaf contains finite values outside [0, 1]")
    if fixed_point:
        _overflow_positions_match(
            "data.zarr/z",
            np.concatenate(overflow_positions) if overflow_positions
            else np.empty(0, dtype=np.int64),
            ZOverflowTable.read(root),
            errors,
        )
    if eaf_plane is not None and encoding.eaf.is_residual:
        before = len(errors)
        _overflow_positions_match(
            "data.zarr/eaf",
            np.concatenate(eaf_exception_positions) if eaf_exception_positions
            else np.empty(0, dtype=np.int64),
            EafExceptionTable.read(root),
            errors,
            what="eaf exception",
            table_name="eaf exception",
        )
        if eaf_undecodable and len(errors) == before:
            errors.append("data.zarr/eaf cannot be decoded under the declared plan")


def _validate_top_hits(root: Any, errors: list[str], encoding: StoreEncoding) -> None:
    if "top_hits" not in root:
        return
    top = root["top_hits"]
    z_plane = DenseZPlane.open(root, encoding)
    z_arr = z_plane.array
    imputed_arr = root["imputed"] if "imputed" in root else None
    n_variants, n_analyses = int(z_arr.shape[0]), int(z_arr.shape[1])

    # Load each threshold group's (comparatively small) index arrays and run the
    # checks that don't need the matrix: array-length consistency, descending-|z|
    # ranking, and cell uniqueness. Keep the survivors for the streamed matrix pass.
    groups: dict[str, dict[str, Any]] = {}
    for key in top:
        group = top[key]
        threshold = float(group.attrs.get("threshold", key.replace("p_", "").replace("_", "-")))
        required = {
            "analysis_offsets", "variant_index", "analysis_index", "abs_z",
            "z", "se", "p_value",
        }
        missing = sorted(required.difference(group.keys()))
        if missing:
            errors.append(f"top-hit index {key} is missing {', '.join(missing)}")
            continue
        rows = group["variant_index"][:].astype(np.int64)
        cols = group["analysis_index"][:].astype(np.int64)
        z_values = group["z"][:].astype("float32")
        abs_z = group["abs_z"][:].astype("float32")
        offsets = group["analysis_offsets"][:].astype(np.int64)
        imputed_values = (
            group["imputed"][:].astype(np.uint8) if "imputed" in group else None
        )
        if imputed_arr is not None and imputed_values is None:
            errors.append(f"top-hit index {key} is missing imputed completion status")
            continue
        if (
            len(rows) != len(cols)
            or len(rows) != len(z_values)
            or len(rows) != len(abs_z)
            or len(rows) != len(group["se"])
            or len(rows) != len(group["p_value"])
            or (imputed_values is not None and len(rows) != len(imputed_values))
        ):
            errors.append(f"top-hit index {key} has inconsistent array lengths")
            continue
        if (
            len(offsets) != n_analyses + 1
            or len(offsets) == 0
            or offsets[0] != 0
            or np.any(offsets[:-1] > offsets[1:])
            or offsets[-1] != len(rows)
        ):
            errors.append(f"top-hit index {key} has invalid analysis offsets")
            continue
        ordered = True
        for analysis_index in range(n_analyses):
            start, stop = int(offsets[analysis_index]), int(offsets[analysis_index + 1])
            if np.any(cols[start:stop] != analysis_index):
                ordered = False
                break
            if stop - start > 1 and np.any(rows[start : stop - 1] >= rows[start + 1 : stop]):
                ordered = False
                break
        if not ordered:
            errors.append(f"top-hit index {key} has incorrect or non-genomic analysis slices")
            continue
        if imputed_values is not None and not np.all((imputed_values == 0) | (imputed_values == 1)):
            errors.append(f"top-hit index {key} has invalid imputed completion status")
            continue
        if len(rows):
            flat = rows * n_analyses + cols
            if len(np.unique(flat)) != len(flat):
                errors.append(f"top-hit index {key} does not match stored z values")
                continue
        groups[key] = {
            "z_crit": z_critical(threshold),
            "rows": rows,
            "cols": cols,
            "z_values": z_values,
            "imputed_values": imputed_values,
            "n": len(rows),
            "pass_count": 0,
            "consistent": True,
            "imputed_consistent": True,
        }
    if not groups:
        return

    # Single streamed pass over the matrix in row-bands: per band, count cells that
    # pass each threshold (completeness) and verify every index cell in that band
    # (matrix z finite, matches the stored index z, and itself clears the cutoff).
    for r0 in range(0, n_variants, _VALIDATE_BAND_ROWS):
        r1 = min(r0 + _VALIDATE_BAND_ROWS, n_variants)
        z_band = z_plane.band(r0, r1)
        abs_band = np.abs(z_band)
        for g in groups.values():
            g["pass_count"] += int(np.count_nonzero(abs_band >= g["z_crit"]))
            in_band = (g["rows"] >= r0) & (g["rows"] < r1)
            if not np.any(in_band):
                continue
            gathered = z_band[g["rows"][in_band] - r0, g["cols"][in_band]]
            matches = np.isclose(gathered, g["z_values"][in_band], rtol=1e-3, atol=1e-3)
            if not (
                np.all(np.isfinite(gathered))
                and np.all(matches)
                and np.all(np.abs(gathered) >= g["z_crit"])
            ):
                g["consistent"] = False
            if imputed_arr is not None and g["imputed_values"] is not None:
                imputed_band = imputed_arr[r0:r1]
                gathered_imputed = imputed_band[
                    g["rows"][in_band] - r0, g["cols"][in_band]
                ].astype(np.uint8)
                if not np.array_equal(gathered_imputed, g["imputed_values"][in_band]):
                    g["imputed_consistent"] = False

    for key, g in groups.items():
        if not g["consistent"]:
            errors.append(f"top-hit index {key} contains z value inconsistent with z array")
        elif not g["imputed_consistent"]:
            errors.append(
                f"top-hit index {key} contains imputed value inconsistent with imputed array"
            )
        elif g["n"] != g["pass_count"]:
            errors.append(f"top-hit index {key} does not match stored z values")


_RHO_REQUIRED_ATTRS = (
    "z_thresh", "min_nulls", "grid_n", "window_bp",
    "n_variants_used", "observed_only", "method", "n_analyses",
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
        errors.append(
            f"rho finiteness is inconsistent with n_null support (min_nulls={min_nulls})"
        )
    if finite.any():
        vals = rho_vals[finite]
        if np.any(vals < -1.0) or np.any(vals > 1.0):
            errors.append("rho contains values outside [-1, 1]")

    n_variants_used = int(group.attrs["n_variants_used"])
    variant_index = group["variant_index"]
    if len(variant_index) != n_variants_used:
        errors.append(
            f"rho variant_index has {len(variant_index)} entries but "
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
            yield (assoc.analysis_id, v.chromosome, int(v.position), a1, a2,
                   float(assoc.z), float(assoc.se))


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
    store_path = store.path
    manifest = store.manifest
    units = _source_units(source, errors)
    if errors or not units:
        return
    rng = np.random.default_rng(seed)
    samples = _sample_sources(units, n_samples, rng)
    if not samples:
        errors.append("source-fidelity: no associations could be read from the source")
        return

    store_asm = _normalise_assembly(manifest.reference_assembly)
    src_asm = _normalise_assembly(source_assembly) if source_assembly else store_asm

    # Resolve each sampled source variant to a store row.
    src_alids = {f"{c}:{p}:{a1}:{a2}" for (_a, c, p, a1, a2, _z, _s) in samples}
    row_by_alid: dict[str, int] = {}
    if src_asm == store_asm:
        va = VariantAxis(store_path)
        try:
            for alid in src_alids:
                rec = va.by_identifier(alid)
                if rec is not None:
                    row_by_alid[alid] = rec.variant_index
        finally:
            va.close()
    else:
        origin_map = _resolve_via_origin(store_path, src_alids)
        if origin_map is not None:
            row_by_alid = origin_map  # store carries provenance → no liftover needed
        else:
            from opengwasdb.build.liftover import build_liftover_lookup

            tuples = {(c, p, a1, a2) for (_a, c, p, a1, a2, _z, _s) in samples}
            lut = build_liftover_lookup(
                tuples, from_build=src_asm, to_build=store_asm,
                failure_threshold=1.0, chain_file=chain_file,
            )
            va = VariantAxis(store_path)
            try:
                for (c, p, a1, a2) in tuples:
                    hg38 = lut.get((c, p, a1, a2))
                    if hg38 is None:
                        continue
                    rec = va.by_identifier(hg38)
                    if rec is not None:
                        row_by_alid[f"{c}:{p}:{a1}:{a2}"] = rec.variant_index
            finally:
                va.close()

    # Resolve analysis ids to columns.
    col_by_aid = {
        row["analysis_id"]: int(row["analysis_index"])
        for row in read_analyses(store_path / "analyses.tsv").rows
    }

    pairs: list[tuple[int, int, float, float, str]] = []
    n_unmatched_variant = 0
    n_unmatched_analysis = 0
    for (aid, c, p, a1, a2, z, se) in samples:
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
        return

    # Gather store z/se for the matched cells and compare (bounded block).
    root = store.arrays(mode="r")
    urows = sorted({pr[0] for pr in pairs})
    ucols = sorted({pr[1] for pr in pairs})
    ri = {r: i for i, r in enumerate(urows)}
    ci = {c: i for i, c in enumerate(ucols)}
    codec = StoreCodec(store.manifest.encoding)
    z_blk = DenseZPlane.open(root, store.manifest.encoding).block(urows, ucols)
    se_blk = root["se"].oindex[urows, :][:, ucols].astype("float32")

    n_compared = 0
    n_missing_in_store = 0
    mismatches: list[str] = []
    for (row, col, z_src, se_src, alid) in pairs:
        sz = z_blk[ri[row], ci[col]]
        sse = se_blk[ri[row], ci[col]]
        if not (np.isfinite(sz) and np.isfinite(sse)):
            n_missing_in_store += 1
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


def default_top_hit_key(threshold: float = 5e-8) -> str:
    return threshold_key(threshold)
