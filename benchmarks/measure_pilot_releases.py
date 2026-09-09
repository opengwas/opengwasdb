#!/usr/bin/env python3
"""Record what the rebuilt pilot Store Releases measure, as one artifact.

ADR 0037's byte tables were measured on the #117 pilot rebuilds and quoted in
the ADR and CHANGELOG, but the measurement itself was never committed: no
artifact said which store, which cells, which encoding plan, or which commit
the B/cell figures came from. This is the repository-side measurement harness
issue #117's evidence needed -- one command that re-records, for every Store
Release named on the command line, the per-store and per-component cell
counts, the encoding plans the release declares, the compressed bytes on disk
(per whole release, per component and per statistic plane), the standalone
validation outcome, the source identity and checksums the release itself
records, and the measured commit and timestamp.

Cells are counted the way the ADR's B/cell figures count them: a Dense
component's cells are the full variant x analysis grid (its planes store one
value per grid cell, missing included), and a Ragged CSR component's cells are
its stored associations (one z/se value per association). `bytes / n_cells`
for the `eaf` plane of a Dense grid is therefore the "rebuilt-pilot plane
B/cell" figure of ADR 0037 section 2, and `bytes / n_cells` for a Ragged
store's per-variant array such as `eaf_reference` is its section 4 figure
(e.g. 3.206 B/cell on completed `eqtlgen`). Every array's `n_cells` is
recorded beside its bytes so a reader can re-derive any ratio without
re-reading the store.

The driver refuses rather than degrades: a store path that is not a Store
Release, a manifest that will not open, or a component with no `z` plane to
count is a hard failure with a message naming the store, and nothing is
written. Validation outcomes are evidence, not gates: a store that fails
standalone validation is still fully measured and its errors recorded, because
the #117 rebuilds predate later validator rules (#127, #135) and an artifact
that silently left such a store out would misrepresent what was measured.

Usage:

    pixi run -e dev python benchmarks/measure_pilot_releases.py STORE [STORE ...] \
        [--output docs/benchmark-output/opengwasdb_pilot_rebuild_measurements.json]
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Any

import zarr

from benchmarks._artifact import provenance, write_artifact
from opengwasdb.model.enums import PrimaryStorageLayout
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store

DEFAULT_OUTPUT = Path(
    "docs/benchmark-output/opengwasdb_pilot_rebuild_measurements.json"
)

#: provenance keys recorded as the release's own account of where its data came
#: from. The set is deliberately layout-independent: builders write different
#: shapes of provenance and the only thing a reader may assume is that these
#: keys, when present, mean what their names say.
_SOURCE_IDENTITY_KEYS = (
    "builder",
    "chain_file",
    "source_release_id",
    "source_besd_prefix",
    "source_build",
    "source_manifest",
)


def _dir_bytes(path: Path) -> int:
    """Apparent compressed bytes under ``path`` (`du -sb`, like the existing
    storage benchmarks): the byte size of the zstd-compressed zarr chunks plus
    their metadata, which is what the ADR's B/cell figures are measured in."""
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(provenance: dict[str, Any]) -> dict[str, str]:
    """The source identity a release records about itself.

    Ragged/SSF builds name their source release and prefix, VCF builds name
    the liftover chain, and EAF-orientation evidence (ADR 0037 section 6) pins
    the reference the release was checked against by id and sha256 checksum.
    What a particular store records varies by layout; keys that are absent are
    simply absent, never defaulted -- a blank reference_checksum is a real
    recorded value and is carried through as the empty string it is.
    """
    identity: dict[str, str] = {}
    for key in _SOURCE_IDENTITY_KEYS:
        if key in provenance:
            identity[key] = str(provenance[key])
    orientation = provenance.get("eaf_orientation")
    if isinstance(orientation, dict):
        for key in ("method", "reference_id", "reference_checksum", "n_sites"):
            if key in orientation:
                identity[f"eaf_orientation.{key}"] = str(orientation[key])
    completion = provenance.get("completion")
    if isinstance(completion, dict):
        for key in ("ancestry", "ld_panel_id", "method"):
            if key in completion:
                identity[f"completion.{key}"] = str(completion[key])
    return identity


def _encoding_record(manifest: StoreManifest) -> dict[str, Any] | None:
    """The encoding plan as written on disk, or ``None`` for a legacy release
    whose absence of a block *is* its plan (float16 throughout, pre-#114)."""
    return manifest.to_dict().get("encoding")


def _zarr_group_bytes(group: zarr.Group, root: Path) -> dict[str, int]:
    """Apparent bytes of every subgroup and group under `group` (e.g. the
    `top_hits` group nested beside the statistic planes), keyed by name."""
    return {name: _dir_bytes(root / name) for name in sorted(group.group_keys())}


def _dense_grid_cells(store_path: Path, data_path: Path, manifest: StoreManifest) -> dict[str, Any]:
    """Measure one Dense grid component (a Dense Store Release, or the nested
    Dense Component of a Hybrid release)."""
    root = zarr.open_group(str(data_path), mode="r")
    if "z" not in root:
        raise SystemExit(f"{store_path}: no data.zarr/z plane to count a Dense component by")
    rows, analyses = (int(n) for n in root["z"].shape)
    arrays = sorted(root.array_keys())
    return {
        "kind": "dense_grid",
        "path": str(data_path.relative_to(store_path)),
        "encoding": _encoding_record(manifest),
        "cells": {
            "n_variants": rows,
            "n_analyses": analyses,
            "n_cells": rows * analyses,
        },
        "data_bytes": _dir_bytes(data_path),
        "group_bytes": _zarr_group_bytes(root, data_path),
        "arrays": {
            name: _array_size(root[name], data_path / name) for name in arrays
        },
    }


def _csr_cells(
    store_path: Path, zarr_root: Path, group_name: str, manifest: StoreManifest
) -> dict[str, Any]:
    """Measure one Ragged CSR component: a Ragged Store Release's arrays at
    ``data.zarr/<group_name>``, or a Hybrid release's Ragged Overflow."""
    group_path = zarr_root / group_name
    root = zarr.open_group(str(group_path), mode="r")
    if "offsets" not in root:
        raise SystemExit(f"{store_path}: no data.zarr/{group_name}/offsets to count a CSR by")
    if root["offsets"].size < 2:
        raise SystemExit(f"{store_path}: data.zarr/{group_name}/offsets has no analyses")
    analyses = int(root["offsets"].size) - 1
    cells = int(root["offsets"][-1])
    arrays = sorted(root.array_keys())
    return {
        "kind": "ragged_csr",
        "path": str(group_path.relative_to(store_path)),
        "encoding": _encoding_record(manifest),
        "cells": {
            "n_analyses": analyses,
            "n_cells": cells,
        },
        "data_bytes": _dir_bytes(group_path),
        "group_bytes": _zarr_group_bytes(root, group_path),
        "arrays": {
            name: _array_size(root[name], group_path / name) for name in arrays
        },
    }


def _array_size(array: zarr.Array, path: Path) -> dict[str, Any]:
    """One plane/array's bytes, element count and bytes-per-cell.

    ``n_cells`` is the array's own element count, so the ratios reproduce the
    ADR's figures without guessing at layout: a Dense grid plane's count is the
    grid's, a Ragged association plane's is the store's stored associations,
    and a per-variant array's (``eaf_baseline``, ``eaf_reference``) is the
    variant count.
    """
    n_cells = int(array.size)
    bytes_ = _dir_bytes(path)
    return {
        "bytes": bytes_,
        "n_cells": n_cells,
        "bytes_per_cell": round(bytes_ / n_cells, 6) if n_cells else None,
    }


def _validation_record(path: Path) -> dict[str, Any]:
    """The standalone validation outcome, verbatim enough to be evidence.

    Errors are recorded in full (there are ever few); warnings are capped in
    the artifact because eaf-orientation audits legitimately emit one per
    Analysis (the metabolome Ragged pilot carries 1,000+).
    """
    result = validate_store(path)
    capped = result.warnings[:50]
    return {
        "ok": result.ok,
        "n_errors": len(result.errors),
        "errors": result.errors,
        "n_warnings": len(result.warnings),
        "warnings": capped,
        "warnings_truncated": len(result.warnings) > len(capped),
    }


def measure_store(path: str | Path) -> dict[str, Any]:
    """Measure one Store Release: identity, cells, encodings, bytes,
    validation and source identity."""
    store_path = Path(path).resolve()
    if not store_path.is_dir():
        raise SystemExit(f"{store_path}: no such Store Release directory")
    manifest_path = store_path / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{store_path}: no manifest.json; not a Store Release")
    analyses_path = store_path / "analyses.tsv"
    if not analyses_path.is_file():
        raise SystemExit(f"{store_path}: no analyses.tsv; not a Store Release")
    release = open_store(store_path)  # raises on an unreadable format or missing manifest
    manifest = release.manifest
    layout = manifest.primary_layout

    if layout is PrimaryStorageLayout.DENSE:
        components = [
            _dense_grid_cells(store_path, store_path / "data.zarr", manifest)
        ]
    elif layout is PrimaryStorageLayout.RAGGED:
        components = [
            _csr_cells(store_path, store_path / "data.zarr", "ragged", manifest)
        ]
    elif layout is PrimaryStorageLayout.HYBRID:
        dense_manifest = StoreManifest.load(store_path / "dense")
        components = [
            _dense_grid_cells(
                store_path, store_path / "dense" / "data.zarr", dense_manifest
            ),
            _csr_cells(store_path, store_path / "data.zarr", "ragged", manifest),
        ]
    else:
        # PrimaryStorageLayout is a closed enum; a new layout must be measured
        # explicitly or it fails loudly rather than publishing empty components.
        raise SystemExit(f"{store_path}: unknown primary_layout {layout!r}")

    cells_total = sum(component["cells"]["n_cells"] for component in components)
    return {
        "path": str(store_path),
        "store_id": manifest.store_id,
        "release_id": manifest.release_id,
        "format_version": manifest.format_version,
        "primary_layout": layout.value,
        "association_coverage": manifest.association_coverage.value,
        "completion_state": manifest.completion_state.value,
        "reference_assembly": manifest.reference_assembly,
        "created_at": manifest.created_at,
        "encoding": _encoding_record(manifest),
        "store_bytes": _dir_bytes(store_path),
        "cells": {"n_cells": cells_total},
        "components": components,
        "validation": _validation_record(store_path),
        "identity": {
            "manifest_sha256": _sha256(manifest_path),
            "analyses_tsv_sha256": _sha256(analyses_path),
            "sources": _source_identity(manifest.provenance),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stores", type=Path, nargs="+", help="Store Release directories")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    records = [measure_store(path) for path in args.stores]
    payload = {
        "artifact": "rebuilt-pilot-releases",
        "generator": "benchmarks/measure_pilot_releases.py",
        **provenance(),
        "stores": records,
    }
    write_artifact(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
