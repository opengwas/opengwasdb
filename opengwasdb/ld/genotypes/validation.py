"""Validate a Genotype Store's canonical variants, immutable inputs and adjustment arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import zarr
from ogquery.ld.genotypes.layout import ARRAYS, VERSION, check_input, mapped_arrays

from opengwasdb.ld.format import VALIDATION_RULES, REncoding
from opengwasdb.ld.store import Finding, layout
from opengwasdb.ld.store.validation import validate_group

_VARIANT_RULES = tuple(
    name
    for name in VALIDATION_RULES
    if name.startswith(("variants.", "alid_index."))
    or name
    in (
        "provenance.encoding",
        "provenance.distances",
        "provenance.threshold",
        "provenance.partner_threshold",
    )
)


def validate_sidecar(
    path: Path, root: zarr.Group, record: dict[str, Any], *, threads: int = 1
) -> dict[str, list[Finding]]:
    """Validate an unpublished candidate using the same checks as on-demand validation."""
    findings = validate_group(root, record, threads=threads, rules=_VARIANT_RULES)
    checks = {
        "genotypes.provenance": _metadata,
        "genotypes.inputs": _inputs,
        "genotypes.arrays": _arrays,
    }
    for name, check in checks.items():
        findings[name] = []
        try:
            check(path, root, record)
        except (OSError, ValueError, TypeError, KeyError, IndexError) as failed:
            findings[name].append(Finding(name, str(path), str(failed)))
    return findings


def _metadata(path: Path, root: zarr.Group, record: dict[str, Any]) -> None:
    if root.metadata.zarr_format != 3 or "ld_store" in root.attrs:
        raise ValueError(f"{path}: needs Zarr v3 and only the genotype_store kind")
    if record["format_version"] != VERSION:
        raise ValueError(f"{path}: unsupported Genotype Store version")
    for key in (
        "source",
        "panel",
        "adjustment",
        "variant_filters",
        "variants_dropped",
        "reference_assembly",
        "software",
        "build",
        "arrays",
        "filesets",
        "inputs",
    ):
        if not isinstance(record[key], (dict, list, str)):
            raise ValueError(f"{path}: invalid provenance {key}")
    if not record["filesets"]:
        raise ValueError(f"{path}: no pgen filesets recorded")
    floor = REncoding.of(record.get("encoding")).partner_code_floor(
        record["t_partner"], record["t_far"]
    )
    if record["partner_code_floor"] != floor:
        raise ValueError(f"{path}: partner_code_floor differs from the recorded thresholds")


def _inputs(path: Path, root: zarr.Group, record: dict[str, Any]) -> None:
    for fileset in record["filesets"]:
        for name in ("pgen", "psam", "pvar"):
            check_input(path, fileset[name], hashes=True)
    for pin in record["inputs"]:
        check_input(path, pin, hashes=True)


def _arrays(path: Path, root: zarr.Group, record: dict[str, Any]) -> None:
    arrays = mapped_arrays(path, record["arrays"])
    n, k, count = record["panel"]["n"], record["adjustment"]["k"], record["n_variants"]
    shapes = {"basis": (n, k + 1), "projection": (count, k + 1)}
    for name in ARRAYS:
        file = path / "genotypes" / f"{name}.npy"
        if arrays[name].shape != shapes.get(name, (count,)):
            raise ValueError(f"{file}: shape does not match n, k and the Variant Set")
        if layout.file_sha256(file) != record["arrays"][name]["sha256"]:
            raise ValueError(f"{file}: SHA-256 changed")
    q = arrays["basis"]
    if not np.isfinite(q).all() or not np.allclose(q.T @ q, np.eye(k + 1), atol=1e-12):
        raise ValueError(f"{path}: invalid covariate basis")
    for first in range(0, count, 8192):
        block = {name: array[first : first + 8192] for name, array in arrays.items()}
        _values(path, block, len(record["filesets"]))


def _values(path: Path, block: dict[str, npt.NDArray[Any]], filesets: int) -> None:
    for name in ("projection", "norm"):
        if not np.isfinite(block[name]).all():
            raise ValueError(f"{path}: nonfinite {name}")
    if (block["norm"] <= 0).any():
        raise ValueError(f"{path}: nonpositive residual norm")
    if not np.isin(block["sign"], [-1, 1]).all():
        raise ValueError(f"{path}: invalid A1 sign")
    files = block["fileset"]
    if ((files < 0) | (files >= filesets)).any():
        raise ValueError(f"{path}: invalid fileset index")
    if (block["pgen_row"] < 0).any():
        raise ValueError(f"{path}: invalid pgen row")
