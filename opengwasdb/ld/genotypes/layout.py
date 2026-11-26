"""The Genotype Store sidecar's own layout: root kind, arrays and relative input pins
(opengwas/ogld#91).

It sits below both the writer and validation so either may import it, and neither has to
import the other to name the root key or an array.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from ogquery.ld.format import InputError

from opengwasdb.ld.genotypes.pgen import PgenFileset
from opengwasdb.ld.store import layout

KEY = "genotype_store"
VERSION = "0.1.0"
ARRAYS = {
    "pgen_row": "<i8",
    "fileset": "<i4",
    "projection": "<f8",
    "norm": "<f8",
    "sign": "<f8",
    "basis": "<f8",
}


def input_pin(path: Path, root: Path) -> dict[str, Any]:
    """Paths move with the sidecar's tree; size/mtime are cheap open-time guards (opengwas/ogld#91
    ruling)."""
    stat = path.stat()
    return {
        "path": Path(os.path.relpath(path, root)).as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": layout.file_sha256(path),
    }


def check_input(root: Path, pin: dict[str, Any], *, hashes: bool = False) -> Path:
    path = root / str(pin["path"])
    try:
        stat = path.stat()
    except OSError as missing:
        raise InputError(
            f"{path}: recorded genotype input is missing or unreadable: {missing}"
        ) from missing
    for key, actual in (("size", stat.st_size), ("mtime_ns", stat.st_mtime_ns)):
        if actual != pin[key]:
            raise InputError(f"{path}: {key} changed: recorded {pin[key]}, found {actual}")
    if hashes and layout.file_sha256(path) != pin["sha256"]:
        raise InputError(f"{path}: SHA-256 changed from the recorded genotype input hash")
    return path


@dataclass(frozen=True)
class Fileset:
    """Persisted sample selection, without reparsing a mutable psam at query time."""

    pgen: Path
    psam: Path
    subset: npt.NDArray[np.uint32]
    order: npt.NDArray[np.int64] | None
    samples: int
    panel: str


def fileset_record(fileset: PgenFileset, pvar: Path, root: Path) -> dict[str, Any]:
    return {
        "pgen": input_pin(fileset.pgen, root),
        "psam": input_pin(fileset.psam, root),
        "pvar": input_pin(pvar, root),
        "subset": fileset.subset.tolist(),
        "order": None if fileset.order is None else fileset.order.tolist(),
        "samples": fileset.samples,
        "panel": fileset.panel,
    }


def open_filesets(root: Path, records: list[dict[str, Any]]) -> list[Fileset]:
    out = []
    for record in records:
        paths = {name: check_input(root, record[name]) for name in ("pgen", "psam", "pvar")}
        out.append(
            Fileset(
                paths["pgen"],
                paths["psam"],
                np.asarray(record["subset"], dtype=np.uint32),
                None if record["order"] is None else np.asarray(record["order"], dtype=np.int64),
                record["samples"],
                record["panel"],
            )
        )
    return out


def mapped_arrays(root: Path, metadata: dict[str, Any]) -> dict[str, npt.NDArray[Any]]:
    out = {}
    for name, dtype in ARRAYS.items():
        path = root / "genotypes" / f"{name}.npy"
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as broken:
            raise InputError(
                f"{path}: missing or unreadable Genotype Store array: {broken}"
            ) from broken
        if array.dtype != np.dtype(dtype) or list(array.shape) != metadata[name]["shape"]:
            raise InputError(f"{path}: array dtype or shape differs from its recorded descriptor")
        out[name] = array
    return out
