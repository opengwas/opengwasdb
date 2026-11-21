"""Local Genotype Store storage: relative input pins and shareable adjustment arrays (ADR 0015)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import zarr

from ogld.format import InputError, Provenance, REncoding, Sourced
from ogld.format.geometry import chromosome_ranges
from ogld.genotypes.adjust import Block, Covariates
from ogld.genotypes.pgen import PgenFileset
from ogld.store import layout
from ogld.store.assembly import (
    Cutter,
    Tally,
    derive_alid_index,
    derive_variant_companion,
    provenance,
)
from ogld.store.tables import Variants
from ogld.store.writer import write_variant_table

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
    """Paths move with the sidecar's tree; size/mtime are cheap open-time guards (#91 ruling)."""
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


class SidecarWriter:
    """The shared variant writer plus adjustment data; never writes LD components (#91)."""

    def __init__(
        self,
        root: Path,
        variants: Variants,
        cov: Covariates,
        encoding: REncoding,
        *,
        near_bp: int,
        far_bp: int,
        t_partner: float,
    ) -> None:
        self.path = root
        ranges = chromosome_ranges(variants.chrom)
        self.cutter = Cutter(
            variants.position,
            ranges,
            (near_bp, far_bp),
            encoding,
            n=len(cov.basis),
            k=cov.k,
            t_partner=t_partner,
        )
        self.root = write_variant_table(root, variants, ranges)
        self.root.create_group("genotypes")
        self.count = len(variants.position)
        shapes = {
            "pgen_row": (self.count,),
            "fileset": (self.count,),
            "projection": (self.count, cov.k + 1),
            "norm": (self.count,),
            "sign": (self.count,),
            "basis": cov.basis.shape,
        }
        self.arrays = {
            name: np.lib.format.open_memmap(
                root / "genotypes" / f"{name}.npy", mode="w+", dtype=dtype, shape=shapes[name]
            )
            for name, dtype in ARRAYS.items()
        }
        self.arrays["basis"][:] = cov.basis
        self.next = 0

    def write(self, rows: npt.NDArray[np.int64], fileset: int, block: Block) -> None:
        stop = self.next + len(rows)
        self.arrays["pgen_row"][self.next : stop] = rows
        self.arrays["fileset"][self.next : stop] = fileset
        for name in ("projection", "norm", "sign"):
            self.arrays[name][self.next : stop] = getattr(block, name)
        self.next = stop

    def finish(
        self, sourced: Sourced, filesets: list[dict[str, Any]], inputs: list[dict[str, Any]]
    ) -> Provenance:
        if self.next != self.count:
            raise InputError(f"{self.path}: unwritten adjustment rows from {self.next}")
        descriptors = {}
        for name, array in self.arrays.items():
            array.flush()
            descriptors[name] = {
                "shape": list(array.shape),
                "sha256": layout.file_sha256(self.path / "genotypes" / f"{name}.npy"),
            }
        group = self.root["variants"]
        assert isinstance(group, zarr.Group)
        left = derive_alid_index(group, group.create_group("alid_index"))
        made: dict[str, Any] = dict(
            provenance(sourced, self.cutter, self.count, Tally(), alids_left_out=left)
        )
        made.update(
            {
                "format_version": VERSION,
                "filesets": filesets,
                "inputs": inputs,
                "arrays": descriptors,
            }
        )
        # Counts describe materialised data only; no fictitious zero count of measured pairs.
        made["counts"] = {"alids_left_out": left}
        from ogld.genotypes.validation import validate_sidecar
        from ogld.store.validation import ValidationFailed

        findings = validate_sidecar(self.path, self.root, made)
        if any(findings.values()):
            raise ValidationFailed(findings)
        self.root.attrs[KEY] = made
        derive_variant_companion(self.path, provenance_key=KEY)
        return cast(Provenance, made)
