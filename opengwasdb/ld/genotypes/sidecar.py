"""Local Genotype Store storage: relative input pins and shareable adjustment arrays (ADR 0014)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import zarr

from ogld.format import InputError, Provenance, REncoding, Sourced
from ogld.format.geometry import chromosome_ranges
from ogld.genotypes.adjust import Block, Covariates
from ogld.genotypes.layout import ARRAYS, KEY, VERSION
from ogld.genotypes.validation import validate_sidecar
from ogld.store import layout
from ogld.store.assembly import (
    Cutter,
    Tally,
    derive_alid_index,
    derive_variant_companion,
    provenance,
)
from ogld.store.tables import Variants
from ogld.store.validation import ValidationFailed
from ogld.store.writer import write_variant_table


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
        findings = validate_sidecar(self.path, self.root, made)
        if any(findings.values()):
            raise ValidationFailed(findings)
        self.root.attrs[KEY] = made
        derive_variant_companion(self.path, provenance_key=KEY)
        return cast(Provenance, made)
