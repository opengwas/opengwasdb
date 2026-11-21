"""Build/validate and corruption refusals at the public sidecar boundary (#91)."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from test_genotype_store import genotype_store as genotype_store

import ogld
from ogld.cli import main


def test_built_sidecar_validates(genotype_store: Path) -> None:
    assert main(["validate", str(genotype_store)]) == 0
    findings = ogld.validate(genotype_store)
    assert len(findings) >= 10
    assert not any(findings.values())


@pytest.mark.parametrize("fault", ["projection", "hash", "missing"])
def test_corrupted_sidecar_fails_validation(
    genotype_store: Path, tmp_path: Path, fault: str
) -> None:
    copy = tmp_path / "sidecar"
    # Preserve relative input paths when copying the sidecar into a different directory.
    shutil.copytree(genotype_store, copy)
    root = zarr.open_group(copy, mode="a")
    record = dict(root.attrs["genotype_store"])
    for fileset in record["filesets"]:
        for name in ("pgen", "psam", "pvar"):
            pin = fileset[name]
            pin["path"] = os.path.relpath((genotype_store / pin["path"]).resolve(), copy)
    for pin in record["inputs"]:
        pin["path"] = os.path.relpath((genotype_store / pin["path"]).resolve(), copy)
    if fault == "projection":
        array = np.load(copy / "genotypes/projection.npy", mmap_mode="r+")
        array[0, 0] += 1
        array.flush()
    elif fault == "hash":
        record["filesets"][0]["pgen"]["sha256"] = "0" * 64
    else:
        (copy / "genotypes/norm.npy").unlink()
    root.attrs["genotype_store"] = record
    # Root pins deliberately changed: the companion must be re-derived on this private copy.
    del root.attrs["alid_mmap"]
    assert main(["validate", str(copy)]) == 1
    findings = ogld.validate(copy)
    assert findings["genotypes.inputs" if fault == "hash" else "genotypes.arrays"]
