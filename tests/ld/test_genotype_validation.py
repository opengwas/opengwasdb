"""Build/validate and corruption refusals at the public sidecar boundary (#91)."""

from pathlib import Path

import numpy as np
import pytest
import zarr
from store_copies import private_sidecar_copy

import opengwasdb.ld
from opengwasdb.ld.cli import main


def test_built_sidecar_validates(genotype_store: Path) -> None:
    assert main(["validate", str(genotype_store)]) == 0
    findings = opengwasdb.ld.validate(genotype_store)
    assert len(findings) >= 10
    assert not any(findings.values())


@pytest.mark.parametrize("fault", ["projection", "hash", "missing"])
def test_corrupted_sidecar_fails_validation(
    genotype_store: Path, tmp_path: Path, fault: str
) -> None:
    copy, root, record = private_sidecar_copy(genotype_store, tmp_path)
    if fault == "projection":
        array = np.load(copy / "genotypes/projection.npy", mmap_mode="r+")
        array[0, 0] += 1
        array.flush()
    elif fault == "hash":
        record["filesets"][0]["pgen"]["sha256"] = "0" * 64
    else:
        (copy / "genotypes/norm.npy").unlink()
    root.attrs["genotype_store"] = record
    assert main(["validate", str(copy)]) == 1
    findings = opengwasdb.ld.validate(copy)
    assert findings["genotypes.inputs" if fault == "hash" else "genotypes.arrays"]


def _copy(genotype_store: Path, tmp_path: Path) -> tuple[Path, zarr.Group, dict]:
    """A private copy whose pins point at the original inputs, its companion dropped."""
    return private_sidecar_copy(genotype_store, tmp_path)


def _rewrite(copy: Path, record: dict, name: str, change) -> None:
    """Change one array's contents and record the new file's hash, so later checks are reached."""
    from opengwasdb.ld.store import layout

    path = copy / "genotypes" / f"{name}.npy"
    array = np.load(path, mmap_mode="r+")
    change(array)
    array.flush()
    del array
    record["arrays"][name]["sha256"] = layout.file_sha256(path)


def _save(root: zarr.Group, record: dict) -> None:
    root.attrs["genotype_store"] = record


def test_a_sidecar_claiming_an_ld_store_is_refused(genotype_store: Path, tmp_path: Path) -> None:
    copy, root, _ = _copy(genotype_store, tmp_path)
    root.attrs["ld_store"] = {}
    findings = opengwasdb.ld.validate(copy)
    assert findings["genotypes.provenance"], "a root of both kinds is not a Genotype Store"


def test_an_unsupported_sidecar_version_is_refused(genotype_store: Path, tmp_path: Path) -> None:
    copy, root, record = _copy(genotype_store, tmp_path)
    record["format_version"] = "9.9.9"
    _save(root, record)
    findings = opengwasdb.ld.validate(copy)
    assert findings["genotypes.provenance"], "an unknown Genotype Store version is refused"


@pytest.mark.parametrize("key", ["panel", "adjustment", "filesets", "arrays", "inputs"])
def test_a_provenance_key_of_the_wrong_type_is_refused(
    genotype_store: Path, tmp_path: Path, key: str
) -> None:
    copy, root, record = _copy(genotype_store, tmp_path)
    record[key] = 5
    _save(root, record)
    findings = opengwasdb.ld.validate(copy)
    assert findings["genotypes.provenance"], f"a numeric {key} is not a provenance record"


def test_a_sidecar_recording_no_fileset_is_refused(genotype_store: Path, tmp_path: Path) -> None:
    copy, root, record = _copy(genotype_store, tmp_path)
    record["filesets"] = []
    _save(root, record)
    findings = opengwasdb.ld.validate(copy)
    assert findings["genotypes.provenance"], "a sidecar without a fileset has no inputs"


def test_an_array_whose_shape_disagrees_with_n_is_refused(
    genotype_store: Path, tmp_path: Path
) -> None:
    copy, root, record = _copy(genotype_store, tmp_path)
    record["panel"]["n"] += 1
    _save(root, record)
    findings = opengwasdb.ld.validate(copy)
    assert findings["genotypes.arrays"], "the basis is not n × (k + 1) any more"


def test_a_basis_that_is_not_orthonormal_is_refused(genotype_store: Path, tmp_path: Path) -> None:
    copy, root, record = _copy(genotype_store, tmp_path)
    _rewrite(copy, record, "basis", lambda array: array.__setitem__((slice(None), 0), 0.0))
    _save(root, record)
    findings = opengwasdb.ld.validate(copy)
    assert findings["genotypes.arrays"], "a basis whose columns are not orthonormal is refused"


@pytest.mark.parametrize(
    ("name", "change", "message"),
    [
        ("projection", lambda array: array.__setitem__((0, 0), np.nan), "nonfinite projection"),
        ("norm", lambda array: array.__setitem__(0, np.inf), "nonfinite norm"),
        ("norm", lambda array: array.__setitem__(0, 0.0), "nonpositive residual norm"),
        ("sign", lambda array: array.__setitem__(0, 2.0), "A1 sign"),
        ("fileset", lambda array: array.__setitem__(0, 99), "fileset index"),
        ("pgen_row", lambda array: array.__setitem__(0, -1), "pgen row"),
    ],
)
def test_a_bad_adjustment_value_is_refused(
    genotype_store: Path, tmp_path: Path, name: str, change, message: str
) -> None:
    copy, root, record = _copy(genotype_store, tmp_path)
    _rewrite(copy, record, name, change)
    _save(root, record)
    findings = opengwasdb.ld.validate(copy)
    assert findings["genotypes.arrays"], f"a bad {name} is refused"
    why = findings["genotypes.arrays"][0].message
    assert message in why, why
