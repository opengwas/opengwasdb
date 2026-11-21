"""Kind dispatch, immutable inputs and query refusals at ogld.open (#91)."""

import os
import shutil
import sys
from pathlib import Path

import build_fixtures
import numpy as np
import pytest
import testdata
import zarr
from contract.support import TEST_STORE

import ogld
from ogld.build import build_genotypes
from ogld.genotypes.reader import GenotypeReader
from ogld.store import StoreReader


@pytest.fixture
def private_sidecar(genotype_store: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "sidecar"
    shutil.copytree(genotype_store, copy)
    root = zarr.open_group(copy, mode="a")
    record = root.attrs["genotype_store"]
    for fileset in record["filesets"]:
        for name in ("pgen", "psam", "pvar"):
            pin = fileset[name]
            pin["path"] = os.path.relpath((genotype_store / pin["path"]).resolve(), copy)
    for pin in record["inputs"]:
        pin["path"] = os.path.relpath((genotype_store / pin["path"]).resolve(), copy)
    root.attrs["genotype_store"] = record
    del root.attrs["alid_mmap"]
    return copy


def test_a_wrong_partner_floor_is_not_validated_as_an_old_ld_format(private_sidecar: Path) -> None:
    root = zarr.open_group(private_sidecar, mode="a")
    record = root.attrs["genotype_store"]
    record["partner_code_floor"] += 1
    root.attrs["genotype_store"] = record
    assert ogld.validate(private_sidecar)["genotypes.provenance"]


def test_kind_dispatch_and_explicit_wrong_kind_refusals(genotype_store: Path) -> None:
    assert isinstance(ogld.open(genotype_store)._reader, GenotypeReader)
    assert type(ogld.open(TEST_STORE)._reader) is StoreReader
    with pytest.raises(ogld.InputError, match="LD Store"):
        StoreReader.open(genotype_store)
    with pytest.raises(ogld.InputError, match="Genotype Store"):
        GenotypeReader.open(TEST_STORE)


def test_sidecar_and_inputs_move_together(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    pgen = build_fixtures.copy_panel(tree / "panel")
    sidecar = tree / "sidecar"
    build_genotypes(build_fixtures.inputs(pgen), sidecar)
    recorded = zarr.open_group(sidecar, mode="r").attrs["genotype_store"]
    for fileset in recorded["filesets"]:
        for name in ("pgen", "psam", "pvar"):
            assert fileset[name]["path"].startswith("../panel.")
    names = build_fixtures.alids(np.flatnonzero(testdata.variants().passes_filters)[:10])
    before = ogld.open(sidecar).matrix(names)
    moved = tmp_path / "moved"
    tree.rename(moved)
    after = ogld.open(moved / "sidecar").matrix(names)
    np.testing.assert_array_equal(before.r, after.r)
    assert not any(ogld.validate(moved / "sidecar").values())


@pytest.mark.parametrize("fault", ["missing", "size", "mtime_ns"])
def test_changed_or_missing_pgen_is_named_on_open(private_sidecar: Path, fault: str) -> None:
    root = zarr.open_group(private_sidecar, mode="a")
    record = root.attrs["genotype_store"]
    pin = record["filesets"][0]["pgen"]
    if fault == "missing":
        pin["path"] = "missing-panel.pgen"
    else:
        pin[fault] += 1
    root.attrs["genotype_store"] = record
    with pytest.raises(ogld.InputError, match=rf"pgen.*{fault}"):
        ogld.open(private_sidecar)


def test_missing_pgenlib_names_the_build_extra(
    genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pgenlib", None)
    with pytest.raises(ogld.InputError, match=r"pgenlib.*build.*extra"):
        ogld.open(genotype_store)


@pytest.mark.parametrize("query", ["clump", "proxies"])
@pytest.mark.parametrize(
    ("r2", "kb"),
    [
        (-1, 10),
        (float("nan"), 10),
        (0.1, -1),
        (0.1, float("inf")),
        (0.1, 10001),
        (0.001, 3000),
        (0.01, 5000),
    ],
)
def test_threshold_refusals_equal_the_test_store(
    genotype_store: Path, query: str, r2: float, kb: float
) -> None:
    reasons = []
    for path in (TEST_STORE, genotype_store):
        store = ogld.open(path)
        call = getattr(store, query)
        args = (["absent"], [1]) if query == "clump" else (["absent"],)
        with pytest.raises(ogld.Refusal) as refused:
            call(*args, r2=r2, kb=kb)
        reasons.append(refused.value.reason)
    assert reasons[0] == reasons[1]


@pytest.mark.parametrize("radius", [-1, 5000001])
def test_region_refusal_equals_the_test_store(genotype_store: Path, radius: int) -> None:
    reasons = []
    for path in (TEST_STORE, genotype_store):
        with pytest.raises(ogld.Refusal) as refused:
            ogld.open(path).region("22", 25000000, radius)
        reasons.append(refused.value.reason)
    assert reasons[0] == reasons[1]


@pytest.mark.parametrize("score", [[-1], [np.inf], [np.nan], [1, 2]])
def test_invalid_scores_equal_the_test_store(genotype_store: Path, score: list[float]) -> None:
    for path in (TEST_STORE, genotype_store):
        with pytest.raises(ogld.Refusal, match="score"):
            ogld.open(path).clump(["absent"], score, r2=0.1, kb=3000)


def test_missing_threadpoolctl_names_the_build_extra(
    genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = build_fixtures.alids(np.flatnonzero(testdata.variants().passes_filters)[:5])
    store = ogld.open(genotype_store)
    monkeypatch.setitem(sys.modules, "threadpoolctl", None)
    with pytest.raises(ogld.InputError, match=r"threadpoolctl.*build.*extra"):
        store.matrix(names)
