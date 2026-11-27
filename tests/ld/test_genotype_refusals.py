"""Validation of an edited sidecar, a sidecar moved with its inputs, and a build without pgenlib
(#91).

The read refusals -- the kind dispatch, a changed or missing pgen, the threshold, region and
score refusals -- are ogquery's (opengwas/ogquery#11), held in its own suite on its own copy of
the Genotype Store."""

import re
import sys
from pathlib import Path

import build_fixtures
import numpy as np
import pytest
import testdata
import zarr
from ogquery.opening import GenotypeExtraMissing
from store_copies import private_sidecar_copy

import opengwasdb.ld
from opengwasdb.ld.build import build_genotypes
from opengwasdb.ld.cli import main


@pytest.fixture
def private_sidecar(genotype_store: Path, tmp_path: Path) -> Path:
    copy, root, record = private_sidecar_copy(genotype_store, tmp_path)
    root.attrs["genotype_store"] = record
    return copy


def test_a_wrong_partner_floor_is_not_validated_as_an_old_ld_format(private_sidecar: Path) -> None:
    root = zarr.open_group(private_sidecar, mode="a")
    record = root.attrs["genotype_store"]
    record["partner_code_floor"] += 1
    root.attrs["genotype_store"] = record
    assert opengwasdb.ld.validate(private_sidecar)["genotypes.provenance"]


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
    before = opengwasdb.ld.open(sidecar).matrix(names)
    moved = tmp_path / "moved"
    tree.rename(moved)
    after = opengwasdb.ld.open(moved / "sidecar").matrix(names)
    np.testing.assert_array_equal(before.r, after.r)
    assert not any(opengwasdb.ld.validate(moved / "sidecar").values())


def test_missing_pgenlib_names_the_extra_a_build_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build that reaches `Genotypes` without pgenlib raises ogquery's `GenotypeExtraMissing`,
    which names `ogquery[genotype]`, the extra `opengwasdb[ld-build]` installs."""
    monkeypatch.setitem(sys.modules, "pgenlib", None)
    pgen = build_fixtures.copy_panel(tmp_path / "panel")
    named = r"pip install 'ogquery\[genotype\]'"
    with pytest.raises(GenotypeExtraMissing, match=named) as missing:
        build_genotypes(build_fixtures.inputs(pgen), tmp_path / "sidecar")
    assert isinstance(missing.value, ImportError)


def test_the_cli_refuses_a_genotype_build_without_pgenlib_naming_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 2 and `opengwasdb ld build: <the extra to install>`, as a missing input is refused,
    not a traceback."""
    monkeypatch.setitem(sys.modules, "pgenlib", None)
    panel = testdata.panel()
    argv = [
        "build", str(tmp_path / "sidecar"), "--genotypes", "--pgen", str(panel.pgen),
        "--keep", str(panel.keep), "--pcs", str(panel.pcs), "--pc-source", "Test Panel",
        "--panel-id", "Test Panel", "--ancestry", "EUR", "--genotype-source", "Test Panel",
        "--r-scale", "511", "--t-partner", "0.01", "--info-min", "none",
    ]  # fmt: skip
    assert main(argv) == 2
    assert re.match(
        r"opengwasdb ld build: .*pip install 'ogquery\[genotype\]'", capsys.readouterr().err
    )
