"""The committed Genotype Store, its recipe, and this repository's build agree (ogquery#11).

`data/test-panel/genotype-store/` is the store ogquery's reader is tested on, built once by
opengwasdb `964ac1b2`; `data/test-panel/genotype-store.SOURCE` records the command, the inputs and
every output's git blob id. This holds the three to each other, so none drifts silently:

* the store on disk is exactly the files the recipe lists, each byte for byte;
* every input the recipe names is the file committed here, so a changed input means the store must
  be built again;
* this repository's build, run now with the recipe's command on those inputs, writes the same
  store: every file the same bytes and every array the same values, but for what a run records of
  itself, and the last bits of the float64 arrays: a QR basis and what is projected on it differ
  by a few ulps with the process's BLAS (2e-16 to 7e-15 here, up to 2 ulps of the array's largest
  element; under the project's `test` task, with `PYTHONNOUSERSITE=1` and one BLAS thread, they
  match to the byte), and the sidecar records each float array's SHA-256, which follows.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
from pathlib import Path
from typing import Any

import genotype_fixture as fixture
import numpy as np
import zarr
from store_copies import arrays
from test_store_k0_source import blob_id

from opengwasdb.ld.cli import main

ROOT = Path(__file__).resolve().parents[2]
SOURCE = fixture.TEST_PANEL / "genotype-store.SOURCE"
DATA = "tests/ld/data/test-panel/"


def lines(kind: str) -> dict[str, str]:
    """The recipe's `<kind> <blob> <path>` lines, path to blob id."""
    text = SOURCE.read_text(encoding="utf-8")
    return {
        path: blob for blob, path in re.findall(rf"^{kind} ([0-9a-f]{{40}}) (\S+)$", text, re.M)
    }


def test_the_store_is_exactly_the_files_the_recipe_lists() -> None:
    outputs = lines("output")
    assert len(outputs) == 28, "the store's files, as ogquery's SOURCE lists them"
    held = {
        p.relative_to(ROOT).as_posix(): blob_id(p)
        for p in fixture.COMMITTED.rglob("*")
        if p.is_file()
    }
    assert held == outputs


def test_every_input_the_recipe_names_is_the_file_committed_here() -> None:
    inputs = lines("input")
    assert len(inputs) == 5, inputs
    changed = {p: b for p, b in inputs.items() if blob_id(ROOT / p) != b}
    assert changed == {}, (
        "an input changed since the store was built: build it again, and ogquery's"
    )


def test_the_blob_id_is_git_s() -> None:
    assert hashlib.sha1(b"blob 0\0").hexdigest() == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    lock = fixture.COMMITTED / "variants" / "alid_mmap" / ".lock"
    assert blob_id(lock) == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def _companion(store: Path) -> Path:
    (directory,) = [p for p in (store / "variants" / "alid_mmap").iterdir() if p.is_dir()]
    return directory


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


#: The arrays the sidecar holds as float64. A few ulps of BLAS drift move them, and so the SHA-256
#: the root records for each; the integer arrays and every input pin's SHA-256 never move.
FLOAT_ARRAYS = ("basis", "norm", "projection", "sign")
#: Float64 drift allowed, in units of the last place of the largest value compared: the QR basis
#: and what is projected on it moved by up to 2 ulps of their largest element here.
ULPS = 8
#: The PC Check's ratios and quantiles, which the arrays' drift moves, derived values: relative
#: 3e-15 at most here, held to 1e-14.
DOCUMENT_RTOL = 1e-14


def _same_floats(first: Any, second: Any, name: str) -> None:
    """`first` and `second` equal to `ULPS` ulps of the larger's largest element."""
    assert first.shape == second.shape, name
    largest = max(float(np.abs(first).max(initial=0.0)), float(np.abs(second).max(initial=0.0)))
    np.testing.assert_allclose(
        first, second, rtol=0, atol=ULPS * np.spacing(largest or 1.0), err_msg=name
    )


def _same_array(first: np.ndarray[Any, Any], second: np.ndarray[Any, Any], name: str) -> None:
    """Integers exactly; float64 to `ULPS` ulps."""
    assert first.dtype == second.dtype and first.shape == second.shape, name
    if first.dtype.kind == "f":
        _same_floats(first, second, name)
    else:
        np.testing.assert_array_equal(first, second, err_msg=name)


def _same_document(first: Any, second: Any, path: str = "") -> None:
    """Two root documents equal, but for what a build records of itself (when and how long it ran,
    the library versions, the ALID companion's generation, which hashes this very document) and
    the float64 arrays' recorded SHA-256, with floats to `DOCUMENT_RTOL`. Every other key, an input
    pin's SHA-256 included, is compared exactly."""
    if isinstance(first, dict):
        assert isinstance(second, dict) and first.keys() == second.keys(), path
        for key in first:
            where = f"{path}/{key}"
            if where in SELF_RECORDED or where.endswith(
                tuple(f"/arrays/{n}/sha256" for n in FLOAT_ARRAYS)
            ):
                continue
            _same_document(first[key], second[key], where)
    elif isinstance(first, list):
        assert isinstance(second, list) and len(first) == len(second), path
        for index, (one, other) in enumerate(zip(first, second, strict=True)):
            _same_document(one, other, f"{path}[{index}]")
    elif isinstance(first, float):
        np.testing.assert_allclose(first, second, rtol=DOCUMENT_RTOL, atol=0, err_msg=path)
    else:
        assert first == second, path


SELF_RECORDED = (
    "/attributes/alid_mmap/generation",
    "/attributes/genotype_store/build",
    "/attributes/genotype_store/software",
)


def _recipe_arguments(tree: Path) -> list[str]:
    """The recipe's `opengwasdb ld` arguments, its inputs and its output placed in `tree`."""
    found = re.search(r"^generated-command opengwasdb ld (.*)$", SOURCE.read_text(), re.M)
    assert found, "the recipe has no generated-command"
    arguments = [
        part.replace(DATA + "genotype-store", str(tree / fixture.COMMITTED.name)).replace(
            DATA, f"{tree}/"
        )
        for part in shlex.split(found.group(1))
    ]
    assert arguments[0] == "build" and "--genotypes" in arguments
    return arguments


def test_this_repository_s_build_writes_the_committed_store(tmp_path: Path) -> None:
    """Built from the inputs at their recorded mtime, with the recipe's own command: the same files,
    the same bytes in each (the ALID companion's manifest apart, which hashes the root document),
    the same arrays in the compressed chunks, and the same root document but for the run's
    record."""
    for name in fixture.BUILT_FROM:
        shutil.copy2(fixture.TEST_PANEL / name, tmp_path / name)
    fixture.pin_mtimes(tmp_path, fixture.recorded_mtime_ns(fixture.COMMITTED))
    assert main(_recipe_arguments(tmp_path)) == 0

    committed, built = fixture.COMMITTED, tmp_path / fixture.COMMITTED.name
    was, now = _companion(committed).name, _companion(built).name
    assert {f.replace(was, now) for f in _files(committed)} == _files(built), "the same files"

    for name in sorted(_files(committed) - {"zarr.json"}):
        other = built / name.replace(was, now)
        if name.endswith(f"alid_mmap/{was}/manifest.json") or "/c/" in name:
            continue  # the companion's manifest hashes the root; chunks are compared as arrays
        if name.startswith("genotypes/") and name.endswith(".npy"):
            _same_array(np.load(committed / name), np.load(other), name)
        else:
            assert (committed / name).read_bytes() == other.read_bytes(), name
    mine, yours = (dict(arrays(zarr.open_group(root, mode="r"))) for root in (committed, built))
    assert mine.keys() == yours.keys() and mine, "the same arrays, and some"
    for key in mine:
        _same_array(np.asarray(mine[key][...]), np.asarray(yours[key][...]), key)
    documents = [json.loads((root / "zarr.json").read_text("utf-8")) for root in (committed, built)]
    _same_document(documents[0], documents[1])
    # What the rebuilt root records for each float64 array is that array's own SHA-256.
    for name in FLOAT_ARRAYS:
        recorded = documents[1]["attributes"]["genotype_store"]["arrays"][name]["sha256"]
        assert (
            recorded
            == hashlib.sha256((built / "genotypes" / f"{name}.npy").read_bytes()).hexdigest()
        )
    manifests = [
        json.loads((_companion(root) / "manifest.json").read_text("utf-8"))
        for root in (committed, built)
    ]
    for manifest in manifests:
        manifest.pop("sources")
    assert manifests[0] == manifests[1]
