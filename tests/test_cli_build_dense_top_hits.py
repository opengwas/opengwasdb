"""`build-dense-top-hits` from the command line: its `--n-workers` process pool,
and its refusal of a release it cannot write an index into (issue #313).

The index is the same whichever worker count builds it, and a 0.1.0 release --
Zarr v2, so nothing the v3 index's arrays can be added to -- is refused before
anything is written. A 0.2.0 release stays indexable: what decides is the
release's on-disk layout, not its `format_version`'s distance from
`CURRENT_FORMAT_VERSION` (ADR 0057).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import zarr
from conftest import SOURCE_HEADER
from dense_sources import build_dense_fixture
from legacy_fixtures import relayout_as_0_1_0
from typer.testing import CliRunner

from opengwasdb.cli.main import app
from opengwasdb.encoding import EafEncoding, SeEncoding, StoreEncoding, ZEncoding
from opengwasdb.model.enums import AssociationCoverage, CompletionState, PrimaryStorageLayout
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.top_hits import TOP_HIT_TIER_ARRAYS

#: `build_top_hit_indexes` scans the `z` plane in bands of `max(z.chunks[0],
#: 250_000)` variants, so this fixture's chunk size is also its band size, and
#: 600,000 variants make three bands.
_BAND_ROWS = 250_000
_N_VARIANTS = 600_000
_N_ANALYSES = 2

#: `(variant row, Analysis column, stored z)`: hits in all three bands, spread
#: over the three significance tiers. The second is negative because a tier is
#: decided by |z|.
_HITS = [
    (10, 0, 8.0),  # band 0
    (123_456, 1, -3.6),  # band 0
    (300_000, 1, 6.0),  # band 1
    (400_000, 0, 4.0),  # band 1
    (550_000, 0, 5.0),  # band 2
]

#: The six associations of `_SOURCE_ROWS`, |z| 8, 6, 11, 7, 5 and 4; 5.45, 4.56
#: and 3.48 are the 5e-8, 5e-6 and 5e-4 |z| cutoffs, so every tier is larger
#: than the one above it.
_SOURCE_ROWS = [
    "a1\tp1\tHeight\tHeight primary\t1\t100\tA\tG\t8.0\t0.1\trs1\tsd",
    "a2\tp2\tDisease\tDisease primary\t1\t100\tA\tG\t6.0\t0.2\trs1\tlog_or",
    "a1\tp1\tHeight\tHeight primary\t1\t200\tT\tC\t11.0\t0.2\trs2\tsd",
    "a2\tp2\tDisease\tDisease primary\t1\t300\tG\tA\t-7.0\t0.5\trs3\tlog_or",
    "a1\tp1\tHeight\tHeight primary\t1\t400\tG\tA\t5.0\t0.3\trs4\tsd",
    "a2\tp2\tDisease\tDisease primary\t1\t500\tA\tG\t4.0\t0.2\trs5\tlog_or",
]


def _release_fixture_store(root: Path) -> Path:
    """The six-association Dense Store Release `_SOURCE_ROWS` describes."""
    source = root / "associations.tsv"
    source.write_text(SOURCE_HEADER + "\n" + "\n".join(_SOURCE_ROWS) + "\n", encoding="utf-8")
    store = root / "release.opengwasdb"
    build_dense_fixture(
        [source],
        store,
        store_id="banded",
        release_id="observed-v1",
        reference_assembly="GRCh37",
    )
    return store


@pytest.fixture(scope="module")
def release(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One Dense release, built once: its readers copy it before writing."""
    yield _release_fixture_store(tmp_path_factory.mktemp("release"))


def _copy(source: Path, destination: Path) -> Path:
    """The same release twice, so two runs index byte-identical input."""
    shutil.copytree(source, destination)
    return destination


def _banded_store(tmp_path: Path, name: str) -> Path:
    """A Dense release whose `z` plane spans three scan bands, hits detached."""
    store = tmp_path / name
    store.mkdir()
    (store / "manifest.json").write_text(
        json.dumps(
            StoreManifest(
                store_id="banded",
                release_id="observed-v1",
                format_version="0.2.0",
                primary_layout=PrimaryStorageLayout.DENSE,
                association_coverage=AssociationCoverage.GWAS,
                completion_state=CompletionState.OBSERVED_ONLY,
                reference_assembly="GRCh37",
                encoding=StoreEncoding(
                    z=ZEncoding("float16"),
                    se=SeEncoding("float16"),
                    eaf=EafEncoding("float32"),
                ),
            ).to_dict()
        ),
        encoding="utf-8",
    )
    root = zarr.open_group(str(store / "data.zarr"), mode="w", zarr_format=3)
    rng = np.random.default_rng(20261107)
    z = rng.normal(0.0, 0.5, (_N_VARIANTS, _N_ANALYSES)).astype(np.float16)
    for row, column, value in _HITS:
        z[row, column] = np.float16(value)
    chunks = (_BAND_ROWS, _N_ANALYSES)
    root.create_array("z", data=z, chunks=chunks)
    root.create_array(
        "se", data=np.full((_N_VARIANTS, _N_ANALYSES), 0.1, dtype=np.float16), chunks=chunks
    )
    root.create_array(
        "eaf",
        data=rng.uniform(0.05, 0.95, (_N_VARIANTS, _N_ANALYSES)).astype(np.float32),
        chunks=chunks,
    )
    return store


def _tier_arrays(store: Path) -> dict[str, np.ndarray]:
    """Every array of every tier, keyed `<tier>/<array>`."""
    root = zarr.open_group(str(store / "data.zarr"), mode="r")
    return {
        f"{tier}/{name}": np.asarray(root[f"top_hits/{tier}/{name}"][:])
        for tier in root["top_hits"]
        for name in root[f"top_hits/{tier}"]
    }


def _index_hits(store: Path) -> dict[str, int]:
    """Each tier's hit count, from the length of its `variant_index`."""
    root = zarr.open_group(str(store / "data.zarr"), mode="r")
    return {
        tier: int(root[f"top_hits/{tier}/variant_index"].shape[0])
        for tier in root["top_hits"]
    }


def _snapshot(release_path: Path) -> dict[str, str]:
    """Every file of a release, by relative path, hashed."""
    return {
        str(path.relative_to(release_path)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(release_path.rglob("*"))
        if path.is_file()
    }


def _run(*arguments: str):
    return CliRunner().invoke(app, ["build-dense-top-hits", *arguments])


def test_the_index_is_identical_at_one_and_three_workers(tmp_path: Path) -> None:
    source = _banded_store(tmp_path, "source")
    serial = _copy(source, tmp_path / "serial")
    parallel = _copy(source, tmp_path / "parallel")

    # The fixture is meaningful only if the scan has several bands with hits in
    # every one: a single band would make the worker count immaterial.
    assert _N_VARIANTS > 2 * _BAND_ROWS
    assert {row // _BAND_ROWS for row, _, _ in _HITS} == {0, 1, 2}

    one = _run(str(serial), "--n-workers", "1")
    assert one.exit_code == 0, one.output
    three = _run(str(parallel), "--n-workers", "3")
    assert three.exit_code == 0, three.output

    serial_arrays = _tier_arrays(serial)
    parallel_arrays = _tier_arrays(parallel)
    assert _index_hits(serial) == {"p_5e_08": 2, "p_5e_06": 3, "p_5e_04": 5}
    assert set(serial_arrays) == {
        f"{tier}/{name}"
        for tier in ("p_5e_08", "p_5e_06", "p_5e_04")
        for name in TOP_HIT_TIER_ARRAYS | {"eaf"}
    }
    assert set(parallel_arrays) == set(serial_arrays)
    for name, values in serial_arrays.items():
        np.testing.assert_array_equal(parallel_arrays[name], values, err_msg=name)


def test_the_scan_log_names_the_requested_worker_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = _banded_store(tmp_path, "logged")
    with caplog.at_level(logging.INFO, logger="opengwasdb.layouts.dense.top_hits"):
        result = _run(str(store), "--n-workers", "3")

    assert result.exit_code == 0, result.output
    messages = [record.getMessage() for record in caplog.records]
    assert f"Top-hit index: scanning {_N_VARIANTS} variants (n_workers=3)" in messages


def test_a_0_1_0_release_is_refused_and_left_untouched(tmp_path: Path, release: Path) -> None:
    legacy = relayout_as_0_1_0(release, tmp_path / "legacy")
    assert json.loads((legacy / "manifest.json").read_text())["format_version"] == "0.1.0"
    assert (legacy / "data.zarr" / ".zgroup").is_file(), "the fixture must be Zarr v2"
    assert not (legacy / "data.zarr" / "zarr.json").exists()
    before = _snapshot(legacy)

    refused = _run(str(legacy))

    assert refused.exit_code != 0
    assert "0.1.0" in refused.output, refused.output
    assert "convert_store_to_0_2_0" in refused.output, refused.output
    assert _snapshot(legacy) == before, "the refused release must be untouched"
    assert not (legacy / "data.zarr" / "zarr.json").exists(), (
        "a v3 zarr.json beside the v2 .zgroup is the corruption being refused"
    )


def test_a_0_2_0_release_is_indexed(tmp_path: Path, release: Path) -> None:
    store = _copy(release, tmp_path / "released")
    assert _index_hits(store) == {"p_5e_08": 4, "p_5e_06": 5, "p_5e_04": 6}, (
        "the fixture's own build must have written the three tiers"
    )
    root = zarr.open_group(str(store / "data.zarr"), mode="a", zarr_format=3)
    del root["top_hits"]

    result = _run(str(store), "--n-workers", "2")

    assert result.exit_code == 0, result.output
    assert _index_hits(store) == {"p_5e_08": 4, "p_5e_06": 5, "p_5e_04": 6}
