"""`build-dense-top-hits`: its `--n-workers` pool, its scan's band, and its refusal
of a release it cannot write an index into (issue #313).

The index is the same whichever worker count builds it and whatever band the
scan reads, and that band comes from the decoded-band budget
`opengwasdb.encoding.banding` shares with validation. A 0.1.0 release -- Zarr v2,
so nothing the v3 index's arrays can be added to -- is refused before anything is
queued; a 0.2.0 release stays indexable whatever `CURRENT_FORMAT_VERSION` says.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import zarr
from conftest import SOURCE_HEADER
from dense_sources import build_dense_fixture, write_release_manifest
from legacy_fixtures import relayout_as_0_1_0
from typer.testing import CliRunner

from opengwasdb.cli.main import app
from opengwasdb.encoding import banding
from opengwasdb.encoding.banding import BYTES_PER_CELL
from opengwasdb.layouts.dense.top_hits import build_top_hit_indexes
from opengwasdb.top_hits import TOP_HIT_TIER_ARRAYS

_N_ANALYSES = 2
_N_VARIANTS = 200
_ROW_CHUNK = 20

#: A budget of 40 rows per band for this fixture: several bands, and well under
#: the 250,000-row cap the budget never reaches here.
_FORTY_ROW_BUDGET = 40 * BYTES_PER_CELL * _N_ANALYSES

#: `(variant row, Analysis column, stored z)`: hits in different bands, spread
#: over the significance tiers. The second is negative because a tier is decided
#: by |z|.
_HITS = [
    (10, 0, 8.0),
    (50, 1, -3.6),
    (125, 0, 6.0),
    (170, 1, 4.0),
    (190, 0, 5.0),
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

_BAND_LINE = re.compile(r"band \d+ of \d+, (\d+) rows from variant (\d+)")
_PLAN_LINE = re.compile(r"(\d+) rows per band, (\d+)-byte decoded-band budget")


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


@pytest.fixture
def forty_row_bands(monkeypatch: pytest.MonkeyPatch) -> int:
    """Shrink the decoded-band budget to this fixture's 40-row band.

    The budget is the knob a caller with a different one passes, so replacing it
    here is the same move as `build_top_hit_indexes(working_bytes=...)` and puts
    the scan in several bands without a fixture large enough to reach the cap.
    """
    monkeypatch.setattr(banding, "DECODED_BAND_BYTES", _FORTY_ROW_BUDGET)
    return _FORTY_ROW_BUDGET


def _copy(source: Path, destination: Path) -> Path:
    """The same release twice, so two runs index byte-identical input."""
    shutil.copytree(source, destination)
    return destination


def _store(
    tmp_path: Path,
    name: str,
    *,
    n_analyses: int = _N_ANALYSES,
    n_variants: int = _N_VARIANTS,
    row_chunk: int = _ROW_CHUNK,
    with_hits: bool = True,
) -> Path:
    """A Dense release with a manifest, a `z`/`se`/`eaf` plane and some hits."""
    store = tmp_path / name
    store.mkdir()
    write_release_manifest(store, store_id="banded")
    root = zarr.open_group(str(store / "data.zarr"), mode="w", zarr_format=3)
    rng = np.random.default_rng(20261107)
    z = rng.normal(0.0, 0.5, (n_variants, n_analyses)).astype(np.float16)
    if with_hits:
        for row, column, value in _HITS:
            if row < n_variants:
                z[row, column % n_analyses] = np.float16(value)
    chunks = (row_chunk, n_analyses)
    root.create_array("z", data=z, chunks=chunks)
    root.create_array(
        "se", data=np.full((n_variants, n_analyses), 0.1, dtype=np.float16), chunks=chunks
    )
    root.create_array(
        "eaf",
        data=rng.uniform(0.05, 0.95, (n_variants, n_analyses)).astype(np.float32),
        chunks=chunks,
    )
    return store


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def _bands_read(caplog: pytest.LogCaptureFixture) -> list[tuple[int, int]]:
    """Every band the scan read, as `(rows, first variant)`, in scan order."""
    found = [_BAND_LINE.search(message) for message in _messages(caplog)]
    return [(int(match.group(1)), int(match.group(2))) for match in found if match]


def _plan(caplog: pytest.LogCaptureFixture) -> tuple[int, int]:
    """The band and the budget the scan logged, as `(rows, budget)`."""
    for message in _messages(caplog):
        match = _PLAN_LINE.search(message)
        if match:
            return int(match.group(1)), int(match.group(2))
    raise AssertionError(f"no scan plan in {_messages(caplog)}")


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


def test_the_index_is_identical_at_one_and_three_workers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, forty_row_bands: int
) -> None:
    source = _store(tmp_path, "source")
    serial = _copy(source, tmp_path / "serial")
    parallel = _copy(source, tmp_path / "parallel")

    read: dict[str, list[tuple[int, int]]] = {}
    for label, store, workers in (("serial", serial, "1"), ("parallel", parallel, "3")):
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="opengwasdb.layouts.dense.top_hits"):
            result = _run(str(store), "--n-workers", workers)
        assert result.exit_code == 0, result.output
        read[label] = _bands_read(caplog)

    # Each run's own bands, not both runs' lines pooled: a run in one band would
    # make the worker count immaterial.
    for label, bands in read.items():
        assert len(bands) > 1, (label, bands)
        assert [start for _, start in bands] == sorted({start for _, start in bands})
        for rows, _ in bands:
            assert rows * _N_ANALYSES * BYTES_PER_CELL <= forty_row_bands, (label, rows)
    assert len({row // read["serial"][0][0] for row, _, _ in _HITS}) >= 3

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


def test_the_scan_logs_the_worker_count_and_the_band_the_budget_gives(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, "logged")
    plans: list[tuple[int, int]] = []
    for budget, workers in ((20 * BYTES_PER_CELL * _N_ANALYSES, "3"), (_FORTY_ROW_BUDGET, "1")):
        monkeypatch.setattr(banding, "DECODED_BAND_BYTES", budget)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="opengwasdb.layouts.dense.top_hits"):
            result = _run(str(store), "--n-workers", workers)
        assert result.exit_code == 0, result.output
        assert any("n_workers=" + workers in message for message in _messages(caplog))
        plans.append(_plan(caplog))

    # The budget reaches the scan through the module, at call time: a smaller
    # budget is a smaller band, in the scan's own log.
    assert plans == [(20, 20 * BYTES_PER_CELL * _N_ANALYSES), (40, _FORTY_ROW_BUDGET)], plans


def test_a_wide_release_gets_a_narrower_band_than_a_narrow_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    budget = 2 * 1024 * 1024
    narrow = _store(tmp_path, "narrow", n_analyses=2, n_variants=400, row_chunk=50)
    wide = _store(tmp_path, "wide", n_analyses=48, n_variants=400, row_chunk=50)

    bands = {}
    for label, store in (("narrow", narrow), ("wide", wide)):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="opengwasdb.layouts.dense.top_hits"):
            build_top_hit_indexes(store, working_bytes=budget)
        bands[label] = _plan(caplog)[0]

    assert bands["wide"] < bands["narrow"] <= 250_000, bands
    # Each band is as large as its budget allows, not merely smaller: one more
    # whole row chunk of the same width would not fit.
    assert bands["wide"] * 48 * BYTES_PER_CELL <= budget
    assert (bands["wide"] + 50) * 48 * BYTES_PER_CELL > budget


def test_a_narrow_release_keeps_the_row_cap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = _store(tmp_path, "narrow", n_analyses=2, n_variants=400, row_chunk=50)

    with caplog.at_level(logging.INFO, logger="opengwasdb.layouts.dense.top_hits"):
        build_top_hit_indexes(store)

    assert _plan(caplog)[0] == 250_000


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
