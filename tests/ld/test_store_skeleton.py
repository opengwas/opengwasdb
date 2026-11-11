"""Seams 2 and 3 of the walking skeleton: `StoreWriter` → `StoreReader`, and the Test Store script.

The Test Store is written once per module (about a second) from the Test Panel's `reference_r(10)`;
every expected value here is the reference r, quantised and signed to A1 by `ogld.identity.orient`,
or a literal from spec §6.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import testdata
import zarr
from script_modules import load_script
from store_copies import FLOOR, R_SCALE, codes_at, stored_at

from ogld.format import NEAR_BP, InputError, REncoding
from ogld.identity import orient
from ogld.store import StoreReader, StoreWriter, layout

K = 10
#: The Test Store's encoding, which the stores written here share.
TEST_STORE_ENCODING = REncoding(R_SCALE)


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return load_script("make_test_store").write_test_store(
        K, tmp_path_factory.mktemp("s") / "store", R_SCALE
    )


def passing() -> npt.NDArray[np.int64]:
    return np.flatnonzero(testdata.variants().passes_filters).astype(np.int64)


def signs() -> npt.NDArray[np.float64]:
    v = testdata.variants()
    return np.array(
        [
            1.0
            if orient(str(v.chrom[r]), int(v.pos[r]), str(v.ref[r]), str(v.alt[r])).a1_is_alt
            else -1.0
            for r in range(len(v))
        ]
    )


def arrays(group: zarr.Group, prefix: str = "") -> Iterator[tuple[str, zarr.Array]]:
    for name, member in group.members():
        if isinstance(member, zarr.Array):
            yield prefix + name, member
        else:
            yield from arrays(member, f"{prefix}{name}/")


def test_the_fixture_reaches_every_array_kind(written: Path) -> None:
    names = {name for name, _ in arrays(zarr.open_group(written, mode="r"))}
    expected = {f"variants/{c}" for c in ("position", "alid", "a1", "a2", "a1_frequency")}
    for chrom in ("21", "22"):
        expected |= {f"near/{chrom}/{a}" for a in ("indptr", "r")}
        expected |= {f"index/{chrom}/{a}" for a in ("indptr", "partner", "r")}
    assert names == expected


def test_every_array_has_the_fill_value_and_codecs_spec_6_says(written: Path) -> None:
    fills = {
        "indptr": -1,
        "partner": -1,
        "r": -32768,
        "position": 0,
        "alid": "",
        "a1": "",
        "a2": "",
    }
    for name, array in arrays(zarr.open_group(written, mode="r")):
        leaf = name.rsplit("/", 1)[1]
        if leaf == "a1_frequency":
            assert np.isnan(array.fill_value), name
        else:
            assert array.fill_value == fills[leaf], name
        meta = array.metadata.to_dict()["codecs"][0]
        assert meta["name"] == "sharding_indexed", name
        inner = [c["name"] for c in meta["configuration"]["codecs"]]
        assert inner == (
            ["vlen-utf8", "zstd"] if leaf in {"alid", "a1", "a2"} else ["bytes", "zstd"]
        ), name
        assert [c["name"] for c in meta["configuration"]["index_codecs"]] == ["bytes", "crc32c"], (
            name
        )


def test_the_variants_read_back_as_the_engine_forms_them(written: Path) -> None:
    reader = StoreReader.open(written)
    v = testdata.variants()
    rows = passing()
    found = [orient(str(v.chrom[r]), int(v.pos[r]), str(v.ref[r]), str(v.alt[r])) for r in rows]
    table = reader.variants(np.arange(len(rows)))
    assert list(table.alid) == [f.alid for f in found]
    assert list(table.a1) == [f.a1 for f in found]
    assert list(table.a2) == [f.a2 for f in found]
    np.testing.assert_array_equal(table.position, [f.position for f in found])
    assert (np.diff(table.position[table.chrom == "22"]) >= 0).all()
    np.testing.assert_array_equal(reader.rows_of([f.alid for f in found]), np.arange(len(rows)))
    assert reader.rows_of(["22:1:A:G", "nonsense"]).tolist() == [-1, -1]


def test_the_whole_store_reads_back_the_reference_codes_exactly(written: Path) -> None:
    reader = StoreReader.open(written)
    rows = passing()
    ref = testdata.reference_r(K)
    where = np.full(len(testdata.variants()), -1)
    where[rows] = np.arange(len(rows))
    sign = signs()
    i, j = where[ref.row_i], where[ref.row_j]
    pos = testdata.variants().pos
    distance = pos[ref.row_j] - pos[ref.row_i]
    code = codes_at(ref.r * sign[ref.row_i] * sign[ref.row_j])
    held = (distance <= NEAR_BP) | (np.abs(code) >= FLOOR)
    assert (np.abs(code) >= FLOOR)[distance > NEAR_BP].any(), "the fixture needs a Far Pair"
    assert ((distance > NEAR_BP) & (np.abs(code) < FLOOR)).any(), "the fixture needs a Bound"
    expected = np.full((len(rows), len(rows)), np.nan)
    expected[i[held], j[held]] = code[held] / R_SCALE
    expected[j[held], i[held]] = code[held] / R_SCALE
    np.fill_diagonal(expected, 1.0)
    np.testing.assert_array_equal(reader.pair_r(np.arange(len(rows))), expected)


def test_the_proxy_index_holds_each_pair_both_ways_at_the_threshold(written: Path) -> None:
    reader = StoreReader.open(written)
    n = reader.provenance["n_variants"]
    index = reader.index(np.arange(n))
    held = np.repeat(np.arange(n), np.diff(index.indptr))
    assert len(index.partner) == reader.provenance["counts"]["index_entries"] > 100
    assert (np.abs(codes_at(index.r)) >= FLOOR).all()
    forward = dict(zip(zip(held, index.partner, strict=True), index.r, strict=True))
    assert all(forward[(m, j)] == r for (j, m), r in forward.items())


def test_an_unwritten_near_band_chunk_reads_as_the_reserved_code(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "g", mode="w", zarr_format=3)
    length = layout.NEAR_R[1] + 5
    array = layout.new_r_array(group, "r", length, layout.NEAR_R, TEST_STORE_ENCODING)
    assert array.fill_value == -32768
    assert (np.asarray(array[:10]) == -32768).all()
    assert (np.asarray(array[layout.NEAR_R[1] :]) == -32768).all()
    with pytest.raises(ValueError, match="reserved"):
        TEST_STORE_ENCODING.dequantise(np.asarray(array[:10]))


def test_a_store_without_its_provenance_is_not_read_as_a_short_one(
    tmp_path: Path, written: Path
) -> None:
    variants = StoreReader.open(written).variants(np.arange(3))
    StoreWriter.create(
        tmp_path / "half",
        variants=variants,
        near_bp=NEAR_BP,
        far_bp=10_000_000,
        encoding=TEST_STORE_ENCODING,
        n=771,
        k=K,
    )
    with pytest.raises(ValueError, match="provenance"):
        StoreReader.open(tmp_path / "half")


def test_the_script_writes_the_same_arrays_and_the_same_provenance_again(tmp_path: Path) -> None:
    script = load_script("make_test_store")
    first = script.write_test_store(0, tmp_path / "one", R_SCALE)
    second = script.write_test_store(0, tmp_path / "two", R_SCALE)
    one, two = zarr.open_group(first, mode="r"), zarr.open_group(second, mode="r")
    names = {name: np.asarray(array[:]) for name, array in arrays(one)}
    assert len(names) == 15, "every array must be compared for this to mean anything"
    again = {name: np.asarray(array[:]) for name, array in arrays(two)}
    assert names.keys() == again.keys()
    for name, array in names.items():
        np.testing.assert_array_equal(again[name], array, err_msg=name)
    assert measured_free(one.attrs["ld_store"]) == measured_free(two.attrs["ld_store"])


def measured_free(provenance: dict[str, Any]) -> dict[str, Any]:
    """The provenance without what a run measures: its time, its cost and its software."""
    kept = {k: v for k, v in provenance.items() if k not in {"build", "software"}}
    kept["build"] = {k: v for k, v in provenance["build"].items() if k == "threads"}
    return kept


# ------------------------------------------------------------------ fix round 1


def store_rows() -> npt.NDArray[np.int64]:
    """Each Test Panel row's row in the Test Store, −1 where the filters drop it."""
    where = np.full(len(testdata.variants()), -1, dtype=np.int64)
    where[passing()] = np.arange(len(passing()))
    return where


@pytest.mark.parametrize("rows", [[-1, 5], [0, 1056], [-1054, 2]])
def test_the_reader_refuses_a_row_outside_the_variant_set(written: Path, rows: list[int]) -> None:
    # rows_of reports Absent as −1; read as an index it would be the last variant.
    reader = StoreReader.open(written)
    assert reader.provenance["n_variants"] == 1056
    given = np.array(rows, dtype=np.int64)
    for read in (reader.pair_r, reader.variants, reader.index, reader.band):
        with pytest.raises(ValueError, match="outside the Variant Set"):
            read(given)


def test_rows_in_refuses_a_label_that_is_not_canonical(written: Path) -> None:
    reader = StoreReader.open(written)
    assert len(reader.rows_in("22", 0, 100_000_000)) == 588
    with pytest.raises(InputError, match="chr22"):
        reader.rows_in("chr22", 0, 100_000_000)
    assert len(reader.rows_in("7", 0, 100_000_000)) == 0, "a canonical chromosome with no variants"


def test_the_band_is_both_sides_of_a_row_within_near_bp(written: Path) -> None:
    reader = StoreReader.open(written)
    ref = testdata.reference_r(K)
    where, sign = store_rows(), signs()
    a, b = where[ref.row_i], where[ref.row_j]
    near = (testdata.variants().pos[ref.row_j] - testdata.variants().pos[ref.row_i]) <= NEAR_BP
    stored = stored_at(ref.r * sign[ref.row_i] * sign[ref.row_j])
    some = reader.rows_in("22", 0, 100_000_000)[[200, 300, 400]]
    band = reader.band(some)
    for n, row in enumerate(some):
        mine = near & ((a == row) | (b == row))
        partners = np.where(a[mine] == row, b[mine], a[mine])
        order = np.argsort(partners)
        got = slice(band.indptr[n], band.indptr[n + 1])
        assert len(partners) > 5, "the row must have Near Band partners for this to mean anything"
        np.testing.assert_array_equal(band.partner[got], partners[order])
        np.testing.assert_array_equal(band.r[got], stored[mine][order])
    assert (band.partner[band.partner < some[1]] != some[1]).all()
    assert (band.partner[band.partner < some[1]]).size > 0, "an upstream partner must exist"


def test_the_proxy_index_is_exactly_the_reference_pairs_at_the_threshold(written: Path) -> None:
    reader = StoreReader.open(written)
    ref = testdata.reference_r(K)
    where, sign = store_rows(), signs()
    code = codes_at(ref.r * sign[ref.row_i] * sign[ref.row_j])
    keep = np.abs(code) >= FLOOR
    assert keep.sum() > 100, "the reference must hold many pairs at the threshold"
    a, b = where[ref.row_i[keep]], where[ref.row_j[keep]]
    expected = {
        (int(x), int(y)): float(c) / R_SCALE for x, y, c in zip(a, b, code[keep], strict=True)
    }
    expected |= {(y, x): r for (x, y), r in expected.items()}
    n = reader.provenance["n_variants"]
    index = reader.index(np.arange(n))
    held = np.repeat(np.arange(n), np.diff(index.indptr))
    found = {
        (int(i), int(j)): float(r) for i, j, r in zip(held, index.partner, index.r, strict=True)
    }
    assert found == expected


def test_a1_frequency_is_the_a1_share_of_the_dosages(written: Path) -> None:
    d = testdata.dosages()
    rows = d.rows[store_rows()[d.rows] >= 0]
    alt_freq = d.alt_counts[:, np.isin(d.rows, rows)].mean(axis=0) / 2
    is_alt = signs()[rows] > 0
    assert (~is_alt).any(), "the dosage rows must hold an A1 that is REF"
    assert is_alt.any(), "and one that is ALT"
    got = StoreReader.open(written).variants(store_rows()[rows]).a1_frequency
    np.testing.assert_allclose(got, np.where(is_alt, alt_freq, 1 - alt_freq), atol=1e-6)


def test_every_input_is_hashed_in_the_provenance(written: Path) -> None:
    import hashlib

    recorded = StoreReader.open(written).provenance["source"]["sha256"]
    names = ["panel.pgen", "panel.pvar", "panel.psam", "panel.keep", "reference/r.npz"]
    assert sorted(recorded) == sorted(names)
    for name in names:
        digest = hashlib.sha256((testdata.TEST_PANEL / name).read_bytes()).hexdigest()
        assert recorded[name] == digest, name


def test_a_store_that_is_not_one_this_ogld_reads_is_an_input_error(
    tmp_path: Path, written: Path
) -> None:
    import json
    import shutil

    for edit, match in (
        (lambda p: p.update(format_version="9.0.0"), r"9\.0\.0"),
        (lambda p: p["encoding"].update(r_dtype="int8"), "encoding"),
        (lambda p: p.clear(), "provenance|format"),
    ):
        copy = tmp_path / match.replace("|", "_").replace("\\", "")
        shutil.copytree(written, copy)
        doc = json.loads((copy / "zarr.json").read_text())
        edit(doc["attributes"]["ld_store"])
        (copy / "zarr.json").write_text(json.dumps(doc))
        with pytest.raises(InputError, match=match):
            StoreReader.open(copy)
    bare = tmp_path / "bare"
    shutil.copytree(written, bare)
    doc = json.loads((bare / "zarr.json").read_text())
    del doc["attributes"]["ld_store"]
    (bare / "zarr.json").write_text(json.dumps(doc))
    with pytest.raises(InputError, match="provenance"):
        StoreReader.open(bare)


def test_the_build_cost_is_measured_and_the_version_is_real(written: Path) -> None:
    import datetime

    import ogld

    p = StoreReader.open(written).provenance
    made = datetime.datetime.strptime(p["build"]["created_at"], "%Y-%m-%dT%H:%M:%SZ")
    assert datetime.datetime(2026, 11, 8) <= made <= datetime.datetime.now(), "a real time"
    assert p["build"]["wall_seconds"] > 0
    assert p["build"]["peak_memory_bytes"] > 10_000_000
    assert p["software"]["ogld"] == ogld.__version__


def test_the_pc_diagnostic_is_measured_on_the_test_panel_s_cross_chromosome_pairs(
    written: Path,
) -> None:
    import math

    d = StoreReader.open(written).provenance["pc_diagnostic"]
    assert d is not None
    assert d["pairs"]["n_pairs"] == 468 * 588
    assert d["pairs"]["seed"] == 405
    assert "chr21" in d["pairs"]["source"].replace("chromosome 21", "chr21")
    assert d["rate_r2_ge_0_01_sampling"] == pytest.approx(math.erfc(math.sqrt(0.01 * 760 / 2)))
    assert 0.0058 < d["rate_r2_ge_0_01_sampling"] < 0.0059
    for key in ("rate_r2_ge_0_01_unadjusted", "rate_r2_ge_0_01_adjusted"):
        assert 0.0 < d[key] < 0.5, key
    change = d["near_100kb_change"]
    assert list(change) == ["0", "0.5", "0.9", "0.99", "1"]
    assert list(change.values()) == sorted(change.values())
    assert change["0"] < 0 < change["1"], "adjustment moves r both ways"


@pytest.mark.parametrize("k", [0, 10])
def test_the_diagnostic_s_adjusted_r_is_the_reference_r(k: int) -> None:
    # The machinery behind the diagnostic, held to the Test Panel's independent reference.
    script = load_script("make_test_store")
    vs = script.variant_set()
    z = script.unit_columns(vs, k)
    ref = testdata.reference_r(k)
    where = store_rows()
    i, j = where[ref.row_i], where[ref.row_j]
    sign = signs()
    # The 1,056 × 1,056 matrix of products, then the pairs read from it: the per-pair product
    # (146,593 × 771, about 1 GB of temporaries) took 1.5 s and hit the 5 s budget under load.
    got = (z @ z.T)[i, j]
    np.testing.assert_allclose(got, ref.r * sign[ref.row_i] * sign[ref.row_j], atol=1e-9)
