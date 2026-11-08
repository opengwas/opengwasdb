"""Seams 2 and 3 of the walking skeleton: `StoreWriter` → `StoreReader`, and the Test Store script.

The Test Store is written once per module (about a second) from the Test Panel's `reference_r(10)`;
every expected value here is the reference r, quantised and signed to A1 by `ogld.identity.orient`,
or a literal from spec §6.
"""

from __future__ import annotations

import filecmp
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import testdata
import zarr
from script_modules import load_script

from ogld.format import NEAR_BP, RESERVED_CODE, dequantise, far_code_floor, quantise, t_far
from ogld.identity import orient
from ogld.store import StoreReader, StoreWriter, layout

K = 10
FLOOR = far_code_floor(t_far(771, K))


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return load_script("make_test_store").write_test_store(
        K, tmp_path_factory.mktemp("s") / "store"
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
        "r": RESERVED_CODE,
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
    code = quantise(ref.r * sign[ref.row_i] * sign[ref.row_j])
    held = (distance <= NEAR_BP) | (np.abs(code) >= FLOOR)
    assert (np.abs(code) >= FLOOR)[distance > NEAR_BP].any(), "the fixture needs a Far Pair"
    assert ((distance > NEAR_BP) & (np.abs(code) < FLOOR)).any(), "the fixture needs a Bound"
    expected = np.full((len(rows), len(rows)), np.nan)
    expected[i[held], j[held]] = dequantise(code[held])
    expected[j[held], i[held]] = dequantise(code[held])
    np.fill_diagonal(expected, 1.0)
    np.testing.assert_array_equal(reader.pair_r(np.arange(len(rows))), expected)


def test_the_proxy_index_holds_each_pair_both_ways_at_the_threshold(written: Path) -> None:
    reader = StoreReader.open(written)
    n = reader.provenance["n_variants"]
    index = reader.index(np.arange(n))
    held = np.repeat(np.arange(n), np.diff(index.indptr))
    assert len(index.partner) == reader.provenance["counts"]["index_entries"] > 100
    assert (np.abs(quantise(index.r)) >= FLOOR).all()
    forward = dict(zip(zip(held, index.partner, strict=True), index.r, strict=True))
    assert all(forward[(m, j)] == r for (j, m), r in forward.items())


def test_an_unwritten_near_band_chunk_reads_as_the_reserved_code(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "g", mode="w", zarr_format=3)
    length = layout.NEAR_R[1] + 5
    array = layout.new_r_array(group, "r", length, layout.NEAR_R)
    assert array.fill_value == RESERVED_CODE
    assert (np.asarray(array[:10]) == RESERVED_CODE).all()
    assert (np.asarray(array[layout.NEAR_R[1] :]) == RESERVED_CODE).all()
    with pytest.raises(ValueError, match="reserved"):
        dequantise(np.asarray(array[:10]))


def test_a_store_without_its_provenance_is_not_read_as_a_short_one(
    tmp_path: Path, written: Path
) -> None:
    variants = StoreReader.open(written).variants(np.arange(3))
    StoreWriter.create(tmp_path / "half", variants=variants, near_bp=NEAR_BP, far_bp=10_000_000)
    with pytest.raises(ValueError, match="provenance"):
        StoreReader.open(tmp_path / "half")


def test_the_writer_refuses_an_unfinished_chromosome_naming_it(
    tmp_path: Path, written: Path
) -> None:
    reader = StoreReader.open(written)
    writer = StoreWriter.create(
        tmp_path / "s", variants=reader.variants(np.arange(3)), near_bp=NEAR_BP, far_bp=10_000_000
    )
    with pytest.raises(ValueError, match=r"Near Band of chromosome 21"):
        writer.finish(reader.provenance)


def test_the_writer_refuses_a_near_band_row_of_the_wrong_length(
    tmp_path: Path, written: Path
) -> None:
    reader = StoreReader.open(written)
    rows = np.arange(4)
    writer = StoreWriter.create(
        tmp_path / "s", variants=reader.variants(rows), near_bp=NEAR_BP, far_bp=10_000_000
    )
    with pytest.raises(ValueError, match="every later variant within near_bp"):
        writer.write_near(0, np.array([1, 1, 1, 0]), np.zeros(3, dtype=np.int16))


def test_the_script_is_deterministic_to_the_byte(tmp_path: Path) -> None:
    script = load_script("make_test_store")
    first = script.write_test_store(0, tmp_path / "one")
    second = script.write_test_store(0, tmp_path / "two")
    names = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    assert len(names) > 30, "the store must hold many files for this to mean anything"
    assert names == sorted(p.relative_to(second) for p in second.rglob("*") if p.is_file())
    match, mismatch, errors = filecmp.cmpfiles(first, second, names, shallow=False)
    assert (mismatch, errors) == ([], [])
    assert len(match) == len(names)


def four_row_writer(tmp_path: Path, written: Path) -> StoreWriter:
    """A writer over the Test Store's first four rows, all on chromosome 21, 300 kb apart."""
    variants = StoreReader.open(written).variants(np.arange(4))
    assert set(variants.chrom) == {"21"}, "the fixture's first rows must share a chromosome"
    return StoreWriter.create(tmp_path / "s", variants=variants, near_bp=NEAR_BP, far_bp=10_000_000)


GOOD_LENGTHS = np.array([3, 2, 1, 0], dtype=np.int64)
SIX_CODES = np.arange(6, dtype=np.int16)


@pytest.mark.parametrize(
    ("first_row", "lengths", "codes", "message"),
    [
        (1, GOOD_LENGTHS[1:], SIX_CODES[:3], "in order"),
        (0, np.append(GOOD_LENGTHS, 0), SIX_CODES, "past the end"),
        (0, GOOD_LENGTHS, SIX_CODES[:5], "disagree"),
        (0, GOOD_LENGTHS, np.full(6, RESERVED_CODE, dtype=np.int16), "reserved"),
    ],
)
def test_the_writer_refuses_a_near_band_write_that_is_out_of_order_or_unsound(
    tmp_path: Path,
    written: Path,
    first_row: int,
    lengths: np.ndarray,
    codes: np.ndarray,
    message: str,
) -> None:
    writer = four_row_writer(tmp_path, written)
    with pytest.raises(ValueError, match=message):
        writer.write_near(first_row, lengths, codes)


def test_the_writer_accepts_the_near_band_these_refusals_are_the_exception_to(
    tmp_path: Path, written: Path
) -> None:
    writer = four_row_writer(tmp_path, written)
    writer.write_near(0, GOOD_LENGTHS, SIX_CODES)
    assert (tmp_path / "s" / "near" / "21" / "r").is_dir()


@pytest.mark.parametrize(
    ("chrom", "indptr", "partner", "codes", "message"),
    [
        ("7", [0] * 5, [], [], "not in the Variant Set"),
        ("21", [0, 0], [], [], "indptr must span"),
        ("21", [0, 1, 1, 1, 1], [1], [], "disagree"),
        ("21", [0, 1, 1, 1, 1], [9], [5], "outside chromosome 21"),
        ("21", [0, 1, 1, 1, 1], [1], [RESERVED_CODE], "reserved"),
    ],
)
def test_the_writer_refuses_a_proxy_index_that_does_not_fit_its_chromosome(
    tmp_path: Path,
    written: Path,
    chrom: str,
    indptr: list[int],
    partner: list[int],
    codes: list[int],
    message: str,
) -> None:
    writer = four_row_writer(tmp_path, written)
    with pytest.raises(ValueError, match=message):
        writer.write_index(
            chrom,
            np.array(indptr, dtype=np.int64),
            np.array(partner, dtype=np.int64),
            np.array(codes, dtype=np.int16),
        )
