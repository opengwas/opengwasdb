"""The Far Pairs beside the Proxy Index (ogld#31; spec §5, §8, §10; ADR 0007).

Each chromosome's `far/<chrom>/` holds its pairs more than near_bp and at most far_bp apart whose
code reaches the far code floor, once each, in the earlier variant's row: the pairs a Proxy Index
held beyond near_bp, in their earlier rows. `matrix` and `region` read those pairs from the Far
Pairs alone; from #72 no store holds an index at all, and a store that still carries a leftover
one answers the same (spec §12).

Every expected value here is read from the store's own arrays with zarr, from the Test Panel's
reference r, or is the answer the committed Test Store gives; never the reader's far part.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import zarr
from ld_script_modules import load_script
from store_copies import TEST_STORE, copy_store, resharded

import opengwasdb.ld
from opengwasdb.ld.format import REncoding
from opengwasdb.ld.format.geometry import near_pair_of
from opengwasdb.ld.store import StoreReader, layout

#: Inner chunks of 512 elements, 4 to a shard, as `test_store_reads.py` re-shards the Test Store.
INNER, PER_SHARD = 512, 4

_Pairs = tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]


def provenance(store: Path) -> dict[str, object]:
    found: dict[str, object] = json.loads((store / "zarr.json").read_text())["attributes"][
        "ld_store"
    ]
    return found


def chromosomes(store: Path) -> list[tuple[str, int, int]]:
    ranges = zarr.open_group(store / "variants", mode="r").attrs["chromosomes"]
    return [(c["name"], c["row_start"], c["row_stop"]) for c in ranges]


def rows_of(store: Path, kind: str, chrom: str) -> _Pairs:
    """`kind/<chrom>/`'s entries as (row, partner, code), local rows, in stored order."""
    group = zarr.open_group(store / kind / chrom, mode="r")
    indptr = np.asarray(group["indptr"][:]).astype(np.int64)
    owner = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    partner = np.asarray(group["partner"][:]).astype(np.int64)
    return owner, partner, np.asarray(group["r"][:]).astype(np.int64)


def write_a_proxy_index(store: Path) -> None:
    """Give `store` the Proxy Index a store written before #72 held, rebuilt from its own Near
    Band and Far Pairs: every pair within far_bp at the far code floor, both ways, ascending by
    (row, partner). No store this package writes has one; it is the input `derive_far_pairs`
    migrates, which `without_far_pairs` below hands it."""
    floor = int(str(provenance(store)["far_code_floor"]))
    far_bp = int(str(provenance(store)["far_bp"]))
    encoding = REncoding.of(provenance(store)["encoding"])
    root = zarr.open_group(store, mode="r+")
    for chrom, start, stop in chromosomes(store):
        rows = stop - start
        position = np.asarray(root["variants/position"][start:stop], dtype=np.int64)
        indptr = np.asarray(root[f"near/{chrom}/indptr"][:], dtype=np.int64)
        codes = np.asarray(root[f"near/{chrom}/r"][:])
        at = np.flatnonzero(np.abs(codes.astype(np.int64)) >= floor)
        lo, hi = near_pair_of(indptr, at)
        owner = [lo.astype(np.int64)]
        partner = [hi.astype(np.int64)]
        keeper = [codes[at].astype(np.int64)]
        far_owner, far_partner, far_code = rows_of(store, "far", chrom)
        owner += [far_owner, far_partner]
        partner += [far_partner, far_owner]
        keeper += [far_code, far_code]
        owner_a = np.concatenate(owner)
        partner_a = np.concatenate(partner)
        code_a = np.concatenate(keeper)
        assert np.abs(position[partner_a] - position[owner_a]).max() <= far_bp, (
            "every index entry is within far_bp"
        )
        order = np.lexsort((partner_a, owner_a))
        counts = np.bincount(owner_a, minlength=rows)
        group = root.create_group(f"index/{chrom}")
        layout.put_numbers(group, "indptr", np.concatenate([[0], np.cumsum(counts)]), -1)
        layout.put_numbers(group, "partner", partner_a[order].astype(np.int32), -1)
        codes_of = code_a[order].astype(encoding.dtype)
        layout.new_r_array(group, "r", len(order), layout.SIDE, encoding)[:] = codes_of


# ------------------------------------------------------------------ what the store holds


def test_the_test_store_holds_its_far_pairs_once_each_and_at_the_floor() -> None:
    counts = provenance(TEST_STORE)["counts"]
    assert isinstance(counts, dict)
    assert counts["far_pairs"] > 0, "the Test Store must hold Far Pairs for this to mean anything"
    held = 0
    for chrom, start, _ in chromosomes(TEST_STORE):
        owner, partner, code = rows_of(TEST_STORE, "far", chrom)
        position = np.asarray(zarr.open_array(TEST_STORE / "variants" / "position", mode="r"))
        near_bp = int(str(provenance(TEST_STORE)["near_bp"]))
        gap = position[start + partner].astype(np.int64) - position[start + owner]
        assert (np.abs(gap) > near_bp).all(), "a Far Pair is beyond near_bp"
        assert (np.abs(code) >= int(str(provenance(TEST_STORE)["far_code_floor"]))).all()
        assert (partner > owner).all(), "each in its earlier row"
        held += len(owner)
    assert held == counts["far_pairs"], "each Far Pair once, as provenance counts them"


# ------------------------------------------------------------------ what a matrix reads


@pytest.fixture(scope="module")
def small_chunks(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return resharded(tmp_path_factory.mktemp("resharded") / "store", INNER, PER_SHARD)


def every_alid() -> list[str]:
    reader = StoreReader.open(TEST_STORE)
    return list(reader.variants(np.arange(reader.provenance["n_variants"])).alid)


#: Regions 10 Mb wide whose variants hold Far Pairs among them: chr21's two at 26.0 and 33.8 Mb,
#: and chr22's three at 33.5–42.3 Mb.
WIDE_REGIONS = (("21", 29_000_000, 5_000_000), ("22", 38_000_000, 5_000_000))


def far_stored(m: opengwasdb.ld.LDMatrix) -> int:
    """The pairs of `m` more than 3 Mb apart that hold a Stored r."""
    gap = np.abs(m.variants.position[:, None] - m.variants.position[None, :])
    return int(((m.kinds() == "stored") & (gap > 3_000_000)).sum()) // 2


def test_a_matrix_and_a_region_answer_alike_from_a_store_holding_a_leftover_index(
    small_chunks: Path, tmp_path: Path
) -> None:
    # A 0.2.0 store derived before #72 still holds `index/`. Nothing reads it (spec §12), so a
    # matrix and a region answer exactly as the store without one does — even when every entry
    # of it is gone, which a read would have to refuse (spec §6).
    copy = tmp_path / "leftover-index"
    shutil.copytree(small_chunks, copy)
    write_a_proxy_index(copy)
    lost = 0
    for chrom, _, _ in chromosomes(copy):
        for name in ("partner", "r"):
            shutil.rmtree(copy / "index" / chrom / name / "c")
            lost += 1
    assert lost == 4, "both chromosomes' index entries must be lost"
    whole, plain = opengwasdb.ld.open(TEST_STORE), opengwasdb.ld.open(copy)
    names = every_alid()
    expected = whole.matrix(names)
    assert far_stored(expected) == 7, "the matrix must hold the Test Store's seven Far Pairs"
    np.testing.assert_array_equal(plain.matrix(names).r, expected.r)
    for chrom, centre, radius in WIDE_REGIONS:
        region = whole.region(chrom, centre, radius)
        assert far_stored(region) >= 2, f"the chr{chrom} region must hold Far Pairs"
        np.testing.assert_array_equal(plain.region(chrom, centre, radius).r, region.r)


def test_a_matrix_fetches_far_chunks_and_no_proxy_index_chunk(small_chunks: Path) -> None:
    # Counted at the chunk cache, which keeps every chunk it fetched at its default budget. The
    # copy holds a leftover index, so a read of it would be seen here as a fetched chunk.
    copy = small_chunks.parent / "with-index"
    if not copy.exists():
        shutil.copytree(small_chunks, copy)
        write_a_proxy_index(copy)
    reader = StoreReader.open(copy)
    rows = np.arange(reader.provenance["n_variants"])
    reader.pair_r(rows)
    for _, start, stop in chromosomes(copy):
        reader.region_r(start, stop)
    # The cache keeps shard indexes beside the chunks since ogld#32; only chunks name an array.
    fetched = {key[0] for key in reader._cache._kept if isinstance(key, tuple)}
    assert {"far/21/partner", "far/22/partner", "far/21/r", "far/22/r"} <= fetched
    assert not [path for path in fetched if path.startswith("index/")], sorted(fetched)


def test_a_store_without_far_pairs_is_refused_naming_them(tmp_path: Path) -> None:
    copy = copy_store(tmp_path / "store")
    shutil.rmtree(copy / "far")
    with pytest.raises(ValueError, match=r"far/2[12]/indptr is missing"):
        opengwasdb.ld.open(copy).matrix(every_alid())


# ------------------------------------------------------------------ stores written before them


def files(store: Path, under: str = "") -> dict[str, bytes]:
    """Every file of `store` under `under`, by path, with its bytes."""
    root = store / under
    return {str(p.relative_to(store)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def without_far_pairs(to: Path) -> Path:
    """A copy of the Test Store as a store written before ADR 0007 holds it: its Proxy Index,
    which no store this code writes has, and no far/."""
    copy = copy_store(to)
    write_a_proxy_index(copy)
    shutil.rmtree(copy / "far")
    return copy


def test_far_pairs_derived_from_the_index_are_the_ones_the_store_writes(tmp_path: Path) -> None:
    # The five HGDP+1kGP stores were written before the Far Pairs came back; deriving theirs
    # from each store's own index must write what a build writes, byte for byte, and nothing
    # else: not an array, and not the provenance.
    committed = files(TEST_STORE, "far")
    assert len(committed) == 15, "the Test Store's far/ must be compared whole"
    copy = without_far_pairs(tmp_path / "store")
    before = files(copy)
    assert load_script("derive_far_pairs").derive(copy) == 7, "the Test Store's seven"
    after = files(copy)
    assert {path: data for path, data in after.items() if not path.startswith("far/")} == before
    assert {path: data for path, data in after.items() if path.startswith("far/")} == committed
    assert not any(opengwasdb.ld.validate(copy).values())


def test_a_store_that_holds_far_pairs_is_not_derived_again(tmp_path: Path) -> None:
    copy = copy_store(tmp_path / "store")
    before = files(copy)
    with pytest.raises(SystemExit, match="far/ already"):
        load_script("derive_far_pairs").derive(copy)
    assert files(copy) == before


def test_a_lost_index_chunk_is_refused_never_read_as_no_far_pair(tmp_path: Path) -> None:
    # A lost chunk reads as the fill value, partner −1, which is no pair above its row: read as
    # data, it would silently drop the Far Pairs it held.
    copy = without_far_pairs(tmp_path / "store")
    shutil.rmtree(copy / "index" / "22" / "partner" / "c")
    with pytest.raises(ValueError, match=r"index/22/partner read .* fill value -1"):
        load_script("derive_far_pairs").derive(copy)


def test_a_derivation_that_fails_leaves_no_far_pairs_behind(tmp_path: Path) -> None:
    # The Far Pairs are written under another name and renamed only when every chromosome's are
    # written and counted, so a failure leaves the store as it was, and a run can begin again.
    copy = without_far_pairs(tmp_path / "store")
    meta = copy / "zarr.json"
    doc = json.loads(meta.read_text())
    doc["attributes"]["ld_store"]["counts"]["far_pairs"] = 8
    meta.write_text(json.dumps(doc))
    before = files(copy)
    script = load_script("derive_far_pairs")
    with pytest.raises(SystemExit, match="7 Far Pairs, but its provenance counts 8"):
        script.derive(copy)
    assert files(copy) == before, "nothing left behind: no far/, no partial far/"


def test_a_partial_derivation_left_by_a_killed_run_is_replaced(tmp_path: Path) -> None:
    copy = without_far_pairs(tmp_path / "store")
    script = load_script("derive_far_pairs")
    stale = copy / script.PARTIAL / "22" / "partner"
    stale.mkdir(parents=True)
    (stale / "zarr.json").write_text("{ not json")
    assert script.derive(copy) == 7
    assert not (copy / script.PARTIAL).exists()
    assert files(copy, "far") == files(TEST_STORE, "far")


def test_a_derivation_that_raises_part_way_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = without_far_pairs(tmp_path / "store")
    before = files(copy)
    script = load_script("derive_far_pairs")
    calls = []
    real = script.derive_far_pairs

    def second_fails(*args: object) -> int:
        calls.append(1)
        if len(calls) == 2:
            raise OSError("disk full")
        return int(real(*args))

    monkeypatch.setattr(script, "derive_far_pairs", second_fails)
    with pytest.raises(OSError, match="disk full"):
        script.derive(copy)
    assert len(calls) == 2, "chr21's Far Pairs were written before chr22's failed"
    assert files(copy) == before, "nothing left behind: no far/, no partial far/"
