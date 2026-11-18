"""ogld#4's contract: read, write and validate LD Stores; LD matrices with NaN and fill.

Against the Test Store, which holds the Test Panel's reference r at k = 10 (support.py).
Expected values are the reference r signed to A1, never what the reader computes. Every
test calls ogld first; the whole-store matrices are read once per worker and cached.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import support
import zarr
from support import alids, panel_data, passing_rows, reference, reference_matrix

import ogld
from ogld import cli
from ogld.format import VALIDATION_RULES, RefusalReason, REncoding, t_far
from ogld.store import StoreWriter


def two(store: ogld.LDStore, i: int, j: int) -> tuple[float, str]:
    """The store's r and answer kind for Test Panel rows i and j."""
    m = store.matrix(alids([i, j]))
    assert m.absent == ()
    return float(m.r[0, 1]), str(m.kinds()[0, 1])


@cache
def opened() -> ogld.LDStore:
    return ogld.open(support.TEST_STORE)


@cache
def whole_store(fill: float | None = None) -> ogld.LDMatrix:
    """Every Test Store variant's matrix, both chromosomes, in Test Panel order."""
    store = opened()
    return store.matrix(alids(passing_rows()), fill=fill)


def expected() -> np.ndarray:
    return reference_matrix(support.K, tuple(int(row) for row in passing_rows()))


def with_provenance(tmp_path: Path, edit: Callable[[dict[str, Any]], None]) -> Path:
    """A copy of the Test Store whose provenance `edit` has changed."""
    copy = tmp_path / "store"
    shutil.copytree(support.TEST_STORE, copy)
    meta = copy / "zarr.json"
    doc = json.loads(meta.read_text())
    edit(doc["attributes"]["ld_store"])
    meta.write_text(json.dumps(doc))
    return copy


# ------------------------------------------------------------------ opening


def test_the_test_store_records_its_panel_and_its_distances() -> None:
    p = opened().provenance
    assert p["format_version"] == "0.2.0"
    assert p["panel"]["n"] == support.N_SAMPLES
    assert p["adjustment"]["k"] == support.K
    assert p["adjustment"]["covariates"] == [f"PC{i}" for i in range(1, 11)]
    assert p["t_far"] == support.T_FAR
    assert p["far_code_floor"] == support.FLOOR
    assert p["t_partner"] == support.T_PARTNER
    assert p["partner_code_floor"] == support.PARTNER_FLOOR
    assert p["counts"]["partner_entries"] > 0
    assert p["encoding"] == {
        "r_dtype": "int16",
        "r_scale": support.R_SCALE,
        "r_reserved_code": -32768,
        "rounding": "half_to_even",
    }
    assert (p["near_bp"], p["far_bp"]) == (3_000_000, 10_000_000)
    assert p["source"]["kind"] == "genotypes"
    assert p["n_variants"] == len(passing_rows())


def test_a_reader_refuses_a_format_version_it_does_not_implement(tmp_path: Path) -> None:
    opened()

    def future(p: dict[str, Any]) -> None:
        p["format_version"] = "9.0.0"

    with pytest.raises(ValueError, match=r"9\.0\.0"):
        ogld.open(with_provenance(tmp_path, future))


def test_a_0_1_0_store_is_read_as_it_was_before_the_partner_rows(tmp_path: Path) -> None:
    # Spec §13: the bump lands with the writer, so a store published before the Partner Rows is
    # still read, from its Near Band, its Proxy Index and its Far Pairs.
    def earlier(p: dict[str, Any]) -> None:
        p["format_version"] = "0.1.0"

    location = with_provenance(tmp_path, earlier)
    shutil.rmtree(location / "partners")
    opened_early = ogld.open(location)
    assert opened_early.provenance["format_version"] == "0.1.0"
    rows = alids(passing_rows()[:20])
    assert opened_early.matrix(rows).r.shape == (20, 20)


def test_the_test_store_reads_the_same_over_http(served_store: str) -> None:
    # Spec §1: a local path or an HTTP(S) URL. A static Range-serving server, as #9 may use.
    remote = ogld.open(served_store)
    assert remote.provenance == opened().provenance
    some = alids(passing_rows()[:40])
    np.testing.assert_array_equal(remote.matrix(some).r, opened().matrix(some).r)


def test_every_array_decodes_as_zarr_over_http(served_store: str) -> None:
    from ogld.store.location import ChunkCache, open_group

    cache = ChunkCache()
    remote = open_group(served_store, cache=cache)
    local = zarr.open_group(support.TEST_STORE, mode="r")
    arrays = [(p, a) for p, a in local.members(max_depth=None) if isinstance(a, zarr.Array)]
    assert len(arrays) > 20
    for path, array in arrays:
        served = remote[path]
        assert isinstance(served, zarr.Array)
        np.testing.assert_array_equal(cache.take(served, np.arange(array.shape[0])), array[:])


# ------------------------------------------------------------------ pair answers (spec §8)


def test_a_pair_exactly_3_mb_apart_is_the_near_band_s_exact_r() -> None:
    store = opened()
    pair = reference().where(reference().distance == 3_000_000, "a pair exactly 3 Mb apart")
    r, kind = two(store, int(pair.row_i[0]), int(pair.row_j[0]))
    assert kind == "stored"
    assert abs(r - pair.r[0]) <= support.HALF_STEP + support.SLACK


def test_a_pair_beyond_3_mb_below_t_far_is_nan_and_the_bound() -> None:
    store = opened()
    ref = reference()
    far = (ref.distance > 3_000_000) & (ref.distance <= 10_000_000)
    pair = ref.where(far & (np.abs(ref.code) < support.FLOOR), "a 3–10 Mb pair below t_far")
    r, kind = two(store, int(pair.row_i[0]), int(pair.row_j[0]))
    assert math.isnan(r), "an unstored pair is NaN, never a number"
    assert kind == "bound"


def test_a_far_pair_is_stored_and_reads_back_its_r() -> None:
    store = opened()
    ref = reference()
    far = (ref.distance > 3_000_000) & (ref.distance <= 10_000_000)
    pair = ref.where(far & (np.abs(ref.code) >= support.FLOOR), "a real Far Pair at k = 10")
    r, kind = two(store, int(pair.row_i[0]), int(pair.row_j[0]))
    assert kind == "stored"
    assert abs(r - pair.r[0]) <= support.HALF_STEP + support.SLACK


def a_pair_apart(distance: int) -> tuple[int, int] | None:
    """Two passing Test Panel rows on one chromosome exactly `distance` bp apart, if any."""
    v = panel_data().variants()
    rows = passing_rows()
    pos, chrom = v.pos[rows].astype(np.int64), v.chrom[rows]
    gap = (pos[None, :] - pos[:, None] == distance) & (chrom[None, :] == chrom[:, None])
    found = np.argwhere(gap)
    return (int(rows[found[0][0]]), int(rows[found[0][1]])) if len(found) else None


def test_exactly_10_mb_is_measured_and_one_base_more_is_not() -> None:
    store = opened()
    at = reference().where(reference().distance == 10_000_000, "a pair exactly 10 Mb apart")
    assert two(store, int(at.row_i[0]), int(at.row_j[0]))[1] in {"stored", "bound"}
    beyond = a_pair_apart(10_000_001)
    assert beyond is not None, "the Test Panel must hold a pair 10,000,001 bp apart"
    r, kind = two(store, *beyond)
    assert math.isnan(r)
    assert kind == "not_measured"


def test_another_chromosome_is_not_measured() -> None:
    store = opened()
    v = panel_data().variants()
    rows = passing_rows()
    chr21 = rows[v.chrom[rows] == "21"]
    chr22 = rows[v.chrom[rows] == "22"]
    assert len(chr21), "the Test Panel must hold chr21"
    assert len(chr22), "the Test Panel must hold chr22"
    r, kind = two(store, int(chr21[0]), int(chr22[0]))
    assert math.isnan(r)
    assert kind == "not_measured"


def test_two_records_at_one_position_are_a_near_band_pair() -> None:
    store = opened()
    pair = reference().where(reference().distance == 0, "two records at one position")
    r, kind = two(store, int(pair.row_i[0]), int(pair.row_j[0]))
    assert kind == "stored"
    assert abs(r - pair.r[0]) <= support.HALF_STEP + support.SLACK


# ------------------------------------------------------------------ matrices, NaN and fill


def test_the_whole_store_reads_back_within_half_a_step() -> None:
    # The round trip: every pair of both chromosomes against the reference r, signed to A1.
    m = whole_store()
    ref = expected()
    stored = m.kinds() == "stored"
    assert stored.sum() > 1000, "the store must hold many pairs for this to mean anything"
    assert not np.isnan(ref[stored]).any(), "every stored pair is in the reference"
    assert np.all(np.abs(m.r[stored] - ref[stored]) <= support.HALF_STEP + support.SLACK)
    assert np.isnan(m.r[~stored]).all(), "every pair the store does not hold is NaN"


def test_every_stored_cell_is_exactly_what_the_codes_say() -> None:
    m = whole_store()
    ref = expected()
    stored = ~np.isnan(m.r)
    np.fill_diagonal(stored, False)
    assert stored.any()
    np.testing.assert_array_equal(m.r[stored], support.stored_at(ref[stored]))


def test_unstored_cells_are_nan_and_a_fill_replaces_exactly_them() -> None:
    plain = whole_store()
    kinds = plain.kinds()
    bound, nowhere = kinds == "bound", kinds == "not_measured"
    assert bound.any(), "the matrix must hold a Bound cell for this to mean anything"
    assert nowhere.any(), "the matrix must hold a Not Measured cell for this to mean anything"
    unstored = bound | nowhere
    assert np.isnan(plain.r[unstored]).all()
    assert not np.isnan(plain.r[~unstored]).any()
    # A non-zero fill, so a matrix that writes 0 for any fill cannot pass.
    filled = whole_store(fill=-9.0)
    assert filled.fill == -9.0
    assert (filled.r[unstored] == -9.0).all()
    np.testing.assert_array_equal(filled.r[~unstored], plain.r[~unstored])
    with pytest.raises(ValueError, match="fill"):
        filled.kinds()


def test_absent_alids_are_listed_and_are_not_rows() -> None:
    store = opened()
    v = panel_data().variants()
    failing = np.flatnonzero(~v.passes_filters)
    assert len(failing), "the Test Panel must hold variants the filters remove"
    present = alids(passing_rows()[:3])
    asked = [present[0], *alids(failing[:1]), "not-an-alid", *present[1:]]
    m = store.matrix(asked)
    assert m.absent == (asked[1], "not-an-alid")
    assert list(m.variants.alid) == present
    assert m.r.shape == (3, 3)


# ------------------------------------------------------------------ regions


def test_a_region_is_the_matrix_of_its_variants_in_genomic_order() -> None:
    store = opened()
    v = panel_data().variants()
    rows = passing_rows()
    middle = rows[len(rows) // 2]
    centre, chrom = int(v.pos[middle]), str(v.chrom[middle])
    region = store.region(chrom, centre, 2_000_000)
    offset = np.abs(v.pos[rows].astype(np.int64) - centre)
    inside = rows[(v.chrom[rows] == chrom) & (offset <= 2_000_000)]
    assert len(inside) > 1, "the region must hold variants"
    assert list(region.variants.alid) == alids(inside)
    np.testing.assert_array_equal(region.r, store.matrix(alids(inside)).r)


def test_a_region_is_at_most_10_mb_wide() -> None:
    store = opened()
    store.region("22", 30_000_000, 5_000_000)
    with pytest.raises(ogld.Refusal) as wide:
        store.region("22", 30_000_000, 5_000_001)
    assert wide.value.reason is RefusalReason.REGION_OVER_FAR
    with pytest.raises(ogld.Refusal) as negative:
        store.region("22", 30_000_000, -1)
    assert negative.value.reason is RefusalReason.INVALID


# ------------------------------------------------------------------ writing


def test_a_lost_chunk_is_refused_never_read_as_zero(tmp_path: Path) -> None:
    opened()
    copy = tmp_path / "store"
    shutil.copytree(support.TEST_STORE, copy)
    shards = sorted((copy / "near" / "22" / "r" / "c").iterdir())
    assert shards, "the Test Store's chr22 Near Band must have a shard to lose"
    shards[0].unlink()
    v = panel_data().variants()
    rows = passing_rows()[v.chrom[passing_rows()] == "22"]
    with pytest.raises(ValueError, match=r"-32768|reserved"):
        ogld.open(copy).matrix(alids(rows[:50]))


def test_the_writer_refuses_a_location_that_exists(tmp_path: Path) -> None:
    # Against a copy: a broken writer is never pointed at the committed Test Store.
    variants = opened().matrix(alids(passing_rows()[:2])).variants
    copy = tmp_path / "store"
    shutil.copytree(support.TEST_STORE, copy)
    before = sorted(p.relative_to(copy) for p in copy.rglob("*"))
    with pytest.raises(FileExistsError):
        StoreWriter.create(
            copy,
            variants=variants,
            near_bp=3_000_000,
            far_bp=10_000_000,
            encoding=REncoding(support.R_SCALE),
            n=771,
            k=10,
            t_partner=0.01,
        )
    assert sorted(p.relative_to(copy) for p in copy.rglob("*")) == before


# ------------------------------------------------------------------ validation (spec §12)


@cache
def committed_findings() -> dict[str, list[ogld.Finding]]:
    """The committed Test Store validated, once per worker: every test below starts from it."""
    return ogld.validate(support.TEST_STORE)


def test_the_test_store_passes_every_rule() -> None:
    findings = committed_findings()
    assert tuple(findings) == VALIDATION_RULES
    assert {rule: found for rule, found in findings.items() if found} == {}


# The breakers edit a copy's JSON documents or rewrite one of its arrays whole, which the
# Test Store's size allows. Each changes exactly what its description says.


def edit_json(store: Path, node: str, edit: Callable[[dict[str, Any]], None]) -> None:
    """Edit the metadata document of `node` ("" for the root) in place."""
    meta = store / node / "zarr.json"
    doc = json.loads(meta.read_text())
    edit(doc)
    meta.write_text(json.dumps(doc))


def edit_provenance(edit: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
    return lambda store: edit_json(store, "", lambda doc: edit(doc["attributes"]["ld_store"]))


def rewrite(store: Path, path: str, change: Callable[[np.ndarray], np.ndarray]) -> None:
    """Replace the array at `path` with `change` of its data, resizing it to fit."""
    array = zarr.open_array(store / path, mode="r+")
    data = change(np.asarray(array[:]).copy())
    array.resize(data.shape)
    array[:] = data


def chr22(store: Path) -> tuple[int, np.ndarray]:
    """chr22's first row and its positions in the copy."""
    group = zarr.open_group(store / "variants", mode="r")
    (start, stop), *_ = [(c["row_start"], c["row_stop"]) for c in group.attrs["chromosomes"]
                         if c["name"] == "22"]  # fmt: skip
    return start, np.asarray(group["position"][start:stop]).astype(np.int64)


def index22(store: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """chr22's Proxy Index as entries: owner, partner (local rows) and code."""
    group = zarr.open_group(store / "index" / "22", mode="r")
    indptr = np.asarray(group["indptr"][:])
    owner = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    return owner, np.asarray(group["partner"][:]).astype(np.int64), np.asarray(group["r"][:])


def write_index22(store: Path, owner: np.ndarray, partner: np.ndarray, code: np.ndarray) -> None:
    """Write chr22's Proxy Index from entries, in row order and ascending partner."""
    order = np.lexsort((partner, owner))
    size = len(chr22(store)[1])
    counts = np.bincount(owner, minlength=size)
    rewrite(store, "index/22/indptr", lambda _: np.concatenate([[0], np.cumsum(counts)]))
    rewrite(store, "index/22/partner", lambda _: partner[order].astype(np.int32))
    rewrite(store, "index/22/r", lambda _: code[order].astype(np.int16))


def far22(store: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """chr22's Far Pairs as entries: row, partner (local rows) and code."""
    group = zarr.open_group(store / "far" / "22", mode="r")
    indptr = np.asarray(group["indptr"][:])
    owner = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    return owner, np.asarray(group["partner"][:]).astype(np.int64), np.asarray(group["r"][:])


def write_far22(store: Path, owner: np.ndarray, partner: np.ndarray, code: np.ndarray) -> None:
    """Write chr22's Far Pairs from entries, in row order and ascending partner."""
    order = np.lexsort((partner, owner))
    counts = np.bincount(owner, minlength=len(chr22(store)[1]))
    rewrite(store, "far/22/indptr", lambda _: np.concatenate([[0], np.cumsum(counts)]))
    rewrite(store, "far/22/partner", lambda _: partner[order].astype(np.int32))
    rewrite(store, "far/22/r", lambda _: code[order].astype(np.int16))


def floor_of(store: Path) -> int:
    """The far code floor the copy records: a breaker sets a code by its own store's floor."""
    floor: int = json.loads((store / "zarr.json").read_text())["attributes"]["ld_store"][
        "far_code_floor"
    ]
    return floor


def add_pair(
    gap: Callable[[np.ndarray], np.ndarray], coded: Callable[[int], int]
) -> Callable[[Path], None]:
    """A breaker adding, both ways, the first chr22 pair not in the index whose distance
    `gap` accepts, with the code `coded` gives for the store's far code floor."""

    def breaker(store: Path) -> None:
        code = coded(floor_of(store))
        _, pos = chr22(store)
        owner, partner, codes = index22(store)
        held = set(zip(owner.tolist(), partner.tolist(), strict=True))
        apart = np.argwhere(np.triu(gap(pos[None, :] - pos[:, None]), k=1))
        lo, hi = next((int(a), int(b)) for a, b in apart if (int(a), int(b)) not in held)
        both = (np.append(owner, [lo, hi]), np.append(partner, [hi, lo]))
        write_index22(store, *both, np.append(codes, [code, code]).astype(np.int16))

    return breaker


def add_far_pair(
    gap: Callable[[np.ndarray], np.ndarray], coded: Callable[[int], int]
) -> Callable[[Path], None]:
    """A breaker adding to chr22's Far Pairs, in its earlier row, the first pair not among them
    whose distance `gap` accepts, with the code `coded` gives for the store's far code floor."""

    def breaker(store: Path) -> None:
        _, pos = chr22(store)
        owner, partner, codes = far22(store)
        held = set(zip(owner.tolist(), partner.tolist(), strict=True))
        apart = np.argwhere(np.triu(gap(pos[None, :] - pos[:, None]), k=1))
        lo, hi = next((int(a), int(b)) for a, b in apart if (int(a), int(b)) not in held)
        code = np.append(codes, coded(floor_of(store))).astype(np.int16)
        write_far22(store, np.append(owner, lo), np.append(partner, hi), code)

    return breaker


def move_a_far_pair_to_its_later_row(store: Path) -> None:
    owner, partner, codes = far22(store)
    owner[0], partner[0] = partner[0], owner[0]
    write_far22(store, owner, partner, codes)


def drop_a_far_pair(store: Path) -> None:
    owner, partner, codes = far22(store)
    write_far22(store, owner[1:], partner[1:], codes[1:])


def drop_entry(near: bool) -> Callable[[Path], None]:
    """A breaker dropping the first chr22 entry, earlier row first, within 3 Mb or beyond."""

    def breaker(store: Path) -> None:
        _, pos = chr22(store)
        owner, partner, codes = index22(store)
        within = np.abs(pos[partner] - pos[owner]) <= 3_000_000
        first = np.flatnonzero((within == near) & (partner > owner))[0]
        keep = np.arange(len(owner)) != first
        write_index22(store, owner[keep], partner[keep], codes[keep])

    return breaker


def shorten_a_near_row(store: Path) -> None:
    indptr = np.asarray(zarr.open_array(store / "near/22/indptr", mode="r")[:])
    row = int(np.flatnonzero(np.diff(indptr) > 0)[0])
    rewrite(store, "near/22/r", lambda r: np.delete(r, indptr[row + 1] - 1))
    rewrite(store, "near/22/indptr", lambda p: p - (np.arange(len(p)) > row))


def swap_two_positions(store: Path) -> None:
    start, pos = chr22(store)
    row = start + int(np.flatnonzero(np.diff(pos) > 0)[0])
    rewrite(store, "variants/position", lambda p: p[np.r_[:row, row + 1, row, row + 2 : len(p)]])


def reverse_an_alid(store: Path) -> None:
    start, _ = chr22(store)

    def reverse(alid: np.ndarray) -> np.ndarray:
        chrom, pos, a1, a2 = str(alid[start]).split(":")
        alid[start] = f"{chrom}:{pos}:{a2}:{a1}"
        return alid

    rewrite(store, "variants/alid", reverse)


def repeat_a_partner(store: Path) -> None:
    owner, partner, codes = index22(store)
    k = int(np.flatnonzero(owner[1:] == owner[:-1])[0])
    partner[k + 1] = partner[k]
    write_index22(store, owner, partner, codes)


def alid_index_entries(store: Path) -> tuple[np.ndarray, np.ndarray]:
    """The ALID search index as entries: its keys, a row of bytes each, and each entry's row."""
    group = zarr.open_group(store / "variants/alid_index", mode="r")
    keys = np.asarray(group["alid"][:]).reshape(-1, 32)
    return keys, np.asarray(group["row"][:]).astype(np.int64)


def write_alid_index(store: Path, keys: np.ndarray, rows: np.ndarray) -> None:
    rewrite(store, "variants/alid_index/alid", lambda _: keys.ravel())
    rewrite(store, "variants/alid_index/row", lambda _: rows.astype(np.int32))


def shorten_the_key_array(store: Path) -> None:
    """`variants/alid_index/alid` one key short of `row`'s entries, which still name every row: the
    two arrays do not agree, so `alid_index.count` names it and a rule that reads the keys says it
    cannot be checked rather than failing on the shorter array (ogld#42's fix round 3)."""
    rewrite(store, "variants/alid_index/alid", lambda data: data[:-32])


def drop_an_index_entry(store: Path) -> None:
    """The last entry removed from both arrays: the row it named has no entry, and every entry
    left is its row's own ALID."""
    keys, rows = alid_index_entries(store)
    write_alid_index(store, keys[:-1], rows[:-1])


def swap_two_index_entries(store: Path) -> None:
    """The first two entries in each other's place, key and row together: every entry is still its
    row's own ALID, and the keys no longer ascend."""
    keys, rows = alid_index_entries(store)
    order = np.arange(len(rows))
    order[[0, 1]] = [1, 0]
    write_alid_index(store, keys[order], rows[order])


def invent_a_key(store: Path) -> None:
    """One entry's key replaced by an ALID the store does not hold, naming the row it already
    named: the row's own ALID is no longer in the index, and a lookup by the invented one finds
    it (ogld#42's spec review, reproduction 2)."""
    keys, rows = alid_index_entries(store)
    keys[474] = np.frombuffer(b"22:1:A:C".ljust(32, b"\0"), dtype=np.uint8)
    order = np.argsort(keys.view("S32").ravel(), kind="stable")
    write_alid_index(store, keys[order], rows[order])


def append_a_truncated_key(*, adjust_count: bool) -> Callable[[Path], None]:
    """A wide row's key cut into the slot, appended as an entry of its own naming that row — the
    defect the width exists to prevent, and ogld#42's spec review reproduction 1. It passes all
    28 rules before fix round 1. `adjust_count` also records one fewer ALID left out, which is
    how the review made the older count rule pass; the count is the table's now, so that is a
    second finding."""

    def breaker(store: Path) -> None:
        keys, rows = alid_index_entries(store)
        wide = np.flatnonzero(
            np.array([len(str(alid).encode()) > 32 for alid in alids_of(store)], dtype=bool)
        )
        row = int(wide[1])
        key = np.frombuffer(alids_of(store)[row].encode()[:32], dtype=np.uint8)
        at = int(np.searchsorted(keys.view("S32").ravel(), key.view("S32")[0]))
        write_alid_index(
            store,
            np.insert(keys, at, key, axis=0),
            np.insert(rows, at, row),
        )
        if adjust_count:
            edit_provenance(
                lambda p: p["counts"].update(alids_left_out=p["counts"]["alids_left_out"] - 1)
            )(store)

    return breaker


def alids_of(store: Path) -> list[str]:
    """The Test Store's ALIDs, in row order, from its own table."""
    values = np.asarray(zarr.open_array(store / "variants/alid", mode="r")[:]).tolist()
    return [str(alid) for alid in values]


def set_element(path: str, at: int, value: float) -> Callable[[Path], None]:
    def breaker(store: Path) -> None:
        def change(data: np.ndarray) -> np.ndarray:
            data[at] = value
            return data

        rewrite(store, path, change)

    return breaker


#: One way to break each rule, applied to a copy of the Test Store. ogld#4 writes each.
BREAKERS: dict[str, Callable[[Path], None]] = {
    # lacks index/22/partner
    "envelope.arrays": lambda store: shutil.rmtree(store / "index" / "22" / "partner"),
    # sets near/22/r's fill value to 0
    "envelope.fill": lambda store: edit_json(
        store, "near/22/r", lambda doc: doc.update(fill_value=0)
    ),
    # drops the key adjustment.covariates
    "provenance.complete": edit_provenance(lambda p: p["adjustment"].pop("covariates")),
    # records r_scale 32768
    "provenance.encoding": edit_provenance(lambda p: p["encoding"].update(r_scale=32768)),
    # records far_bp 12,000,000
    "provenance.distances": edit_provenance(lambda p: p.update(far_bp=12_000_000)),
    # records t_far 0.02 for n = 771, k = 10
    "provenance.threshold": edit_provenance(lambda p: p.update(t_far=0.02)),
    # gives chr22's range a gap
    "variants.chromosomes": lambda store: edit_json(
        store,
        "variants",
        lambda doc: doc["attributes"]["chromosomes"][1].update(row_start=469),
    ),
    # swaps two positions on chr22
    "variants.order": swap_two_positions,
    # writes an ALID with its alleles reversed
    "variants.identity": reverse_an_alid,
    # sets one a1_frequency to 0
    "variants.frequency": set_element("variants/a1_frequency", 7, 0.0),
    # drops an entry, so the row it named has none
    "alid_index.rows": drop_an_index_entry,
    # puts two entries in each other's place, so the keys do not ascend
    "alid_index.order": swap_two_index_entries,
    # gives one entry an ALID the store does not hold, naming a row it has
    "alid_index.keys": invent_a_key,
    # leaves the key array one key short of the entries it must hold
    "alid_index.count": shorten_the_key_array,
    # shortens one Near Band row by one code
    "near.rows": shorten_a_near_row,
    # writes −32768 into near/22/r
    "near.codes": set_element("near/22/r", 5, -32768),
    # repeats a partner within one index row
    "index.rows": repeat_a_partner,
    # writes −32768 into index/22/r
    "index.codes": set_element("index/22/r", 3, -32768),
    # adds an index entry 10,000,001 bp apart, with a code above the floor
    "index.distance": add_pair(lambda gap: gap == 10_000_001, lambda floor: floor + 100),
    # adds an index entry with |code| = far_code_floor − 1
    "index.threshold": add_pair(
        lambda gap: (gap > 3_000_000) & (gap <= 10_000_000), lambda floor: -(floor - 1)
    ),
    # drops one direction of one index pair
    "index.symmetric": drop_entry(near=False),
    # drops an index entry within 3 Mb whose Near Band code reaches the floor
    "index.near": drop_entry(near=True),
    # holds one chr22 Far Pair in its later row
    "far.rows": move_a_far_pair_to_its_later_row,
    # writes −32768 into far/22/r
    "far.codes": set_element("far/22/r", 0, -32768),
    # adds a Far Pair 10,000,001 bp apart, with a code above the floor
    "far.distance": add_far_pair(lambda gap: gap == 10_000_001, lambda floor: floor + 100),
    # adds a Far Pair 3–10 Mb apart with |code| = far_code_floor − 1
    "far.threshold": add_far_pair(
        lambda gap: (gap > 3_000_000) & (gap <= 10_000_000), lambda floor: -(floor - 1)
    ),
    # drops a Far Pair the index holds
    "far.index": drop_a_far_pair,
}


def test_the_provenance_count_is_a_breaker_of_its_own_rule(tmp_path: Path) -> None:
    # A recorded count the table does not give fails `alid_index.count` alone: the keys and rows
    # are untouched (ogld#42's fix round 1).
    copy = tmp_path / "store"
    shutil.copytree(support.TEST_STORE, copy)
    edit_provenance(lambda p: p["counts"].update(alids_left_out=0))(copy)
    failing = {name for name, found in ogld.validate(copy).items() if found}
    assert failing == {"alid_index.count"}, failing


def test_every_rule_has_a_way_to_break_it() -> None:
    # Not a contract test: it holds the table above to the spec's rule list now.
    assert tuple(BREAKERS) == VALIDATION_RULES


@pytest.mark.parametrize("rule", VALIDATION_RULES)
def test_a_store_that_breaks_a_rule_fails_that_rule(rule: str, tmp_path: Path) -> None:
    assert not any(committed_findings().values())
    copy = tmp_path / "store"
    shutil.copytree(support.TEST_STORE, copy)
    BREAKERS[rule](copy)
    findings = ogld.validate(copy)
    assert findings[rule], f"{rule} passed a store built to break it"


@pytest.mark.parametrize("rule", [r for r in VALIDATION_RULES if r.startswith("alid_index.")])
def test_an_index_rule_s_breaker_fails_only_that_rule(rule: str, tmp_path: Path) -> None:
    # The four index rules judge one array each, so a store built to break one must not be
    # refused for another: a reader trusts what they agree on (ogld#42's fix round 1). A rule whose
    # array the breaker shortened says it cannot be checked, which spec §12 requires of it, and
    # that is not a judgement against the store.
    copy = tmp_path / "store"
    shutil.copytree(support.TEST_STORE, copy)
    BREAKERS[rule](copy)
    found = ogld.validate(copy)
    assert found[rule], f"{rule} passed a store built to break it"
    for name, findings in found.items():
        if name == rule or not name.startswith("alid_index."):
            continue
        assert all(f.message.startswith("cannot be checked") for f in findings), (name, findings)


#: The two stores ogld#42's spec review built, which passed all 28 rules before fix round 1, with
#: the rules each fails now. Both are a lookup that answers falsely: an ALID the store does not
#: hold, found at a row it does.
REPRODUCTIONS: dict[str, tuple[Callable[[Path], None], set[str]]] = {
    # a wide ALID's key cut into the slot, appended as an entry of its own: the row's ALID is
    # wider than the slot, so no entry may name it
    "a truncated key appended": (append_a_truncated_key(adjust_count=False), {"alid_index.keys"}),
    # and with the count adjusted, as the review had to make it for the older count rule: the
    # record is no longer the table's count either
    "a truncated key appended, count adjusted": (
        append_a_truncated_key(adjust_count=True),
        {"alid_index.keys", "alid_index.count"},
    ),
}


@pytest.mark.parametrize("case", list(REPRODUCTIONS))
def test_the_review_s_reproductions_are_caught(case: str, tmp_path: Path) -> None:
    assert not any(committed_findings().values())
    breaker, expected = REPRODUCTIONS[case]
    copy = tmp_path / "store"
    shutil.copytree(support.TEST_STORE, copy)
    breaker(copy)
    failing = {name for name, found in ogld.validate(copy).items() if found}
    assert failing == expected, failing


def as_unadjusted(k_min: int | None) -> Callable[[dict[str, Any]], None]:
    """The Test Store's provenance recorded as a k = 0 build's, t_far and floor included."""

    def edit(p: dict[str, Any]) -> None:
        p["adjustment"].update(
            covariates=[], k=0, k_target=0, k_min=k_min, pc_file=None, pc_check=None
        )
        p["t_far"] = t_far(p["panel"]["n"], 0)
        p["far_code_floor"] = support.FLOOR_K0

    return edit


def as_import(k_target: int | None) -> Callable[[dict[str, Any]], None]:
    """The Test Store's provenance recorded as an imported matrix's: its 10 covariates, k = 10."""

    def edit(p: dict[str, Any]) -> None:
        p["source"]["kind"] = "ld_matrix"
        p["adjustment"].update(k_target=k_target, k_min=None, pc_file=None, pc_check=None)
        p["pc_diagnostic"] = None

    return edit


#: Spec §7 and §12: `k_min` applies only when a PC File was read, and an import has no
#: `k_target`. Each pair differs in that one field, so the passing one is the control.
K_RECORDS: dict[str, tuple[Callable[[dict[str, Any]], None], bool]] = {
    "unadjusted, k_min null": (as_unadjusted(None), True),
    "unadjusted, k_min 5": (as_unadjusted(5), False),
    "import, k_target null": (as_import(None), True),
    "import, k_target 10": (as_import(10), False),
}


@pytest.mark.parametrize("case", list(K_RECORDS))
def test_provenance_threshold_holds_k_min_and_k_target_to_the_source(
    case: str, tmp_path: Path
) -> None:
    assert not committed_findings()["provenance.threshold"]
    edit, passes = K_RECORDS[case]
    findings = ogld.validate(with_provenance(tmp_path, edit))["provenance.threshold"]
    assert (not findings) is passes, f"{case}: {findings}"


# ------------------------------------------------------------------ the CLI


def test_ogld_validate_and_info_on_the_test_store(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["validate", str(support.TEST_STORE)]) == 0
    assert cli.main(["info", str(support.TEST_STORE)]) == 0
    out = capsys.readouterr().out
    assert "ok   index.symmetric" in out
    assert '"t_far"' in out
