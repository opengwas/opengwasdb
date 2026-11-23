"""ogld#4's contract: read, write and validate LD Stores; LD matrices with NaN and fill.

Against the Test Store, which holds the Test Panel's reference r at k = 10 (support.py).
Expected values are the reference r signed to A1, never what the reader computes. Every
test calls ogld first; the whole-store matrices are read once per worker and cached.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import numcodecs
import numpy as np
import pytest
import support
import zarr
from support import alids, panel_data, passing_rows, reference, reference_matrix

import opengwasdb.ld
from opengwasdb.ld import cli
from opengwasdb.ld.format import VALIDATION_RULES, RefusalReason, REncoding, t_far
from opengwasdb.ld.store import StoreWriter, layout
from opengwasdb.ld.store.frames import decode_frame, encode_frame


def two(store: opengwasdb.ld.LDStore, i: int, j: int) -> tuple[float, str]:
    """The store's r and answer kind for Test Panel rows i and j."""
    m = store.matrix(alids([i, j]))
    assert m.absent == ()
    return float(m.r[0, 1]), str(m.kinds()[0, 1])


@cache
def opened() -> opengwasdb.ld.LDStore:
    return opengwasdb.ld.open(support.TEST_STORE)


@cache
def whole_store(fill: float | None = None) -> opengwasdb.ld.LDMatrix:
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
        opengwasdb.ld.open(with_provenance(tmp_path, future))


def test_a_0_1_0_store_is_read_as_it_was_before_the_partner_rows(tmp_path: Path) -> None:
    # ld-store §13: the bump lands with the writer, so a store published before the Partner Rows is
    # still read, from its Near Band, its Proxy Index and its Far Pairs.
    def earlier(p: dict[str, Any]) -> None:
        p["format_version"] = "0.1.0"

    location = with_provenance(tmp_path, earlier)
    shutil.rmtree(location / "partners")
    opened_early = opengwasdb.ld.open(location)
    assert opened_early.provenance["format_version"] == "0.1.0"
    rows = alids(passing_rows()[:20])
    assert opened_early.matrix(rows).r.shape == (20, 20)


def test_the_test_store_reads_the_same_over_http(served_store: str) -> None:
    # ld-store §1: a local path or an HTTP(S) URL. A static Range-serving server, as opengwas/ogld#9
    # may use.
    remote = opengwasdb.ld.open(served_store)
    assert remote.provenance == opened().provenance
    some = alids(passing_rows()[:40])
    np.testing.assert_array_equal(remote.matrix(some).r, opened().matrix(some).r)


def test_every_array_decodes_as_zarr_over_http(served_store: str) -> None:
    from opengwasdb.ld.store.location import ChunkCache, open_group

    cache = ChunkCache()
    remote = open_group(served_store, cache=cache)
    local = zarr.open_group(support.TEST_STORE, mode="r")
    arrays = [(p, a) for p, a in local.members(max_depth=None) if isinstance(a, zarr.Array)]
    assert len(arrays) > 20
    for path, array in arrays:
        served = remote[path]
        assert isinstance(served, zarr.Array)
        np.testing.assert_array_equal(cache.take(served, np.arange(array.shape[0])), array[:])


# ------------------------------------------------------------------ pair answers (ld-store §8)


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
    with pytest.raises(opengwasdb.ld.Refusal) as wide:
        store.region("22", 30_000_000, 5_000_001)
    assert wide.value.reason is RefusalReason.REGION_OVER_FAR
    with pytest.raises(opengwasdb.ld.Refusal) as negative:
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
        opengwasdb.ld.open(copy).matrix(alids(rows[:50]))


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


# ------------------------------------------------------------------ validation (ld-store §12)


@cache
def committed_findings() -> dict[str, list[opengwasdb.ld.Finding]]:
    """The committed Test Store validated, once per worker: every test below starts from it."""
    return opengwasdb.ld.validate(support.TEST_STORE)


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


def partner_rows(store: Path, chrom: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """One chromosome's Partner Rows as rows: each row's partners and codes, and the group's G.

    Read through the store's own frame decoder, never by parsing a payload here.
    """
    group = zarr.open_group(store / "partners" / chrom, mode="r")
    attrs = dict(group.attrs)
    per_frame = int(attrs["rows_per_frame"])
    offsets = np.asarray(group["frames_idx"][:]).astype(np.int64)
    raw = np.asarray(group["frames"][:]).tobytes()
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    for number in range(int(attrs["frames"])):
        first = number * per_frame
        due = min(per_frame, int(attrs["rows"]) - first)
        frame = decode_frame(
            raw[offsets[number] : offsets[number + 1]],
            chrom=chrom,
            number=number,
            first=first,
            rows=due,
        )
        rows.extend(frame.row(first + at) for at in range(due))
    return rows, per_frame


def frame_bytes(store: Path, chrom: str) -> tuple[list[bytes], dict[str, Any]]:
    """One chromosome's Partner Rows as the frames the group holds, and its attributes."""
    group = zarr.open_group(store / "partners" / chrom, mode="r")
    attrs = dict(group.attrs)
    offsets = np.asarray(group["frames_idx"][:]).astype(np.int64)
    raw = np.asarray(group["frames"][:]).tobytes()
    held = [raw[offsets[n] : offsets[n + 1]] for n in range(len(offsets) - 1)]
    return held, attrs


def write_frames(store: Path, chrom: str, frames: list[bytes], attrs: dict[str, Any]) -> None:
    """Write `chrom`'s Partner Rows from already-encoded `frames`, recording `attrs`' rows, rows a
    frame and entries, and restamping the offsets, the length and both hashes: a store only a test
    makes, which the rules beside the one it breaks still pass."""
    path = store / "partners" / chrom
    shutil.rmtree(path)
    group = zarr.open_group(path, mode="w", zarr_format=3)
    data = b"".join(frames)
    array = layout.new_frames_array(group, len(data))
    target = layout.chunk_file(array)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    offsets = np.concatenate([[0], np.cumsum([len(frame) for frame in frames])])
    index = layout.new_frames_index(group, len(offsets))
    index[:] = offsets.astype("<u8")
    group.attrs.update(
        {
            "rows": attrs["rows"],
            "rows_per_frame": attrs["rows_per_frame"],
            "frames": len(frames),
            "entries": attrs["entries"],
            "frames_bytes": len(data),
            "frames_sha256": hashlib.sha256(data).hexdigest(),
            "frames_idx_sha256": hashlib.sha256(offsets.astype("<u8").tobytes()).hexdigest(),
            "zstd_level": 3,
        }
    )


def write_partner_rows(
    store: Path, chrom: str, rows: list[tuple[np.ndarray, np.ndarray]], per_frame: int
) -> None:
    """Write `chrom`'s Partner Rows from `rows`, re-encoding every frame and recording the group's
    attributes and both hashes: a store only a test makes, which the other rules still pass."""
    frames = [
        encode_frame(at, rows[at : at + per_frame], chrom=chrom, number=at // per_frame)
        for at in range(0, len(rows), per_frame)
    ]
    write_frames(
        store,
        chrom,
        frames,
        {
            "rows": len(rows),
            "rows_per_frame": per_frame,
            "entries": sum(len(partners) for partners, _ in rows),
        },
    )


def window_descriptor(size: int) -> int:
    """A Window_Descriptor byte whose window holds at least `size` bytes (RFC 8878 §3.1.1.1.2)."""
    for descriptor in range(256):
        log = 10 + (descriptor >> 3)
        if (1 << log) + ((1 << log) >> 3) * (descriptor & 0b111) >= size:
            return descriptor
    raise ValueError(f"no zstd window holds {size} bytes")


def frame_without_content_size(frame: bytes) -> bytes:
    """`frame` with its header's content-size field removed and its flag cleared, the blocks and
    the content checksum untouched: the checksum still verifies, so only a rule that requires the
    size in the header can refuse it (RFC 8878 §3.1.1.1.1)."""
    descriptor = frame[4]
    flag, single = descriptor >> 6, bool(descriptor & 0b0010_0000)
    sizes = ({0: 1, 1: 2, 2: 4, 3: 8} if single else {0: 0, 1: 2, 2: 4, 3: 8})[flag]
    field = 5 if single else 6  # a frame that is not single-segment carries a window descriptor
    if single:
        # A single-segment frame's size field is its window, so one cleared of the size needs a
        # window of its own: the smallest that holds the payload the field named.
        size = int.from_bytes(frame[field : field + sizes], "little")
        head = frame[:4] + bytes([descriptor & 0b0001_1111, window_descriptor(size)])
    else:
        head = frame[:4] + bytes([descriptor & 0b0011_1111]) + frame[5:field]
    return head + frame[field + sizes :]


def a_frame_without_a_checksum(store: Path) -> None:
    """Recompresses chr22's third frame with no content checksum, restamping the offsets, the
    length and both hashes: a frame a decoder accepts and never verifies (partner-rows §5.2)."""
    frames, attrs = frame_bytes(store, "22")
    payload = bytes(numcodecs.Zstd(level=3, checksum=True).decode(frames[2]))
    frames[2] = bytes(numcodecs.Zstd(level=3, checksum=False).encode(payload))
    write_frames(store, "22", frames, attrs)


def a_frame_without_a_content_size(store: Path) -> None:
    """Drops chr22's third frame's content-size field, keeping its content checksum: a frame whose
    checksum still verifies, which only a header that requires the size refuses (§5.2)."""
    frames, attrs = frame_bytes(store, "22")
    frames[2] = frame_without_content_size(frames[2])
    write_frames(store, "22", frames, attrs)


def a_frame_with_a_delta_below_row_0(store: Path) -> None:
    """Recompresses chr22's third frame with its first delta one row short, so the frame's first
    partner decodes to −1: a checksummed frame only decoding can refuse (§5.2)."""
    frames, attrs = frame_bytes(store, "22")
    payload = bytearray(numcodecs.Zstd(level=3, checksum=True).decode(frames[2]))
    rows = int(np.frombuffer(bytes(payload[:4]), "<u4")[0])
    lengths = np.frombuffer(bytes(payload[4 : 4 + 4 * rows]), "<u4")
    entries = int(lengths.sum())
    for plane, byte in enumerate((-33).to_bytes(4, "little", signed=True)):
        payload[4 + 4 * rows + plane * entries] = byte
    frames[2] = bytes(numcodecs.Zstd(level=3, checksum=True).encode(bytes(payload)))
    write_frames(store, "22", frames, attrs)


def edit_partner_row(
    store: Path, change: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]]
) -> None:
    """`change` applied to the first chr22 row that has a partner, written back as frames."""
    rows, per_frame = partner_rows(store, "22")
    at = next(at for at, (partners, _) in enumerate(rows) if len(partners))
    rows[at] = change(*rows[at], at)
    write_partner_rows(store, "22", rows, per_frame)


def a_partner_past_the_chromosome(store: Path) -> None:
    """Names a partner past the chromosome's last row: a partner §5.2 never holds."""

    def change(partners: np.ndarray, codes: np.ndarray, at: int) -> tuple[np.ndarray, np.ndarray]:
        moved = partners.astype(np.int64).copy()
        moved[-1] = len(partner_rows(store, "22")[0]) + 5
        return moved, codes

    edit_partner_row(store, change)


def a_reserved_partner_code(store: Path) -> None:
    """Writes the reserved code into a Partner Rows row."""

    def change(partners: np.ndarray, codes: np.ndarray, at: int) -> tuple[np.ndarray, np.ndarray]:
        changed = codes.copy()
        changed[0] = -32768
        return partners, changed

    edit_partner_row(store, change)


def a_partner_beyond_far_bp(store: Path) -> None:
    """Moves one entry's partner to a row more than far_bp away, keeping the row ascending."""
    rows, per_frame = partner_rows(store, "22")
    _, pos = chr22(store)
    far_bp = 10_000_000
    for at, (partners, codes) in enumerate(rows):
        moved = partners.astype(np.int64).copy()
        far = np.flatnonzero(np.abs(pos - pos[at]) > far_bp)[::-1]
        chosen = next((int(j) for j in far if len(moved) == 1 or int(j) > moved[-2]), None)
        if chosen is None:
            continue
        moved[-1] = chosen
        rows[at] = (moved, codes)
        break
    else:
        raise AssertionError("no chr22 row holds a partner more than far_bp away")
    write_partner_rows(store, "22", rows, per_frame)


def a_partner_below_its_floor(store: Path) -> None:
    """Writes one within-near_bp pair's codes below the Partner Floor, both ways."""
    rows, per_frame = partner_rows(store, "22")
    _, pos = chr22(store)
    floor = json.loads((store / "zarr.json").read_text())["attributes"]["ld_store"][
        "partner_code_floor"
    ]
    for at, (partners, codes) in enumerate(rows):
        close = np.abs(pos[partners] - pos[at]) <= 3_000_000
        if not close.any():
            continue
        into = int(np.flatnonzero(close)[0])
        partner = int(partners[into])
        below = (floor - 1) if codes[into] >= 0 else -(floor - 1)
        codes = codes.copy()
        codes[into] = below
        rows[at] = (partners, codes)
        back = np.flatnonzero(rows[partner][0] == at)
        assert len(back) == 1, "a pair is in both its rows"
        other = rows[partner][1].copy()
        other[int(back[0])] = below
        rows[partner] = (rows[partner][0], other)
        break
    write_partner_rows(store, "22", rows, per_frame)


def a_pair_held_with_two_codes(store: Path) -> None:
    """Gives one pair's two entries different codes, both above the floor."""

    def change(partners: np.ndarray, codes: np.ndarray, at: int) -> tuple[np.ndarray, np.ndarray]:
        changed = codes.copy()
        changed[0] = -changed[0] if abs(int(changed[0])) < 32767 else changed[0] - 1
        return partners, changed

    edit_partner_row(store, change)


def a_missing_within_near_bp_entry(store: Path) -> None:
    """Drops one direction of a pair within near_bp whose band code reaches the Partner Floor,
    so the near half is no longer the band's."""
    rows, per_frame = partner_rows(store, "22")
    _, pos = chr22(store)
    for at, (partners, codes) in enumerate(rows):
        close = np.abs(pos[partners] - pos[at]) <= 3_000_000
        if not close.any():
            continue
        into = int(np.flatnonzero(close)[0])
        rows[at] = (np.delete(partners, into), np.delete(codes, into))
        break
    write_partner_rows(store, "22", rows, per_frame)


def a_later_row_code_unlike_the_band(store: Path) -> None:
    """Gives the entry in a pair's later row a code the Near Band does not hold there, the
    earlier row's entry staying as the band holds it."""
    rows, per_frame = partner_rows(store, "22")
    _, pos = chr22(store)
    for at, (partners, codes) in enumerate(rows):
        close = (partners < at) & (np.abs(pos[partners] - pos[at]) <= 3_000_000)
        if not close.any():
            continue
        into = int(np.flatnonzero(close)[0])
        changed = codes.copy()
        changed[into] = -changed[into]
        rows[at] = (partners, changed)
        write_partner_rows(store, "22", rows, per_frame)
        return
    raise AssertionError("no chr22 entry is in a pair's later row within near_bp")


def a_missing_far_entry(store: Path) -> None:
    """Drops one direction of a pair beyond near_bp, so the far half is not the Far Pairs'."""
    rows, per_frame = partner_rows(store, "22")
    _, pos = chr22(store)
    for at, (partners, codes) in enumerate(rows):
        beyond = np.abs(pos[partners] - pos[at]) > 3_000_000
        if not beyond.any():
            continue
        into = int(np.flatnonzero(beyond)[0])
        rows[at] = (np.delete(partners, into), np.delete(codes, into))
        break
    write_partner_rows(store, "22", rows, per_frame)


def flip_a_byte_in_a_frame(store: Path) -> None:
    """Flips a byte inside chr22's third frame: its content checksum then fails."""
    group = zarr.open_group(store / "partners" / "22", mode="r")
    offsets = np.asarray(group["frames_idx"][:]).astype(np.int64)
    chunk = layout.chunk_file(group["frames"])
    data = bytearray(chunk.read_bytes())
    at = int(offsets[2]) + 12
    data[at] ^= 0xFF
    chunk.write_bytes(bytes(data))


def a_group_recording_another_row_count(store: Path) -> None:
    """Records 587 rows for chr22, which has 588."""
    group = zarr.open_group(store / "partners" / "22", mode="r+")
    group.attrs.update(rows=587)


def a_later_row_missing_its_entry(store: Path) -> None:
    """Drops the entry a pair's later row holds for its earlier row, keeping the earlier row's
    own: a pair within near_bp whose band code is in one row only (partner-rows §3)."""
    rows, per_frame = partner_rows(store, "22")
    _, pos = chr22(store)
    for at, (partners, codes) in enumerate(rows):
        close = (partners < at) & (np.abs(pos[partners] - pos[at]) <= 3_000_000)
        if not close.any():
            continue
        into = int(np.flatnonzero(close)[0])
        rows[at] = (np.delete(partners, into), np.delete(codes, into))
        break
    write_partner_rows(store, "22", rows, per_frame)


def a_temporary_group_a_crash_left(store: Path) -> None:
    """A temporary Partner Rows group under a dot-prefixed name, as a writer that crashed before
    its one rename leaves behind, with the file it was building (partner-rows §5.1)."""
    stale = store / "partners" / ".22.tmp-crashed"
    stale.mkdir()
    (stale / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}')
    (stale / "frames.bin").write_bytes(b"\x00" * 16)


def bytes_appended_to_the_frames_file(store: Path) -> None:
    """Twenty-six bytes appended to chr22's frames chunk file: the metadata and both hashes still
    name the frames a writer wrote, and the file holds more than them (partner-rows §5.3)."""
    group = zarr.open_group(store / "partners" / "22", mode="r")
    file = layout.chunk_file(group["frames"])
    with file.open("ab") as stream:
        stream.write(bytes(range(26)))


def a_group_recording_another_entry_count(store: Path) -> None:
    """Records one entry more than chr22's frames hold."""
    group = zarr.open_group(store / "partners" / "22", mode="r+")
    group.attrs.update(entries=int(group.attrs["entries"]) + 1)


#: One way to break each rule, applied to a copy of the Test Store. ogld#4 writes each.
BREAKERS: dict[str, Callable[[Path], None]] = {
    # lacks far/22/partner
    "envelope.arrays": lambda store: shutil.rmtree(store / "far" / "22" / "partner"),
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
    # leaves a crashed writer's temporary group under partners/, and 26 bytes past the frames
    "partners.files": a_group_recording_another_row_count,
    # recompresses chr22's third frame with no checksum in its header
    "partners.frames": a_frame_without_a_checksum,
    # records one entry more than chr22's frames hold
    "partners.rows": a_group_recording_another_entry_count,
    # writes -32768 into a chr22 Partner Rows row
    "partners.codes": a_reserved_partner_code,
    # moves one entry's partner more than far_bp away
    "partners.distance": a_partner_beyond_far_bp,
    # writes a within-near_bp pair's codes below the Partner Floor, both ways
    "partners.threshold": a_partner_below_its_floor,
    # gives one pair's two entries different codes
    "partners.symmetric": a_pair_held_with_two_codes,
    # drops one direction of a pair within 3 Mb the band reaches
    "partners.near": a_missing_within_near_bp_entry,
    # drops one direction of a pair beyond 3 Mb
    "partners.far": a_missing_far_entry,
    # records a Partner Threshold above the store's t_far
    "provenance.partner_threshold": edit_provenance(
        lambda p: p.update(t_partner=p["t_far"] + 0.001)
    ),
}


def test_the_provenance_count_is_a_breaker_of_its_own_rule(tmp_path: Path) -> None:
    # A recorded count the table does not give fails `alid_index.count` alone: the keys and rows
    # are untouched (ogld#42's fix round 1).
    copy = tmp_path / "store"
    shutil.copytree(support.TEST_STORE, copy)
    edit_provenance(lambda p: p["counts"].update(alids_left_out=0))(copy)
    failing = {name for name, found in opengwasdb.ld.validate(copy).items() if found}
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
    findings = opengwasdb.ld.validate(copy)
    assert findings[rule], f"{rule} passed a store built to break it"


@pytest.mark.parametrize("rule", [r for r in VALIDATION_RULES if r.startswith("alid_index.")])
def test_an_index_rule_s_breaker_fails_only_that_rule(rule: str, tmp_path: Path) -> None:
    # The four index rules judge one array each, so a store built to break one must not be
    # refused for another: a reader trusts what they agree on (ogld#42's fix round 1). A rule whose
    # array the breaker shortened says it cannot be checked, which ld-store §12 requires of it, and
    # that is not a judgement against the store.
    copy = tmp_path / "store"
    shutil.copytree(support.TEST_STORE, copy)
    BREAKERS[rule](copy)
    found = opengwasdb.ld.validate(copy)
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
    failing = {name for name, found in opengwasdb.ld.validate(copy).items() if found}
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


#: ld-store §7 and §12: `k_min` applies only when a PC File was read, and an import has no
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
    findings = opengwasdb.ld.validate(with_provenance(tmp_path, edit))["provenance.threshold"]
    assert (not findings) is passes, f"{case}: {findings}"


# ------------------------------------------------------------------ the CLI


def test_ogld_validate_and_info_on_the_test_store(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["validate", str(support.TEST_STORE)]) == 0
    assert cli.main(["info", str(support.TEST_STORE)]) == 0
    out = capsys.readouterr().out
    assert "ok   partners.symmetric" in out
    assert '"t_far"' in out
