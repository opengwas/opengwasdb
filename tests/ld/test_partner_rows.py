"""The Partner Rows a store holds, on the committed Test Store (partner-rows §3, §5, §6; ogld#67).

Every expected entry is reconstructed here from the store's own Near Band and Far Pairs with plain
zarr and numpy, by ld-store §4's and §5's rules written out in this file: the Near Band's addresses,
the Far Pairs' later rows, the Partner Floor at t_partner 0.01, and every kept pair in both
variants' rows. The emitter's selection is never called, so a second layout of the same codes is
held to the codes, not to itself.

The store is the committed one, which the writer wrote (`scripts/ld/make_test_store.py --t-partner
0.01`); nothing here writes into `tests/ld/data/`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import zarr
from partner_oracle import Pairs, chromosomes, decoded_frames, far_pairs, near_pairs, store
from store_copies import FLOOR, TEST_STORE

#: The Test Store's distances (ld-store §7).
NEAR_BP, FAR_BP = 3_000_000, 10_000_000
#: Its Partner Threshold, and the Partner Floor that is the smallest code c ≥ 0 with
#: (c / 511)² ≥ 0.01, computed here rather than read from the store (ld-store §3; partner-rows §3).
T_PARTNER = 0.01
PARTNER_FLOOR = 52
CHROMOSOMES = ("21", "22")


def kept(group: zarr.Group, chrom: str) -> Pairs:
    """The pairs the Partner Rows must hold of one chromosome, each once and both ways: every
    Near Band pair at the Partner Floor, and every Far Pair (which is at t_far, never above it)."""
    near_lo, near_hi, near_codes = near_pairs(group, chrom)
    far_lo, far_hi, far_codes = far_pairs(group, chrom)
    reaches = np.abs(near_codes) >= PARTNER_FLOOR
    near_lo, near_hi = near_lo[reaches], near_hi[reaches]
    near_codes = near_codes[reaches]
    lo = np.concatenate([near_lo, near_hi, far_lo, far_hi])
    hi = np.concatenate([near_hi, near_lo, far_hi, far_lo])
    codes = np.concatenate([near_codes, near_codes, far_codes, far_codes])
    order = np.lexsort((hi, lo))
    return lo[order], hi[order], codes[order]


def decoded(group: zarr.Group, chrom: str) -> Pairs:
    """Every entry a chromosome's Partner Rows hold, frame by frame, as (row, partner, code)."""
    rows_of, partners, codes = [], [], []
    for first, frame in decoded_frames(group["partners"][chrom], chrom):
        for row in range(frame.rows):
            held_partners, held_codes = frame.row(first + row)
            rows_of.append(np.full(len(held_partners), first + row, dtype=np.int64))
            partners.append(held_partners.astype(np.int64))
            codes.append(held_codes)
    return (
        np.concatenate(rows_of),
        np.concatenate(partners),
        np.concatenate(codes),
    )


@pytest.mark.parametrize("chrom", CHROMOSOMES)
def test_every_row_is_the_near_band_and_the_far_pairs_at_their_floors(chrom: str) -> None:
    group = store()
    want = kept(group, chrom)
    got = decoded(group, chrom)
    assert len(want[0]) > 1_000, "the Test Store's Partner Rows must hold many entries"
    assert (want[1] < want[0]).any(), "a row holds partners upstream"
    assert (want[1] > want[0]).any(), "and downstream"
    np.testing.assert_array_equal(got[0], want[0], "row after row")
    np.testing.assert_array_equal(got[1], want[1], "partners ascending within each row")
    np.testing.assert_array_equal(got[2], want[2], "each at its pair's own code")


@pytest.mark.parametrize("chrom", CHROMOSOMES)
def test_the_partner_rows_hold_every_entry_a_proxy_index_held(chrom: str) -> None:
    # t_partner is never above t_far, so every entry the Proxy Index held is an entry here: the
    # Near Band's pairs at the far code floor, and the Far Pairs, each with its own code, both
    # ways (partner-rows §3; ADR 0097, D1). No store holds an index since opengwas/ogld#72, so the
    # set is built from the two components the index held (opengwas/ogld#72): a second reading of
    # the same codes, not the emitter's selection.
    group = store()
    near_lo, near_hi, near_codes = near_pairs(group, chrom)
    far_lo, far_hi, far_codes = far_pairs(group, chrom)
    at_floor = np.abs(near_codes.astype(np.int64)) >= FLOOR
    held = set(
        zip(
            np.concatenate([near_lo[at_floor], near_hi[at_floor], far_lo, far_hi]).tolist(),
            np.concatenate([near_hi[at_floor], near_lo[at_floor], far_hi, far_lo]).tolist(),
            np.concatenate(
                [near_codes[at_floor], near_codes[at_floor], far_codes, far_codes]
            ).tolist(),
            strict=True,
        )
    )
    assert len(held) > 1_000, "the Test Store must hold many entries for this to mean anything"
    mine = decoded(group, chrom)
    here = set(zip(mine[0].tolist(), mine[1].tolist(), mine[2].tolist(), strict=True))
    assert held <= here, (
        f"{len(held - here)} entries a Proxy Index held are not in the Partner Rows"
    )


def test_the_store_says_its_partner_threshold_and_its_floor() -> None:
    recorded: dict[str, Any] = store().attrs["ld_store"]
    assert recorded["format_version"] == "0.2.0"
    assert recorded["t_partner"] == T_PARTNER
    assert recorded["partner_code_floor"] == PARTNER_FLOOR
    entries = sum(int(store()["partners"][c].attrs["entries"]) for c in CHROMOSOMES)
    assert recorded["counts"]["partner_entries"] == entries > 0
    assert isinstance(recorded["software"]["zstd"], str), "a frame's bytes depend on it"


def test_zarr_lists_the_test_stores_partner_rows_as_a_node_with_no_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        members = zarr.open_group(str(TEST_STORE), mode="r").members(max_depth=None)
    warned = [str(w.message) for w in caught]
    assert not warned, f"a node in the hierarchy is listed, never warned of: {warned}"
    listed = {str(path) for path, _ in members}
    for chrom in CHROMOSOMES:
        assert f"partners/{chrom}/frames" in listed, listed
        assert f"partners/{chrom}/frames_idx" in listed, listed


def test_a_rows_entries_lie_within_far_bp_and_at_their_floors() -> None:
    group = store()
    positions = np.asarray(group["variants"]["position"][:], dtype=np.int64)
    for chrom in CHROMOSOMES:
        start, stop = chromosomes(group)[chrom]
        local = positions[start:stop]
        rows, partners, codes = decoded(group, chrom)
        gap = np.abs(local[partners] - local[rows])
        assert (gap <= FAR_BP).all(), "no entry past far_bp"
        assert (partner_floor(codes, gap) <= np.abs(codes)).all(), "each at its distance's floor"


def partner_floor(
    codes: npt.NDArray[np.int16], gap: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """The floor each entry's distance calls for, written out here: the Partner Floor within
    near_bp and the far code floor (85 at × 511 for the Test Store's t_far) beyond it."""
    return np.where(gap <= NEAR_BP, PARTNER_FLOOR, 85).astype(np.int64)
