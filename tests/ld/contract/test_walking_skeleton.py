"""The walking skeleton's contract: the thinnest real path, and the Test Store it writes.

pgen → r at k = 0 → the Near Band → write → read → one pair answered. The skeleton's script,
`scripts/make_test_store.py`, writes the Test Store from the Test Panel's `reference_r(10)`
into `tests/data/test-panel/store/`; the ticket runs it, and no test writes there. Its
`write_test_store(k, out)` also gives ogld#6 a k = 0 store. From then on ogld#4–#6 build
against real data in the format.

These tests go through the store seam, `StoreReader`, not the queries, which are ogld#4's
and ogld#6's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import support
import zarr
from script_modules import load_script
from support import alids, panel_data, passing_rows, reference

import ogld
from ogld.store import StoreReader


def near_pair_r(reader: StoreReader, k: int) -> tuple[float, float]:
    """A Near Band pair's r as the store reads it back, and as the reference has it."""
    ref = reference(k)
    pair = ref.where((ref.distance > 0) & (ref.distance <= 3_000_000), "a Near Band pair")
    rows = reader.rows_of(alids([pair.row_i[0], pair.row_j[0]]))
    assert (rows >= 0).all(), "both variants must be in the Test Store"
    return float(reader.pair_r(rows)[0, 1]), float(pair.r[0])


def test_the_test_store_opens_through_ogld_open() -> None:
    store = ogld.open(support.TEST_STORE)
    assert (store.t_far, store.near_bp, store.far_bp) == (support.T_FAR, 3_000_000, 10_000_000)


def test_one_near_band_pair_reads_back_its_reference_r() -> None:
    got, expected = near_pair_r(StoreReader.open(support.TEST_STORE), 10)
    assert abs(got - expected) <= support.HALF_STEP + support.SLACK


def test_pairs_the_store_does_not_hold_read_as_nan() -> None:
    reader = StoreReader.open(support.TEST_STORE)
    ref = reference()
    far = (ref.distance > 3_000_000) & (ref.distance <= 10_000_000)
    bound = ref.where(far & (np.abs(ref.code) < support.FLOOR), "a 3–10 Mb pair below t_far")
    v = panel_data().variants()
    rows = passing_rows()
    pos, chrom = v.pos[rows].astype(np.int64), v.chrom[rows]
    apart = (pos[None, :] - pos[:, None] > 10_000_000) & (chrom[None, :] == chrom[:, None])
    assert apart.any(), "the Test Panel must hold a pair more than 10 Mb apart"
    a, b = np.argwhere(apart)[0]
    for i, j in ((int(bound.row_i[0]), int(bound.row_j[0])), (int(rows[a]), int(rows[b]))):
        found = reader.rows_of(alids([i, j]))
        assert (found >= 0).all(), "both variants must be in the Test Store"
        assert np.isnan(reader.pair_r(found)[0, 1]), f"rows {i} and {j}: unstored, so NaN"


def test_the_test_store_can_be_written_again_at_k0(tmp_path: Path) -> None:
    StoreReader.open(support.TEST_STORE)
    out = load_script("make_test_store").write_test_store(0, tmp_path / "k0", support.R_SCALE)
    reader = StoreReader.open(out)
    assert reader.provenance["adjustment"]["k"] == 0
    got, expected = near_pair_r(reader, 0)
    assert abs(got - expected) <= support.HALF_STEP + support.SLACK


def arrays(group: zarr.Group, prefix: str = "") -> dict[str, np.ndarray]:
    """Every array under a Zarr group, decoded, by path."""
    out: dict[str, np.ndarray] = {}
    for name, member in group.members():
        if isinstance(member, zarr.Array):
            out[prefix + name] = np.asarray(member[:])
        else:
            out |= arrays(member, f"{prefix}{name}/")
    return out


def test_the_script_writes_the_committed_test_store_again(tmp_path: Path) -> None:
    # Into tmp_path, never over the committed store: the ticket runs the script into
    # tests/data/test-panel/store/, and this holds the two to one set of arrays.
    StoreReader.open(support.TEST_STORE)
    out = load_script("make_test_store").write_test_store(10, tmp_path / "again", support.R_SCALE)
    committed = arrays(zarr.open_group(support.TEST_STORE, mode="r"))
    again = arrays(zarr.open_group(out, mode="r"))
    assert committed, "the committed Test Store must hold arrays for this to mean anything"
    assert committed.keys() == again.keys()
    for path, array in committed.items():
        np.testing.assert_array_equal(again[path], array, err_msg=path)
