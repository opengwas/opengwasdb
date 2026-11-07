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

import numpy as np
import pytest
import support
from support import alids, contract, panel_data, passing_rows, reference, reference_matrix

import ogld
from ogld import cli
from ogld.format import VALIDATION_RULES, RefusalReason, quantise
from ogld.store import StoreWriter

OGLD_4 = contract("ogld#4")


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


# ------------------------------------------------------------------ opening


@OGLD_4
def test_the_test_store_records_its_panel_and_its_distances() -> None:
    p = opened().provenance
    assert p["format_version"] == "0.1.0"
    assert p["panel"]["n"] == support.N_SAMPLES
    assert p["adjustment"]["k"] == support.K
    assert p["adjustment"]["covariates"] == [f"PC{i}" for i in range(1, 11)]
    assert p["t_far"] == support.T_FAR
    assert p["far_code_floor"] == support.FLOOR == 5426
    assert (p["near_bp"], p["far_bp"]) == (3_000_000, 10_000_000)
    assert p["source"]["kind"] == "genotypes"
    assert p["n_variants"] == len(passing_rows())


@OGLD_4
def test_a_reader_refuses_a_format_version_it_does_not_implement(tmp_path: Path) -> None:
    opened()
    copy = tmp_path / "store"
    shutil.copytree(support.TEST_STORE, copy)
    meta = copy / "zarr.json"
    doc = json.loads(meta.read_text())
    doc["attributes"]["ld_store"]["format_version"] = "9.0.0"
    meta.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=r"9\.0\.0"):
        ogld.open(copy)


@OGLD_4
def test_the_test_store_reads_the_same_over_http(served_store: str) -> None:
    # Spec §1: a local path or an HTTP(S) URL. A static Range-serving server, as #9 may use.
    remote = ogld.open(served_store)
    assert remote.provenance == opened().provenance
    some = alids(passing_rows()[:40])
    np.testing.assert_array_equal(remote.matrix(some).r, opened().matrix(some).r)


# ------------------------------------------------------------------ pair answers (spec §8)


@OGLD_4
def test_a_pair_exactly_3_mb_apart_is_the_near_band_s_exact_r() -> None:
    store = opened()
    pair = reference().where(reference().distance == 3_000_000, "a pair exactly 3 Mb apart")
    r, kind = two(store, int(pair.row_i[0]), int(pair.row_j[0]))
    assert kind == "stored"
    assert abs(r - pair.r[0]) <= support.HALF_STEP + support.SLACK


@OGLD_4
def test_a_pair_beyond_3_mb_below_t_far_is_nan_and_the_bound() -> None:
    store = opened()
    ref = reference()
    far = (ref.distance > 3_000_000) & (ref.distance <= 10_000_000)
    pair = ref.where(far & (np.abs(ref.code) < support.FLOOR), "a 3–10 Mb pair below t_far")
    r, kind = two(store, int(pair.row_i[0]), int(pair.row_j[0]))
    assert math.isnan(r), "an unstored pair is NaN, never a number"
    assert kind == "bound"


@OGLD_4
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


@OGLD_4
def test_exactly_10_mb_is_measured_and_one_base_more_is_not() -> None:
    store = opened()
    at = reference().where(reference().distance == 10_000_000, "a pair exactly 10 Mb apart")
    assert two(store, int(at.row_i[0]), int(at.row_j[0]))[1] in {"stored", "bound"}
    beyond = a_pair_apart(10_000_001)
    assert beyond is not None, "the Test Panel must hold a pair 10,000,001 bp apart"
    r, kind = two(store, *beyond)
    assert math.isnan(r)
    assert kind == "not_measured"


@OGLD_4
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


@OGLD_4
def test_two_records_at_one_position_are_a_near_band_pair() -> None:
    store = opened()
    pair = reference().where(reference().distance == 0, "two records at one position")
    r, kind = two(store, int(pair.row_i[0]), int(pair.row_j[0]))
    assert kind == "stored"
    assert abs(r - pair.r[0]) <= support.HALF_STEP + support.SLACK


# ------------------------------------------------------------------ matrices, NaN and fill


@OGLD_4
def test_the_whole_store_reads_back_within_half_a_step() -> None:
    # The round trip: every pair of both chromosomes against the reference r, signed to A1.
    m = whole_store()
    ref = expected()
    stored = m.kinds() == "stored"
    assert stored.sum() > 1000, "the store must hold many pairs for this to mean anything"
    assert not np.isnan(ref[stored]).any(), "every stored pair is in the reference"
    assert np.all(np.abs(m.r[stored] - ref[stored]) <= support.HALF_STEP + support.SLACK)
    assert np.isnan(m.r[~stored]).all(), "every pair the store does not hold is NaN"


@OGLD_4
def test_every_stored_cell_is_exactly_what_the_codes_say() -> None:
    m = whole_store()
    ref = expected()
    stored = ~np.isnan(m.r)
    np.fill_diagonal(stored, False)
    assert stored.any()
    np.testing.assert_array_equal(m.r[stored], quantise(ref[stored]) / 32767)


@OGLD_4
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


@OGLD_4
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


@OGLD_4
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


@OGLD_4
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


@OGLD_4
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


@OGLD_4
def test_the_writer_refuses_a_location_that_exists(tmp_path: Path) -> None:
    # Against a copy: a broken writer is never pointed at the committed Test Store.
    variants = opened().matrix(alids(passing_rows()[:2])).variants
    copy = tmp_path / "store"
    shutil.copytree(support.TEST_STORE, copy)
    before = sorted(p.relative_to(copy) for p in copy.rglob("*"))
    with pytest.raises(FileExistsError):
        StoreWriter.create(copy, variants=variants, near_bp=3_000_000, far_bp=10_000_000)
    assert sorted(p.relative_to(copy) for p in copy.rglob("*")) == before


# ------------------------------------------------------------------ validation (spec §12)


@OGLD_4
def test_the_test_store_passes_every_rule() -> None:
    findings = ogld.validate(support.TEST_STORE)
    assert tuple(findings) == VALIDATION_RULES
    assert {rule: found for rule, found in findings.items() if found} == {}


def unbuilt(how: str) -> Callable[[Path], None]:
    def breaker(store: Path) -> None:
        raise NotImplementedError(f"ogld#4: write a copy of the Test Store that {how}")

    return breaker


#: One way to break each rule, applied to a copy of the Test Store. ogld#4 writes each.
BREAKERS: dict[str, Callable[[Path], None]] = {
    "envelope.arrays": unbuilt("lacks index/22/partner"),
    "envelope.fill": unbuilt("sets near/22/r's fill value to 0"),
    "provenance.complete": unbuilt("drops the key adjustment.covariates"),
    "provenance.encoding": unbuilt("records r_scale 32768"),
    "provenance.distances": unbuilt("records far_bp 12,000,000"),
    "provenance.threshold": unbuilt("records t_far 0.02 for n = 771, k = 10"),
    "variants.chromosomes": unbuilt("gives chr22's range a gap"),
    "variants.order": unbuilt("swaps two positions on chr22"),
    "variants.identity": unbuilt("writes an ALID with its alleles reversed"),
    "variants.frequency": unbuilt("sets one a1_frequency to 0"),
    "near.rows": unbuilt("shortens one Near Band row by one code"),
    "near.codes": unbuilt("writes −32768 into near/22/r"),
    "index.rows": unbuilt("repeats a partner within one index row"),
    "index.codes": unbuilt("writes −32768 into index/22/r"),
    "index.distance": unbuilt("adds an index entry 10,000,001 bp apart"),
    "index.threshold": unbuilt("adds an index entry with |code| = far_code_floor − 1"),
    "index.symmetric": unbuilt("drops one direction of one index pair"),
    "index.near": unbuilt(
        "drops an index entry within 3 Mb whose Near Band code reaches the floor"
    ),
}


def test_every_rule_has_a_way_to_break_it() -> None:
    # Not a contract test: it holds the table above to the spec's rule list now.
    assert tuple(BREAKERS) == VALIDATION_RULES


@OGLD_4
@pytest.mark.parametrize("rule", VALIDATION_RULES)
def test_a_store_that_breaks_a_rule_fails_that_rule(rule: str, tmp_path: Path) -> None:
    assert not any(ogld.validate(support.TEST_STORE).values())
    copy = tmp_path / "store"
    shutil.copytree(support.TEST_STORE, copy)
    BREAKERS[rule](copy)
    findings = ogld.validate(copy)
    assert findings[rule], f"{rule} passed a store built to break it"


# ------------------------------------------------------------------ the CLI


@OGLD_4
def test_ogld_validate_and_info_on_the_test_store(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["validate", str(support.TEST_STORE)]) == 0
    assert cli.main(["info", str(support.TEST_STORE)]) == 0
    out = capsys.readouterr().out
    assert "ok   index.symmetric" in out
    assert '"t_far"' in out
