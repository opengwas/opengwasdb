"""The pair masks have one owner, and the builder, emitter and Genotype reader call it (#91)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import build_fixtures
import numpy as np
import pytest
import testdata

import opengwasdb.ld
from opengwasdb.ld.format import REncoding
from opengwasdb.ld.format.encoding import FAR_BP, NEAR_BP

SCALE = 511


def encoding() -> REncoding:
    return REncoding(SCALE)


def test_the_stored_mask_is_one_rule_for_distance_and_code() -> None:
    e = encoding()
    floors = e.far_code_floor(0.01), e.partner_code_floor(0.01, 0.01)
    far_floor, _ = floors
    assert far_floor > 1
    distance = np.array([0, NEAR_BP, NEAR_BP + 1, FAR_BP, FAR_BP + 1], dtype=np.int64)
    codes = np.array([0, 0, far_floor - 1, far_floor, far_floor], dtype=np.int16)
    stored = e.stored_mask(
        distance, codes, near_bp=NEAR_BP, far_bp=FAR_BP, far_code_floor=far_floor
    )
    # Every pair within near_bp is stored, whatever its code; beyond it the far floor decides.
    np.testing.assert_array_equal(stored, [True, True, False, True, False])


def test_the_partner_mask_chooses_the_floor_at_the_distance() -> None:
    e = encoding()
    far_floor = e.far_code_floor(0.01)
    partner_floor = e.partner_code_floor(0.001, 0.01)
    assert partner_floor < far_floor, "the fixture must separate the two floors"
    distance = np.array([NEAR_BP, NEAR_BP, NEAR_BP + 1, NEAR_BP + 1, FAR_BP + 1], dtype=np.int64)
    codes = np.array([partner_floor, partner_floor - 1, far_floor, far_floor - 1, far_floor])
    kept = e.partner_mask(
        distance,
        codes,
        near_bp=NEAR_BP,
        far_bp=FAR_BP,
        partner_code_floor=partner_floor,
        far_code_floor=far_floor,
    )
    np.testing.assert_array_equal(kept, [True, False, True, False, False])


def counted_masks(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Counts each call of the encoding's two masks, by `stored` and `partner`."""
    calls = {"stored": 0, "partner": 0}
    stored, partner = REncoding.stored_mask, REncoding.partner_mask

    def counted_stored(self: REncoding, *args: Any, **kwargs: Any) -> np.ndarray:
        calls["stored"] += 1
        return stored(self, *args, **kwargs)

    def counted_partner(self: REncoding, *args: Any, **kwargs: Any) -> np.ndarray:
        calls["partner"] += 1
        return partner(self, *args, **kwargs)

    monkeypatch.setattr(REncoding, "stored_mask", counted_stored)
    monkeypatch.setattr(REncoding, "partner_mask", counted_partner)
    return calls


def test_the_builder_and_the_partner_rows_emitter_call_the_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = counted_masks(monkeypatch)
    out = tmp_path / "store"
    opengwasdb.ld.build(build_fixtures.inputs(testdata.panel().pgen), out)
    assert not any(opengwasdb.ld.validate(out).values())
    assert calls["stored"], calls
    assert calls["partner"], calls


def test_the_genotype_reader_calls_the_masks(
    genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = counted_masks(monkeypatch)
    names = build_fixtures.alids(np.flatnonzero(testdata.variants().passes_filters)[:20])
    store = opengwasdb.ld.open(genotype_store)
    assert store.matrix(names).r.shape == (len(names), len(names))
    assert store.proxies([names[0]], r2=0.01, kb=3000) is not None
    assert calls == {"stored": 1, "partner": 1}, calls


def test_the_emitter_keeps_each_partner_at_its_real_distance(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The Partner Rows emitter's boundary distances equal the per-pair rule, pair by pair."""
    from contract.support import TEST_STORE

    from opengwasdb.ld.store import StoreReader

    _, built = build_fixtures.test_panel_build(10, tmp_path_factory)
    reader = StoreReader.open(built)
    provenance = reader.provenance
    oracle = opengwasdb.ld.open(TEST_STORE)
    rows = np.flatnonzero(testdata.variants().passes_filters)
    names = build_fixtures.alids(rows)
    # The built store's rows are the passing panel rows in the same order, so its row of a
    # variant is its index here; the check holds the fixture to that before using it.
    assert reader.alids_of(np.arange(len(names))).tolist() == names
    matrix = oracle.matrix(names)
    position = testdata.variants().pos[rows].astype(np.int64)
    chrom = testdata.variants().chrom[rows]
    e = encoding()
    near_bp, far_bp = provenance["near_bp"], provenance["far_bp"]
    near = far = 0
    for at in range(0, len(rows), 37):
        distance = np.abs(position - position[at])
        judged = np.isfinite(matrix.r[at]) & (distance > 0) & (distance <= far_bp)
        assert judged.any(), "the fixture must judge at least one pair for every sampled row"
        keep = e.partner_mask(
            distance[judged],
            e.quantise(matrix.r[at, judged]),
            near_bp=near_bp,
            far_bp=far_bp,
            partner_code_floor=provenance["partner_code_floor"],
            far_code_floor=provenance["far_code_floor"],
        )
        start, _ = reader.chromosome_range(str(chrom[at]))
        expected = {int(i) - start for i in np.flatnonzero(judged)[keep]}
        partners, _ = reader.partner_row(str(chrom[at]), at - start)
        assert set(partners.tolist()) == expected, f"row {at} keeps another set of partners"
        near += int((keep & (distance[judged] <= near_bp)).sum())
        far += int((keep & (distance[judged] > near_bp)).sum())
    # The probe must exercise both sides of the floor choice, or it proves only one branch.
    assert near > 0, "no kept near partner was judged"
    assert far > 0, "no kept Far Pair was judged"
    reader.close()
