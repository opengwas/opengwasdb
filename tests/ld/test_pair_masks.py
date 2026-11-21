"""The pair masks have one owner, and the builder, emitter and Genotype reader call it (#91)."""

from __future__ import annotations

from pathlib import Path

import build_fixtures
import numpy as np
import pytest
import testdata

import ogld
from ogld.format import REncoding
from ogld.format.encoding import FAR_BP, NEAR_BP

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


def test_the_builder_and_the_partner_rows_emitter_call_the_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"stored": 0, "partner": 0}
    stored, partner = REncoding.stored_mask, REncoding.partner_mask

    def counted_stored(self: REncoding, *args: object, **kwargs: object) -> np.ndarray:
        calls["stored"] += 1
        return stored(self, *args, **kwargs)  # type: ignore[arg-type]

    def counted_partner(self: REncoding, *args: object, **kwargs: object) -> np.ndarray:
        calls["partner"] += 1
        return partner(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(REncoding, "stored_mask", counted_stored)
    monkeypatch.setattr(REncoding, "partner_mask", counted_partner)
    out = tmp_path / "store"
    ogld.build(build_fixtures.inputs(testdata.panel().pgen), out)
    assert not any(ogld.validate(out).values())
    assert calls["stored"], calls
    assert calls["partner"], calls


def test_the_genotype_reader_calls_the_masks(
    genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"stored": 0, "partner": 0}
    stored, partner = REncoding.stored_mask, REncoding.partner_mask

    def counted_stored(self: REncoding, *args: object, **kwargs: object) -> np.ndarray:
        calls["stored"] += 1
        return stored(self, *args, **kwargs)  # type: ignore[arg-type]

    def counted_partner(self: REncoding, *args: object, **kwargs: object) -> np.ndarray:
        calls["partner"] += 1
        return partner(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(REncoding, "stored_mask", counted_stored)
    monkeypatch.setattr(REncoding, "partner_mask", counted_partner)
    names = build_fixtures.alids(np.flatnonzero(testdata.variants().passes_filters)[:20])
    store = ogld.open(genotype_store)
    assert store.matrix(names).r.shape == (len(names), len(names))
    assert store.proxies([names[0]], r2=0.01, kb=3000) is not None
    assert calls == {"stored": 1, "partner": 1}, calls
