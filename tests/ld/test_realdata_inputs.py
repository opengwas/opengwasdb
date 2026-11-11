"""The realdata inputs script's pure parts (ogld#6): the ranking plink2 is given, and which pvar
record of a shared ALID the store holds. The comparison itself is the realdata tier's."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from script_modules import load_script

script = load_script("realdata/make_403_inputs")


def test_minus_log10_p_matches_the_normal_tail_and_does_not_underflow() -> None:
    assert script.minus_log10_p(5.45) == pytest.approx(-math.log10(math.erfc(5.45 / math.sqrt(2))))
    assert 7.2 < script.minus_log10_p(5.45) < 7.4, "5e-8 is z ≈ 5.45"
    assert script.minus_log10_p(342.7) > 25_000, "p = 0 in a double, finite here"


def test_minus_log10_p_is_strictly_increasing_across_the_switch_to_the_asymptotic_form() -> None:
    zs = [29.9, 29.99, 30.0, 30.01, 30.1, 38.0, 38.25, 100.0, 100.25]
    values = [script.minus_log10_p(z) for z in zs]
    assert values == sorted(set(values)), "equal |z| must be equal p and a larger |z| a smaller p"
    assert script.minus_log10_p(30.0) == pytest.approx(
        -math.log10(math.erfc(30.0 / math.sqrt(2))), rel=1e-9
    )


def test_a_hit_the_store_does_not_hold_is_counted_and_dropped() -> None:
    hits = [("1:5:A:C", 6.0, "1"), ("1:9:A:C", 7.0, "1"), ("2:3:A:G", 8.0, "2")]
    kept, unmatched = script.matched(hits, {"1:5:A:C", "2:3:A:G"})
    assert [h[0] for h in kept] == ["1:5:A:C", "2:3:A:G"]
    assert unmatched == 1


def test_of_an_insertion_and_a_deletion_the_one_the_builder_kept_is_the_one_prepared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 1:10:C:CT is two records: C>CT (REF = A1) and CT>C. The builder filters first, so a
    # monomorphic C>CT is dropped and CT>C is the store's; with both polymorphic, REF = A1 wins.
    found = {
        "1:10:C:CT": [("1:10:C:CT", "C", "C"), ("1:10:CT:C", "CT", "C")],
        "1:20:A:G": [("1:20:A:G", "A", "A")],
    }
    monkeypatch.setattr(script, "minor_counts", lambda *_: {"1:10:C:CT": 0, "1:10:CT:C": 40})
    assert script.resolve(found, tmp_path, tmp_path, 20) == {
        "1:10:CT:C": "1:10:C:CT",
        "1:20:A:G": "1:20:A:G",
    }
    monkeypatch.setattr(script, "minor_counts", lambda *_: {"1:10:C:CT": 30, "1:10:CT:C": 40})
    assert script.resolve(found, tmp_path, tmp_path, 20)["1:10:C:CT"] == "1:10:C:CT"
