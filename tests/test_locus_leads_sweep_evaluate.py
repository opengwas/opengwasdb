"""Whether a (D, W) qualifies, through `evaluate` and against the bp-only rule (#403)."""

from __future__ import annotations

from typing import Any

from prototype_modules import load_prototype

sweep = load_prototype("locus_leads", "sweep")

W = 1_000_000
BAND = {"n": 0, "r2_undefined": 0, "r2_ge_0.01": 0, "r2_ge_0.001": 0}
PAIRS = {"n": 0, "r2_undefined": 0, "r2_histogram": {"[0,0.001)": 0},
         "by_distance_band": {"<=1Mb": BAND}, "straddling_a_spike": 0, "straddling": {"TSI": 0}}


def _record(splits: int, lost: int, leads: int = 100, index: int = 100) -> dict[str, Any]:
    return {"locus_leads": leads, "index_variants": index, "spurious_splits": splits,
            "lost_signals": lost, "by_abs_z": {"[0,7)": {"locus_leads": leads}},
            "spurious_split_pairs": PAIRS, "lost_signal_pairs": PAIRS}


def _analyses(configs: dict[Any, tuple[int, int]], n: int = 3) -> list[dict[str, Any]]:
    """`n` identical Analyses; each D's (spurious splits, lost signals) per 100 leads / index."""
    return [{"configs": [{"d_cm": d, "w_bp": W, "by_r2": {"0.01": _record(*counts)}}
                         for d, counts in configs.items()]} for _ in range(n)]


def _row(result: dict[str, Any], d_cm: Any) -> dict[str, Any]:
    row: dict[str, Any] = next(r for r in result["configs"] if r["d_cm"] == d_cm)
    return row


BP = (20, 20)  # 20% of each, 20% disagreement


def test_both_rates_at_ten_percent_and_clearly_better_than_bp_qualifies() -> None:
    result = sweep.evaluate(_analyses({"bp_only": BP, 0.5: (10, 10)}))
    assert _row(result, 0.5)["clearly_better_than_bp_only"]
    assert _row(result, 0.5)["qualifies_pooled"] and _row(result, 0.5)["qualifies_median"]
    assert result["best"] == {"qualifies": True, "d_cm": 0.5, "w_bp": W}


def test_a_lost_signal_rate_over_ten_percent_alone_disqualifies() -> None:
    result = sweep.evaluate(_analyses({"bp_only": BP, 0.5: (10, 11)}))
    assert _row(result, 0.5)["clearly_better_than_bp_only"]
    assert not _row(result, 0.5)["qualifies_pooled"]
    assert not _row(result, 0.5)["qualifies_median"]
    assert result["best"]["qualifies"] is False


def test_a_spurious_split_rate_over_ten_percent_alone_disqualifies() -> None:
    result = sweep.evaluate(_analyses({"bp_only": BP, 0.5: (11, 10)}))
    assert _row(result, 0.5)["clearly_better_than_bp_only"]
    assert not _row(result, 0.5)["qualifies_pooled"]


def test_rates_within_ten_percent_do_not_qualify_when_bp_only_is_as_good() -> None:
    result = sweep.evaluate(_analyses({"bp_only": (10, 10), 0.5: (10, 10)}))
    assert not _row(result, 0.5)["clearly_better_than_bp_only"]
    assert not _row(result, 0.5)["qualifies_pooled"]


def test_a_five_percent_cut_on_bp_only_is_not_clearly_better_even_in_every_analysis() -> None:
    result = sweep.evaluate(_analyses({"bp_only": (10, 10), 0.5: (10, 9)}))  # 19 against 20
    assert not _row(result, 0.5)["clearly_better_than_bp_only"]
    assert not _row(result, 0.5)["qualifies_pooled"]


def test_the_comparison_is_against_the_bp_only_rule_at_the_same_w() -> None:
    analyses = _analyses({"bp_only": BP, 0.5: (10, 10)})
    for analysis in analyses:
        other = [{"d_cm": d, "w_bp": 2 * W, "by_r2": {"0.01": _record(*counts)}}
                 for d, counts in {"bp_only": (10, 10), 0.5: (10, 10)}.items()]
        analysis["configs"] += other
    result = sweep.evaluate(analyses)
    at = {(r["d_cm"], r["w_bp"]): r for r in result["configs"]}
    assert at[(0.5, W)]["qualifies_pooled"]
    assert not at[(0.5, 2 * W)]["qualifies_pooled"]


def test_the_nearest_point_is_the_one_with_the_smallest_larger_rate_when_none_qualifies() -> None:
    result = sweep.evaluate(_analyses({"bp_only": BP, 0.1: (30, 5), 0.5: (12, 14)}))
    assert result["best"] == {"qualifies": False, "nearest_d_cm": 0.5, "nearest_w_bp": W}
