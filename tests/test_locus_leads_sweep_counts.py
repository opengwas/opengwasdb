"""The count criteria C1-C3 (#403), on hand-worked Analyses.

The criteria are fixed in `counts.COUNT_CRITERIA`; each test names what it pins.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from prototype_modules import load_prototype

counts = load_prototype("locus_leads", "counts")

W = 1_000_000
INDEX = [10, 20, 30, 40, 50, 60]
LEADS = [100, 110, 120, 130, 140, 150]


def _analysis(name: str, *, rows: tuple[int, int], index: int, z_band: str = "<38.5",
              rows_band: str = "<1k", at: dict[tuple[Any, int], tuple[int, int]]) -> dict[str, Any]:
    """`at` maps (D, W) to (Locus Leads in the full shared set, Locus Leads on HapMap3 rows)."""
    return {
        "analysis": {"analysis_id": name},
        "selection": {"z_band": z_band, "rows_band": rows_band},
        "density": {"rows_full": rows[0], "rows_hm3": rows[1]},
        "configs": [{"d_cm": d, "w_bp": w, "density": {"leads_full": full, "leads_hm3": hm3},
                     "by_r2": {"0.01": {"locus_leads": full, "index_variants": index}}}
                    for (d, w), (full, hm3) in at.items()],
    }


def _by_ratio(leads: dict[Any, list[int]], ratio: dict[Any, float], *,
              rows_kept: float = 0.1) -> list[dict[str, Any]]:
    """Six Analyses; at each D the HapMap3 lead count is the full count times the ratio."""
    return [_analysis(f"a{i}", rows=(1000, round(1000 * rows_kept)), index=INDEX[i],
                      at={(d, W): (leads[d][i], round(leads[d][i] * ratio[d])) for d in leads})
            for i in range(6)]


def _first(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = counts.evaluate_counts(analyses)["configs"][0]
    return result


def test_spearman_is_the_pearson_of_average_ranks() -> None:
    assert counts.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert counts.spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(1.5 / math.sqrt(3))
    assert counts.spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_refuses_a_constant_series() -> None:
    with pytest.raises(ValueError, match="constant"):
        counts.spearman([2, 2, 2], [1, 2, 3])


def test_c1_needs_rho_of_at_least_point_95() -> None:
    agree = _first(_by_ratio({"bp_only": [5, 9, 14, 22, 30, 41]}, {"bp_only": 1.0}))
    swapped = _first(_by_ratio({"bp_only": [9, 5, 14, 22, 30, 41]}, {"bp_only": 1.0}))
    assert agree["rho"] == pytest.approx(1.0) and agree["c1"]
    assert swapped["rho"] == pytest.approx(1 - 6 * 2 / (6 * 35))  # 0.9429
    assert not swapped["c1"]


def test_c2_needs_the_median_ratio_in_range_and_less_log_distance_than_half_the_rows() -> None:
    def at(ratio: float, rows_kept: float = 0.1) -> dict[str, Any]:
        return _first(_by_ratio({"bp_only": LEADS}, {"bp_only": ratio}, rows_kept=rows_kept))
    # Rows keep 10% (|log2| 3.32); leads keep 100%: ratio 1, |log2| 0.
    assert at(1.0)["median_ratio"] == 1.0 and at(1.0)["c2"]
    # 1.3 is outside [0.8, 1.25] though its |log2| (0.38) is under half of 3.32.
    high = at(1.3)
    assert high["median_abs_log2"] < 0.5 * high["rows_median_abs_log2"] and not high["c2"]
    assert not at(0.5)["c2"]
    # In range (0.85), but rows keep 80%: |log2| 0.234 is above half of the rows' 0.322.
    near = at(0.85, rows_kept=0.8)
    assert 0.8 <= near["median_ratio"] <= 1.25 and not near["c2"]


def test_the_median_is_over_analyses_and_the_ratio_is_hm3_over_full() -> None:
    analyses = [_analysis(f"a{i}", rows=(1000, 100), index=INDEX[i], at={("bp_only", W): (full, hm3)})
                for i, (full, hm3) in enumerate([(100, 90), (200, 200), (300, 330), (400, 480),
                                                 (500, 50)])]
    result = _first(analyses)
    assert result["median_ratio"] == pytest.approx(1.0)
    assert result["median_abs_log2"] == pytest.approx(abs(math.log2(0.9)))


def test_an_analysis_with_no_hm3_lead_is_an_infinite_log_distance_not_a_dropped_one() -> None:
    analyses = [_analysis(f"a{i}", rows=(1000, 100), index=INDEX[i], at={("bp_only", W): (full, hm3)})
                for i, (full, hm3) in enumerate([(100, 0), (200, 0), (300, 300)])]
    result = _first(analyses)
    assert result["analyses_without_hm3_leads"] == 2
    assert result["median_ratio"] == 0.0
    assert result["median_abs_log2"] is None and not result["c2"]  # None: infinite, not a number


def test_an_analysis_with_no_leads_at_all_is_refused() -> None:
    analyses = [_analysis(f"a{i}", rows=(1000, 100), index=INDEX[i], at={("bp_only", W): (n, n)})
                for i, n in enumerate([0, 1, 2])]
    with pytest.raises(ValueError, match="no Locus Lead"):
        counts.evaluate_counts(analyses)


BP_LEADS = [900, 500, 1400, 2200, 3000, 4100]  # rho 0.943 with the index variants
CM_LEADS = [500, 900, 1400, 2200, 3000, 4100]  # rho 1.0: an improvement of 0.057


def _c3(bp: tuple[list[int], float], cm: tuple[list[int], float]) -> dict[str, Any]:
    result = counts.evaluate_counts(_by_ratio({"bp_only": bp[0], 0.5: cm[0]},
                                              {"bp_only": bp[1], 0.5: cm[1]}))
    assert [c["w_bp"] for c in result["c3"]] == [W]
    c3: dict[str, Any] = result["c3"][0]
    return c3


def test_c3_holds_when_cm_improves_rho_by_point_01_and_is_not_worse_in_density() -> None:
    result = _c3((BP_LEADS, 1.0), (CM_LEADS, 1.0))
    assert result["holds"] and result["d_cm_that_holds"] == [0.5]


def test_c3_fails_when_rho_improves_but_density_is_worse_by_more_than_point_1() -> None:
    # |log2| of the density ratio: bp-only 0, cM 0.2 (ratio 1.15).
    assert not _c3((BP_LEADS, 1.0), (CM_LEADS, 1.15))["holds"]


def test_c3_holds_on_density_alone_when_rho_is_no_worse() -> None:
    assert _c3((CM_LEADS, 1.3), (CM_LEADS, 1.0))["holds"]  # |log2| 0.38 down to 0


def test_c3_fails_when_neither_margin_is_reached() -> None:
    assert not _c3((CM_LEADS, 1.0), (CM_LEADS, 1.0))["holds"]


def test_c3_fails_when_rho_is_worse_by_more_than_point_01_even_if_density_improves() -> None:
    assert not _c3((CM_LEADS, 1.3), (BP_LEADS, 1.0))["holds"]


def test_the_named_point_is_the_one_meeting_c1_and_c2_that_loses_the_fewest_leads_to_density() -> None:
    both = counts.evaluate_counts(_by_ratio({"bp_only": CM_LEADS, 0.5: CM_LEADS},
                                            {"bp_only": 0.9, 0.5: 0.95}))
    assert both["named"]["d_cm"] == 0.5
    # A config that fails C2 is never named, however many leads the other loses.
    failing = counts.evaluate_counts(_by_ratio({"bp_only": CM_LEADS, 0.5: CM_LEADS},
                                               {"bp_only": 0.9, 0.5: 0.5}))
    assert failing["named"]["d_cm"] == "bp_only"


def test_when_no_point_meets_c1_and_c2_the_result_says_none() -> None:
    result = counts.evaluate_counts(_by_ratio({"bp_only": BP_LEADS, 0.5: BP_LEADS},
                                              {"bp_only": 1.0, 0.5: 1.0}))
    assert result["named"] is None


def test_leads_per_index_variant_are_pooled_by_max_z_band_and_by_row_band() -> None:
    at = {("bp_only", W): (5, 5)}
    analyses = [
        _analysis("low", rows=(100, 10), index=10, z_band="<38.5", rows_band="<1k", at=at),
        _analysis("high1", rows=(100, 10), index=20, z_band=">=100", rows_band=">=100k",
                  at={("bp_only", W): (10, 10)}),
        _analysis("high2", rows=(100, 10), index=20, z_band=">=100", rows_band=">=100k",
                  at={("bp_only", W): (4, 4)})]
    table = counts.leads_per_index_variant(analyses)
    assert len(table) == 1 and (table[0]["d_cm"], table[0]["w_bp"]) == ("bp_only", W)
    assert table[0]["z"][">=100"] == {"analyses": 2, "locus_leads": 14, "index_variants": 40,
                                      "ratio": pytest.approx(14 / 40)}
    assert table[0]["z"]["<38.5"]["ratio"] == pytest.approx(0.5)
    assert table[0]["rows"][">=100k"]["locus_leads"] == 14
