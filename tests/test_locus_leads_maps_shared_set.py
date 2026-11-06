"""Every map must leave the same rows to judge (prototypes/locus_leads, #404).

The comparison between maps only means something if each map ranks the same
rows: a map with no cM on a chromosome drops that chromosome's rows out of the
shared set, and its rates would then be measured on other hits. So the shared
sets are asserted equal by count and by a digest of (Store, Analysis, Store
Variant Index), and a difference stops the run.
"""

from __future__ import annotations

import numpy as np
import pytest
from prototype_modules import load_prototype

maps = load_prototype("locus_leads", "maps")
genetic_map = load_prototype("locus_leads", "genetic_map")
run = load_prototype("locus_leads", "run")
panel_module = load_prototype("locus_leads", "panel")

KEY = "OGS-00001/a"


def _fields(chromosome: list[str], position: list[int], variant_index: list[int]) -> object:
    return run.TierRows(
        chromosome=np.array(chromosome), position=np.array(position),
        alid=np.array([f"{c}:{p}" for c, p in zip(chromosome, position, strict=True)]),
        allele_1=np.array(["C"] * len(position)), allele_2=np.array(["T"] * len(position)),
        abs_z=np.arange(len(position), dtype="float64"),
        variant_index=np.array(variant_index),
    )


def _panel() -> object:
    """Three panel variants, so a row at 100, 200 or 300 matches."""
    return panel_module.PanelVariants(
        position=np.array([100, 200, 300]),
        variant_id=np.array(["1:100:C:T", "1:200:C:T", "1:300:C:T"]),
        ref=np.array(["C", "C", "C"]), alt=np.array(["T", "T", "T"]),
    )


def _two_chromosomes() -> dict[str, object]:
    return {"1": _panel(), "2": _panel()}


def _map(*chromosomes: str) -> object:
    return genetic_map.GeneticMap({c: (np.array([1, 400]), np.array([0.0, 1.0]))
                                   for c in chromosomes})


def test_the_shared_set_is_the_matched_rows_with_finite_cm_and_the_rest_are_counted() -> None:
    tier = _fields(["1", "1", "1", "2", "X"], [100, 200, 400, 100, 100], [7, 8, 9, 10, 11])

    kept = maps.shared_set_rows(tier, {"1": _panel()}, _map("1"))

    # 100 and 200 are on the panel, 400 is not; chr2 is off the panel here and
    # chrX is not on the map. The fixture must reach both a match and a miss.
    assert tier.variant_index.size == 5
    np.testing.assert_array_equal(kept, [True, True, False, False, False])


def test_the_shared_set_spells_out_the_rule_build_shared_set_applies(tmp_path) -> None:
    # The same tier through the sweep's own function and through this one, so
    # the check cannot pass by measuring something else.
    from locus_leads_panel import write_panel  # a real pgen, for this test alone

    write_panel(tmp_path, "1", [(100, "C", "T"), (200, "C", "T"), (300, "C", "T")],
                [["0|1", "0|0"], ["0|1", "0|1"], ["0|0", "0|1"]])
    written = panel_module.Panel(tmp_path)
    tier = _fields(["1", "1", "1", "1"], [100, 200, 300, 400], [7, 8, 9, 10])
    one = _map("1")
    assert written.chromosomes == ["1"] and written.variants("1").position.size == 3

    shared, accounting = run.build_shared_set(tier, written, one)
    kept = maps.shared_set_rows(tier, {"1": written.variants("1")}, one)

    assert accounting["shared_set"] == 3, "the fixture must reach both matched and absent rows"
    np.testing.assert_array_equal(tier.variant_index[kept], shared.rows.variant_index)


def test_a_map_that_loses_one_row_is_refused_and_the_counts_are_named() -> None:
    tier = _fields(["1", "1", "2"], [100, 200, 100], [7, 8, 9])
    reference = maps.shared_set_of(maps.match_tiers({KEY: tier}, _two_chromosomes()),
                                   _map("1", "2"), name="mean")
    assert reference.analyses[KEY]["shared_set"] == 3

    with pytest.raises(maps.SharedSetMismatch, match=f"{KEY}: 3 rows.*2 rows"):
        maps.check_same_shared_set(reference, {"lifted": maps.shared_set_of(
            maps.match_tiers({KEY: tier}, _two_chromosomes()), _map("1"), name="lifted")})


def test_the_digest_separates_shared_sets_that_share_a_count() -> None:
    one = maps.shared_set_of(
        maps.match_tiers({KEY: _fields(["1", "1", "2"], [100, 200, 100], [7, 8, 9])},
                         _two_chromosomes()), _map("1", "2"), name="mean")
    other_row = maps.shared_set_of(
        maps.match_tiers({KEY: _fields(["1", "1", "2"], [100, 200, 100], [7, 8, 10])},
                         _two_chromosomes()), _map("1", "2"), name="moved")

    assert one.analyses[KEY]["shared_set"] == other_row.analyses[KEY]["shared_set"] == 3
    with pytest.raises(maps.SharedSetMismatch, match="sha256"):
        maps.check_same_shared_set(one, {"moved": other_row})


def test_the_same_shared_set_on_another_map_is_accepted() -> None:
    tier = _fields(["1", "1", "2"], [100, 200, 100], [7, 8, 9])
    matched = maps.match_tiers({KEY: tier}, _two_chromosomes())
    both = {name: maps.shared_set_of(matched, _map("1", "2"), name=name)
            for name in ("mean", "median")}

    maps.check_same_shared_set(both["mean"], {"median": both["median"]})

    assert both["mean"].analyses[KEY]["sha256"] == both["median"].analyses[KEY]["sha256"]
    assert both["mean"].digest() == both["median"].digest()
