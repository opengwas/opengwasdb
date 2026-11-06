"""One Analysis scored end to end: a synthetic 5e-4 tier, a tiny pgen panel, and the record
#403's own sweep makes of its 5e-8 rows (prototypes/locus_leads, #409).

Four candidates clear 5e-8: A, B and C carry one signal 100 kb apart, D another 5 Mb out.
Twenty rows below 5e-8, off HapMap3, fill every 10 kb bin between A and C. HapMap3 holds
A, C and D, so on HapMap3 alone the profile between A and C is empty.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pytest
from locus_leads_panel import write_panel
from prototype_modules import load_prototype

blocks, genetic_map, panel, profile_sweep, run, sweep = (
    load_prototype("locus_leads", name)
    for name in ("blocks", "genetic_map", "panel", "profile_sweep", "run", "sweep"))

MAP = genetic_map.LoadedMap(
    genetic_map.GeneticMap({"1": (np.array([1, 10_000_000]), np.array([0.0, 100.0]))}),
    {"assembly": "GRCh38"})
SIGNAL = ["0/0", "1/1", "0/0", "1/1"] * 10
OTHER = ["0/0", "0/0", "1/1", "1/1"] * 10  # r^2 = 0 with SIGNAL
CANDIDATES = {100_000: 20.0, 200_000: 15.0, 300_000: 10.0, 5_000_000: 8.0}  # A, B, C, D
FILLER = [105_000 + 10_000 * k for k in range(20)]
HM3 = frozenset({"1:100000:A:G", "1:300000:A:G", "1:5000000:A:G"})
BLOCKS = "chr\tstart\tend\tancestry\n1\t0\t1000000\tEUR\n1\t1000000\t10000000\tEUR\n"


def _rows(positions: list[int], abs_z: list[float], variant_index: list[int]) -> Any:
    n = len(positions)
    return run.TierRows(
        np.array(["1"] * n), np.array(positions), np.array([f"1:{p}:A:G" for p in positions]),
        np.array(["A"] * n), np.array(["G"] * n), np.array(abs_z), np.array(variant_index))


TIER = _rows([*CANDIDATES, *FILLER], [*CANDIDATES.values(), *[4.0] * len(FILLER)],
             [0, 1, 2, 3, *range(10, 10 + len(FILLER))])
REFERENCE = _rows(list(CANDIDATES), list(CANDIDATES.values()), [0, 1, 2, 3])


def _identity(threshold: float, rows: int) -> dict[str, Any]:
    return {"store": {"reference_assembly": "GRCh38", "store_id": "S"},
            "analysis": {"analysis_id": "a"},
            "query": {"threshold": threshold, "observed": rows}}


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    directory = tmp_path_factory.mktemp("profile_panel")
    write_panel(directory, "1", [(p, "A", "G") for p in CANDIDATES],
                                  [SIGNAL, SIGNAL, SIGNAL, OTHER], assembly="gnomAD_GRCh38")
    (directory / "blocks.tsv").write_text(BLOCKS)
    record = sweep.analyse(REFERENCE, _identity(5e-8, 4), MAP, panel.Panel(directory),
                           work_dir=directory / "403", threads=1, spikes=sweep.EUR_SPIKES,
                           configs=sweep.grid_configs(), hm3=HM3)
    return {"panel": panel.Panel(directory), "committed": json.loads(json.dumps(record)),
            "blocks": blocks.read_blocks(directory / "blocks.tsv", "EUR"), "work": directory}


def _score(inputs: dict[str, Any], committed: dict[str, Any] | None = None,
           reference: Any = REFERENCE, tier: Any = TIER) -> dict[str, Any]:
    record: dict[str, Any] = profile_sweep.score_profile(
        tier, _identity(5e-4, tier.abs_z.size), reference, committed or inputs["committed"],
        MAP, inputs["panel"], inputs["blocks"], HM3, store_id="S",
        work_dir=inputs["work"] / "409", threads=1)
    return record


def _occupied(positions: np.ndarray, start: int, end: int) -> list[bool]:
    held = {int(p) // 10_000 for p in positions}
    return [b in held for b in range(start // 10_000 + 1, end // 10_000)]


def test_the_fixture_puts_an_occupied_profile_between_a_and_c_that_hapmap3_empties() -> None:
    on_hm3 = np.isin(TIER.alid, list(HM3))
    below = TIER.abs_z < 5.45
    assert list(TIER.position[on_hm3]) == [100_000, 300_000, 5_000_000]
    assert below.sum() == len(FILLER) and not (below & on_hm3).any()
    assert all(_occupied(TIER.position, 100_000, 300_000))
    assert not any(_occupied(TIER.position[on_hm3], 100_000, 300_000))


def test_on_hapmap3_the_profile_is_cut_to_hapmap3_rows_too(inputs: dict[str, Any]) -> None:
    record = _score(inputs)

    density = {r["label"]: r["density"] for r in record["rules"]}
    # Full density: A, B and C are one locus, D another. On HapMap3 the empty stretch
    # between A and C splits them.
    for label in ("B 10 kb, fixed T 0", "B 50 kb, fixed T 0", "B 10 kb, f 0.25"):
        assert density[label] == {"leads_full": 2, "leads_hm3": 3}, label
    assert density["LD blocks (EUR)"] == {"leads_full": 2, "leads_hm3": 2}
    assert record["index_variants"] == 2
    assert record["density"] == {"rows_full": 4, "rows_hm3": 3}


def test_the_shared_set_is_published_as_matching_403s_summaries_with_its_digest(
        inputs: dict[str, Any]) -> None:
    record = _score(inputs)

    assert record["shared_set"] == {
        "rows": 4, "sha256": profile_sweep.shared_set_identity("S", "a", np.arange(4))["sha256"],
        "matches_403_summaries": True}
    assert record["baselines_equal_403"] == [
        {"d_cm": "bp_only", "w_bp": w} for w in sweep.W_GRID] + [{"d_cm": 0.5,
                                                                  "w_bp": 1_000_000}]


@pytest.mark.parametrize("change", ["bp-only 10 Mb leads", "D 0.5 cM HapMap3 leads"])
def test_a_distance_rule_that_does_not_reproduce_403s_record_is_refused(
        inputs: dict[str, Any], change: str) -> None:
    committed = copy.deepcopy(inputs["committed"])
    configs = {(c["d_cm"], c["w_bp"]): c for c in committed["configs"]}
    if change == "bp-only 10 Mb leads":
        configs[("bp_only", 10_000_000)]["by_r2"]["0.01"]["locus_leads"] += 1
    else:
        configs[(0.5, 1_000_000)]["density"]["leads_hm3"] += 1

    with pytest.raises(profile_sweep.BaselineMismatch, match="differs from #403's record"):
        _score(inputs, committed)


@pytest.mark.parametrize("change", ["accounting", "HapMap3 rows", "one reference row",
                                    "an extra reference row off the panel",
                                    "an extra candidate off the panel"])
def test_a_shared_set_that_does_not_reproduce_403s_summaries_is_refused(
        inputs: dict[str, Any], change: str) -> None:
    committed, reference, tier = copy.deepcopy(inputs["committed"]), REFERENCE, TIER
    if change == "accounting":
        committed["rows"]["partition"]["absent_at_position"] += 1
    elif change == "HapMap3 rows":
        committed["density"]["rows_hm3"] -= 1
    elif change == "one reference row":
        reference = _rows(list(CANDIDATES), list(CANDIDATES.values()), [0, 1, 2, 4])
    elif change == "an extra reference row off the panel":
        # The shared sets still agree; only the 5e-8 read's accounting shows the extra row.
        reference = _rows([*CANDIDATES, 400_000], [*CANDIDATES.values(), 6.0], [0, 1, 2, 3, 9])
    else:
        # Likewise, but the extra row is a candidate of the 5e-4 read.
        tier = _rows([*TIER.position, 400_000], [*TIER.abs_z, 6.0], [*TIER.variant_index, 9])

    with pytest.raises(profile_sweep.SharedSetMismatch):
        _score(inputs, committed, reference, tier)
