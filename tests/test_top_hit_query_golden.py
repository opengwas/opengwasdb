"""`top_hits()` answers the same rows, in the same order, as before the Top-Hit
Index module (#53).

`tests/golden/top_hit_queries.json` was written by the code at `eb5845b`; see
`top_hit_fixtures.py`. Asserting the pins are meaningful first: a golden file of
empty answers would pass against any reader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from top_hit_fixtures import STORES, THRESHOLDS, query_matrix

GOLDEN = json.loads((Path(__file__).parent / "golden" / "top_hit_queries.json").read_text())


@pytest.mark.parametrize("layout", sorted(STORES))
def test_golden_pins_are_not_vacuous(layout):
    pins = GOLDEN[layout]
    for threshold in (5e-8, 5e-6, 5e-4):
        assert pins[f"{threshold:g}|None|None|False"]["z"], (layout, threshold)
    # a limit and an Analysis each really cut the answer down
    assert len(pins["0.0005|None|1|False"]["z"]) == 1
    assert len(pins["0.0005|None|None|False"]["z"]) > 1
    assert pins["0.0005|no-such-analysis|None|False"]["z"] == []
    assert set(THRESHOLDS) == {1e-9, 5e-8, 5e-6, 5e-4, 1e-3}


@pytest.mark.parametrize("layout", sorted(STORES))
def test_top_hits_match_the_pins_from_before_the_module(layout, tmp_path):
    build, analysis_ids = STORES[layout]
    actual = query_matrix(build(tmp_path), analysis_ids)
    assert actual.keys() == GOLDEN[layout].keys()
    for key, expected in GOLDEN[layout].items():
        assert actual[key] == expected, (layout, key)
