"""The query facade answers exactly what it answered before the Store Reader move (#464).

`tests/golden/store_reader_digests.json` is what the code at `5d600a1d` -- before
`opengwasdb.store_reader` existed -- answered on six fixture releases: Dense,
Ragged and Hybrid, each Observed-Only and Reference-Completed
(`store_reader_digests`). The pins were written by the old code, never by this
one; a difference names the release, the question and the array.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from store_reader_digests import GOLDEN, _shapes, build_releases, release_digests

from opengwasdb.query import query_store
from opengwasdb.store_reader.stats import association_statistics

PINS: dict[str, Any] = json.loads(GOLDEN.read_text(encoding="utf-8"))
RELEASES = sorted(name for name in PINS if not name.startswith("__"))

#: The relative tolerance `log10_p` is held to: the Store Reader computes it with
#: `math.erfc` where the code before computed scipy's `norm.logsf`, and the two
#: agree to this, not bit for bit (`tests/test_store_reader_stats.py`).
LOG10_P_RTOL = 1e-12


@pytest.fixture(scope="module")
def releases(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return build_releases(tmp_path_factory.mktemp("store-reader-digests"))


def test_the_pins_cover_every_layout_and_completion_state():
    assert RELEASES == [
        f"{layout}_{state}"
        for layout in ("dense", "hybrid", "ragged")
        for state in ("completed", "observed")
    ]
    for name in RELEASES:
        shapes = PINS[name]["shapes"]
        empty = shapes["phewas/1:1:A:C/all"]
        answered = [shape for shape, digests in shapes.items() if digests != empty]
        # A pin of nothing but empty answers would pass whatever the reader did.
        assert len(answered) > 20, (name, len(answered))
        assert sum(len(values) for values in PINS[name]["log10_p"].values()) > 500, name


@pytest.mark.parametrize("name", RELEASES)
def test_every_answer_matches_the_code_before_the_move(releases: dict[str, Path], name: str):
    found = release_digests(releases[name])
    # The pins predate beta and pval (ADR 0104): without them, nothing moved.
    for kind in ("shapes", "resolved"):
        for shape, digests in found[kind].items():
            # A resolved answer of no rows has no columns to digest.
            assert not digests or all(key in digests for key in DERIVED), (kind, shape)
            found[kind][shape] = {k: v for k, v in digests.items() if k not in DERIVED}
    pinned = PINS[name]
    assert found["adapter"] == pinned["adapter"]
    assert found["variants_table"] == pinned["variants_table"]
    assert found["analyses_table"] == pinned["analyses_table"]
    assert sorted(found["shapes"]) == sorted(pinned["shapes"])
    differing = {
        shape: sorted(
            array
            for array in set(digests) | set(pinned["shapes"][shape])
            if digests.get(array) != pinned["shapes"][shape].get(array)
        )
        for shape, digests in found["shapes"].items()
        if digests != pinned["shapes"][shape]
    }
    assert differing == {}
    assert found["resolved"] == pinned["resolved"]
    for shape, values in pinned["log10_p"].items():
        assert len(found["log10_p"][shape]) == len(values), shape
        for got, want in zip(found["log10_p"][shape], values, strict=True):
            # A missing association's log10_p is NaN on both sides, and NaN is
            # never `isclose` to itself.
            same = math.isnan(got) and math.isnan(want)
            assert same or math.isclose(got, want, rel_tol=LOG10_P_RTOL, abs_tol=0.0), (
                shape,
                got,
                want,
            )


#: The arrays the Store Reader derives (ADR 0104), on every record and resolved row.
DERIVED = ("beta", "pval")


@pytest.mark.parametrize("name", RELEASES)
def test_every_answer_carries_beta_and_pval_from_the_one_function(
    releases: dict[str, Path], name: str
):
    counts = {"rows": 0, "missing": 0, "resolved": 0}
    with query_store(releases[name]) as q:
        for shape, ask in _shapes(q).items():
            result = ask()
            derived = association_statistics(result["z"], result["se"])
            for key in DERIVED:
                assert key in result, (shape, key)
                assert result[key].dtype == np.float64, (shape, key)
                assert np.array_equal(result[key], getattr(derived, key), equal_nan=True), (
                    shape,
                    key,
                )
            counts["rows"] += len(result["z"])
            counts["missing"] += int(np.count_nonzero(np.isnan(result["pval"])))
            if shape.startswith("lookup/"):
                rows = list(q.resolve(result))
                for i, row in enumerate(rows):
                    for key in (*DERIVED, "log10_p"):
                        want = float(getattr(derived, key)[i])
                        assert row[key] == want or (math.isnan(want) and math.isnan(row[key]))
                counts["resolved"] += len(rows)
    # A fixture of empty answers would pass whatever the reader did.
    assert counts["rows"] > 200 and counts["resolved"] > 5, counts
    if name == "ragged_completed":
        # The one fixture whose answers hold missing rows (12, where imputation
        # failed): their NaN must reach beta and pval.
        assert counts["missing"] > 0, counts
