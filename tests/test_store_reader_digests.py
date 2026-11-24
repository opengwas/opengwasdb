"""The query facade answers exactly what it answered before the Store Reader move (#464).

`tests/golden/store_reader_digests.json` is what the facade answers on six fixture
releases: Dense, Ragged and Hybrid, each Observed-Only and Reference-Completed
(`store_reader_digests`). The pins were first written by the code at `5d600a1d`,
before `opengwasdb.store_reader` existed. They were regenerated at `826b36bb`, when
every record gained `beta` and `pval` (ADR 0104). With those two dropped, every
digest equalled the `5d600a1d` pins, and `log10_p` was within 9.7e-16. A
difference names the release, the question and the array.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from store_reader_digests import GOLDEN, release_digests

from opengwasdb.query import query_store
from opengwasdb.store.query_digests import expects_refusal, query_shapes
from opengwasdb.store_reader.stats import association_statistics
from opengwasdb.store_reader.top_hits import UnstoredThresholdError

PINS: dict[str, Any] = json.loads(GOLDEN.read_text(encoding="utf-8"))
RELEASES = sorted(name for name in PINS if not name.startswith("__"))

#: The relative tolerance `log10_p` is held to: the Store Reader computes it with
#: `math.erfc` where the code before computed scipy's `norm.logsf`, and the two
#: agree to this, not bit for bit (`tests/test_store_reader_stats.py`).
LOG10_P_RTOL = 1e-12


@pytest.fixture(scope="module")
def releases(digest_releases: dict[str, Path]) -> dict[str, Path]:
    return digest_releases


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


def test_the_variant_reads_are_the_one_section_the_pins_gained_with_484():
    """The pins hold the sections of the code before the move, and `variant_reads` (#484)
    from the facade's variant-only reads; nothing else was added to them."""
    for name in RELEASES:
        assert set(PINS[name]) == {
            "adapter", "variants_table", "analyses_table", "shapes", "resolved", "log10_p",
            "variant_reads",
        }, name
        assert len(PINS[name]["variant_reads"]) > 5, name


@pytest.mark.parametrize("name", RELEASES)
def test_every_answer_matches_the_code_before_the_move(releases: dict[str, Path], name: str):
    found = release_digests(releases[name])
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
    assert found["variant_reads"] == pinned["variant_reads"]  # #484
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
        for shape, ask in query_shapes(q).items():
            try:
                result = ask()
            except UnstoredThresholdError:
                # only the shapes a Dense or Hybrid release is expected to refuse (#483)
                assert expects_refusal(q, shape), shape
                continue
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
