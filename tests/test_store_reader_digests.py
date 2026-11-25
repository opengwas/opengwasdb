"""Every answer carries `beta` and `pval` from the one function that derives them
(ADR 0104), on the six `store_reader_digests` fixture releases.

What those releases answer was pinned in `tests/golden/store_reader_digests.json` until
#474; the Store Reader contract bundle (`tests/store_reader_contract/`) holds it now
(`store_reader_digests`).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.query import query_store
from opengwasdb.store.query_digests import expects_refusal, query_shapes
from opengwasdb.store_reader.stats import association_statistics
from opengwasdb.store_reader.top_hits import UnstoredThresholdError

RELEASES = [
    f"{layout}_{state}"
    for layout in ("dense", "hybrid", "ragged")
    for state in ("completed", "observed")
]


@pytest.fixture(scope="module")
def releases(digest_releases: dict[str, Path]) -> dict[str, Path]:
    return digest_releases


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
