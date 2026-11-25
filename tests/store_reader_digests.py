"""The six fixture Store Releases every Store Reader question is asked of (#464, #474).

`build_releases` builds a Dense, a Ragged and a Hybrid release, each Observed-Only and
Reference-Completed: the `digest_releases` session fixture (`conftest.py`), and the base
of six of the Store Reader contract bundle's releases (`tests/store_reader_contract/`).

What they answer was pinned in `tests/golden/store_reader_digests.json` from #464 until
#474, written by the code before the Store Reader move (`5d600a1d`) and regenerated from
`git archive` at `826b36bb` (`beta`, `pval`) and `1ca2d349` (`source_alid`). The
contract bundle replaced those pins: its expected answers are the same questions
(`opengwasdb.store.query_digests`) asked of these releases with every planted state, and
the generator must reproduce them byte for byte, so a change to a builder or to an
answer shows there, once.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest


class _Factory:
    """The one method `RaggedResidualScenario` asks of pytest's factory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def mktemp(self, name: str) -> Path:
        path = self.root / name
        path.mkdir(parents=True)
        return path


def build_releases(root: Path) -> dict[str, Path]:
    """The six fixture releases, by name: each layout Observed-Only and Reference-Completed."""
    import residual_stores
    from test_hybrid_completion import _residual_hybrid_crossover_source
    from test_ragged_residual_completion import RaggedResidualScenario

    def fresh(name: str) -> Path:
        path = root / name
        path.mkdir(parents=True)
        return path

    ragged = RaggedResidualScenario(cast(pytest.TempPathFactory, _Factory(root / "ragged")))
    hybrid_observed, *_ = _residual_hybrid_crossover_source(fresh("hybrid_observed"))
    dense_completed, *_ = residual_stores.dense_completed(fresh("dense_completed"))
    return {
        "dense_observed": residual_stores.dense_observed_missing(fresh("dense_observed")),
        "dense_completed": dense_completed,
        "ragged_observed": ragged.source,
        "ragged_completed": ragged.completed,
        "hybrid_observed": hybrid_observed,
        "hybrid_completed": residual_stores.hybrid_completed(fresh("hybrid_completed")),
    }
