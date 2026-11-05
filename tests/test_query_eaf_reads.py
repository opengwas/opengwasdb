"""#253 at the query facade: what the association readers' own suite cannot see.

The read-once and alignment properties of a shared EAF read -- each of `eaf`,
`eaf_baseline` and `imputed` read once per region, SE decoded from that one read,
the panel's frequency on imputed cells only, `observed_only` cutting every column
together -- are properties of each reader, and live in the contract suite
(`test_association_readers.py`), region by region and adapter by adapter.

What stays here is composition: a residual Hybrid's one-Analysis answer joins two
components, each decoding its own EAF, and Reference Completion still reads a
whole Analysis through `RaggedCSRReader.get_analysis`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from store_reads import chunk_reads, duplicate_chunk_keys
from test_hybrid_shared_se_plan_e2e import (
    N_OFF_PANEL,
    N_PANEL,
    OFF_BASE,
    _build_hybrid,
    _se_value,
    overflow_alid,
    panel_alid,
)
from test_ragged_residual_completion import RaggedResidualScenario

from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.query import query_store

#: The arrays whose duplicate reads #253 removes.
_EAF_ARRAYS = ("eaf", "eaf_baseline", "imputed")


@pytest.fixture(scope="module")
def ragged_completed(
    residual_ragged_scenario: RaggedResidualScenario,
) -> RaggedResidualScenario:
    """`shared_releases.residual_ragged`, built once per session and read here."""
    return residual_ragged_scenario


@pytest.fixture(scope="module")
def hybrid_residual(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_hybrid(tmp_path_factory.mktemp("eaf_reads_hybrid"), "reads", [dict(), dict()])


def test_hybrid_fixture_is_residual_in_both_components(hybrid_residual: Path) -> None:
    """Both Hybrid components must residual-code se, or the trap is vacuous."""
    encoding = StoreManifest.load(hybrid_residual)
    assert encoding.encoding.se.is_residual, "fixture must residual-code se"
    with query_store(hybrid_residual) as query:
        dense = query.lookup([panel_alid(50)], ["trait_0"])
        overflow = query.lookup([overflow_alid(50)], ["trait_0"])
    assert len(dense["z"]) == 1 and len(overflow["z"]) == 1


def test_ragged_get_analysis_reads_each_eaf_array_once(
    ragged_completed: RaggedResidualScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`RaggedCSRReader.get_analysis` reads one slice's EAF once, not twice."""
    reader = RaggedCSRReader(ragged_completed.completed)
    with chunk_reads(monkeypatch, _EAF_ARRAYS) as reads:
        row = reader.get_analysis(0)
    assert len(row.z) > 0, "the Analysis must return rows for this to mean anything"
    duplicates = duplicate_chunk_keys(reads)
    assert duplicates == {}, f"get_analysis re-read EAF chunks: {duplicates}"
    assert reads["eaf"] and reads["eaf_baseline"] and reads["imputed"], (
        "get_analysis must read all three arrays for this to mean anything"
    )


def test_hybrid_query_answers_match_the_model(hybrid_residual: Path) -> None:
    """A residual Hybrid's one-Analysis read decodes each component's own EAF."""
    frequencies = np.linspace(0.05, 0.95, N_PANEL, dtype=np.float64)
    off_frequencies = np.linspace(0.10, 0.90, N_OFF_PANEL, dtype=np.float64)
    with query_store(hybrid_residual) as query:
        variants = query.variants_table()
        result = query.analysis("trait_0")
    assert len(result["z"]) == N_PANEL + N_OFF_PANEL
    assert np.all(np.isfinite(result["se"])) and np.all(result["se"] > 0)
    for vi, se, eaf in zip(result["variant_index"], result["se"], result["eaf"], strict=True):
        position = int(variants[int(vi)]["position"])
        # The store orients EAF to the stored effect allele (REF=A here), while
        # the fixture's AF column is the ALT frequency, so eaf is 1 - AF.
        if position < OFF_BASE:
            row = position // 1000 - 1
            expected_se = _se_value(0, float(frequencies[row]), row)
            np.testing.assert_allclose(eaf, 1.0 - frequencies[row], rtol=0.02)
        else:
            row = (position - OFF_BASE) // 1000
            expected_se = _se_value(0, float(off_frequencies[row]), row)
            np.testing.assert_allclose(eaf, 1.0 - off_frequencies[row], rtol=0.02)
        np.testing.assert_allclose(se, expected_se, rtol=0.01)
