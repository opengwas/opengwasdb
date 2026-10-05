"""#253: one EAF read per query, shared between SE decoding and the eaf column.

A query on a release with residual SE (ADR 0037 §3) reads `eaf`, its
`eaf_baseline` and, on a Reference-Completed release, the `imputed` mask twice:
once inside SE decoding, because the residual is predicted from the frequency,
and once for the result's `eaf` column. The Ragged and Hybrid Analysis-side
reads do the same. This module pins the two properties the change must hold,
each on a fixture that genuinely has residual SE:

* **read once.** No chunk of those three arrays is fetched twice in one query,
  counted at zarr's store -- the same instrument `test_query_metadata_reads`
  uses for metadata. On the unfixed code this fails for every in-scope shape;
  after the change it passes.
* **no answer changes.** Every returned cell's `z`, `se`, `eaf` and
  `association_status` still agrees with the values the fixture's source rows
  and LD panel put in, with and without `observed_only`.

The identity assertions are against the fixture's own inputs, never against the
query code, so a deliberately wrong shared read fails them: the shared array
indexed with the wrong mask (a neighbouring variant's frequency), the result
column decoded without the panel substitution (NaN where the release holds a
frequency), the substitution applied without its mask (the panel's frequency on
observed cells), or one Hybrid component's read used for the other's. The wrong
variants live in ticket #253's report as diffs, not here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from residual_fixtures import residual_eligible_records
from store_reads import chunk_reads, duplicate_chunk_keys, old_index_without_fields
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
from test_se_residual_encoding import _residual_source_and_panel

from opengwasdb.encoding.codec import EAF_ABSENT
from opengwasdb.layouts.dense.build import build_dense_observed_store
from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.query import query_store

#: The arrays whose duplicate reads #253 removes.
_EAF_ARRAYS = ("eaf", "eaf_baseline", "imputed")

#: The panel frequency shift the Dense trap fixture applies, so an observed
#: cell's own frequency differs from the panel's and an unmasked substitution is
#: visible. Well inside the residual coding's range and the panel's [0, 1].
_PANEL_SHIFT = 0.15


# ── fixtures ────────────────────────────────────────────────────────────────


def _dense_completed_fixture(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """A Reference-Completed residual Dense release whose panel differs.

    Returns the store, the observed frequency of each variant row, and the
    panel frequency now held in `eaf_reference`. The panel is overwritten so it
    is *not* the observed frequency: otherwise the fixture cannot tell a masked
    substitution from an unmasked one, because on the source data they are the
    same number and trap 3 would pass unnoticed (ADR 0037 §4's 3000x error is
    exactly a panel value that differs from the cohort's).
    """
    source, panel_path, _expected = _residual_source_and_panel(tmp_path)
    completed = tmp_path / "comp.opengwasdb"
    complete_dense_store(
        source, completed, panel_path, ancestry="EUR", min_cor=0.0, release_id="comp"
    )
    root = zarr.open_group(str(completed / "data.zarr"), mode="r+", zarr_format=2)
    n_variants = int(root["eaf_reference"].shape[0])
    observed = np.linspace(0.05, 0.95, n_variants, dtype=np.float32)
    panel = np.clip(observed + _PANEL_SHIFT, 0.02, 0.98).astype(np.float32)
    root["eaf_reference"][:] = panel
    return completed, observed, panel


@pytest.fixture(scope="module")
def dense_completed(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, np.ndarray, np.ndarray]:
    return _dense_completed_fixture(tmp_path_factory.mktemp("eaf_reads_dense"))


@pytest.fixture(scope="module")
def ragged_completed(tmp_path_factory: pytest.TempPathFactory) -> RaggedResidualScenario:
    return RaggedResidualScenario(tmp_path_factory)


@pytest.fixture(scope="module")
def hybrid_residual(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_hybrid(tmp_path_factory.mktemp("eaf_reads_hybrid"), "reads", [dict(), dict()])


#: The variant `dense_observed_missing` leaves one Analysis unobserved at, in
#: the middle of the axis so that a contiguous (mask-free) slice of the shared
#: array cannot coincide with the finite mask.
_MISSING_VARIANT = 300


@pytest.fixture(scope="module")
def dense_observed_missing(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A residual Dense observed release with one genuinely missing cell."""
    tmp = tmp_path_factory.mktemp("eaf_reads_missing")
    records, _expected = residual_eligible_records(600)
    kept = [
        r
        for r in records
        if not (r.analysis_id == "b" and r.variant.position == _MISSING_VARIANT)
    ]
    assert len(kept) == len(records) - 1, "the fixture must drop exactly one cell"
    store = tmp / "obs.opengwasdb"
    build_dense_observed_store(
        kept,
        store,
        store_id="s",
        release_id="obs",
        reference_assembly="GRCh38",
        chunk_shape=(100, 2),
    )
    return store


# ── fixture meaningfulness ──────────────────────────────────────────────────


def test_dense_fixture_has_imputed_residual_cells_and_a_differing_panel(
    dense_completed: tuple[Path, np.ndarray, np.ndarray],
) -> None:
    """Assert the fixture is meaningful before anything is asserted about it.

    The Dense trap tests only mean something if the release is residual-coded,
    imputes cells that carry a standard error, stores the absent code where a
    substitution will supply the panel frequency, and has a panel frequency
    that differs from the observed one on variants the source observed.
    """
    store, observed, panel = dense_completed
    manifest = StoreManifest.load(store)
    assert manifest.encoding.se.is_residual, "fixture must residual-code se"
    root = zarr.open_group(str(store / "data.zarr"), mode="r")
    imputed = np.asarray(root["imputed"][:], dtype=bool)
    assert imputed.sum() > 0, "fixture must have imputed cells"
    se = np.asarray(root["se"][:])
    assert np.all(se[imputed] != 0), "imputed cells must carry se codes"
    assert np.all(np.isfinite(panel))
    # The raw eaf entry of an imputed cell is the absent code: without the
    # panel substitution it decodes to NaN, which is trap 2's silent failure.
    raw = np.asarray(root["eaf"][:])
    assert np.all(raw[imputed] == EAF_ABSENT)
    # And the panel genuinely differs on a variant the source observed, so an
    # unmasked substitution (trap 3) is a different number, not the same one.
    observed_variant = int(np.argmax(imputed[:, 1] == 0))
    assert not np.isclose(panel[observed_variant], observed[observed_variant])


def test_ragged_fixture_is_residual_and_completed(ragged_completed: RaggedResidualScenario) -> None:
    """The Ragged fixture must residual-code se and actually impute cells."""
    encoding = StoreManifest.load(ragged_completed.completed).encoding
    assert encoding.se.is_residual, "fixture must residual-code se"
    assert encoding.eaf.reference, "completed fixture must carry the panel's eaf"
    assert ragged_completed.result.n_imputed > 0, "fixture must impute for this to mean anything"


def test_hybrid_fixture_is_residual_in_both_components(hybrid_residual: Path) -> None:
    """Both Hybrid components must residual-code se, or the trap is vacuous."""
    encoding = StoreManifest.load(hybrid_residual)
    assert encoding.encoding.se.is_residual, "fixture must residual-code se"
    with query_store(hybrid_residual) as query:
        dense = query.lookup([panel_alid(50)], ["trait_0"])
        overflow = query.lookup([overflow_alid(50)], ["trait_0"])
    assert len(dense["z"]) == 1 and len(overflow["z"]) == 1


# ── read once ───────────────────────────────────────────────────────────────


_DENSE_SHAPES = {
    "analysis": lambda q: q.analysis("b"),
    "phewas": lambda q: q.phewas("1:197000:A:G"),
    "range_phewas": lambda q: q.range_phewas("1", 0, 10_000_000),
    "lookup": lambda q: q.lookup(["1:1000:A:G", "1:197000:A:G"], ["a", "b"]),
}


@pytest.mark.parametrize("shape", sorted(_DENSE_SHAPES))
def test_dense_query_reads_each_eaf_array_once(
    shape: str,
    dense_completed: tuple[Path, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every in-scope Dense shape reads each of the three arrays once."""
    store, _observed, _panel = dense_completed
    with query_store(store) as query, chunk_reads(monkeypatch, _EAF_ARRAYS) as reads:
        _DENSE_SHAPES[shape](query)
    duplicates = duplicate_chunk_keys(reads)
    assert duplicates == {}, f"{shape} re-read EAF chunks: {duplicates}"
    assert reads["eaf"] and reads["eaf_baseline"] and reads["imputed"], (
        f"{shape} must read all three arrays for this to mean anything"
    )


def test_dense_old_index_top_hits_reads_each_eaf_array_once(
    dense_completed: tuple[Path, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The older-index top-hit fallback reads the plane once, not twice."""
    store, _observed, _panel = dense_completed
    with (
        query_store(store) as query,
        old_index_without_fields(monkeypatch, _EAF_ARRAYS),
        chunk_reads(monkeypatch, _EAF_ARRAYS) as reads,
    ):
        query.top_hits(threshold=5e-8, analysis_id="a")
    duplicates = duplicate_chunk_keys(reads)
    assert duplicates == {}, f"older-index top_hits re-read EAF chunks: {duplicates}"
    assert reads["eaf"] and reads["eaf_baseline"] and reads["imputed"]


def test_ragged_analysis_and_lookup_read_each_eaf_array_once(
    ragged_completed: RaggedResidualScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Analysis-side Ragged reads share one decoded EAF per query."""
    # A lookup identifier that resolves on this store's own axis; the Ragged
    # fixture's variants are not at the Dense fixture's positions.
    with query_store(ragged_completed.completed) as query:
        identifier = str(next(iter(query.variants_table().values()))["alid"])
        shapes = {
            "analysis": lambda q: q.analysis("b"),
            # One Analysis: `lookup` calls `analysis` per requested Analysis, so
            # a request for two Analyses of one CSR legitimately reads the same
            # chunk twice -- once as each Analysis's own rows. The duplicate
            # this test is about is the second read *of one Analysis*, which
            # `analysis` would make on its own.
            "lookup": lambda q: q.lookup([identifier], ["a"]),
        }
        for name, call in shapes.items():
            with chunk_reads(monkeypatch, _EAF_ARRAYS) as reads:
                call(query)
            duplicates = duplicate_chunk_keys(reads)
            assert duplicates == {}, f"Ragged {name} re-read EAF chunks: {duplicates}"
            # The fixture's eaf plane is `float32` (a sparse store writes no
            # baseline), so `eaf` itself is the array this test guarantees is
            # read; the imputed mask is read because the plan carries reference
            # EAF, and reading it once is what the duplicate check pins.
            assert reads["eaf"], f"Ragged {name} must read eaf for this to mean anything"
            assert reads["imputed"], f"Ragged {name} must read the imputed mask"


def test_hybrid_analysis_reads_each_eaf_array_once(
    hybrid_residual: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Hybrid one-Analysis read shares each component's EAF read."""
    with query_store(hybrid_residual) as query, chunk_reads(monkeypatch, _EAF_ARRAYS) as reads:
        query.analysis("trait_0")
    duplicates = duplicate_chunk_keys(reads)
    assert duplicates == {}, f"Hybrid analysis re-read EAF chunks: {duplicates}"
    assert reads["eaf"], "Hybrid analysis must read eaf for this to mean anything"


# ── no answer changes (the trap tests) ──────────────────────────────────────


def _status_split(result: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    status = np.asarray(result["association_status"])
    return status == "observed", status == "imputed"


def _check_dense_cells(
    result: dict[str, np.ndarray],
    observed: np.ndarray,
    panel: np.ndarray,
    z_at,
) -> None:
    """Every returned Dense cell decodes to the source or the panel value."""
    rows = np.asarray(result["variant_index"], dtype=np.int64)
    assert result["se"].dtype == np.dtype("float32")
    assert result["eaf"].dtype == np.dtype("float32")
    is_observed, is_imputed = _status_split(result)
    assert is_observed.any(), "every Dense shape here must return observed cells"
    assert np.all(is_observed | is_imputed), "a returned cell is neither observed nor imputed"
    # An observed cell's frequency is the cohort's own; an imputed one's is the
    # panel's (ADR 0037 §4). Trap 3 replaces the first with the second; trap 2
    # leaves the second NaN.
    np.testing.assert_allclose(result["eaf"][is_observed], observed[rows[is_observed]], rtol=0.02)
    if is_imputed.any():
        assert np.all(np.isfinite(result["eaf"][is_imputed]))
        np.testing.assert_allclose(result["eaf"][is_imputed], panel[rows[is_imputed]], rtol=0.02)
    assert np.all(np.isfinite(result["se"])) and np.all(result["se"] > 0)
    # z is only the source's on observed cells; completion imputes its own z.
    np.testing.assert_allclose(
        result["z"][is_observed], [z_at(int(r)) for r in rows[is_observed]], rtol=1e-3
    )


def test_dense_query_answers_match_the_source_and_panel(
    dense_completed: tuple[Path, np.ndarray, np.ndarray],
) -> None:
    """analysis/phewas/range/lookup agree cell for cell with the fixture inputs."""
    store, observed, panel = dense_completed

    def z_at(row: int) -> float:
        return 8.0 if row % 50 == 0 else 1.0

    with query_store(store) as query:
        shapes = {
            "analysis_a": lambda: query.analysis("a"),
            "analysis_b": lambda: query.analysis("b"),
            "phewas_observed": lambda: query.phewas("1:1000:A:G"),
            "phewas_imputed": lambda: query.phewas("1:197000:A:G"),
            "range": lambda: query.range_phewas("1", 0, 10_000_000),
            "lookup": lambda: query.lookup(["1:1000:A:G", "1:197000:A:G"], ["a", "b"]),
        }
        seen_imputed = False
        for name, call in shapes.items():
            result = call()
            assert len(result["z"]) > 0, f"{name} returned nothing"
            _check_dense_cells(result, observed, panel, z_at)
            seen_imputed |= bool(_status_split(result)[1].any())
        assert seen_imputed, "the shapes above must exercise imputed cells"
        # observed_only keeps the same cells, minus the imputed ones.
        full = query.range_phewas("1", 0, 10_000_000)
        filtered = query.range_phewas("1", 0, 10_000_000, observed_only=True)
        assert len(filtered["z"]) == int(_status_split(full)[0].sum())
        assert "imputed" not in set(filtered["association_status"])


def test_dense_old_index_top_hits_answers_match_the_source(
    dense_completed: tuple[Path, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The older-index top-hit fallback decodes the same physical values."""
    store, observed, panel = dense_completed

    def z_at(row: int) -> float:
        return 8.0 if row % 50 == 0 else 1.0

    with query_store(store) as query, old_index_without_fields(monkeypatch, _EAF_ARRAYS):
        result = query.top_hits(threshold=5e-8, analysis_id="a")
    assert len(result["z"]) > 0
    _check_dense_cells(result, observed, panel, z_at)


def test_dense_observed_only_filters_the_shared_eaf_column(
    dense_completed: tuple[Path, np.ndarray, np.ndarray],
) -> None:
    """`observed_only` must cut the shared eaf column in lockstep with z/se.

    The shared region is read once and indexed by the query's masks; if the
    result column is cut with a different mask from SE decoding, an observed
    cell gets a neighbouring or imputed cell's frequency. Every array is
    compared cell for cell against the unfiltered result restricted to its
    observed rows.
    """
    store, _observed, _panel = dense_completed
    with query_store(store) as query:
        pairs = {
            "analysis_b": (
                lambda: query.analysis("b"),
                lambda: query.analysis("b", observed_only=True),
            ),
            "phewas_imputed": (
                lambda: query.phewas("1:197000:A:G"),
                lambda: query.phewas("1:197000:A:G", observed_only=True),
            ),
            "range": (
                lambda: query.range_phewas("1", 0, 10_000_000),
                lambda: query.range_phewas("1", 0, 10_000_000, observed_only=True),
            ),
            "lookup": (
                lambda: query.lookup(["1:1000:A:G", "1:197000:A:G"], ["a", "b"]),
                lambda: query.lookup(
                    ["1:1000:A:G", "1:197000:A:G"], ["a", "b"], observed_only=True
                ),
            ),
        }
        for name, (full_call, filtered_call) in pairs.items():
            full = full_call()
            filtered = filtered_call()
            keep = _status_split(full)[0]
            assert keep.any() and not keep.all(), f"{name} must have both kinds of cell"
            assert len(filtered["z"]) == int(keep.sum()), name
            for key in ("variant_index", "analysis_index", "z", "se", "eaf", "association_status"):
                np.testing.assert_array_equal(
                    np.asarray(full[key])[keep], filtered[key], err_msg=f"{name}:{key}"
                )


def test_dense_finite_mask_indexes_the_shared_eaf_column(
    dense_observed_missing: Path,
) -> None:
    """A missing Dense cell drops from the shared read, not shifts it.

    The shared frequency region is the whole column, a superset of the finite
    cells. Indexing it with a contiguous slice instead of the finite mask would
    give every cell after the missing one its neighbour's frequency -- wrong,
    plausible, and silent, which is #253's alignment trap on the finite mask.
    """
    frequencies = np.linspace(0.05, 0.95, 600, dtype=np.float32)
    with query_store(dense_observed_missing) as query:
        result = query.analysis("b")
    returned = set(int(v) for v in result["variant_index"])
    assert _MISSING_VARIANT - 1 not in returned, "the missing cell must stay absent"
    assert len(returned) == 599
    for vi, eaf in zip(result["variant_index"], result["eaf"], strict=True):
        np.testing.assert_allclose(eaf, frequencies[int(vi)], rtol=0.02)


def test_ragged_query_answers_match_the_source_and_panel(
    ragged_completed: RaggedResidualScenario,
) -> None:
    """Analysis-side Ragged reads decode the source and the panel, unmixed."""
    expected = ragged_completed.expected_observed()
    panel_eaf = ragged_completed.panel_eaf
    with query_store(ragged_completed.completed) as query:
        analyses = query.analyses_table()
        analysis_by_index = {i: row["analysis_id"] for i, row in analyses.items()}
        variant_by_index = query.variants_table()
        first = query.analysis("a")
        observed_alid = str(variant_by_index[int(first["variant_index"][0])]["alid"])
        results = (
            ("a", query.analysis("a")),
            ("b", query.analysis("b")),
            ("lookup", query.lookup([observed_alid], ["a", "b"])),
        )
    alid_by_index = {i: row["alid"] for i, row in variant_by_index.items()}
    seen_imputed = 0
    for name, result in results:
        assert len(result["z"]) > 0, f"{name} returned nothing"
        for vi, ai, se, eaf, status in zip(
            result["variant_index"],
            result["analysis_index"],
            result["se"],
            result["eaf"],
            result["association_status"],
            strict=True,
        ):
            alid = alid_by_index[int(vi)]
            analysis_id = analysis_by_index[int(ai)]
            if status == "missing":
                assert np.isnan(se) and np.isnan(eaf)
                continue
            assert np.isfinite(se) and se > 0
            if status == "imputed":
                seen_imputed += 1
                assert np.isfinite(eaf), "an imputed cell's panel frequency must be decoded"
                np.testing.assert_allclose(eaf, panel_eaf[alid], rtol=0.02)
            else:
                expected_se, expected_eaf = expected[analysis_id][alid]
                np.testing.assert_allclose(eaf, expected_eaf, rtol=0.02)
                np.testing.assert_allclose(se, expected_se, rtol=0.02)
    assert seen_imputed > 0, "the shapes above must exercise imputed cells"


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
