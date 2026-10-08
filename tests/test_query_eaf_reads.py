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

import csv
from pathlib import Path

import numpy as np
import pytest
import zarr
from residual_fixtures import residual_eligible_records
from store_reads import chunk_reads, duplicate_chunk_keys, old_index_without_fields
from test_hybrid_completion import (
    _residual_hybrid_crossover_source,
    _residual_ld_panel_with_crossover,
)
from test_hybrid_shared_se_plan_e2e import (
    N_OFF_PANEL,
    N_PANEL,
    OFF_BASE,
    _build_hybrid,
    _se_value,
    overflow_alid,
    panel_alid,
)
from test_ragged_residual_completion import _TRAIT_BP, RaggedResidualScenario
from test_se_residual_encoding import _residual_source_and_panel

from opengwasdb.encoding.codec import EAF_ABSENT
from opengwasdb.encoding.planes import DenseZPlane
from opengwasdb.layouts.dense.build import build_dense_observed_store
from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.layouts.dense.top_hits import build_top_hit_indexes, threshold_key
from opengwasdb.layouts.hybrid.complete import complete_hybrid_store
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
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
    root = zarr.open_group(str(completed / "data.zarr"), mode="r+", zarr_format=3)
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


#: Where `ragged_single_trait` moves analysis `b`'s Trait position to.
_RELOCATED_TRAIT_BP = 2_000_000


def _relocate_trait_bp(store: Path, analysis_id: str, bp: int) -> None:
    """Move one Analysis's Trait position in a built store's analyses.tsv.

    `range_by_analysis` selects Analyses by `trait_chr`/`trait_bp`, and the
    Ragged residual fixture puts both of them at the same position. Relocating
    one lets the read-count test select a single Analysis: without that, a CSR
    chunk shared by two Analyses is legitimately read once per Analysis, and a
    duplicate-key assertion could not tell that from the twice-per-Analysis
    defect #253 removes (review round 1).
    """
    path = store / "analyses.tsv"
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    header = rows[0]
    bp_col, id_col = header.index("trait_bp"), header.index("analysis_id")
    moved = 0
    for row in rows[1:]:
        if row[id_col] == analysis_id:
            row[bp_col] = str(bp)
            moved += 1
    assert moved == 1, f"{analysis_id} must appear exactly once in analyses.tsv"
    path.write_text("\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def ragged_single_trait(tmp_path_factory: pytest.TempPathFactory) -> RaggedResidualScenario:
    """The completed Ragged fixture with analysis `b` relocated off `a`'s trait."""
    scenario = RaggedResidualScenario(tmp_path_factory)
    _relocate_trait_bp(scenario.completed, "b", _RELOCATED_TRAIT_BP)
    return scenario


@pytest.fixture(scope="module")
def hybrid_completed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A Reference-Completed Hybrid with residual SE and imputed Dense cells.

    The residual in both components is what makes SE decoding read the EAF
    plane at all; the imputed Dense cells are what make an `observed_only`
    check non-vacuous (the simple Hybrid fixture is observed-only and has no
    `imputed` array).
    """
    tmp = tmp_path_factory.mktemp("eaf_reads_hybrid_completed")
    src, crossover_alid, _crossover_se, crossover_eaf = _residual_hybrid_crossover_source(tmp)
    ld = _residual_ld_panel_with_crossover(tmp, crossover_alid, crossover_eaf)
    dst = tmp / "comp.opengwasdb"
    complete_hybrid_store(src, dst, ld, min_cor=0.0, thresh=0.9)
    return dst


@pytest.fixture(scope="module")
def dense_completed_imputed_hit(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, np.ndarray, np.ndarray]:
    """The Dense trap fixture, with one imputed cell made a decisive top hit.

    The residual Dense fixture's imputed cells are all low-|z|, so a top-hit
    query never returned one and `observed_only` had nothing to filter. Patching
    one imputed cell's `z` to 8 and rebuilding the tier from the store's own
    plane puts an imputed row in the index the older-index fallback decodes.
    """
    store, observed, panel = _dense_completed_fixture(tmp_path_factory.mktemp("eaf_reads_hit"))
    encoding = StoreManifest.load(store).encoding
    root = zarr.open_group(str(store / "data.zarr"), mode="r+", zarr_format=3)
    imputed = np.asarray(root["imputed"][:], dtype=bool)
    rows, cols = np.where(imputed)
    assert len(rows) > 0, "the fixture must have an imputed cell to promote"
    DenseZPlane.open(root, encoding).patch(
        rows[:1].astype("int64"), cols[:1].astype("int64"), np.array([8.0], dtype=np.float32)
    )
    build_top_hit_indexes(store)
    tier = zarr.open_group(str(store / "data.zarr"), mode="r")["top_hits"][threshold_key(5e-8)]
    assert int(np.asarray(tier["imputed"][:]).sum()) > 0, "the promoted cell must be a top hit"
    return store, observed, panel


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


def test_ragged_range_by_analysis_reads_each_eaf_array_once(
    ragged_single_trait: RaggedResidualScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`range_by_analysis` decodes one Analysis at a time and reads once."""
    with query_store(ragged_single_trait.completed) as query:
        selected = query._analysis_indices_in_range("1", _TRAIT_BP, _TRAIT_BP)
        assert selected == [0], "the relocated fixture must select exactly one Analysis"
        with chunk_reads(monkeypatch, _EAF_ARRAYS) as reads:
            result = query.range_by_analysis("1", _TRAIT_BP, _TRAIT_BP)
    assert len(result["z"]) > 0, "the selected Analysis must return rows"
    duplicates = duplicate_chunk_keys(reads)
    assert duplicates == {}, f"range_by_analysis re-read EAF chunks: {duplicates}"
    assert reads["eaf"] and reads["eaf_baseline"] and reads["imputed"], (
        "range_by_analysis must read all three arrays for this to mean anything"
    )


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


# ── no answer changes (the trap tests) ──────────────────────────────────────


def _status_split(result: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    status = np.asarray(result["association_status"])
    return status == "observed", status == "imputed"


def _assert_observed_only_lockstep(
    full: dict[str, np.ndarray], filtered: dict[str, np.ndarray], label: str
) -> None:
    """`observed_only` drops the imputed rows from every returned array.

    The complement (`association_status == "missing"`) is kept: the Dense
    facade never returns a non-finite cell, the Ragged facade returns missing
    cells under `observed_only` too, and the filter is on the imputed mask, not
    on finiteness.
    """
    keep = np.asarray(full["association_status"]) != "imputed"
    assert keep.any() and not keep.all(), f"{label} must have both observed and imputed cells"
    assert len(filtered["z"]) == int(keep.sum()), label
    for key in ("variant_index", "analysis_index", "z", "se", "eaf", "association_status"):
        np.testing.assert_array_equal(
            np.asarray(full[key])[keep], filtered[key], err_msg=f"{label}:{key}"
        )


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


def test_ragged_observed_only_keeps_arrays_in_lockstep(
    ragged_completed: RaggedResidualScenario,
) -> None:
    """The Ragged parallel arrays stay aligned under `observed_only`."""
    with query_store(ragged_completed.completed) as query:
        variants = query.variants_table()
        full_a = query.analysis("a")
        full_b = query.analysis("b")
        filtered_a = query.analysis("a", observed_only=True)
        filtered_b = query.analysis("b", observed_only=True)
        imputed_b = np.asarray(full_b["association_status"]) == "imputed"
        assert imputed_b.any(), "analysis b must have imputed cells for this to mean anything"
        # A variant analysis `a` observed and `b` had imputed: a panel-only
        # variant is imputed for both, and a lookup of one would have no
        # observed cell left to keep.
        observed_a = {
            int(v)
            for v, status in zip(full_a["variant_index"], full_a["association_status"], strict=True)
            if status == "observed"
        }
        shared = [int(v) for v in full_b["variant_index"][imputed_b] if int(v) in observed_a]
        assert shared, "the fixture must impute a variant the other Analysis observed"
        lookup_alid = str(variants[shared[0]]["alid"])
        full_lookup = query.lookup([lookup_alid], ["a", "b"])
        filtered_lookup = query.lookup([lookup_alid], ["a", "b"], observed_only=True)
    _assert_observed_only_lockstep(full_a, filtered_a, "analysis a")
    _assert_observed_only_lockstep(full_b, filtered_b, "analysis b")
    _assert_observed_only_lockstep(full_lookup, filtered_lookup, "lookup")


def test_hybrid_completed_observed_only_keeps_arrays_in_lockstep(hybrid_completed: Path) -> None:
    """A completed Hybrid's Dense imputed cells drop in lockstep."""
    with query_store(hybrid_completed) as query:
        full = query.analysis("trait_b")
        filtered = query.analysis("trait_b", observed_only=True)
    _assert_observed_only_lockstep(full, filtered, "hybrid analysis")


def test_dense_old_index_top_hits_observed_only_keeps_arrays_in_lockstep(
    dense_completed_imputed_hit: tuple[Path, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The older-index top-hit fallback drops an imputed hit in lockstep."""
    store, observed, panel = dense_completed_imputed_hit
    with query_store(store) as query, old_index_without_fields(monkeypatch, _EAF_ARRAYS):
        full = query.top_hits(threshold=5e-8, analysis_id="b")
        filtered = query.top_hits(threshold=5e-8, analysis_id="b", observed_only=True)
    _assert_observed_only_lockstep(full, filtered, "old-index top_hits")
    _check_dense_cells(full, observed, panel, lambda row: 8.0 if row % 50 == 0 else 1.0)


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
