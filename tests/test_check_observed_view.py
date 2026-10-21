"""The Observed-View Check (issue #289, stores ADR 0037).

Every test goes through the CLI or the module's one public entry point. The
fixture pair is the Dense observed -> completed pair of `test_dense_completion`
with an EAF column added, so that all three statistics can be perturbed, and
its expected counts are worked by hand from the source rows and the LD panel
below -- never counted with the code under test.
"""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copytree

import numpy as np
import pytest
import test_dense_completion as tdc
from cli_output import normalize_cli_output
from typer.testing import CliRunner

from opengwasdb.build.observed import build_dense_observed_from_sources
from opengwasdb.cli.main import app
from opengwasdb.encoding import DenseEafPlane, DenseSePlane, DenseZPlane
from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.query import query_store
from opengwasdb.store.open import open_store
from opengwasdb.validation.observed_view import check_observed_view

HEADER = tdc.SOURCE_HEADER + "\teaf"
# The `test_dense_completion` source, one frequency per row. Five distinct variants
# (1:1000000:A:G is observed by both Analyses), six observed cells, two Analyses.
EAFS = ["0.30", "0.31", "0.32", "0.34", "0.36", "0.38"]
ROWS = [f"{row}\t{eaf}" for row, eaf in zip(tdc.SOURCE_ROWS, EAFS, strict=True)]
# The LD panel adds 1:1050000:C:T between the source rows 1:1000000 and 1:1100000, and two
# chr2 rows after every source row. Only the first lies inside a source chunk-row's span.
N_PARENT, N_CHILD, N_ANALYSES, N_OBSERVED, N_INSERTED_IN_SPAN = 5, 8, 2, 6, 1


def _build_pair(root: Path, rows: list[str], panel: Path) -> tuple[Path, Path]:
    source = root / "associations.tsv"
    source.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    observed, completed = root / "obs.opengwasdb", root / "comp.opengwasdb"
    build_dense_observed_from_sources(
        [source], observed, store_id="test", release_id="obs-v1", reference_assembly="GRCh38"
    )
    complete_dense_store(
        observed, completed, panel, ancestry="EUR", min_cor=0.0, release_id="comp-v1"
    )
    return observed, completed


@pytest.fixture(scope="module")
def _pair_seed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("observed-view-pair")
    return _build_pair(root, ROWS, tdc._make_ld_panel(root))


@pytest.fixture
def pair(tmp_path: Path, _pair_seed: tuple[Path, Path]) -> tuple[Path, Path]:
    observed, completed = _pair_seed
    return (
        copytree(observed, tmp_path / "obs.opengwasdb"),
        copytree(completed, tmp_path / "comp.opengwasdb"),
    )


def _run(source: Path, target: Path, out: Path, *extra: str, seed: int = 7):
    return CliRunner().invoke(
        app,
        ["check-observed-view", str(source), str(target), "--seed", str(seed),
         "--output", str(out), *extra],
    )


def _verdict(result, out: Path) -> dict:
    assert out.exists(), result.output
    return json.loads(out.read_text())


def _cell(store: Path, alid: str, analysis: str) -> tuple[np.ndarray, np.ndarray]:
    found = query_store(store).lookup([alid], [analysis])
    assert len(found["z"]) == 1, f"fixture must observe {alid} in {analysis}"
    return (
        np.array([int(found["variant_index"][0])]),
        np.array([int(found["analysis_index"][0])]),
    )


def _plane(store: Path, cls):
    opened = open_store(store)
    return cls.open(opened.arrays(mode="r+"), opened.manifest.encoding)


def _row_of(store: Path, alid: str) -> int:
    from opengwasdb.variants import VariantAxis

    axis = VariantAxis(store)
    try:
        record = axis.by_identifier(alid)
        assert record is not None, f"fixture must hold {alid}"
        return record.variant_index
    finally:
        axis.close()


def _failed(verdict: dict) -> dict[str, int]:
    return {k: v for k, v in verdict["totals"].items() if v and k in verdict["failures"]}


# --- seam 1: a fixture pair passes -----------------------------------------


def test_a_completed_fixture_pair_passes_with_hand_counted_totals(tmp_path, pair):
    observed, completed = pair
    assert open_store(completed).arrays(mode="r")["imputed"].shape == (N_CHILD, N_ANALYSES)
    out = tmp_path / "verdict.json"

    result = _run(observed, completed, out)

    assert result.exit_code == 0, result.output
    verdict = _verdict(result, out)
    assert json.loads(result.stdout) == verdict
    assert verdict["passed"] is True and verdict["failures"] == []
    assert verdict["seed"] == 7
    assert verdict["axis"]["n_parent"] == N_PARENT
    assert verdict["axis"]["n_child"] == N_CHILD
    assert verdict["axis"]["parent_rows_unmapped"] == 0
    assert verdict["axis"]["order_preserved"] is True
    assert verdict["analyses"]["identical"] is True
    # One parent chunk-row holds the whole grid, so it is every stratum but the boundary one.
    assert verdict["bands_checked"] == 1
    assert verdict["bands_by_stratum"] == {
        "random": 1, "chrom_edge": 1, "final_partial": 1, "writer_band_boundary": 0,
    }
    totals = verdict["totals"]
    assert totals["cells_compared"] == N_PARENT * N_ANALYSES
    assert totals["parent_observed"] == N_OBSERVED
    assert totals["inserted_rows"] == N_INSERTED_IN_SPAN
    assert all(totals[name] == 0 for name in (
        "z_mismatch", "se_mismatch", "eaf_mismatch", "observed_marked_imputed",
        "observed_missing_in_child", "missing_became_observed", "inserted_observed_cells",
    ))
    assert verdict["max_abs_diff"] == {"z": 0.0, "se": 0.0, "eaf": 0.0}
    assert verdict["source"]["release_id"] == "obs-v1"
    assert verdict["target"]["release_id"] == "comp-v1"
    assert verdict["source"]["store_id"] == verdict["target"]["store_id"] == "test"


# --- seam 2: negative controls ---------------------------------------------


def _perturb_z(store: Path) -> None:
    rows, cols = _cell(store, "1:1000000:A:G", "a1")
    plane = _plane(store, DenseZPlane)
    plane.patch(rows, cols, plane.points(rows, cols) + 1.0)


def _perturb_se(store: Path) -> None:
    rows, cols = _cell(store, "1:1000000:A:G", "a1")
    plane = _plane(store, DenseSePlane)
    plane.patch(rows, cols, plane.points(rows, cols) * 2.0)


def _perturb_eaf(store: Path) -> None:
    rows, cols = _cell(store, "1:1000000:A:G", "a1")
    plane = _plane(store, DenseEafPlane)
    plane.patch(rows, cols, plane.points(rows, cols) * 0.5)


def _flag_observed_cell_imputed(store: Path) -> None:
    rows, cols = _cell(store, "1:1000000:A:G", "a1")
    open_store(store).arrays(mode="r+")["imputed"][int(rows[0]), int(cols[0])] = 1


def _blank_observed_cell(store: Path) -> None:
    rows, cols = _cell(store, "1:1000000:A:G", "a1")
    _plane(store, DenseZPlane).patch(rows, cols, np.array([np.nan], dtype=np.float32))


def _observe_a_missing_cell(store: Path) -> None:
    # a2 observes only 1:1000000, so 1:900000 is a source-missing cell in a2.
    rows, cols = np.array([_row_of(store, "1:900000:A:G")]), np.array([1])
    _plane(store, DenseZPlane).patch(rows, cols, np.array([2.5], dtype=np.float32))
    _plane(store, DenseSePlane).patch(rows, cols, np.array([0.3], dtype=np.float32))


def _observe_an_inserted_row(store: Path) -> None:
    # The panel-only row 1:1050000:C:T sits inside the sampled span, and a2 is never
    # imputed, so this cell is missing and unflagged until it is patched.
    rows, cols = np.array([_row_of(store, "1:1050000:C:T")]), np.array([1])
    _plane(store, DenseZPlane).patch(rows, cols, np.array([2.5], dtype=np.float32))
    _plane(store, DenseSePlane).patch(rows, cols, np.array([0.3], dtype=np.float32))


@pytest.mark.parametrize(
    ("perturb", "counter"),
    [
        (_perturb_z, "z_mismatch"),
        (_perturb_se, "se_mismatch"),
        (_perturb_eaf, "eaf_mismatch"),
        (_flag_observed_cell_imputed, "observed_marked_imputed"),
        (_blank_observed_cell, "observed_missing_in_child"),
        (_observe_a_missing_cell, "missing_became_observed"),
        (_observe_an_inserted_row, "inserted_observed_cells"),
    ],
    ids=lambda value: getattr(value, "__name__", value),
)
def test_a_perturbed_completed_store_fails_on_the_right_counter(tmp_path, pair, perturb, counter):
    observed, completed = pair
    perturb(completed)
    out = tmp_path / "verdict.json"

    result = _run(observed, completed, out)

    assert result.exit_code == 1, result.output
    verdict = _verdict(result, out)
    assert verdict["passed"] is False
    assert verdict["totals"][counter] == 1
    assert counter in verdict["failures"]
    assert verdict["bands_with_mismatch"] == [0]
    if counter.endswith("_mismatch") and counter != "missing_became_observed":
        plane = counter.removesuffix("_mismatch")
        assert verdict["max_abs_diff"][plane] > 0.0


def test_imputed_cells_are_not_failures(tmp_path, pair):
    # The fixture's completion imputes nothing (its LD is too weak), so impute by hand:
    # a source-missing cell and a cell on an inserted row, each finite and flagged.
    observed, completed = pair
    _observe_a_missing_cell(completed)
    _observe_an_inserted_row(completed)
    flags = open_store(completed).arrays(mode="r+")["imputed"]
    flags[_row_of(completed, "1:900000:A:G"), 1] = 1
    flags[_row_of(completed, "1:1050000:C:T"), 1] = 1
    out = tmp_path / "verdict.json"

    result = _run(observed, completed, out)

    assert result.exit_code == 0, result.output
    totals = _verdict(result, out)["totals"]
    assert totals["missing_now_imputed"] == 1
    assert totals["missing_became_observed"] == totals["inserted_observed_cells"] == 0


def _point_alids_at_each_others_rows(store: Path, first: bytes, second: bytes) -> None:
    alids = np.load(store / "variant_alid_bytes.npy")
    rows = np.load(store / "variant_alid_rows.npy")
    i, j = (int(np.flatnonzero(alids == alid)[0]) for alid in (first, second))
    rows[[i, j]] = rows[[j, i]]
    np.save(store / "variant_alid_rows.npy", rows)


def test_a_misaligned_axis_fails_and_still_writes_a_verdict(tmp_path, pair):
    observed, completed = pair
    _point_alids_at_each_others_rows(completed, b"1:900000:A:G", b"1:950000:A:C")
    out = tmp_path / "verdict.json"

    result = _run(observed, completed, out)

    assert result.exit_code == 1, result.output
    verdict = _verdict(result, out)
    assert verdict["passed"] is False
    assert verdict["axis"]["order_preserved"] is False
    assert "order_not_preserved" in verdict["failures"]
    assert verdict["bands_checked"] == 0  # rows that do not line up are not compared


def test_a_row_map_shifted_by_an_inserted_row_fails_on_the_cells(tmp_path, pair):
    # Order is preserved here -- the source row is simply mapped onto the panel row after it --
    # so only the cell comparison can see it (the ADR's shift-by-one negative control).
    observed, completed = pair
    _point_alids_at_each_others_rows(completed, b"1:1000000:A:G", b"1:1050000:C:T")
    out = tmp_path / "verdict.json"

    result = _run(observed, completed, out)

    assert result.exit_code == 1, result.output
    verdict = _verdict(result, out)
    assert verdict["axis"]["order_preserved"] is True
    assert verdict["totals"]["z_mismatch"] > 0
    assert verdict["totals"]["observed_missing_in_child"] > 0


def test_a_different_analyses_order_fails(tmp_path, pair):
    observed, completed = pair
    path = completed / "analyses.tsv"
    header, *lines = path.read_text().splitlines()
    assert len(lines) == N_ANALYSES
    path.write_text("\n".join([header, *reversed(lines)]) + "\n")
    out = tmp_path / "verdict.json"

    result = _run(observed, completed, out)

    assert result.exit_code == 1, result.output
    verdict = _verdict(result, out)
    assert verdict["analyses"]["identical"] is False
    assert verdict["failures"] == ["analyses_mismatch"]


# --- seam 3: refusals ------------------------------------------------------


def _assert_refused(result, out: Path, reason: str) -> None:
    assert result.exit_code == 2, result.output
    assert reason in normalize_cli_output(result.output)
    assert not out.exists(), "nothing was checked, so no verdict is written"


def test_a_ragged_store_is_refused(tmp_path, pair):
    import test_ragged_completion as trc

    from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd

    ragged = tmp_path / "ragged.opengwasdb"
    build_ragged_from_besd(
        trc._make_besd_fixture(tmp_path), ragged, store_id="r", release_id="r1", tissue="Blood"
    )
    out = tmp_path / "verdict.json"

    _assert_refused(_run(ragged, pair[1], out), out, "is not Dense (primary_layout=ragged)")


def test_a_hybrid_root_is_refused_as_not_supported_yet(tmp_path, pair):
    import test_hybrid_completion as thc

    hybrid = thc._build_source(tmp_path)
    out = tmp_path / "verdict.json"

    result = _run(hybrid, pair[1], out)

    _assert_refused(result, out, "Hybrid Dense Components are not supported yet")


def test_an_observed_only_target_is_refused(tmp_path, pair):
    observed, _ = pair
    out = tmp_path / "verdict.json"

    _assert_refused(
        _run(observed, observed, out), out,
        "is not Reference-Completed (completion_state=observed_only)",
    )


def test_a_reference_completed_source_is_refused(tmp_path, pair):
    _, completed = pair
    out = tmp_path / "verdict.json"

    _assert_refused(
        _run(completed, completed, out), out,
        "is not Observed-Only (completion_state=reference_completed)",
    )


# --- seam 4: strata --------------------------------------------------------

# 12 source rows on chr1 with LD-panel rows interleaved, so a child row is not its parent row:
#   child row  0  1  2  3* 4  5  6  7  8  9* 10 11 12* 13 14 15* 16*   (* panel-only)
#   parent row 0  1  2     3  4  5  6  7     8  9     10 11
# at positions 900000 + 20000 * i (i < 10), 1110000 and 1120000; the panel adds 1:950000:A:C,
# 1:1050000:C:T and 1:1100000:A:C inside the span and two chr2 rows after it.
_STRATA_POSITIONS = [900_000 + 20_000 * i for i in range(10)] + [1_110_000, 1_120_000]
_STRATA_PARENT_TO_CHILD = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14]


@pytest.fixture
def small_chunked_pair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    """The geometry of `test_a_dense_completion_writes_whole_shards`, with a wider axis.

    Parent chunk-rows of 2 rows; the child shard is 4 rows and the writer's memory target
    6 rows, so the writer band is one 4-row shard. The check reads that off the child, so
    the patch must still be in place when it runs.
    """
    import opengwasdb.layouts.dense.complete as complete_module
    import opengwasdb.store.arrays as arrays_module
    from opengwasdb.build.source import stream_normalised_associations
    from opengwasdb.layouts.dense.build import build_dense_observed_store

    monkeypatch.setattr(arrays_module, "DENSE_SHARD_SHAPE", (4, 2))
    monkeypatch.setattr(arrays_module, "DENSE_CHUNK_SHAPE", (2, 1))
    monkeypatch.setattr(complete_module, "DEFAULT_CHUNK_SHAPE", (2, 1))
    monkeypatch.setattr(complete_module, "_BAND_ROWS", 6)
    rows = [
        f"a1\tp1\tTrait\tTrait primary\t1\t{pos}\tA\tG\t2.0\t0.1\trs{i}\tsd\t0.3"
        for i, pos in enumerate(_STRATA_POSITIONS)
    ]
    source = tmp_path / "associations.tsv"
    source.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    observed, completed = tmp_path / "obs.opengwasdb", tmp_path / "comp.opengwasdb"
    build_dense_observed_store(
        stream_normalised_associations([source]), observed, store_id="s", release_id="obs",
        reference_assembly="GRCh38", chunk_shape=(2, 1),
    )
    complete_dense_store(
        observed, completed, tdc._make_ld_panel(tmp_path), ancestry="EUR", min_cor=0.0
    )
    return observed, completed


def test_strata_are_chosen_from_the_childs_own_band_geometry(small_chunked_pair):
    observed, completed = small_chunked_pair
    assert _row_of(completed, "1:1100000:A:C") == 12, "fixture must interleave panel rows"
    assert [_row_of(completed, f"1:{p}:A:G") for p in _STRATA_POSITIONS] == (
        _STRATA_PARENT_TO_CHILD
    )

    # n_random=0 leaves only the deliberate strata: the one contig's two edges, the last
    # chunk-row, and the parent chunk-rows either side of the child's band boundaries at
    # child rows 4, 8, 12 (16 maps past the last parent row). A 250,000-row band would
    # have no boundary in 17 child rows at all.
    verdict = check_observed_view(observed, completed, seed=1, n_random=0).record

    assert verdict["geometry"] == {
        "parent_chunk_rows": 2, "n_parent_chunk_rows": 6, "completion_band_rows": 4,
    }
    chosen = {entry["band"]: entry["strata"] for entry in verdict["bands"]}
    assert chosen == {
        0: ["chrom_edge"],
        1: ["writer_band_boundary"],
        3: ["writer_band_boundary"],
        4: ["writer_band_boundary"],
        5: ["chrom_edge", "final_partial", "writer_band_boundary"],
    }
    assert verdict["bands_by_stratum"] == {
        "random": 0, "chrom_edge": 2, "final_partial": 1, "writer_band_boundary": 4,
    }
    assert verdict["bands_checked"] == 5
    assert verdict["totals"]["cells_compared"] == 5 * 2  # five two-row chunk-rows, one Analysis
    assert verdict["passed"], verdict["failures"]


def test_the_same_seed_gives_the_same_verdict_and_the_whole_grid_when_it_is_small(
    small_chunked_pair,
):
    observed, completed = small_chunked_pair

    first = check_observed_view(observed, completed, seed=3, n_random=3).record
    again = check_observed_view(observed, completed, seed=3, n_random=3).record
    everything = check_observed_view(observed, completed, seed=3, n_random=1000).record

    for record in (first, again, everything):
        record.pop("seconds")
    assert first == again
    assert first["bands_by_stratum"]["random"] == 3
    assert everything["bands_by_stratum"]["random"] == 6  # fewer chunk-rows than asked: all
    assert everything["totals"]["cells_compared"] == len(_STRATA_POSITIONS)
    # Only 1:950000 lies inside a chunk-row's span. 1:1050000 and 1:1100000 fall between two
    # chunk-rows (child rows 9 and 12), which no span covers, as in the prototype's counts.
    assert everything["totals"]["inserted_rows"] == 1
    assert everything["passed"], everything["failures"]


# --- seam 5: wide ALIDs ----------------------------------------------------

_LONG_ALLELE = "ACGT" * 10  # 1:1150000:A:ACGT... is 53 bytes, wider than the 32-byte index key


def _with_a_long_indel(root: Path) -> tuple[Path, Path]:
    long_row = (
        f"a1\tp1\tHeight\tHeight primary\t1\t1150000\t{_LONG_ALLELE}\tA\t1.2\t0.3\trsL\tsd\t0.4"
    )
    return _build_pair(root, [*ROWS, long_row], tdc._make_ld_panel(root))


def test_a_parent_row_wider_than_the_alid_index_is_mapped_not_unmapped(tmp_path):
    observed, completed = _with_a_long_indel(tmp_path)
    indexed = len(np.load(observed / "variant_alid_bytes.npy"))
    assert indexed == N_PARENT, "fixture must leave the long ALID out of the index"

    verdict = check_observed_view(observed, completed, seed=1).record

    assert verdict["axis"]["n_parent"] == N_PARENT + 1
    assert verdict["axis"]["parent_rows_unmapped"] == 0
    assert verdict["axis"]["parent_rows_unindexed"] == 1
    assert verdict["axis"]["child_rows_unindexed"] == 1
    assert verdict["axis"]["order_preserved"] is True
    assert verdict["totals"]["parent_observed"] == N_OBSERVED + 1
    assert verdict["passed"], verdict["failures"]


def test_stores_with_no_alid_index_are_still_checked(tmp_path, pair):
    observed, completed = pair
    for store in pair:
        (store / "variant_alid_bytes.npy").unlink()
        (store / "variant_alid_rows.npy").unlink()

    verdict = check_observed_view(observed, completed, seed=1).record

    assert verdict["axis"]["parent_rows_unindexed"] == N_PARENT
    assert verdict["axis"]["child_rows_unindexed"] == N_CHILD
    assert verdict["axis"]["parent_rows_unmapped"] == 0
    assert verdict["totals"]["parent_observed"] == N_OBSERVED
    assert verdict["passed"], verdict["failures"]
