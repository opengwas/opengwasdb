"""The Observed-View Check on a Hybrid Store (issue #295, stores ADR 0037).

Every test goes through the CLI or the check's one public entry point. The
fixture is an Observed-Only Hybrid built from three GWAS-VCF Analyses with
frequencies, and the `complete-hybrid` output made from it with an LD panel
that extends the Dense axis. Every count asserted below is worked by hand from
the tables in this docstring, never counted with the code under test.

Build panel (the source Dense axis): 12 variants, 1:1000:A:G .. 1:12000:A:G.
Off-panel (the source Overflow): 6 variants, 1:100000:A:G .. 1:105000:A:G.

    Analysis   panel observed          off-panel observed
    trait_a    all 12                  all 6
    trait_b    all 12                  100000, 101000, 102000, 103000
    trait_c    10 (not 11000, 12000)   105000

So the source Dense Component holds 12 x 3 = 36 cells, 34 observed, and the
Overflow 6 + 4 + 1 = 11 associations.

The LD panel covers the 12 panel variants plus 1:6500:A:G (in no Analysis), in
one block, and 1:101000:A:G and 1:104000:A:G in a second. Completion extends the
Dense axis by those three rows. The two off-panel variants are crossovers: their
3 Overflow associations (101000 in trait_a and trait_b, 104000 in trait_a) move
into the Dense Component, leaving 11 - 3 = 8 in the Overflow. Completion imputes
trait_c's two unobserved panel variants; the crossover block has no Dense
observation to impute from.

The Dense Component is built in chunk-rows of 2 source rows, and the completed one
in shards of 4 rows, which the completion writer's band then is. So the Dense grid
has six chunk-rows, and its strata come from the Stores' own geometry. The
completed axis interleaves the three inserted rows:

    child row   0  1  2  3  4  5  6* 7  8  9  10 11 12 13* 14*   (* inserted)
    parent row  0  1  2  3  4  5     6  7  8  9  10 11
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from shutil import copytree

import numpy as np
import pytest
from cli_output import normalize_cli_output
from residual_fixtures import write_gwas_vcf_with_eaf
from test_check_observed_view import _assert_refused, _run, _verdict
from test_hybrid_completion import _write_ld_block

import opengwasdb.layouts.dense.complete as dense_complete
import opengwasdb.store.arrays as arrays_module
from opengwasdb.encoding import (
    DenseEafPlane,
    DenseSePlane,
    DenseZPlane,
    EafExceptionBuilder,
    StoreCodec,
    positions_pairs,
)
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.hybrid.complete import complete_hybrid_store
from opengwasdb.layouts.ragged.by_variant import build_variant_index
from opengwasdb.layouts.ragged.zarr_csr import (
    AnalysisAssociations,
    RaggedCSRReader,
    RaggedCSRWriter,
)
from opengwasdb.query import query_store
from opengwasdb.store.open import open_store
from opengwasdb.validation.observed_view import check_observed_view
from opengwasdb.variants import VariantAxis

ANALYSES = ("trait_a", "trait_b", "trait_c")
PANEL_POSITIONS = [1000 * (i + 1) for i in range(12)]
OFF_PANEL_POSITIONS = [100_000 + 1000 * j for j in range(6)]
PANEL_ONLY_POSITION = 6500
CROSSOVER_POSITIONS = (101_000, 104_000)
CROSSOVER_CELLS = ((101_000, "trait_a"), (101_000, "trait_b"), (104_000, "trait_a"))
OBSERVED_PANEL = {
    "trait_a": PANEL_POSITIONS,
    "trait_b": PANEL_POSITIONS,
    "trait_c": PANEL_POSITIONS[:10],
}
OBSERVED_OFF_PANEL = {
    "trait_a": OFF_PANEL_POSITIONS,
    "trait_b": OFF_PANEL_POSITIONS[:4],
    "trait_c": OFF_PANEL_POSITIONS[5:],
}
N_PANEL, N_ANALYSES, N_DENSE_OBSERVED, N_INSERTED = 12, 3, 34, 3
N_OVERFLOW, N_CROSSOVERS, N_CROSSOVER_ROWS, N_IMPUTED = 11, 3, 2, 2
# Varied z, so completion's fit has signal to find; scaled per Analysis.
_Z = [2.1, -1.3, 0.8, 3.2, -0.5, 1.7, 2.4, -2.2, 0.9, 1.1, -1.8, 2.6]
_OFF_Z = [1.4, -2.7, 0.6, 2.2, -1.1, 3.5]


def _alid(position: int) -> str:
    return f"1:{position}:A:G"


def _frequency(position: int, column: int) -> float:
    """A frequency per variant, nudged per Analysis so every residual is non-zero."""
    base = 0.08 + 0.8 * ((position // 1000) % 37) / 37
    return base * (1.0 + 0.03 * (column - 1))


def _vcf_rows(column: int, analysis: str) -> list[str]:
    observed = set(OBSERVED_PANEL[analysis]) | set(OBSERVED_OFF_PANEL[analysis])
    rows = []
    for positions, zs in ((PANEL_POSITIONS, _Z), (OFF_PANEL_POSITIONS, _OFF_Z)):
        for position, z in zip(positions, zs, strict=True):
            if position not in observed:
                continue
            se = 0.05 + 0.01 * column + 0.002 * (position // 1000 % 7)
            effect = z * (1.0 + 0.1 * column) * se
            freq = _frequency(position, column)
            rows.append(
                f"1\t{position}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t{effect:.6f}:{se:.6f}:{freq:.6f}\n"
            )
    return rows


def _build_observed(root: Path, chunk_shape: tuple[int, int] | None = None) -> Path:
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\tsource_assembly"
    ]
    for column, analysis in enumerate(ANALYSES):
        vcf = write_gwas_vcf_with_eaf(root / f"{analysis}.vcf", _vcf_rows(column, analysis))
        lines.append(f"{analysis}\t{vcf}\t{analysis}\t1000\tsd\tdeclared_standardised\thg38")
    manifest = root / "manifest.tsv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    panel = root / "panel.txt"
    panel.write_text("\n".join(_alid(p) for p in PANEL_POSITIONS) + "\n", encoding="utf-8")
    source = root / "obs.opengwasdb"
    shape = {} if chunk_shape is None else {"chunk_shape": chunk_shape}
    build_hybrid_from_vcf_manifest(
        manifest, source, reference_panel=panel, store_id="hyb", release_id="obs-v1", **shape
    )
    return source


def _extending_ld_panel(root: Path) -> Path:
    ld = root / "ld_panel"
    block = sorted([*PANEL_POSITIONS, PANEL_ONLY_POSITION])
    for positions, seed in ((block, 0), (CROSSOVER_POSITIONS, 1)):
        _write_ld_block(
            ld / "EUR" / "1", f"{positions[0]}-{positions[-1]}",
            [(_alid(p), _frequency(p, 1), p) for p in positions], seed=seed,
        )
    return ld


def _complete(observed: Path, root: Path) -> Path:
    completed = root / "comp.opengwasdb"
    complete_hybrid_store(
        observed, completed, _extending_ld_panel(root), min_cor=0.0, thresh=0.9,
        release_id="comp-v1",
    )
    return completed


@pytest.fixture(scope="module")
def _pair_seed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("hybrid-pair")
    # The geometry of `test_check_observed_view._small_chunked_pair`. The Stores carry it
    # once written, so it is patched only while they are built.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(arrays_module, "DENSE_SHARD_SHAPE", (4, 2))
        patch.setattr(arrays_module, "DENSE_CHUNK_SHAPE", (2, 1))
        patch.setattr(dense_complete, "DEFAULT_CHUNK_SHAPE", (2, 1))
        patch.setattr(dense_complete, "_BAND_ROWS", 6)
        observed = _build_observed(root, chunk_shape=(2, 1))
        return observed, _complete(observed, root)


@pytest.fixture
def pair(tmp_path: Path, _pair_seed: tuple[Path, Path]) -> tuple[Path, Path]:
    return (
        copytree(_pair_seed[0], tmp_path / "obs.opengwasdb"),
        copytree(_pair_seed[1], tmp_path / "comp.opengwasdb"),
    )


def _dense_alids(store: Path) -> list[str]:
    axis = VariantAxis(store / "dense")
    try:
        return [record.alid for record in axis.all()]
    finally:
        axis.close()


def _dense_cell(store: Path, position: int, analysis: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([_dense_alids(store).index(_alid(position))]),
        np.array([ANALYSES.index(analysis)]),
    )


def _dense_plane(store: Path, cls):
    component = open_store(store).dense_component()
    return cls.open(component.arrays(mode="r+"), component.manifest.encoding)


def _imputed(store: Path):
    return open_store(store).dense_component().arrays(mode="r+")["imputed"]


def _overflow(store: Path):
    return open_store(store).arrays(mode="r+")["ragged"]


def _shared_row(store: Path, position: int) -> int:
    axis = VariantAxis(store)
    try:
        record = axis.by_identifier(_alid(position))
        assert record is not None, f"fixture must hold {_alid(position)}"
        return int(record.variant_index)
    finally:
        axis.close()


def _n_shared(store: Path) -> int:
    axis = VariantAxis(store)
    try:
        return int(axis.n_variants)
    finally:
        axis.close()


def _overflow_position(store: Path, position: int, analysis: str) -> int:
    shared = _shared_row(store, position)
    group = _overflow(store)
    column = ANALYSES.index(analysis)
    lo, hi = (int(v) for v in group["offsets"][column : column + 2])
    hits = np.flatnonzero(np.asarray(group["variant_index"][lo:hi]) == shared)
    assert len(hits) == 1, f"fixture's Overflow must hold {_alid(position)} in {analysis}"
    return lo + int(hits[0])


def _counter(verdict: dict, failure: str) -> int:
    section, name = failure.split(".")
    if section == "crossovers":
        return verdict["crossovers"]["totals"][name]
    component = verdict["components"][section]
    return {**component["totals"], **component.get("all_analyses", {})}[name]


# --- seam 1: a completed pair with crossovers passes ------------------------


def test_the_fixture_extends_the_dense_axis_and_moves_the_crossovers(pair):
    observed, completed = pair
    panel = sorted([*PANEL_POSITIONS, PANEL_ONLY_POSITION, *CROSSOVER_POSITIONS])
    assert _dense_alids(observed) == [_alid(p) for p in PANEL_POSITIONS]
    assert _dense_alids(completed) == [_alid(p) for p in panel]
    assert int(_overflow(observed)["offsets"][-1]) == N_OVERFLOW
    assert int(_overflow(completed)["offsets"][-1]) == N_OVERFLOW - N_CROSSOVERS
    flags = np.asarray(_imputed(completed)[:])
    for position in (11_000, 12_000):
        row, col = _dense_cell(completed, position, "trait_c")
        assert flags[row[0], col[0]] == 1, "completion must impute trait_c's missing cells"


def test_a_completed_hybrid_with_crossovers_passes_with_hand_counted_totals(tmp_path, pair):
    observed, completed = pair
    out = tmp_path / "verdict.json"

    result = _run(observed, completed, out)

    assert result.exit_code == 0, result.output
    verdict = _verdict(result, out)
    assert verdict["passed"] is True and verdict["failures"] == []
    assert verdict["layout"] == "hybrid" and verdict["seed"] == 7
    assert verdict["analyses"] == {"n_parent": 3, "n_child": 3, "identical": True}
    dense = verdict["components"]["dense"]
    assert (dense["axis"]["n_parent"], dense["axis"]["n_child"]) == (12, 12 + N_INSERTED)
    assert dense["bands_checked"] == 6  # fewer chunk-rows than --n-random: all of them
    assert {k: v for k, v in dense["totals"].items() if v} == {
        "cells_compared": N_PANEL * N_ANALYSES,
        "parent_observed": N_DENSE_OBSERVED,
        "inserted_rows": N_INSERTED,
        "missing_now_imputed": N_IMPUTED,
        "crossover_cells": N_CROSSOVERS,
    }
    overflow = verdict["components"]["overflow"]
    assert overflow["n_parent_associations"] == N_OVERFLOW
    assert overflow["n_child_associations"] == N_OVERFLOW - N_CROSSOVERS
    assert overflow["axis"]["n_parent"] == 18 and overflow["axis"]["n_child"] == 19
    assert overflow["analyses"] == [0, 1, 2]  # three Analyses, fewer than --n-random: all
    assert {k: v for k, v in overflow["totals"].items() if v} == {
        "parent_associations": N_OVERFLOW,
        "crossovers": N_CROSSOVERS,
        "child_associations": N_OVERFLOW - N_CROSSOVERS,
    }
    assert overflow["all_analyses"] == dict.fromkeys(
        ("observed_in_both", "imputed_cells", "analyses_count_mismatch"), 0
    )
    crossovers = verdict["crossovers"]
    assert (crossovers["n"], crossovers["rows"]) == (N_CROSSOVERS, N_CROSSOVER_ROWS)
    assert not any(crossovers["totals"].values())


def _lookup(store: Path, position: int, analysis: str) -> dict[str, np.ndarray]:
    found = query_store(store).lookup([_alid(position)], [analysis])
    assert list(found["association_status"]) == ["observed"]
    return {plane: found[plane] for plane in ("z", "se", "eaf")}


def test_each_crossover_is_its_source_value_through_the_dense_codec(pair):
    """The rule for a crossover, confirmed cell by cell, through the query path.

    The fold writes a crossover through the completed Dense planes: Z, then EAF
    against the Dense Component's baseline for that row, then SE. Encoding the
    source Overflow value with that codec and decoding it gives the completed
    cell exactly. This fixture's planes are `int16_fixed` Z, `int8_residual` EAF
    and `float16` SE, so the EAF goes through the baseline-relative coding.
    """
    observed, completed = pair
    component = open_store(completed).dense_component()
    encoding, root = component.manifest.encoding, component.arrays(mode="r")
    assert (encoding.z.kind, encoding.eaf.kind, encoding.se.kind) == (
        "int16_fixed", "int8_residual", "float16"
    )
    codec = StoreCodec(encoding)
    for position, analysis in CROSSOVER_CELLS:
        source, completed_cell = _lookup(observed, position, analysis), _lookup(
            completed, position, analysis
        )
        rows, cols = _dense_cell(completed, position, analysis)
        where = positions_pairs(rows, cols, N_ANALYSES)
        baseline = np.asarray(root["eaf_baseline"].oindex[rows], dtype=np.float32)
        exceptions = EafExceptionBuilder()
        codes = codec.encode_eaf(
            source["eaf"], baseline=baseline, positions=where, exceptions=exceptions
        )
        eaf = StoreCodec(encoding, eaf_exceptions=exceptions.table()).decode_eaf(
            codes, baseline=baseline, positions=where,
            imputed=np.zeros(1, np.uint8), reference=np.full(1, np.nan, np.float32),
        )
        assert np.array_equal(completed_cell["z"], codec.quantise_z(source["z"]))
        assert np.array_equal(completed_cell["eaf"], eaf)
        assert np.array_equal(completed_cell["se"], source["se"].astype(np.float16))


def test_a_crossover_coded_against_a_dense_baseline_passes_though_its_value_moved(
    tmp_path, pair
):
    """Identity for a crossover is the Dense codec's, not the source's bits.

    Give a crossover row an EAF baseline and write its crossovers through the
    planes again, as the fold would have against that baseline: the cells now
    differ from the source by the quantisation, and must still pass.
    """
    observed, completed = pair
    rows, cols = _dense_cell(completed, 101_000, "trait_a")
    rows, cols = np.r_[rows, rows], np.r_[cols, ANALYSES.index("trait_b")]
    root = open_store(completed).dense_component().arrays(mode="r+")
    root["eaf_baseline"][int(rows[0])] = 0.31
    values = [_lookup(observed, 101_000, analysis) for analysis in ("trait_a", "trait_b")]
    for cls, plane in ((DenseEafPlane, "eaf"), (DenseSePlane, "se")):
        _dense_plane(completed, cls).patch(rows, cols, np.concatenate([v[plane] for v in values]))
    moved = _lookup(completed, 101_000, "trait_a")["eaf"]
    assert moved != values[0]["eaf"], "the fixture must move the crossover's EAF"
    out = tmp_path / "verdict.json"

    result = _run(observed, completed, out)

    assert result.exit_code == 0, result.output
    assert _verdict(result, out)["crossovers"]["reencoding_max_abs_diff"]["eaf"] > 0.0


# --- seam 2: negative controls ----------------------------------------------


def _change_a_dense_observation(store: Path) -> None:
    rows, cols = _dense_cell(store, 3000, "trait_b")
    plane = _dense_plane(store, DenseZPlane)
    plane.patch(rows, cols, plane.points(rows, cols) + 1.0)


def _change_an_overflow_observation(store: Path) -> None:
    position = _overflow_position(store, 102_000, "trait_a")
    z = _overflow(store)["z"]
    z[position] = int(z[position]) + 10  # ten codes of the int16 fixed-point plane


def _observe_a_dense_cell_that_is_no_crossover(store: Path) -> None:
    # 1:104000 is a crossover in trait_a only; trait_b never observed it anywhere.
    rows, cols = _dense_cell(store, 104_000, "trait_b")
    _dense_plane(store, DenseZPlane).patch(rows, cols, np.array([2.5], dtype=np.float32))
    _dense_plane(store, DenseSePlane).patch(rows, cols, np.array([0.06], dtype=np.float32))


def _mark_an_overflow_association_imputed(store: Path) -> None:
    group = _overflow(store)
    flags = group.create_array(
        "imputed", shape=group["z"].shape, dtype="uint8", fill_value=0
    )
    flags[_overflow_position(store, 105_000, "trait_c")] = 1
    assert "imputed" in open_store(store).arrays(mode="r")["ragged"]


def _alter_a_crossover(store: Path) -> None:
    rows, cols = _dense_cell(store, 101_000, "trait_b")
    plane = _dense_plane(store, DenseZPlane)
    plane.patch(rows, cols, plane.points(rows, cols) * 1.5)


def _flag_a_crossover_imputed(store: Path) -> None:
    rows, cols = _dense_cell(store, 104_000, "trait_a")
    _imputed(store)[int(rows[0]), int(cols[0])] = 1


def _blank_a_crossover(store: Path) -> None:
    rows, cols = _dense_cell(store, 101_000, "trait_a")
    _dense_plane(store, DenseZPlane).patch(rows, cols, np.array([np.nan], dtype=np.float32))


def _failed_verdict(observed: Path, completed: Path, out: Path, *extra: str) -> dict:
    """Run the CLI on a pair that must fail, and the verdict it still wrote."""
    result = _run(observed, completed, out, *extra)
    assert result.exit_code == 1, result.output
    verdict = _verdict(result, out)
    assert verdict["passed"] is False
    return verdict


@pytest.mark.parametrize(
    ("perturb", "failure"),
    [
        (_change_a_dense_observation, "dense.z_mismatch"),
        (_change_an_overflow_observation, "overflow.z_mismatch"),
        (_observe_a_dense_cell_that_is_no_crossover, "dense.inserted_observed_cells"),
        (_mark_an_overflow_association_imputed, "overflow.imputed_cells"),
        (_alter_a_crossover, "crossovers.z_mismatch"),
        (_flag_a_crossover_imputed, "crossovers.observed_marked_imputed"),
        (_blank_a_crossover, "crossovers.observed_missing_in_child"),
    ],
    ids=lambda value: getattr(value, "__name__", value),
)
def test_a_perturbed_completed_hybrid_fails_on_the_right_counter(
    tmp_path, pair, perturb, failure
):
    observed, completed = pair
    perturb(completed)

    verdict = _failed_verdict(observed, completed, tmp_path / "verdict.json")

    assert failure in verdict["failures"]
    assert _counter(verdict, failure) == 1


def _rewrite_overflow(
    store: Path, edit: Callable[[str, AnalysisAssociations], AnalysisAssociations]
) -> None:
    """Rewrite `store`'s Overflow through the Ragged writer, each Analysis edited first.

    Every association is encoded again from its decoded value, under the Store's own
    plan and `eaf_baseline`, so one the edit leaves alone decodes as it did
    (`test_rewriting_the_overflow_unchanged_keeps_the_pair_passing`). The variant index
    is rebuilt, as completion builds it, so the Store can still be queried.
    """
    reader = RaggedCSRReader(store)
    edited = [edit(name, reader.get_analysis(i)) for i, name in enumerate(ANALYSES)]
    baseline = np.asarray(_overflow(store)["eaf_baseline"][:], dtype=np.float32)
    writer = RaggedCSRWriter(_n_shared(store))
    for rows in edited:
        order = np.argsort(rows.variant_index, kind="stable")
        writer.add_analysis(*(np.asarray(column)[order] for column in rows))
    writer.flush(store, open_store(store).manifest.encoding, eaf_baseline=baseline)
    build_variant_index(store, n_axis=_n_shared(store))


def _without(analysis: str, shared_row: int):
    def edit(name: str, rows: AnalysisAssociations) -> AnalysisAssociations:
        if name != analysis:
            return rows
        keep = np.asarray(rows.variant_index) != shared_row
        return AnalysisAssociations(*(np.asarray(column)[keep] for column in rows))

    return edit


def _with(added: dict[str, list[tuple[int, dict[str, np.ndarray]]]]):
    def edit(name: str, rows: AnalysisAssociations) -> AnalysisAssociations:
        if name not in added:
            return rows
        shared = np.array([row for row, _ in added[name]], dtype=np.int32)
        values = [value for _, value in added[name]]
        return AnalysisAssociations(
            np.r_[rows.variant_index, shared],
            *(np.r_[getattr(rows, p), [v[p][0] for v in values]] for p in ("z", "se", "eaf")),
        )

    return edit


def _drop_an_overflow_association(store: Path) -> None:
    # trait_a's 1:100000, which is not a crossover.
    _rewrite_overflow(store, _without("trait_a", _shared_row(store, 100_000)))


def _add_an_overflow_association(store: Path) -> None:
    # trait_c gains 1:100000, which the source holds for trait_a and trait_b only.
    value = {"z": np.array([1.5]), "se": np.array([0.07]), "eaf": np.array([0.3])}
    _rewrite_overflow(store, _with({"trait_c": [(_shared_row(store, 100_000), value)]}))


def _keep_the_crossovers_in_the_overflow(store: Path) -> None:
    # Each crossover stays in the Overflow as well as being folded into the Dense Component.
    added: dict[str, list[tuple[int, dict[str, np.ndarray]]]] = {}
    for position, analysis in CROSSOVER_CELLS:
        value = _lookup(store, position, analysis)
        added.setdefault(analysis, []).append((_shared_row(store, position), value))
    _rewrite_overflow(store, _with(added))


def _move_an_overflow_observation_onto_an_existing_dense_row(store: Path) -> None:
    # trait_c's one Overflow association, 1:105000, leaves the Overflow and lands in
    # trait_c's cell at 1:11000, a row of the source Dense axis and not one the panel added.
    moved = _lookup(store, 105_000, "trait_c")
    _rewrite_overflow(store, _without("trait_c", _shared_row(store, 105_000)))
    rows, cols = _dense_cell(store, 11_000, "trait_c")
    _imputed(store)[int(rows[0]), int(cols[0])] = 0
    for cls, plane in ((DenseZPlane, "z"), (DenseEafPlane, "eaf"), (DenseSePlane, "se")):
        _dense_plane(store, cls).patch(rows, cols, moved[plane])


def test_rewriting_the_overflow_unchanged_keeps_the_pair_passing(tmp_path, pair):
    observed, completed = pair
    _rewrite_overflow(completed, lambda name, rows: rows)

    result = _run(observed, completed, tmp_path / "verdict.json")

    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    ("defect", "counts"),
    [
        (_drop_an_overflow_association, {
            "overflow.observed_missing_in_child": 1, "overflow.analyses_count_mismatch": 1,
        }),
        (_add_an_overflow_association, {
            "overflow.missing_became_observed": 1, "overflow.analyses_count_mismatch": 1,
        }),
        # trait_a then holds 6 Overflow associations where 4 belong, trait_b 4 where 3 do.
        (_keep_the_crossovers_in_the_overflow, {
            "overflow.observed_in_both": N_CROSSOVERS, "overflow.analyses_count_mismatch": 2,
        }),
    ],
    ids=lambda value: getattr(value, "__name__", ""),
)
def test_an_overflow_that_differs_from_the_sources_fails(tmp_path, pair, defect, counts):
    observed, completed = pair
    defect(completed)

    verdict = _failed_verdict(observed, completed, tmp_path / "verdict.json")

    assert {failure: _counter(verdict, failure) for failure in counts} == counts
    assert set(counts) <= set(verdict["failures"])


@pytest.mark.parametrize(
    ("defect", "failure", "count"),
    [
        (_drop_an_overflow_association, "overflow.analyses_count_mismatch", 1),
        (_keep_the_crossovers_in_the_overflow, "overflow.observed_in_both", N_CROSSOVERS),
        (_mark_an_overflow_association_imputed, "overflow.imputed_cells", 1),
    ],
    ids=lambda value: getattr(value, "__name__", value),
)
def test_an_overflow_defect_outside_the_sample_still_fails(
    tmp_path, pair, defect, failure, count
):
    observed, completed = pair
    defect(completed)

    verdict = _failed_verdict(
        observed, completed, tmp_path / "verdict.json", "--n-random", "0"
    )

    overflow = verdict["components"]["overflow"]
    assert overflow["analyses"] == [], "no Overflow Analysis may be sampled"
    assert not any(overflow["totals"].values())
    assert failure in verdict["failures"]
    assert _counter(verdict, failure) == count


def test_an_overflow_observation_moved_onto_an_existing_dense_row_fails(tmp_path, pair):
    observed, completed = pair
    assert _alid(11_000) in _dense_alids(observed), "the row must be a source Dense row"
    found = query_store(observed).lookup([_alid(11_000)], ["trait_c"])
    assert len(found["z"]) == 0, "trait_c must not observe 1:11000 in the source"
    _move_an_overflow_observation_onto_an_existing_dense_row(completed)

    verdict = _failed_verdict(observed, completed, tmp_path / "verdict.json")

    assert {failure: _counter(verdict, failure) for failure in (
        "dense.missing_became_observed", "dense.inserted_observed_cells",
        "overflow.observed_missing_in_child", "overflow.analyses_count_mismatch",
    )} == {
        "dense.missing_became_observed": 1, "dense.inserted_observed_cells": 0,
        "overflow.observed_missing_in_child": 1, "overflow.analyses_count_mismatch": 1,
    }


# --- seam 3: refusals --------------------------------------------------------


@pytest.mark.parametrize("dense_side", ["source", "target"])
def test_a_dense_and_hybrid_pair_is_refused(tmp_path, pair, dense_side):
    observed, completed = pair
    # Each Hybrid's Dense Component is itself a Dense Store in the matching state.
    source, target = (observed / "dense", completed) if dense_side == "source" else (
        observed, completed / "dense"
    )
    layouts = ("Hybrid", "dense") if dense_side == "source" else ("Dense", "hybrid")
    out = tmp_path / "verdict.json"

    result = _run(source, target, out)

    _assert_refused(result, out, "is not {} (primary_layout={})".format(*layouts))
    assert "a mixed-layout pair cannot be checked" in normalize_cli_output(result.output)


def test_a_hybrid_source_that_is_not_observed_only_is_refused(tmp_path, pair):
    _, completed = pair
    out = tmp_path / "verdict.json"

    _assert_refused(
        _run(completed, completed, out), out,
        "is not Observed-Only (completion_state=reference_completed)",
    )


def test_a_hybrid_target_that_is_not_reference_completed_is_refused(tmp_path, pair):
    observed, _ = pair
    out = tmp_path / "verdict.json"

    _assert_refused(
        _run(observed, observed, out), out,
        "is not Reference-Completed (completion_state=observed_only)",
    )


# --- seam 4: the sample ------------------------------------------------------

# The completion writer's bands start at child rows 4, 8 and 12; the chunk-rows owning
# the child rows either side of each are 1|2, 3|3 and 5|5. One contig: its edges are
# chunk-rows 0 and 5.
_STRATA_CHOSEN = {
    0: ["chrom_edge"],
    1: ["writer_band_boundary"],
    2: ["writer_band_boundary"],
    3: ["writer_band_boundary"],
    5: ["chrom_edge", "final_partial", "writer_band_boundary"],
}


def test_the_dense_strata_come_from_the_components_own_geometry(monkeypatch, _pair_seed):
    observed, completed = _pair_seed
    # The writer's band is whole shards up to its memory target, which the check reads
    # with the shard height; the target must be the one the pair was completed under.
    monkeypatch.setattr(dense_complete, "_BAND_ROWS", 6)

    verdict = check_observed_view(observed, completed, seed=1, n_random=0).record

    dense = verdict["components"]["dense"]
    assert dense["geometry"] == {
        "parent_chunk_rows": 2, "n_parent_chunk_rows": 6, "completion_band_rows": 4,
    }
    assert {entry["band"]: entry["strata"] for entry in dense["bands"]} == _STRATA_CHOSEN
    # Five chunk-rows of two source rows. Their spans, [0, 2) [2, 4) [4, 6) [6, 9) and
    # [11, 15), hold all three inserted rows and so every crossover cell.
    assert dense["totals"]["cells_compared"] == 5 * 2 * N_ANALYSES
    assert dense["totals"]["inserted_rows"] == N_INSERTED
    assert dense["totals"]["crossover_cells"] == N_CROSSOVERS
    assert verdict["components"]["overflow"]["analyses_checked"] == 0
    assert verdict["crossovers"]["n"] == N_CROSSOVERS  # every one, whatever the sample
    assert verdict["passed"], verdict["failures"]


def test_the_same_seed_gives_the_same_verdict(_pair_seed):
    observed, completed = _pair_seed

    first = check_observed_view(observed, completed, seed=3, n_random=2).record
    again = check_observed_view(observed, completed, seed=3, n_random=2).record

    for record in (first, again):
        record.pop("seconds")
    assert first == again
    assert first["components"]["dense"]["bands_by_stratum"]["random"] == 2
    assert first["components"]["overflow"]["analyses_checked"] == 2
    assert first["passed"], first["failures"]
