"""Reference Completion and an SE above float16's range (issue #338, ADR 0068).

A completed release holds no association whose SE is above 65,504: a source
cell holding one -- only a release before format 0.2.3 does -- is not carried,
a fill with one is left missing, and each Analysis's `n_missing_se_out_of_range`
is its source's count carried forward plus both, which the completion records
apart (`provenance.completion.se_out_of_range`). The Observed-View Check holds
the count to exactly that.

A release before 0.2.3 is built here by this package with the gate disabled
(`_as_before_0_2_3`): `storable` keeps every cell and the encoders narrow as
they did, so a `float16` plane holds `+inf` and a residual one an exact
exception, consistently across every array that holds the cell. It is then
stamped 0.2.2 and its count column removed, as such a release has none.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import test_dense_completion as tdc
import test_hybrid_completion as thc
import test_ragged_completion as trc
from dense_sources import build_dense_fixture
from format_0_2_3 import follow_stamp
from stored_rows import stored_rows
from test_check_observed_view import HEADER, ROWS

import opengwasdb.completion.block as block_module
from opengwasdb.encoding import association_codec, codec, se
from opengwasdb.encoding.association_codec import CellValues, Storable
from opengwasdb.encoding.sd_units import PhenotypeSd
from opengwasdb.layouts.dense import build_vcf
from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.layouts.hybrid import build as hybrid_build
from opengwasdb.layouts.hybrid.complete import complete_hybrid_store
from opengwasdb.layouts.ragged import zarr_csr
from opengwasdb.layouts.ragged.complete import complete_ragged_store
from opengwasdb.model.analyses import (
    N_MISSING_SE_OUT_OF_RANGE,
    read_analysis_records,
    write_analyses,
)
from opengwasdb.store.open import open_store
from opengwasdb.store_reader.analyses import AnalysesTable, read_analyses
from opengwasdb.validation import validate_store
from opengwasdb.validation.observed_view import check_observed_view

#: An SE above the bound; `float16` narrows it to `+inf`.
HUGE = 100_000.0


# ── a release written before format 0.2.3 ───────────────────────────────────


@contextmanager
def _as_before_0_2_3(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Build as this package did before issue #338: every cell stored, an SE above
    the bound narrowed by the plan, and nothing counted."""
    real = association_codec.storable

    def kept_whole(
        cells: CellValues, analyses, n_analyses: int, *, where: str, phenotype_sd: PhenotypeSd
    ) -> Storable:
        owner = np.broadcast_to(np.asarray(analyses, dtype=np.int64), np.shape(cells.z))
        se_values = phenotype_sd.standardised(np.asarray(cells.se), owner, n_analyses)
        cells = CellValues(cells.z, se_values, cells.eaf, cells.imputed)
        with np.errstate(invalid="ignore"):
            stand_in = np.where(se_values > 65504.0, np.float32(1.0), se_values)
        judged = real(  # Admission's other rules still hold of every cell
            CellValues(cells.z, stand_in, cells.eaf, cells.imputed), analyses, n_analyses,
            where=where, phenotype_sd=PhenotypeSd.APPLIED,
        )
        none = np.zeros(n_analyses, dtype=np.int64)
        return Storable(cells, np.ones(np.shape(cells.z), dtype=bool) & judged.stored, none, none)

    with monkeypatch.context() as patched:
        # The Dense build reaches the gate through `association_codec` itself.
        for module in (association_codec, build_vcf, hybrid_build, zarr_csr):
            patched.setattr(module, "storable", kept_whole)
        for module in (codec, se):
            patched.setattr(module, "require_se_in_range", lambda *_a, **_k: None)
        # Nor did a residual encoder hold a code reading above the bound exactly.
        patched.setattr(codec, "se_above_bound", lambda v: np.zeros(np.shape(v), dtype=bool))
        yield


def _stamp_0_2_2(store: Path) -> None:
    """Stamp a release (and a Hybrid's Dense Component) 0.2.2, without the count."""
    for release in (store, store / "dense"):
        manifest_path = release / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_path.write_text(
            json.dumps({**manifest, "format_version": "0.2.2"}), encoding="utf-8"
        )
        follow_stamp(release)
        table = read_analyses(release / "analyses.tsv")
        fields = tuple(f for f in table.fieldnames if f != N_MISSING_SE_OUT_OF_RANGE)
        rows = tuple({k: v for k, v in row.items() if k in fields} for row in table.rows)
        write_analyses(release / "analyses.tsv", AnalysesTable(fields, rows))


def _recorded(store: Path) -> list[str]:
    return [a.n_missing_se_out_of_range for a in read_analysis_records(store / "analyses.tsv")]


def _set_recorded(store: Path, counts: list[str]) -> None:
    table = read_analyses(store / "analyses.tsv")
    rows = tuple(
        {**row, N_MISSING_SE_OUT_OF_RANGE: count}
        for row, count in zip(table.rows, counts, strict=True)
    )
    write_analyses(store / "analyses.tsv", AnalysesTable(table.fieldnames, rows))


def _parts(store: Path) -> dict:
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    return manifest["provenance"]["completion"]["se_out_of_range"]


def _set_parts(store: Path, parts: dict | None) -> None:
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if parts is None:
        del manifest["provenance"]["completion"]["se_out_of_range"]
    else:
        manifest["provenance"]["completion"]["se_out_of_range"] = parts
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _quality(store: Path) -> tuple[int, int]:
    with sqlite3.connect(store / "index.sqlite") as db:
        return db.execute(
            "SELECT COALESCE(SUM(n_imputed), 0), COALESCE(SUM(n_missing), 0) "
            "FROM completion_quality"
        ).fetchone()


def _validates_cleanly(store: Path) -> None:
    result = validate_store(store)
    assert result.ok, result.errors
    assert not [w for w in result.warnings if "infinite" in w.lower()], result.warnings


# ── Dense: a source cell above the bound ─────────────────────────────────────

#: a1's association at 1:1000000, which the source carries with SE 100,000.
OVER_CELL = ("1:1000000:A:G", 0)


@pytest.fixture(scope="module")
def dense_pair(tmp_path_factory) -> tuple[Path, Path]:
    """A 0.2.2 Dense source holding `+inf` at a1's 1:1000000, and its completion."""
    root = tmp_path_factory.mktemp("se-out-of-range-dense")
    fields = [row.split("\t") for row in ROWS]
    assert (fields[2][0], fields[2][5]) == ("a1", "1000000")
    fields[2][9] = str(HUGE)
    rows = root / "associations.tsv"
    rows.write_text(HEADER + "\n" + "\n".join("\t".join(f) for f in fields) + "\n")
    observed = root / "obs.opengwasdb"
    with pytest.MonkeyPatch.context() as monkeypatch, _as_before_0_2_3(monkeypatch):
        build_dense_fixture(
            [rows], observed, store_id="t", release_id="obs", reference_assembly="GRCh38"
        )
    _stamp_0_2_2(observed)
    completed = root / "comp.opengwasdb"
    complete_dense_store(
        observed, completed, tdc._make_ld_panel(root), ancestry="EUR", min_cor=0.0,
        release_id="comp",
    )
    return observed, completed


def _dense_cell(store: Path, alid: str, analysis: int) -> tuple[float, float, int]:
    from opengwasdb.store_reader.encoding.planes import DenseSePlane, DenseZPlane

    opened = open_store(store)
    axis = opened.variant_axis(aliases=False)  # a 0.3.0 Component's is the shared one
    try:
        row = axis.by_identifier(alid).variant_index
    finally:
        axis.close()
    root, encoding = opened.arrays(mode="r"), opened.manifest.encoding
    rows, cols = np.array([row]), np.array([analysis])
    imputed = int(root["imputed"][row, analysis]) if "imputed" in root else 0
    return (
        float(DenseZPlane.open(root, encoding).points(rows, cols)[0]),
        float(DenseSePlane.open(root, encoding).points(rows, cols)[0]),
        imputed,
    )


def _check(observed: Path, completed: Path, **extra) -> dict:
    return check_observed_view(observed, completed, seed=7, **extra).record


def test_a_dense_completion_leaves_out_a_source_cell_above_the_bound(dense_pair) -> None:
    observed, completed = dense_pair
    # The fixture is meaningful first: the source holds the cell, as +inf beside a z.
    z, se_value, _ = _dense_cell(observed, *OVER_CELL)
    assert np.isfinite(z) and np.isinf(se_value)
    assert open_store(observed).manifest.format_version == "0.2.2"
    assert _recorded(observed) == ["", ""]

    z, se_value, imputed = _dense_cell(completed, *OVER_CELL)
    assert np.isnan(z) and np.isnan(se_value) and imputed == 0
    assert open_store(completed).manifest.format_version == "0.3.0"
    assert _recorded(completed) == ["1", "0"]
    assert _parts(completed) == {"observed": [1, 0], "imputed": [0, 0]}
    _validates_cleanly(completed)


@pytest.mark.parametrize(
    ("observed", "imputed"), [([2], [-1]), ([-1], [2]), ([1], [-1])],
    ids=["negative-imputed", "negative-observed", "zero-sum"],
)
def test_each_recorded_part_is_a_non_negative_count(observed, imputed) -> None:
    """Each part is judged on its own, not through the sum (spec §7a)."""
    from opengwasdb.completion.se_out_of_range import LeftOut

    block = {"se_out_of_range": {"observed": observed, "imputed": imputed}}
    with pytest.raises(ValueError, match="non-negative"):
        LeftOut.from_provenance(block, 1)


def test_a_source_before_0_2_3_has_an_unknown_count_never_0() -> None:
    """Its completion counts what it meets instead (spec §7a): the source's own
    count is unknown, so nothing reads it, and no 0 stands in for it."""
    from opengwasdb.completion.se_out_of_range import LeftOut, carried_counts
    from opengwasdb.model.analyses import Analysis

    analyses = [Analysis("a1"), Analysis("a2")]
    assert carried_counts(analyses, "0.2.2", source="s") is None
    met = LeftOut.of([1, 0], [0, 2])
    assert [a.n_missing_se_out_of_range for a in met.recorded(analyses, None)] == ["1", "2"]
    counted = met.recorded(analyses, None)
    assert carried_counts(counted, "0.2.3", source="s").tolist() == [1, 2]


def test_the_observed_view_check_accounts_for_it_exactly(dense_pair) -> None:
    record = _check(*dense_pair)
    assert record["passed"], record["failures"]
    assert record["se_out_of_range"]["source_counts"] is None, "unknown, not 0"
    assert record["totals"]["se_out_of_range_left_out"] == 1
    assert record["se_out_of_range"]["found"] == {"a1": 1}
    assert record["se_out_of_range"]["every_source_cell_compared"] is True


@pytest.mark.parametrize(
    ("counts", "parts"),
    [
        (["999999", "0"], None),  # an over-count
        (["0", "0"], None),  # an under-count
        (["", "0"], None),  # blank: unknown, never read as 0
        (["1", "0"], "absent"),  # the parts unrecorded
        (["2", "0"], {"observed": [2, 0], "imputed": [0, 0]}),  # more than the cells found
        (["1", "0"], {"observed": [0, 0], "imputed": [1, 0]}),  # a cell found, none recorded
        # A negative part cancels the cell found, so the sum alone looks right.
        (["0", "0"], {"observed": [1, 0], "imputed": [-1, 0]}),
        (["1", "0"], {"observed": [1, -1], "imputed": [0, 1]}),
        (["1", "0"], {"observed": [1, 0], "imputed": [0.0, 0]}),  # not an integer
    ],
    ids=[
        "over", "under", "blank", "unrecorded", "observed-over", "observed-under",
        "negative-imputed", "negative-observed", "float-part",
    ],
)
def test_the_check_fails_any_count_but_the_exact_one(tmp_path, dense_pair, counts, parts) -> None:
    import shutil

    observed, completed = dense_pair
    edited = shutil.copytree(completed, tmp_path / "edited.opengwasdb")
    _set_recorded(edited, counts)
    if parts is not None:
        _set_parts(edited, None if parts == "absent" else parts)

    record = _check(observed, edited)

    assert not record["passed"]
    assert set(record["failures"]) <= {
        "se_out_of_range_miscounted", "se_out_of_range_unaccounted"
    }, record["failures"]


def test_a_completion_before_0_2_3_must_leave_nothing_out(tmp_path, dense_pair) -> None:
    """A target stamped before 0.2.3 was written by a completion that carried every
    cell, so a source cell it lacks is unaccounted, not left out."""
    import shutil

    observed, completed = dense_pair
    edited = shutil.copytree(completed, tmp_path / "edited.opengwasdb")
    manifest = json.loads((edited / "manifest.json").read_text(encoding="utf-8"))
    (edited / "manifest.json").write_text(json.dumps({**manifest, "format_version": "0.2.2"}))
    follow_stamp(edited)

    record = _check(observed, edited)

    assert "observed_missing_in_child" in record["failures"], record["failures"]


# ── Dense: a fill above the bound ────────────────────────────────────────────


def _oversized_fills(monkeypatch: pytest.MonkeyPatch) -> None:
    """An imputed SE above the bound needs an unrealistic panel, so the imputation's
    SE is scaled at its seam, `completion.block.scalar_n_se`."""
    real = block_module.scalar_n_se
    monkeypatch.setattr(
        block_module, "scalar_n_se", lambda *a, **k: np.asarray(real(*a, **k)) * 1e9
    )


def test_the_fill_merge_holds_what_storable_returns(monkeypatch) -> None:
    """The merge leaves missing exactly the fills `storable` leaves out, by its
    mask and counts: no second gate (ADR 0077). A gate that also leaves out the
    fill with SE 0.5 shows whose judgement the merge holds."""
    from opengwasdb.completion import fill_shards

    real = association_codec.storable
    judged: list[str] = []

    def gate(
        cells: CellValues, analyses, n_analyses: int, *, where: str, phenotype_sd: PhenotypeSd
    ) -> Storable:
        judged.append(where)
        kept = real(cells, analyses, n_analyses, where=where, phenotype_sd=phenotype_sd)
        also = np.asarray(cells.se) == np.float32(0.5)
        owners = np.broadcast_to(np.asarray(analyses), also.shape)[also]
        extra = np.bincount(owners, minlength=n_analyses).astype(np.int64)
        return Storable(
            kept.cells, kept.stored & ~also, kept.se_out_of_range + extra,
            kept.imputed_out_of_range + extra,
        )

    monkeypatch.setattr(association_codec, "storable", gate)
    fills = (
        np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1]),
        np.array([1.0, 2.0, 3.0, 4.0], np.float32), np.array([0.1, HUGE, 0.5, 0.2], np.float32),
    )
    quality = [(0, "b", None, 2, 0), (1, "b", None, 2, 0)]

    kept, adjusted, left_out = fill_shards._without_se_out_of_range(fills, quality, Path("c"))

    assert judged, "the merge asks storable"
    assert kept is not None and kept[0].tolist() == [0, 3]
    assert left_out == {0: 1, 1: 1}
    assert adjusted == [(0, "b", None, 1, 1), (1, "b", None, 1, 1)]


def _signal_source(tmp_path: Path) -> Path:
    """`test_dense_completion`'s AR(1) source: one Analysis observing every position
    of one block but the held-out one."""
    rows = [tdc.SOURCE_HEADER]
    for i, (pos, z) in enumerate(zip(tdc._SIGNAL_POSITIONS, tdc._SIGNAL_Z_TRUE, strict=True)):
        if i != tdc._SIGNAL_MISSING_IDX:
            rows.append(f"a1\tp1\tHeight\tHeight primary\t1\t{pos}\tA\tG\t{z:.6f}\t0.2\trs{i}\tsd")
    path = tmp_path / "signal_associations.tsv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    out = tmp_path / "signal_obs.opengwasdb"
    build_dense_fixture([path], out, store_id="t", release_id="obs", reference_assembly="GRCh38")
    return out


def test_a_dense_fill_above_the_bound_is_left_missing_and_counted(tmp_path, monkeypatch) -> None:
    """One Analysis, one held-out position: the baseline imputes exactly that cell."""
    source = _signal_source(tmp_path)
    panel = tdc.write_signal_panel(tmp_path / "signal_panel")

    def complete(name: str) -> Path:
        out = tmp_path / name
        complete_dense_store(source, out, panel, ancestry="EUR", min_cor=0.0, release_id="c")
        return out

    baseline = complete("baseline.opengwasdb")
    assert int(np.asarray(open_store(baseline).arrays(mode="r")["imputed"][:]).sum()) == 1
    assert _quality(baseline)[0] == 1 and _recorded(baseline) == ["0"]

    _set_recorded(source, ["3"])  # what the source's build recorded, carried forward
    _oversized_fills(monkeypatch)
    completed = complete("oversized.opengwasdb")

    assert int(np.asarray(open_store(completed).arrays(mode="r")["imputed"][:]).sum()) == 0
    assert _recorded(completed) == ["4"]
    assert _parts(completed) == {"observed": [0], "imputed": [1]}
    assert _quality(completed) == (0, _quality(baseline)[1] + 1)
    assert read_analyses(completed / "analyses.tsv").rows[0]["completion_n_imputed_total"] == "0"
    _validates_cleanly(completed)
    record = _check(source, completed)
    assert record["passed"], record["failures"]


# ── Ragged ───────────────────────────────────────────────────────────────────


def _besd_source(tmp_path: Path, probe_assocs, **build) -> Path:
    prefix = trc._make_besd_fixture(tmp_path, freqs=(0.3, 0.45, 0.2), probe_assocs=probe_assocs)
    observed = tmp_path / "obs.opengwasdb"
    trc.build_ragged_from_besd(
        prefix, observed, store_id="t", release_id="obs", tissue="Blood",
        association_coverage="cis", **build,
    )
    return observed


def _complete_ragged(observed: Path, tmp_path: Path, name: str) -> tuple[Path, Path]:
    panel = trc._make_ld_panel(tmp_path, "1", 900_000, 1_300_000)
    out = tmp_path / name
    complete_ragged_store(
        observed, out, panel, ancestry="EUR", cis_window_bp=500_000, min_cor=0.0,
        release_id="comp",
    )
    return out, panel


def test_a_ragged_completion_leaves_out_a_source_row_above_the_bound(tmp_path, monkeypatch):
    """The first probe's rs1002 carries SE 100,000 in a 0.2.2 source."""
    assocs = [[(0, 0.1, 0.02), (1, -0.2, HUGE)], [(1, 0.5, 0.05), (2, -0.15, 0.025)]]
    with _as_before_0_2_3(monkeypatch):
        observed = _besd_source(tmp_path, assocs)
    _stamp_0_2_2(observed)
    first = stored_rows(observed, 0)
    assert len(first.variant_index) == 2 and np.isinf(first.se).sum() == 1, "fixture: +inf held"

    completed, panel = _complete_ragged(observed, tmp_path, "comp.opengwasdb")

    held = stored_rows(completed, 0)
    assert not np.isinf(held.se).any()
    assert held.se[~np.isnan(held.se)].tolist().count(np.float32(0.02)) <= 1
    assert _recorded(completed) == ["1", "0"]
    assert _parts(completed) == {"observed": [1, 0], "imputed": [0, 0]}
    _validates_cleanly(completed)
    record = _check(observed, completed, ld_panel=panel, ancestry="EUR")
    assert record["passed"], record["failures"]
    assert record["se_out_of_range"]["found"] == {"ENSG00000000001::Blood": 1}


def test_a_ragged_fill_above_the_bound_is_left_missing_and_counted(tmp_path, monkeypatch):
    """The Ragged Observed-View fixture: the panel-only SNP is imputed for `eur_rich`
    and `eur_traitless`, the two Analyses observing five block SNPs."""
    from test_observed_view_ragged import SOURCES, _build_source

    observed = _build_source(tmp_path / "src", SOURCES)
    panel = trc._make_rich_ld_panel(tmp_path)

    def complete(name: str) -> Path:
        out = tmp_path / name
        complete_ragged_store(
            observed, out, panel, ancestry="EUR", cis_window_bp=500_000, min_cor=0.0,
            release_id="comp",
        )
        return out

    baseline = complete("baseline.opengwasdb")
    flags = np.asarray(open_store(baseline).arrays(mode="r")["ragged"]["imputed"][:])
    assert int(flags.sum()) == 2, "fixture: one imputed row in each of two Analyses"
    assert _recorded(baseline) == ["0", "0", "0", "0"] and _quality(baseline)[0] == 2

    _oversized_fills(monkeypatch)
    completed = complete("oversized.opengwasdb")

    group = open_store(completed).arrays(mode="r")["ragged"]
    assert int(np.asarray(group["imputed"][:]).sum()) == 0
    assert not np.isinf(np.asarray(group["se"][:], dtype=np.float32)).any()
    assert _recorded(completed) == ["1", "0", "1", "0"]
    assert _parts(completed) == {"observed": [0, 0, 0, 0], "imputed": [1, 0, 1, 0]}
    assert _quality(completed) == (0, _quality(baseline)[1] + 2)
    _validates_cleanly(completed)
    record = _check(observed, completed, ld_panel=panel, ancestry="EUR")
    assert record["passed"], record["failures"]


def test_a_residual_ragged_analysis_left_with_one_se_completes_against_its_sources_model(
    tmp_path, monkeypatch
):
    """The refit gap ADR 0077 recorded, which ADR 0078 closed (issue #301).

    The first probe's two SEs fit the source's residual model; leaving out the one
    above the bound leaves it one. Completion fits nothing -- it codes every cell
    against its source's `se_coefficients` -- so it neither fails nor falls back to
    float16: the model is the source's bit for bit, and the SE it keeps decodes as
    the source's does.
    """
    assocs = [
        [(0, 0.1, 0.02), (1, -0.2, HUGE)],
        [(0, 0.3, 0.04), (1, 0.5, 0.05), (2, -0.15, 0.025)],
    ]
    with _as_before_0_2_3(monkeypatch):
        observed = _besd_source(tmp_path, assocs, store_esi_eaf=True, allow_unverified_eaf=True)
    _stamp_0_2_2(observed)
    source = open_store(observed)
    assert source.manifest.encoding.se.kind == "int8_residual", "fixture: a residual plan"
    first = stored_rows(observed, 0)
    assert len(first.se) == 2 and float(first.se[1]) == pytest.approx(HUGE, rel=1e-5), (
        "fixture: the exception held"
    )

    completed, _ = _complete_ragged(observed, tmp_path, "comp.opengwasdb")

    target = open_store(completed)
    assert target.manifest.encoding.se == source.manifest.encoding.se
    coefficients = [
        np.asarray(store.arrays(mode="r")["ragged"]["se_coefficients"][:])
        for store in (source, target)
    ]
    assert np.array_equal(*coefficients)
    held = stored_rows(completed, 0)
    observed_rows = held.flags == 0 if held.flags is not None else np.ones(len(held.se), bool)
    assert held.se[observed_rows & ~np.isnan(held.se)].tolist() == [float(first.se[0])]
    assert _recorded(completed) == ["1", "0"]
    _validates_cleanly(completed)


# ── Hybrid ───────────────────────────────────────────────────────────────────


def test_a_hybrid_completion_counts_what_each_component_left_out(tmp_path, monkeypatch):
    """trait_a's hg19 2,000,000 is an Overflow row the LD panel makes a crossover, and
    its 1,200,000 one that stays off the panel; a 0.2.2 source holds both with SE
    100,000. Neither is carried: the crossover is not folded, so its Dense cell is
    Dense completion's, and the Overflow row is dropped. The Dense Component's
    completion records the crossover, which would have been its cell, in its parts;
    the root counts both (issue #338, ADR 0068 defect 1, #472)."""
    with _as_before_0_2_3(monkeypatch):
        src = thc.build_two_trait_source(
            tmp_path,
            {
                "trait_a": [
                    (100_000, "2.0:0.5"), (1_000_000, "1.5:0.3"), (1_200_000, f"1.0:{HUGE}"),
                    (2_000_000, f"1.2:{HUGE}"),
                ],
                "trait_c": [(100_000, "2.2:0.5"), (1_000_000, "1.6:0.3")],
            },
        )
    _stamp_0_2_2(src)
    overflow = open_store(src).arrays(mode="r")["ragged"]
    held = np.asarray(overflow["se"][:], dtype=np.float32)
    assert np.isinf(held).sum() == 2 and int(overflow["offsets"][1]) == 2, (
        "fixture: trait_a's two Overflow rows hold +inf"
    )

    dst = tmp_path / "dst.opengwasdb"
    complete_hybrid_store(
        src, dst, thc._make_ld_panel_with_crossover(tmp_path), min_cor=0.0, thresh=0.9
    )

    # The completed Dense Component counts the crossover in its completion's parts; at
    # 0.3.0 it has no analyses.tsv of its own, and the root counts both (#472).
    assert not (dst / "dense" / "analyses.tsv").exists()
    assert _recorded(dst) == ["2", "0"]
    assert _parts(dst / "dense") == {"observed": [1, 0], "imputed": [0, 0]}
    assert _parts(dst) == {"observed": [2, 0], "imputed": [0, 0]}
    z, se_value, imputed = _dense_cell(dst / "dense", thc.OFF_PANEL_ALID, 0)
    assert not np.isinf(se_value)
    assert imputed == 1 or (np.isnan(z) and np.isnan(se_value)), "Dense completion's own"
    assert int(open_store(dst).arrays(mode="r")["ragged"]["offsets"][1]) == 0
    _validates_cleanly(dst)
    record = _check(src, dst)
    assert record["passed"], record["failures"]
    assert record["se_out_of_range"]["found"] == {"trait_a": 2}


# ── the converter ────────────────────────────────────────────────────────────


def test_the_converter_refuses_a_dense_source_holding_one_naming_the_analysis(
    tmp_path, monkeypatch
) -> None:
    from legacy_fixtures import relayout_as_0_1_0

    from opengwasdb.store.convert import ConversionError, convert_release

    fields = [row.split("\t") for row in ROWS]
    fields[2][9] = str(HUGE)  # a1 at 1:1000000
    rows = tmp_path / "associations.tsv"
    rows.write_text(HEADER + "\n" + "\n".join("\t".join(f) for f in fields) + "\n")
    built = tmp_path / "built.opengwasdb"
    with _as_before_0_2_3(monkeypatch):
        build_dense_fixture(
            [rows], built, store_id="t", release_id="r", reference_assembly="GRCh38"
        )
    source = relayout_as_0_1_0(built, tmp_path / "source.opengwasdb")
    assert json.loads((source / "manifest.json").read_text())["format_version"] == "0.1.0"
    assert np.isinf(np.asarray(open_store(source).arrays(mode="r")["se"][:])).sum() == 1

    destination = tmp_path / "converted.opengwasdb"
    with pytest.raises(ConversionError, match=r"in Analyses a1: 1 \(arrays data\.zarr/se: 1"):
        convert_release(source, destination)
    assert not destination.exists()


def test_the_converter_refuses_a_ragged_source_holding_one_naming_the_analysis(
    tmp_path, monkeypatch
) -> None:
    from legacy_fixtures import relayout_as_0_1_0

    from opengwasdb.store.convert import ConversionError, convert_release

    assocs = [[(0, 0.1, 0.02), (1, -0.2, HUGE)], [(1, 0.5, 0.05), (2, -0.15, 0.025)]]
    with _as_before_0_2_3(monkeypatch):
        built = _besd_source(tmp_path, assocs)
    source = relayout_as_0_1_0(built, tmp_path / "source.opengwasdb")

    destination = tmp_path / "converted.opengwasdb"
    with pytest.raises(
        ConversionError,
        match=r"in Analyses ENSG00000000001::Blood: 1 \(arrays .*data\.zarr/ragged/se: 1",
    ):
        convert_release(source, destination)
    assert not destination.exists()
