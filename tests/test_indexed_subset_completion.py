"""Indexed Variant Subsets on Reference-Completed Dense stores (issue #266).

Issue #264 built the artifact for Observed-Only Dense releases and #265 wired
the selected-Analysis query to it.  This module pins the Reference-Completed
half: the index must carry the per-cell imputed mask and the subset reference
EAF, decode an imputed cell to the panel's frequency while leaving an observed
cell whose source reported none absent, and reproduce the ordinary
selected-Analysis result field for field -- including ``association_status``
and ``observed_only``.

Two fixtures reach the two EAF shapes a completed Dense release can have:

* ``completed_store`` carries an ``eaf`` plane (its source reported
  frequencies) *and* reference EAF: observed cells read the cohort's value,
  an observed cell with no cohort value stays NaN, and the one imputed cell
  reads the panel's.
* ``reference_only_completed_store`` carries reference EAF with **no** ``eaf``
  plane at all (issue #113): every observed cell is NaN and only the imputed
  cell reads a frequency.  A reader that skipped the reference on the array's
  absence would return NaN there -- plausible, wrong, and exactly the class of
  defect this project refuses.

Each fixture asserts it reaches the states below before anything is asserted
about the index (CONTRIBUTING.md, "A test that cannot fail is worse than no
test").
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from test_dense_completion import (
    _SIGNAL_N,
    _SIGNAL_POSITIONS,
    _SIGNAL_Z_TRUE,
    write_signal_panel,
)
from test_indexed_subset_query import _assert_results_equal, _invoke, _ordinary_filtered
from test_indexed_subsets import (
    INDEXED_SUBSETS_GROUP,
    _authoritative_digest,
    _digest,
    _index_group,
)

from opengwasdb.build.source import NormalisedAssociation
from opengwasdb.layouts.dense.build import build_dense_observed_store
from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.layouts.dense.indexed_subsets import (
    IndexedSubsetError,
    build_indexed_subset,
    list_indexed_subsets,
    open_indexed_subset,
)
from opengwasdb.model.analyses import read_analyses
from opengwasdb.query import query_store
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store
from opengwasdb.variants import CanonicalVariant, VariantAxis

#: The imputation target, left out of a1's source rows and held by the panel.
IMPUTED = 6
#: An observed cell whose source reported no frequency; the panel holds one.
NO_COHORT_EAF = 3
#: A canonical ALID the Store Variant Table does not carry.
ABSENT_ALID = "1:9999999:A:G"


def alid(variant_index: int) -> str:
    return f"1:{_SIGNAL_POSITIONS[variant_index]}:A:G"


def _record(
    analysis_id: str, variant_index: int, z: float, eaf: float | None
) -> NormalisedAssociation:
    return NormalisedAssociation(
        analysis_id,
        CanonicalVariant("1", _SIGNAL_POSITIONS[variant_index], "A", "G"),
        z=z,
        se=0.2,
        eaf=eaf,
    )


def _source_with_cohort_eaf(store: Path) -> Path:
    """a1 observes every position but the target; a2 observes the target, not completed.

    a1's cell at ``NO_COHORT_EAF`` carries no cohort frequency.  a2 is left
    observed-only by ``impute_analysis_ids={'a1'}`` below, so the fixture spans
    a completed Analysis and an unmatched one.
    """
    records: list[NormalisedAssociation] = []
    for i, z in enumerate(_SIGNAL_Z_TRUE):
        if i == IMPUTED:
            continue
        records.append(_record("a1", i, float(z), None if i == NO_COHORT_EAF else 0.2))
    records.append(_record("a2", IMPUTED, 2.0, 0.25))
    records.append(_record("a2", 0, 1.0, 0.15))
    build_dense_observed_store(
        records,
        store,
        store_id="completion-fixture",
        release_id="observed-v1",
        reference_assembly="GRCh38",
    )
    return store


def _source_without_cohort_eaf(store: Path) -> Path:
    """a1 observes every position but the target and reports no frequency at all."""
    records = [
        _record("a1", i, float(z), None)
        for i, z in enumerate(_SIGNAL_Z_TRUE)
        if i != IMPUTED
    ]
    build_dense_observed_store(
        records,
        store,
        store_id="reference-only-fixture",
        release_id="observed-v1",
        reference_assembly="GRCh38",
    )
    return store


@pytest.fixture
def signal_panel_npz_only(tmp_path: Path) -> Path:
    """The AR(1) panel `test_dense_completion` uses, built in this test's tmp dir."""
    return write_signal_panel(tmp_path / "signal_panel")


@pytest.fixture
def cohort_eaf_store(tmp_path: Path) -> Path:
    return _source_with_cohort_eaf(tmp_path / "cohort-obs.opengwasdb")


@pytest.fixture
def reference_only_store(tmp_path: Path) -> Path:
    return _source_without_cohort_eaf(tmp_path / "reference-only-obs.opengwasdb")


@pytest.fixture
def completed_store(
    tmp_path: Path,
    cohort_eaf_store: Path,
    signal_panel_npz_only: Path,
) -> Path:
    """A completed release whose source reported frequencies (plus reference EAF)."""
    dst = tmp_path / "cohort-comp.opengwasdb"
    complete_dense_store(
        cohort_eaf_store,
        dst,
        signal_panel_npz_only,
        ancestry="EUR",
        min_cor=0.0,
        release_id="comp-v1",
        impute_analysis_ids={"a1"},
    )
    return dst


@pytest.fixture
def reference_only_completed_store(
    tmp_path: Path,
    reference_only_store: Path,
    signal_panel_npz_only: Path,
) -> Path:
    """A completed release whose source reported no frequency: no `eaf` plane."""
    dst = tmp_path / "reference-only-comp.opengwasdb"
    complete_dense_store(
        reference_only_store,
        dst,
        signal_panel_npz_only,
        ancestry="EUR",
        min_cor=0.0,
        release_id="comp-v1",
    )
    return dst


def _write_alid_list(tmp_path: Path, name: str = "hm3.alid.txt") -> Path:
    path = tmp_path / name
    path.write_text(
        "\n".join([*(alid(i) for i in range(_SIGNAL_N)), ABSENT_ALID]) + "\n",
        encoding="utf-8",
    )
    return path


def _build_subset(
    store: Path, tmp_path: Path, name: str = "hm3", *, overwrite: bool = False
) -> Path:
    build_indexed_subset(
        store,
        name,
        _write_alid_list(tmp_path),
        reference_assembly="GRCh38",
        overwrite=overwrite,
    )
    return store


def _analyse(store: Path, analysis_id: str, **kwargs: object) -> dict[str, np.ndarray]:
    with query_store(store) as query:
        return query.analysis(analysis_id, **kwargs)


def _variant_index_of(store: Path, variant: int) -> int:
    axis = VariantAxis(store)
    try:
        record = axis.by_identifier(alid(variant))
    finally:
        axis.close()
    assert record is not None, f"fixture must hold {alid(variant)}"
    return int(record.variant_index)


# ── Fixture is meaningful ───────────────────────────────────────────────────


def test_cohort_eaf_fixture_reaches_every_completed_state(completed_store: Path) -> None:
    manifest = open_store(completed_store).manifest
    assert manifest.completion_state.value == "reference_completed"
    assert manifest.encoding.eaf.reference, "the fixture must declare reference EAF"
    root = open_store(completed_store).arrays(mode="r")
    assert "imputed" in root, "the fixture must carry the per-cell status mask"
    assert "eaf_reference" in root, "the fixture must carry panel frequencies"
    assert int(root["imputed"][:].sum()) == 1, "the fixture must hold exactly one imputed cell"
    assert bool(root["imputed"][_variant_index_of(completed_store, IMPUTED), 0]) is True
    assert bool(root["imputed"][_variant_index_of(completed_store, 0), 0]) is False
    reference = root["eaf_reference"][:]
    assert np.isfinite(reference[_variant_index_of(completed_store, IMPUTED)])
    assert np.isfinite(reference[_variant_index_of(completed_store, NO_COHORT_EAF)])
    # a1@NO_COHORT_EAF is observed with no frequency, a1@IMPUTED imputed.
    with query_store(completed_store) as query:
        result = query.analysis("a1")
    by_variant = {v: (e, s) for v, e, s in zip(
        result["variant_index"], result["eaf"], result["association_status"], strict=True
    )}
    assert np.isnan(by_variant[_variant_index_of(completed_store, NO_COHORT_EAF)][0])
    assert by_variant[_variant_index_of(completed_store, IMPUTED)][1] == "imputed"
    assert validate_store(completed_store).ok


def test_reference_only_fixture_has_no_eaf_plane_but_imputes(
    reference_only_completed_store: Path,
) -> None:
    manifest = open_store(reference_only_completed_store).manifest
    assert manifest.encoding.eaf.reference, "the fixture must declare reference EAF"
    assert manifest.encoding.eaf.is_absent, "the fixture must reach the no-eaf-plane branch"
    root = open_store(reference_only_completed_store).arrays(mode="r")
    assert "eaf" not in root, "this branch is only meaningful with no eaf plane"
    assert "eaf_reference" in root and "imputed" in root
    assert int(root["imputed"][:].sum()) == 1, "the fixture must actually impute a cell"


# ── Build ───────────────────────────────────────────────────────────────────


def test_build_carries_status_and_reference_eaf(tmp_path: Path, completed_store: Path) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    subset = open_indexed_subset(store, "hm3")
    assert subset.has_imputed is True
    assert subset.has_eaf is True
    assert subset.n_analyses == 2
    assert subset.resolved_count == _SIGNAL_N
    assert subset.absent_count == 1

    group = _index_group(store, "hm3")
    assert "imputed" in group
    assert "eaf_reference" in group
    assert str(group["imputed"].dtype) == "uint8"
    assert tuple(group["imputed"].shape) == (2, _SIGNAL_N)
    assert group["eaf_reference"].shape[0] == _SIGNAL_N
    assert validate_store(store).ok


def test_build_on_reference_only_release_carries_reference_without_eaf(
    tmp_path: Path, reference_only_completed_store: Path
) -> None:
    store = _build_subset(
        shutil.copytree(reference_only_completed_store, tmp_path / "s.opengwasdb"), tmp_path
    )
    group = _index_group(store, "hm3")
    assert "eaf" not in group
    assert "eaf_reference" in group
    assert "imputed" in group
    subset = open_indexed_subset(store, "hm3")
    assert subset.has_eaf is False
    decoded = subset.decode_analysis(0)
    assert decoded.eaf is not None, "reference-only EAF still has liveness on imputed cells"
    assert np.isnan(decoded.z).sum() == 0, "the fixture's axis is observed-only where finite"
    assert validate_store(store).ok


def test_absent_from_store_is_counted_while_absent_from_source_is_indexed(
    tmp_path: Path, completed_store: Path
) -> None:
    """Two different "absent"s: not in the Store, versus not observed in a source."""
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    subset = open_indexed_subset(store, "hm3")
    assert subset.resolved_count == _SIGNAL_N
    assert subset.absent_count == 1
    # The target is absent from every source row yet still on the index axis;
    # query filters it by the finite Z/SE rule, not by dropping the variant.
    assert _variant_index_of(store, IMPUTED) in subset.variant_index.tolist()


# ── Equivalence with filtering ordinary analysis() ──────────────────────────


@pytest.mark.parametrize("analysis_id", ["a1", "a2"])
def test_indexed_equals_ordinary_filtered_by_default(
    tmp_path: Path, completed_store: Path, analysis_id: str
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    ordinary = _analyse(store, analysis_id)
    indexed = _analyse(store, analysis_id, indexed_subset="hm3")
    expected = _ordinary_filtered(ordinary, open_indexed_subset(store, "hm3").variant_index)
    assert len(expected["z"]) > 0, "the fixture must return rows for this to mean anything"
    _assert_results_equal(indexed, expected)


@pytest.mark.parametrize("analysis_id", ["a1", "a2"])
def test_observed_only_selector_is_equivalent(
    tmp_path: Path, completed_store: Path, analysis_id: str
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    ordinary = _analyse(store, analysis_id, observed_only=True)
    indexed = _analyse(store, analysis_id, observed_only=True, indexed_subset="hm3")
    expected = _ordinary_filtered(ordinary, open_indexed_subset(store, "hm3").variant_index)
    _assert_results_equal(indexed, expected)


def test_observed_only_drops_exactly_the_imputed_cell(
    tmp_path: Path, completed_store: Path
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    default = _analyse(store, "a1", indexed_subset="hm3")
    observed = _analyse(store, "a1", observed_only=True, indexed_subset="hm3")
    assert "imputed" in set(default["association_status"]), (
        "the fixture must return an imputed row for the drop to mean anything"
    )
    assert "imputed" not in set(observed["association_status"])
    assert set(observed["variant_index"]) == {
        v
        for v, status in zip(default["variant_index"], default["association_status"], strict=True)
        if status != "imputed"
    }


def test_imputed_cell_reads_reference_eaf(
    tmp_path: Path, completed_store: Path
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    target = _variant_index_of(store, IMPUTED)
    indexed = _analyse(store, "a1", indexed_subset="hm3")
    eaf_by_variant = dict(zip(indexed["variant_index"], indexed["eaf"], strict=True))
    status_by_variant = dict(
        zip(indexed["variant_index"], indexed["association_status"], strict=True)
    )
    reference = open_store(store).arrays(mode="r")["eaf_reference"][:]
    assert status_by_variant[target] == "imputed"
    assert eaf_by_variant[target] == pytest.approx(float(reference[target]))
    # a2 observed the same variant and was not completed; a1's imputed cell
    # must not make a2's observed cell read as imputed.
    a2 = _analyse(store, "a2", indexed_subset="hm3")
    a2_status = dict(zip(a2["variant_index"], a2["association_status"], strict=True))
    assert target in a2_status
    assert a2_status[target] == "observed"


def test_observed_cell_without_cohort_eaf_never_takes_reference(
    tmp_path: Path, completed_store: Path
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    target = _variant_index_of(store, NO_COHORT_EAF)
    reference = open_store(store).arrays(mode="r")["eaf_reference"][:]
    assert np.isfinite(reference[target]), "the panel must hold a frequency to leak"
    indexed = _analyse(store, "a1", indexed_subset="hm3")
    eaf_by_variant = dict(zip(indexed["variant_index"], indexed["eaf"], strict=True))
    status_by_variant = dict(
        zip(indexed["variant_index"], indexed["association_status"], strict=True)
    )
    assert status_by_variant[target] == "observed"
    assert np.isnan(eaf_by_variant[target]), (
        "an observed cell with no cohort frequency must stay absent, never the panel's"
    )


def test_unmatched_analysis_reads_only_its_observed_cells(
    tmp_path: Path, completed_store: Path
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    indexed = _analyse(store, "a2", indexed_subset="hm3")
    assert set(indexed["association_status"]) == {"observed"}
    assert indexed["variant_index"].tolist() == [
        _variant_index_of(store, 0),
        _variant_index_of(store, IMPUTED),
    ], "missing cells are filtered by the paired Z/SE rule, not inferred from a1"


def test_reference_only_index_decodes_the_panel_frequency(
    tmp_path: Path, reference_only_completed_store: Path
) -> None:
    store = _build_subset(
        shutil.copytree(reference_only_completed_store, tmp_path / "s.opengwasdb"), tmp_path
    )
    target = _variant_index_of(store, IMPUTED)
    ordinary = _analyse(store, "a1")
    indexed = _analyse(store, "a1", indexed_subset="hm3")
    expected = _ordinary_filtered(ordinary, open_indexed_subset(store, "hm3").variant_index)
    _assert_results_equal(indexed, expected)
    eaf_by_variant = dict(zip(indexed["variant_index"], indexed["eaf"], strict=True))
    assert eaf_by_variant[target] == pytest.approx(0.3, abs=1e-6)
    assert np.isnan(eaf_by_variant[_variant_index_of(store, 0)])


# ── Validation ──────────────────────────────────────────────────────────────


def _errors_for(store: Path) -> list[str]:
    return validate_store(store).errors


def test_validation_accepts_the_untouched_index(
    tmp_path: Path, completed_store: Path
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    assert validate_store(store).ok


def test_validation_rejects_a_flipped_imputed_mask(
    tmp_path: Path, completed_store: Path
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    group = _index_group(store, "hm3")
    # Mark a1's observed first cell imputed.
    mask = np.asarray(group["imputed"][:])
    mask[0, 0] = 1
    group["imputed"][:] = mask
    errors = _errors_for(store)
    assert any("imputed mask disagrees" in e for e in errors), errors


def test_validation_rejects_imputed_true_at_a_missing_cell(
    tmp_path: Path, completed_store: Path
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    group = _index_group(store, "hm3")
    # a2 has no association at slot 1 (position 1): set that missing cell imputed.
    mask = np.asarray(group["imputed"][:])
    mask[1, 1] = 1
    group["imputed"][:] = mask
    errors = _errors_for(store)
    assert any("marks a missing Z/SE cell imputed" in e for e in errors), errors


def test_validation_rejects_a_non_boolean_imputed_mask(
    tmp_path: Path, completed_store: Path
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    group = _index_group(store, "hm3")
    mask = np.asarray(group["imputed"][:])
    mask[0, 0] = 2
    group["imputed"][:] = mask
    errors = _errors_for(store)
    assert any("values other than 0 and 1" in e for e in errors), errors


# ── Read-path mask structure/domain (issue #266 review) ─────────────────────
#
# The read path refuses a mask the codec and Association Status would read
# differently -- a float mask, or a value outside {0,1} -- before decoding the
# Analysis's column.  Comparing the mask's or `eaf_reference`'s *content*
# against the release stays `validate`'s job.


def _set_mask_value(store: Path, value: int) -> None:
    group = _index_group(store, "hm3")
    mask = np.asarray(group["imputed"][:])
    mask[0, 0] = value
    group["imputed"][:] = mask


def _replace_mask_with_float(store: Path) -> None:
    """Rewrite the mask as float32 holding a 0.5 cell.

    A decoder that coerced it to uint8 would see 0 (observed) while a truthy
    test would see imputed -- the inconsistency this refuses.
    """
    group = _index_group(store, "hm3")
    n_analyses = int(group.attrs["n_analyses"])
    n_subset = int(group.attrs["n_subset_variants"])
    values = np.zeros((n_analyses, n_subset), dtype="float32")
    values[0, 0] = 0.5
    del group["imputed"]
    group.create_array("imputed", chunks=(1, n_subset), shards=(1, n_subset), data=values)


def _damage_mask(store: Path, damage: str) -> None:
    if damage == "value2":
        _set_mask_value(store, 2)
    else:
        _replace_mask_with_float(store)


def test_open_refuses_a_float_mask_before_decoding(
    tmp_path: Path, completed_store: Path
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    _replace_mask_with_float(store)
    with pytest.raises(IndexedSubsetError, match="uint8") as excinfo:
        open_indexed_subset(store, "hm3")
    message = str(excinfo.value)
    assert str(store) in message and "hm3" in message, message


@pytest.mark.parametrize("observed_only", [False, True])
@pytest.mark.parametrize("damage", ["value2", "float"])
def test_damaged_mask_is_refused_by_the_api(
    tmp_path: Path, completed_store: Path, damage: str, observed_only: bool
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    _damage_mask(store, damage)
    with pytest.raises(IndexedSubsetError) as excinfo:
        _analyse(store, "a1", indexed_subset="hm3", observed_only=observed_only)
    message = str(excinfo.value)
    assert str(store) in message and "hm3" in message, message


@pytest.mark.parametrize("damage", ["value2", "float"])
def test_damaged_mask_is_refused_by_the_cli(
    tmp_path: Path, completed_store: Path, damage: str
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    _damage_mask(store, damage)
    code, output = _invoke(store, "a1", "--indexed-subset", "hm3", "--format", "json")
    assert code == 1, output
    assert "error:" in output, output
    assert str(store) in output and "hm3" in output, output


def test_validation_rejects_a_corrupt_reference_eaf(
    tmp_path: Path, completed_store: Path
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    group = _index_group(store, "hm3")
    reference = np.asarray(group["eaf_reference"][:])
    reference[IMPUTED] = 0.9
    group["eaf_reference"][:] = reference
    errors = _errors_for(store)
    assert any("eaf_reference disagrees" in e for e in errors), errors


def test_validation_rejects_a_corrupt_indexed_eaf_value(
    tmp_path: Path, completed_store: Path
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    group = _index_group(store, "hm3")
    values = np.asarray(group["eaf"][:])
    values[0, 0] = values[0, 0] + 1
    group["eaf"][:] = values
    errors = _errors_for(store)
    assert any("eaf values differ" in e for e in errors), errors


def test_validation_rejects_a_missing_imputed_mask(
    tmp_path: Path, completed_store: Path
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    group = _index_group(store, "hm3")
    del group["imputed"]
    errors = _errors_for(store)
    assert any("imputed" in e and "missing" in e for e in errors), errors


def test_validation_rejects_a_missing_reference_eaf(
    tmp_path: Path, completed_store: Path
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    group = _index_group(store, "hm3")
    del group["eaf_reference"]
    errors = _errors_for(store)
    assert any("eaf_reference" in e for e in errors), errors


def test_validation_rejects_a_short_reference_eaf(
    tmp_path: Path, completed_store: Path
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    group = _index_group(store, "hm3")
    values = np.asarray(group["eaf_reference"][: _SIGNAL_N - 1])
    del group["eaf_reference"]
    group.create_array(
        "eaf_reference",
        chunks=(_SIGNAL_N - 1,),
        shards=(_SIGNAL_N - 1,),
        data=values,
    )
    errors = _errors_for(store)
    assert any("eaf_reference has" in e and "variants" in e for e in errors), errors


def test_completion_metadata_stays_in_the_analyses_table(
    tmp_path: Path, completed_store: Path
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    group = _index_group(store, "hm3")
    assert "completed_against" not in group.attrs, (
        "completion provenance is Analytical Metadata; the index must not duplicate it"
    )
    assert all("completion" not in key for key in group.keys()), (
        "completion metadata lives in analyses.tsv, not the index"
    )
    assert all("analysis" not in key for key in group.keys()), (
        "the index carries no Analysis-level metadata table"
    )
    rows = {row["analysis_id"]: row for row in read_analyses(store / "analyses.tsv").rows}
    assert rows["a1"]["completed_against"] == "EUR"
    assert rows["a2"]["completed_against"] == ""


# ── Lifecycle: a completed build is atomic like any other index ─────────────


def test_failed_completed_replacement_preserves_the_published_group(
    tmp_path: Path,
    completed_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    published = store / "data.zarr" / INDEXED_SUBSETS_GROUP / "hm3"
    before = _digest(published)

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        "opengwasdb.layouts.dense.indexed_subsets.write_indexed_subset_group", boom
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _build_subset(store, tmp_path, name="hm3", overwrite=True)

    assert _digest(published) == before
    assert not [p for p in published.parent.iterdir() if p.name.startswith(".")]
    assert validate_store(store).ok


def test_completed_index_removal_leaves_ordinary_queries_unchanged(
    tmp_path: Path, completed_store: Path
) -> None:
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    with query_store(store) as query:
        before = query.analysis("a1")
    authoritative = _authoritative_digest(store)

    from opengwasdb.layouts.dense.indexed_subsets import remove_indexed_subset

    assert remove_indexed_subset(store, "hm3") is True
    assert list_indexed_subsets(store) == ()
    assert _authoritative_digest(store) == authoritative
    assert validate_store(store).ok
    with query_store(store) as query:
        after = query.analysis("a1")
    for field in before:
        np.testing.assert_array_equal(before[field], after[field], err_msg=field)


def test_read_seam_refuses_a_missing_status_array(
    tmp_path: Path, completed_store: Path
) -> None:
    """The read path refuses a completed index that is missing its status mask."""
    store = _build_subset(shutil.copytree(completed_store, tmp_path / "s.opengwasdb"), tmp_path)
    group = _index_group(store, "hm3")
    del group["imputed"]
    with pytest.raises(IndexedSubsetError, match="imputed"):
        open_indexed_subset(store, "hm3")
