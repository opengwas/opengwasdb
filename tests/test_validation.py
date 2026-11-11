from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import plane_edits
import pytest
from dense_sources import build_dense_from_records
from residual_fixtures import model_se

from opengwasdb.encoding import DenseSePlane
from opengwasdb.encoding.codec import SeExceptionTable, se_float16
from opengwasdb.layouts.dense.build import NormalisedAssociation
from opengwasdb.layouts.dense.rho import build_dense_rho
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.model.analyses import read_analyses, write_analyses
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store.arrays import ArrayRole, compressor, create_array
from opengwasdb.store.open import CURRENT_FORMAT_VERSION, open_store
from opengwasdb.validation import Severity, validate_store
from opengwasdb.validation.rules.side_tables import (
    FLOAT16_EXACT_MINIMUM,
    FLOAT16_EXACT_SHARE,
)
from opengwasdb.variants import CanonicalVariant


@pytest.mark.parametrize("fixture", ["dense_store_path", "ragged_store_path", "hybrid_store_path"])
def test_a_release_stamped_0_2_1_without_a_region_table_fails_its_stamp(request, fixture):
    """Spec §21.3: a release is stamped the lowest version its features use, so one
    without `regions.tsv` is 0.2.0, and one stamped otherwise disagrees with itself. For
    a Hybrid, the outer release's manifest is the one restamped."""
    from region_tables import edit_manifest

    store = request.getfixturevalue(fixture)
    assert validate_store(store).ok, "the fixture must validate before it is restamped"
    assert open_store(store).manifest.format_version == "0.2.0"
    edit_manifest(store, lambda manifest: {**manifest, "format_version": "0.2.1"})

    result = validate_store(store)

    assert result.failed() == {"manifest.format_stamp"}, result.errors
    assert any(
        "stamped 0.2.1, but a release carrying no region table is 0.2.0" in error
        for error in result.errors
    ), result.errors


def test_validator_rejects_missing_required_array(dense_store_path):
    root = open_store(dense_store_path).arrays(mode="a")
    del root["se"]

    result = validate_store(dense_store_path)

    assert not result.ok
    assert "missing data.zarr/se" in result.errors


def test_a_release_without_its_data_directory_reports_it_rather_than_raising(dense_store_path):
    """`top_hits.arrays` asks whether there is an index by opening `data.zarr`; that is
    not asked once the envelope rule has found `data.zarr` missing."""
    shutil.rmtree(dense_store_path / "data.zarr")

    result = validate_store(dense_store_path)

    assert result.failed() == {"envelope.required"}
    assert result.errors == ["missing data.zarr"]


def test_validator_rejects_misshaped_z_plane(dense_store_path):
    # The z/se/eaf planes are parallel (n_variants, n_analyses) matrices; a
    # release whose z plane does not span the variant axis cannot be read
    # against that axis, and the guard must say so before any band is decoded.
    root = open_store(dense_store_path).arrays(mode="a")
    n_variants, n_analyses = root["z"].shape
    root["z"].resize((n_variants - 1, n_analyses))

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any(
        f"z shape ({n_variants - 1}, {n_analyses}) does not match" in e
        for e in result.errors
    ), result.errors


def test_validator_rejects_negative_se(dense_store_path):
    root = open_store(dense_store_path).arrays(mode="a")
    root["se"][0, 0] = -0.1

    result = validate_store(dense_store_path)

    assert result.failed() == {"associations.se_negative"}
    assert "se contains negative finite values" in result.errors


def test_validator_rejects_inconsistent_missingness(dense_store_path):
    root = open_store(dense_store_path).arrays(mode="a")
    root["se"][0, 0] = float("nan")

    result = validate_store(dense_store_path)

    assert result.failed() == {"associations.markers_paired"}
    assert "z and se missingness is inconsistent" in result.errors


def test_validator_rejects_invalid_stored_effect_scale(dense_store_path):
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    for row in table.rows:
        if row["analysis_id"] == "a1":
            row["stored_effect_scale"] = "kg"
    write_analyses(analyses_path, table)

    result = validate_store(dense_store_path)

    assert result.failed() == {"analyses.vocabularies"}
    assert any("invalid stored_effect_scale" in error for error in result.errors)


def test_validator_rejects_inconsistent_top_hit_index(dense_store_path):
    root = open_store(dense_store_path).arrays(mode="a")
    root["top_hits"]["p_5e_08"]["z"][0] = 0.0

    result = validate_store(dense_store_path)

    assert result.failed() == {"top_hits.z"}
    assert result.errors == ["top-hit index p_5e_08 contains z value inconsistent with z array"]


def test_a_bad_variant_offset_and_a_bad_top_hit_tier_are_both_reported(dense_store_path):
    # Issue #55's acceptance case: two independent defects in one release. The
    # Top-Hit Index was checked only when nothing else had failed, so the offset
    # error hid the tier's.
    offsets_path = dense_store_path / "variant_offsets.npy"
    offsets = np.load(offsets_path)
    offsets[1] = offsets[0]
    np.save(offsets_path, offsets)
    root = open_store(dense_store_path).arrays(mode="a")
    root["top_hits"]["p_5e_08"]["z"][0] = 0.0

    result = validate_store(dense_store_path)

    assert result.failed() == {"variant_table.representatives", "top_hits.z"}


def test_validator_rejects_invalid_top_hit_offsets(dense_store_path):
    root = open_store(dense_store_path).arrays(mode="r+")
    offsets = root["top_hits/p_5e_08/analysis_offsets"][:]
    offsets[-1] -= 1
    root["top_hits/p_5e_08/analysis_offsets"][:] = offsets

    result = validate_store(dense_store_path)

    assert result.failed() == {"top_hits.structure"}
    assert any("invalid analysis offsets" in error for error in result.errors)


def test_the_zarr_format_rule_reports_nodes_in_name_order(
    dense_store_path, monkeypatch: pytest.MonkeyPatch
):
    """A walk of the array tree is sorted, so the two nodes come out in one order.

    The rule reads the filesystem rather than zarr, and a directory listing's order
    is the filesystem's: a half-converted release used to name its nodes in whichever
    order the walk reached them (#378 review 2).
    """
    for name in ("z", "se"):
        (dense_store_path / "data.zarr" / name / ".zarray").write_text("{}", encoding="utf-8")
    walked = os.walk
    monkeypatch.setattr(
        os,
        "walk",
        lambda top, *a, **k: reversed(sorted(walked(top, *a, **k), key=lambda node: node[0])),
    )

    result = validate_store(dense_store_path)

    assert result.failed() == {"zarr.format"}
    assert result.errors == [
        f"data.zarr/{node}: Zarr v2 metadata ['.zarray'] under format_version 0.2.1; a "
        "0.2.1 release is Zarr v3, and a half-converted release is invalid (issue #245)"
        for node in ("se", "z")
    ]


def test_validator_rejects_missing_variant_axis_file(dense_store_path):
    (dense_store_path / "variants.tsv.gz").unlink()

    result = validate_store(dense_store_path)

    assert result.failed() == {"envelope.required"}
    assert "missing variants.tsv.gz" in result.errors


@pytest.mark.parametrize("name", ["variant_alid_bytes.npy", "variant_alid_rows.npy"])
def test_validator_rejects_missing_alid_index_file_for_dense(dense_store_path, name):
    (dense_store_path / name).unlink()

    result = validate_store(dense_store_path)

    assert result.failed() == {"envelope.required"}
    assert any(
        error == f"missing {name} — rebuild the store to generate the ALID search index"
        for error in result.errors
    )


def test_validator_rejects_bad_variant_offset(dense_store_path):
    offsets_path = dense_store_path / "variant_offsets.npy"
    offsets = np.load(offsets_path)
    offsets[1] = offsets[0]
    np.save(offsets_path, offsets)

    result = validate_store(dense_store_path)

    assert result.failed() == {"variant_table.representatives"}
    assert any("variant offset for row 1" in error for error in result.errors)


def test_validator_rejects_leftover_sqlite_analyses_table(dense_store_path):
    # ADR 0030/issue #22: analyses.tsv is the sole source of truth for
    # Analytical Metadata; a store carrying a leftover SQLite `analyses`
    # table (store-format spec §20) is invalid even though nothing else
    # about it is malformed.
    connection = open_store(dense_store_path).index_connection()
    try:
        connection.execute("CREATE TABLE analyses (analysis_index INTEGER PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()

    result = validate_store(dense_store_path)

    assert result.failed() == {"sqlite.analyses_table"}


def test_validator_accepts_store_without_sqlite_analyses_table(dense_store_path):
    result = validate_store(dense_store_path)

    assert result.ok, result.errors


def test_validator_rejects_leftover_sqlite_analyses_table_ragged(ragged_store_path):
    # Issue #69/#72: Ragged retired its SQLite `analyses` table onto
    # analyses.tsv too -- the same rule Dense already enforces must catch a
    # Ragged release that still carries one.
    connection = open_store(ragged_store_path).index_connection()
    try:
        connection.execute("CREATE TABLE analyses (analysis_index INTEGER PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()

    result = validate_store(ragged_store_path)

    assert result.failed() == {"sqlite.analyses_table"}


def test_a_ragged_release_has_its_variant_table_checked(ragged_store_path):
    # Issue #55: the variant-table rules ran on Dense and Hybrid releases only; a
    # Ragged release's offset sidecar could point at the wrong row unremarked.
    offsets_path = ragged_store_path / "variant_offsets.npy"
    offsets = np.load(offsets_path)
    assert len(offsets) >= 2, "the fixture must have a second row to misdirect"
    offsets[1] = offsets[0]
    np.save(offsets_path, offsets)

    result = validate_store(ragged_store_path)

    assert result.failed() == {"variant_table.representatives"}


def test_a_ragged_release_has_no_alias_table_and_is_not_said_to_pass_it(ragged_store_path):
    result = validate_store(ragged_store_path)

    assert result.ok, result.errors
    assert result.skipped["sqlite.variant_aliases"] == (
        "index.sqlite has no variant_aliases table",
    )


def test_validator_rejects_missing_ragged_csr_array(ragged_store_path):
    # The Ragged validator's CSR-structure seam must not turn a missing
    # parallel array into a decode crash in a later seam.
    result = validate_store(ragged_store_path)
    assert result.ok, result.errors
    ragged = open_store(ragged_store_path).arrays(mode="a")["ragged"]
    assert "se" in ragged and ragged["se"].shape[0] == int(ragged["offsets"][-1])  # fixture sanity
    del ragged["se"]

    result = validate_store(ragged_store_path)

    assert result.failed() == {"csr.arrays"}
    assert "missing data.zarr/ragged/se" in result.errors


def _replace_ragged_z(store_path: object, values: np.ndarray) -> None:
    """Rewrite a Ragged release's `z` sequence through the seam.

    Through the seam, so the replacement is a shard-valid 0.2.0 array and the
    error under test is the structural one rather than the format rule the
    corruption would otherwise trip (#247).
    """
    ragged = open_store(store_path).arrays(mode="a")["ragged"]
    del ragged["z"]
    create_array(
        ragged, "z", ArrayRole.ASSOCIATION_SEQUENCE, data=values, compressor=compressor()
    )


def test_validator_rejects_ragged_csr_array_length_mismatch(ragged_store_path):
    # A parallel array shorter than `offsets` implies is a structural error
    # that must be reported, not read as the store silently losing rows.
    ragged = open_store(ragged_store_path).arrays(mode="r")["ragged"]
    n_assoc = int(ragged["offsets"][-1])
    assert n_assoc > 1  # fixture sanity: the truncation must actually shorten z
    assert ragged["z"].shape[0] == n_assoc
    z = np.asarray(ragged["z"][:])
    _replace_ragged_z(ragged_store_path, z[:-1])

    result = validate_store(ragged_store_path)

    assert not result.ok
    assert any(
        "data.zarr/ragged/z has" in error and "entries but offsets imply" in error
        for error in result.errors
    )


def test_validator_rejects_ragged_csr_array_outside_declared_plan(ragged_store_path):
    # The manifest's encoding is the contract (issue #119): an array whose
    # dtype disagrees with it must be reported as such, and later seams must
    # not try to decode values under a plan the structure already contradicted.
    ragged = open_store(ragged_store_path).arrays(mode="r")["ragged"]
    assert str(ragged["z"].dtype) == "int16"  # fixture sanity
    z = np.asarray(ragged["z"][:]).astype(np.float32)
    _replace_ragged_z(ragged_store_path, z)

    result = validate_store(ragged_store_path)

    assert result.failed() == {"encoding.dtypes"}
    assert any(
        "data.zarr/ragged/z has dtype float32 but the manifest declares int16_fixed"
        in error
        for error in result.errors
    )


def test_validator_rejects_negative_ragged_se(ragged_store_path):
    # Same rule as the Dense band pass: a decoded negative finite se is a
    # wrong answer a query would serve unchanged, so it is a hard error.
    ragged = open_store(ragged_store_path).arrays(mode="r")["ragged"]
    assert np.all(np.asarray(ragged["se"][:]) > 0)  # fixture sanity
    ragged = open_store(ragged_store_path).arrays(mode="a")["ragged"]
    ragged["se"][0] = np.float16(-0.1)

    result = validate_store(ragged_store_path)

    assert not result.ok
    assert "se contains negative finite values" in result.errors


def _infinite_se_warnings(result) -> list[str]:
    return [warning for warning in result.warnings if "infinite se" in warning]


def test_validator_warns_about_an_infinite_dense_se(dense_store_path):
    """An SE above float16's range is `+inf` beside a finite Z: a warning, not an error.

    `beta = z * se` reads as infinity although the source value was finite (issue
    #338), and nothing else in a Store says so. The cell is present under spec §15,
    so the release stays valid and the warning names the count and the Analysis
    (issue #337).

    Mutation that makes this fail: count nothing in `associations.se_infinite`'s
    measure, or stop judging its total.
    """
    root = open_store(dense_store_path).arrays(mode="a")
    assert root["se"].dtype == np.dtype("float16")  # fixture sanity
    root["se"][0, 0] = float("inf")

    result = validate_store(dense_store_path)

    assert result.ok, result.errors
    warnings = _infinite_se_warnings(result)
    # The fixture holds no `eaf` plane, so the frequency rule is skipped beside it.
    assert [f.rule for f in result.findings if f.severity is Severity.WARNING] == [
        "associations.se_infinite"
    ]
    assert result.errors == []
    assert "1 cell(s) of data.zarr/se" in warnings[0]
    assert "'a1'" in warnings[0]


def test_validator_warns_about_an_infinite_ragged_se(ragged_store_path):
    """An infinite SE in a Ragged plane's CSR, where no band pass reads it.

    Mutation that makes this fail: drop the Ragged plane from `_ragged_se_planes`, or
    scan the plane with the Analysis boundaries ignored (the id would be wrong).
    """
    ragged = open_store(ragged_store_path).arrays(mode="a")["ragged"]
    index = ragged["by_variant"]
    assert ragged["se"].dtype == np.dtype("float16")  # fixture sanity
    assert int(ragged["offsets"][-1]) == int(index["offsets"][-1]) == 2  # two associations
    # Both copies of the cell move: the Analysis-sorted CSR and the variant-centric
    # index hold the same cell (spec §11.1), so writing one alone is an error of its own.
    variant = int(ragged["variant_index"][0])
    hold = slice(int(index["offsets"][variant]), int(index["offsets"][variant + 1]))
    assert list(np.asarray(index["analysis_index"][hold])) == [0]  # fixture sanity: one Analysis
    ragged["se"][0] = np.float16(np.inf)
    index["se"][hold.start] = np.float16(np.inf)

    result = validate_store(ragged_store_path)

    assert result.ok, result.errors
    warnings = _infinite_se_warnings(result)
    assert len(warnings) == 1, result.warnings
    assert "1 cell(s) of data.zarr/ragged/se" in warnings[0]
    assert "'t1'" in warnings[0]


def _hybrid_store_with_both_components(tmp_path: Path) -> Path:
    """An Observed-Only Hybrid holding one Dense Component association and one Overflow one.

    Two Analyses over the one panel variant, plus `1:2000000` off the panel in `trait_a`.
    Every effect is tiny, so nothing here is a top-hit and writing an infinite SE moves no
    index: the finding under test is the warning, not an index disagreement.
    """
    import test_hybrid_completion as thc

    trait_a = thc._vcf(tmp_path, "trait_a", [
        "1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE\t0.002:0.5\n",
        "1\t2000000\t.\tA\tG\t.\tPASS\t.\tES:SE\t0.002:0.5\n",  # off-panel: the Overflow
    ])
    trait_c = thc._vcf(tmp_path, "trait_c", [
        "1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE\t0.002:0.5\n",
    ])
    manifest = thc.write_manifest(
        tmp_path, [(trait_a, "trait_a", "Trait A"), (trait_c, "trait_c", "Trait C")]
    )
    panel = tmp_path / "panel.txt"
    panel.write_text("1:100000:A:G\n", encoding="utf-8")
    store = tmp_path / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=panel, store_id="hyb", release_id="v1"
    )
    return store


def test_validator_warns_about_an_infinite_se_in_both_hybrid_components(tmp_path):
    """One infinite SE cell in each component: two warnings, each naming its own plane.

    Mutation that makes this fail: dropping the Overflow plane from
    `_ragged_se_planes` (one warning), dropping the Dense Component's grid from the
    Hybrid's targets (one warning), yielding or reporting a plane twice (three), or
    labelling both planes alike (the label assertions).
    """
    store = _hybrid_store_with_both_components(tmp_path)
    component = open_store(store).dense_component()
    dense_se = DenseSePlane.open(component.arrays(mode="r+"), component.manifest.encoding)
    assert component.manifest.encoding.se.kind == "float16"  # fixture sanity
    rows, cols = np.array([0]), np.array([1])  # trait_c observes the one panel variant
    assert np.isfinite(dense_se.points(rows, cols)[0])  # fixture sanity: an observed cell
    group, encoding = component.arrays(mode="r+"), component.manifest.encoding
    plane_edits.patch(group, encoding, DenseSePlane, rows, cols, np.array([np.inf], np.float32))
    ragged = open_store(store).arrays(mode="r+")["ragged"]
    assert ragged["se"].dtype == np.dtype("float16")  # fixture sanity
    overflow = int(ragged["offsets"][0])  # trait_a's off-panel association comes first
    assert int(ragged["variant_index"][overflow]) != 0  # fixture sanity: a real Overflow row
    ragged["se"][overflow] = np.float16(np.inf)
    index = ragged["by_variant"]
    variant = int(ragged["variant_index"][overflow])
    hold = slice(int(index["offsets"][variant]), int(index["offsets"][variant + 1]))
    index["se"][hold.start] = np.float16(np.inf)

    result = validate_store(store)

    assert result.ok, result.errors
    warnings = _infinite_se_warnings(result)
    assert len(warnings) == 2, result.warnings
    # Order-independent: the Overflow is read before the release's own validators run.
    by_plane = {warning.split(" hold ")[0]: warning for warning in warnings}
    assert set(by_plane) == {
        "1 cell(s) of data.zarr/ragged/se", "1 cell(s) of dense/data.zarr/se",
    }
    assert "'trait_a'" in by_plane["1 cell(s) of data.zarr/ragged/se"]
    assert "'trait_c'" in by_plane["1 cell(s) of dense/data.zarr/se"]


def test_a_store_without_an_infinite_se_gets_no_infinite_se_warning(
    dense_store_path, ragged_store_path
):
    # The warning is about the arrays, not about the layout: a release that holds no
    # infinite cell must not be told it does.
    assert validate_store(dense_store_path).ok
    assert _infinite_se_warnings(validate_store(dense_store_path)) == []
    assert _infinite_se_warnings(validate_store(ragged_store_path)) == []


def test_validator_rejects_ragged_analyses_row_count_mismatch(ragged_store_path):
    # analyses.tsv is the sole source of truth for Analytical Metadata, so a
    # row count that disagrees with the CSR offsets is a store whose metadata
    # no longer describes its arrays (ADR 0030, spec §11/§20).
    result = validate_store(ragged_store_path)
    assert result.ok, result.errors
    analyses_path = ragged_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    assert len(table.rows) == 1  # fixture sanity: dropping it must change the count
    write_analyses(analyses_path, type(table)(fieldnames=table.fieldnames, rows=()))

    result = validate_store(ragged_store_path)

    assert not result.ok
    assert any(
        "analyses.tsv has 0 rows but" in error and "offsets imply 1" in error
        for error in result.errors
    )


def test_validator_accepts_migrated_ragged_store(ragged_store_path):
    result = validate_store(ragged_store_path)

    assert result.ok, result.errors


def test_validator_accepts_migrated_hybrid_store(hybrid_store_path):
    # Issue #72: "a correctly-migrated Dense, Hybrid, and Ragged release...
    # succeeds" -- Hybrid's Dense Component reuses _validate_dense_store,
    # so this covers the same two new rules via that shared path.
    result = validate_store(hybrid_store_path)

    assert result.ok, result.errors


def test_validator_rejects_leftover_sqlite_analyses_table_hybrid(hybrid_store_path):
    from opengwasdb.layouts.hybrid.layout import dense_component_path

    connection = open_store(dense_component_path(hybrid_store_path)).index_connection()
    try:
        connection.execute("CREATE TABLE analyses (analysis_index INTEGER PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()

    result = validate_store(hybrid_store_path)

    assert result.failed() == {"sqlite.analyses_table"}
    assert [f.target for f in result.of("sqlite.analyses_table")] == ["dense"]


def test_validator_rejects_duplicate_analysis_id_hybrid(hybrid_store_path):
    from opengwasdb.layouts.hybrid.layout import dense_component_path

    analyses_path = dense_component_path(hybrid_store_path) / "analyses.tsv"
    table = read_analyses(analyses_path)
    rows = [{**row, "analysis_id": table.rows[0]["analysis_id"]} for row in table.rows]
    write_analyses(analyses_path, type(table)(fieldnames=table.fieldnames, rows=tuple(rows)))

    result = validate_store(hybrid_store_path)

    assert result.failed() == {"analyses.analysis_id"}
    assert [f.target for f in result.of("analyses.analysis_id")] == ["dense"]


def test_validator_rejects_duplicate_analysis_id(dense_store_path):
    # Issue #72: analysis_id uniqueness within analyses.tsv is enforced for
    # every layout, not just documented (store-format spec §7a).
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    rows = list(table.rows)
    rows[1] = {**rows[1], "analysis_id": rows[0]["analysis_id"]}
    write_analyses(analyses_path, type(table)(fieldnames=table.fieldnames, rows=tuple(rows)))

    result = validate_store(dense_store_path)

    assert result.failed() == {"analyses.analysis_id"}


def test_validator_rejects_stray_file_in_dense_store(dense_store_path):
    # Issue #80: the closed envelope catches a file no build path writes --
    # this is the class of bug that let a stray Ragged traits.tsv.gz side-file
    # go unnoticed until a human spotted it (issues #69-#73 / PR #78).
    (dense_store_path / "traits.tsv.gz").write_bytes(b"stray")

    result = validate_store(dense_store_path)

    assert result.failed() == {"envelope.closed"}
    assert any(
        "unexpected store entry" in error and "traits.tsv.gz" in error
        for error in result.errors
    )


def test_validator_rejects_stray_directory_in_ragged_store(ragged_store_path):
    (ragged_store_path / "extra_dir").mkdir()

    result = validate_store(ragged_store_path)

    assert result.failed() == {"envelope.closed"}
    assert any(
        "unexpected store entry" in error and "extra_dir" in error for error in result.errors
    )


def test_validator_rejects_stray_file_at_hybrid_top_level(hybrid_store_path):
    (hybrid_store_path / "traits.tsv.gz").write_bytes(b"stray")

    result = validate_store(hybrid_store_path)

    assert result.failed() == {"envelope.closed"}
    assert any(
        "unexpected store entry" in error and "traits.tsv.gz" in error
        for error in result.errors
    )


def test_validator_rejects_stray_file_in_hybrid_dense_component(hybrid_store_path):
    from opengwasdb.layouts.hybrid.layout import dense_component_path

    (dense_component_path(hybrid_store_path) / "stray.txt").write_text("x")

    result = validate_store(hybrid_store_path)

    assert result.failed() == {"envelope.closed"}
    assert any(
        "unexpected store entry" in error and "stray.txt" in error for error in result.errors
    )


def test_validator_rejects_coverage_the_layout_cannot_carry(
    dense_store_path, hybrid_store_path
):
    # Issue #298: a Dense or Hybrid release has a cell for every (variant,
    # Analysis), so a coverage narrower than `gwas` would leave cells that are
    # indistinguishable from associations the source never reported. No builder
    # may write one, so a release that declares one is invalid rather than a
    # release whose missing cells mean something else.
    import json

    for store_path, layout in ((dense_store_path, "dense"), (hybrid_store_path, "hybrid")):
        manifest_path = store_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["association_coverage"] = "cis"
        manifest_path.write_text(json.dumps(manifest))

        result = validate_store(store_path)

        assert result.failed() == {"manifest.coverage"}, result.errors
        assert any(
            "cannot carry association_coverage=cis" in error
            and layout in error
            for error in result.errors
        ), result.errors


def test_validator_accepts_a_ragged_release_declaring_cis(ragged_store_path):
    # The same rule must not reject a Ragged release, whose sequence states its
    # own absences: all four coverages are representable there (issue #298).
    import json

    manifest_path = ragged_store_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["association_coverage"] = "cis"
    manifest_path.write_text(json.dumps(manifest))

    result = validate_store(ragged_store_path)

    assert result.ok, result.errors


def test_a_fixture_0_1_0_release_still_spelling_full_opens_and_validates(
    tmp_path, dense_store_path
):
    """The brief's case, on a real 0.1.0 release: every release built before
    #298 says `full`, and none of them needs rewriting.

    `relayout_as_0_1_0` writes a genuine 0.1.0 (Zarr v2) release from a current
    build, so this exercises the version a published Store actually carries
    rather than a hand-edited manifest.
    """
    import json

    from legacy_fixtures import relayout_as_0_1_0

    legacy = relayout_as_0_1_0(dense_store_path, tmp_path / "legacy-0.1.0.opengwasdb")
    manifest_path = legacy / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["format_version"] == "0.1.0"
    # The fixture must be the thing under test: an old release that says `full`.
    manifest["association_coverage"] = "full"
    manifest_path.write_text(json.dumps(manifest))

    store = open_store(legacy)
    assert store.manifest.association_coverage.value == "gwas"

    result = validate_store(legacy)
    assert result.ok, result.errors


def test_validator_accepts_a_release_still_spelling_full(dense_store_path):
    # Every release built before #298 says `full` outside the two-valued
    # vocabulary. It still opens and validates, as `gwas` (issue #298).
    import json

    manifest_path = dense_store_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["association_coverage"] = "full"
    manifest_path.write_text(json.dumps(manifest))

    result = validate_store(dense_store_path)

    assert result.ok, result.errors
    assert open_store(dense_store_path).manifest.association_coverage.value == "gwas"


def test_validator_accepts_clean_dense_ragged_hybrid_stores(
    dense_store_path, ragged_store_path, hybrid_store_path
):
    # Issue #80 acceptance criterion: a correctly-built release passes with no
    # false positives -- overview.html, the Hybrid dense/ subdirectory, and
    # its dense_to_shared.npy must all be recognised as legitimate, not just
    # rejected as unexpected.
    for store_path in (dense_store_path, ragged_store_path, hybrid_store_path):
        result = validate_store(store_path)
        assert result.ok, (store_path, result.errors)


def test_validator_rejects_retired_gene_id_column(dense_store_path):
    # Issue #81/ADR 0035: a built store's analyses.tsv carrying the retired
    # gene_id/gene_name columns (superseded by analysis_label/
    # trait_ontology_id/trait_ontology_label) is invalid against the current
    # schema, the same way ADR 0034's phenotype_id/phenotype_label/trait_id
    # retirement already is.
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    fieldnames = (*table.fieldnames, "gene_id")
    rows = tuple({**row, "gene_id": "ENSG00000000001"} for row in table.rows)
    write_analyses(analyses_path, type(table)(fieldnames=fieldnames, rows=rows))

    result = validate_store(dense_store_path)

    assert result.failed() == {"analyses.retired_columns"}
    assert any("retired column" in error and "gene_id" in error for error in result.errors)


def _give_every_analysis_an_invalid_effect_scale(store_path: Path) -> None:
    """An `analyses.tsv` defect no plane rule reads: the unrelated error beside another."""
    analyses_path = store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    for row in table.rows:
        row["stored_effect_scale"] = "kg"
    write_analyses(analyses_path, table)


def test_a_stray_file_no_longer_hides_an_invalid_vocabulary(dense_store_path):
    # Issue #55: a stray entry stopped every Dense check, so a release with two
    # independent defects reported one. Each rule now runs unless a rule it reads
    # after failed, and nothing reads a stray file.
    (dense_store_path / "traits.tsv.gz").write_bytes(b"stray")
    _give_every_analysis_an_invalid_effect_scale(dense_store_path)

    result = validate_store(dense_store_path)

    assert result.failed() == {"envelope.closed", "analyses.vocabularies"}


def test_a_z_overflow_table_defect_is_reported_beside_an_unrelated_error(dense_store_path):
    # Issue #55: the fixed-point z plan was checked only when no error of any kind
    # had been found before it, so an overflow table holding a non-finite exact
    # value hid behind any other defect. It needs only `z` and its plan.
    root = open_store(dense_store_path).arrays(mode="a")
    for name in ("z_overflow_index", "z_overflow_value"):
        if name in root:
            del root[name]
    create_array(
        root, "z_overflow_index", ArrayRole.EXCEPTION_TABLE,
        data=np.array([3, 5], dtype=np.int64), compressor=compressor(),
    )
    create_array(
        root, "z_overflow_value", ArrayRole.EXCEPTION_TABLE,
        data=np.array([np.nan, 41.0], dtype=np.float32), compressor=compressor(),
    )
    _give_every_analysis_an_invalid_effect_scale(dense_store_path)

    result = validate_store(dense_store_path)

    assert result.failed() == {"encoding.z", "analyses.vocabularies"}
    assert "data.zarr z overflow table contains non-finite values" in result.errors


def test_validator_accepts_duplicate_trait_ontology_id(dense_store_path):
    # Issue #72: duplicate trait_ontology_id is expected and valid (several
    # Analyses of the same curated Trait) as long as analysis_id stays unique.
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    rows = [{**row, "trait_ontology_id": "EFO:0000001"} for row in table.rows]
    write_analyses(analyses_path, type(table)(fieldnames=table.fieldnames, rows=tuple(rows)))

    result = validate_store(dense_store_path)

    assert result.ok, result.errors


def test_validator_rejects_invalid_original_sd_method(dense_store_path):
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    for row in table.rows:
        if row["analysis_id"] == "a1":
            row["original_sd_method"] = "bogus_method"
    write_analyses(analyses_path, table)

    result = validate_store(dense_store_path)

    assert result.failed() == {"analyses.vocabularies"}
    assert any("invalid original_sd_method" in error for error in result.errors)


def test_validator_accepts_valid_original_sd_method(dense_store_path):
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    for row in table.rows:
        row["original_sd_method"] = "declared_standardised"
    write_analyses(analyses_path, table)

    result = validate_store(dense_store_path)

    assert result.ok, result.errors


def test_validator_rejects_invalid_ancestry_assignment_method(dense_store_path):
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    for row in table.rows:
        if row["analysis_id"] == "a2":
            row["ancestry_assignment_method"] = "bogus_method"
    write_analyses(analyses_path, table)

    result = validate_store(dense_store_path)

    assert result.failed() == {"analyses.vocabularies"}
    assert any("invalid ancestry_assignment_method" in error for error in result.errors)


def test_validator_accepts_valid_ancestry_assignment_method(dense_store_path):
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    for row in table.rows:
        row["ancestry_assignment_method"] = "af_assigned"
    write_analyses(analyses_path, table)

    result = validate_store(dense_store_path)

    assert result.ok, result.errors


def test_validator_rejects_analyses_tsv_missing_referenced_analysis_index(dense_store_path):
    # store-format spec §20: analyses.tsv must cover every analysis_index
    # referenced by the store's positional indexing -- a gap (here, a2's
    # index jumps from 1 to 2, leaving 1 uncovered) is the one relationship
    # SQLite cannot enforce as a foreign key, since analyses.tsv is a
    # separate file.
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    for row in table.rows:
        if row["analysis_id"] == "a2":
            row["analysis_index"] = "2"
    write_analyses(analyses_path, table)

    result = validate_store(dense_store_path)

    assert "analyses.analysis_index" in result.failed()
    assert any(
        "analysis_index values are not exactly the range" in error for error in result.errors
    )


def test_validator_accepts_analyses_tsv_covering_every_analysis_index(dense_store_path):
    result = validate_store(dense_store_path)

    assert result.ok, result.errors


# ── Rho Matrix (issue 049) ────────────────────────────────────────────────────
# window_bp=1 keeps every variant (100/200/300bp spacing); z_thresh=10.0 makes
# every finite z "null" -- the fixture then has exactly one shared-null variant
# (rs1, the only row both a1 and a2 have data for) for the single a1/a2 pair.


def _build_fixture_rho(dense_store_path, min_nulls: int = 1) -> None:
    build_dense_rho(dense_store_path, window_bp=1, z_thresh=10.0, min_nulls=min_nulls, n_workers=1)


def test_validator_accepts_store_without_rho_group(dense_store_path):
    result = validate_store(dense_store_path)

    assert result.ok, result.errors


def test_validator_accepts_well_formed_rho_matrix(dense_store_path):
    _build_fixture_rho(dense_store_path)

    result = validate_store(dense_store_path)

    assert result.ok, result.errors


def test_validator_rejects_rho_wrong_packed_length(dense_store_path):
    _build_fixture_rho(dense_store_path)
    root = open_store(dense_store_path).arrays(mode="a")
    del root["rho"]["rho"]
    create_array(
        root["rho"],
        "rho",
        ArrayRole.RHO_ARRAY,
        data=np.array([0.1, 0.2], dtype="float16"),
        compressor=compressor(),
    )

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("rho array has" in error for error in result.errors)


def test_validator_rejects_rho_finite_with_insufficient_support(dense_store_path):
    _build_fixture_rho(dense_store_path)
    root = open_store(dense_store_path).arrays(mode="a")
    root["rho"]["n_null"][:] = 0  # below min_nulls=1, but rho stays finite

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("rho finiteness is inconsistent" in error for error in result.errors)


def test_validator_rejects_rho_out_of_range(dense_store_path):
    _build_fixture_rho(dense_store_path)
    root = open_store(dense_store_path).arrays(mode="a")
    root["rho"]["rho"][:] = np.array([1.5], dtype="float16")

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("rho contains values outside" in error for error in result.errors)


def test_validator_rejects_rho_missing_provenance_attr(dense_store_path):
    _build_fixture_rho(dense_store_path)
    root = open_store(dense_store_path).arrays(mode="a")
    del root["rho"].attrs["window_bp"]

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("missing provenance attrs" in error for error in result.errors)


def test_validator_rejects_rho_variant_index_length_mismatch(dense_store_path):
    _build_fixture_rho(dense_store_path)
    root = open_store(dense_store_path).arrays(mode="a")
    root["rho"].attrs["n_variants_used"] = 999

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("rho variant_index has" in error for error in result.errors)


# -- the SE exception table's precision (issue #355) -------------------------

#: A residual plan keeps at most 2% of an Analysis's cells as exceptions, so a
#: table the precision rule judges needs at least 50,000 cells in the Analysis.
ROUNDED_SE_VARIANTS = 60_000
ROUNDED_SE_EXCEPTIONS = 1_100


def _se_exception_records() -> list[NormalisedAssociation]:
    """One Analysis whose SE tracks its MAF, with a block the plan cannot code.

    The block's SE is `exp(2.2)` times the model's, so its residual sits beyond
    every candidate range and the build must store it exactly -- which is what
    makes the table's precision the thing under test.
    """
    frequencies = np.linspace(0.05, 0.95, ROUNDED_SE_VARIANTS, dtype=np.float64)
    records = []
    for row in range(ROUNDED_SE_VARIANTS):
        se = model_se(0, float(frequencies[row]), row)
        if row < ROUNDED_SE_EXCEPTIONS:
            se *= math.exp(2.2)
        records.append(
            NormalisedAssociation(
                analysis_id="a",
                variant=CanonicalVariant("1", row + 1, "A", "G"),
                z=1.0,
                se=se,
                eaf=float(frequencies[row]),
            )
        )
    return records


@pytest.fixture(scope="module")
def se_exception_stores(tmp_path_factory) -> tuple[Path, Path]:
    """A residual release with over 1,000 exceptions, and a copy of it rounded.

    The copy holds what a build before #355 wrote: the same cells marked `-127`,
    every table entry the `float16` rounding of its SE rather than the source's.
    """
    directory = tmp_path_factory.mktemp("rounded-se-exceptions")
    exact = directory / "exact.opengwasdb"
    build_dense_from_records(
        _se_exception_records(),
        exact,
        store_id="s",
        release_id="r",
        reference_assembly="GRCh38",
        chunk_shape=(1000, 1),
    )
    rounded = directory / "rounded.opengwasdb"
    shutil.copytree(exact, rounded)
    root = open_store(rounded).arrays(mode="r+")
    values = np.asarray(root[SeExceptionTable.value_name][:], dtype=np.float32)
    root[SeExceptionTable.value_name][:] = se_float16(values)
    return exact, rounded


def _exception_values(store: Path) -> np.ndarray:
    """The store's SE exception values, asserted to be enough for the rule to judge."""
    values = np.asarray(
        open_store(store).arrays(mode="r")[SeExceptionTable.value_name][:], dtype=np.float32
    )
    assert len(values) >= FLOAT16_EXACT_MINIMUM, (
        f"the fixture must hold {FLOAT16_EXACT_MINIMUM} exceptions, not {len(values)}, "
        "or the precision rule has nothing to judge"
    )
    return values


def test_an_exact_se_exception_table_passes_the_precision_rule(
    se_exception_stores: tuple[Path, Path],
) -> None:
    """A build that held the source's `float32` is the shape the rule expects."""
    exact, _rounded = se_exception_stores
    values = _exception_values(exact)
    exact_share = float(np.count_nonzero(se_float16(values).astype(np.float32) == values)) / len(
        values
    )
    assert exact_share < FLOAT16_EXACT_SHARE, (
        f"the exact fixture's share is {exact_share:.2%}, which the rule would flag"
    )
    result = validate_store(exact)
    assert "side_tables.se_exceptions_rounded" not in result.failed()
    assert not [w for w in result.warnings if "float16 plane stores" in w]
    assert result.ok, result.errors


def test_a_rounded_se_exception_table_fails_a_0_2_1_release(
    se_exception_stores: tuple[Path, Path],
) -> None:
    """Issue #355: every entry of a rounded table is what a `float16` plane stores.

    One rounded exception reads exactly like a right one, so nothing per cell can
    find this; the table's share is 100% against the 2**-13 a source value is
    `float16`-exact by chance.
    """
    _exact, rounded = se_exception_stores
    assert StoreManifest.load(rounded).format_version == CURRENT_FORMAT_VERSION
    values = _exception_values(rounded)
    assert (se_float16(values).astype(np.float32) == values).all(), (
        "every fixture entry must be a float16 rounding, or the share is not 100%"
    )

    result = validate_store(rounded)

    assert result.failed() == {"side_tables.se_exceptions_rounded"}
    message = result.errors[0]
    assert f"{len(values)} of {len(values)} se exceptions" in message
    assert "float16 plane stores" in message
    assert "rebuild the release" in message


def test_a_rounded_se_exception_table_warns_on_an_earlier_release(
    se_exception_stores: tuple[Path, Path], tmp_path: Path
) -> None:
    """A release from before the fix is reported, not failed: #367 owns it.

    Re-validating a published Store must say what it is without failing a Store
    nobody can rebuild in place.
    """
    _exact, rounded = se_exception_stores
    store = tmp_path / "restamped.opengwasdb"
    shutil.copytree(rounded, store)
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    manifest["format_version"] = "0.2.0"
    (store / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _exception_values(store)

    result = validate_store(store)

    assert "side_tables.se_exceptions_rounded" not in result.failed()
    warnings = [w for w in result.warnings if "float16 plane stores" in w]
    assert len(warnings) == 1
    assert "predates the fix" in warnings[0]
    assert result.ok, result.errors
