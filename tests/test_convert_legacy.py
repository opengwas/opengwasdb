"""`ogdb rewrite-release --convert-legacy` converts a 0.2.0 Dense release to 0.3.0 (#492).

The sources are the Dense `store_reader_digests` releases in the shape of the hosted
0.2.0 Stores (`format_0_2_0.as_0_2_0`): stamped 0.2.0, no `n_missing_se_out_of_range`,
a blank `source_alid` on every row and a chain label in provenance, with SEs above
65,504 stored as exact exceptions (`plant_se`), one of them a top hit. A conversion masks
and counts exactly those cells, names an unlifted release's own ALIDs (state 1) or keeps
a lifted one's none (state 0), records its deviations, proves every answer equal to the
source's with exactly those changes, and validates with the recorded rules as warnings.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from format_0_2_0 import a_top_hit, as_0_2_0, plant_se, sorted_indel
from format_0_2_3 import as_0_2_3, write_variant_offsets
from typer.testing import CliRunner

from opengwasdb.cli.main import app
from opengwasdb.query import query_store
from opengwasdb.store.legacy_conversion import LegacyConversion
from opengwasdb.store.open import open_store
from opengwasdb.store.rewrite import RewriteError, RewriteVerificationError, rewrite_release
from opengwasdb.store_reader.analyses import read_analyses
from opengwasdb.store_reader.associations import Cells, DenseGridReader, State
from opengwasdb.store_reader.source_alid_sidecar import SourceAlidSidecar
from opengwasdb.validation import validate_store
from opengwasdb.validation.rule import Severity
from opengwasdb.variants.legacy_table import iter_variant_records

UNLIFTED = LegacyConversion(lifted=False)
LIFTED = LegacyConversion(lifted=True)
DENSE = ["dense_observed", "dense_completed"]


@pytest.fixture(scope="module")
def sources(digest_releases: dict[str, Path], tmp_path_factory: pytest.TempPathFactory):
    """Each release as a 0.2.3 build wrote it. Never written to: copy one to change it."""
    root = tmp_path_factory.mktemp("legacy-sources")
    return {
        name: as_0_2_3(Path(shutil.copytree(path, root / name)))
        for name, path in digest_releases.items()
    }


def _files(release: Path) -> dict[str, str]:
    return {
        p.relative_to(release).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(release.rglob("*"))
        if p.is_file()
    }


def _quiet_cell(release: Path, avoid: tuple[int, int]) -> tuple[int, int]:
    """An observed cell no tier holds, other than `avoid`."""
    q = query_store(release)
    try:
        tiered = q.top_hits(threshold=5e-4)
        held = set(
            zip(tiered["variant_index"].tolist(), tiered["analysis_index"].tolist(), strict=True)
        )
        for analysis_id in [row["analysis_id"] for row in q.analyses_table().values()]:
            found = q.analysis(analysis_id, observed_only=True)
            for row, col in zip(found["variant_index"].tolist(), found["analysis_index"].tolist(),
                                strict=True):
                if (row, col) not in held and (row, col) != avoid:
                    return int(row), int(col)
    finally:
        q.close()
    raise AssertionError("the fixture must hold an observed cell no tier holds")


def _legacy(sources: dict[str, Path], tmp_path: Path, name: str) -> tuple[Path, list]:
    """`name` as a 0.2.0 release, with two SEs above 65,504 planted: one a top hit."""
    source = as_0_2_0(Path(shutil.copytree(sources[name], tmp_path / "source")))
    hit = a_top_hit(source)
    quiet = _quiet_cell(source, hit)
    plant_se(source, *hit, 1.0e5)
    plant_se(source, *quiet, 70_000.0)
    return source, [hit, quiet]


def _states(release: Path, cells: list) -> list[int]:
    store = open_store(release)
    completed = store.manifest.completion_state.value == "reference_completed"
    grid = DenseGridReader.open(
        store.arrays(), store.manifest.encoding, requires_mask=completed
    )
    rows, cols = (np.asarray(axis, dtype=np.int64) for axis in zip(*cells, strict=True))
    found = grid.inspect(Cells.pairs(rows, cols))
    return [int(state) for state in found.state], found.se


def _count(release: Path) -> dict[str, str]:
    return {
        row["analysis_index"]: row.get("n_missing_se_out_of_range", "<absent>")
        for row in read_analyses(release / "analyses.tsv").rows
    }


def _missing_counted(release: Path) -> int:
    """Every `completion_quality` row's `n_missing`, summed."""
    from opengwasdb.index.release_index import ReleaseIndex

    with ReleaseIndex(release) as index:
        return sum(int(row[3]) for row in index.completion_quality())


def _hits(release: Path) -> set[tuple[int, int]]:
    q = query_store(release)
    try:
        found = q.top_hits(threshold=5e-4)
        return set(
            zip(found["variant_index"].tolist(), found["analysis_index"].tolist(), strict=True)
        )
    finally:
        q.close()


@pytest.mark.parametrize("name", DENSE)
def test_an_unlifted_0_2_0_release_is_masked_counted_and_named(sources, tmp_path, name):
    source, cells = _legacy(sources, tmp_path, name)
    states, se = _states(source, cells)
    assert states == [State.OBSERVED, State.OBSERVED] and (se > 65_504).all(), "planted"
    assert cells[0] in _hits(source) and cells[1] not in _hits(source)
    assert set(_count(source).values()) == {"<absent>"}
    rows = list(iter_variant_records(source / "variants.tsv.gz"))
    assert rows and all(r.source_alid is None for r in rows), "every source row is blank"
    before = _files(source)

    destination = rewrite_release(source, tmp_path / "converted", legacy=UNLIFTED)

    assert _files(source) == before, "the conversion wrote its source"
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == "0.3.0"
    record = manifest["provenance"]["converted_from"]
    assert record["source_format_version"] == "0.2.0"
    assert record["deviations"] == ["variant_table.indel_order"]
    assert (record["lifted"], record["se_out_of_range_masked"]) == (False, 2)
    assert manifest["provenance"]["format_rewrite"]["steps"][:2] == [
        "se_out_of_range", "own_source_alid",
    ]
    assert _states(destination, cells)[0] == [State.MISSING, State.MISSING]
    counted = {str(c): str(sum(1 for _, a in cells if a == c)) for c in range(2)}
    assert _count(destination) == counted
    assert _hits(destination) == _hits(source) - {cells[0]}
    hit_column = {r["analysis_index"]: r["n_hits_5e4"] for r in read_analyses(
        destination / "analyses.tsv").rows}
    assert int(hit_column[str(cells[0][1])]) == int({r["analysis_index"]: r["n_hits_5e4"] for r in
        read_analyses(source / "analyses.tsv").rows}[str(cells[0][1])]) - 1
    sidecar = SourceAlidSidecar.open(destination)
    assert sidecar is not None and set(np.asarray(sidecar.state).tolist()) == {1}
    if name == "dense_completed":
        assert _missing_counted(destination) == _missing_counted(source) + 2
    q = query_store(destination)
    try:
        assert all(row["source_alid"] == row["alid"] for row in q.variants_table().values())
    finally:
        q.close()
    result = validate_store(destination)
    assert result.ok, result.errors


def test_a_lifted_release_keeps_state_0_and_the_rule_warns_naming_the_conversion(
    sources, tmp_path
):
    source, _cells = _legacy(sources, tmp_path, "dense_observed")

    destination = rewrite_release(source, tmp_path / "converted", legacy=LIFTED)

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["converted_from"]["deviations"] == [
        "variant_table.indel_order", "variant_table.source_alid",
    ]
    sidecar = SourceAlidSidecar.open(destination)
    assert sidecar is not None and set(np.asarray(sidecar.state).tolist()) == {0}
    result = validate_store(destination)
    assert result.ok, result.errors
    (finding,) = [f for f in result.findings if f.rule == "variant_table.source_alid"]
    assert finding.severity is Severity.WARNING
    assert "converted this release from format_version 0.2.0" in finding.message


def test_without_the_recorded_deviation_the_same_release_fails(sources, tmp_path):
    """The warning is the record's, not the conversion's: drop the rule from
    `deviations` and the lifted release fails it again."""
    source, _cells = _legacy(sources, tmp_path, "dense_observed")
    destination = rewrite_release(source, tmp_path / "converted", legacy=LIFTED)
    path = destination / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["provenance"]["converted_from"]["deviations"] = ["variant_table.indel_order"]
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_store(destination).failed() == {"variant_table.source_alid"}


def test_a_converted_release_fails_every_other_rule_as_before(sources, tmp_path):
    source, _cells = _legacy(sources, tmp_path, "dense_observed")
    destination = rewrite_release(source, tmp_path / "converted", legacy=LIFTED)
    table = read_analyses(destination / "analyses.tsv")
    lines = (destination / "analyses.tsv").read_text(encoding="utf-8").splitlines()
    column = table.fieldnames.index("n_missing_se_out_of_range")
    blanked = [
        "\t".join(v if i != column or n == 0 else "" for i, v in enumerate(line.split("\t")))
        for n, line in enumerate(lines)
    ]
    (destination / "analyses.tsv").write_text("\n".join(blanked) + "\n", encoding="utf-8")

    result = validate_store(destination)

    assert "analyses.se_out_of_range_count" in result.failed()


def _skip_last_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    from opengwasdb.store import legacy_masking

    real = legacy_masking.mask_cells

    def all_but_last(group, positions, *args):
        real(group, positions[:-1], *args)

    monkeypatch.setattr(legacy_masking, "mask_cells", all_but_last)


def test_the_proof_refuses_a_conversion_that_masks_one_cell_too_few(
    sources, tmp_path, monkeypatch
):
    source, _cells = _legacy(sources, tmp_path, "dense_completed")
    _skip_last_cell(monkeypatch)

    with pytest.raises(RewriteVerificationError) as refused:
        rewrite_release(source, tmp_path / "converted", legacy=UNLIFTED)

    assert any(d.startswith("shapes/") for d in refused.value.differences)
    assert not (tmp_path / "converted").exists()


def test_the_proof_refuses_a_conversion_that_names_another_source_alid(
    sources, tmp_path, monkeypatch
):
    from opengwasdb.variants import axis

    source, _cells = _legacy(sources, tmp_path, "dense_observed")
    real = axis.rewrite_source_alid_column

    def first_row_blank(store, name):
        return real(store, lambda row, alid, held: None if row == 0 else name(row, alid, held))

    monkeypatch.setattr(axis, "rewrite_source_alid_column", first_row_blank)

    with pytest.raises(RewriteVerificationError) as refused:
        rewrite_release(source, tmp_path / "converted", legacy=UNLIFTED)

    assert "variants_table/source_alid" in refused.value.differences


def test_a_subset_holding_a_masked_cell_answers_it_missing(sources, tmp_path):
    from test_rewrite_release import _with_subset

    staged = Path(shutil.copytree(sources["dense_completed"], tmp_path / "with-subset"))
    _with_subset(staged, tmp_path)
    source = as_0_2_0(Path(shutil.copytree(staged, tmp_path / "source")))
    plant_se(source, 0, 0, 1.0e5)

    destination = rewrite_release(source, tmp_path / "converted", legacy=UNLIFTED)

    q = query_store(destination)
    try:
        answered = q.analysis(q.analyses_table()[0]["analysis_id"], indexed_subset="hm3")
    finally:
        q.close()
    assert 0 not in answered["variant_index"].tolist()
    result = validate_store(destination)
    assert result.ok, result.errors


def test_a_sorted_indel_is_carried_as_written(tmp_path):
    """A 0.2.0 release spells an indel sorted; the conversion carries the ALID byte for
    byte and records the rule, which this row cannot trip: the table agrees with itself."""
    from test_indel_alid import DELETION_AT_950000, _observed_deletion_store

    built = _observed_deletion_store(tmp_path, "dense")
    source = as_0_2_0(as_0_2_3(Path(shutil.copytree(built, tmp_path / "source"))))
    spelt = sorted_indel(source, DELETION_AT_950000)

    destination = rewrite_release(source, tmp_path / "converted", legacy=UNLIFTED)

    alids = [r.alid for r in iter_variant_records(destination / "variants.tsv.gz")]
    assert spelt in alids and DELETION_AT_950000 not in alids
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["converted_from"]["deviations"] == [
        "variant_table.indel_order"
    ]
    result = validate_store(destination)
    assert result.ok, result.errors


def test_a_recorded_indel_order_deviation_warns_and_an_unrecorded_one_fails(tmp_path):
    from test_indel_alid import _observed_deletion_store, _rewrite_table, _sorted_deletion

    store = _observed_deletion_store(tmp_path, "dense")
    _rewrite_table(store, _sorted_deletion)
    assert validate_store(store).failed() == {"variant_table.indel_order"}, "a fault"
    path = store / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for recorded, fails in (
        (["variant_table.source_alid"], {"variant_table.indel_order"}),
        (["variant_table.indel_order"], set()),
    ):
        manifest["provenance"]["converted_from"] = {
            "source_format_version": "0.2.0", "source_release_id": "r", "deviations": recorded,
        }
        path.write_text(json.dumps(manifest), encoding="utf-8")
        result = validate_store(store)
        assert result.failed() == fails
    (finding,) = [f for f in result.findings if f.rule == "variant_table.indel_order"]
    assert finding.severity is Severity.WARNING and "--convert-legacy" in finding.message


def test_a_0_2_0_dense_release_is_refused_without_the_flag(sources, tmp_path):
    source, _cells = _legacy(sources, tmp_path, "dense_observed")
    with pytest.raises(RewriteError, match="--convert-legacy"):
        rewrite_release(source, tmp_path / "converted")
    assert not (tmp_path / "converted").exists()


def test_a_0_2_3_release_is_refused_with_the_flag(sources, tmp_path):
    with pytest.raises(RewriteError, match="without --convert-legacy"):
        rewrite_release(sources["dense_observed"], tmp_path / "converted", legacy=UNLIFTED)


@pytest.mark.parametrize("name", ["ragged_observed", "hybrid_completed"])
def test_a_legacy_release_not_dense_is_refused(sources, tmp_path, name):
    source = Path(shutil.copytree(sources[name], tmp_path / "source"))
    for manifest in source.rglob("manifest.json"):
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace('"0.2.3"', '"0.2.0"'), "utf-8"
        )
    with pytest.raises(RewriteError, match="only a Dense release"):
        rewrite_release(source, tmp_path / "converted", legacy=UNLIFTED)


def test_an_imputed_cell_above_the_bound_is_refused(sources, tmp_path):
    source = as_0_2_0(Path(shutil.copytree(sources["dense_completed"], tmp_path / "source")))
    q = query_store(source)
    try:
        found = q.range_phewas("1", 1, 10**9)
        imputed = found["association_status"] == "imputed"
        assert imputed.any(), "the fixture must hold an imputed cell"
        row, col = (int(found[k][imputed][0]) for k in ("variant_index", "analysis_index"))
    finally:
        q.close()
    plant_se(source, row, col, 1.0e5)

    with pytest.raises(RewriteError, match="imputed cell"):
        rewrite_release(source, tmp_path / "converted", legacy=UNLIFTED)


def test_panel_rows_that_are_not_the_completions_count_are_refused(sources, tmp_path):
    source = as_0_2_0(Path(shutil.copytree(sources["dense_completed"], tmp_path / "source")))
    path = source / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["provenance"]["completion"]["n_variants_new"] += 1
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RewriteError, match="n_variants_new"):
        rewrite_release(source, tmp_path / "converted", legacy=UNLIFTED)


def test_an_unlifted_conversion_of_a_table_naming_another_alid_is_refused(sources, tmp_path):
    from opengwasdb.variants.axis import rewrite_source_alid_column

    source = as_0_2_0(Path(shutil.copytree(sources["dense_observed"], tmp_path / "source")))
    rewrite_source_alid_column(source, lambda row, alid, held: "1:1:C:T" if row == 5 else None)
    write_variant_offsets(source)

    with pytest.raises(RewriteError, match="converted with --lifted"):
        rewrite_release(source, tmp_path / "converted", legacy=UNLIFTED)
    assert not (tmp_path / "converted").exists()


def test_the_cli_needs_a_lift_answer_and_converts_with_one(sources, tmp_path):
    source, _cells = _legacy(sources, tmp_path, "dense_observed")
    runner = CliRunner()
    args = ["rewrite-release", str(source), str(tmp_path / "converted"), "--convert-legacy"]

    asked = runner.invoke(app, args)
    assert asked.exit_code == 2 and "--lifted or --not-lifted" in asked.output
    converted = runner.invoke(app, [*args, "--not-lifted"])
    assert converted.exit_code == 0, converted.output
    assert "Published" in converted.output
