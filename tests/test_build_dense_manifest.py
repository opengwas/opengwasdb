"""`build-dense` reads a build manifest: what it takes from a row, and what it refuses."""

from __future__ import annotations

from pathlib import Path

import pytest

from opengwasdb.admission.rules import AnalysisRules
from opengwasdb.build.observed import build_dense_observed_from_manifest
from opengwasdb.layouts.dense import AnalysisSource, build_dense_observed_store
from opengwasdb.model.analyses import Analysis, read_analyses
from opengwasdb.query import query_store
from opengwasdb.readers.interface import StreamTally
from opengwasdb.validation import validate_store

_HEADER = "chromosome\tposition\teffect_allele\tother_allele\tz\tse\n"
_ROW = "1\t100\tA\tG\t2.0\t0.2\n"
_BUILD = {"store_id": "s", "release_id": "r", "reference_assembly": "GRCh37"}


def _analysis(tmp_path: Path, name: str, body: str = _ROW) -> dict[str, str]:
    path = tmp_path / f"{name}.tsv"
    path.write_text(_HEADER + body, encoding="utf-8")
    return {
        "analysis_id": name, "source_file": str(path), "stored_effect_scale": "sd",
        "original_sd_method": "declared_standardised",
    }


def _build(tmp_path: Path, *rows: dict[str, str], **options: str) -> Path:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "\t".join(columns) + "\n"
        + "".join("\t".join(row.get(c, "") for c in columns) + "\n" for row in rows),
        encoding="utf-8",
    )
    store = tmp_path / "store.opengwasdb"
    build_dense_observed_from_manifest(
        manifest, store, store_id="s", release_id="r", admission_record=tmp_path / "adm", **options
    )
    return store


def test_the_stores_columns_are_the_manifests_rows_in_order(tmp_path: Path) -> None:
    store = _build(tmp_path, _analysis(tmp_path, "zeta"), _analysis(tmp_path, "alpha"))

    assert [r["analysis_id"] for r in read_analyses(store / "analyses.tsv").rows] == [
        "zeta", "alpha",
    ]


def test_an_sd_rescaling_method_divides_the_analysis_standard_errors(tmp_path: Path) -> None:
    """The phenotype SD standardises an Analysis's SEs as in `build-dense-vcf` (#18)."""
    rescaled = {
        **_analysis(tmp_path, "rescaled"),
        "original_sd_method": "source_provided", "original_sd": "2",
    }
    store = _build(tmp_path, rescaled, _analysis(tmp_path, "plain"))

    found = query_store(store).lookup(["1:100:A:G"], ["rescaled", "plain"])
    se = dict(zip(found["analysis_index"].tolist(), found["se"].tolist(), strict=True))
    assert se[0] == pytest.approx(0.1, rel=1e-3)
    assert se[1] == pytest.approx(0.2, rel=1e-3)


def test_a_row_in_another_assembly_is_refused_because_build_dense_does_not_lift(
    tmp_path: Path,
) -> None:
    row = {**_analysis(tmp_path, "a"), "source_assembly": "hg38"}
    with pytest.raises(ValueError, match="analysis 'a' is hg38, but build-dense does not lift"):
        _build(tmp_path, row)


def test_a_row_in_the_reference_assembly_is_accepted_under_any_spelling(tmp_path: Path) -> None:
    row = {**_analysis(tmp_path, "a"), "source_assembly": "b38"}
    _build(tmp_path, row, reference_assembly="GRCh38")


def test_a_row_naming_another_reader_is_refused(tmp_path: Path) -> None:
    row = {**_analysis(tmp_path, "a"), "source_reader_capability": "opengwasdb.gwas-ssf"}
    with pytest.raises(ValueError, match="does not read; it reads 'opengwasdb.association-tsv'"):
        _build(tmp_path, row)


def test_a_manifest_naming_an_analysis_twice_is_refused_naming_it_and_its_rows(
    tmp_path: Path,
) -> None:
    row = _analysis(tmp_path, "a")
    with pytest.raises(ValueError, match=r"analysis 'a' is named by rows 2 and 3"):
        _build(tmp_path, row, row, _analysis(tmp_path, "z"))


def test_the_writer_refuses_an_analysis_named_twice(tmp_path: Path) -> None:
    source = AnalysisSource(
        analysis=Analysis(analysis_id="a", stored_effect_scale="sd"),
        rules=AnalysisRules("a"),
        se_divisor=1.0,
        batches=[],
        tally=StreamTally(),
    )
    with pytest.raises(ValueError, match="Analyses named more than once: a"):
        build_dense_observed_store([source, source], tmp_path / "store.opengwasdb", **_BUILD)


def test_an_analysis_whose_file_holds_no_association_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="analysis 'empty'.*holds no associations"):
        _build(tmp_path, _analysis(tmp_path, "empty", body=""))


def test_fidelity_compares_the_source_after_the_declared_standardisation(tmp_path: Path) -> None:
    """A Store built from a manifest declaring SD 2 holds SE 0.1 of a source SE of 0.2,
    and validates against that manifest."""
    rescaled = {
        **_analysis(tmp_path, "rescaled"),
        "original_sd_method": "source_provided", "original_sd": "2",
    }
    store = _build(tmp_path, rescaled)

    result = validate_store(store, source=tmp_path / "manifest.tsv")

    assert result.ok, result.errors


def test_fidelity_does_not_fail_a_row_the_build_dropped_as_malformed(tmp_path: Path) -> None:
    """The build logs a row whose allele no ALID can spell and leaves it out; it
    is not a source association the Store failed to keep."""
    store = _build(
        tmp_path, _analysis(tmp_path, "a", body=_ROW + "1\t200\tA\tN\t2.0\t0.2\n")
    )

    result = validate_store(store, source=tmp_path / "manifest.tsv")

    assert result.ok, result.errors


def test_fidelity_refuses_a_bare_association_tsv(tmp_path: Path) -> None:
    store = _build(tmp_path, _analysis(tmp_path, "a"))

    result = validate_store(store, source=tmp_path / "a.tsv")

    assert result.failed() == {"fidelity.source"}
    assert "names no Analysis" in result.errors[0]


def test_fidelity_refuses_a_manifest_with_no_rows(tmp_path: Path) -> None:
    store = _build(tmp_path, _analysis(tmp_path, "a"))
    empty = tmp_path / "empty-manifest.tsv"
    empty.write_text(
        "analysis_id\tsource_file\tstored_effect_scale\toriginal_sd_method\n", encoding="utf-8"
    )

    result = validate_store(store, source=empty)

    assert result.failed() == {"fidelity.source"}
    assert "contains no rows" in result.errors[0]


def test_fidelity_refuses_a_source_that_names_nothing_to_check(tmp_path: Path) -> None:
    store = _build(tmp_path, _analysis(tmp_path, "a"))

    result = validate_store(store, source=[])

    assert result.failed() == {"fidelity.source"}
    assert "names no Analysis to check" in result.errors[0]
