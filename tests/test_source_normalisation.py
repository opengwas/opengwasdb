"""How `build-dense` reads an association TSV into canonical associations.

The reader's own cases are in `test_association_tsv_reader.py`. These are the
orientation to canonical A1, and the Analysis metadata a manifest row carries into
`analyses.tsv` where the TSV once repeated it on every association row.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from opengwasdb.model.analyses import read_analyses
from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.association_tsv import (
    ASSOCIATION_TSV_CAPABILITY,
    SourceRowError,
)
from opengwasdb.readers.interface import StreamTally
from opengwasdb.readers.registry import resolve_reader


def _batches(path: Path):
    reader = resolve_reader(ASSOCIATION_TSV_CAPABILITY, path, StoredEffectScale.SD)
    (batch,) = reader.stream_batches(StreamTally())
    return batch


def test_source_reader_normalises_alid_and_flips_signed_statistics(source_path, tmp_path):
    from dense_sources import write_dense_manifest

    from opengwasdb.layouts.build_manifest import read_manifest

    rows = read_manifest(write_dense_manifest(source_path))
    batch = _batches(Path(rows[0].file_path))  # analysis a1

    assert batch.alid.tolist() == ["1:100:A:G", "1:200:C:T"]
    assert batch.z.tolist() == [2.0, -3.0]
    assert batch.se.tolist() == [0.1, 0.2]


def test_source_reader_derives_z_from_beta_when_needed(tmp_path):
    source = tmp_path / "beta.tsv"
    source.write_text(
        "chromosome\tposition\teffect_allele\tother_allele\tbeta\tse\n1\t10\tG\tA\t0.6\t0.2\n",
        encoding="utf-8",
    )

    batch = _batches(source)

    assert batch.alid.tolist() == ["1:10:A:G"]
    assert batch.z[0] == pytest.approx(-3.0)
    assert batch.se[0] == 0.2


def test_source_reader_supports_gzipped_tsv(tmp_path):
    source = tmp_path / "source.tsv.gz"
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write(
            "chromosome\tposition\teffect_allele\tother_allele\tz\tse\n1\t10\tA\tG\t2\t0.5\n"
        )

    batch = _batches(source)

    assert batch.alid.tolist() == ["1:10:A:G"]
    assert batch.z.tolist() == [2.0]


def test_source_reader_rejects_invalid_rows(tmp_path):
    source = tmp_path / "invalid.tsv"
    source.write_text(
        "chromosome\tposition\teffect_allele\tother_allele\tz\tse\n1\t10\tA\tA\t1\t0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(SourceRowError, match="identical"):
        _batches(source)


def _analysis_file(directory: Path, name: str) -> str:
    path = directory / f"{name}.tsv"
    path.write_text(
        "chromosome\tposition\teffect_allele\tother_allele\tz\tse\n1\t10\tA\tG\t2\t0.5\n",
        encoding="utf-8",
    )
    return str(path)


_ATTRIBUTION = {
    "analysis_label": "Height", "trait_ontology_id": "EFO:0001073",
    "trait_ontology_label": "body height", "license": "CC0",
    "publication_doi": "10.1000/xyz", "publication_pmid": "12345678",
    "consortium": "GIANT", "first_author": "J. Smith",
}


@pytest.fixture(scope="module")
def analyses(tmp_path_factory) -> dict[str, dict[str, str]]:
    """The `analyses.tsv` rows, by Analysis, of one Store built from a manifest whose
    rows declare everything (`full`), nothing (`bare`) and another effect scale."""
    from opengwasdb.build.observed import build_dense_observed_from_manifest

    directory = tmp_path_factory.mktemp("metadata")
    rows = [
        {"analysis_id": "full", "stored_effect_scale": "sd", **_ATTRIBUTION},
        {"analysis_id": "bare", "stored_effect_scale": "sd"},
        {"analysis_id": "binary", "stored_effect_scale": "log_or"},
    ]
    columns = list({key for row in rows for key in row} | {"source_file", "original_sd_method"})
    manifest = directory / "manifest.tsv"
    manifest.write_text(
        "\t".join(columns) + "\n"
        + "".join(
            "\t".join(
                {
                    **row,
                    "source_file": _analysis_file(directory, row["analysis_id"]),
                    "original_sd_method": "declared_standardised",
                }.get(c, "")
                for c in columns
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    store = directory / "store.opengwasdb"
    build_dense_observed_from_manifest(manifest, store, store_id="s", release_id="r")
    return {row["analysis_id"]: row for row in read_analyses(store / "analyses.tsv").rows}


def test_trait_ontology_and_attribution_columns_resolved_when_present(analyses):
    assert {key: analyses["full"][key] for key in _ATTRIBUTION} == _ATTRIBUTION


def test_trait_ontology_and_attribution_columns_blank_when_absent(analyses):
    for column in (
        "trait_ontology_id", "trait_ontology_label", "license", "publication_doi",
        "publication_pmid", "consortium", "first_author",
    ):
        assert analyses["bare"][column] == ""


def test_each_analysis_has_the_effect_scale_its_manifest_row_declares(analyses):
    assert {name: row["stored_effect_scale"] for name, row in analyses.items()} == {
        "full": "sd", "bare": "sd", "binary": "log_or",
    }
