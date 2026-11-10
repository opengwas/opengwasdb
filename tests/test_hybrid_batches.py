"""A Hybrid build's Pass 2 reads a reader's association batches too (#318).

FinnGen R13 offers batches. Two Hybrid builds of the same FinnGen sources, one
whose Pass 2 reads the batches and one that reads the associations one at a
time, must write the same spills -- Dense, Overflow, off-reference and folded
-- the same Admission record and the same release.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from extraction_fixtures import extract_from_manifest
from finngen_cases import HEADER, _row
from hybrid_pass2_fixtures import captured_spills
from test_column_map_build import _store_contents
from test_finngen_batches import RowsOnly

from opengwasdb.admission import admission_record_for
from opengwasdb.layouts.hybrid import build as hybrid_build
from opengwasdb.readers import registry
from opengwasdb.readers.finngen import FINNGEN_R13_CAPABILITY, FinnGenR13Reader

_ROWS_ONLY = "opengwasdb.test-finngen-rows"

#: ``(position, ref, alt, beta)`` on chromosome 1.
ALPHA = [
    (100, "A", "G", "0.5"),
    (200, "C", "T", "-0.3"),
    (300, "g", "a", "0.2"),  # lower case
    (400, "AT", "A", "0.1"),  # a deletion: Overflow, the panel being SNVs
    (500, "A", "ACGT", "-0.4"),  # an insertion
    (600, "T", "C", "0.6"),
    (600, "T", "C", "0.6"),  # agrees: collapsed
    (700, "G", "T", "0.7"),
]
BETA = [
    (100, "A", "G", "0.55"),
    (300, "G", "A", "0.25"),
    (400, "AT", "A", "0.15"),
    (800, "C", "A", "0.8"),  # only BETA names these: off-reference in a single pass
    (900, "A", "AT", "-0.9"),
    (1000, "T", "G", "1.0"),
    (1000, "T", "G", "-1.0"),  # disagrees: both dropped
]
SNVS = ["1:100:A:G", "1:200:C:T", "1:300:A:G", "1:600:C:T", "1:700:G:T"]


def _source(path: Path, rows: list[tuple[int, str, str, str]]) -> Path:
    lines = [HEADER, *(_row("1", str(p), r, a, b, rsids="") for p, r, a, b in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _manifest(directory: Path, capability: str, names: list[str]) -> Path:
    path = directory / f"manifest-{capability}-{'-'.join(names)}.tsv"
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\toriginal_sd\tsource_reader_capability\tsource_assembly"
    ]
    lines += [
        f"{name}\t{directory / f'{name}.tsv'}\t{name}\t1000\tlog_or\tbinary_trait\t\t"
        f"{capability}\thg38"
        for name in names
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build(
    directory: Path, capability: str, single_pass: bool, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, dict[int, dict[str, np.ndarray]]], Path, Path]:
    panel = directory / "panel.txt"
    panel.write_text("\n".join(SNVS) + "\n", encoding="utf-8")
    options: dict[str, object] = {"reference_panel": panel}
    if single_pass:
        reference = directory / f"alpha-{capability}.variant-ref.tsv.gz"
        extract_from_manifest(_manifest(directory, capability, ["alpha"]), reference)
        options["variant_reference"] = reference
    store = directory / f"{capability}-{single_pass}.opengwasdb"
    with captured_spills(monkeypatch) as spills:
        hybrid_build.build_hybrid_from_vcf_manifest(
            _manifest(directory, capability, ["alpha", "beta"]), store,
            store_id="s", release_id="r", on_rsid_conflict="drop", **options,
        )
    return spills, admission_record_for(store), store


def _refuse_rows(self: FinnGenR13Reader, tally: object = None) -> None:
    raise AssertionError("Pass 2 read a reader that offers batches one association at a time")


@pytest.mark.parametrize("single_pass", [False, True], ids=["two-pass", "single-pass"])
def test_hybrid_pass2_reads_batches_as_it_reads_associations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, single_pass: bool
) -> None:
    _source(tmp_path / "alpha.tsv", ALPHA)
    _source(tmp_path / "beta.tsv", BETA)
    monkeypatch.setitem(
        registry._READERS, _ROWS_ONLY, lambda path, scale: RowsOnly(FinnGenR13Reader(path, scale))
    )
    rows, rows_record, rows_store = _build(tmp_path, _ROWS_ONLY, single_pass, monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(FinnGenR13Reader, "stream_associations", _refuse_rows)
        batches, batches_record, batches_store = _build(
            tmp_path, FINNGEN_R13_CAPABILITY, single_pass, monkeypatch
        )

    assert rows["dense"] and rows["overflow"][0]["variant_index"].size
    if single_pass:
        assert any(spill["keys"].size for spill in rows["off_reference"].values())
    for part, columns in rows.items():
        assert sorted(columns) == sorted(batches[part]), part
        for column, arrays in columns.items():
            for name, values in arrays.items():
                other = batches[part][column][name]
                assert values.dtype == other.dtype and values.tobytes() == other.tobytes(), (
                    part, column, name,
                )
    record = (rows_record / "admission.json").read_bytes()
    assert b"duplicate_conflicting" in record and b"duplicate_collapsed" in record
    assert record == (batches_record / "admission.json").read_bytes()
    analyses = ["alpha", "beta"]
    assert _store_contents(rows_store, analyses) == _store_contents(batches_store, analyses)
