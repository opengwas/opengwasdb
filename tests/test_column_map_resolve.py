"""`resolve-analyses` reading a collection through a Column Map (issue #281).

The map is part of how a Resolution Record was measured: the same bytes read
through a different map are a different measurement. So its checksum is in the
record's fingerprint, and `--resume` never reuses a record resolved under
another map.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_resolve_manifest import _STUDY_N, _mixture, _se_for, _write_panel, _write_ssf
from typer.testing import CliRunner

from opengwasdb.cli.main import app

_MAP = """\
column_map_version = 1
name = "plain-provider"

[columns]
chromosome = "chr"
position = "bp"
effect_allele = "a1"
other_allele = "a2"
beta = "b"
standard_error = "se"
frequency = "freq"

[conventions]
missing_values = ["", "NA"]
unmapped_columns = "refuse"
alleles = "effect_and_other"
frequency_allele = "effect_allele"
position_notation = "integer"
chromosome_spelling = "adr-0052"
genome_build = "GRCh38"
"""

_HEADER = "analysis_id\tsource_file\tsource_reader_capability\tstored_effect_scale\t"
_HEADER += "original_sd_method\tsample_size\n"


def _write_provider(path: Path, freqs: np.ndarray) -> Path:
    """The `_write_ssf` source, in a provider's own column names and order."""
    lines = ["se\tfreq\tchr\tbp\ta2\ta1\tb\n"]
    for index, (frequency, se) in enumerate(zip(freqs, _se_for(freqs), strict=True)):
        lines.append(f"{se:.6g}\t{frequency:.6g}\t1\t{1000 + index}\tC\tA\t0.2\n")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.writelines(lines)
    return path


@pytest.fixture
def setup(tmp_path: Path) -> dict[str, Any]:
    ref_path, grp_path, ref = _write_panel(tmp_path / "reference")
    freqs = _mixture(ref, {"North-West Europe": 0.8, "North-East Europe": 0.2})
    provider = _write_provider(tmp_path / "provider.tsv.gz", freqs)
    ssf = _write_ssf(tmp_path / "same.h.tsv.gz", freqs)
    column_map = tmp_path / "provider.toml"
    column_map.write_text(_MAP, encoding="utf-8")
    mapped = tmp_path / "mapped.tsv"
    mapped.write_text(
        _HEADER + f"MAPPED\t{provider}\topengwasdb.column-map\tsd\t"
        f"estimated_from_source_maf\t{int(_STUDY_N)}\n",
        encoding="utf-8",
    )
    reference = tmp_path / "reference.tsv"
    reference.write_text(
        _HEADER + f"MAPPED\t{ssf}\topengwasdb.gwas-ssf\tsd\t"
        f"estimated_from_source_maf\t{int(_STUDY_N)}\n",
        encoding="utf-8",
    )
    return {
        "tmp": tmp_path, "ref": ref_path, "groups": grp_path, "map": column_map,
        "mapped": mapped, "reference": reference,
    }


def _resolve(setup: dict[str, Any], manifest: Path, records: Path, *flags: str) -> Any:
    return CliRunner().invoke(
        app,
        [
            "resolve-analyses", str(manifest), str(records),
            "--ancestry-reference", str(setup["ref"]), "--ancestry-groups", str(setup["groups"]),
            "--reference-build", "GRCh38", "--source-build", "GRCh38",
            "--n-min", "10", "--residual-max", "0.25", "--n-workers", "1", *flags,
        ],
    )


def _record(records: Path) -> dict[str, Any]:
    with open(records / "MAPPED.json", encoding="utf-8") as fh:
        record: dict[str, Any] = json.load(fh)
    return record


def test_a_mapped_source_resolves_as_its_gwas_ssf_twin_does(setup: dict[str, Any]) -> None:
    mapped = _resolve(setup, setup["mapped"], setup["tmp"] / "m", "--column-map", str(setup["map"]))
    reference = _resolve(setup, setup["reference"], setup["tmp"] / "r")
    assert reference.exit_code == 0, reference.output
    assert mapped.exit_code == 0, mapped.output

    got, want = _record(setup["tmp"] / "m"), _record(setup["tmp"] / "r")

    assert want["ancestry"]["assigned_ancestry"] == "EUR", "the twin must resolve to mean anything"
    for section in ("diagnostics", "ancestry", "phenotype_sd"):
        expected = want[section]
        if section == "diagnostics":
            expected = {**expected, "source_file": got[section]["source_file"]}
        assert got[section] == expected, section


def test_the_map_s_checksum_is_in_the_fingerprint(setup: dict[str, Any]) -> None:
    result = _resolve(setup, setup["mapped"], setup["tmp"] / "m", "--column-map", str(setup["map"]))
    assert result.exit_code == 0, result.output

    recorded = _record(setup["tmp"] / "m")["fingerprints"]["resolution_config"]["column_map"]

    assert recorded["name"] == "plain-provider"
    assert len(recorded["sha256"]) == 64


def test_a_changed_map_is_not_resumed(setup: dict[str, Any]) -> None:
    records = setup["tmp"] / "m"
    flags = ("--column-map", str(setup["map"]), "--resume")
    assert _resolve(setup, setup["mapped"], records, *flags).exit_code == 0
    unchanged = _resolve(setup, setup["mapped"], records, *flags)
    assert json.loads(unchanged.output)["n_resumed"] == 1, "an unchanged map must resume"

    setup["map"].write_text(_MAP + "# reviewed again\n", encoding="utf-8")
    changed = _resolve(setup, setup["mapped"], records, *flags)

    assert changed.exit_code == 0, changed.output
    assert json.loads(changed.output)["n_resumed"] == 0


def test_a_mapped_row_without_a_map_is_refused(setup: dict[str, Any]) -> None:
    result = _resolve(setup, setup["mapped"], setup["tmp"] / "m")

    assert result.exit_code == 1
    assert "--column-map" in result.output


def test_a_map_no_row_reads_through_is_refused(setup: dict[str, Any]) -> None:
    """A map that silently reads nothing is a run that looks mapped and is not."""
    result = _resolve(
        setup, setup["reference"], setup["tmp"] / "r", "--column-map", str(setup["map"])
    )

    assert result.exit_code == 1
    assert "opengwasdb.column-map" in result.output


def test_a_row_declaring_another_build_than_its_map_is_refused(setup: dict[str, Any]) -> None:
    manifest = setup["tmp"] / "hg19.tsv"
    manifest.write_text(
        setup["mapped"].read_text().replace("sample_size\n", "sample_size\tsource_genome_build\n")
        .replace(f"{int(_STUDY_N)}\n", f"{int(_STUDY_N)}\tGRCh37\n"),
        encoding="utf-8",
    )

    result = _resolve(setup, manifest, setup["tmp"] / "m", "--column-map", str(setup["map"]))

    assert result.exit_code == 1
    assert "genome_build" in result.output


def test_a_built_in_map_is_named_by_name(setup: dict[str, Any]) -> None:
    manifest = setup["tmp"] / "builtin.tsv"
    manifest.write_text(
        setup["reference"].read_text().replace("opengwasdb.gwas-ssf", "opengwasdb.column-map"),
        encoding="utf-8",
    )

    result = _resolve(setup, manifest, setup["tmp"] / "b", "--column-map", "gwas-ssf")

    assert result.exit_code == 0, result.output
    recorded = _record(setup["tmp"] / "b")["fingerprints"]["resolution_config"]["column_map"]
    assert recorded["name"] == "gwas-ssf"
