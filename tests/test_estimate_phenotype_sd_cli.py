"""Tests for the estimate-phenotype-sd command (issue #176).

Mirrors tests/test_assign_ancestry_cli.py's shape: a synthetic source with a
known true SD, driven through the real CLI, so the test proves the command's
whole path -- canonical manifest in, SourceReader resolution, source read,
estimate, shared-core-named TSV out -- rather than any one internal function.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from opengwasdb.build.phenotype_sd_pipeline import SD_OUTPUT_COLUMNS
from opengwasdb.cli.main import app
from opengwasdb.model.enums import OriginalSdMethod
from opengwasdb.readers.finngen import FINNGEN_R13_CAPABILITY
from opengwasdb.readers.gwas_ssf import GWAS_SSF_CAPABILITY
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY

_N_SITES = 200
_SAMPLE_SIZE = 20_000.0
_SD_TRUE = 2.5

_SSF_HEADER = (
    "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta"
    "\tstandard_error\teffect_allele_frequency"
)
_FINNGEN_HEADER = "#chrom\tpos\tref\talt\trsids\tbeta\tsebeta\taf_alt"
_VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size">\n'
    '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">\n'
    '##FORMAT=<ID=EZ,Number=A,Type=Float,Description="Z-score">\n'
    '##FORMAT=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n'
    "##SAMPLE=<ID=STUDY,StudyType=Continuous>\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSTUDY\n"
)


def _af_series() -> np.ndarray:
    """Frequencies spread across (0, 1) so ``2f(1-f)`` genuinely varies.

    A fixture with a narrow frequency range would let a broken estimator pass
    on a near-constant ``se`` scale, so the spread is asserted meaningful
    before any estimate is.
    """
    af = np.linspace(0.05, 0.95, _N_SITES)
    assert af.size >= 50, "fixture must have enough variants to estimate over"
    assert af.max() - af.min() > 0.5, "fixture frequencies must genuinely vary"
    return af


def _se_for(af: np.ndarray, sd_true: float = _SD_TRUE) -> np.ndarray:
    """`se` implied by the ADR-0029 model, the estimator's inverse."""
    return sd_true / np.sqrt(2.0 * af * (1.0 - af) * _SAMPLE_SIZE)


def _write_ssf(path: Path, af: np.ndarray, se: np.ndarray, *, with_af: bool = True) -> None:
    header = _SSF_HEADER if with_af else _SSF_HEADER.rsplit("\t", 1)[0]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for i, (f, s) in enumerate(zip(af, se, strict=True)):
            cells = ["1", str(100_000 + i), "A", "G", "0.1", f"{float(s):.10g}"]
            if with_af:
                cells.append(f"{float(f):.10g}")
            fh.write("\t".join(cells) + "\n")


def _write_finngen(path: Path, af: np.ndarray, se: np.ndarray) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(_FINNGEN_HEADER + "\n")
        for i, (f, s) in enumerate(zip(af, se, strict=True)):
            fh.write(
                f"1\t{100_000 + i}\tG\tA\trs{i}\t0.1\t{float(s):.10g}\t{float(f):.10g}\n"
            )


def _write_vcf(path: Path, af: np.ndarray, se: np.ndarray) -> None:
    """A plain (unindexed) GWAS-VCF; REF=G ALT=A so A1=ALT and AF is A's."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_VCF_HEADER)
        for i, (f, s) in enumerate(zip(af, se, strict=True)):
            fh.write(
                f"1\t{100_000 + i}\t.\tG\tA\t.\tPASS\t.\tES:SE:AF"
                f"\t0.1:{float(s):.10g}:{float(f):.10g}\n"
            )


def _write_manifest(path: Path, rows: list[tuple[str, Path, str, str, str]]) -> None:
    """`(analysis_id, source_file, capability, sample_size, method)` rows."""
    header = [
        "analysis_id",
        "source_file",
        "source_reader_capability",
        "sample_size",
        "original_sd_method",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for analysis_id, source, capability, sample_size, method in rows:
            fh.write(
                "\t".join([analysis_id, str(source), capability, sample_size, method]) + "\n"
            )


def _standard_manifest(
    tmp_path: Path,
    *,
    sample_size: str,
    method: str,
    capability: str = GWAS_SSF_CAPABILITY,
) -> Path:
    """One SSF source carrying the true SD, plus a matching manifest."""
    af = _af_series()
    source = tmp_path / "study.tsv.gz"
    _write_ssf(source, af, _se_for(af))
    manifest = tmp_path / "analyses.tsv"
    _write_manifest(manifest, [("a1", source, capability, sample_size, method)])
    return manifest


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"), strict=True)) for line in lines[1:]]
    return header, rows


def _run(manifest: Path, out: Path, *extra: str) -> Any:
    return CliRunner().invoke(
        app, ["estimate-phenotype-sd", str(manifest), str(out), *extra]
    )


@pytest.mark.parametrize(
    "capability,writer,name",
    [
        (GWAS_SSF_CAPABILITY, _write_ssf, "study.tsv.gz"),
        (FINNGEN_R13_CAPABILITY, _write_finngen, "study.tsv.gz"),
        (GWAS_VCF_CAPABILITY, _write_vcf, "study.vcf"),
    ],
)
def test_recovers_known_sd_through_each_capability(tmp_path, capability, writer, name):
    """Issue #176 AC6: a known true SD is recovered through every capability.

    The fixture is asserted meaningful first -- the raw estimator over the
    same synthetic se/af recovers the SD -- so a green result cannot come from
    a fixture that never reached the estimator.
    """
    from opengwasdb.build.phenotype_sd import estimate_phenotype_sd

    af = _af_series()
    se = _se_for(af)
    raw = estimate_phenotype_sd(
        OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF, _SAMPLE_SIZE, se=se, af=af
    )
    assert raw.sd == pytest.approx(_SD_TRUE, rel=1e-6), "fixture must encode the true SD"

    source = tmp_path / name
    writer(source, af, se)
    manifest = tmp_path / "analyses.tsv"
    _write_manifest(
        manifest,
        [("a1", source, capability, str(int(_SAMPLE_SIZE)), "estimated_from_source_maf")],
    )
    out = tmp_path / "sd.tsv"

    result = _run(manifest, out)
    assert result.exit_code == 0, result.output

    header, rows = _read_tsv(out)
    assert header == list(SD_OUTPUT_COLUMNS)
    assert len(rows) == 1
    assert rows[0]["analysis_id"] == "a1"
    assert rows[0]["original_sd_method"] == "estimated_from_source_maf"
    assert float(rows[0]["original_sd"]) == pytest.approx(_SD_TRUE, rel=1e-6)
    # Text-format round-tripping leaves a tiny dispersion; it must stay far
    # below the SD itself, not be exactly zero.
    assert float(rows[0]["original_sd_dispersion"]) < 1e-3


def test_unknown_capability_fails_naming_the_known_ones(tmp_path):
    """Issue #176 AC2: an unknown capability fails naming the registered set."""
    from opengwasdb.build.phenotype_sd_pipeline import read_sd_manifest

    manifest = _standard_manifest(
        tmp_path,
        sample_size="20000",
        method="estimated_from_source_maf",
        capability="opengwasdb.some-future-format",
    )

    with pytest.raises(ValueError, match=r"known: .*opengwasdb\.gwas-vcf"):
        read_sd_manifest(manifest)
    with pytest.raises(ValueError, match="opengwasdb.some-future-format"):
        read_sd_manifest(manifest)


@pytest.mark.parametrize("sample_size", ["", "0", "-5"])
def test_missing_or_unusable_sample_size_reports_unavailable(tmp_path, sample_size):
    """Issue #176 AC4: no usable N is `unavailable`, never a fabricated SD."""
    manifest = _standard_manifest(
        tmp_path, sample_size=sample_size, method="estimated_from_source_maf"
    )
    out = tmp_path / "sd.tsv"

    result = _run(manifest, out)
    assert result.exit_code == 0, result.output

    _header, rows = _read_tsv(out)
    assert rows[0]["original_sd_method"] == "unavailable"
    assert rows[0]["original_sd"] == ""
    assert rows[0]["original_sd_dispersion"] == ""
    assert rows[0]["notes"], "an unavailable estimate must say why"


def test_result_is_independent_of_worker_count(tmp_path):
    """Issue #176 AC5: order-preserving and worker-count independent."""
    af = _af_series()
    first = tmp_path / "first.tsv.gz"
    second = tmp_path / "second.tsv.gz"
    _write_ssf(first, af, _se_for(af))
    _write_finngen(second, af, _se_for(af, 1.3))
    manifest = tmp_path / "analyses.tsv"
    _write_manifest(
        manifest,
        [
            ("b_second", second, FINNGEN_R13_CAPABILITY, "20000", "estimated_from_source_maf"),
            ("a_first", first, GWAS_SSF_CAPABILITY, "20000", "estimated_from_source_maf"),
        ],
    )
    serial_out = tmp_path / "serial.tsv"
    parallel_out = tmp_path / "parallel.tsv"

    assert _run(manifest, serial_out, "--n-workers", "1").exit_code == 0
    assert _run(manifest, parallel_out, "--n-workers", "3").exit_code == 0

    assert serial_out.read_bytes() == parallel_out.read_bytes()
    _header, rows = _read_tsv(serial_out)
    # Input order, not completion order.
    assert [r["analysis_id"] for r in rows] == ["b_second", "a_first"]
    assert float(rows[0]["original_sd"]) == pytest.approx(1.3, rel=1e-6)


def test_reference_af_source_recovers_known_sd(tmp_path):
    """Issue #176: `--af-source reference` substitutes reference frequencies.

    The source carries no frequency column at all; the estimator must take AF
    from the reference table keyed by canonical ALID.
    """
    af = _af_series()
    source = tmp_path / "no_af.tsv.gz"
    _write_ssf(source, af, _se_for(af), with_af=False)

    reference = tmp_path / "ref.tsv"
    with open(reference, "w", encoding="utf-8") as fh:
        fh.write("alid\teaf\n")
        for i, f in enumerate(af):
            fh.write(f"1:{100_000 + i}:A:G\t{float(f):.10g}\n")

    manifest = tmp_path / "analyses.tsv"
    _write_manifest(
        manifest,
        [("a1", source, GWAS_SSF_CAPABILITY, "20000", "estimated_from_reference_maf")],
    )
    out = tmp_path / "sd.tsv"

    result = _run(
        manifest, out, "--af-source", "reference", "--af-reference", str(reference)
    )
    assert result.exit_code == 0, result.output

    _header, rows = _read_tsv(out)
    assert rows[0]["original_sd_method"] == "estimated_from_reference_maf"
    assert float(rows[0]["original_sd"]) == pytest.approx(_SD_TRUE, rel=1e-6)


def test_af_source_must_match_the_requested_tier(tmp_path):
    """Issue #176: reference AF must not be labelled with a source-MAF tier."""
    from opengwasdb.build.phenotype_sd_pipeline import (
        AfSource,
        SdManifestRow,
        estimate_manifest_phenotype_sd,
    )

    row = SdManifestRow(
        analysis_id="a1",
        source_file=tmp_path / "study.tsv.gz",
        source_reader_capability=GWAS_SSF_CAPABILITY,
        sample_size=20_000.0,
        method=OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF,
    )
    with pytest.raises(ValueError, match="needs --af-source source"):
        estimate_manifest_phenotype_sd(
            [row], af_source=AfSource.reference, af_reference={"1:1:A:G": 0.2}
        )


def test_cli_emits_machine_readable_summary(tmp_path):
    """Issue #176: the command's stdout summary is a single JSON object."""
    manifest = _standard_manifest(
        tmp_path, sample_size="20000", method="estimated_from_source_maf"
    )
    out = tmp_path / "sd.tsv"

    result = _run(manifest, out)
    assert result.exit_code == 0, result.output

    summary = json.loads(result.output.strip().splitlines()[-1])
    assert summary["out_path"] == str(out)
    assert summary["n_analyses"] == 1
    assert summary["n_estimated"] == 1


def test_reference_af_source_requires_a_reference(tmp_path):
    """Issue #176: `--af-source reference` without a table fails at parse time."""
    manifest = _standard_manifest(
        tmp_path, sample_size="20000", method="estimated_from_reference_maf"
    )
    result = _run(manifest, tmp_path / "sd.tsv", "--af-source", "reference")
    assert result.exit_code != 0
    clean = " ".join(result.output.replace("│", " ").split())
    assert "--af-source reference requires --af-reference" in clean


def test_malformed_sample_size_fails_loudly_naming_analysis(tmp_path):
    """Issue #176: a non-numeric N is a manifest defect, not a missing value."""
    from opengwasdb.build.phenotype_sd_pipeline import read_sd_manifest

    manifest = _standard_manifest(
        tmp_path, sample_size="31,684", method="estimated_from_source_maf"
    )
    with pytest.raises(ValueError, match=r"analysis 'a1' has invalid sample_size '31,684'"):
        read_sd_manifest(manifest)


def test_duplicate_analysis_id_fails_loudly_naming_the_id(tmp_path):
    """Issue #176: a repeated analysis_id is refused, never silently merged.

    `analysis_id` keys the output, so two rows for one Analysis would make the
    output ambiguous. The failure names the repeated ID.
    """
    from opengwasdb.build.phenotype_sd_pipeline import read_sd_manifest

    source = tmp_path / "study.tsv.gz"
    _write_ssf(source, _af_series(), _se_for(_af_series()))
    manifest = tmp_path / "duplicate_manifest.tsv"
    _write_manifest(
        manifest,
        [
            ("a1", source, GWAS_SSF_CAPABILITY, "20000", "estimated_from_source_maf"),
            ("a1", source, GWAS_SSF_CAPABILITY, "20000", "estimated_from_source_maf"),
        ],
    )
    with pytest.raises(ValueError, match=r"duplicate analysis_id: 'a1'"):
        read_sd_manifest(manifest)


def test_streams_more_sites_than_a_command_line_array_would_hold(tmp_path):
    """Issue #176: the file/stream path works above the old CLI-array scale.

    The registry shim this replaces passed `se`/`af` JSON arrays on the command
    line, which hits the OS argument-size limit somewhere in the low thousands
    of sites. The new path carries the source in a file and streams it, so it
    is not bounded by ARG_MAX -- 4,001 sites is deliberately above that old
    scale, and the estimate must still be the true SD.
    """
    af = np.linspace(0.05, 0.95, 4_001)
    assert af.size > 4_000, "fixture must exceed the old command-line array scale"
    se = _se_for(af)
    source = tmp_path / "large.tsv.gz"
    _write_ssf(source, af, se)
    manifest = tmp_path / "analyses.tsv"
    _write_manifest(
        manifest,
        [("a1", source, GWAS_SSF_CAPABILITY, "20000", "estimated_from_source_maf")],
    )
    out = tmp_path / "sd.tsv"

    result = _run(manifest, out)
    assert result.exit_code == 0, result.output

    _header, rows = _read_tsv(out)
    assert float(rows[0]["original_sd"]) == pytest.approx(_SD_TRUE, rel=1e-6)
