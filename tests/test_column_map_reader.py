"""The generic tabular reader, driven by a built-in Column Map (issue #281).

The built-in `gwas-ssf` and `finngen-r13` maps exist to say, as data, what the
two dedicated readers have always done. So the bar is byte identity: every
stream either reader offers, read through the built-in map, is the same value
the dedicated reader gives on the same file -- on the fixtures the dedicated
readers' own tests already found worth covering.
"""

from __future__ import annotations

import gzip
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from test_effect_recovery import _beta_file, _odds_ratio_file
from test_imputation_score_reader import _declared
from test_imputation_score_reader import _file as _scored_file
from test_projected_metric_chunks import _HEADER, _Z_HEADER, _ordinary_rows, _write
from test_readers import _METRICS_FIXTURE_ROWS, _write_metrics_fixture

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.column_map import load_column_map
from opengwasdb.readers.column_map_reader import ColumnMapReader
from opengwasdb.readers.finngen import FinnGenR13Reader
from opengwasdb.readers.gwas_ssf import GwasSsfReader
from opengwasdb.readers.tabular import MetricsChunk

_FIXTURES = Path(__file__).parent / "fixtures"


def _chunk_bytes(chunks: list[MetricsChunk]) -> list[bytes]:
    """Every chunk array as bytes: dtype and values, so `0.0` and `-0.0` differ."""
    out: list[bytes] = []
    for chunk in chunks:
        for name in (
            "alid", "flipped", "palindromic", "af_alt", "beta", "se", "imputation_score",
            "imputation_score_status", "effect_from_odds_ratio_fallback", "se_from_ci",
            "se_from_p_value",
        ):
            array: np.ndarray = getattr(chunk, name)
            body = repr(array.tolist()).encode() if array.dtype == object else array.tobytes()
            out.append(f"{name}:{array.dtype.str}:".encode() + body)
    return out


def _ssf_streams(reader: GwasSsfReader | ColumnMapReader) -> dict[str, object]:
    metrics = list(reader.stream_metrics())
    alids = [row.alid for row in metrics]
    return {
        "associations": repr(list(reader.stream_associations())),
        "variants": repr(list(reader.stream_variants())),
        "metrics": repr(metrics),
        "chunks": _chunk_bytes(list(reader.stream_metric_chunks())),
        "sites": repr(sorted(reader.extract_at_sites(alids).items())),
        "effect_kind": None if reader.effect_source is None else reader.effect_source.kind,
    }


def _legacy_hm_file(tmp_path: Path) -> Path:
    header = [
        "hm_variant_id", "hm_rsid", "hm_chrom", "hm_pos", "hm_other_allele",
        "hm_effect_allele", "hm_beta", "hm_effect_allele_frequency", "hm_code",
        "chromosome", "base_pair_location", "variant_id", "effect_allele", "beta",
        "standard_error", "other_allele", "effect_allele_frequency",
    ]
    rows = [
        ["10_1_A_G", "rs1", "10", "1", "A", "G", "0.1", "0.2", "7",
         "10", "1", "rs1", "G", "0.1", "0.03", "A", "0.2"],
        ["10_2_C_T", "rs2", "10", "2", "C", "T", "0.2", "0.3", "7",
         "10", "2", "10_2_C_T", "T", "0.2", "0.04", "C", "0.3"],
    ]
    return _write(tmp_path / "legacy.tsv.gz", header, rows)


def _quoted_ragged_file(tmp_path: Path) -> Path:
    rows = [
        ["1", "100", "A", "G", "0.1", "0.05", "0.2"],
        ["1", "200", "A"],
        ["1", "300", '"A\tG"', "C", "0.1", "0.05", "0.2"],
        ["1", "400", "A", "C", "0.1", "0.05"],
        ["1", "500", "A", "C", '"0.2"', "0.05", "0.3"],
    ]
    return _write(tmp_path / "ragged.tsv.gz", _HEADER, rows)


def _upper_beta_with_rsids_file(tmp_path: Path) -> Path:
    header = [
        "rsid", "chromosome", "base_pair_location", "effect_allele", "other_allele",
        "BETA", "standard_error", "effect_allele_frequency", "variant_id", "hm_code",
    ]
    rows = [
        ["rs10", "chr1", "100", "a", "g", "0.1", "0.05", "0.2", "1_100_A_G", "1"],
        ["", "23", "200", "T", "A", "-0.2", "0.06", "0.7", "rs20", "2"],
        [".", "MT", "300", "C", "G", "0.3", "NA", "0.1", "1_300_C_G", "3"],
    ]
    return _write(tmp_path / "upper.tsv.gz", header, rows)


def _z_file(tmp_path: Path) -> Path:
    rows = [
        ["1", "100", "A", "G", "-2.0", "0.25", "1000"],
        ["1", "200", "A", "C", "1.5", "0.4", "25000"],
        ["1", "400", "A", "C", "-1.0", "0", "1000"],
        ["1", "700", "A", "C", "NA", "0.25", "1000"],
    ]
    return _write(tmp_path / "z.tsv.gz", _Z_HEADER, rows)


def _metrics_file(tmp_path: Path) -> Path:
    path = tmp_path / "metrics.tsv"
    _write_metrics_fixture(path, _METRICS_FIXTURE_ROWS)
    return path


_SSF_FIXTURES: dict[str, Callable[[Path], Path]] = {
    "ordinary": lambda tmp: _write(tmp / "ordinary.tsv.gz", _HEADER, _ordinary_rows()),
    "recovery with beta": _beta_file,
    "recovery odds ratio only": _odds_ratio_file,
    "reader metrics rows": _metrics_file,
    "legacy hm layout": _legacy_hm_file,
    "quoted and ragged": _quoted_ragged_file,
    "BETA spelling and rsids": _upper_beta_with_rsids_file,
    "signed z": _z_file,
}


#: Every fixture on both scales, except a signed z on a case-control scale,
#: which both readers refuse (tested below).
_SSF_CASES = [
    (fixture, scale)
    for fixture in sorted(_SSF_FIXTURES)
    for scale in (StoredEffectScale.SD, StoredEffectScale.LOG_OR)
    if not (fixture == "signed z" and scale is StoredEffectScale.LOG_OR)
]


@pytest.mark.parametrize(("fixture", "scale"), _SSF_CASES)
def test_the_gwas_ssf_built_in_map_reads_what_the_gwas_ssf_reader_reads(
    tmp_path: Path, fixture: str, scale: StoredEffectScale
) -> None:
    path = _SSF_FIXTURES[fixture](tmp_path)
    reference = _ssf_streams(GwasSsfReader(path, scale, chunk_rows=2))
    assert reference["associations"] != "[]", "the fixture must yield associations"

    mapped = ColumnMapReader(path, load_column_map("gwas-ssf"), scale, chunk_rows=2)

    assert _ssf_streams(mapped) == reference


def test_a_signed_z_is_refused_on_a_case_control_scale_through_the_map(tmp_path: Path) -> None:
    path = _z_file(tmp_path)
    reader = ColumnMapReader(path, load_column_map("gwas-ssf"), StoredEffectScale.LOG_OR)

    with pytest.raises(ValueError, match="log_or"):
        list(reader.stream_associations())


def test_a_declared_imputation_score_reads_identically_through_the_map(tmp_path: Path) -> None:
    values = ["0", "0.6", "1", "", "NA", "#NA", "bogus", "inf", "-0.1", "1.01"]
    path = _scored_file(tmp_path / "quality.tsv", "info", values)
    declared = _declared()
    reference = _ssf_streams(
        GwasSsfReader(path, chunk_rows=3, imputation_score_declaration=declared)
    )
    assert "usable" in repr(reference["metrics"]), "the fixture must carry a usable score"

    mapped = ColumnMapReader(
        path, load_column_map("gwas-ssf"), chunk_rows=3, imputation_score_declaration=declared
    )

    assert _ssf_streams(mapped) == reference


def _finngen_streams(reader: FinnGenR13Reader | ColumnMapReader) -> dict[str, str]:
    associations = list(reader.stream_associations())
    variants = list(reader.stream_variants())
    alids = sorted({f"{v.chromosome}:{v.position}:{min(v.ref, v.alt).upper()}:"
                    f"{max(v.ref, v.alt).upper()}" for v in variants})
    return {
        "associations": repr(associations),
        "variants": repr(variants),
        "sites": repr(sorted(reader.extract_at_sites(alids).items())),
    }


def _finngen_crafted(tmp_path: Path) -> Path:
    path = tmp_path / "finngen.tsv.bgz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(
            "chrom\tpos\tref\talt\trsids\tnearest_genes\tpval\tmlogp\tbeta\tsebeta\taf_alt\n"
            "1\t100\tG\tA\trs100,rs101\tGENE1\t0.1\t1\t0.5\t0.2\t0.3\n"
            "1\t200\ta\tg\t.\tGENE2\t0.1\t1\t-0.5\t0.2\t0.6\n"
            "23\t300\tA\tC\trs300\tGENEX\t0.1\t1\t0.1\t0\t0.1\n"
            "1\t400\tA\tC\trs400\tG\t0.1\t1\tNA\t0.5\tNA\n"
            "1\tbad\tA\tG\trs500\tG\t0.1\t1\t1.0\t0.5\t0.2\n"
            "1\t600\tA\tN\trs600\tG\t0.1\t1\t1.0\t0.5\t0.2\n"
            "1\t700\tT\tA\t rs700 ,rs701\tG\t0.1\t1\t1.0\t0.5\t0.0\n"
        )
    return path


def _finngen_quoted(tmp_path: Path) -> Path:
    path = tmp_path / "quoted.tsv"
    path.write_text(
        '"extra"\t"alt"\t"rsids"\t"pos"\t"#chrom"\t"ref"\t'
        '"beta"\t"sebeta"\t"af_alt"\t"pval"\t"mlogp"\t"nearest_genes"\n'
        "x\tA\trs100,rs101\t100\t1\tG\t0.1\t0.2\t0.3\t0.1\t1\tGENE1\n"
        "x\tC\trsX\t300\t23\tA\t0.2\t0.1\t0.4\t0.1\t1\tGENEX\n"
        '"x"\t"A"\t"rsQuoted"\t"600"\t"1"\t"G"\t0.3\t0.1\t0.5\t0.1\t1\tGENE\n',
        encoding="utf-8",
    )
    return path


_FINNGEN_FIXTURES: dict[str, Callable[[Path], Path]] = {
    "published fixture": lambda _tmp: _FIXTURES / "finngen_r13.tsv",
    "crafted, bgzipped, legacy chrom": _finngen_crafted,
    "quoted": _finngen_quoted,
}


@pytest.mark.parametrize("fixture", sorted(_FINNGEN_FIXTURES))
def test_the_finngen_built_in_map_reads_what_the_finngen_reader_reads(
    tmp_path: Path, fixture: str
) -> None:
    path = _FINNGEN_FIXTURES[fixture](tmp_path)
    reference = _finngen_streams(FinnGenR13Reader(path))
    assert reference["associations"] != "[]", "the fixture must yield associations"

    mapped = ColumnMapReader(path, load_column_map("finngen-r13"), StoredEffectScale.LOG_OR)

    assert _finngen_streams(mapped) == reference


def test_the_finngen_map_takes_the_first_of_several_rsids(tmp_path: Path) -> None:
    """The one FinnGen rule GWAS-SSF does not share, so it must come from the map."""
    path = _finngen_crafted(tmp_path)
    reader = ColumnMapReader(path, load_column_map("finngen-r13"), StoredEffectScale.LOG_OR)

    rsids = {variant.position: variant.rsid for variant in reader.stream_variants()}

    assert rsids[100] == "rs100"
    assert rsids[700] == "rs700"


_PLAIN_MAP = """\
column_map_version = 1
name = "plain"

[columns]
chromosome = "chr"
position = "bp"
effect_allele = "a1"
other_allele = "a2"
beta = "b"
standard_error = "se"

[optional_columns]
frequency = "freq"

[conventions]
missing_values = "any_non_numeric"
unmapped_columns = "ignore"
alleles = "effect_and_other"
frequency_allele = "effect_allele"
position_notation = "integer"
chromosome_spelling = "adr-0052"
genome_build = "GRCh37"
"""


def _plain_map(tmp_path: Path, text: str = _PLAIN_MAP) -> Path:
    path = tmp_path / "plain.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_an_undeclared_column_is_never_read_even_under_a_gwas_ssf_name(tmp_path: Path) -> None:
    """`standard_error` and `p_value` here are provider columns the map does not
    name; reading either would be a plausible wrong answer."""
    path = tmp_path / "plain.tsv"
    path.write_text(
        "chr\tbp\ta1\ta2\tb\tse\tstandard_error\tp_value\n"
        "1\t100\tA\tG\t0.2\t0.1\t9.0\t0.5\n"
        "1\t200\tA\tG\t0.2\tNA\t9.0\t0.5\n",
        encoding="utf-8",
    )
    reader = ColumnMapReader(path, load_column_map(_plain_map(tmp_path)))

    rows = [(row.position, row.beta, row.se) for row in reader.stream_metrics()]

    assert rows == [(100, 0.2, 0.1), (200, 0.2, None)]


def test_a_declared_column_the_file_lacks_is_a_loud_failure(tmp_path: Path) -> None:
    path = tmp_path / "plain.tsv"
    path.write_text("chr\tbp\ta1\ta2\tb\n1\t100\tA\tG\t0.2\n", encoding="utf-8")
    reader = ColumnMapReader(path, load_column_map(_plain_map(tmp_path)))

    with pytest.raises(ValueError, match="'se'"):
        list(reader.stream_associations())


def test_a_field_the_file_names_twice_is_refused(tmp_path: Path) -> None:
    """`se ` and `se` are one column named twice (the GCST006329 shape, ADR 0049)."""
    path = tmp_path / "plain.tsv"
    path.write_text(
        "chr\tbp\ta1\ta2\tb\tse \tse\n1\t100\tA\tG\t0.2\t0.1\tNA\n", encoding="utf-8"
    )
    reader = ColumnMapReader(path, load_column_map(_plain_map(tmp_path)))

    with pytest.raises(ValueError, match="named 2 times"):
        list(reader.stream_metric_chunks())


def test_unmapped_columns_refuse_rejects_a_column_the_map_did_not_expect(
    tmp_path: Path,
) -> None:
    text = _PLAIN_MAP.replace('unmapped_columns = "ignore"', 'unmapped_columns = "refuse"')
    text = text.replace('name = "plain"', 'name = "plain"\nignored_columns = ["note"]')
    path = tmp_path / "plain.tsv"
    path.write_text("chr\tbp\ta1\ta2\tb\tse\tnote\tnew\n1\t100\tA\tG\t0.2\t0.1\tx\ty\n")
    reader = ColumnMapReader(path, load_column_map(_plain_map(tmp_path, text)))

    with pytest.raises(ValueError, match="'new'"):
        list(reader.stream_variants())
