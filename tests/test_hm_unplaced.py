"""A GWAS Catalog row its harmoniser could not place is dropped and counted (#410).

In the Catalog's pre-GWAS-SSF layout -- the study's alleles in the main columns,
the harmonised ones in `hm_*` columns -- a row whose `hm_code` is 15 ("No
matching variants in reference VCF; Not orientated") carries `NA` in every
`hm_*` column, and its main-column position is whatever the submitter or an
earlier lift wrote. On real files that position is often wrong: a GRCh37
position left unlifted, or a position on an alt or random scaffold under the
primary chromosome's label (OGS-00011 stored `1:5:G:T rs138647955`, lifted onto
`chr4_GL000008v2_random:5`). Every read path drops such a row and names it
`hm_unplaced`; none falls back to the submitter's position.

The rows below are copied from real sources: GCST90014325 (`1:5`, `1:2864`) and
GCST90012790 (`1:13124793`, GRCh37 for rs373657650).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

import opengwasdb.readers.gwas_ssf as gwas_ssf
from opengwasdb.admission import AnalysisRules, Gate, Reason, admission_record_for
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.readers.gwas_ssf import GWAS_SSF_CAPABILITY, GwasSsfReader
from opengwasdb.readers.interface import StreamTally
from opengwasdb.variants.axis import VariantAxis

pytestmark = pytest.mark.assembly_gaps

#: The pre-GWAS-SSF harmonised header, as GCST90014325 spells it.
HEADER = [
    "hm_variant_id", "hm_rsid", "hm_chrom", "hm_pos", "hm_other_allele", "hm_effect_allele",
    "hm_beta", "hm_odds_ratio", "hm_ci_lower", "hm_ci_upper", "hm_effect_allele_frequency",
    "hm_code", "variant_id", "p_value", "chromosome", "base_pair_location", "effect_allele",
    "other_allele", "effect_allele_frequency", "beta", "standard_error", "odds_ratio",
    "ci_lower", "ci_upper",
]
_NA = ["NA"] * 11


def _placed(chrom: str, pos: int, other: str, effect: str, code: str, rsid: str) -> list[str]:
    """A row the harmoniser placed: its `hm_*` cells filled."""
    hm = [f"{chrom}_{pos}_{other}_{effect}", rsid, chrom, str(pos), other, effect, "0.1",
          "NA", "NA", "NA", "0.3"]
    return [*hm, code, rsid, "0.01", chrom, str(pos), effect, other, "0.3", "0.1", "0.05",
            "NA", "NA", "NA"]


def _unplaced(chrom: str, pos: int, effect: str, other: str, rsid: str) -> list[str]:
    """A code-15 row: every `hm_*` cell `NA`, the submitter's position in the main columns."""
    return [*_NA, "15", rsid, "0.8", chrom, str(pos), effect, other, "0.08", "-0.004",
            "0.018", "NA", "NA", "NA"]


ROWS = [
    _placed("1", 1_000_000, "A", "G", "10", "rs1"),
    _placed("1", 2_000_000, "C", "T", "11", "rs2"),
    _unplaced("1", 5, "G", "T", "rs138647955"),
    _unplaced("1", 13_124_793, "A", "G", "rs373657650"),
    _unplaced("1", 2_864, "G", "A", "rs373657650"),
]
PLACED = {"1:1000000:A:G", "1:2000000:C:T"}
PLACED_SITES = {("1", 1_000_000), ("1", 2_000_000)}
#: How each code-15 row is named: ``chromosome:position:other:effect``, as the source spells it.
UNPLACED_KEYS = ["1:5:T:G", "1:13124793:G:A", "1:2864:A:G"]
UNPLACED_POSITIONS = {5, 13_124_793, 2_864}


@pytest.fixture()
def source(tmp_path: Path) -> Path:
    """The rows gzipped with CR LF line ends, as the Catalog ships them."""
    path = tmp_path / "GCST-hm.h.tsv.gz"
    lines = ["\t".join(cells) for cells in [HEADER, *ROWS]]
    path.write_bytes(gzip.compress(("\r\n".join(lines) + "\r\n").encode("utf-8")))
    return path


def test_the_fixture_is_the_hm_columns_layout_with_code_15_rows(source: Path) -> None:
    assert gwas_ssf._declaration(source) == gwas_ssf.HM_COLUMNS_REFERENCE
    with gzip.open(source, "rt", encoding="utf-8") as fh:
        header, *rows = [line.rstrip("\r\n").split("\t") for line in fh]
    code = header.index("hm_code")
    assert sorted(row[code] for row in rows) == ["10", "11", "15", "15", "15"]
    # A code-15 row's position is on a primary chromosome label, so no contig rule drops it.
    position = header.index("base_pair_location")
    assert {int(row[position]) for row in rows if row[code] == "15"} == UNPLACED_POSITIONS


def _faults(tally: StreamTally) -> list[tuple[str, str]]:
    return list(tally.faults.rows)


def test_associations_drop_and_name_each_code_15_row(source: Path) -> None:
    tally = StreamTally()
    associations = list(GwasSsfReader(source).stream_associations(tally))
    assert {(a.chromosome, a.position) for a in associations} == PLACED_SITES
    assert _faults(tally) == [(key, "hm_unplaced") for key in UNPLACED_KEYS]


def test_association_batches_drop_and_name_each_code_15_row(source: Path) -> None:
    tally = StreamTally()
    batches = list(GwasSsfReader(source).stream_association_batches(tally))
    sites = {
        (str(c), int(p))
        for batch in batches
        for c, p in zip(batch.chromosome, batch.position, strict=True)
    }
    assert sites == PLACED_SITES
    assert _faults(tally) == [(key, "hm_unplaced") for key in UNPLACED_KEYS]


def test_the_variant_and_metrics_projections_leave_code_15_rows_out(source: Path) -> None:
    reader = GwasSsfReader(source)
    assert {(v.chromosome, v.position) for v in reader.stream_variants()} == PLACED_SITES
    assert {m.alid for m in reader.stream_metrics()} == PLACED
    assert {str(a) for chunk in reader.stream_metric_chunks() for a in chunk.alid} == PLACED


def test_a_gate_counts_and_logs_the_drops_as_hm_unplaced(source: Path) -> None:
    tally = StreamTally()
    list(GwasSsfReader(source).stream_associations(tally))
    gate = Gate(AnalysisRules("a"))
    gate.streamed(tally)
    assert gate.counts.dropped == {Reason.HM_UNPLACED: 3}
    assert gate.counts.examined == 3
    assert gate.dropped == [(key, Reason.HM_UNPLACED) for key in UNPLACED_KEYS]


def test_a_build_stores_no_code_15_row_at_its_submitted_position(
    tmp_path: Path, source: Path
) -> None:
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "trait_id\ttrait_name\tfile_path\tn\tstored_effect_scale\toriginal_sd_method"
        "\toriginal_sd\tsource_reader_capability\tsource_assembly\n"
        f"a\tTrait\t{source}\t1000\tlog_or\tbinary_trait\t\t{GWAS_SSF_CAPABILITY}\thg38\n",
        encoding="utf-8",
    )
    store = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(manifest, store, store_id="s", release_id="r")
    assert {v.alid for v in VariantAxis(store).all()} == PLACED
    record = admission_record_for(store)
    (analysis,) = json.loads((record / "admission.json").read_text())["analyses"]
    assert analysis["dropped"] == {"hm_unplaced": 3}
    assert (analysis["examined"], analysis["admitted"]) == (5, 2)


def _coded(directory: Path, code: str) -> Path:
    """A placed row, and one at `1:2500000` whose `hm_code` is ``code``, every `hm_*` cell NA."""
    row = [*_NA, code, "rs3", "0.5", "1", "2500000", "G", "A", "0.3", "0.1", "0.05",
           "NA", "NA", "NA"]
    path = directory / f"GCST-hm-{code}.h.tsv.gz"
    lines = ["\t".join(cells) for cells in [HEADER, ROWS[0], row]]
    path.write_bytes(gzip.compress(("\r\n".join(lines) + "\r\n").encode("utf-8")))
    return path


def _read_both_ways(path: Path) -> tuple[set[tuple[str, int]], list[tuple[str, str]]]:
    """The sites and faults of the per-row read, asserted equal to the batches'."""
    rows, batches = StreamTally(), StreamTally()
    sites = {(a.chromosome, a.position) for a in GwasSsfReader(path).stream_associations(rows)}
    batched = {
        (str(c), int(p))
        for batch in GwasSsfReader(path).stream_association_batches(batches)
        for c, p in zip(batch.chromosome, batch.position, strict=True)
    }
    assert sites == batched
    assert _faults(rows) == _faults(batches)
    return sites, _faults(rows)


@pytest.mark.parametrize(
    "code",
    [
        # "Required fields are not known; Not orientated" (EBI's harmonisation codes).
        "14",
        # "Multiple matching variants in reference VCF; Not orientated".
        "16",
    ],
)
def test_a_row_of_another_code_the_harmoniser_placed_nowhere_is_dropped(
    tmp_path: Path, code: str
) -> None:
    sites, faults = _read_both_ways(_coded(tmp_path, code))
    assert sites == {("1", 1_000_000)}
    assert faults == [("1:2500000:A:G", "hm_unplaced")]


@pytest.mark.parametrize("code", ["9", "17", "18"])
def test_a_palindrome_code_places_its_row_and_it_is_kept(tmp_path: Path, code: str) -> None:
    sites, faults = _read_both_ways(_coded(tmp_path, code))
    assert sites == {("1", 1_000_000), ("1", 2_500_000)}
    assert faults == []
