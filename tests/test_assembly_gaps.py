"""No Store holds a variant where GRCh38 has no sequence (#410).

GRCh38 has no sequence in its assembly gaps, the runs of `N` that start every
chromosome (chr1 1-10,000) and fill each centromere, and none past a contig's
end. A source row on GRCh38 placed there is dropped and counted at admission
(`in_assembly_gap`, `past_contig_end`), and `validate` fails a release that
holds one (`variant_table.assembly_gap`). A single `N` base is not a gap: it is
an IUPAC ambiguity code UCSC's FASTA writes as `N`, and real variants sit on it
(`17:83128871:G:T` rs9747886, held by OGS-00008, -09, -11 and -27).

Every expected boundary below is a literal from the N runs of UCSC's
`hg38.fa.gz` (sha256 `c1dd8706...27d4`), read once on 27 Nov 2026.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from liftover_fixtures import vcf_manifest, vcf_row
from residual_fixtures import write_gwas_vcf_with_eaf

import opengwasdb.variants.assembly_gaps as assembly_gaps
import opengwasdb.variants.routing.union as union
from opengwasdb.admission import AnalysisRules, Gate, Reason, admission_record_for
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.validation.validate import validate_store
from opengwasdb.variants.assembly_gaps import grch38, sequence_of
from opengwasdb.variants.axis import VariantAxis

pytestmark = pytest.mark.assembly_gaps

#: chr21's length in GRCh38.
CHR21_LENGTH = 46_709_983
GAP_RULE = "variant_table.assembly_gap"


# --- the gap table ------------------------------------------------------------


def _in_gap(chromosome: str, position: int) -> bool:
    return bool(grch38().in_gap(np.array([chromosome]), np.array([position]))[0])


def _past_end(chromosome: str, position: int) -> bool:
    return bool(grch38().past_end(np.array([chromosome]), np.array([position]))[0])


@pytest.mark.parametrize(
    ("chromosome", "position", "gap"),
    [
        ("1", 1, True), ("1", 5, True), ("1", 9_999, True), ("1", 10_000, True),
        ("1", 10_001, False),
        # chr1's second run, and the 18 Mb run around its centromere.
        ("1", 207_666, False), ("1", 207_667, True), ("1", 257_666, True), ("1", 257_667, False),
        ("1", 125_184_588, True), ("1", 143_120_946, True), ("1", 143_184_587, True),
        ("1", 143_184_588, False),
        # The acrocentric short arms.
        ("13", 16_000_000, True), ("14", 16_000_000, True), ("15", 17_000_000, True),
        ("15", 17_000_001, False), ("21", 5_010_000, True), ("22", 10_510_000, True),
        ("22", 10_510_001, False),
        # A run of two bases is a gap; a single N base, the one beside it too, is not.
        ("1", 121_804_801, True), ("1", 121_804_802, True), ("1", 121_804_799, False),
        ("17", 83_128_871, False), ("13", 100_972_571, False), ("3", 16_816_464, False),
        # chrMT's one N run is its base 3,107.
        ("MT", 3_107, False), ("MT", 3_106, False),
    ],
)
def test_the_table_holds_grch38s_gaps(chromosome: str, position: int, gap: bool) -> None:
    assert _in_gap(chromosome, position) is gap


def test_the_table_holds_every_primary_contig_and_its_length() -> None:
    sequence = grch38()
    assert set(sequence.lengths) == {*map(str, range(1, 23)), "X", "Y", "MT"}
    assert sequence.lengths["1"] == 248_956_422
    assert sequence.lengths["21"] == CHR21_LENGTH
    assert sequence.lengths["MT"] == 16_569
    # 1,002 N runs, 216 of them a single base.
    assert sum(len(starts) for starts, _ends in sequence.gaps.values()) == 786


def test_a_position_past_its_contigs_end_is_past_it() -> None:
    assert not _past_end("21", CHR21_LENGTH)
    assert _past_end("21", CHR21_LENGTH + 1)
    assert not _in_gap("21", CHR21_LENGTH + 1)


def test_a_contig_the_table_does_not_hold_is_not_judged() -> None:
    """A GRCh38 source's alt contig is a chromosome (spec §4); its sequence is not here."""
    chromosomes = np.array(["11_KI270927v1_alt", "1"])
    positions = np.array([5, 5])
    assert grch38().in_gap(chromosomes, positions).tolist() == [False, True]
    assert grch38().past_end(chromosomes, positions).tolist() == [False, False]


def test_only_grch38_has_a_table() -> None:
    assert sequence_of("hg38") is grch38()
    assert sequence_of("GRCh38") is grch38()
    assert sequence_of("hg19") is None


def test_the_derivation_reads_runs_of_two_or_more_n(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fa.gz"
    body = ">chr1\nNNNNACGTnnAC\nGNTA\n>chr1_KI270706v1_random\nNNNN\n>chrM\nACNNA\n"
    fasta.write_bytes(gzip.compress(body.encode("ascii")))
    lengths, runs = assembly_gaps.derive(fasta)
    assert lengths == {"1": 16, "MT": 5}
    # 1-based, inclusive: chr1's NNNN, its nn, not its single N; chrM's NN.
    assert runs == [("1", 1, 4), ("1", 9, 10), ("MT", 3, 4)]


def test_outside_the_suite_the_package_reads_the_committed_table_with_its_gaps() -> None:
    """The suite's gap-free table is its own fixture's (tests/conftest.py): a process
    that never loads the suite reads the committed table, gaps and all."""
    import subprocess
    import sys

    probe = (
        "from opengwasdb.variants import assembly_gaps as g; t = g.grch38(); "
        "print(g.GRCH38_TABLE.name, g.GRCH38_TABLE.parent.name, "
        "sum(len(s) for s, _ in t.gaps.values()), len(t.lengths))"
    )
    ran = subprocess.run(
        [sys.executable, "-I", "-c", probe], capture_output=True, text=True, check=True,
        cwd=Path(assembly_gaps.__file__).parents[2],
    )
    assert ran.stdout.split() == ["grch38-assembly-gaps.tsv", "variants", "786", "25"]


def test_nothing_in_the_package_points_the_table_elsewhere() -> None:
    """`GRCH38_TABLE` and its recorded checksum are each assigned once, where they are
    defined; only the suite's fixture repoints them, so no code path can load the
    gap-free table."""
    import re

    write = re.compile(
        r"\bGRCH38_TABLE(_SHA256)?\s*(:[^=]*)?=(?!=)|setattr\(.*GRCH38_TABLE"
    )
    package = Path(assembly_gaps.__file__).parents[1]
    writes = [
        (str(path.relative_to(package)), line.split("=", 1)[0].strip())
        for path in sorted(package.rglob("*.py"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if write.search(line)
    ]
    assert writes == [
        ("variants/assembly_gaps.py", "GRCH38_TABLE: Final"),
        ("variants/assembly_gaps.py", "GRCH38_TABLE_SHA256: Final"),
    ]


def test_the_committed_table_is_the_one_its_checksum_records() -> None:
    import hashlib

    digest = hashlib.sha256(assembly_gaps.GRCH38_TABLE.read_bytes()).hexdigest()
    assert digest == assembly_gaps.GRCH38_TABLE_SHA256


@pytest.mark.parametrize(
    "keep",
    [
        pytest.param(lambda line: line.startswith(("#", "kind")), id="header only"),
        pytest.param(lambda line: not line.startswith("gap\t"), id="no gap rows"),
        pytest.param(lambda line: line != "gap\t1\t1\t10000", id="one gap row lost"),
    ],
)
def test_a_table_that_lost_rows_is_refused_not_read_as_fewer_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep: Callable[[str], bool]
) -> None:
    lines = assembly_gaps.GRCH38_TABLE.read_text(encoding="utf-8").splitlines()
    damaged = tmp_path / "grch38-assembly-gaps.tsv"
    damaged.write_text("\n".join(filter(keep, lines)) + "\n", encoding="utf-8")
    assert damaged.read_text() != assembly_gaps.GRCH38_TABLE.read_text()
    monkeypatch.setattr(assembly_gaps, "GRCH38_TABLE", damaged)
    with pytest.raises(ValueError, match="sha256"):
        grch38()


# --- admission ----------------------------------------------------------------


@dataclass(frozen=True)
class _Batch:
    """A routed batch, as Variant Routing hands a gate one."""

    chromosome: list[str]
    position: list[int]
    ref: list[str] = field(default_factory=list)
    alt: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.chromosome)
        object.__setattr__(self, "ref", ["G"] * n)
        object.__setattr__(self, "alt", ["T"] * n)

    @property
    def z(self) -> np.ndarray:
        return np.full(len(self.chromosome), 2.0)

    @property
    def se(self) -> np.ndarray:
        return np.full(len(self.chromosome), 0.1)

    @property
    def eaf(self) -> np.ndarray:
        return np.full(len(self.chromosome), 0.2)

    @property
    def imputation_score(self) -> np.ndarray:
        return np.full(len(self.chromosome), np.nan)

    @property
    def imputation_status(self) -> np.ndarray:
        return np.zeros(len(self.chromosome), dtype=np.uint8)


ROWS = _Batch(["1", "1", "1", "21"], [5, 9_999, 10_001, CHR21_LENGTH + 1])


def test_a_gate_on_grch38_drops_and_counts_a_gap_row_and_a_past_end_row() -> None:
    gate = Gate(AnalysisRules("a", sequence=grch38()))
    keep = gate.admit(ROWS)
    assert keep.tolist() == [False, False, True, False]
    assert gate.counts.dropped == {Reason.IN_ASSEMBLY_GAP: 2, Reason.PAST_CONTIG_END: 1}
    assert (gate.counts.examined, gate.counts.admitted) == (4, 1)
    assert gate.dropped == [
        ("1:5:G:T", Reason.IN_ASSEMBLY_GAP),
        ("1:9999:G:T", Reason.IN_ASSEMBLY_GAP),
        (f"21:{CHR21_LENGTH + 1}:G:T", Reason.PAST_CONTIG_END),
    ]


def test_a_gate_that_is_given_no_sequence_judges_no_position() -> None:
    """A lifted source's rows reach the gate at their source coordinates."""
    gate = Gate(AnalysisRules("a"))
    assert gate.admit(ROWS).tolist() == [True] * 4


def _grch38_vcf_build(directory: Path, rows: list[str]) -> Path:
    vcf = write_gwas_vcf_with_eaf(directory / "a.vcf", rows)
    store = directory / "store.opengwasdb"
    build_dense_from_vcf_manifest(
        vcf_manifest(directory, {"a": vcf}, assembly="hg38"), store, store_id="s", release_id="r"
    )
    return store


GAP_ROWS = [
    vcf_row("1", 5, "G", "T"), vcf_row("1", 9_999, "G", "T"), vcf_row("1", 10_001, "G", "T"),
    vcf_row("21", CHR21_LENGTH + 1, "G", "T"),
]


def test_a_build_drops_and_counts_grch38_rows_outside_the_sequence(tmp_path: Path) -> None:
    store = _grch38_vcf_build(tmp_path, GAP_ROWS)
    assert {v.alid for v in VariantAxis(store).all()} == {"1:10001:G:T"}
    record = admission_record_for(store)
    (analysis,) = json.loads((record / "admission.json").read_text())["analyses"]
    assert analysis["dropped"] == {"in_assembly_gap": 2, "past_contig_end": 1}
    assert (analysis["examined"], analysis["admitted"]) == (4, 1)


def test_a_hybrid_build_drops_and_counts_them_too(tmp_path: Path) -> None:
    from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest

    vcf = write_gwas_vcf_with_eaf(tmp_path / "h.vcf", GAP_ROWS)
    panel = tmp_path / "panel.txt"
    panel.write_text("1:10001:G:T\n", encoding="utf-8")
    store = tmp_path / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        vcf_manifest(tmp_path, {"a": vcf}, assembly="hg38"), store, store_id="h",
        release_id="v1", reference_panel=panel,
    )
    assert {v.alid for v in VariantAxis(store).all()} == {"1:10001:G:T"}
    record = admission_record_for(store)
    (analysis,) = json.loads((record / "admission.json").read_text())["analyses"]
    assert analysis["dropped"] == {"in_assembly_gap": 2, "past_contig_end": 1}


def test_a_ragged_ssf_build_drops_and_counts_them_too(tmp_path: Path) -> None:
    from test_lift_off_primary import _admission, _alids, _ssf_build, _ssf_row

    rows = [
        _ssf_row("1", 5, "G", "T", "rs_gap"), _ssf_row("1", 9_999, "G", "T", "rs_gap2"),
        _ssf_row("1", 10_001, "G", "T", "rs_on"),
        _ssf_row("21", CHR21_LENGTH + 1, "G", "T", "rs_past"),
    ]
    store = _ssf_build(tmp_path, [("a", "GRCh38", rows)])
    assert _alids(store) == {"1:10001:G:T"}
    (analysis,) = _admission(store)
    assert analysis["dropped"] == {"in_assembly_gap": 2, "past_contig_end": 1}


def _admitted(store: Path) -> dict:
    record = admission_record_for(store)
    (analysis,) = json.loads((record / "admission.json").read_text())["analyses"]
    return analysis


#: The gap-and-end rows every builder below reads: two in chr1's first gap, one on
#: sequence, one past chr21's end.
SITES = [("1", 5), ("1", 9_999), ("1", 10_001), ("21", CHR21_LENGTH + 1)]
GAP_DROPS = {"in_assembly_gap": 2, "past_contig_end": 1}


def test_a_dense_tsv_build_drops_and_counts_them_too(tmp_path: Path) -> None:
    from dense_sources import build_dense_fixture

    source = tmp_path / "records.tsv"
    source.write_text(
        "analysis_id\tchromosome\tposition\teffect_allele\tother_allele\tz\tse\n"
        + "".join(f"A\t{c}\t{p}\tT\tG\t2.0\t0.1\n" for c, p in SITES),
        encoding="utf-8",
    )
    store = tmp_path / "dense.opengwasdb"
    build_dense_fixture([source], store, store_id="s", release_id="r", reference_assembly="GRCh38")
    assert {v.alid for v in VariantAxis(store).all()} == {"1:10001:G:T"}
    assert _admitted(store)["dropped"] == GAP_DROPS


def _besd(tmp_path: Path, build: str) -> Path:
    import besd_fixtures

    from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    besd_fixtures.write_esi(fixture / "t.esi", [
        {"chr": c, "snp_id": f"rs{i}", "bp": p, "a1": "T", "a2": "G"}
        for i, (c, p) in enumerate(SITES)
    ])
    besd_fixtures.write_epi(fixture / "t.epi", [{"chr": "1", "probe_id": "ENSG0", "bp": 1000}])
    besd_fixtures.write_besd_sparse_3f(
        fixture / "t.besd", 1, [[(i, 0.1, 0.02) for i in range(len(SITES))]]
    )
    store = tmp_path / "besd.opengwasdb"
    build_ragged_from_besd(
        fixture / "t", store, store_id="s", release_id="r",
        association_coverage="cis_and_signals", source_build=build,
    )
    return store


def test_a_besd_build_drops_and_counts_them_too(tmp_path: Path) -> None:
    """An ESI row in a gap is on no axis, and each probe's row there is dropped for
    the gap, named by its ESI index, not counted `off_axis`."""
    store = _besd(tmp_path, "hg38")
    assert {v.alid for v in VariantAxis(store).all()} == {"1:10001:G:T"}
    analysis = _admitted(store)
    assert analysis["dropped"] == GAP_DROPS
    with gzip.open(admission_record_for(store) / "dropped" / "0.tsv.gz", "rt") as fh:
        logged = fh.read().splitlines()[1:]
    assert logged == [
        "esi:0\tin_assembly_gap", "esi:1\tin_assembly_gap", "esi:3\tpast_contig_end"
    ]


# --- validate -----------------------------------------------------------------


def _release_holding(directory: Path, monkeypatch: pytest.MonkeyPatch, rows: list[str]) -> Path:
    """A 0.3.0 release a build that judged no position wrote: what OGS-00011 is."""
    monkeypatch.setattr(assembly_gaps, "sequence_of", lambda _assembly: None)
    monkeypatch.setattr(union, "_off_sequence", lambda _sites: set())
    return _grch38_vcf_build(directory, rows)


def _gap_findings(store: Path) -> list[str]:
    return [f.message for f in validate_store(store, rules=[GAP_RULE]).of(GAP_RULE)]


def test_validate_fails_a_release_holding_a_variant_in_a_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _release_holding(
        tmp_path, monkeypatch, [vcf_row("1", 5, "G", "T"), vcf_row("1", 10_001, "G", "T")]
    )
    assert "1:5:G:T" in {v.alid for v in VariantAxis(store).all()}
    result = validate_store(store, rules=[GAP_RULE])
    assert GAP_RULE in result.failed()
    (message,) = _gap_findings(store)
    assert "1 variant(s)" in message and "1:5:G:T" in message


def test_validate_fails_a_release_holding_a_variant_past_a_contigs_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _release_holding(tmp_path, monkeypatch, [vcf_row("21", CHR21_LENGTH + 1, "G", "T")])
    assert GAP_RULE in validate_store(store, rules=[GAP_RULE]).failed()


def _relabelled(directory: Path, monkeypatch: pytest.MonkeyPatch, assembly: str) -> Path:
    """A release holding `1:5:G:T` whose manifest names ``assembly``."""
    store = _release_holding(
        directory, monkeypatch, [vcf_row("1", 5, "G", "T"), vcf_row("1", 10_001, "G", "T")]
    )
    manifest = store / "manifest.json"
    written = json.loads(manifest.read_text(encoding="utf-8"))
    written["reference_assembly"] = assembly
    manifest.write_text(json.dumps(written), encoding="utf-8")
    return store


@pytest.mark.parametrize("assembly", ["hg17", ""])
def test_validate_says_it_cannot_judge_a_release_whose_assembly_it_cannot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, assembly: str
) -> None:
    """Not a quiet pass: the rule is skipped, and says why."""
    result = validate_store(_relabelled(tmp_path, monkeypatch, assembly), rules=[GAP_RULE])
    assert GAP_RULE not in result.failed()
    (reason,) = result.skipped[GAP_RULE]
    assert "names no assembly" in reason and repr(assembly) in reason


def test_validate_does_not_apply_the_rule_to_a_release_on_another_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GRCh38 is the one assembly with a gap table; a GRCh37 release is not judged by
    it, as a rule gated to a layout is not judged on another."""
    result = validate_store(_relabelled(tmp_path, monkeypatch, "GRCh37"), rules=[GAP_RULE])
    assert result.of(GAP_RULE) == []


def test_validate_passes_a_release_holding_a_variant_on_a_single_n_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _release_holding(
        tmp_path, monkeypatch, [vcf_row("17", 83_128_871, "G", "T"), vcf_row("1", 10_001, "G", "T")]
    )
    assert "17:83128871:G:T" in {v.alid for v in VariantAxis(store).all()}
    result = validate_store(store, rules=[GAP_RULE])
    assert GAP_RULE not in result.failed()
    assert _gap_findings(store) == []
