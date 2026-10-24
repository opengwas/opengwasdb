"""Tests for the build-collision reader of the OGS-00009/OGS-00010 comparison (#312, #325).

A collision row holds one of two source variants' statistics. The reader must
name which one from the source VCFs, refuse to report a row whose statistics
match neither, and never lift with today's code under an older revision's name.
"""

from __future__ import annotations

import gzip
import math
from pathlib import Path

import pysam
import pytest

from benchmarks import _build_collisions
from benchmarks._build_collisions import (
    archived_lift,
    collision_rows,
    describe_collision,
    oriented_z,
    store_lift,
    vcf_records,
)

REPO = Path(__file__).resolve().parents[1]
ALID = "7:5000:A:G"


def _vcf(tmp_path: Path) -> Path:
    plain = tmp_path / "source.vcf"
    plain.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=7,length=100000>\n"
        '##FORMAT=<ID=ES,Number=A,Type=Float,Description="ES">\n'
        '##FORMAT=<ID=SE,Number=A,Type=Float,Description="SE">\n'
        '##FORMAT=<ID=EZ,Number=A,Type=Float,Description="EZ">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n"
        "7\t100\trsA\tA\tG\t.\tPASS\t.\tES:SE\t0.2:0.1\n"
        "7\t200\trsB\tC\tT\t.\tPASS\t.\tES:SE\t-0.3:0.1\n"
        "7\t300\trsC\tG\tT\t.\tPASS\t.\tEZ:ES:SE\t1.5:0.9:0.1\n",
        encoding="utf-8",
    )
    return Path(pysam.tabix_index(str(plain), preset="vcf", force=True))


def _lift(key: tuple[str, int, str, str]) -> tuple[str, bool] | None:
    """Both sources land on ALID; the second through a reverse block, so it swaps."""
    return {("7", 100, "A", "G"): (ALID, False), ("7", 200, "C", "T"): (ALID, True)}.get(key)


def _sources(chrom: str, pos: int) -> list[tuple[str, int, str]]:
    assert (chrom, pos) == ("7", 5000)
    return [("7", 100, "+"), ("7", 200, "-"), ("7", 300, "+")]


def test_vcf_records_prefer_ez_and_fall_back_to_es_over_se(tmp_path):
    vcf = _vcf(tmp_path)
    assert [r["z_alt"] for r in vcf_records(vcf, "7", 100)] == pytest.approx([2.0])
    assert [r["z_alt"] for r in vcf_records(vcf, "7", 300)] == pytest.approx([1.5])
    assert vcf_records(vcf, "7", 150) == []


def test_describe_collision_names_the_source_whose_statistics_the_row_holds(tmp_path):
    vcf = _vcf(tmp_path)
    second = oriented_z(("7", 200, "C", "T"), -3.0, True)
    assert second == pytest.approx(-3.0) and oriented_z(("7", 100, "A", "G"), 2.0, False) == -2.0
    record = describe_collision(ALID, lift=_lift, sources=_sources,
                                vcfs={"a1": vcf, "a2": vcf}, stored_z=lambda *_: second)
    assert [c["source"] for c in record["sources"]] == ["7:100:A:G", "7:200:C:T"]
    assert record["statistics_from"] == "7:200:C:T"
    assert record["matches_neither"] is False


def test_describe_collision_flags_statistics_matching_neither_source(tmp_path):
    vcf = _vcf(tmp_path)
    record = describe_collision(ALID, lift=_lift, sources=_sources, vcfs={"a1": vcf},
                                stored_z=lambda *_: 0.5)
    assert record["matches_neither"] is True
    assert record["statistics_from"] == "not determined"


def test_describe_collision_without_a_stored_value_is_not_determined(tmp_path):
    vcf = _vcf(tmp_path)
    record = describe_collision(ALID, lift=_lift, sources=_sources, vcfs={"a1": vcf},
                                stored_z=lambda *_: math.nan)
    assert record["statistics_from"] == "not determined"
    assert record["matches_neither"] is False


def test_collision_rows_are_the_blank_source_alids(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    header = "\t".join(["#chromosome", "position", "variant_index", "effect_allele",
                        "other_allele", "alid", "rsid", "source_alid"]) + "\n"
    rows = ["7\t5000\t0\tA\tG\t7:5000:A:G\trsA\t.\n",
            "7\t6000\t1\tC\tT\t7:6000:C:T\trsB\t7:1:C:T\n"]
    with gzip.open(parent / "variants.tsv.gz", "wt") as fh:
        fh.write(header + "".join(rows))
    assert list(collision_rows(parent)["alid"]) == ["7:5000:A:G"]
    with gzip.open(parent / "variants.tsv.gz", "wt") as fh:
        fh.write(header + rows[0])
    with pytest.raises(SystemExit, match="no row records a source_alid"):
        collision_rows(parent)


def _reverse_chain(path: Path) -> Path:
    """hg19 chr5:[1000, 3000) mapped in reverse onto hg38 chr5, as in the comparison's tests."""
    path.write_text("chain 1000 chr5 100000 + 1000 3000 chr5 90000 - 2000 4000 1\n"
                    "1000 100 100\n900\n\n", encoding="utf-8")
    return path


def test_an_older_lift_reextracts_a_tampered_copy_before_running(tmp_path):
    """A workdir copy that is not the revision's blob must never run under its name."""
    chain = _reverse_chain(tmp_path / "rev.chain")
    copy = tmp_path / "f812532" / _build_collisions.LIFTOVER
    copy.parent.mkdir(parents=True)
    copy.write_text("def build_liftover_lookup(variants, chain_file=None, failure_threshold=0):\n"
                    "    return {key: '1:1:A:C' for key in variants}\n", encoding="utf-8")
    lift = archived_lift(REPO, "f812532", tmp_path, chain)
    # f812532's own lift: the chain's position, the source's alleles uncomplemented.
    assert lift(("5", 1500, "A", "C")) == ("5:87501:A:C", False)
    assert "1:1:A:C" not in copy.read_text(encoding="utf-8")


def test_an_older_lift_that_imports_the_package_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="imports the package"):
        archived_lift(REPO, "d405de0", tmp_path, tmp_path / "chain")


def test_store_lift_uses_the_installed_lift_only_for_identical_files(tmp_path):
    assert "installed package" in store_lift(REPO, "d405de0", tmp_path, tmp_path / "c").how
    older = store_lift(REPO, "f812532", tmp_path, tmp_path / "c")
    assert "git archive" in older.how
    assert (tmp_path / "f812532" / _build_collisions.LIFTOVER).is_file()
