"""Unit tests for the BESD ESI/EPI index readers (issue #130 fix lane).

read_esi/read_epi used to share no code and silently `continue` past any
non-comment row they could not parse, so a row with too few columns, a
malformed bp, or an unparsable frequency disappeared without a trace — and
because row_idx is assigned by enumeration, every *later* row silently
shifted against the `.besd` association file. This file pins the loud
behaviour: comments and blanks stay skippable, everything else either parses
or raises a BESDMetadataError that names the file and the physical line.
"""

import pytest

from opengwasdb.layouts.ragged.besd_reader import (
    BESDMetadataError,
    read_epi,
    read_esi,
)

# ── Happy paths ──────────────────────────────────────────────────────────────

def test_read_esi_skips_comments_and_blanks_but_numbers_data_rows(tmp_path):
    p = tmp_path / "s.esi"
    p.write_text(
        "# eQTLGen-style BESD index\n"
        "\n"
        "1\trs1\t0\t100\tA\tG\t0.1\n"
        "   \n"
        "2\trs2\t0\t200\tC\tT\t0.05\n"
    )
    snps = read_esi(p)
    assert [s.snp_id for s in snps] == ["rs1", "rs2"]
    # row_idx counts data rows only (that is what .besd associations index)
    assert [s.row_idx for s in snps] == [0, 1]
    assert snps[0].chromosome == "1" and snps[0].bp == 100
    assert snps[0].a1 == "A" and snps[0].a2 == "G"
    assert snps[0].freq == 0.1


def test_read_esi_frequency_absent_or_na_parses_to_none(tmp_path):
    p = tmp_path / "s.esi"
    p.write_text(
        "1\trs1\t0\t100\tA\tG\n"       # 6 columns: no frequency column at all
        "1\trs2\t0\t200\tC\tT\tNA\n"   # explicit NA frequency
    )
    snps = read_esi(p)
    assert [s.snp_id for s in snps] == ["rs1", "rs2"]
    assert [s.freq for s in snps] == [None, None]


def test_read_epi_parses_rows_and_numbers_probes(tmp_path):
    p = tmp_path / "s.epi"
    p.write_text(
        "# probe index\n"
        "1\tENSG00000000001\t0\t1050000\tGENE1\t+\n"
        "\n"
        "1\tENSG00000000002\t0\t1150000\tGENE2\t-\n"
    )
    probes = read_epi(p)
    assert [pr.probe_id for pr in probes] == ["ENSG00000000001", "ENSG00000000002"]
    assert [pr.row_idx for pr in probes] == [0, 1]
    assert probes[1].probe_bp == 1150000
    assert probes[1].gene == "GENE2" and probes[1].orientation == "-"


# ── Fail loudly: file and line context ──────────────────────────────────────

def test_read_esi_rejects_row_with_fewer_than_minimum_columns(tmp_path):
    """A row cut off inside the allele columns must not be dropped silently."""
    p = tmp_path / "s.esi"
    p.write_text(
        "1\trs1\t0\t100\tA\tG\t0.1\n"
        "1\trs2\t0\t200\tA\n"  # a2 (and frequency) truncated away
    )
    with pytest.raises(BESDMetadataError) as excinfo:
        read_esi(p)
    msg = str(excinfo.value)
    assert "s.esi" in msg and ":2" in msg
    assert "ESI" in msg and "column" in msg


def test_read_esi_rejects_malformed_bp_naming_file_line_and_value(tmp_path):
    p = tmp_path / "s.esi"
    p.write_text(
        "1\trs1\t0\t100\tA\tG\t0.1\n"
        "1\trs2\t0\tnope\tA\tG\t0.2\n"
    )
    with pytest.raises(BESDMetadataError) as excinfo:
        read_esi(p)
    msg = str(excinfo.value)
    assert "s.esi" in msg and ":2" in msg and "nope" in msg


def test_read_esi_rejects_malformed_frequency_but_accepts_na(tmp_path):
    p = tmp_path / "s.esi"
    p.write_text(
        "1\trs1\t0\t100\tA\tG\tNA\n"
        "1\trs2\t0\t200\tC\tT\t0.3\n"
        "1\trs3\t0\t300\tG\tA\tbanana\n"
    )
    with pytest.raises(BESDMetadataError) as excinfo:
        read_esi(p)
    msg = str(excinfo.value)
    assert "s.esi" in msg and ":3" in msg and "banana" in msg


def test_read_esi_rejects_rows_with_more_columns_than_the_format_defines(tmp_path):
    p = tmp_path / "s.esi"
    p.write_text("1\trs1\t0\t100\tA\tG\t0.1\textra\n")
    with pytest.raises(BESDMetadataError) as excinfo:
        read_esi(p)
    msg = str(excinfo.value)
    assert "s.esi" in msg and ":1" in msg


def test_read_epi_rejects_truncated_probe_rows(tmp_path):
    """A probe row missing gene/orientation (or cut before probe_bp) must raise."""
    for lineno, row in [
        (2, "1\tENSG00000000001\t0\t1050000"),  # no gene, no orientation
        (2, "1\tENSG00000000001\t0\t1050000\tGENE1"),  # no orientation
    ]:
        p = tmp_path / "s.epi"
        p.write_text(
            "1\tENSG00000000000\t0\t1000\tGENE0\t+\n"
            f"{row}\n"
        )
        with pytest.raises(BESDMetadataError) as excinfo:
            read_epi(p)
        msg = str(excinfo.value)
        assert "s.epi" in msg and f":{lineno}" in msg and "EPI" in msg


def test_read_epi_rejects_malformed_probe_bp(tmp_path):
    p = tmp_path / "s.epi"
    p.write_text("1\tENSG00000000001\t0\tbp_not_int\tGENE1\t+\n")
    with pytest.raises(BESDMetadataError) as excinfo:
        read_epi(p)
    msg = str(excinfo.value)
    assert "s.epi" in msg and ":1" in msg and "bp_not_int" in msg


# ── Callers surface the failure ─────────────────────────────────────────────

def test_build_ragged_from_besd_surfaces_malformed_esi(tmp_path):
    """build_ragged_from_besd must not swallow the metadata error."""
    from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd

    (tmp_path / "bad.esi").write_text(
        "1\trs1\t0\t100\tA\tG\t0.1\n"
        "1\trs2\t0\t200\tA\n"
    )
    (tmp_path / "bad.epi").write_text("# no probes\n")
    with pytest.raises(BESDMetadataError) as excinfo:
        build_ragged_from_besd(
            tmp_path / "bad", tmp_path / "out.opengwasdb", store_id="s", release_id="r"
        )
    msg = str(excinfo.value)
    assert "bad.esi" in msg and ":2" in msg
