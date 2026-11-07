"""The indel-reference table an LD panel's indels are placed by (issue #381).

`scripts/build_panel_indel_reference.py` settles each panel indel by GRCh38
evidence -- the HGDP+1kGP pvars and FinnGen's declared `ref` -- only where a
source reads it one way and none reads it otherwise, and writes the table and
its provenance beside the panel's blocks, never over a file already there. An
evidence file it cannot read to its end fails the build.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pysam
import pytest
from script_modules import load_script

from opengwasdb.completion.ld_panel import load_block

builder = load_script("build_panel_indel_reference")

#: ``(panel id, how the evidence reads it)``: one panel indel per outcome, and
#: an SNP the table never names.
PANEL = [
    ("1:100:A:AT", "settled_pvar"),
    ("1:200:C:CA", "settled_finngen"),
    ("1:300:G:GT", "settled_agree"),
    ("1:400:T:TA", "both_ways"),
    ("1:500:A:AC", "disagree"),
    ("1:600:C:CT", "no_evidence"),
    ("1:700:A:G", None),
]
#: pvar records: 100 a deletion; 300 an insertion; 400 both readings; 500 an
#: insertion that FinnGen reads as a deletion; and an unrelated SNP.
PVAR = [
    ("chr1", 100, "AT", "A"),
    ("chr1", 300, "G", "GT"),
    ("chr1", 400, "T", "TA"),
    ("chr1", 400, "TA", "T"),
    ("chr1", 500, "A", "AC"),
    ("chr1", 700, "A", "G"),
]
FINNGEN = [
    ("1", 200, "CA", "C"),
    ("1", 300, "G", "GT"),
    ("1", 500, "AC", "A"),
]


def _panel(root: Path, *, ukb_ids: bool = False) -> Path:
    """One block of `PANEL`'s variants, named ``chr:pos:a1:a2``, or as UKB
    names them, ``chr:pos_a1_a2``."""
    block_dir = root / "EUR" / "1"
    block_dir.mkdir(parents=True)
    lines = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
    for snp, _outcome in PANEL:
        c, pos, a1, a2 = snp.split(":")
        snp_id = f"{c}:{pos}_{a1}_{a2}" if ukb_ids else snp
        lines.append(f"1\t{snp_id}\t{a2}\t{a1}\t0.3\t{pos}")
    (block_dir / "1-1000.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Loading a block needs its LD to exist; these tests never read it.
    (block_dir / "1-1000.unphased.vcor1.gz").touch()
    return root / "EUR"


def _evidence(root: Path) -> tuple[Path, Path]:
    pvar = root / "chr1.pvar"
    pvar.write_text(
        "##fileformat=PVARv1.0\n#CHROM\tPOS\tID\tREF\tALT\n"
        + "".join(f"{c}\t{p}\t.\t{r}\t{a}\n" for c, p, r, a in PVAR),
        encoding="utf-8",
    )
    finngen = root / "finngen.tsv"
    finngen.write_text(
        "#chrom\tpos\tref\talt\trsids\n"
        + "".join(f"{c}\t{p}\t{r}\t{a}\t\n" for c, p, r, a in FINNGEN),
        encoding="utf-8",
    )
    return pvar, finngen


def _table(panel: Path) -> list[list[str]]:
    """The table's rows, through its tabix index, as completion reads it."""
    with pysam.TabixFile(str(panel / "indel_reference.tsv.gz")) as tabix:
        return [line.split("\t") for line in tabix.fetch("1")]


def test_each_panel_indel_is_settled_or_left_with_its_reason(tmp_path: Path):
    panel = _panel(tmp_path / "panel")
    pvar, finngen = _evidence(tmp_path)

    provenance = builder.build(panel, [pvar], finngen)

    assert _table(panel) == [
        ["1", "100", "A", "AT", "AT", "settled_pvar"],
        ["1", "200", "C", "CA", "CA", "settled_finngen"],
        ["1", "300", "G", "GT", "G", "settled_agree"],
        ["1", "400", "T", "TA", ".", "both_ways"],
        ["1", "500", "A", "AC", ".", "disagree"],
        ["1", "600", "C", "CT", ".", "no_evidence"],
    ]
    assert provenance["n_panel_indels"] == 6
    assert provenance["counts"] == {outcome: 1 for outcome in builder.OUTCOMES}
    recorded = json.loads((panel / "indel_reference.json").read_text())
    assert recorded["counts"] == provenance["counts"]
    assert [(e["role"], Path(e["path"]).name) for e in recorded["evidence"]] == [
        ("pvar", "chr1.pvar"), ("finngen_ref", "finngen.tsv"),
    ]
    assert all(len(e["sha256"]) == 64 for e in recorded["evidence"])


@pytest.mark.parametrize("ukb_ids", [False, True], ids=["colon-ids", "ukb-ids"])
def test_completion_places_the_indels_the_table_settles(tmp_path: Path, ukb_ids: bool):
    """The builder keys a panel id as completion does, in either id form."""
    panel = _panel(tmp_path / "panel", ukb_ids=ukb_ids)
    pvar, finngen = _evidence(tmp_path)
    builder.build(panel, [pvar], finngen)

    block = load_block(panel / "1" / "1-1000.tsv")

    assert block is not None
    assert block.alids == [
        "1:100:AT:A", "1:200:CA:C", "1:300:G:GT", None, None, None, "1:700:A:G",
    ]
    # The two deletions' A1 is not their EA, the sorted-first allele.
    assert block.flipped.tolist() == [True, True, False, False, False, False, False]
    assert (block.indel_reference, block.n_indels, block.n_indels_placed) == ("table", 6, 3)


def test_the_builder_never_replaces_a_table(tmp_path: Path):
    panel = _panel(tmp_path / "panel")
    pvar, finngen = _evidence(tmp_path)
    builder.build(panel, [pvar], finngen)
    before = (panel / "indel_reference.tsv.gz").read_bytes()

    with pytest.raises(FileExistsError, match="never replaces"):
        builder.build(panel, [pvar], finngen)

    assert (panel / "indel_reference.tsv.gz").read_bytes() == before


def _zstd(path: Path) -> Path:
    subprocess.run(["zstd", "-q", "-f", str(path), "-o", f"{path}.zst"], check=True)
    return Path(f"{path}.zst")


def test_a_compressed_pvar_is_read_to_its_end(tmp_path: Path):
    panel = _panel(tmp_path / "panel")
    pvar, finngen = _evidence(tmp_path)

    provenance = builder.build(panel, [_zstd(pvar)], finngen)

    assert provenance["counts"] == {outcome: 1 for outcome in builder.OUTCOMES}


def test_a_truncated_compressed_pvar_fails_the_build(tmp_path: Path):
    """Read short, the pvar would hold one reading of 1:400 where it holds
    two, and settle an indel it cannot."""
    panel = _panel(tmp_path / "panel")
    pvar, finngen = _evidence(tmp_path)
    with open(pvar, "a", encoding="utf-8") as fh:
        fh.writelines(f"chr2\t{p}\t.\tA\tG\n" for p in range(1, 20_000))
    compressed = _zstd(pvar)
    data = compressed.read_bytes()
    compressed.write_bytes(data[: len(data) // 2])

    with pytest.raises(subprocess.CalledProcessError):
        builder.build(panel, [compressed], finngen)

    assert not (panel / "indel_reference.tsv.gz").exists()


@pytest.mark.parametrize(
    ("text", "fault"),
    [
        ("##fileformat=PVARv1.0\n", "ends before its header"),
        ("#CHROM\tPOS\tID\tREF\tALT\nchr1\t400\t.\tT\tTA\nchr1\t400\t.\tTA", "cut short"),
    ],
    ids=["no-header", "cut-record"],
)
def test_an_evidence_file_that_ends_early_fails_the_build(tmp_path: Path, text: str, fault: str):
    panel = _panel(tmp_path / "panel")
    _pvar, finngen = _evidence(tmp_path)
    short = tmp_path / "chr2.pvar"
    short.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=fault):
        builder.build(panel, [short], finngen)

    assert not (panel / "indel_reference.tsv.gz").exists()
