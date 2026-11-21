"""Published LD import through the CLI, existing validator and reader (#93)."""

import json
from pathlib import Path

import numpy as np

import ogld
from ogld.cli import main
from ogld.format import REQUIRED_KEYS

FIXTURE = Path(__file__).parent / "data" / "panukb"


def import_args(out: Path, work: Path) -> list[str]:
    return [
        "import",
        str(out),
        "--work",
        str(work),
        "--metadata",
        str(FIXTURE / "metadata.json"),
        "--variants",
        str(FIXTURE / "variants.tsv"),
        "--chain",
        str(FIXTURE / "identity.chain"),
        "--fasta",
        str(FIXTURE / "reference.fa.gz"),
        "--ledger",
        str(FIXTURE / "ledger.json"),
        "--parts",
        str(FIXTURE),
    ]


def test_cli_import_writes_literal_signed_codes_and_complete_provenance(tmp_path, capsys) -> None:
    out = tmp_path / "import.store"
    assert main(import_args(out, tmp_path / "work")) == 0
    assert main(["validate", str(out)]) == 0
    with ogld.open(out) as store:
        assert store.provenance["n_variants"] == 6
        got = store.matrix(["1:1:A:C", "1:2:A:G", "1:4000001:A:G"]).r
        np.testing.assert_array_equal(
            got * 2047, [[2047, -1024, -1024], [-1024, 2047, -512], [-1024, -512, 2047]]
        )
        record = store.provenance
    for dotted in REQUIRED_KEYS:
        value = record
        for key in dotted.split("."):
            value = value[key]
    source = json.loads(record["source"]["description"])
    assert source["producer_commit"] == "ca9a04cd4c95ebcd6eb92b2096281508b8becf86"
    assert source["far_bp_by_chromosome"] == {"1": 10_000_000, "2": 10_000_000}
    assert source["transformation"] == "r_ij = C_ij / sqrt(C_ii * C_jj); oriented to engine A1"
    assert source["url"].startswith("https://pan-ukb")
    assert len(record["source"]["sha256"]) == 6  # metadata, TSV, chain, FASTA and two parts
    assert record["far_code_floor"] == 65
    assert record["t_far"] == record["t_partner"] == 0.001
    assert "ok  " in capsys.readouterr().out
