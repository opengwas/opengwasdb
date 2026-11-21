"""Published LD import through the CLI, existing validator and reader (#93)."""

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pytest

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


@pytest.mark.parametrize("alias", ["direct", "supplied-symlink", "cache-symlink"])
def test_cli_delete_preserves_supplied_parts_and_deletes_only_downloads(
    tmp_path, monkeypatch, alias
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cache = work / "parts"
    if alias == "cache-symlink":
        supplied = tmp_path / "supplied"
        supplied.mkdir()
        cache.symlink_to(supplied, target_is_directory=True)
    else:
        cache.mkdir()
        supplied = cache
        if alias == "supplied-symlink":
            supplied = tmp_path / "supplied-alias"
            supplied.symlink_to(cache, target_is_directory=True)
    original = (FIXTURE / "part-0").read_bytes()
    (supplied / "part-0").write_bytes(original)
    fetches = []

    def fetch(url, *, timeout):
        name = url.rsplit("/", 1)[-1]
        fetches.append(name)
        return io.BytesIO((FIXTURE / name).read_bytes())

    monkeypatch.setattr(urllib.request, "urlopen", fetch)
    args = import_args(tmp_path / "import.store", work)
    args[args.index("--parts") + 1] = str(supplied)
    assert main([*args, "--delete-parts"]) == 0
    assert fetches == ["part-1"]
    assert (supplied / "part-0").read_bytes() == original
    assert not (cache / "part-1").exists()
    assert not (supplied / "part-1").exists()
    assert json.loads((cache / ".sha256/part-0.json").read_text())["downloaded"] is False
    assert json.loads((cache / ".sha256/part-1.json").read_text())["downloaded"] is True


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
