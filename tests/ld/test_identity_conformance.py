"""ogld#3: ogld's identity function held to the engine's own (ADR 0003).

`tests/data/identity/` holds the engine's answers (opengwas/opengwasdb at the commit named on
the files' first line) for awkward inputs, made by `make_fixture.py`. These tests read it;
no expected value here comes from ogld.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from ogld.identity import alid_locus, normalise_chromosome, orient

DATA = Path(__file__).resolve().parent / "data" / "identity"


def _rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as handle:
        header = handle.readline()
        assert header.startswith("# opengwasdb "), "the fixture must name its engine commit"
        return list(csv.DictReader(handle, delimiter="\t"))


ORIENT = _rows("orient.tsv")
CHROMOSOMES = _rows("chromosomes.tsv")


def test_the_fixture_is_meaningful() -> None:
    # Asserted before anything is asserted about it: both outcomes, both signs, long ALIDs.
    refused = [r for r in ORIENT if r["error"]]
    kept = [r for r in ORIENT if not r["error"]]
    assert len(kept) > 40
    assert len(refused) > 15
    assert {r["a1_is_alt"] for r in kept} == {"true", "false"}
    assert any(len(r["alid"]) > 32 for r in kept)
    assert any(r["ref"] != r["ref"].upper() for r in kept)
    assert any(len(r["ref"]) != len(r["alt"]) for r in kept)
    assert sum(1 for r in CHROMOSOMES if r["error"]) > 10
    assert {r["canonical"] for r in CHROMOSOMES} >= {"1", "22", "X", "Y", "MT", ""}


@pytest.mark.parametrize(
    "row", ORIENT, ids=lambda r: f"{r['chrom']}:{r['position']}:{r['ref']}>{r['alt']}"
)
def test_orient_is_the_engine_s(row: dict[str, str]) -> None:
    args = (row["chrom"], int(row["position"]), row["ref"], row["alt"])
    if row["error"]:
        with pytest.raises(ValueError, match=rf"\A{re.escape(row['error'])}\Z"):
            orient(*args)
        return
    o = orient(*args)
    assert o.alid == row["alid"]
    assert (o.a1, o.a2) == (row["a1"], row["a2"])
    assert o.a1_is_alt is (row["a1_is_alt"] == "true")
    assert alid_locus(o.alid) == (o.chrom, o.position)


@pytest.mark.parametrize("row", CHROMOSOMES, ids=lambda r: repr(r["label"]))
def test_normalise_chromosome_is_the_engine_s(row: dict[str, str]) -> None:
    if row["error"]:
        with pytest.raises(ValueError, match=rf"\A{re.escape(row['error'])}\Z"):
            normalise_chromosome(row["label"])
    else:
        assert normalise_chromosome(row["label"]) == row["canonical"]


@pytest.mark.parametrize(
    "alid",
    [
        "22:16050075:A",
        "22:16050075:A:G:T",
        "chr22:100:A:G",
        "27:100:A:G",
        "22:0:A:G",
        "22:-1:A:G",
        "22:0100:A:G",
        "22:\u0661\u0660\u0660:A:G",
        "22:100:A:",
        "22:100::G",
        "22:100:a:G",
        "22:100:A:N",
        ":100:A:G",
        "22: 100:A:G",
    ],
)
def test_a_malformed_alid_has_no_locus(alid: str) -> None:
    assert alid_locus(alid) is None


def test_an_alid_over_32_bytes_names_its_locus() -> None:
    long = [r for r in ORIENT if r["alid"] and len(r["alid"]) > 32]
    assert long
    for row in long:
        chrom, pos = row["alid"].split(":")[:2]
        assert alid_locus(row["alid"]) == (chrom, int(pos))
