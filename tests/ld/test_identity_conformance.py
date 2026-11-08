"""ogld#3: ogld's identity function held to the engine's own (ADR 0003).

`tests/data/identity/` holds the engine's answers (opengwas/opengwasdb at the commit named on
the files' first line) for awkward inputs, made by `make_fixture.py`. These tests read it;
no expected value here comes from ogld.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pytest

from ogld.identity import (
    IdentityError,
    alid_locus,
    normalise_chromosome,
    orient,
    resolve_reversed_indels,
)

DATA = Path(__file__).resolve().parent / "data" / "identity"


def _rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as handle:
        header = handle.readline()
        assert header.startswith("# opengwasdb "), "the fixture must name its engine commit"
        return list(csv.DictReader(handle, delimiter="\t"))


ORIENT = _rows("orient.tsv")
CHROMOSOMES = _rows("chromosomes.tsv")
REVERSED = _rows("reversed.tsv")


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
    assert {r["reason"] for r in ORIENT} == {"", "allele_invalid", "chromosome_missing"}
    # Rows with two faults, whose message names the one the engine judges first.
    messages = {(r["chrom"], r["position"], r["ref"], r["alt"]): r["error"] for r in refused}
    assert messages[("22", "0", "N", "G")] == "unsupported allele 'N'"
    assert messages[("22", "100", "N", "X")] == "unsupported allele 'X'"
    assert messages[("27", "100", "N", "G")] == "unsupported chromosome '27'"
    assert {r["canonical"] for r in CHROMOSOMES if "+22" in r["label"] or "-1" in r["label"]}
    assert sum(1 for r in CHROMOSOMES if r["error"]) > 10
    assert len(REVERSED) >= 8
    assert {r["kept"] for r in REVERSED} == {"true", "false"}
    assert {r["canonical"] for r in CHROMOSOMES} >= {"1", "22", "X", "Y", "MT", ""}


@pytest.mark.parametrize(
    "row", ORIENT, ids=lambda r: f"{r['chrom']}:{r['position']}:{r['ref']}>{r['alt']}"
)
def test_orient_is_the_engine_s(row: dict[str, str]) -> None:
    position: Any = row["position"]
    if position.lstrip("-").isdigit():
        position = int(position)
    args = (row["chrom"], position, row["ref"], row["alt"])
    if row["error"]:
        with pytest.raises(IdentityError, match=rf"\A{re.escape(row['error'])}\Z") as fault:
            orient(*args)
        assert fault.value.reason == row["reason"]
        return
    o = orient(*args)
    assert o.alid == row["alid"]
    assert (o.a1, o.a2) == (row["a1"], row["a2"])
    assert o.a1_is_alt is (row["a1_is_alt"] == "true")
    # A label the engine passes through (`chr 22`, `-1`, `HLA-A`) forms an ALID no reader finds.
    named = o.chrom in {*map(str, range(1, 23)), "X", "Y", "MT"}
    assert alid_locus(o.alid) == ((o.chrom, o.position) if named else None)


@pytest.mark.parametrize("row", CHROMOSOMES, ids=lambda r: repr(r["label"]))
def test_normalise_chromosome_is_the_engine_s(row: dict[str, str]) -> None:
    if row["error"]:
        with pytest.raises(IdentityError, match=rf"\A{re.escape(row['error'])}\Z") as fault:
            normalise_chromosome(row["label"])
        assert fault.value.reason == row["reason"]
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
        "22:100:G:A",
        "22:100:A:A",
        "foo:100:A:G",
        "22:2147483648:A:G",
        "22:" + "9" * 5000 + ":A:G",
        "22:" + "1" * 4301 + ":A:G",
        "22:100:A:N",
        ":100:A:G",
        "22: 100:A:G",
    ],
)
def test_a_malformed_alid_has_no_locus(alid: str) -> None:
    assert alid_locus(alid) is None


@pytest.mark.parametrize(
    ("alid", "locus"),
    [("22:2147483647:A:G", ("22", 2147483647)), ("MT:1:C:T", ("MT", 1)), ("X:5:A:AC", ("X", 5))],
)
def test_an_alid_at_the_bounds_names_its_locus(alid: str, locus: tuple[str, int]) -> None:
    assert alid_locus(alid) == locus


def test_an_alid_over_32_bytes_names_its_locus() -> None:
    long = [r for r in ORIENT if r["alid"] and len(r["alid"]) > 32]
    assert long
    for row in long:
        chrom, pos = row["alid"].split(":")[:2]
        assert alid_locus(row["alid"]) == (chrom, int(pos))


def _records(group: str) -> list[dict[str, str]]:
    return [r for r in REVERSED if r["group"] == group]


@pytest.mark.parametrize("group", sorted({r["group"] for r in REVERSED}))
def test_a_reversed_indel_pair_keeps_the_record_whose_ref_the_alid_spells_first(
    group: str,
) -> None:
    rows = _records(group)
    assert len(rows) == 2
    assert rows[0]["alid"] == rows[1]["alid"], "the fixture's pair must share one ALID"
    records = [(r["chrom"], int(r["position"]), r["ref"], r["alt"]) for r in rows]
    for order in ([0, 1], [1, 0]):
        shuffled = [records[i] for i in order]
        kept, dropped = resolve_reversed_indels(shuffled)
        assert [rows[order[i]]["kept"] for i in kept] == ["true"]
        assert [rows[order[i]]["kept"] for i in dropped] == ["false"]


def test_records_with_no_shared_alid_are_all_kept() -> None:
    records = [
        ("22", 5, "T", "TA"),
        ("22", 6, "TA", "T"),
        ("21", 5, "TA", "T"),
        ("22", 5, "T", "G"),
    ]
    assert resolve_reversed_indels(records) == ([0, 1, 2, 3], [])


@pytest.mark.parametrize(
    "records",
    [
        [("22", 5, "TA", "T"), ("22", 5, "ta", "t")],
        [("22", 5, "T", "TA"), ("22", 5, "TA", "T"), ("22", 5, "TA", "T")],
        [("22", 5, "A", "G"), ("22", 5, "G", "A")],
    ],
    ids=["identical", "third-copy", "snv-pair"],
)
def test_two_records_no_split_gives_are_refused(records: list[tuple[str, int, str, str]]) -> None:
    with pytest.raises(IdentityError) as fault:
        resolve_reversed_indels(records)
    assert fault.value.reason == "duplicate_record"


def test_a_record_the_engine_refuses_is_refused_here_too() -> None:
    with pytest.raises(IdentityError, match="unsupported allele 'N'"):
        resolve_reversed_indels([("22", 5, "N", "A")])


def test_the_cli_exits_2_for_an_identity_error_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ogld import cli

    def refuse(location: str) -> None:
        normalise_chromosome("27")

    monkeypatch.setattr(cli, "open_store", refuse)
    assert cli.main(["info", "somewhere"]) == 2
    assert "unsupported chromosome '27'" in capsys.readouterr().err
