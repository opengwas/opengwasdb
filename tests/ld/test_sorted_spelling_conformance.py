"""The sorted spelling an LD Store's ALIDs use, held to ogld's committed answers (ADR 0089).

`tests/ld/data/identity/` holds the engine's answers at opengwasdb `1ab2b09`, before ADR 0079
made indels reference first, when every ALID was spelt sorted (the commit is on each file's
first line; the README there says how they were made). They are the independent source: no
expected value here comes from `opengwasdb.variants`. LD Stores keep that spelling until #462
(owner, 23 November 2026, on #456), so `opengwasdb.variants.sorted_spelling` must give every
answer they hold, and the engine's reference-first rule must differ from them only where the
two spellings differ.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pytest

from opengwasdb.ld import cli
from opengwasdb.store_reader.identity import VariantNormalisationError, normalise_chromosome
from opengwasdb.variants.normalise import orient_to_canonical
from opengwasdb.variants.sorted_spelling import (
    DuplicateRecord,
    resolve_sorted_reversed_indels,
    sorted_alid_locus,
    sorted_identity,
    spells_indel_reference_first,
)

DATA = Path(__file__).resolve().parent / "data" / "identity"


def _rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as handle:
        header = handle.readline()
        assert header.startswith("# opengwasdb 1ab2b09:"), "the fixture is pinned before ADR 0079"
        return list(csv.DictReader(handle, delimiter="\t"))


ORIENT = _rows("orient.tsv")
CHROMOSOMES = _rows("chromosomes.tsv")
REVERSED = _rows("reversed.tsv")
CANONICAL = {*map(str, range(1, 23)), "X", "Y", "MT"}


def _position(text: str) -> Any:
    return int(text) if text.lstrip("-").isdigit() else text


def _alleles(row: dict[str, str]) -> tuple[str, str]:
    return row["ref"].strip().upper(), row["alt"].strip().upper()


def test_the_fixture_is_meaningful() -> None:
    # Asserted before anything is asserted about it: both outcomes, both signs, long ALIDs, and
    # indels on both sides of the line between the spellings.
    refused = [r for r in ORIENT if r["error"]]
    kept = [r for r in ORIENT if not r["error"]]
    assert (len(kept), len(refused)) == (49, 44)
    assert {r["a1_is_alt"] for r in kept} == {"true", "false"}
    assert any(len(r["alid"]) > 32 for r in kept)
    assert any(r["ref"] != r["ref"].upper() for r in kept)
    longer_ref = [_alleles(r) for r in kept if len(r["ref"].strip()) > len(r["alt"].strip())]
    assert sum(ref > alt for ref, alt in longer_ref) == 10
    assert sum(ref < alt for ref, alt in longer_ref) == 2  # 22:403 GAT>GT, 22:404 AGT>AT
    assert {r["reason"] for r in ORIENT} == {"", "allele_invalid", "chromosome_missing"}
    # Rows with two faults, whose message names the one judged first.
    messages = {(r["chrom"], r["position"], r["ref"], r["alt"]): r["error"] for r in refused}
    assert messages[("22", "0", "N", "G")] == "unsupported allele 'N'"
    assert messages[("22", "100", "N", "X")] == "unsupported allele 'X'"
    assert messages[("27", "100", "N", "G")] == "unsupported chromosome '27'"
    assert {r["canonical"] for r in CHROMOSOMES if "+22" in r["label"] or "-1" in r["label"]}
    assert sum(1 for r in CHROMOSOMES if r["error"]) > 10
    assert {r["canonical"] for r in CHROMOSOMES} >= {"1", "22", "X", "Y", "MT", ""}
    assert len(REVERSED) == 8
    assert {r["kept"] for r in REVERSED} == {"true", "false"}


@pytest.mark.parametrize(
    "row", ORIENT, ids=lambda r: f"{r['chrom']}:{r['position']}:{r['ref']}>{r['alt']}"
)
def test_the_sorted_identity_is_the_committed_one(row: dict[str, str]) -> None:
    args = (row["chrom"], _position(row["position"]), row["ref"], row["alt"])
    if row["error"]:
        with pytest.raises(
            VariantNormalisationError, match=rf"\A{re.escape(row['error'])}\Z"
        ) as fault:
            sorted_identity(*args)
        assert getattr(fault.value, "reason", "") == row["reason"]
        return
    o = sorted_identity(*args)
    assert o.alid == row["alid"]
    assert (o.a1, o.a2) == (row["a1"], row["a2"])
    assert o.a1_is_alt is (row["a1_is_alt"] == "true")
    assert o.alid == f"{o.chrom}:{o.position}:{o.a1}:{o.a2}"
    # A label the engine passes through (`chr 22`, `-1`, `HLA-A`) forms an ALID no reader finds.
    named = o.chrom in CANONICAL
    assert sorted_alid_locus(o.alid) == ((o.chrom, o.position) if named else None)


@pytest.mark.parametrize("row", CHROMOSOMES, ids=lambda r: repr(r["label"]))
def test_the_engine_s_chromosome_label_is_the_committed_one(row: dict[str, str]) -> None:
    if row["error"]:
        with pytest.raises(
            VariantNormalisationError, match=rf"\A{re.escape(row['error'])}\Z"
        ) as fault:
            normalise_chromosome(row["label"])
        assert getattr(fault.value, "reason", "") == row["reason"]
    else:
        assert normalise_chromosome(row["label"]) == row["canonical"]


def test_the_reference_first_rule_differs_only_where_the_spellings_do() -> None:
    # The engine's rule today (ADR 0079) against the sorted answers: the 10 rows where REF is
    # longer and sorts after ALT, and no other. A longer REF that sorts first (`GAT>GT`) is spelt
    # alike by both, so "REF longer than ALT" alone would name two rows too many.
    differing, expected = [], []
    for row in ORIENT:
        if row["error"]:
            continue
        ref, alt = _alleles(row)
        args = (row["chrom"], _position(row["position"]), row["alt"], row["ref"])
        engine = orient_to_canonical(*args, other_is_reference=True).variant.alid
        if engine != row["alid"]:
            differing.append(row["alid"])
            assert spells_indel_reference_first(engine)
            assert engine.split(":")[2:] == [ref, alt]
        if len(ref) > len(alt) and ref > alt:
            expected.append(row["alid"])
    assert len(differing) == 10
    assert differing == expected


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
        "22:١٠٠:A:G",
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
        "1:827587:GGGGCC:G",
    ],
)
def test_a_malformed_alid_has_no_locus(alid: str) -> None:
    assert sorted_alid_locus(alid) is None


@pytest.mark.parametrize(
    ("alid", "locus"),
    [("22:2147483647:A:G", ("22", 2147483647)), ("MT:1:C:T", ("MT", 1)), ("X:5:A:AC", ("X", 5))],
)
def test_an_alid_at_the_bounds_names_its_locus(alid: str, locus: tuple[str, int]) -> None:
    assert sorted_alid_locus(alid) == locus


def test_an_alid_over_32_bytes_names_its_locus() -> None:
    long = [r for r in ORIENT if r["alid"] and len(r["alid"]) > 32]
    assert long
    for row in long:
        chrom, pos = row["alid"].split(":")[:2]
        assert sorted_alid_locus(row["alid"]) == (chrom, int(pos))


@pytest.mark.parametrize(
    ("alid", "reference_first"),
    [
        ("1:827587:GGGGCC:G", True),
        ("22:400:CAA:CA", True),
        ("22:5:T:AT", True),  # an unanchored insertion whose REF sorts last
        ("1:827587:G:GGGGCC", False),
        ("22:403:GAT:GT", False),  # spelt alike by both
        ("22:100:G:A", False),  # an unsorted SNV: malformed in both, so Absent
        ("27:5:CAA:CA", False),  # no chromosome of the format: Absent
        ("22:5:CAA", False),
    ],
)
def test_only_an_indel_alid_no_sorted_spelling_forms_is_reference_first(
    alid: str, reference_first: bool
) -> None:
    assert spells_indel_reference_first(alid) is reference_first


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
    for r, record in zip(rows, records, strict=True):
        identity = sorted_identity(*record)
        assert identity.alid == r["alid"]
        assert identity.a1_is_alt is (r["a1_is_alt"] == "true")
    for order in ([0, 1], [1, 0]):
        shuffled = [records[i] for i in order]
        kept, dropped = resolve_sorted_reversed_indels(shuffled)
        assert [rows[order[i]]["kept"] for i in kept] == ["true"]
        assert [rows[order[i]]["kept"] for i in dropped] == ["false"]


def test_records_with_no_shared_alid_are_all_kept() -> None:
    records = [
        ("22", 5, "T", "TA"),
        ("22", 6, "TA", "T"),
        ("21", 5, "TA", "T"),
        ("22", 5, "T", "G"),
    ]
    assert resolve_sorted_reversed_indels(records) == ([0, 1, 2, 3], [])


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
    with pytest.raises(DuplicateRecord) as fault:
        resolve_sorted_reversed_indels(records)
    assert fault.value.reason == "duplicate_record"


def test_a_record_the_engine_refuses_is_refused_here_too() -> None:
    with pytest.raises(VariantNormalisationError, match="unsupported allele 'N'"):
        resolve_sorted_reversed_indels([("22", 5, "N", "A")])


def test_the_cli_exits_2_for_an_identity_refusal_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(location: str) -> None:
        normalise_chromosome("27")

    monkeypatch.setattr(cli, "open_store", refuse)
    assert cli.main(["info", "somewhere"]) == 2
    assert "unsupported chromosome '27'" in capsys.readouterr().err
