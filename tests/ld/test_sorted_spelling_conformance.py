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
