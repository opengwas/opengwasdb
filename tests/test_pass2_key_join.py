"""Dense Pass 2 finds each association's row by an integer key join (#316).

A Routing Key becomes one ``uint64``: its chromosome, its position, and its
alleles -- exactly, for a one-base SNV, or by a hash of the whole key for
anything else. A hash only narrows the search: a row whose key is hashed is
matched only where its whole Routing Key equals the table's, so two keys a
hash sends to one integer can never trade rows.

The spills are pinned against the ones the binary search over Routing Key bytes
wrote at `39c74f2` (`pass2_fixtures`), and each fixture is shown to hold the
case it is there for before the spills are compared.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pass2_fixtures as fixtures
import pytest
from admission_helpers import record_analyses

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.interface import ReaderAssociation, ReferenceAllele
from opengwasdb.variants.routing import RouteTable
from opengwasdb.variants.routing import keys as routing_keys
from opengwasdb.variants.routing.keys import MAX_CHROMOSOMES, IntegerKeyCodec, routing_key

GOLDEN = json.loads(fixtures.GOLDEN.read_text(encoding="utf-8"))


def _built(directory: Path, which: str) -> tuple[Path, fixtures.Spills]:
    with pytest.MonkeyPatch.context() as patch, fixtures.captured_spills(patch) as spills:
        store = fixtures.build(directory, which)
    return store, spills


@pytest.fixture(scope="module")
def builds(tmp_path_factory) -> dict[str, tuple[Path, fixtures.Spills]]:
    root = tmp_path_factory.mktemp("pass2")
    return {which: _built(root / which, which) for which in fixtures.BUILDS}


def _dropped(store: Path) -> dict[str, dict[str, int]]:
    return {entry["analysis_id"]: entry["dropped"] for entry in record_analyses(store)}


def _assert_spills_match_golden(which: str, store: Path, spills: fixtures.Spills) -> None:
    pinned = GOLDEN[which]
    assert fixtures.axis(store) == pinned["alids"]
    assert sorted(map(str, spills)) == sorted(pinned["columns"])
    for col, arrays in spills.items():
        for name, array in arrays.items():
            expected = pinned["columns"][str(col)][name]
            assert str(array.dtype) == expected["dtype"], (which, col, name)
            np.testing.assert_array_equal(
                array, np.array(expected["values"], dtype=expected["dtype"]),
                err_msg=f"{which} column {col} {name}",
            )


# ── the fixtures hold their cases ────────────────────────────────────────────


def test_the_golden_spills_are_not_vacuous():
    for which in fixtures.BUILDS:
        columns = GOLDEN[which]["columns"]
        assert sorted(columns) == ["0", "1"], which
        assert all(len(column["rows"]["values"]) >= 5 for column in columns.values()), which


def test_the_sources_spell_long_indels_and_lower_and_mixed_case():
    rows = fixtures.ALPHA + fixtures.BETA
    longest = max(len(routing_key((chrom, pos, ref, alt))) for chrom, pos, ref, alt, *_ in rows)
    assert longest > 300
    assert any(ref.islower() and alt.islower() for _, _, ref, alt, *_ in rows)
    assert any(
        not (ref + alt).islower() and not (ref + alt).isupper() for _, _, ref, alt, *_ in rows
    )


@pytest.mark.parametrize("which", fixtures.BUILDS)
def test_each_build_meets_every_case(builds, which):
    store, spills = builds[which]
    alids = fixtures.axis(store)
    # An insertion and a deletion at one position reach two rows, long or short.
    assert {"3:6000:G:GGGGCC", "3:6000:GGGGCC:G"} <= set(alids)
    assert {f"3:5000:A:{fixtures.INSERTED}", f"3:5000:{fixtures.DELETED}:A"} <= set(alids)
    # A site spelt in two cases is one row.
    assert alids.count("3:7000:C:T") == 1 and "3:10000:ACGT:A" in alids
    # The reverse-strand lift re-signs alpha's 2:1000 G>A, whose source z is +2.
    at = alids.index("2:999001:C:T")
    alpha = spills[0]
    assert alpha["z"][alpha["rows"].tolist().index(at)] == pytest.approx(-2.0)
    # Admission's duplicate rule collapses copies and drops conflicts; some
    # rows reach no axis.
    dropped = _dropped(store)
    assert dropped["alpha"]["duplicate_collapsed"] == 2
    assert dropped["alpha"]["duplicate_conflicting"] == 2
    assert all(reasons["off_axis"] >= 1 for reasons in dropped.values())


def test_the_single_pass_build_leaves_the_second_analyses_own_sites_unmatched(builds):
    store, _ = builds["single_pass"]
    assert "3:12000:C:T" not in fixtures.axis(store)
    assert _dropped(store)["beta"]["off_axis"] == 5


# ── the spills are the search's ──────────────────────────────────────────────


@pytest.mark.parametrize("which", fixtures.BUILDS)
def test_pass2_spills_are_the_ones_the_byte_key_search_wrote(builds, which):
    store, spills = builds[which]
    _assert_spills_match_golden(which, store, spills)


# ── a colliding hash ─────────────────────────────────────────────────────────


def _every_key_collides(key: bytes) -> int:
    return 0


def test_a_hash_sending_every_key_to_one_value_leaves_every_spill_unchanged(
    tmp_path, monkeypatch
):
    """Every hashed key at a site then shares one integer, so only the full
    Routing Key tells the long insertion from the long deletion, or `A>AT`
    from `AT>A`."""
    monkeypatch.setattr(routing_keys, "key_hash", _every_key_collides)
    for which in fixtures.BUILDS:
        store, spills = _built(tmp_path / which, which)
        _assert_spills_match_golden(which, store, spills)


#: Indels at one position, and one an association names that no source did.
_INDELS = [("1", 100, "A", "AT"), ("1", 100, "AT", "A"), ("1", 100, "A", "ATT"),
           ("1", 100, "ATTT", "A"), ("1", 100, "a", "ac")]
_ABSENT = ("1", 100, "A", "AG")


def _association(site: tuple[str, int, str, str], z: float) -> ReaderAssociation:
    chrom, pos, ref, alt = site
    return ReaderAssociation(
        chrom, pos, ref, alt, z, 0.1, StoredEffectScale.SD, reference=ReferenceAllele.REF
    )


def _encoded(sites: list[tuple[str, int, str, str]], chromosomes: list[str]):
    """``sites``' integer keys, under the hash a table built now would use."""
    codec = IntegerKeyCodec(chromosomes, routing_keys.key_hash)
    return codec.encode(*map(list, zip(*sites, strict=True)))


def test_keys_a_hash_sends_to_one_integer_each_reach_their_own_row(monkeypatch):
    monkeypatch.setattr(routing_keys, "key_hash", _every_key_collides)
    alids = [f"1:100:{ref.upper()}:{alt.upper()}" for _, _, ref, alt in _INDELS]
    table = RouteTable.build(
        dict(zip(_INDELS, alids, strict=True)), (), [alids], keep_unmatched=True
    )
    queried = [*_INDELS, _ABSENT]
    encoded = _encoded(queried, ["1"])
    # Fixture first: every query, the absent one too, shares one integer key.
    assert len(set(encoded.values.tolist())) == 1
    assert encoded.hashed.tolist() == list(range(len(queried)))

    column = table.route_column(
        [_association(site, float(i)) for i, site in enumerate(reversed(queried))]
    )
    placed = column.placed[0]
    by_index = dict(zip(placed.index.tolist(), placed.z.tolist(), strict=True))
    n = len(queried)
    assert by_index == {i: float(n - 1 - i) for i in range(len(_INDELS))}
    assert column.unmatched.keys == [routing_key(_ABSENT)]


# ── the integer key's limits ─────────────────────────────────────────────────


def test_a_table_codes_at_most_max_chromosomes_labels():
    """A chromosome code is 16 bits: one label more than they hold is
    refused, never wrapped onto another chromosome's code."""
    labels = [f"contig{i}" for i in range(MAX_CHROMOSOMES + 1)]
    IntegerKeyCodec(labels[:-1], routing_keys.key_hash)
    with pytest.raises(ValueError, match=f"names {MAX_CHROMOSOMES + 1} chromosomes"):
        IntegerKeyCodec(labels, routing_keys.key_hash)


def test_a_position_the_field_cannot_hold_is_hashed_and_matched_on_the_full_key():
    """A position is 30 bits. One past them is hashed and confirmed, so it
    reaches its own row and no other: written into the field, 2^30 + 1 on
    chromosome 1 would carry into the chromosome bits and read as 2:1."""
    big, edge = ("1", 2**30, "A", "G"), ("1", 2**30 - 1, "A", "G")
    absent, other = ("1", 2**30 + 1, "A", "G"), ("2", 1, "A", "G")
    sites = {big: "1:1073741824:A:G", edge: "1:1073741823:A:G", other: "2:1:A:G"}
    alids = list(sites.values())
    table = RouteTable.build(sites, (), [alids], keep_unmatched=True)
    queried = [big, edge, absent, other]
    # Fixture first: the two positions past the field are hashed, the edge is exact.
    assert _encoded(queried, ["1", "2"]).hashed.tolist() == [0, 2]

    column = table.route_column([_association(site, float(i)) for i, site in enumerate(queried)])

    placed = column.placed[0]
    assert dict(zip(placed.index.tolist(), placed.z.tolist(), strict=True)) == {
        0: 0.0, 1: 1.0, 2: 3.0
    }
    assert column.unmatched.keys == [routing_key(absent)]
