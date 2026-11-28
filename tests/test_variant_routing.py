"""Variant Routing at its interface: `route_variants`, `RoutedNames`, `RouteTable`.

Every source is a `FakeReader`, opened through `route_variants`' reader seam,
and every lift goes through a chain whose strands are known by construction
(`liftover_fixtures`), so no Store is built and no fixture VCF is parsed.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from liftover_fixtures import (
    write_collision_chain,
    write_identity_collision_chain,
    write_reverse_collision_chain,
    write_two_strand_chain,
)

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.fake import FakeReader
from opengwasdb.readers.interface import ReaderAssociation, ReferenceAllele, SourceVariant
from opengwasdb.readers.registry import SourceSpec
from opengwasdb.variants.reference import VariantReference, read_variant_reference
from opengwasdb.variants.routing import (
    AssociationBatch,
    FromReference,
    LiftedUnion,
    RoutedNames,
    RouteTable,
    extract_variant_reference,
    place_sites,
    route_variants,
)
from opengwasdb.variants.routing import table as route_table
from opengwasdb.variants.routing.lift import (
    DroppedSource,
    LiftDrops,
    require_one_locus_per_alid,
)
from opengwasdb.variants.rsid import RsidConflictPolicy

#: The fake sources of the running test, by path. Forked Pass 1 workers
#: inherit it, so the reader factory below works at any worker count.
_SOURCES: dict[str, FakeReader] = {}

Site = tuple[str, int, str, str]


@pytest.fixture(autouse=True)
def _fresh_sources() -> Iterator[None]:
    _SOURCES.clear()
    yield
    _SOURCES.clear()


def _fake_reader(spec: SourceSpec) -> FakeReader:
    return _SOURCES[spec.path]


def _source(
    name: str, assembly: str, variants: list[tuple[str, int, str, str, str]]
) -> SourceSpec:
    """A fake source naming ``variants`` as ``(chrom, pos, ref, alt, rsid)``."""
    _SOURCES[name] = FakeReader(
        variants=[SourceVariant(*variant) for variant in variants]
    )
    return SourceSpec(name, name, "fake", assembly, StoredEffectScale.SD)


def _association(site: Site, z: float, *, eaf: float | None = None) -> ReaderAssociation:
    chrom, pos, ref, alt = site
    return ReaderAssociation(
        chrom, pos, ref, alt, z, 0.1, StoredEffectScale.SD, eaf=eaf,
    )


def _route(sources, authority, **kwargs):
    return route_variants(sources, authority, readers=_fake_reader, **kwargs)


def _reference(source_lookup: dict[Site, str]) -> VariantReference:
    """A reference that names no rsids, routing exactly ``source_lookup``."""
    alids = sorted(set(source_lookup.values()))
    return VariantReference(
        alids, source_lookup, {}, explicit_source_keys=True, reoriented=frozenset()
    )


# ── route_variants and RoutedNames ───────────────────────────────────────────


@pytest.mark.parametrize("n_workers", [1, 2])
def test_the_first_source_to_name_a_site_names_its_variant(n_workers):
    """First named wins, in source order -- not rsid order, and not whichever
    worker finished first (issues #109, #192)."""
    first = _source("a", "hg38", [("1", 100, "A", "G", ""), ("1", 200, "C", "T", "rsZ")])
    second = _source("b", "hg38", [("1", 100, "A", "G", "rs9"), ("1", 200, "C", "T", "rsA")])
    third = _source("c", "hg38", [("1", 100, "A", "G", "rs1"), ("2", 50, "G", "A", "")])

    routing = _route(
        [first, second, third], LiftedUnion(), n_workers=n_workers
    )

    assert routing.alids == ["1:100:A:G", "1:200:C:T", "2:50:A:G"]
    assert routing.names.rsid_by_alid == {"1:100:A:G": "rs9", "1:200:C:T": "rsZ"}
    assert not routing.names.pending


def test_names_and_axis_do_not_depend_on_the_worker_count(tmp_path):
    """One worker or several give the same axis, provenance, names and table.

    Every source names one shared site, each by a name that sorts before the
    names of the sources ahead of it, so only the sources' own order -- fixed
    before any worker starts -- can give the first source's name (issue #195).
    """
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    sources = [
        _source(
            f"s{i}",
            "hg19",
            [("2", 1_000 + i, "G", "A", f"rs{i}"), ("3", 500, "C", "T", f"rs{9 - i}")],
        )
        for i in range(5)
    ]

    serial = _route(sources, LiftedUnion(chain), n_workers=1)
    parallel = _route(sources, LiftedUnion(chain), n_workers=3)

    assert serial.alids == parallel.alids
    assert serial.source_alids() == parallel.source_alids()
    assert serial.names.rsid_by_alid == parallel.names.rsid_by_alid
    assert parallel.names.rsid_by_alid["3:500:C:T"] == "rs9"
    associations = [_association(("2", 1_000 + i, "G", "A"), 1.0 + i) for i in range(5)]
    by_serial = serial.table([serial.alids]).route_column(associations).placed[0]
    by_parallel = parallel.table([parallel.alids]).route_column(associations).placed[0]
    for a, b in zip(
        (by_serial.index, by_serial.z), (by_parallel.index, by_parallel.z), strict=True
    ):
        np.testing.assert_array_equal(a, b)


def test_a_tuple_both_assemblies_declare_is_dropped_from_both(tmp_path):
    """The same raw string under hg19 and hg38 is two loci: neither is routed,
    and neither names a variant (issue #85)."""
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    shared = ("3", 1_000, "G", "A", "rsShared")
    hg38 = _source("hg38", "hg38", [shared, ("3", 2_000, "C", "T", "")])
    hg19 = _source("hg19", "hg19", [shared, ("3", 3_000, "C", "T", "")])

    routing = _route([hg38, hg19], LiftedUnion(chain))

    assert routing.alids == ["3:2000:C:T", "3:3000:C:T"]
    assert routing.names.rsid_by_alid == {}


def test_no_spelling_of_a_key_both_assemblies_declare_is_routed(tmp_path):
    """A tuple both assemblies declare makes its routing key two loci, so no
    spelling of the key routes: an hg38 source's `g`/`a` beside the shared
    `G`/`A` would otherwise draw the hg19 locus's associations, which fold to
    the same key, onto the hg38 locus (issue #360)."""
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    shared = ("3", 1_000, "G", "A", "")
    hg38 = _source("hg38", "hg38", [shared, ("3", 1_000, "g", "a", ""), ("3", 2_000, "C", "T", "")])
    hg19 = _source("hg19", "hg19", [shared, ("3", 3_000, "C", "T", "")])

    routing = _route([hg38, hg19], LiftedUnion(chain))

    assert routing.alids == ["3:2000:C:T", "3:3000:C:T"]
    column = routing.table([routing.alids]).route_column([_association(shared[:4], 1.0)])
    assert column.n_unmatched == 1


def test_a_harvest_names_no_variant_from_a_tuple_both_assemblies_declare(tmp_path):
    """The harvest drops the same tuples the lift drops: one string under hg19
    and hg38 is two loci, so neither row names the variant the reference
    routes that string to (issue #255 rounds 4-5)."""
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    hg38 = _source("hg38", "hg38", [("3", 1_000, "G", "A", "rsHG38")])
    hg19 = _source("hg19", "hg19", [("3", 1_000, "G", "A", "rsHG19"), ("3", 7, "C", "T", "rs7")])
    reference = _reference({("3", 1_000, "G", "A"): "3:1000:A:G", ("3", 7, "C", "T"): "3:7:C:T"})

    routing = _route([hg38, hg19], FromReference(reference, chain))

    assert routing.names.finish().rsid_by_alid == {"3:7:C:T": "rs7"}


@pytest.mark.parametrize("reference_routed", [False, True], ids=["union", "reference"])
def test_one_site_spelt_in_two_cases_under_two_assemblies_is_dropped(tmp_path, reference_routed):
    """Two different raw tuples, one per assembly, that one routing key spells
    are one identity declared under both: two loci, so neither routes nor names
    a variant, whatever routes the build -- as for a tuple both declare (issue
    #255 rounds 4-5, issue #360)."""
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    hg38 = _source("hg38", "hg38", [("3", 1_000, "g", "a", "rs1"), ("3", 2_000, "C", "T", "")])
    hg19 = _source("hg19", "hg19", [("3", 1_000, "G", "A", "rs2"), ("3", 3_000, "C", "T", "rs3")])
    authority: LiftedUnion | FromReference = LiftedUnion(chain)
    if reference_routed:
        routes = {("3", 1_000, "g", "a"): "3:1000:A:G", ("3", 3_000, "C", "T"): "3:3000:C:T"}
        authority = FromReference(_reference(routes), chain)

    routing = _route([hg38, hg19], authority)

    names = routing.names.finish() if routing.names.pending else routing.names
    if not reference_routed:
        assert routing.alids == ["3:2000:C:T", "3:3000:C:T"]
    assert names.rsid_by_alid == {"3:3000:C:T": "rs3"}


def test_a_harvested_name_follows_the_reference_routing_not_the_lift(tmp_path):
    """The reference decides where an association lands, so its name lands
    there too, wherever Pass 1's own lift would have put it (issue #255)."""
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    source = _source("a", "hg19", [("3", 1_000, "G", "A", "rsSOURCE")])
    # The identity chain lifts 3:1000 to 3:1000; the reference says 3:5000.
    reference = _reference({("3", 1_000, "G", "A"): "3:5000:A:G"})

    routing = _route([source], FromReference(reference, chain))

    assert routing.names.pending
    assert routing.names.finish().rsid_by_alid == {"3:5000:A:G": "rsSOURCE"}
    assert routing.alids == ["3:5000:A:G"]


def _named_reference() -> VariantReference:
    """An artifact routing 1:100 and 1:200, and naming only 1:100: rsREF."""
    return VariantReference(
        ["1:100:A:G", "1:200:C:T"],
        {("1", 100, "A", "G"): "1:100:A:G", ("1", 200, "C", "T"): "1:200:C:T"},
        {"1:100:A:G": "rsREF"},
        True,
        frozenset(),
    )


def _off_reference_route(key: str) -> str | None:
    """The fold's placement of the one off-reference key these tests' sources declare."""
    return "1:300:A:G" if key == "1:300:G:A" else None


def test_a_named_reference_fills_what_it_leaves_unnamed_and_its_names_win(caplog):
    """The source names all three variants: 1:100 differently from the reference, 1:200
    (a reference row the reference leaves unnamed) and 1:300 (off the reference). The
    reference's name wins where it has one; the harvest fills the rest (#258)."""
    import logging

    source = _source(
        "a", "hg38",
        [("1", 100, "A", "G", "rsSOURCE"), ("1", 200, "C", "T", "rsFILLED"),
         ("1", 300, "G", "A", "rsOFF")],
    )

    with caplog.at_level(logging.WARNING):
        routing = _route([source], FromReference(_named_reference()))
        assert routing.names.pending, "the harvest ran"
        assert routing.names.rsid_by_alid == {"1:100:A:G": "rsREF", "1:200:C:T": "rsFILLED"}
        names = routing.names.finish(_off_reference_route)

    assert names.rsid_by_alid == {
        "1:100:A:G": "rsREF", "1:200:C:T": "rsFILLED", "1:300:A:G": "rsOFF"
    }
    assert (names.conflicts.n_keys, names.conflicts.n_variants) == (0, 0)
    assert "1 variant(s) the Variant Reference names differently" in caplog.text


@pytest.mark.parametrize(
    ("disputed", "expected", "n_variants"),
    [
        ("1:100", {"1:100:A:G": "rsREF", "1:200:C:T": "rsAGREED"}, 0),
        ("1:200", {"1:100:A:G": "rsREF"}, 1),
    ],
    ids=["reference-named", "reference-unnamed"],
)
def test_drop_under_a_named_reference_keeps_its_name_and_unnames_the_rest(
    disputed, expected, n_variants
):
    """Under ``drop`` a variant the sources name two ways is unnamed, unless the
    reference names it: then the reference's name is stored, and no variant is
    counted as left unnamed (#258)."""
    chrom, pos = disputed.split(":")
    other = "1:200" if disputed == "1:100" else "1:100"
    alleles = {"1:100": ("A", "G"), "1:200": ("C", "T")}
    # Two spellings of one Routing Key: one site's names are first-named per source.
    rows_a = [(chrom, int(pos), *alleles[disputed], "rsONE")]
    rows_b = [(chrom, int(pos), *(a.lower() for a in alleles[disputed]), "rsTWO")]
    o_chrom, o_pos = other.split(":")
    rows_a.append((o_chrom, int(o_pos), *alleles[other], "rsAGREED"))
    sources = [_source("a", "hg38", rows_a), _source("b", "hg38", rows_b)]

    routing = _route(
        sources, FromReference(_named_reference()), on_rsid_conflict=RsidConflictPolicy.DROP
    )
    names = routing.names.finish()

    assert names.rsid_by_alid == expected
    assert (names.conflicts.n_keys, names.conflicts.n_variants) == (1, n_variants)


def test_refuse_under_a_named_reference_refuses_a_key_its_sources_dispute():
    """The sources' names are compared by the one rule whichever name is stored: a
    reference that names the variant does not hide its sources' dispute (#258)."""
    sources = [
        _source("a", "hg38", [("1", 100, "A", "G", "rsONE")]),
        _source("b", "hg38", [("1", 100, "a", "g", "rsTWO")]),
    ]
    with pytest.raises(ValueError, match="'1:100:A:G' with different rsids"):
        _route(sources, FromReference(_named_reference()))


def test_finish_gives_a_variant_the_globally_first_name(tmp_path):
    """A name the fold routes, earlier in ``(rank, site)`` order, beats a
    reference-routed one that comes later (issue #255)."""
    first = _source("a", "hg38", [("1", 100, "A", "G", "rsFOLDED")])
    second = _source("b", "hg38", [("1", 100, "G", "A", "rsREFERENCED")])
    reference = _reference({("1", 100, "G", "A"): "1:100:A:G"})

    routing = _route([first, second], FromReference(reference))

    # Before the fold only the reference-routed name is known.
    assert routing.names.rsid_by_alid == {"1:100:A:G": "rsREFERENCED"}
    with pytest.raises(ValueError, match="pending"):
        routing.names.require_written(tmp_path, ["1:100:A:G"])
    folded = routing.names.finish(lambda key: "1:100:A:G" if key == "1:100:A:G" else None)
    assert folded.rsid_by_alid == {"1:100:A:G": "rsFOLDED"}
    with pytest.raises(ValueError, match="already final"):
        folded.finish()


def _disputed_sources() -> list[SourceSpec]:
    """Two spellings of one folded key named two ways, then the other allele
    order naming the same variant again."""
    return [
        _source("a", "hg38", [("1", 100, "A", "G", "rs1")]),
        _source("b", "hg38", [("1", 100, "a", "g", "rs2")]),
        _source("c", "hg38", [("1", 100, "G", "A", "rs1")]),
    ]


def _authority(kind: str) -> LiftedUnion | FromReference:
    """The sources' own union, or a reference naming none of its variants that
    routes every site these tests' sources declare where the union does."""
    if kind == "union":
        return LiftedUnion()
    return FromReference(
        _reference(
            {
                ("1", 100, "A", "G"): "1:100:A:G",
                ("1", 100, "G", "A"): "1:100:A:G",
                ("1", 300, "C", "T"): "1:300:C:T",
            }
        )
    )


def _final(names: RoutedNames) -> RoutedNames:
    """The names once every key is routed: a reference's harvest has nothing
    else to place here."""
    return names.finish() if names.pending else names


#: Both harvests compare the names a site's spellings carry (issue #379).
AUTHORITIES = ["union", "reference"]


@pytest.mark.parametrize("authority", AUTHORITIES)
def test_a_key_named_two_ways_is_refused_naming_both(authority):
    with pytest.raises(ValueError, match="'1:100:A:G' with different rsids: .*'rs1' but .* 'rs2'"):
        _route(_disputed_sources(), _authority(authority))


@pytest.mark.parametrize("authority", AUTHORITIES)
def test_drop_unnames_the_variant_even_through_the_other_allele_order(authority):
    """Under ``drop`` the disputed key's variant stores no name, although the
    other allele order -- a different key -- names it undisputed (issue #306)."""
    routing = _route(
        _disputed_sources(), _authority(authority), on_rsid_conflict=RsidConflictPolicy.DROP
    )
    names = _final(routing.names)

    assert "1:100:A:G" in routing.alids
    assert names.rsid_by_alid == {}
    assert names.conflicts.n_keys == 1
    assert names.conflicts.n_variants == 1
    assert names.conflicts.sample[0].names == ("rs1", "rs2")


@pytest.mark.parametrize("authority", AUTHORITIES)
def test_an_undisputed_harvest_reports_zero(authority):
    """A harvest compared every name and found none disputed: 0 and 0, not
    the null of a comparison never made, whichever authority routes."""
    sources = [
        _source("a", "hg38", [("1", 100, "A", "G", "rs1")]),
        _source("b", "hg38", [("1", 100, "a", "g", "rs1")]),
    ]

    names = _final(_route(sources, _authority(authority)).names)

    assert names.rsid_by_alid == {"1:100:A:G": "rs1"}
    assert (names.conflicts.n_keys, names.conflicts.n_variants) == (0, 0)


@pytest.mark.parametrize("policy", list(RsidConflictPolicy))
@pytest.mark.parametrize("authority", AUTHORITIES)
def test_swapping_the_sources_changes_no_name(authority, policy):
    """A site's two spellings, named two ways, then a variant both name alike.
    In either order the disputed variant is refused or left unnamed, and the
    other keeps its name: no name depends on which source is listed first."""
    upper = [("1", 100, "A", "G", "rsUPPER"), ("1", 300, "C", "T", "rsBOTH")]
    lower = [("1", 100, "a", "g", "rsLOWER"), ("1", 300, "c", "t", "rsBOTH")]
    outcomes = []
    for order in ((upper, lower), (lower, upper)):
        _SOURCES.clear()
        sources = [_source(name, "hg38", rows) for name, rows in zip("ab", order, strict=True)]
        try:
            routing = _route(sources, _authority(authority), on_rsid_conflict=policy)
        except ValueError as refusal:
            assert "'1:100:A:G' with different rsids" in str(refusal)
            outcomes.append(None)
            continue
        outcomes.append(_final(routing.names).rsid_by_alid)

    expected = None if policy is RsidConflictPolicy.REFUSE else {"1:300:C:T": "rsBOTH"}
    assert outcomes == [expected, expected]


def test_a_liftover_collision_leaves_no_source_alid_for_its_alid(tmp_path):
    """Source loci that lift onto one ALID are all dropped (#325), so the ALID has no
    provenance to blank -- unless the sources are one locus, its alleles reported
    either way round."""
    chain = write_collision_chain(tmp_path / "collision.chain")
    source = _source(
        "a",
        "hg19",
        [
            ("1", 501, "A", "G", ""),  # lifts to 1:5501
            ("1", 1_001, "A", "G", ""),  # also lifts to 1:5501: both dropped
            ("1", 1_501, "C", "T", ""),  # lifts to 1:6001
            ("1", 1_501, "T", "C", ""),  # the same locus, reported flipped
        ],
    )

    routing = _route([source], LiftedUnion(chain))

    assert routing.source_alids() == {"1:6001:C:T": "1:1501:C:T"}
    assert [d.source_key for d in routing.lift_drops.sources] == ["1:501:A:G", "1:1001:A:G"]


COLLIDED = "1:5501:A:G"
CLEAN = "1:6001:C:T"
#: (chrom, pos, ref, alt, rsid): hg19 501 and 1001 both lift to 1:5501:A:G, as the
#: forward duplicate at 4:74450339 does; 1501's two spellings are one locus.
FORWARD_COLLISION = [
    ("1", 501, "A", "G", "rs_first"),
    ("1", 1_001, "A", "G", "rs_second"),
    ("1", 1_501, "C", "T", "rs_clean"),
    ("1", 1_501, "T", "C", "rs_clean"),
]


def test_a_forward_duplicate_drops_both_sources_and_records_each(tmp_path):
    chain = write_collision_chain(tmp_path / "forward.chain")
    # Fixture first: each colliding source reaches the ALID on its own.
    for site in (("1", 501, "A", "G"), ("1", 1_001, "A", "G")):
        alone = place_sites({"hg19": [site]}, chain_file=chain, failure_threshold=1.0)
        assert alone.alid == {site: COLLIDED}

    routing = _route([_source("a", "hg19", FORWARD_COLLISION)], LiftedUnion(chain))

    assert routing.alids == [CLEAN], "the collided ALID is on no axis"
    assert routing.source_alids() == {CLEAN: "1:1501:C:T"}
    assert routing.lift_drops.sources == (
        DroppedSource(COLLIDED, "1:501:A:G", "rs_first"),
        DroppedSource(COLLIDED, "1:1001:A:G", "rs_second"),
    )
    assert routing.lift_drops.keys == {"1:501:A:G", "1:1001:A:G"}
    # Neither source's rsid names anything: the ALID is not stored.
    assert routing.names.rsid_by_alid == {CLEAN: "rs_clean"}


def test_a_reverse_strand_block_onto_another_sources_alid_drops_both(tmp_path):
    chain = write_reverse_collision_chain(tmp_path / "reverse.chain")
    forward, reverse = ("1", 501, "A", "G"), ("1", 1_001, "T", "C")
    for site in (forward, reverse):
        alone = place_sites({"hg19": [site]}, chain_file=chain, failure_threshold=1.0)
        assert alone.alid == {site: COLLIDED}
    assert alone.reoriented == {reverse}, "fixture: the second source is the reverse-strand one"

    routing = _route(
        [_source("a", "hg19", [("1", 501, "A", "G", "rs_fwd"), ("1", 1_001, "T", "C", "rs_rev")])],
        LiftedUnion(chain),
    )

    assert routing.alids == []
    assert [tuple(d) for d in routing.lift_drops.sources] == [
        (COLLIDED, "1:501:A:G", "rs_fwd"),
        (COLLIDED, "1:1001:T:C", "rs_rev"),
    ]
    table = routing.table([[COLLIDED]], keep_unmatched=True)
    stream = [
        _association(("1", 501, "A", "G"), 3.0, eaf=0.2),
        _association(("1", 1_001, "T", "C"), -7.0, eaf=0.2),
    ]
    routed = table.route_column(stream)
    assert routed.placed[0].index.tolist() == [], "no row of either source takes the ALID"
    assert sorted(routed.unmatched.keys) == ["1:1001:T:C", "1:501:A:G"]


def test_a_folds_collisions_are_named_as_the_harvest_named_them():
    """A Hybrid fold's spill keeps a site upper-cased (#360), so its drops take the
    harvest's spelling and hg19 rsid for the same Routing Key at the same ALID: a named
    spelling first, else the one there is. A site the harvest's lift never met there is
    refused, not recorded under the fold's spelling."""
    harvest = LiftDrops(
        (
            DroppedSource(COLLIDED, "1:501:a:g", ""),
            DroppedSource(COLLIDED, "1:501:A:G", "rs_first"),
            DroppedSource(COLLIDED, "1:1001:a:g", ""),
        )
    )
    found = harvest.found_by({COLLIDED: (("1", 501, "A", "G"), ("1", 1_001, "A", "G"))})
    assert found.sources == (
        DroppedSource(COLLIDED, "1:501:A:G", "rs_first"),
        DroppedSource(COLLIDED, "1:1001:a:g", ""),
    )
    assert found.spelled == ("1:501:A:G", "1:1001:A:G"), "rows are found by the fold's keys"
    with pytest.raises(ValueError, match="1:2001:A:G .* harvest's own lift did not"):
        harvest.found_by({COLLIDED: (("1", 2_001, "A", "G"),)})
    with pytest.raises(ValueError, match="1:501:A:G as a lift collision at 1:9:A:G"):
        harvest.found_by({"1:9:A:G": (("1", 501, "A", "G"),)})


def test_one_locus_in_both_allele_orders_is_not_a_collision(tmp_path):
    chain = write_collision_chain(tmp_path / "forward.chain")
    source = _source("a", "hg19", [("1", 1_501, "C", "T", "rs_a"), ("1", 1_501, "T", "C", "")])
    routing = _route([source], LiftedUnion(chain))
    assert routing.alids == [CLEAN]
    assert not routing.lift_drops.sources
    assert routing.source_alids() == {CLEAN: "1:1501:C:T"}


def test_the_guard_refuses_two_loci_on_one_alid():
    with pytest.raises(ValueError, match=r"1:9:A:G by 1:1:A:G, 1:2:A:G"):
        require_one_locus_per_alid(
            {("1", 1, "A", "G"): "1:9:A:G", ("1", 2, "A", "G"): "1:9:A:G"}, where="test"
        )
    # One locus under both allele orders, and under two spellings of its chromosome, pass.
    require_one_locus_per_alid(
        {("1", 1, "A", "G"): "1:9:A:G", ("1", 1, "G", "A"): "1:9:A:G"}, where="test"
    )
    require_one_locus_per_alid(
        {("chr1", 1, "A", "G"): "1:9:A:G", ("1", 1, "G", "A"): "1:9:A:G"}, where="test"
    )
    with pytest.raises(ValueError, match="reached by two or more source loci"):
        require_one_locus_per_alid(
            {("chr1", 1, "A", "G"): "1:9:A:G", ("1", 2, "A", "G"): "1:9:A:G"}, where="test"
        )


def test_a_grch38_variant_on_the_collided_alid_keeps_it_with_its_own_name(tmp_path):
    """The rule holds the lifted group to itself: a source already on GRCh38 that names
    the ALID is one variant seen from another build, and neither lifted source may
    give it their statistics or rsid."""
    chain = write_collision_chain(tmp_path / "forward.chain")
    sources = [
        _source("a", "hg19", FORWARD_COLLISION[:2]),
        _source("b", "hg38", [("1", 5_501, "A", "G", "rs_native")]),
    ]

    routing = _route(sources, LiftedUnion(chain))

    assert routing.alids == [COLLIDED]
    assert routing.source_alids() == {COLLIDED: COLLIDED}, "only the GRCh38 source's own origin"
    assert routing.names.rsid_by_alid == {COLLIDED: "rs_native"}
    assert [d.source_key for d in routing.lift_drops.sources] == ["1:501:A:G", "1:1001:A:G"]


def test_a_collided_source_never_routes_by_its_raw_key_onto_a_native_row(tmp_path):
    """hg19 501 and 1001 collide at 1:5501, and GRCh38 declares 1:501:A:G. The dropped
    hg19 501 spells the native row's key, so (as any key both assemblies declare) neither
    owns it: the hg19 row's statistics must not land on the native variant."""
    chain = write_collision_chain(tmp_path / "forward.chain")
    sources = [
        _source("a", "hg19", FORWARD_COLLISION[:2]),
        _source("b", "hg38", [("1", 501, "A", "G", "rs_native")]),
    ]

    routing = _route(sources, LiftedUnion(chain))

    assert routing.alids == []
    placed = routing.table([["1:501:A:G"]], keep_unmatched=True).route_column(
        [_association(("1", 501, "A", "G"), 9.0)]
    )
    assert placed.placed[0].index.tolist() == []
    assert placed.unmatched.keys == ["1:501:A:G"]


def test_a_source_the_reference_routes_under_another_case_is_not_recorded_as_dropped(tmp_path):
    """hg19 501 and 1001 collide at 1:5501. The reference routes 501 under a lower-case
    spelling, which Pass 2 matches case-folded, so only 1001 is a drop of the build."""
    chain = write_collision_chain(tmp_path / "forward.chain")
    reference = _reference({("1", 501, "a", "g"): "1:501:A:G", ("1", 1_501, "C", "T"): CLEAN})
    source = _source("a", "hg19", FORWARD_COLLISION[:2])

    routing = _route([source], FromReference(reference, chain))

    assert [d.source_key for d in routing.lift_drops.sources] == ["1:1001:A:G"]
    assert routing.lift_drops.keys == {"1:1001:A:G"}


def _identity_reference(alids: list[str], rsid_by_alid: dict[str, str]) -> VariantReference:
    """A reference with no source keys, as a Store Variant Table reads (#260)."""
    lookup = {(c, int(p), a, b): alid for alid in alids for c, p, a, b in [alid.split(":")]}
    lookup |= {(c, int(p), b, a): alid for alid in alids for c, p, a, b in [alid.split(":")]}
    return VariantReference(alids, lookup, rsid_by_alid, False, frozenset())


def test_a_reference_with_no_source_keys_lifts_an_hg19_source_and_fills_its_names(tmp_path):
    """hg19 2:1000 G>A lifts onto the reverse strand to 2:999001:C:T. A Store Variant
    Table that names its rsids and also lists 2:1000:A:G routes the row by the lift, not
    by its own coordinate, re-oriented (#260). Its name wins over the source's; the row
    it leaves unnamed, 3:1000:A:G, takes its source's name (#258)."""
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    source = _source(
        "a", "hg19", [("2", 1_000, "G", "A", "rsSOURCE"), ("3", 1_000, "G", "A", "rsTHREE")]
    )
    reference = _identity_reference(
        ["2:1000:A:G", "2:999001:C:T", "3:1000:A:G"], {"2:999001:C:T": "rsREF"}
    )
    assert reference.source_lookup[("2", 1_000, "G", "A")] == "2:1000:A:G", "fixture: the trap"

    routing = _route([source], FromReference(reference, chain))

    assert routing.placed_every_site
    assert routing.alids == ["2:1000:A:G", "2:999001:C:T", "3:1000:A:G"]
    assert routing.names.rsid_by_alid == {"2:999001:C:T": "rsREF", "3:1000:A:G": "rsTHREE"}
    assert routing.names.conflicts.n_keys == 0, "the sources' names were compared"
    column = routing.table([routing.alids]).route_column(
        [_association(("2", 1_000, "G", "A"), 1.0)]
    )
    assert column.placed[0].index.tolist() == [1]
    assert column.placed[0].z.tolist() == [-1.0], "re-oriented by the reverse strand"


def test_a_reference_with_no_source_keys_routes_grch38_sources_by_identity_and_fills():
    """With no hg19 source nothing is lifted: the reference routes by identity, and the
    harvest fills the row it leaves unnamed (#258)."""
    reference = _identity_reference(["1:100:A:G", "1:200:C:T"], {"1:100:A:G": "rsREF"})
    source = _source(
        "a", "hg38", [("1", 100, "A", "G", "rsSOURCE"), ("1", 200, "T", "C", "rsFILLED")]
    )

    routing = _route([source], FromReference(reference))

    assert not routing.placed_every_site
    assert routing.names.finish().rsid_by_alid == {
        "1:100:A:G": "rsREF", "1:200:C:T": "rsFILLED"
    }


def test_a_reference_with_no_source_keys_refuses_a_source_of_another_assembly():
    """Neither identity nor the hg19 lift places a source declared hg18 (#260)."""
    reference = _identity_reference(["1:100:A:G"], {})
    source = _source("a", "hg18", [("1", 100, "A", "G", "")])
    with pytest.raises(ValueError, match="'hg18'"):
        _route([source], FromReference(reference))


def test_an_artifact_routing_two_lifted_loci_to_one_alid_routes_neither(tmp_path):
    """An artifact frozen before #325 routes hg19 501 and 1001 to 1:5501:A:G; a GRCh38
    source's own key at an ALID is one variant from two builds, and is kept (#85). The
    harvest's lift places neither hg19 site under the default chain (the threshold
    admits that), so the artifact alone says what collided, and the chain only that
    1:6001 is not an hg19 coordinate it leaves in place."""
    lookup = {
        ("1", 501, "A", "G"): "1:5501:A:G",
        ("1", 1_001, "A", "G"): "1:5501:A:G",
        ("1", 6_001, "C", "T"): CLEAN,
        ("1", 1_501, "C", "T"): CLEAN,
    }
    reference = replace(_reference(lookup), rsid_by_alid={CLEAN: "rsC"})
    hg19 = _source("b", "hg19", [("1", 501, "A", "G", ""), ("1", 1_001, "A", "G", "")])
    hg38 = _source("a", "hg38", [("1", 6_001, "C", "T", "")])

    routing = _route([hg38, hg19], FromReference(reference, failure_threshold=1.0))

    assert routing.alids == ["1:5501:A:G", CLEAN], "the reference keeps its ALIDs"
    assert sorted(routing.lift_drops.keys) == ["1:1001:A:G", "1:501:A:G"]
    assert {d.alid for d in routing.lift_drops.sources} == {"1:5501:A:G"}
    assert routing.source_alids() == {CLEAN: None}


@pytest.mark.parametrize(
    ("write_chain", "dropped"),
    [
        pytest.param(write_identity_collision_chain, ["1:501:A:G", "1:5501:A:G"], id="kept"),
        pytest.param(write_collision_chain, [], id="moved"),
    ],
)
def test_an_artifact_key_at_its_own_locus_is_lifted_where_the_chain_keeps_it(
    tmp_path, write_chain, dropped
):
    """A mixed manifest's artifact routes hg19 501 (lifted to 5501) and 1:5501:A:G to
    1:5501:A:G. The artifact cannot say which assembly spelled the second key. Where the
    chain leaves hg19 5501 at 5501, it is a second lifted locus and both are dropped;
    where the chain places hg19 5501 elsewhere, it is a GRCh38 key and both route."""
    lookup = {
        ("1", 501, "A", "G"): "1:5501:A:G",
        ("1", 5_501, "A", "G"): "1:5501:A:G",
        ("1", 9_001, "A", "C"): "1:9001:A:C",
    }
    reference = replace(_reference(lookup), rsid_by_alid={"1:9001:A:C": "rs9001"})
    hg19 = _source("b", "hg19", [("1", 501, "A", "G", ""), ("1", 5_501, "A", "G", "")])
    hg38 = _source("a", "hg38", [("1", 9_001, "A", "C", "")])

    # The harvest's lift may fail hg19 5501, which only the reference routes.
    authority = FromReference(reference, write_chain(tmp_path / "c"), failure_threshold=1.0)
    routing = _route([hg38, hg19], authority)

    assert sorted(routing.lift_drops.keys) == dropped
    routed = {"1:9001:A:C"} | (set() if dropped else {"1:5501:A:G"})
    assert set(routing.source_alids()) == routed


# ── RouteTable ───────────────────────────────────────────────────────────────


def _hg38_routing(sites: list[Site]):
    return _route(
        [_source("a", "hg38", [(*site, "") for site in sites])],
        LiftedUnion(),
    )


def test_duplicates_for_an_index_collapse_or_drop_across_batches(monkeypatch):
    """Rows reaching one index that agree on z and SE collapse to the first, and
    rows that disagree are all dropped and named, even when a batch boundary
    falls between them."""
    monkeypatch.setattr(route_table, "_RESOLVE_BATCH", 2)
    routing = _hg38_routing([("1", 100, "A", "G"), ("1", 300, "C", "T"), ("1", 100, "G", "A")])
    assert routing.alids == ["1:100:A:G", "1:300:C:T"]
    table = routing.table([routing.alids])
    stream = [
        _association(("1", 100, "A", "G"), 1.0),
        _association(("1", 300, "C", "T"), 2.0),
        _association(("1", 300, "C", "T"), 2.0),  # identical: agrees
        _association(("1", 100, "G", "A"), 5.0),  # the other allele order, later
    ]

    placed = table.route_column(stream).placed[0]

    assert placed.index.tolist() == [1]
    assert placed.z.tolist() == [2.0]
    assert placed.conflicting.tolist() == [0]
    assert placed.dropped.tolist() == [0, 0]
    assert placed.collapsed == 1


def _read_place_by_place(place, z, se):
    """D5's rule read one place at a time: keep, conflicting, dropped, collapsed."""
    rows_at: dict[int, list[int]] = {}
    for row, at in enumerate(place.tolist()):
        rows_at.setdefault(at, []).append(row)

    def agrees(row: int, first: int) -> bool:
        return all(
            a == b or (np.isnan(a) and np.isnan(b))
            for a, b in ((z[row], z[first]), (se[row], se[first]))
        )

    keep, conflicting, dropped, collapsed = [], [], [], 0
    for at, rows in rows_at.items():
        if all(agrees(r, rows[0]) for r in rows):
            keep.append(rows[0])
            collapsed += len(rows) - 1
        else:
            conflicting.append(at)
            dropped += rows
    return sorted(keep), sorted(conflicting), sorted(dropped), collapsed


def test_collapse_duplicates_reads_each_place_as_the_rule_says():
    """`collapse_duplicates` touches only the rows that share a place; on random
    columns of repeated places, NaNs, agreeing and disagreeing rows, it answers
    as a plain reading of each place does."""
    rng = np.random.default_rng(357)
    conflicts_seen = 0
    for _ in range(300):
        n = int(rng.integers(0, 40))
        place = rng.integers(0, 12, n)
        z = rng.choice([1.0, 2.0, np.nan], n)
        se = rng.choice([0.1, 0.2, np.nan], n)

        found = route_table.collapse_duplicates(place, z, se)

        assert (
            found.keep.tolist(), found.conflicting.tolist(), found.dropped.tolist(),
            found.collapsed,
        ) == _read_place_by_place(place, z, se)
        conflicts_seen += len(found.conflicting)
    assert conflicts_seen > 0


def _indel_association(site: Site, z: float) -> ReaderAssociation:
    chrom, pos, ref, alt = site
    return ReaderAssociation(
        chrom, pos, ref, alt, z, 0.1, StoredEffectScale.SD
    )


#: BMI_IRN (FinnGen r13) at 1:827587: an insertion and a deletion of one repeat.
#: A sorted ALID spelt both `1:827587:G:GGGGCC`; written reference first they
#: are two variants (ADR 0079).
INSERTION: Site = ("1", 827587, "G", "GGGGCC")
DELETION: Site = ("1", 827587, "GGGGCC", "G")


@pytest.mark.parametrize("order", ["insertion first", "deletion first"])
def test_an_insertion_and_a_deletion_at_one_position_reach_two_places(order):
    """Each keeps its own row, wherever it falls in the stream: neither is a
    duplicate of the other, so no row is dropped."""
    routing = _hg38_routing([INSERTION, DELETION])
    assert routing.alids == ["1:827587:G:GGGGCC", "1:827587:GGGGCC:G"]
    stream = [_indel_association(INSERTION, 1.5), _indel_association(DELETION, -0.5)]
    if order == "deletion first":
        stream.reverse()

    placed = routing.table([routing.alids]).route_column(stream).placed[0]

    z_by_alid = dict(
        zip([routing.alids[i] for i in placed.index.tolist()], placed.z.tolist(), strict=True)
    )
    assert z_by_alid == {"1:827587:G:GGGGCC": 1.5, "1:827587:GGGGCC:G": -0.5}
    assert placed.conflicting.tolist() == []
    assert (placed.dropped.tolist(), placed.collapsed) == ([], 0)


def test_an_insertion_and_a_deletion_resolve_their_duplicates_apart():
    """Each indel's rows are judged at its own place: the insertion's copies
    collapse to one association, and the deletion's two distinct rows are both
    dropped, without either place's rows deciding the other's."""
    routing = _hg38_routing([INSERTION, DELETION])
    stream = [
        _indel_association(INSERTION, 1.5),
        _indel_association(DELETION, -0.5),
        _indel_association(INSERTION, 1.5),
        _indel_association(DELETION, -0.7),
    ]

    placed = routing.table([routing.alids]).route_column(stream).placed[0]

    assert [routing.alids[i] for i in placed.index.tolist()] == ["1:827587:G:GGGGCC"]
    assert placed.z.tolist() == [1.5]
    assert placed.collapsed == 1
    assert [routing.alids[i] for i in placed.conflicting.tolist()] == ["1:827587:GGGGCC:G"]
    assert placed.dropped.tolist() == [1, 1]


def test_no_reader_declares_its_alt_the_reference():
    """A reader declares its `ref` the reference, or nothing (#316): #357's
    reversed-indel rule, the one reader of an `alt` declaration, is gone."""
    assert [declared.value for declared in ReferenceAllele] == ["undeclared", "ref"]


def test_snvs_spelling_the_reference_both_ways_still_drop_both_rows():
    """Two SNV rows at one ALID are one variant seen from opposite alleles, so
    a declared reference tells them nothing: both are dropped."""
    routing = _hg38_routing([("1", 100, "A", "G"), ("1", 100, "G", "A")])
    table = routing.table([routing.alids])
    stream = [
        _indel_association(("1", 100, "A", "G"), 1.0),
        _indel_association(("1", 100, "G", "A"), 3.0),
    ]

    placed = table.route_column(stream).placed[0]

    assert placed.index.tolist() == []
    assert placed.dropped.tolist() == [0, 0]


def test_duplicates_that_differ_only_in_frequency_are_one_association():
    """The frequency is annotation, not the finding (ADR 0036): rows agreeing on
    z and SE collapse to the first, frequency and all."""
    routing = _hg38_routing([("1", 100, "A", "G")])
    table = routing.table([routing.alids])
    stream = [
        _association(("1", 100, "A", "G"), 1.0, eaf=0.2),
        _association(("1", 100, "A", "G"), 1.0, eaf=0.3),
    ]

    placed = table.route_column(stream).placed[0]

    assert placed.index.tolist() == [0]
    assert placed.eaf.tolist() == pytest.approx([0.2])
    assert placed.collapsed == 1
    assert placed.conflicting.tolist() == []


def test_a_reverse_strand_lift_reorients_statistics_on_every_axis(tmp_path):
    """z is negated and EAF complemented for a re-oriented key, on the panel
    axis and the shared axis alike; a forward key is untouched (issue #284)."""
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    sites = [("2", 1_000, "G", "A"), ("2", 2_000, "T", "A"), ("3", 1_000, "G", "A")]
    routing = _route(
        [_source("a", "hg19", [(*site, "") for site in sites])],
        LiftedUnion(chain),
    )
    panel = ["2:999001:C:T"]
    table = routing.table([panel, routing.alids])

    column = table.route_column([_association(site, 2.0, eaf=0.2) for site in sites])
    on_panel, shared = column.placed

    assert on_panel.index.tolist() == [0]
    assert on_panel.z.tolist() == pytest.approx([-2.0])
    assert on_panel.eaf.tolist() == pytest.approx([0.8])
    shared_alids = [routing.alids[i] for i in shared.index.tolist()]
    assert shared_alids == ["2:998001:A:T", "3:1000:A:G"]
    assert shared.z.tolist() == pytest.approx([-2.0, 2.0])
    assert shared.eaf.tolist() == pytest.approx([0.8, 0.2])


def test_a_reverse_strand_lift_orients_a_mixed_case_site_on_its_upper_cased_alleles(tmp_path):
    """`g`/`A` complements to `c`/`T` on chr2's reverse block, so the variant is
    `C:T`; the source's A1, `A` (compared upper-cased), projects onto `T`, its
    A2, so z is negated and EAF complemented (issues #284, #360)."""
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    site = ("2", 1_000, "g", "A")
    routing = _route([_source("a", "hg19", [(*site, "")])], LiftedUnion(chain))
    table = routing.table([routing.alids])

    placed = table.route_column([_association(site, 2.0, eaf=0.2)]).placed[0]

    assert routing.alids == ["2:999001:C:T"]
    assert placed.z.tolist() == pytest.approx([-2.0])
    assert placed.eaf.tolist() == pytest.approx([0.8])


def test_a_reverse_strand_lift_signs_statistics_by_the_source_readers_rule(
    tmp_path, monkeypatch
):
    """`readers.orientation` is the only code that signs a statistic by an
    allele swap (ADR 0076), and a lift's re-orientation is one: routing hands
    each batch's swapped rows to its `swap`."""
    from opengwasdb.readers import orientation

    real, judged = orientation.swap, []

    def spy(flipped, z, eaf):
        judged.append(np.asarray(flipped).tolist())
        return real(flipped, z, eaf)

    monkeypatch.setattr(orientation, "swap", spy)
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    sites = [("2", 1_000, "G", "A"), ("3", 1_000, "G", "A")]
    routing = _route([_source("a", "hg19", [(*site, "") for site in sites])], LiftedUnion(chain))
    table = routing.table([routing.alids])

    placed = table.route_column([_association(site, 2.0, eaf=0.2) for site in sites]).placed[0]

    # Fixture first: the reverse block re-orients the first site, not the second.
    assert placed.z.tolist() == pytest.approx([-2.0, 2.0])
    assert placed.eaf.tolist() == pytest.approx([0.8, 0.2])
    assert judged == [[True, False]]


def test_two_spellings_reaching_one_variant_oriented_two_ways_are_refused():
    """Both spellings of a site share one routing key; were only one of them
    re-oriented, a lookup would sign an association by whichever it found
    first, so the table refuses."""
    sites = {("1", 100, "a", "g"): "1:100:A:G", ("1", 100, "A", "G"): "1:100:A:G"}

    with pytest.raises(ValueError, match="conflicting variants after case folding"):
        RouteTable.build(sites, {("1", 100, "a", "g")}, [["1:100:A:G"]], keep_unmatched=False)


def test_one_site_spelt_in_two_cases_is_one_routing_key():
    """`a`/`g` and `A`/`G` are one site (issue #360): the union names one ALID,
    and either spelling of an association reaches it. Two that disagree are
    resolved as any two associations on one index are (#357, D5): neither is
    kept, and the index is reported conflicting."""
    routing = _route(
        [_source("a", "hg38", [("1", 100, "a", "g", ""), ("1", 100, "A", "G", "")])],
        LiftedUnion(),
    )
    assert routing.alids == ["1:100:A:G"]
    table = routing.table([routing.alids])

    placed = table.route_column(
        [_association(("1", 100, "a", "g"), 1.0), _association(("1", 100, "A", "G"), 2.0)]
    ).placed[0]

    assert placed.index.tolist() == []
    assert placed.conflicting.tolist() == [0]
    assert placed.dropped.tolist() == [0, 0]  # both rows, each at index 0


def test_a_reference_sending_one_routing_key_two_ways_is_refused():
    """Two spellings of one site are one key, so a reference routing them to two
    variants is ambiguous, and the table refuses it rather than let insertion
    order choose."""
    reference = VariantReference(
        ["1:100:A:G", "1:200:A:G"],
        {("1", 100, "a", "g"): "1:100:A:G", ("1", 100, "A", "G"): "1:200:A:G"},
        {"1:100:A:G": "rs1"},
        explicit_source_keys=True,
        reoriented=frozenset(),
    )
    routing = _route([_source("a", "hg38", [("1", 100, "A", "G", "")])], FromReference(reference))

    with pytest.raises(ValueError, match="conflicting variants after case folding"):
        routing.table([routing.alids])


def test_admission_judges_the_reader_orientation_before_matching(tmp_path):
    """``admit`` sees the source's own statistics, before a lift re-orients
    them, and a row it refuses is neither placed nor counted unmatched."""
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    reverse = ("2", 1_000, "G", "A")
    routing = _route(
        [_source("a", "hg19", [(*reverse, ""), ("3", 1_000, "G", "A", "")])],
        LiftedUnion(chain),
    )
    table = routing.table([routing.alids])
    seen: list[list[float]] = []

    def admit(batch: AssociationBatch) -> np.ndarray:
        seen.append(batch.z.tolist())
        return batch.z > 0

    stream = [
        _association(reverse, 2.0),
        _association(("3", 1_000, "G", "A"), -1.0),  # refused
        _association(("9", 9, "A", "C"), 3.0),  # admitted, then unmatched
    ]
    column = table.route_column(stream, admit=admit)

    assert seen == [[2.0, -1.0, 3.0]]
    assert column.placed[0].z.tolist() == pytest.approx([-2.0])
    assert column.n_unmatched == 1


def test_an_analysis_without_a_declared_score_hands_admit_no_scores():
    """``imputation_scores=False`` says no score was declared: the batch carries
    every row's score as undeclared, whatever the associations hold, and the
    default hands ``admit`` the scores the reader parsed."""
    from opengwasdb.readers.interface import ImputationScore, ImputationScoreStatus

    routing = _hg38_routing([("1", 100, "A", "G")])
    table = routing.table([routing.alids])
    scored = ReaderAssociation(
        "1", 100, "A", "G", 1.0, 0.1, StoredEffectScale.SD,
        imputation_score=ImputationScore(0.9, ImputationScoreStatus.USABLE),
    )
    seen: dict[bool, tuple[list[float], list[ImputationScoreStatus]]] = {}
    for collect in (True, False):

        def admit(batch: AssociationBatch, collect: bool = collect) -> np.ndarray:
            seen[collect] = (batch.imputation_score.tolist(), batch.imputation_status.tolist())
            return np.ones(len(batch), dtype=bool)

        column = table.route_column([scored], admit=admit, imputation_scores=collect)
        assert column.placed[0].z.tolist() == [1.0]

    assert seen[True] == ([0.9], [ImputationScoreStatus.USABLE])
    scores, statuses = seen[False]
    assert np.isnan(scores).all() and statuses == [ImputationScoreStatus.UNDECLARED]


def test_unmatched_associations_are_kept_by_key_only_when_asked():
    routing = _route(
        [_source("a", "hg38", [("1", 100, "A", "G", "")])],
        LiftedUnion(),
    )
    alids = routing.alids
    stream = [
        _association(("1", 200, "c", "t"), 1.0),
        _association(("1", 100, "A", "G"), 2.0),
        _association(("1", 200, "C", "T"), 3.0),
    ]
    kept = routing.table([alids], keep_unmatched=True).route_column(stream)

    assert kept.unmatched.keys == ["1:200:C:T", "1:200:C:T"]
    assert kept.unmatched.z.tolist() == [1.0, 3.0]
    assert kept.n_unmatched == 2


def test_unmatched_duplicates_that_disagree_are_named_and_all_rows_kept():
    """Two off-reference associations for one key with different statistics
    are a conflict, reported by key as a placed index's would be; rows that
    agree are not. Every row still comes back, in stream order, for the
    caller's own last-wins (the reviewer's probe)."""
    table = RouteTable.build({}, (), [[]], keep_unmatched=True)
    stream = [
        _association(("1", 200, "A", "G"), 1.0),
        _association(("1", 300, "C", "T"), 4.0),
        _association(("1", 200, "A", "G"), 2.0),
        _association(("1", 300, "C", "T"), 4.0),
    ]

    column = table.route_column(stream)

    assert column.unmatched.conflicting == ["1:200:A:G"]
    assert column.unmatched.keys == ["1:200:A:G", "1:300:C:T", "1:200:A:G", "1:300:C:T"]
    assert column.unmatched.z.tolist() == [1.0, 4.0, 2.0, 4.0]
    assert column.n_unmatched == 4


def test_long_alleles_are_routed_by_their_whole_key():
    """A 300-byte allele is matched by its whole key, beside short ones, and a
    key the table does not hold is never snapped to a neighbour."""
    long_alt = "A" * 300
    sites = [("1", 10, "A", "G"), ("1", 10, "A", long_alt), ("2", 5, "C", "T")]
    routing = _hg38_routing(sites)
    table = routing.table([routing.alids])
    stream = [_association(site, float(i)) for i, site in enumerate(sites)]
    stream.append(_association(("1", 10, "A", "A" * 299), 9.0))

    column = table.route_column(stream)

    placed = column.placed[0]
    routed = dict(zip([routing.alids[i] for i in placed.index], placed.z.tolist(), strict=True))
    assert routed == {"1:10:A:G": 0.0, f"1:10:A:{long_alt}": 1.0, "2:5:C:T": 2.0}
    assert column.n_unmatched == 1
    with pytest.raises(ValueError, match="consumed"):
        routing.table([routing.alids])


@pytest.mark.parametrize(
    "mask",
    [np.full(1, np.nan), np.ones(1), np.ones(1, dtype=np.int8), [True]],
    ids=["nan", "float", "int", "list"],
)
def test_an_admit_decision_that_is_not_a_boolean_array_is_refused(mask):
    """A decision is kept only when it says so: a NaN, a number or a list is
    not a boolean mask, and coercing it would keep a row nobody admitted."""
    routing = _hg38_routing([("1", 100, "A", "G")])
    table = routing.table([routing.alids])
    with pytest.raises(TypeError, match="boolean array"):
        table.route_column([_association(("1", 100, "A", "G"), 1.0)], admit=lambda batch: mask)


def test_an_admit_mask_of_the_wrong_length_is_refused():
    routing = _hg38_routing([("1", 100, "A", "G")])
    table = routing.table([routing.alids])
    with pytest.raises(ValueError, match="mask of shape"):
        table.route_column(
            [_association(("1", 100, "A", "G"), 1.0)], admit=lambda batch: np.ones(2, bool)
        )


def test_a_lift_failing_beyond_its_threshold_is_refused(tmp_path: Path):
    """The lift's failure threshold is the authority's: a union whose hg19
    group fails the lift beyond it is refused."""
    from opengwasdb.build.liftover import LiftoverFailureError

    chain = write_two_strand_chain(tmp_path / "strands.chain")
    off_chain = _source("a", "hg19", [("5", 100, "A", "G", ""), ("3", 100, "A", "G", "")])
    with pytest.raises(LiftoverFailureError):
        _route([off_chain], LiftedUnion(chain, failure_threshold=0.1))


# ── place_sites ──────────────────────────────────────────────────────────────


def test_place_sites_names_hg38_sites_and_lifts_hg19_ones(tmp_path):
    """An hg38 site is its own ALID; an hg19 site is lifted, a reverse-strand
    one re-oriented, and one the chain does not cover left out."""
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    placement = place_sites(
        {
            "hg38": [("5", 10, "G", "A")],
            "hg19": [("2", 1_000, "G", "A"), ("3", 1_000, "G", "A"), ("9", 5, "A", "G")],
        },
        chain_file=chain,
        failure_threshold=0.5,
    )

    assert placement.alid == {
        ("5", 10, "G", "A"): "5:10:A:G",
        ("2", 1_000, "G", "A"): "2:999001:C:T",
        ("3", 1_000, "G", "A"): "3:1000:A:G",
    }
    assert placement.reoriented == {("2", 1_000, "G", "A")}


def test_place_sites_judges_the_lift_on_the_hg19_group_alone(tmp_path):
    """Two hg19 sites, one failing, are 50% failed however many hg38 sites
    ride along; an assembly the lift does not know is refused."""
    from opengwasdb.build.liftover import LiftoverFailureError

    chain = write_two_strand_chain(tmp_path / "strands.chain")
    hg38 = [("5", position, "G", "A") for position in range(1, 101)]
    with pytest.raises(LiftoverFailureError):
        place_sites(
            {"hg38": hg38, "hg19": [("3", 1_000, "G", "A"), ("9", 5, "A", "G")]},
            chain_file=chain,
            failure_threshold=0.1,
        )
    with pytest.raises(ValueError, match="unhandled source_assembly"):
        place_sites({"hg18": [("3", 1_000, "G", "A")]})


def test_place_sites_refuses_a_tuple_given_under_both_assemblies(tmp_path):
    """One string under hg19 and hg38 is two loci with two places; keeping
    either would lose the other in silence, so it is refused, named."""
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    shared = ("3", 1_000, "G", "A")
    with pytest.raises(ValueError, match=r"both hg38 and hg19.*'3', 1000, 'G', 'A'"):
        place_sites({"hg38": [shared], "hg19": [shared, ("3", 2_000, "C", "T")]}, chain_file=chain)


# ── extract_variant_reference ────────────────────────────────────────────────


@pytest.mark.parametrize("n_workers", [1, 2])
def test_an_extracted_reference_routes_as_the_union_it_froze(tmp_path, n_workers):
    """A Variant Reference is a routing, frozen: read back, it names the same
    ALIDs and rsids, and places every source key where the lifted union does."""
    chain = write_two_strand_chain(tmp_path / "strands.chain")
    sources = [
        _source("a", "hg19", [("3", 1_000, "G", "A", "rs1"), ("3", 2_000, "C", "T", "")]),
        _source("b", "hg38", [("5", 10, "G", "A", "rs5"), ("5", 10, "A", "G", "rs9")]),
    ]
    artifact = tmp_path / f"union-{n_workers}.variant-ref.tsv.gz"

    result = extract_variant_reference(
        sources, artifact, chain_file=chain, n_workers=n_workers, readers=_fake_reader
    )
    union = _route(sources, LiftedUnion(chain))
    frozen = _route(
        sources, FromReference(read_variant_reference(artifact), chain)
    )

    assert result.n_variants == len(union.alids) == 3
    assert frozen.alids == union.alids
    # Both allele orders of 5:10 are one site each; in source order the
    # A:G site sorts first, so its name is the first named.
    assert frozen.names.finish().rsid_by_alid == union.names.rsid_by_alid == {
        "3:1000:A:G": "rs1",
        "5:10:A:G": "rs9",
    }
    sites = [
        ("3", 1_000, "G", "A"),
        ("3", 2_000, "C", "T"),
        ("5", 10, "G", "A"),
        ("5", 10, "A", "G"),
    ]
    # The two spellings of 5:10 reach one ALID; equal statistics make them one
    # association, so where it lands is observable.
    stream = [_association(site, z) for site, z in zip(sites, (0.0, 1.0, 3.0, 3.0), strict=True)]

    def routed(routing) -> dict[str, float]:
        placed = routing.table([routing.alids]).route_column(stream).placed[0]
        return {
            routing.alids[i]: z
            for i, z in zip(placed.index.tolist(), placed.z.tolist(), strict=True)
        }

    expected = {"3:1000:A:G": 0.0, "3:2000:C:T": 1.0, "5:10:A:G": 3.0}
    assert routed(union) == routed(frozen) == expected
