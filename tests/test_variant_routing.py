"""Variant Routing at its interface: `route_variants`, `RoutedNames`, `RouteTable`.

Every source is a `FakeReader`, opened through `route_variants`' reader seam,
and every lift goes through a chain whose strands are known by construction
(`liftover_fixtures`), so no Store is built and no fixture VCF is parsed.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from liftover_fixtures import write_collision_chain, write_two_strand_chain

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.fake import FakeReader
from opengwasdb.readers.interface import ReaderAssociation, ReferenceAllele, SourceVariant
from opengwasdb.readers.registry import SourceSpec
from opengwasdb.variants.reference import VariantReference, read_variant_reference
from opengwasdb.variants.routing import (
    AssociationBatch,
    FromReference,
    LiftedUnion,
    RouteTable,
    extract_variant_reference,
    place_sites,
    route_variants,
)
from opengwasdb.variants.routing import table as route_table
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
    return ReaderAssociation(chrom, pos, ref, alt, z, 0.1, StoredEffectScale.SD, eaf=eaf)


def _route(sources, authority, **kwargs):
    return route_variants(sources, authority, readers=_fake_reader, **kwargs)


def _reference(source_lookup: dict[Site, str]) -> VariantReference:
    """A reference that names no rsids, routing exactly ``source_lookup``."""
    alids = sorted(set(source_lookup.values()))
    return VariantReference(alids, source_lookup, {}, explicit_source_keys=True)


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


def test_a_named_reference_is_used_as_given_and_nothing_is_read():
    reference = VariantReference(
        ["1:100:A:G"], {("1", 100, "A", "G"): "1:100:A:G"}, {"1:100:A:G": "rsREF"}, True
    )

    def _no_read(_spec: SourceSpec) -> FakeReader:
        raise AssertionError("a reference that names its variants read a source")

    spec = SourceSpec("a", "a", "fake", "hg38", StoredEffectScale.SD)
    routing = route_variants(
        [spec], FromReference(reference), readers=_no_read
    )

    assert routing.names.reference_named
    assert routing.names.rsid_by_alid == {"1:100:A:G": "rsREF"}
    assert routing.names.conflicts.n_keys is None


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


def test_a_key_named_two_ways_is_refused_naming_both():
    reference = _reference({("1", 100, "A", "G"): "1:100:A:G", ("1", 100, "G", "A"): "1:100:A:G"})
    with pytest.raises(ValueError, match="'rs1' but .* names 'rs2'"):
        _route(_disputed_sources(), FromReference(reference))


def test_drop_unnames_the_variant_even_through_the_other_allele_order():
    """Under ``drop`` the disputed key's variant stores no name, although the
    other allele order -- a different key -- names it undisputed (issue #306)."""
    reference = _reference({("1", 100, "A", "G"): "1:100:A:G", ("1", 100, "G", "A"): "1:100:A:G"})

    routing = _route(
        _disputed_sources(),
        FromReference(reference),
        on_rsid_conflict=RsidConflictPolicy.DROP,
    )
    names = routing.names.finish()

    assert names.rsid_by_alid == {}
    assert names.conflicts.n_keys == 1
    assert names.conflicts.n_variants == 1
    assert names.conflicts.sample[0].names == ("rs1", "rs2")


def test_source_alids_blank_an_ambiguous_liftover_collision(tmp_path):
    """A stored variant several source loci lift onto has an ambiguous origin,
    left blank rather than guessed -- unless the sources are one locus, its
    alleles reported either way round (issue #85)."""
    chain = write_collision_chain(tmp_path / "collision.chain")
    source = _source(
        "a",
        "hg19",
        [
            ("1", 501, "A", "G", ""),  # lifts to 1:5501
            ("1", 1_001, "A", "G", ""),  # also lifts to 1:5501: ambiguous
            ("1", 1_501, "C", "T", ""),  # lifts to 1:6001
            ("1", 1_501, "T", "C", ""),  # the same locus, reported flipped
        ],
    )

    routing = _route([source], LiftedUnion(chain))

    assert routing.source_alids() == {"1:5501:A:G": None, "1:6001:C:T": "1:1501:C:T"}


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


def _read_place_by_place(place, z, se, reference=None):
    """D5's rule read one place at a time: keep, conflicting, dropped, collapsed,
    and the rows a reversed indel's kept row displaced."""
    rows_at: dict[int, list[int]] = {}
    for row, at in enumerate(place.tolist()):
        rows_at.setdefault(at, []).append(row)

    def agrees(row: int, first: int) -> bool:
        return all(
            a == b or (np.isnan(a) and np.isnan(b))
            for a, b in ((z[row], z[first]), (se[row], se[first]))
        )

    def reference_first(rows: list[int]) -> list[int] | None:
        """A reversed indel's rows spelling their reference first, or None."""
        if reference is None:
            return None
        codes = [int(reference[r]) for r in rows]
        firsts = [r for r, code in zip(rows, codes, strict=True) if code == 1]
        if 0 in codes or 2 not in codes or not firsts:
            return None
        return firsts if all(agrees(r, firsts[0]) for r in firsts) else None

    keep, conflicting, dropped, reversed_indels, collapsed = [], [], [], [], 0
    for at, rows in rows_at.items():
        if all(agrees(r, rows[0]) for r in rows):
            keep.append(rows[0])
            collapsed += len(rows) - 1
        elif (firsts := reference_first(rows)) is not None:
            keep.append(firsts[0])
            collapsed += len(firsts) - 1
            reversed_indels += [r for r in rows if r not in firsts]
        else:
            conflicting.append(at)
            dropped += rows
    return sorted(keep), sorted(conflicting), sorted(dropped), collapsed, sorted(reversed_indels)


@pytest.mark.parametrize("declared", [False, True], ids=["undeclared", "declared"])
def test_collapse_duplicates_reads_each_place_as_the_rule_says(declared):
    """`collapse_duplicates` touches only the rows that share a place; on random
    columns of repeated places, NaNs, agreeing and disagreeing rows, and rows
    whose reference allele the ALID spells first, second or not at all, it
    answers as a plain reading of each place does."""
    rng = np.random.default_rng(357)
    reversed_seen = 0
    for _ in range(300):
        n = int(rng.integers(0, 40))
        place = rng.integers(0, 12, n)
        z = rng.choice([1.0, 2.0, np.nan], n)
        se = rng.choice([0.1, 0.2, np.nan], n)
        reference = rng.choice([0, 1, 1, 2, 2], n).astype(np.int8) if declared else None

        found = route_table.collapse_duplicates(place, z, se, reference)

        assert (
            found.keep.tolist(), found.conflicting.tolist(), found.dropped.tolist(),
            found.collapsed, found.reversed_indels.tolist(),
        ) == _read_place_by_place(place, z, se, reference)
        reversed_seen += len(found.reversed_indels)
    assert (reversed_seen > 0) == declared


def _indel_association(
    site: Site, z: float, reference: ReferenceAllele = ReferenceAllele.REF
) -> ReaderAssociation:
    chrom, pos, ref, alt = site
    return ReaderAssociation(
        chrom, pos, ref, alt, z, 0.1, StoredEffectScale.SD, reference=reference
    )


#: BMI_IRN (FinnGen r13) at 1:827587: an insertion and a deletion of one repeat,
#: whose sorted alleles make one ALID, `1:827587:G:GGGGCC`.
INSERTION: Site = ("1", 827587, "G", "GGGGCC")
DELETION: Site = ("1", 827587, "GGGGCC", "G")


def _reversed_pair_column(stream):
    routing = _hg38_routing([INSERTION, DELETION])
    (alid,) = routing.alids
    assert alid == "1:827587:G:GGGGCC"
    return routing.table([routing.alids]).route_column(stream).placed[0]


@pytest.mark.parametrize("order", ["insertion first", "deletion first"])
def test_a_reversed_indel_keeps_the_row_whose_declared_reference_the_alid_spells_first(order):
    """D5 for an insertion and a deletion sharing an ALID: when the source
    declares which allele is the reference, the insertion -- the reading the
    ALID spells -- is kept wherever it falls, and the deletion is dropped."""
    stream = [_indel_association(INSERTION, 1.5), _indel_association(DELETION, -0.5)]
    if order == "deletion first":
        stream.reverse()

    placed = _reversed_pair_column(stream)

    assert placed.index.tolist() == [0]
    assert placed.z.tolist() == [1.5]
    assert placed.reversed_indels.tolist() == [0]
    assert placed.conflicting.tolist() == []
    assert (placed.dropped.tolist(), placed.collapsed) == ([], 0)
    assert placed.reference.tolist() == [route_table.REFERENCE_FIRST]


def test_a_reversed_indel_with_no_declared_reference_drops_both_rows():
    """Effect and other alleles do not say which is the genome's: D5 unchanged."""
    stream = [
        _indel_association(INSERTION, 1.5, ReferenceAllele.UNDECLARED),
        _indel_association(DELETION, -0.5, ReferenceAllele.UNDECLARED),
    ]

    placed = _reversed_pair_column(stream)

    assert placed.index.tolist() == []
    assert placed.conflicting.tolist() == [0]
    assert placed.dropped.tolist() == [0, 0]
    assert placed.reversed_indels.tolist() == []


def test_a_declared_alt_reference_is_judged_on_the_alt_allele():
    """A row whose source declares its `alt` the reference spells the ALID's
    first allele when its `alt` sorts first: `GGGGCC>G` with `G` the reference
    is the insertion, and is kept over `G>GGGGCC` whose reference is `GGGGCC`."""
    stream = [
        _indel_association(INSERTION, 1.5, ReferenceAllele.ALT),
        _indel_association(DELETION, -0.5, ReferenceAllele.ALT),
    ]

    placed = _reversed_pair_column(stream)

    assert placed.z.tolist() == [-0.5]
    assert placed.reversed_indels.tolist() == [0]


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
    assert placed.reversed_indels.tolist() == []


def test_a_reversed_indel_is_resolved_across_batches(monkeypatch):
    """The declaration travels with each row through a batch boundary, and a
    reader that changes it mid-stream changes it from that row on."""
    monkeypatch.setattr(route_table, "_RESOLVE_BATCH", 2)
    routing = _hg38_routing([INSERTION, DELETION, ("1", 900, "A", "AT"), ("1", 900, "AT", "A")])
    table = routing.table([routing.alids])
    stream = [
        _indel_association(("1", 900, "A", "AT"), 1.0, ReferenceAllele.UNDECLARED),
        _indel_association(DELETION, -0.5),
        _indel_association(("1", 900, "AT", "A"), 2.0, ReferenceAllele.UNDECLARED),
        _indel_association(INSERTION, 1.5),
    ]

    placed = table.route_column(stream).placed[0]
    alids = [routing.alids[i] for i in placed.index.tolist()]

    assert alids == ["1:827587:G:GGGGCC"]
    assert placed.z.tolist() == [1.5]
    assert [routing.alids[i] for i in placed.conflicting.tolist()] == ["1:900:A:AT"]


def test_a_lift_that_reorients_a_site_moves_its_reference_to_the_other_side():
    """A reverse-strand lift swaps which allele the stored ALID spells first,
    so a re-oriented insertion's reference sits second."""
    alid = "1:827587:G:GGGGCC"
    table = route_table.RouteTable.build(
        {INSERTION: alid}, {INSERTION}, [[alid]], keep_unmatched=False
    )

    placed = table.route_column([_indel_association(INSERTION, 1.5)]).placed[0]

    assert placed.z.tolist() == [-1.5]
    assert placed.reference.tolist() == [route_table.REFERENCE_SECOND]


def test_an_unmatched_row_carries_its_reference_code():
    """A row no axis holds is coded against its own key's alleles, which is
    what the Hybrid's fold resolves it under."""
    routing = _hg38_routing([("1", 100, "A", "G")])
    table = routing.table([routing.alids], keep_unmatched=True)
    stream = [
        _indel_association(INSERTION, 1.5),
        _indel_association(DELETION, -0.5),
        _indel_association(DELETION, -0.5, ReferenceAllele.UNDECLARED),
    ]

    unmatched = table.route_column(stream).unmatched

    assert unmatched.reference.tolist() == [
        route_table.REFERENCE_FIRST, route_table.REFERENCE_SECOND, route_table.REFERENCE_UNKNOWN,
    ]


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
        sources, FromReference(read_variant_reference(artifact))
    )

    assert result.n_variants == len(union.alids) == 3
    assert frozen.alids == union.alids
    # Both allele orders of 5:10 are one site each; in source order the
    # A:G site sorts first, so its name is the first named.
    assert frozen.names.rsid_by_alid == union.names.rsid_by_alid == {
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
