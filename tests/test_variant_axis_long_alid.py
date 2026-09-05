"""Long indel ALIDs must not resolve each other's lookups (issue #127).

The ALID search index is a fixed-width mmap'd array, which is what makes
`np.searchsorted` possible on it. Anything wider than a slot used to be cast
into one silently: two indels at one position whose alleles agree over the
slot width collapsed to a single key, and a lookup by either full ALID
returned whichever row sorted first. Found on the published
`finngen-r13/r13-pilot-20` release -- 6,216 truncated ALIDs, 248 keys shared
by more than one variant, 342 variants answering to another's name.
"""

from __future__ import annotations

from opengwasdb.variants import VariantAxis
from opengwasdb.variants.axis import _ALID_WIDTH, is_indexable_alid, write_variant_axis
from opengwasdb.variants.normalise import CanonicalVariant


def _variant(chrom: str, pos: int, a1: str, a2: str) -> CanonicalVariant:
    return CanonicalVariant(chromosome=chrom, position=pos, effect_allele=a1, other_allele=a2)


def _shared_prefix_pair(pos: int = 1_175_982) -> tuple[CanonicalVariant, CanonicalVariant]:
    """Two indels at one position, identical well past the index slot width.

    The shape of the real collision: `10:1175982:G:GGCC…` twice, the second an
    extension of the first.
    """
    shared = "G" + "GCCTGGCCTGAAGGGAGGTAGGGTGGGACTGGAGACGCGGGGTGGACGAG" * 2
    return (
        _variant("10", pos, "G", shared),
        _variant("10", pos, "G", shared + "AGCAACACCTCCAAACGCAGGCACCCGCA"),
    )


def test_the_fixture_alids_are_wider_than_the_index_slot():
    """Meaningfulness: a pair that fits the slot would never collide."""
    short, long_ = _shared_prefix_pair()
    assert len(short.alid) > _ALID_WIDTH, (short.alid, _ALID_WIDTH)
    assert short.alid[:_ALID_WIDTH] == long_.alid[:_ALID_WIDTH], "the pair must share a full slot"
    assert not is_indexable_alid(short.alid)
    assert not is_indexable_alid(long_.alid)


def test_a_long_indel_resolves_to_itself_not_its_neighbour(tmp_path):
    """Each ALID returns its own row, or nothing -- never the other variant's."""
    short, long_ = _shared_prefix_pair()
    write_variant_axis(tmp_path, [short, long_], {})

    va = VariantAxis(tmp_path)
    try:
        for expected in (short, long_):
            got = va.by_identifier(expected.alid)
            assert got is not None, f"{expected.alid[:40]}… became unreachable"
            assert got.alid == expected.alid, (
                f"asked for {expected.alid[-20:]!r}, got {got.alid[-20:]!r} -- "
                "one variant answered to another's name"
            )
    finally:
        va.close()


def test_variants_that_fit_the_slot_still_use_the_fast_index(tmp_path):
    """The exact-scan fallback must not become the normal path."""
    short, long_ = _shared_prefix_pair()
    ordinary = _variant("10", 2_000_000, "A", "G")
    write_variant_axis(tmp_path, [short, long_, ordinary], {})

    va = VariantAxis(tmp_path)
    try:
        assert is_indexable_alid(ordinary.alid)
        assert va.by_identifier(ordinary.alid).alid == ordinary.alid
        # The index holds only the indexable one; the wide pair is not truncated into it.
        assert len(va._alid_bytes) == 1, "over-wide ALIDs must be left out, not truncated in"
    finally:
        va.close()


def test_the_batched_lookup_path_resolves_long_indels_too(tmp_path):
    """`indices_by_identifiers` is the fast path dense `lookup()` uses (#3).

    It cast every query to the slot width in one vectorised `searchsorted`, so
    it carried the same collision as `by_alid` -- and it is the path that
    serves bulk queries.
    """
    short, long_ = _shared_prefix_pair()
    ordinary = _variant("10", 2_000_000, "A", "G")
    write_variant_axis(tmp_path, [short, long_, ordinary], {})

    va = VariantAxis(tmp_path)
    try:
        got = va.indices_by_identifiers([short.alid, long_.alid, ordinary.alid])
        assert sorted(int(i) for i in got) == [0, 1, 2], (
            "each ALID must resolve to its own row"
        )
        # And asked for only the longer one, it must not return the shorter one's row.
        only_long = va.indices_by_identifiers([long_.alid])
        assert [int(i) for i in only_long] == [1], only_long
    finally:
        va.close()


def test_validation_rejects_an_index_that_truncated_two_variants_together(tmp_path):
    """The detection rule for stores built before the guard (issue #128).

    A store carrying the old truncated index answers one variant's lookup with
    another's row. Nothing said so; `validate_store` passed. It is checked
    against the ALID index rather than SQLite because that is the structure a
    query actually reads.
    """
    import numpy as np

    from opengwasdb.variants.axis import variant_alid_bytes_path, variant_alid_rows_path

    short, long_ = _shared_prefix_pair()
    write_variant_axis(tmp_path, [short, long_], {})

    # Rebuild the index the old way: both wide ALIDs cast in, and so truncated
    # to the same key -- exactly what the published finngen-r13 release holds.
    keys = np.array([short.alid, long_.alid], dtype=f"|S{_ALID_WIDTH}")
    np.save(variant_alid_bytes_path(tmp_path), keys)
    np.save(variant_alid_rows_path(tmp_path), np.array([0, 1], dtype="int32"))
    assert keys[0] == keys[1], "fixture must actually collide, or nothing is under test"

    from opengwasdb.validation.validate import _validate_variant_axis

    va = VariantAxis(tmp_path)
    errors: list[str] = []
    try:
        _validate_variant_axis(va, errors)
    finally:
        va.close()
    assert any("shared by more than one variant" in e for e in errors), errors
