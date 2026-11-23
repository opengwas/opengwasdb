"""The variant axis carries an optional source_alid (build-assembly provenance)."""

from __future__ import annotations

from opengwasdb.store_reader.variant_axis import parse_variant_line as _parse_variant_line
from opengwasdb.variants.axis import VariantAxis, write_variant_axis
from opengwasdb.variants.normalise import CanonicalVariant


def _variant(chrom: str, pos: int, a1: str, a2: str) -> CanonicalVariant:
    return CanonicalVariant(chromosome=chrom, position=pos, effect_allele=a1, other_allele=a2)


def test_source_alid_round_trips(tmp_path):
    variants = [
        _variant("1", 1_000, "A", "G"),
        _variant("1", 2_000, "C", "T"),
        _variant("1", 3_000, "A", "T"),
    ]
    # row 0 lifted from an hg19 coordinate; row 1 blank (e.g. collision); row 2 given
    source_alids = ["1:900:A:G", None, "1:2900:A:T"]
    write_variant_axis(tmp_path, variants, {}, source_alids)

    va = VariantAxis(tmp_path)
    try:
        assert va.by_index(0).source_alid == "1:900:A:G"
        assert va.by_index(1).source_alid is None  # "." → None
        assert va.by_index(2).source_alid == "1:2900:A:T"
        # source_alid also surfaces through range() and the alid lookup path
        assert va.by_identifier("1:1000:A:G").source_alid == "1:900:A:G"
    finally:
        va.close()


def test_source_alids_length_mismatch_raises(tmp_path):
    variants = [_variant("1", 1_000, "A", "G")]
    try:
        write_variant_axis(tmp_path, variants, {}, ["1:900:A:G", "1:extra:A:G"])
    except ValueError as exc:
        assert "source_alids" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on length mismatch")


def test_legacy_seven_field_row_parses_with_none_source_alid():
    # Stores built before the source_alid column have 7 tab-separated fields.
    line = "1\t1000\t0\tA\tG\t1:1000:A:G\t.\n"
    record = _parse_variant_line(line)
    assert record.alid == "1:1000:A:G"
    assert record.source_alid is None
