"""rsid resolution through the Store Variant Table's own index (issue #109).

Before this, rsids reached `variants.tsv.gz` but nothing indexed them, so
`by_identifier("rs123")` returned None -- an empty result indistinguishable
from a real "no association here". The index is written by
`write_variant_axis`, the one function every builder and every completion
path already calls, so a layout cannot write rsids and forget to index them.
"""

from __future__ import annotations

from pathlib import Path

from opengwasdb.store_reader.format import CURRENT_FORMAT_VERSION
from opengwasdb.variants.axis import VariantAxis, write_variant_axis
from opengwasdb.variants.normalise import CanonicalVariant


def _variants() -> list[CanonicalVariant]:
    return [
        CanonicalVariant(chromosome="1", position=100, effect_allele="A", other_allele="G"),
        CanonicalVariant(chromosome="1", position=200, effect_allele="C", other_allele="T"),
        # Same position as row 1, other allele order: a legitimate second row
        # for one rsid (the multi-allelic / both-orders collision case).
        CanonicalVariant(chromosome="1", position=200, effect_allele="A", other_allele="C"),
        CanonicalVariant(chromosome="2", position=300, effect_allele="G", other_allele="T"),
    ]


def _axis(tmp_path: Path) -> VariantAxis:
    variants = _variants()
    write_variant_axis(
        tmp_path,
        variants,
        {
            variants[0].alid: "rs1",
            variants[1].alid: "rs2",
            variants[2].alid: "rs2",
            # variants[3] deliberately has no rsid
        },
    )
    return VariantAxis(tmp_path, format_version=CURRENT_FORMAT_VERSION)


def test_rsid_resolves_to_its_variant(tmp_path: Path):
    axis = _axis(tmp_path)
    record = axis.by_identifier("rs1")
    assert record is not None
    assert record.alid == "1:100:A:G"
    assert record.rsid == "rs1"


def test_unknown_rsid_resolves_to_nothing(tmp_path: Path):
    assert _axis(tmp_path).by_identifier("rs404") is None


def test_variant_without_an_rsid_is_not_reachable_by_a_dot(tmp_path: Path):
    """`.` is the table's blank marker, not an identifier anyone can look up."""
    assert _axis(tmp_path).by_identifier(".") is None


def test_one_rsid_on_several_rows_resolves_to_all_of_them(tmp_path: Path):
    """An rsid can legitimately name more than one stored row (multi-allelic
    site, or the same position stored under both allele orders). The batch
    path returns every one; `by_identifier`, whose contract is a single
    record, returns the lowest variant_index of them."""
    axis = _axis(tmp_path)
    assert sorted(axis.indices_by_identifiers(["rs2"]).tolist()) == [1, 2]
    record = axis.by_identifier("rs2")
    assert record is not None and record.variant_index == 1


def test_alid_and_rsid_identifiers_batch_together(tmp_path: Path):
    axis = _axis(tmp_path)
    resolved = axis.indices_by_identifiers(["1:100:A:G", "rs2", "rs404"])
    assert sorted(resolved.tolist()) == [0, 1, 2]


def test_axis_without_an_rsid_index_still_opens(tmp_path: Path):
    """Stores built before this index exists must keep working -- alid lookups
    unaffected, rsid lookups simply find nothing (the pre-#109 behaviour)."""
    axis = _axis(tmp_path)
    from opengwasdb.store_reader.variant_axis import variant_rsid_bytes_path, variant_rsid_rows_path

    axis.close()
    variant_rsid_bytes_path(tmp_path).unlink()
    variant_rsid_rows_path(tmp_path).unlink()

    stale = VariantAxis(tmp_path, format_version=CURRENT_FORMAT_VERSION)
    assert stale.by_identifier("1:100:A:G") is not None
    assert stale.by_identifier("rs1") is None
