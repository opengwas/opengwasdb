"""Variant References declare their indel semantics before routing (#381)."""

import gzip
from functools import partial
from pathlib import Path

import pytest

from opengwasdb.build.liftover import read_reference_alleles
from opengwasdb.build.resolve_manifest import load_variant_reference
from opengwasdb.layouts.hybrid.build import read_reference_panel_alids
from opengwasdb.variants.reference import read_variant_reference, write_variant_reference

MARKER = "#opengwasdb-variant-reference\t1\n"


def _reference(tmp_path: Path, shape: str, marked: bool, alids: list[str]) -> Path:
    text = MARKER if marked else ""
    if shape == "artifact":
        text += "#alid\tchromosome\tposition\ta1\ta2\trouted_source_keys\treoriented_keys\n"
        text += "".join(f"{alid}\t" + "\t".join(alid.split(":")) + "\t\t\n" for alid in alids)
    elif shape == "table":
        text += "#rsid\talid\n" + "".join(f"rs1\t{alid}\n" for alid in alids)
    else:
        text += "\n".join(alids) + "\n"
    path = tmp_path / "reference.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)
    return path


@pytest.mark.parametrize("shape", ["plain", "artifact", "table"])
@pytest.mark.parametrize("indel", ["1:827587:G:GGGGCC", "1:827587:GGGGCC:G"])
def test_an_unmarked_indel_is_refused_before_it_can_define_a_routing(
    tmp_path: Path, shape: str, indel: str,
) -> None:
    reference = _reference(tmp_path, shape, False, ["1:100:A:G", indel])

    with pytest.raises(ValueError, match=rf"{indel}.*regenerated.*reference-first"):
        read_variant_reference(reference)


def test_the_writer_declares_semantics_and_preserves_routing_on_read(tmp_path: Path) -> None:
    path = tmp_path / "new.variant-ref.tsv.gz"
    alids = ["1:100:A:G", "1:827587:G:GGGGCC", "1:827587:GGGGCC:G"]
    source_lookup = {
        ("1", 100, "G", "A"): alids[0],
        ("1", 827587, "G", "GGGGCC"): alids[1],
        ("1", 827587, "GGGGCC", "G"): alids[2],
    }
    reoriented = frozenset({("1", 100, "G", "A")})
    write_variant_reference(path, alids, source_lookup, reoriented=reoriented)

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert handle.readline() == MARKER
    reference = read_variant_reference(path)
    assert set(reference.alids) == set(alids)
    assert reference.source_lookup == source_lookup
    assert reference.reoriented == reoriented


@pytest.mark.parametrize(
    "read",
    [
        read_reference_panel_alids,
        load_variant_reference,
        partial(read_reference_alleles, positions={("1", 100)}),
        partial(read_reference_alleles, positions=set()),
    ],
    ids=["hybrid-panel", "resolver", "besd-unrequested-row", "besd-no-positions"],
)
@pytest.mark.parametrize("shape", ["plain", "artifact", "table"])
def test_every_reference_consumer_refuses_an_unmarked_indel(
    tmp_path: Path, read, shape: str,
) -> None:
    reference = _reference(tmp_path, shape, False, ["1:100:A:G", "1:827587:G:GGGGCC"])

    with pytest.raises(ValueError, match="1:827587:G:GGGGCC.*regenerated.*reference-first"):
        read(reference)


@pytest.mark.parametrize("shape", ["plain", "artifact", "table"])
def test_a_declared_reference_preserves_both_indels_at_one_locus(
    tmp_path: Path, shape: str,
) -> None:
    alids = ["1:100:A:G", "1:827587:G:GGGGCC", "1:827587:GGGGCC:G"]
    reference = _reference(tmp_path, shape, True, alids)

    assert read_variant_reference(reference).alids == alids
    assert read_reference_panel_alids(reference) == set(alids)
    assert load_variant_reference(reference) == set(alids)
    assert read_reference_alleles(reference, {("1", 827587)}) == {
        ("1", 827587): {frozenset({"G", "GGGGCC"})}
    }


@pytest.mark.parametrize("shape", ["plain", "artifact", "table"])
def test_an_unmarked_snv_reference_stays_usable(tmp_path: Path, shape: str) -> None:
    reference = _reference(tmp_path, shape, False, ["1:100:A:G"])

    assert read_variant_reference(reference).alids == ["1:100:A:G"]
    assert read_reference_panel_alids(reference) == {"1:100:A:G"}
    assert load_variant_reference(reference) == {"1:100:A:G"}
    assert read_reference_alleles(reference, {("1", 100)}) == {
        ("1", 100): {frozenset({"A", "G"})}
    }


@pytest.mark.parametrize("declaration", ["0", "2", "1\textra"])
def test_an_unsupported_declaration_is_refused(tmp_path: Path, declaration: str) -> None:
    path = tmp_path / "unsupported.txt"
    path.write_text(f"#opengwasdb-variant-reference\t{declaration}\n1:100:A:G\n")

    with pytest.raises(ValueError, match="unsupported semantics marker"):
        read_variant_reference(path)


def test_a_late_declaration_cannot_authorize_earlier_rows(tmp_path: Path) -> None:
    path = tmp_path / "late.txt"
    path.write_text("1:100:A:G\n" + MARKER + "1:827587:G:GGGGCC\n")

    with pytest.raises(ValueError, match="misplaced semantics marker"):
        read_variant_reference(path)


def test_a_detached_table_does_not_inherit_a_nearby_format_stamp(tmp_path: Path) -> None:
    reference = _reference(tmp_path, "table", False, ["1:827587:G:GGGGCC"])
    (tmp_path / "manifest.json").write_text('{"format_version": "0.2.2"}')

    with pytest.raises(ValueError, match="1:827587:G:GGGGCC.*regenerated"):
        read_variant_reference(reference)
