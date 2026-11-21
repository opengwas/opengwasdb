"""One root read for the kind dispatch, and a genotype reader that composes, not inherits (#91)."""

from __future__ import annotations

from pathlib import Path

import pytest
from contract.support import TEST_STORE

import ogld
from ogld.genotypes.reader import GenotypeReader
from ogld.store import StoreReader
from ogld.store import location as open_module
from ogld.store.variants import VariantTable


def count_root_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The root openings `open_group` makes, counted where zarr is first asked for the group."""
    reads: list[Path] = []
    original = open_module._opened

    def counted(path: str | Path) -> object:
        reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(open_module, "_opened", counted)
    return reads


def test_an_open_and_a_validate_read_an_ld_store_s_root_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = count_root_reads(monkeypatch)
    ogld.open(TEST_STORE)
    assert len(reads) == 1, f"an LD Store open reads its root metadata {len(reads)} times"
    reads.clear()
    assert not any(ogld.validate(TEST_STORE).values())
    assert len(reads) == 1, f"an LD Store validate reads its root metadata {len(reads)} times"


def test_an_open_and_a_validate_read_a_genotype_store_s_root_once(
    genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = count_root_reads(monkeypatch)
    ogld.open(genotype_store)
    assert len(reads) == 1, f"a Genotype Store open reads its root metadata {len(reads)} times"
    reads.clear()
    assert not any(ogld.validate(genotype_store).values())
    assert len(reads) == 1, f"a Genotype Store validate reads its root {len(reads)} times"


def test_the_genotype_reader_composes_the_variant_table_and_store(genotype_store: Path) -> None:
    from ogld.query.reader import Reader

    reader = ogld.open(genotype_store)._reader
    assert isinstance(reader, GenotypeReader)
    assert isinstance(reader, Reader)
    assert not isinstance(reader, StoreReader), "a genotype reader is not an LD Store reader"
    assert isinstance(reader._variant_table, VariantTable)
    assert reader._opened.root is reader._variant_table._opened.root
    # The inherited members that meant nothing here are gone, not silently wrong.
    assert not hasattr(reader, "r_of_pairs")
    assert not hasattr(reader, "frame_cache_bytes")
    assert not hasattr(reader, "has_partner_rows")
    # And an argument that had no meaning on this kind is refused, not ignored.
    with pytest.raises(TypeError):
        GenotypeReader.open(genotype_store, frame_cache_bytes=1)
