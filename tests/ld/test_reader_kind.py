"""One root read for the kind dispatch, and a genotype reader that composes, not inherits
(opengwas/ogld#91)."""

from __future__ import annotations

from pathlib import Path

import build_fixtures
import numpy as np
import pytest
import testdata
from contract.support import TEST_STORE
from ogquery.ld.store import StoreReader
from ogquery.ld.store import location as open_module
from ogquery.ld.store.variants import VariantTable

import opengwasdb.ld
from opengwasdb.ld.genotypes.reader import GenotypeReader


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
    opengwasdb.ld.open(TEST_STORE)
    assert len(reads) == 1, f"an LD Store open reads its root metadata {len(reads)} times"
    reads.clear()
    assert not any(opengwasdb.ld.validate(TEST_STORE).values())
    assert len(reads) == 1, f"an LD Store validate reads its root metadata {len(reads)} times"


def test_an_open_and_a_validate_read_a_genotype_store_s_root_once(
    genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = count_root_reads(monkeypatch)
    opengwasdb.ld.open(genotype_store)
    assert len(reads) == 1, f"a Genotype Store open reads its root metadata {len(reads)} times"
    reads.clear()
    assert not any(opengwasdb.ld.validate(genotype_store).values())
    assert len(reads) == 1, f"a Genotype Store validate reads its root {len(reads)} times"


def test_the_genotype_reader_composes_the_variant_table_and_store(genotype_store: Path) -> None:
    from ogquery.ld.query.reader import Reader

    reader = opengwasdb.ld.open(genotype_store)._reader
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


def test_a_store_closes_its_reader_and_its_pgen_handles(
    genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import closing

    from opengwasdb.ld.genotypes.pgen import Genotypes

    closed: list[object] = []
    original_close = Genotypes.close

    def counted_close(self: Genotypes) -> None:
        closed.append(self)
        original_close(self)

    monkeypatch.setattr(Genotypes, "close", counted_close)
    names = build_fixtures.alids(np.flatnonzero(testdata.variants().passes_filters)[:10])
    store = opengwasdb.ld.open(genotype_store)
    store.matrix(names)
    reader = store._reader
    handles = len(reader._handles._idle)
    assert handles >= 1, "the fixture must open at least one pgen handle"
    store.close()
    assert reader._closed
    assert reader._handles._idle == [], "a closed store keeps no idle pgen handle"
    assert len(closed) == handles, "close() closes every handle the reader kept"
    with pytest.raises(opengwasdb.ld.InputError):
        store.matrix(names)
    with closing(opengwasdb.ld.open(TEST_STORE)) as ld:
        assert ld.provenance["format_version"]
