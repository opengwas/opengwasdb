"""Validation reads a store's root once, whichever kind it holds (opengwas/ogld#91).

The kind dispatch is ogquery's (`open_kind`), and what `open` does with the root, and the
Genotype Store reader's composition and closing, are held in ogquery's own suite
(opengwas/ogquery#11). What stays here is `opengwasdb.ld.validate`, which reads the same one
root `open` does and then runs the rules of the kind it found.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from contract.support import TEST_STORE
from ogquery.ld.store import location as open_module

import opengwasdb.ld


def count_root_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The root openings `open_group` makes, counted where zarr is first asked for the group."""
    reads: list[Path] = []
    original = open_module._opened

    def counted(path: str | Path) -> object:
        reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(open_module, "_opened", counted)
    return reads


def test_a_validate_reads_an_ld_store_s_root_once(monkeypatch: pytest.MonkeyPatch) -> None:
    reads = count_root_reads(monkeypatch)
    assert not any(opengwasdb.ld.validate(TEST_STORE).values())
    assert len(reads) == 1, f"an LD Store validate reads its root metadata {len(reads)} times"


def test_a_validate_reads_a_genotype_store_s_root_once(
    committed_genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = count_root_reads(monkeypatch)
    assert not any(opengwasdb.ld.validate(committed_genotype_store).values())
    assert len(reads) == 1, f"a Genotype Store validate reads its root {len(reads)} times"
