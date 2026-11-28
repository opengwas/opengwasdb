"""Which files of an LD reference panel a completion may read (#327, spec §12.1).

The one owner of that question: the block loader, the EAF reader and the panel's digest
all take their files from here, so none can read a file the others do not know of. It
sits in `model` so that the EAF reader, which sits below completion, can use it.
"""

from __future__ import annotations

from pathlib import Path

#: The indel-reference table and its provenance, beside a panel's blocks in
#: ``ld_dir/{ancestry}``: bgzipped and tabix-indexed, one row per panel indel
#: (`scripts/build_panel_indel_reference.py`).
INDEL_REFERENCE_TABLE = "indel_reference.tsv.gz"
INDEL_REFERENCE_PROVENANCE = "indel_reference.json"


def list_chromosomes(ld_dir: str | Path, ancestry: str) -> list[str]:
    """Return the chromosome names present in the panel for *ancestry*."""
    ancestry_dir = Path(ld_dir) / ancestry
    if not ancestry_dir.is_dir():
        return []
    return sorted(
        (p.name for p in ancestry_dir.iterdir() if p.is_dir()),
        key=lambda c: (len(c), c),
    )


def chromosome_tables(ld_dir: str | Path, ancestry: str, chrom: str) -> list[Path]:
    """Every block table (``*.tsv``) in one chromosome directory, sorted. Whether a table is
    a loadable block is `load_block`'s to say; the EAF reader reads them all."""
    return sorted((Path(ld_dir) / ancestry / chrom).glob("*.tsv"))


def block_matrix_files(table: Path) -> list[Path]:
    """The files beside a block table that carry its LD, those that exist: the stored
    eigendecomposition, then the LD matrix. A table with neither is not a loadable block."""
    return [
        path for path in (table.with_suffix(".ldeig.npz"), table.with_suffix(".unphased.vcor1.gz"))
        if path.exists()
    ]


def panel_input_files(ld_dir: str | Path, ancestry: str) -> list[Path]:
    """Every file a completion may read from the panel, in the order it reads them.

    In each chromosome directory `list_chromosomes` finds (links followed, as there), every
    block table `chromosome_tables` lists, each followed by its `block_matrix_files`. Then
    the indel-reference table beside them, its tabix index and its provenance. Nothing else
    under the directory is an input: a stray ``.rds`` or log changes nothing completion does.
    """
    root = Path(ld_dir) / ancestry
    files: list[Path] = []
    for chrom in list_chromosomes(ld_dir, ancestry):
        for table in chromosome_tables(ld_dir, ancestry, chrom):
            files += [table, *block_matrix_files(table)]
    reference = root / INDEL_REFERENCE_TABLE
    files += [
        path for path in (
            reference, reference.with_name(INDEL_REFERENCE_TABLE + ".tbi"),
            reference.with_name(INDEL_REFERENCE_PROVENANCE),
        ) if path.exists()
    ]
    return files
