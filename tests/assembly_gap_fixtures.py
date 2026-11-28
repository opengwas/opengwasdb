"""The GRCh38 table with no gaps that the suite and its generators read (#410).

Test infrastructure only. The suite's synthetic sources, and the Store Reader contract
bundle's (`tests/store_reader_contract/generate.py`), place GRCh38 rows at positions
like `1:100` or `1:1000`, which lie in chr1's first gap (1-10,000), where the
committed table would have every build drop them. Their positions stand for any
position, so they keep them, and read a table holding GRCh38's contig lengths and no
gap, under its own checksum. The package itself only ever reads the committed,
checksummed table (`opengwasdb.variants.assembly_gaps`); moving the test positions out
of the gaps is a named follow-up.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

import opengwasdb.variants.assembly_gaps as assembly_gaps

#: The committed table, beside its module whatever has repointed `GRCH38_TABLE`, and its
#: recorded checksum, read when this module is first imported: by `tests/conftest.py`,
#: before any fixture repoints it.
COMMITTED_TABLE = Path(assembly_gaps.__file__).with_name("grch38-assembly-gaps.tsv")
COMMITTED_SHA256 = assembly_gaps.GRCH38_TABLE_SHA256


@contextmanager
def grch38_without_gaps(directory: Path) -> Iterator[Path]:
    """While open, `assembly_gaps.grch38()` reads a table written in ``directory`` with
    the committed table's contig rows and none of its gap rows."""
    lines = COMMITTED_TABLE.read_text(encoding="utf-8").splitlines()
    no_gaps = directory / "grch38-without-gaps.tsv"
    no_gaps.write_text(
        "\n".join(line for line in lines if not line.startswith("gap\t")) + "\n",
        encoding="utf-8",
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assembly_gaps, "GRCH38_TABLE", no_gaps)
        patch.setattr(
            assembly_gaps, "GRCH38_TABLE_SHA256", hashlib.sha256(no_gaps.read_bytes()).hexdigest()
        )
        yield no_gaps


def read_committed_grch38(patch: pytest.MonkeyPatch) -> None:
    """Until ``patch`` is undone, `assembly_gaps.grch38()` reads the committed table."""
    patch.setattr(assembly_gaps, "GRCH38_TABLE", COMMITTED_TABLE)
    patch.setattr(assembly_gaps, "GRCH38_TABLE_SHA256", COMMITTED_SHA256)
