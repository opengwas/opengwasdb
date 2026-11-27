"""What a Genotype Store's writer records of its inputs: relative pins and a fileset's record
(opengwas/ogld#91).

The sidecar's own layout a read opens -- the root key, the arrays, `check_input`, `Fileset` and
the mapped arrays -- is ogquery's (`ogquery.ld.genotypes.layout`; opengwas/ogquery#11), where
the writer and validation import it from. This module keeps the pins only a writer makes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ogquery.ld.genotypes.pgen import PgenFileset

from opengwasdb.ld.store import layout


def input_pin(path: Path, root: Path) -> dict[str, Any]:
    """Paths move with the sidecar's tree; size/mtime are cheap open-time guards (opengwas/ogld#91
    ruling)."""
    stat = path.stat()
    return {
        "path": Path(os.path.relpath(path, root)).as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": layout.file_sha256(path),
    }


def fileset_record(fileset: PgenFileset, pvar: Path, root: Path) -> dict[str, Any]:
    return {
        "pgen": input_pin(fileset.pgen, root),
        "psam": input_pin(fileset.psam, root),
        "pvar": input_pin(pvar, root),
        "subset": fileset.subset.tolist(),
        "order": None if fileset.order is None else fileset.order.tolist(),
        "samples": fileset.samples,
        "panel": fileset.panel,
    }
