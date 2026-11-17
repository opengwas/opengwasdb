"""Two completed Store Releases are the same release (#321).

"The same" is what the release comparison tool compares: every array's
metadata (on-disk size included) and decoded values, `analyses.tsv` and
`variants.tsv.gz`; and beside it the `completion_quality` rows in the order
they were written, and the manifest but for when it was written. Stored bytes
are not compared: a chunk the parent compresses with Blosc threads writes its
blocks in the order they finish (ADR 0056), so two runs of the same code
already differ there.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.compare_store_releases import compare_releases


@dataclass(frozen=True)
class CompletionInputs:
    """A completion's source and panel, the release one worker makes of them,
    and how the test completes them (`run(source, panel, dst, n_workers)`)."""

    source: Path
    panel: Path
    serial: Path
    run: Callable[[Path, Path, Path, int], Path]

    def complete(self, dst: Path, n_workers: int) -> Path:
        return self.run(self.source, self.panel, dst, n_workers)


def quality_rows(release: Path) -> list[tuple[Any, ...]]:
    with sqlite3.connect(release / "index.sqlite") as db:
        return db.execute("SELECT * FROM completion_quality ORDER BY rowid").fetchall()


def _manifest_without_timestamp(release: Path) -> dict[str, Any]:
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("created_at")
    return manifest


def assert_same_release(left: Path, right: Path) -> None:
    """The release comparison tool finds nothing, and neither does the rest."""
    report = compare_releases(left, right)
    assert report["arrays_only_left"] == [] and report["arrays_only_right"] == []
    assert report["arrays_with_metadata_differences"] == []
    assert report["arrays_with_value_differences"] == []
    assert report["analyses_tsv"]["differing_cells_by_column"] == {}
    assert report["variants_tsv_gz"]["identical"]
    assert report["encoding_plan"]["left"] == report["encoding_plan"]["right"]
    assert quality_rows(left) == quality_rows(right)
    assert _manifest_without_timestamp(left) == _manifest_without_timestamp(right)
