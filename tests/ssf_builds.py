"""`build_ragged_from_ssf` for fixtures whose manifests never state their SD scale.

A Ragged SSF build takes each Analysis's `stored_effect_scale` and
`original_sd_method` from its manifest row and refuses a row that names none
(issue #366, ADR 0084). Most fixtures here test something else and write rows
without either column. This wrapper completes such a manifest, in place and
only where a column is absent or a cell blank, as a source already in SD Units
(`sd` / `declared_standardised`), or as `log_or` / `binary_trait` when the
caller passes `stored_effect_scale="log_or"`. A cell the manifest states is left alone,
and the tests of the refusal itself call the package's `build_ragged_from_ssf` directly.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf as _build

_METHOD_FOR = {
    "sd": "declared_standardised",
    "log_or": "binary_trait",
    "log_hazard": "binary_trait",
}


def declare_scale(manifest_path: str | Path, scale: str = "sd") -> None:
    """Fill in the scale and method a manifest leaves out: a missing column, or a blank cell."""
    path = Path(manifest_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    header = rows[0]
    wanted = {"stored_effect_scale": scale, "original_sd_method": _METHOD_FOR[scale]}
    for name in wanted:
        if name not in header:
            header.append(name)
            for row in rows[1:]:
                if row:
                    row.append("")
    for name, value in wanted.items():
        at = header.index(name)
        for row in rows[1:]:
            if row and not row[at]:
                row[at] = value
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)


def build_ragged_from_ssf(manifest_path: str | Path, *args: Any, **kwargs: Any) -> Any:
    """The package's build, over a manifest completed by `declare_scale`."""
    scale = kwargs.pop("stored_effect_scale", "sd")
    declare_scale(manifest_path, scale)
    return _build(manifest_path, *args, **kwargs)
