"""`build_ragged_from_ssf` for fixtures whose manifests never state their SD scale.

A Ragged SSF build takes each Analysis's `stored_effect_scale` and
`original_sd_method` from its manifest row and refuses a row that names none
(issue #366, ADR 0084). Most fixtures here test something else and write rows
without either column. This wrapper completes such a manifest in place (see
`declare_scale`): a row with no scale is a source already in SD Units
(`sd` / `declared_standardised`), or `log_or` / `binary_trait` when the caller
passes `stored_effect_scale="log_or"`. A cell the manifest states is left alone,
and the tests of the refusal itself call the package's `build_ragged_from_ssf`
directly.
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
    """Fill in what a fixture manifest leaves out, row by row.

    A blank or absent scale is `scale`, and its method follows from it. A scale the row
    states keeps its own, and a blank method is filled from *that* scale -- except
    `sd`, whose blank method is the case the build refuses, so it is left to reach it.
    """
    path = Path(manifest_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    header = rows[0]
    for name in ("stored_effect_scale", "original_sd_method"):
        if name not in header:
            header.append(name)
            for row in rows[1:]:
                if row:
                    row.append("")
    scale_at = header.index("stored_effect_scale")
    method_at = header.index("original_sd_method")
    for row in rows[1:]:
        if not row:
            continue
        if not row[scale_at]:
            row[scale_at] = scale
            if not row[method_at]:
                row[method_at] = _METHOD_FOR[scale]
        elif not row[method_at] and row[scale_at] != "sd":
            row[method_at] = _METHOD_FOR.get(row[scale_at], "")
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)


def build_ragged_from_ssf(manifest_path: str | Path, *args: Any, **kwargs: Any) -> Any:
    """The package's build, over a manifest completed by `declare_scale`."""
    scale = kwargs.pop("stored_effect_scale", "sd")
    declare_scale(manifest_path, scale)
    return _build(manifest_path, *args, **kwargs)
