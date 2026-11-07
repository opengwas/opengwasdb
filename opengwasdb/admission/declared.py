"""The filters each Analysis declared, read from its build manifest (stores #175, #176).

Every builder judges a row against the same declared INFO and MAF policies, so
they are parsed from the manifest one way: `declared_rules` reads each row's
policy columns, keyed by the row's Analysis id, and returns one
`AnalysisRules` per Analysis a builder parsed. A partial or contradictory
declaration is a manifest error naming its Analysis, never a filter silently
not applied.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

from opengwasdb.admission.rules import AnalysisRules
from opengwasdb.model.info_score_policy import parse_info_score_policy
from opengwasdb.model.maf_policy import parse_maf_policy
from opengwasdb.model.manifest_columns import manifest_column

__all__ = ["declared_rules"]


def declared_rules(
    manifest_path: str | Path, capabilities: Mapping[str, str | None]
) -> dict[str, AnalysisRules]:
    """Each Analysis's declared rules, by Analysis id.

    ``capabilities`` names the Analyses the builder parsed from the same
    manifest, each with its resolved reader capability (the row's, or the
    build's default), which decides whether a declared score can be read: a
    numeric INFO threshold needs a GWAS-SSF source. Rows are matched by their
    `analysis_id` (or legacy `trait_id`) column, so a builder's own row order
    does not matter, and a manifest whose Analyses differ from the builder's
    is refused.
    """
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        id_column = manifest_column(reader.fieldnames or [], "analysis_id")
        raw_rows = list(reader)
    _refuse_repeated(manifest_path, id_column, raw_rows)
    found = {raw.get(id_column or "", "") for raw in raw_rows}
    if id_column is None or found != set(capabilities) or len(raw_rows) != len(capabilities):
        raise ValueError(
            f"manifest {manifest_path} changed while it was read: its Analyses are not "
            f"the {len(capabilities)} the build parsed"
        )
    return {
        raw[id_column]: _rules(manifest_path, raw[id_column], raw, capabilities[raw[id_column]])
        for raw in raw_rows
    }


def _refuse_repeated(
    manifest_path: str | Path, id_column: str | None, raw_rows: list[dict[str, str]]
) -> None:
    """Refuse an Analysis a manifest names in more than one row, naming it and the rows."""
    rows: dict[str, list[int]] = {}
    for number, raw in enumerate(raw_rows, start=2):  # the header is line 1
        rows.setdefault(raw.get(id_column or "", ""), []).append(number)
    for analysis_id, numbers in rows.items():
        if len(numbers) > 1:
            named = ", ".join(map(str, numbers[:-1])) + f" and {numbers[-1]}"
            raise ValueError(
                f"manifest {manifest_path}: analysis {analysis_id!r} is named by rows {named}; "
                "one row each"
            )


def _rules(
    manifest_path: str | Path, analysis_id: str, raw: Mapping[str, str], capability: str | None
) -> AnalysisRules:
    try:
        info = parse_info_score_policy(raw, reader_capability=capability)
    except ValueError as exc:
        raise ValueError(
            f"manifest {manifest_path}: analysis {analysis_id!r} has invalid INFO policy: {exc}"
        ) from exc
    try:
        maf = parse_maf_policy(raw)
    except ValueError as exc:
        raise ValueError(
            f"manifest {manifest_path}: analysis {analysis_id!r} has invalid MAF policy: {exc}"
        ) from exc
    return AnalysisRules(analysis_id, info=info, maf=maf)
