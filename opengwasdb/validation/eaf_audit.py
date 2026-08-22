"""Re-run the EAF orientation check against a built store (issue #115).

`opengwasdb.validation.validate` checks the evidence a build *recorded*,
because a Store Release has to be interpretable on its own -- the reference
panel it was built against may have moved, been superseded, or never been
available to the person reading the store. This module is the other half: given
a panel now, it re-derives the answer from the store's own arrays.

Two things it is for:

* A store built before the check existed carries no evidence at all, and
  validation says so. This is how that store gets an answer without a rebuild.
* A store that recorded `passed` recorded it against one particular reference.
  Auditing against a different one is how that claim gets tested rather than
  taken on trust.

It reads through the layout-independent query API, so what it correlates is
what a user reads back -- not what the builder believed it was writing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opengwasdb.build.eaf_orientation import (
    DEFAULT_MIN_OVERLAP,
    DEFAULT_MIN_VARIANCE,
    DEFAULT_SAMPLE_SITES,
    EafOrientationReport,
    check_eaf_orientation,
    load_eaf_reference,
    select_rows,
    site_hash,
)
from opengwasdb.model.analyses import read_analyses
from opengwasdb.model.enums import EafOrientationOutcome, EafScope
from opengwasdb.query import query_store
from opengwasdb.store.open import open_store
from opengwasdb.variants import iter_variant_records, variant_table_path

log = logging.getLogger(__name__)

__all__ = ["AuditResult", "audit_eaf_orientation"]


@dataclass(frozen=True)
class AuditResult:
    """The freshly computed report, next to what the store already claimed.

    `disagreements` names the Analyses whose recorded outcome this audit
    *contradicts* — a store that recorded `passed` where the audit reads
    `failed`, or the reverse. An audit that could not verify what the store
    verified is not a contradiction: it means this panel had too little to say
    about that Analysis, which is a fact about the panel, not about the store.
    """

    report: EafOrientationReport
    recorded: dict[str, str]
    disagreements: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.report.failures and not self.disagreements


def _axis_hashes(store_path: Path) -> np.ndarray:
    """`select_sites`' hash per `variant_index`, for the whole variant axis.

    Streamed rather than materialised as ALIDs: the store this most needs to
    run on has twenty million variants, and only the sampled ones are ever
    needed as strings.
    """
    table_path = variant_table_path(store_path)
    hashes: dict[int, int] = {
        record.variant_index: site_hash(record.alid)
        for record in iter_variant_records(table_path)
    }
    out = np.zeros(max(hashes) + 1 if hashes else 0, dtype=np.uint64)
    for index, value in hashes.items():
        out[index] = value
    return out


def _alids_at(store_path: Path, rows: set[int]) -> dict[int, str]:
    """`{variant_index: alid}` for the rows the per-Analysis samples chose."""
    return {
        record.variant_index: record.alid
        for record in iter_variant_records(variant_table_path(store_path))
        if record.variant_index in rows
    }


def audit_eaf_orientation(
    store_path: str | Path,
    reference_path: str | Path,
    *,
    ancestry: str | None = None,
    n_sites: int = DEFAULT_SAMPLE_SITES,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
    min_variance: float = DEFAULT_MIN_VARIANCE,
) -> AuditResult:
    """Correlate each Analysis's *stored* EAF against `reference_path`.

    Only Analyses whose `eaf_scope` is `association` are audited: one that
    stores no frequency has no orientation, and reporting `unverified` for it
    would bury the Analyses that genuinely could not be checked.
    """
    path = Path(store_path)
    store = open_store(path)
    table = read_analyses(store.analyses_path)
    analysis_ids = [
        row["analysis_id"]
        for row in table.rows
        if row.get("eaf_scope", "") == EafScope.ASSOCIATION.value
    ]
    if not analysis_ids:
        log.warning("No analysis in %s stores per-association EAF; nothing to audit", path)

    # Sampled per Analysis, from each Analysis's own EAF-carrying variants —
    # the same rule the builders apply, and for the same reason: an Analysis
    # covering a fraction of a percent of the axis would barely appear in a
    # sample of the axis.
    hashes = _axis_hashes(path)
    query = query_store(path)
    sampled: dict[str, dict[int, float]] = {}
    for analysis_id in analysis_ids:
        result = query.analysis(analysis_id, observed_only=True)
        rows = np.asarray(result["variant_index"], dtype=np.int64)
        eaf = np.asarray(result["eaf"], dtype=np.float64)
        finite = np.isfinite(eaf)
        chosen = select_rows(hashes, rows[finite], k=n_sites)
        by_row = dict(zip(rows[finite].tolist(), eaf[finite].tolist(), strict=True))
        sampled[analysis_id] = {int(row): by_row[row] for row in chosen.tolist()}

    alid_by_row = _alids_at(path, {row for s in sampled.values() for row in s})
    observations = {
        analysis_id: {alid_by_row[row]: value for row, value in rows.items()}
        for analysis_id, rows in sampled.items()
    }
    sites = sorted({alid for observed in observations.values() for alid in observed})
    reference = load_eaf_reference(reference_path, sites, ancestry=ancestry)

    report = check_eaf_orientation(
        observations,
        reference=reference,
        n_sites=len(sites),
        min_overlap=min_overlap,
        min_variance=min_variance,
    )

    recorded = {
        row["analysis_id"]: row.get("eaf_orientation", "")
        for row in table.rows
        if row["analysis_id"] in set(analysis_ids)
    }
    conclusive = {EafOrientationOutcome.PASSED.value, EafOrientationOutcome.FAILED.value}
    disagreements = tuple(
        evidence.analysis_id
        for evidence in report.evidence
        if evidence.outcome.value in conclusive
        and recorded.get(evidence.analysis_id, "") in conclusive
        and recorded[evidence.analysis_id] != evidence.outcome.value
    )
    return AuditResult(report=report, recorded=recorded, disagreements=disagreements)
