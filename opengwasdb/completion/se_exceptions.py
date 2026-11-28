"""The share of SEs a Reference Completion holds as exact exceptions, per Analysis (issue #301).

A residual SE plane's 2% exception budget is the criterion a build uses to
choose its range. Completion keeps its source's range and model, so it measures
the share it produced rather than enforcing it: the counts are taken as each
region or band of codes is written, and the record in
`provenance.completion.se_exceptions` and the logged warning are how a breach by
imputed cells becomes visible. Ragged completion has recorded it since #301;
Dense and Hybrid completion do too (#390), since they carry observed SEs exactly
and so can hold more exceptions than their source did.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from opengwasdb.encoding import SE_EXCEPTION_BUDGET
from opengwasdb.encoding.se_counts import SeExceptionCounts

log = logging.getLogger(__name__)


def se_exception_record(counts: SeExceptionCounts, analysis_ids: list[str]) -> dict[str, Any]:
    """Each Analysis's share of SEs held as exact exceptions, in `analyses.tsv`
    order, and the Analyses above the 2% budget (spec §6a, issue #301).

    The budget is a build's criterion for choosing a residual range. Completion
    keeps its source's range and model, so it measures the share it produced
    rather than enforcing it: the record and `report_se_exceptions` are how a
    breach by imputed cells becomes visible. An Analysis with no SE has no
    share, recorded as null rather than 0. The writer counts as it codes each
    region (`SeExceptionCounts`).
    """
    n_se, n_exceptions = counts.n_se.tolist(), counts.n_exceptions.tolist()
    rates = [exact / n if n else None for exact, n in zip(n_exceptions, n_se, strict=True)]
    analyses = [
        {
            "analysis_id": analysis_id,
            "n_se": int(n_se[i]),
            "n_exceptions": int(n_exceptions[i]),
            "n_imputed_se": int(counts.n_imputed_se[i]),
            "n_imputed_exceptions": int(counts.n_imputed_exceptions[i]),
            "exception_rate": rates[i],
        }
        for i, analysis_id in enumerate(analysis_ids)
    ]
    over = [
        analysis_id
        for analysis_id, rate in zip(analysis_ids, rates, strict=True)
        if rate is not None and rate > SE_EXCEPTION_BUDGET
    ]
    return {"budget": SE_EXCEPTION_BUDGET, "analyses": analyses, "over_budget": over}


def corrected_se_exception_record(
    record: dict[str, Any], removed: SeExceptionCounts, added: SeExceptionCounts
) -> dict[str, Any]:
    """`record` less the cells a Hybrid crossover fold overwrote and plus the cells it
    wrote, so it counts the release's final cells, as #362 made the imputed totals do.

    The Dense pass counted what it wrote; the fold then replaces some of those cells
    with the Ragged Overflow's observations, coded against the Dense Component's own
    baseline and model (#390).
    """
    entries = record["analyses"]
    counts = SeExceptionCounts(
        *(
            np.array([entry[key] for entry in entries], dtype=np.int64)
            for key in ("n_se", "n_exceptions", "n_imputed_se", "n_imputed_exceptions")
        )
    )
    counts.merge(added)
    counts.merge(removed, sign=-1)
    return se_exception_record(counts, [entry["analysis_id"] for entry in entries])


def report_se_exceptions(record: dict[str, Any], *, where: str) -> None:
    """The exception tally on the console, and each Analysis over budget by name."""
    analyses = record["analyses"]
    total = sum(entry["n_exceptions"] for entry in analyses)
    imputed = sum(entry["n_imputed_exceptions"] for entry in analyses)
    cells = sum(entry["n_se"] for entry in analyses)
    print(
        f"SE exceptions: {total:,} of {cells:,} SEs ({imputed:,} on imputed cells); "
        f"{len(record['over_budget']):,} Analyses over the {record['budget']:.0%} budget"
    )
    over = [entry for entry in analyses if entry["analysis_id"] in set(record["over_budget"])]
    if not over:
        return
    log.warning(
        "%s: %s Analyses hold more than %s of their SEs as exact exceptions, "
        "the residual budget (spec §6a); the source's SE model and range are kept, never "
        "refitted or narrowed to float16 (issue #301): %s",
        where,
        f"{len(over):,}",
        f"{record['budget']:.0%}",
        "; ".join(
            f"{entry['analysis_id']}: {entry['exception_rate']:.2%} "
            f"({entry['n_exceptions']:,} of {entry['n_se']:,}, "
            f"{entry['n_imputed_exceptions']:,} imputed)"
            for entry in over
        ),
    )
