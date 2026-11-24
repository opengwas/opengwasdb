"""Resolve index-keyed query results to human-readable rows (issue #104).

Every association-returning method on `StoreQuery`/`RaggedStoreQuery`/
`HybridStoreQuery` (`opengwasdb.store_reader.facade`) returns index-keyed parallel
arrays (ADR-0020) -- `variant_index`/`analysis_index` rather than variant/
analysis identity. `resolve_rows()` is the join back to identity.

chromosome/position/effect_allele/other_allele/alid come from
`variant_axis.identity_by_indices()` -- an ALID *is*
`chromosome:position:effect_allele:other_allele`, and that method recovers
it from the store's already-loaded mmap'd ALID search index with zero
`variants.tsv.gz` I/O. rsid is the one identity field that index can't
produce (it isn't part of the alid), so it's resolved via
`variant_axis.rsids_by_indices()`, and only when `include_variant_info=True`;
so is `source_alid`, the row's pre-liftover ALID (`source_alids_by_indices`, #470).
On 0.3.0 those invert the rsID index, sized to the rows asked, and slice the
`source_alid` sidecar, and read no table (#469, #470). On the 0.2 series they read
the table's rows (`by_indices`, which
adaptively picks a bounded random-access resolution or one sequential scan),
the one part of this join that can cost anywhere from milliseconds to tens of
seconds on a large dense query. `eaf` always rides the row: it is already
materialised in the query result the facade returned, so resolving it costs
no extra I/O and hiding it saved nothing (issue #136). It rides `result`
rather than the variant table, so a store that carries none yields `.` per
cell rather than paying for a lookup. The
analysis side comes from `analyses.all()`, already fully loaded at
store-open (ADR 0030, no extra I/O). Rows are yielded lazily, so the CLI's
CSV writer streams rather than materialising the whole resolved table. `n`
rides the row the same way, but only from a result that carries it -- a Ragged
release with a per-association sample-size plane (ADR 0067).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np

from opengwasdb.store_reader.analyses import AnalysesIndex
from opengwasdb.store_reader.associations.record import statistics_of
from opengwasdb.store_reader.panel_axis import AnyVariantAxis


def resolve_rows(
    analyses: AnalysesIndex,
    variant_axis: AnyVariantAxis,
    result: Mapping[str, np.ndarray],
    *,
    include_variant_info: bool = False,
) -> Iterator[dict[str, object]]:
    """Yield one human-readable row per association in `result`.

    Each row carries analysis_id/analysis_label, variant identity
    (chromosome/position/alid/alleles, plus rsid and source_alid when
    `include_variant_info=True` -- "." when the store has none for that
    association, matching the Store Variant Table's own missing marker), eaf,
    z, se, beta, pval, log10_p, and association_status. beta, pval and log10_p
    are the query record's own (`statistics_of`, ADR 0104): NaN for a missing
    association or an infinite SE, and `pval` 0.0 where p is below float64's
    range while `log10_p` stays exact. `rsid` is omitted from the row
    entirely when `include_variant_info=False` (the default) rather than
    filled with a placeholder: resolving it is the expensive part of this
    join at scale. `eaf` is always present -- it is already in `result`, so
    resolving it costs nothing and a missing cell is the same "." (#136).
    """
    analysis_rows = analyses.all()
    variant_index = result["variant_index"]
    derived = statistics_of(result)

    identity = variant_axis.identity_by_indices(variant_index)
    info = _variant_info(variant_axis, variant_index) if include_variant_info else {}

    for i in range(len(variant_index)):
        analysis = analysis_rows[int(result["analysis_index"][i])]
        row: dict[str, object] = {
            "analysis_id": analysis["analysis_id"],
            "analysis_label": analysis.get("analysis_label", ""),
            **_variant_identity(identity, i),
            "z": float(result["z"][i]),
            "se": float(result["se"][i]),
            "beta": float(derived.beta[i]),
            "pval": float(derived.pval[i]),
            "log10_p": float(derived.log10_p[i]),
            "eaf": _eaf_or_missing(result, i),
            "association_status": str(result["association_status"][i]),
        }
        if "n" in result:
            row["n"] = _n_or_missing(result["n"], i)
        row.update({name: values[i] or "." for name, values in info.items()})
        yield row


def _variant_info(
    variant_axis: AnyVariantAxis, variant_index: np.ndarray
) -> dict[str, np.ndarray]:
    """The variant info a row resolved with it carries: each row's rsID and `source_alid`,
    None where it has none, which the row spells ``.``."""
    return {
        "rsid": variant_axis.rsids_by_indices(variant_index),
        "source_alid": variant_axis.source_alids_by_indices(variant_index),
    }


def _variant_identity(identity: dict[str, np.ndarray], i: int) -> dict[str, object]:
    """One row's chromosome, position, ALID and alleles, in the row's column order,
    from the ALID index (`VariantAxis.identity_by_indices`)."""
    return {
        "chromosome": identity["chromosome"][i],
        "position": int(identity["position"][i]),
        "alid": identity["alid"][i],
        "effect_allele": identity["effect_allele"][i],
        "other_allele": identity["other_allele"][i],
    }


def _n_or_missing(n: np.ndarray, i: int) -> object:
    """This association's stored sample size as an `int`, or "." where it has none."""
    value = float(n[i])
    return "." if not np.isfinite(value) else int(value)


def _eaf_or_missing(result: Mapping[str, np.ndarray], i: int) -> object:
    """This association's stored EAF, or "." (ADR 0036).

    NaN is the store's per-cell "no EAF here" marker, and `eaf` is absent
    from results produced before ADR 0036 landed; both resolve to the same
    "." the rest of this row uses for an unknown value, so a reader never has
    to tell "the store has no EAF array" apart from "this cell has none".
    """
    eaf = result.get("eaf")
    if eaf is None or i >= len(eaf):
        return "."
    value = float(eaf[i])
    return "." if not np.isfinite(value) else value
