"""Resolve index-keyed query results to human-readable rows (issue #104).

Every association-returning method on `StoreQuery`/`RaggedStoreQuery`/
`HybridStoreQuery` (`opengwasdb.query.facade`) returns index-keyed parallel
arrays (ADR-0020) -- `variant_index`/`analysis_index` rather than variant/
analysis identity. `resolve_rows()` is the join back to identity.

chromosome/position/effect_allele/other_allele/alid come from
`variant_axis.identity_by_indices()` -- an ALID *is*
`chromosome:position:effect_allele:other_allele`, and that method recovers
it from the store's already-loaded mmap'd ALID search index with zero
`variants.tsv.gz` I/O. rsid is the one identity field that index can't
produce today (it isn't part of the alid), so it's resolved via
`variant_axis.by_indices()` -- which itself adaptively picks a bounded
random-access resolution or, once the request touches a large-enough share
of the Store Variant Table, one sequential scan instead (see that method's
docstring) -- and only when `include_variant_info=True`, since that's the
one part of this join that can cost anywhere from milliseconds to tens of
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

from collections.abc import Iterator

import numpy as np

from opengwasdb.index import AnalysesIndex
from opengwasdb.stats import log10_p_two_sided
from opengwasdb.variants import VariantAxis, VariantRecord


def resolve_rows(
    analyses: AnalysesIndex,
    variant_axis: VariantAxis,
    result: dict[str, np.ndarray],
    *,
    include_variant_info: bool = False,
) -> Iterator[dict[str, object]]:
    """Yield one human-readable row per association in `result`.

    Each row carries analysis_id/analysis_label, variant identity
    (chromosome/position/alid/alleles, plus rsid when
    `include_variant_info=True` -- "." when the store has none for that
    association, matching the Store Variant Table's own missing marker), eaf,
    z, se, log10_p, and association_status. `rsid` is omitted from the row
    entirely when `include_variant_info=False` (the default) rather than
    filled with a placeholder: resolving it is the expensive part of this
    join at scale. `eaf` is always present -- it is already in `result`, so
    resolving it costs nothing and a missing cell is the same "." (#136).
    """
    analysis_rows = analyses.all()
    variant_index = result["variant_index"]
    log10_p = log10_p_two_sided(result["z"])

    identity = variant_axis.identity_by_indices(variant_index)
    variant_rows = (
        variant_axis.by_indices(variant_index)
        if include_variant_info or identity is None
        else None
    )

    for i in range(len(variant_index)):
        vi = int(variant_index[i])
        analysis = analysis_rows[int(result["analysis_index"][i])]
        row: dict[str, object] = {
            "analysis_id": analysis["analysis_id"],
            "analysis_label": analysis.get("analysis_label", ""),
            **_variant_identity(identity, variant_rows, i, vi),
            "z": float(result["z"][i]),
            "se": float(result["se"][i]),
            "log10_p": float(log10_p[i]),
            "eaf": _eaf_or_missing(result, i),
            "association_status": str(result["association_status"][i]),
        }
        if "n" in result:
            row["n"] = _n_or_missing(result["n"], i)
        if include_variant_info:
            assert variant_rows is not None
            rsid_record = variant_rows.get(vi)
            row["rsid"] = (rsid_record.rsid or ".") if rsid_record is not None else "."
        yield row


def _variant_identity(
    identity: dict[str, np.ndarray] | None,
    variant_rows: dict[int, VariantRecord] | None,
    i: int,
    vi: int,
) -> dict[str, object]:
    """One row's chromosome, position, ALID and alleles, in the row's column order.

    From the ALID search index when it could answer (`identity`), otherwise from
    the Store Variant Table rows `resolve_rows` already fetched.
    """
    if identity is not None:
        return {
            "chromosome": identity["chromosome"][i],
            "position": int(identity["position"][i]),
            "alid": identity["alid"][i],
            "effect_allele": identity["effect_allele"][i],
            "other_allele": identity["other_allele"][i],
        }
    assert variant_rows is not None
    variant = variant_rows[vi]
    return {
        "chromosome": variant.chromosome,
        "position": variant.position,
        "alid": variant.alid,
        "effect_allele": variant.effect_allele,
        "other_allele": variant.other_allele,
    }


def _n_or_missing(n: np.ndarray, i: int) -> object:
    """This association's stored sample size as an `int`, or "." where it has none."""
    value = float(n[i])
    return "." if not np.isfinite(value) else int(value)


def _eaf_or_missing(result: dict[str, np.ndarray], i: int) -> object:
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
