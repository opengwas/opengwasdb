"""Report helpers shared by the full-scale Dense and Hybrid benchmark harnesses.

`benchmark_finngen_dense.py` (OGS-00016, Dense) and
`benchmark_ogs00011_hybrid.py` (OGS-00011, Hybrid) report many of the same
facts about a Store Release: its Analysis table summary, its storage bytes
against the source files, its known-locus checks and its distance-clumped IVW
runs. Keeping the shared shape in one module is what stops two harnesses
disagreeing about what a field means -- and it is what lets both be committed
without one being a near-copy of the other (issue #252 review round 1).

The harness-specific half stays with each harness: which store, which loci,
which MR pairs and which extra fields its report carries. The functions here
return the common part and take an `extra` mapping for a harness's own fields.

The MR half is here for the same reason: `ivw` is the one IVW estimate and
`mr_strata` the one split of a run into observed/imputed strata, so a
harness reporting a stratum beside its whole-set estimate cannot accidentally
use two conventions for the same quantity.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import erfc, log_ndtr

# A stratum of an MR run is only estimated when it holds at least this many
# instruments: a two-instrument "estimate" is the ratio of two ratios, and
# reporting it beside a stratum of ninety invites reading it as one.
MIN_STRATUM_INSTRUMENTS = 3

# The observed/imputed strata an MR run is split into, by the Association
# Status of the exposure cell and of the outcome cell of each instrument.
MR_STRATA = ("both_observed", "both_imputed", "mixed")


def analysis_rows(store: Path) -> list[dict[str, str]]:
    """Every row of a release's `analyses.tsv`, as its own columns."""
    with open(store / "analyses.tsv", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def numeric_column(rows: list[dict[str, str]], column: str) -> np.ndarray:
    """One column as float, skipping rows that leave it blank."""
    return np.array([float(row[column]) for row in rows if row[column] != ""])


def tally(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    """Value counts of one column, most common first, blanks as `(blank)`."""
    counts: dict[str, int] = {}
    for row in rows:
        value = row[column] or "(blank)"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def hits_summary(hits: np.ndarray) -> dict[str, float]:
    """Median/max/zero/total for a per-Analysis hit count."""
    return {
        "median": float(np.median(hits)),
        "max": float(hits.max()),
        "zero": int((hits == 0).sum()),
        "total": float(hits.sum()),
    }


def storage_summary(
    *,
    store_bytes: int,
    source_bytes: int,
    n_source_files: int,
    components: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The source-vs-store footprint every harness reports, plus `extra` keys."""
    out: dict[str, Any] = {
        "store_bytes": store_bytes,
        "store_gb": round(store_bytes / 1e9, 2),
        "source_bytes": source_bytes,
        "source_gb": round(source_bytes / 1e9, 2),
        "n_source_files": n_source_files,
        "compression_ratio": round(source_bytes / store_bytes, 2),
        "components": components,
    }
    if extra:
        out.update(extra)
    return out


def selection_summary(
    *,
    exposure: str,
    phewas_alid: str,
    region: tuple[str, int, int],
    random_lookup_shapes: list[dict[str, int]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The query selection every harness fixes, plus `extra` shapes of its own."""
    out: dict[str, Any] = {
        "exposure": exposure,
        "phewas_alid": phewas_alid,
        "region": {"chrom": region[0], "start": region[1], "end": region[2]},
        "random_lookup_shapes": random_lookup_shapes,
    }
    if extra:
        out.update(extra)
    return out


def neglog10_p(z: float) -> float:
    """-log10 of the two-sided p for a z, computed in the log domain.

    `1 - ndtr(|z|)` underflows to 0 at around z = 38, where this store's
    strongest hits live, and loses its significant digits long before that;
    `log_ndtr` keeps both. Shared so a regional profile and a locus check
    cannot disagree about what a p-value is.
    """
    return float(-(log_ndtr(-abs(z)) + np.log(2.0)) / np.log(10.0))


def oriented_locus(q: Any, alid: str, analysis_id: str, risk_allele: str) -> dict[str, Any] | None:
    """One locus lookup, oriented to its published risk allele, or None if absent.

    The signed effects are multiplied by +1 when the store's effect allele is
    the published risk allele and by -1 otherwise; `risk_allele_frequency` is
    the cohort frequency recast onto the risk allele the same way.
    """
    look = q.lookup([alid], [analysis_id])
    if not len(look["z"]):
        return None
    record = q._variant_axis.by_index(int(look["variant_index"][0]))
    z, se, eaf = float(look["z"][0]), float(look["se"][0]), float(look["eaf"][0])
    sign = 1.0 if record.effect_allele == risk_allele else -1.0
    return {
        "analysis_id": analysis_id,
        "alid": alid,
        "rsid": record.rsid,
        "effect_allele": record.effect_allele,
        "variant_index": int(record.variant_index),
        "z": z,
        "se": se,
        "beta_risk_allele": sign * z * se,
        "z_risk_allele": sign * z,
        "risk_allele_frequency": eaf if record.effect_allele == risk_allele else 1.0 - eaf,
        "neglog10_p": neglog10_p(z),
    }


def ivw(instruments: list[dict[str, Any]]) -> dict[str, float]:
    """Inverse-variance-weighted estimate over `instruments`, as the full-scale MR reports it.

    beta = sum(bx*by/sy^2) / sum(bx^2/sy^2) and se_fixed = sqrt(1 / sum(bx^2/sy^2)).
    `se` is the multiplicative random-effects SE, se_fixed * max(1, sqrt(Q/df))
    with Cochran's Q over the instruments' ratio estimates, and `p` is the
    two-sided normal p from it: the convention `benchmark_finngen_dense.run_mr`
    reports as `ivw_se`/`ivw_pval`, so this store's MR reads beside the other
    full-scale reports. One function for a stratum and for the whole set keeps
    the two comparable.

    `instruments` carry `beta_exp` and `beta_out`/`se_out`; an empty list has
    no estimate and raises rather than returning a NaN a report would print.
    """
    if not instruments:
        raise ValueError("ivw needs at least one instrument")
    bx = np.array([float(i["beta_exp"]) for i in instruments])
    by = np.array([float(i["beta_out"]) for i in instruments])
    sy = np.array([float(i["se_out"]) for i in instruments])
    precision = bx**2 / sy**2
    beta = float(np.sum(bx * by / sy**2) / np.sum(precision))
    se_fixed = float(np.sqrt(1.0 / np.sum(precision)))
    q_stat = float(np.sum(precision * (by / bx - beta) ** 2))
    df = len(instruments) - 1
    se = se_fixed * max(1.0, float(np.sqrt(q_stat / df))) if df > 0 else se_fixed
    return {
        "beta": beta,
        "se": se,
        "se_fixed": se_fixed,
        "p": float(erfc(abs(beta / se) / np.sqrt(2.0))),
        "cochran_q": q_stat,
        "q_df": df,
        "n": len(instruments),
    }


def instrument_stratum(status_exp: str, status_out: str) -> str | None:
    """Which observed/imputed stratum one instrument belongs to, if any.

    An `observed`/`imputed` pair is `mixed`; `observed`/`observed` and
    `imputed`/`imputed` are their own strata. Anything else -- a `missing`
    cell, or a status this build does not know -- belongs to no stratum and is
    reported only in the whole-set estimate.
    """
    if status_exp == status_out == "observed":
        return "both_observed"
    if status_exp == status_out == "imputed":
        return "both_imputed"
    if {status_exp, status_out} == {"observed", "imputed"}:
        return "mixed"
    return None


def default_mr_estimate(
    instruments: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], dict[str, float]]:
    """The instrument filter, instruments and IVW estimate an MR reports by default.

    On a Reference-Completed Store an imputed cell's z is shrunk towards zero, so
    an instrument with one understates its effect. The default is therefore the
    `both_observed` instruments; only when they number fewer than
    `MIN_STRATUM_INSTRUMENTS` does it fall back to `all`, and the filter says so.
    """
    observed = [
        i for i in instruments
        if instrument_stratum(str(i["status_exp"]), str(i["status_out"])) == "both_observed"
    ]
    if len(observed) >= MIN_STRATUM_INSTRUMENTS:
        return "both_observed", observed, ivw(observed)
    return "all", instruments, ivw(instruments)


def mr_strata(instruments: list[dict[str, Any]]) -> dict[str, dict[str, float] | None]:
    """The IVW estimate of each observed/imputed stratum, or None when too small.

    The same `ivw` as the whole-set estimate, over each instrument's exposure
    and outcome Association Status. A stratum holding fewer than
    `MIN_STRATUM_INSTRUMENTS` is `None` rather than a wide estimate that would
    be read as a result.
    """
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in MR_STRATA}
    for instrument in instruments:
        name = instrument_stratum(
            str(instrument["status_exp"]), str(instrument["status_out"])
        )
        if name is not None:
            buckets[name].append(instrument)
    return {
        name: (ivw(bucket) if len(bucket) >= MIN_STRATUM_INSTRUMENTS else None)
        for name, bucket in buckets.items()
    }


def phewas_top(
    q: Any, table: dict[int, dict[str, Any]], variants: list[tuple[str, str, str]], n: int = 8
) -> list[dict[str, Any]]:
    """The strongest `n` Analyses for each of `variants`, by |z|."""
    out = []
    for alid, rsid, gene in variants:
        result = q.phewas(alid)
        order = np.argsort(-np.abs(result["z"]))[:n]
        out.append(
            {
                "alid": alid,
                "rsid": rsid,
                "gene": gene,
                "n_analyses": int(len(result["z"])),
                "n_genome_wide": int((np.abs(result["z"]) > 5.4520).sum()),
                "top": [
                    {
                        "analysis_id": table[int(result["analysis_index"][i])]["analysis_id"],
                        "label": table[int(result["analysis_index"][i])]["analysis_label"],
                        "z": float(result["z"][i]),
                    }
                    for i in order
                ],
            }
        )
    return out
