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
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import log_ndtr


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
        "neglog10_p": float(-(log_ndtr(-abs(z)) + np.log(2.0)) / np.log(10.0)),
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
