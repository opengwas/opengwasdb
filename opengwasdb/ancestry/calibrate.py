"""Calibrate the admission gates against the Reported Population (issue 064).

The **Reported Population** never routes (ADR 0028); it only calibrates and audits.
This module cross-tabulates Assigned vs Reported across a Catalogue, lists the
disagreements for human review, and re-applies a chosen τ/δ to **relabel** the
Catalogue directly from its stored summary statistics — no allele-frequency
re-extraction. A human inspects the cross-tab + disagreements and picks the
operating point; the chosen gates are recorded as Catalogue provenance.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from opengwasdb.ancestry.catalogue import UNASSIGNED
from opengwasdb.ancestry.mixture import Gates, apply_gates

# Free-text Reported Population → comparison category. Substring-matched, longest
# first, lower-cased. "Mixed"/"Unknown" are non-routing categories used only to
# check that admixed/undeclared studies fall to Unassigned.
_REPORTED_PATTERNS: list[tuple[str, str]] = [
    ("european", "EUR"),
    ("europe", "EUR"),
    ("white", "EUR"),
    ("caucasian", "EUR"),
    ("east asian", "EAS"),
    ("japanese", "EAS"),
    ("chinese", "EAS"),
    ("korean", "EAS"),
    ("south asian", "SAS"),
    ("indian", "SAS"),
    ("hispanic", "AMR"),
    ("latin", "AMR"),
    ("african american", "Mixed"),  # admixed → should not route
    ("african", "AFR"),
    ("mixed", "Mixed"),
    ("trans-ethnic", "Mixed"),
    ("trans ethnic", "Mixed"),
    ("admixed", "Mixed"),
    ("multi", "Mixed"),
]

NON_ROUTING = frozenset({"Mixed", "Unknown"})


def normalise_reported(reported: str) -> str:
    """Map a free-text Reported Population to a comparison category."""
    text = (reported or "").strip().lower()
    if not text:
        return "Unknown"
    for needle, category in _REPORTED_PATTERNS:
        if needle in text:
            return category
    return "Unknown"


def crosstab(rows: list[dict[str, str]]) -> Counter[tuple[str, str]]:
    """Count (Reported category, Assigned Ancestry) pairs across the Catalogue."""
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        reported = normalise_reported(row.get("reported_population", ""))
        assigned = row.get("assigned_ancestry", UNASSIGNED) or UNASSIGNED
        counts[(reported, assigned)] += 1
    return counts


@dataclass(frozen=True)
class Disagreement:
    trait_id: str
    reported_population: str
    reported_category: str
    assigned_ancestry: str
    dominant_superpop: str
    dominant_proportion: str
    runner_up_margin: str
    af_overlap: str
    gate_reason: str


def disagreements(rows: list[dict[str, str]]) -> list[Disagreement]:
    """Analyses whose concrete Reported ancestry conflicts with the Assigned one.

    Only rows with a routable Reported category (a super-population, not
    Mixed/Unknown) are audited; a conflict is any mismatch, including Unassigned.
    """
    out: list[Disagreement] = []
    for row in rows:
        category = normalise_reported(row.get("reported_population", ""))
        if category in NON_ROUTING:
            continue
        assigned = row.get("assigned_ancestry", UNASSIGNED) or UNASSIGNED
        if assigned == category:
            continue
        out.append(
            Disagreement(
                trait_id=row.get("trait_id", ""),
                reported_population=row.get("reported_population", ""),
                reported_category=category,
                assigned_ancestry=assigned,
                dominant_superpop=row.get("dominant_superpop", ""),
                dominant_proportion=row.get("dominant_proportion", ""),
                runner_up_margin=row.get("runner_up_margin", ""),
                af_overlap=row.get("af_overlap", ""),
                gate_reason=row.get("gate_reason", ""),
            )
        )
    return out


def relabel(rows: list[dict[str, str]], gates: Gates) -> list[dict[str, str]]:
    """Re-apply ``gates`` to each row from its stored stats (no AF re-extraction).

    Recomputes ``gate_reason`` and ``assigned_ancestry`` and stamps the new gate
    provenance columns. Returns fresh row dicts; input rows are not mutated.

    ``eaf_orientation_r`` is carried into the gates too (issue #115), so no τ/δ
    pick can re-admit an Analysis whose frequency column is reported against the
    other allele -- that verdict is not a threshold anyone is calibrating. A
    Catalogue written before the column existed has no value there, and
    ``apply_gates`` skips the gate rather than failing it, so those rows relabel
    exactly as they did before.
    """
    out: list[dict[str, str]] = []
    for row in rows:
        new = dict(row)
        overlap = int(row.get("af_overlap") or 0)
        residual = _to_float(row.get("nnls_residual"))
        dominant_proportion = _to_float(row.get("dominant_proportion"))
        margin = _to_float(row.get("runner_up_margin"))
        reason = apply_gates(
            gates,
            overlap=overlap,
            residual=residual,
            dominant_proportion=dominant_proportion,
            margin=margin,
            eaf_orientation_r=_to_float(row.get("eaf_orientation_r")),
        )
        dominant = row.get("dominant_superpop", "")
        new["gate_reason"] = reason
        new["assigned_ancestry"] = dominant if reason == "ok" and dominant else UNASSIGNED
        new["gate_tau"] = f"{gates.tau:.6g}"
        new["gate_delta"] = f"{gates.delta:.6g}"
        new["gate_n_min"] = str(gates.n_min)
        new["gate_residual_max"] = f"{gates.residual_max:.6g}"
        new["gate_orientation_flip_r"] = f"{gates.orientation_flip_r:.6g}"
        out.append(new)
    return out


def operating_point(rows: list[dict[str, str]]) -> dict[str, int]:
    """Summary counts for the operating point (reported-EUR admission, etc.)."""
    def _cat(row: dict[str, str]) -> str:
        return normalise_reported(row.get("reported_population", ""))

    reported_eur = [r for r in rows if _cat(r) == "EUR"]
    reported_mixed = [r for r in rows if _cat(r) == "Mixed"]
    eur_admitted = sum(1 for r in reported_eur if (r.get("assigned_ancestry") or "") == "EUR")
    mixed_unassigned = sum(
        1 for r in reported_mixed if (r.get("assigned_ancestry") or UNASSIGNED) == UNASSIGNED
    )
    return {
        "reported_eur": len(reported_eur),
        "reported_eur_admitted_eur": eur_admitted,
        "reported_mixed": len(reported_mixed),
        "reported_mixed_unassigned": mixed_unassigned,
    }


def format_crosstab(counts: Counter[tuple[str, str]]) -> str:
    """Render the cross-tab as a fixed-width text table (Reported × Assigned)."""
    reported = sorted({r for r, _a in counts})
    assigned = sorted({a for _r, a in counts})
    width = max([len("reported\\assigned")] + [len(r) for r in reported]) + 2
    col = max([len(a) for a in assigned] + [5]) + 2
    header = "reported\\assigned".ljust(width) + "".join(a.rjust(col) for a in assigned)
    lines = [header]
    for r in reported:
        cells = "".join(str(counts.get((r, a), 0)).rjust(col) for a in assigned)
        lines.append(r.ljust(width) + cells)
    return "\n".join(lines)


def write_disagreements(path: str | Path, items: list[Disagreement]) -> Path:
    """Write the disagreement report as a TSV for human review."""
    path = Path(path)
    fields = [
        "trait_id",
        "reported_population",
        "reported_category",
        "assigned_ancestry",
        "dominant_superpop",
        "dominant_proportion",
        "runner_up_margin",
        "af_overlap",
        "gate_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for item in items:
            writer.writerow(item.__dict__)
    return path


def write_rows(path: str | Path, rows: list[dict[str, str]], fieldnames: list[str]) -> Path:
    """Write raw Catalogue dict rows back out in the given column order.

    A key present in the rows but absent from `fieldnames` is appended rather
    than raising: `relabel` stamps the gate provenance columns that produced the
    new labels, and a Catalogue written before one of those gates existed has no
    column for it. Dropping it would leave the file claiming an operating point
    it was not relabelled at; raising would mean every new gate breaks
    recalibration of every older Catalogue (issue #115 added one).
    """
    path = Path(path)
    extra = [key for row in rows for key in row if key not in set(fieldnames)]
    ordered = [*fieldnames, *dict.fromkeys(extra)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _to_float(value: str | None) -> float:
    if value is None or value in ("", "nan"):
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")
