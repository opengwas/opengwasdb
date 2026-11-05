"""Linkage disequilibrium between named panel variants, from exported dosages.

`pair_r2` is the squared Pearson correlation of two variants' allele counts
over the samples where both are called, which is what plink2's
`--r2-unphased` computes. The sweep exports the variants it needs once per
Analysis (`plink2 --export A`) and computes every r^2 here.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .clump import PLINK2_MEMORY_MIB
from .panel import Panel, plink2


def export_dosages(panel: Panel, chromosome: str, variant_ids: list[str], work_dir: Path,
                   threads: int = 1) -> np.ndarray:
    """Allele counts (variants x samples) in the order of `variant_ids`; NaN where not called."""
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"chr{chromosome}.dosage"
    wanted = out.with_name(out.name + ".ids")
    wanted.write_text("\n".join(variant_ids) + "\n")
    subprocess.run([plink2(), "--pfile", str(panel.pfile(chromosome)), "vzs",
                    "--extract", str(wanted), "--export", "A", "--threads", str(threads),
                    "--memory", str(PLINK2_MEMORY_MIB), "--out", str(out)],
                   check=True, capture_output=True)
    frame = pd.read_csv(io.StringIO(out.with_name(out.name + ".raw").read_text()), sep=r"\s+",
                        na_values=["NA"])
    counts = frame.iloc[:, 6:]
    named = {column.rsplit("_", 1)[0]: i for i, column in enumerate(counts.columns)}
    if len(named) != len(variant_ids) or set(named) != set(variant_ids):
        raise ValueError(f"chr{chromosome}: plink2 exported {len(named)} of "
                         f"{len(set(variant_ids))} variants asked for")
    matrix = counts.to_numpy(dtype="float64").T
    ordered: np.ndarray = matrix[[named[name] for name in variant_ids]]
    return ordered


def pair_r2(first: np.ndarray, second: np.ndarray) -> float:
    """r^2 over the samples called in both; NaN when fewer than 2 or either is constant."""
    both = ~(np.isnan(first) | np.isnan(second))
    if both.sum() < 2:
        return float("nan")
    x, y = first[both] - first[both].mean(), second[both] - second[both].mean()
    denominator = float(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) ** 2 / denominator) if denominator > 0 else float("nan")


def r2_matrix(dosage: np.ndarray) -> np.ndarray:
    """Every pair's r^2 over the samples both are called in (pairwise complete)."""
    called = (~np.isnan(dosage)).astype("float64")
    filled = np.nan_to_num(dosage)
    n = called @ called.T
    sx, sxx = filled @ called.T, (filled * filled) @ called.T
    sxy = filled @ filled.T
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance = sxy - sx * sx.T / n
        variance = sxx - sx * sx / n
        result = covariance ** 2 / (variance * variance.T)
    return np.asarray(result)
