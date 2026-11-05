"""plink2 clumping of a Top-Hit Query's rows on the genotype panel.

plink2 ranks on p, and a float64 two-sided p is 0 from |z| 38.50 (`p_value_from_z`),
so it is given -log10 p computed from z (`--clump-log10 input-only`), which
stays finite.

Equal values are the harder problem. Stored z is quantised, so most rows of a
real Analysis share their abs_z with another (53,760 of 55,460 for
finngen-r13-BMI_IRN). plink2 breaks a tie by panel order whatever order its
input is in, while the rule breaks it by Store Variant Index. So equal values
are separated in the rule's order: each row of a run of equal values sits
`TIE_STEP_ULPS` ulps below the one before it. A step of 1 ulp did not survive
plink2's arithmetic (102 of 384 trials on the EUR panel's chr22 kept panel
order), 2 ulps failed 9 times, and 3 or more never did; 16 leaves a margin, and
moves no value by more than 4e-11 of itself for runs below 10,000 rows. Whether
plink2 honoured the order is checked afterwards, on every clump
(`check_clumps_follow_rule_order`), rather than trusted.
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opengwasdb.stats import log10_p_two_sided

from .panel import Panel, plink2
from .rule import rule_order

TIE_STEP_ULPS = 16
CLUMP_KB = 10_000
INDEX_P = 5e-8
# Left to itself plink2 reserves half the node's RAM (515 GB here) for every run.
PLINK2_MEMORY_MIB = 4096


class ClumpError(RuntimeError):
    """plink2's clumps do not partition the rows they were given."""


@dataclass(frozen=True)
class RankedScores:
    """-log10 p per row, strictly decreasing in the rule's order."""

    neg_log10_p: np.ndarray
    tied_rows: int  # rows sharing their value with another row
    tied_groups: int
    largest_group: int
    largest_shift: float  # the most any value moved


def neg_log10_p(z: np.ndarray) -> np.ndarray:
    result: np.ndarray = -log10_p_two_sided(np.asarray(z, dtype="float64"))
    return result


def ranked_neg_log10_p(abs_z: np.ndarray, variant_index: np.ndarray) -> RankedScores:
    """-log10 p from abs_z, with equal values separated in the rule's order."""
    order = rule_order(abs_z, variant_index)
    value = neg_log10_p(np.asarray(abs_z))[order]
    n = value.size
    starts = np.flatnonzero(np.r_[True, value[1:] != value[:-1]]) if n else np.array([], int)
    sizes = np.diff(np.r_[starts, n])
    rank = np.arange(n) - np.repeat(starts, sizes)
    separated = value - rank * TIE_STEP_ULPS * np.spacing(value)
    if np.any(np.diff(separated) >= 0):
        raise ClumpError("separating equal -log10 p reordered rows; the step is too large")
    scores = np.empty(n)
    scores[order] = separated
    tied = sizes[sizes > 1]
    return RankedScores(scores, int(tied.sum()), int(tied.size), int(sizes.max(initial=0)),
                        float(np.max(value - separated, initial=0.0)))


def clump_rows(
    panel: Panel, chromosome: np.ndarray, panel_id: np.ndarray, scores: np.ndarray, *,
    r2: float, work_dir: Path, kb: int = CLUMP_KB, p1: float = INDEX_P, p2: float = INDEX_P,
    threads: int = 1,
) -> np.ndarray:
    """For each row, the row of the index variant whose clump holds it.

    One plink2 run per chromosome. Every row must come back exactly once, as an
    index variant or as a member, or the clumping is refused.
    """
    if np.any(scores <= -math.log10(p2)):
        raise ClumpError(f"rows at or above p2 = {p2:g} would be left out of every clump")
    index_of = np.full(len(scores), -1, dtype="int64")
    for label in dict.fromkeys(np.asarray(chromosome).tolist()):
        rows = np.flatnonzero(np.asarray(chromosome) == label)
        clumps = run_clump(panel.pfile(label), panel_id[rows], scores[rows], log10=True, r2=r2,
                           out=work_dir / f"chr{label}.r2_{r2:g}", kb=kb, p1=p1, p2=p2,
                           threads=threads)
        _assign(index_of, dict(zip(panel_id[rows].tolist(), rows.tolist(), strict=True)), clumps)
    unassigned = np.flatnonzero(index_of < 0)
    if unassigned.size:
        raise ClumpError(f"{unassigned.size:,} rows came back in no clump, e.g. "
                         f"{panel_id[unassigned[0]]}")
    return index_of


def check_clumps_follow_rule_order(index_of: np.ndarray, abs_z: np.ndarray,
                                   variant_index: np.ndarray) -> None:
    """Refuse clumps whose index variant comes after one of its members in the rule's order."""
    rank = np.empty(index_of.size, dtype="int64")
    rank[rule_order(abs_z, variant_index)] = np.arange(index_of.size)
    behind = np.flatnonzero(rank[index_of] > rank)
    if behind.size:
        raise ClumpError(f"{behind.size:,} clump members outrank their index variant in the "
                         "rule's order; plink2 did not rank as the rule does")


def run_clump(
    pfile: Path, variant_id: Sequence[str] | np.ndarray, p_column: np.ndarray, *, log10: bool,
    r2: float, out: Path, kb: int = CLUMP_KB, p1: float = INDEX_P, p2: float = INDEX_P,
    threads: int = 1,
) -> dict[str, list[str]]:
    """One plink2 --clump run: each index variant's ID, mapped to its members' IDs."""
    field = "NEG_LOG10_P" if log10 else "P"
    source = out.with_name(out.name + ".input.tsv")
    source.write_text(f"ID\t{field}\n" + "".join(
        f"{name}\t{float(value)!r}\n" for name, value in zip(variant_id, p_column, strict=True)))
    command = [plink2(), "--pfile", str(pfile), "vzs", "--clump", str(source),
               "--clump-id-field", "ID", "--clump-p-field", field, "--clump-p1", f"{p1:g}",
               "--clump-p2", f"{p2:g}", "--clump-r2", f"{r2:g}", "--clump-kb", str(kb),
               "--clump-unphased", "--threads", str(threads), "--memory", str(PLINK2_MEMORY_MIB),
               "--out", str(out)]
    if log10:
        command += ["--clump-log10", "input-only"]
    subprocess.run(command, check=True, capture_output=True)
    _refuse_warnings(out.with_name(out.name + ".log"))
    return _read_clumps(out.with_name(out.name + ".clumps"))


def _assign(index_of: np.ndarray, row_of: dict[str, int], clumps: dict[str, list[str]]) -> None:
    for index_id, members in clumps.items():
        for member in [index_id, *members]:
            row = row_of[member]
            if index_of[row] >= 0:
                raise ClumpError(f"{member} is in two clumps")
            index_of[row] = row_of[index_id]


def _read_clumps(path: Path) -> dict[str, list[str]]:
    lines = path.read_text().splitlines()
    header = lines[0].lstrip("#").split("\t")
    at_id, at_members = header.index("ID"), header.index("SP2")
    clumps = {}
    for line in lines[1:]:
        cells = line.split("\t")
        members = cells[at_members]
        clumps[cells[at_id]] = [] if members == "." else members.split(",")
    return clumps


def _refuse_warnings(log: Path) -> None:
    warnings = [line for line in log.read_text().splitlines() if line.startswith("Warning")]
    if warnings:
        raise ClumpError(f"{log}: plink2 warned: {' | '.join(warnings)}")
