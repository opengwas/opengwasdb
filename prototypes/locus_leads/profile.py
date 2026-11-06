"""Locus boundaries from the association profile: peak prominence (#409).

Under one causal variant a neighbour's expected z is r x z_lead, so a peak's
shape is its LD profile scaled by the lead's strength. Two 5e-8 rows belong to
one locus unless the |z| profile between them falls into a valley.

The profile is the Analysis's 5e-4 Top-Hit tier, read as an envelope: the
largest |z| among the tier's rows in each bin of B bp, with bins fixed per
chromosome from position 0 (`position // B`). A bin with no tier row has
envelope 0: in a Dense Store a missing row means |z| below the tier's floor or a
missing cell, and the envelope cannot tell the two apart.

Candidates are taken in the Locus Lead ranking (`abs_z` descending, then Store
Variant Index ascending). A candidate joins the strongest accepted lead on its
chromosome, at most `MAX_SPAN_BP` away, whose col (the lowest envelope over the
bins from the lead's to the candidate's, inclusive) is above the candidate's
threshold; otherwise it becomes a lead. `DEFINITION` says so in the output.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field, fields

import numpy as np

from .rule import rule_order

# Clumping's window: no merge spans more than this.
MAX_SPAN_BP = 10_000_000

DEFINITION = {
    "profile": "the Analysis's observed rows in the 5e-4 Top-Hit tier, every one of them, "
               "including rows outside the shared set (off the panel, for example)",
    "envelope": "per chromosome, bins of B bp fixed from position 0 (bin = position // B); a "
                "bin's envelope is the largest |z| among the profile's rows in it, and 0 when "
                "it holds none",
    "col": "between two rows on one chromosome, the lowest envelope over the bins from one "
           "row's bin to the other's, both included",
    "order": "abs_z descending, then Store Variant Index ascending (the Locus Lead ranking)",
    "join": "a candidate joins the strongest (first accepted) lead on its chromosome at most "
            f"{MAX_SPAN_BP:,} bp away whose col is above the candidate's threshold; otherwise "
            "it becomes a lead",
    "relative": "threshold = f x the candidate's own |z|",
    "fixed": "threshold = T; with T = 0, a chain of bins that all hold a tier row",
}


@dataclass(frozen=True)
class Profile:
    """The rows that shape the envelope: parallel arrays, one entry per tier row."""

    chromosome: np.ndarray
    position: np.ndarray
    abs_z: np.ndarray

    def take(self, keep: np.ndarray) -> Profile:
        return Profile(*(getattr(self, f.name)[keep] for f in fields(self)))


@dataclass(frozen=True)
class Threshold:
    """What a col must exceed: f x the candidate's |z| (relative), or T (fixed)."""

    kind: str
    value: float

    def __post_init__(self) -> None:
        if self.kind == "relative" and not 0 < self.value < 1:
            raise ValueError(f"a relative threshold needs 0 < f < 1, not {self.value}")
        if self.kind == "fixed" and self.value < 0:
            raise ValueError(f"a fixed threshold needs T >= 0, not {self.value}")
        if self.kind not in ("relative", "fixed"):
            raise ValueError(f"a threshold is relative or fixed, not {self.kind!r}")

    def at(self, abs_z: np.ndarray) -> np.ndarray:
        """Each candidate's threshold."""
        if self.kind == "relative":
            return np.asarray(self.value * abs_z, dtype="float64")
        return np.full(abs_z.shape, self.value, dtype="float64")

    def label(self) -> str:
        return f"f {self.value:g}" if self.kind == "relative" else f"fixed T {self.value:g}"


@dataclass(frozen=True)
class Splits:
    """Candidates made leads although an accepted lead lay within reach, in the order made.

    Each is split from its nearest such lead (ties: the stronger) by a valley: the bin of
    lowest envelope between them, nearest the candidate when several share it.
    """

    row: np.ndarray
    lead: np.ndarray
    chromosome: np.ndarray
    bin: np.ndarray
    envelope: np.ndarray

    @property
    def kind(self) -> np.ndarray:
        """`no_row` where the valley bin holds no tier row, `low_z` where its rows fall short."""
        return np.where(self.envelope == 0, "no_row", "low_z")


@dataclass(frozen=True)
class Prominence:
    """Each candidate's lead (itself when it is one), and the splits that made leads."""

    lead_of: np.ndarray
    splits: Splits
    bin_bp: int


class Envelope:
    """One chromosome's envelope, and where a candidate's col first fails on each side."""

    def __init__(self, position: np.ndarray, abs_z: np.ndarray, bin_bp: int, n_bins: int) -> None:
        self.bin_bp = bin_bp
        self.values = np.zeros(n_bins, dtype="float64")
        np.maximum.at(self.values, position // bin_bp, abs_z)
        # mins[j][i] is the lowest envelope over bins i .. i + 2**j - 1.
        self.mins = [self.values]
        while 2 ** len(self.mins) <= n_bins:
            half = 2 ** (len(self.mins) - 1)
            last = self.mins[-1]
            self.mins.append(np.minimum(last[:-half], last[half:]))

    def check_holds(self, position: np.ndarray, abs_z: np.ndarray) -> None:
        """Refuse candidates the profile does not hold: their own bin must reach their |z|."""
        short = self.values[position // self.bin_bp] < abs_z
        if short.any():
            raise ValueError(f"{int(short.sum()):,} candidates are not in the profile, e.g. at "
                             f"{int(position[short][0]):,}")

    def blocking(self, bins: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per candidate, the nearest bin at or left of its own whose envelope is at most its
        threshold (-1 if none), and the nearest at or right of it (n_bins if none)."""
        n = self.values.size
        start, end = bins + 1, bins.copy()
        for level in range(len(self.mins) - 1, -1, -1):
            step, table = 2 ** level, self.mins[level]
            left = start - step
            clear = (left >= 0) & (table[np.clip(left, 0, table.size - 1)] > threshold)
            start = np.where(clear, left, start)
            clear = (end + step <= n) & (table[np.clip(end, 0, table.size - 1)] > threshold)
            end = np.where(clear, end + step, end)
        return start - 1, end

    def valley(self, lead_bin: int, row_bin: int) -> tuple[int, float]:
        """The bin of lowest envelope from the lead's bin to the row's, nearest the row."""
        if lead_bin <= row_bin:
            stretch = self.values[lead_bin:row_bin + 1][::-1]
            at = row_bin - int(np.argmin(stretch))
        else:
            at = row_bin + int(np.argmin(self.values[row_bin:lead_bin + 1]))
        return at, float(self.values[at])


def prominence_leads(chromosome: np.ndarray, position: np.ndarray, abs_z: np.ndarray,
                     variant_index: np.ndarray, profile: Profile, *, bin_bp: int,
                     threshold: Threshold) -> Prominence:
    """Each candidate's lead under the prominence rule; `DEFINITION` states it."""
    position = np.asarray(position, dtype="int64")
    abs_z = np.asarray(abs_z, dtype="float64")
    lead_of = np.full(position.size, -1, dtype="int64")
    splits: list[tuple[int, int, str, int, float]] = []
    order = rule_order(abs_z, variant_index)
    for label in dict.fromkeys(chromosome[order].tolist()):
        rows = order[chromosome[order] == label]
        on = profile.chromosome == label
        n_bins = int(max(position[rows].max(), profile.position[on].max(initial=0))) // bin_bp + 1
        envelope = Envelope(profile.position[on], profile.abs_z[on], bin_bp, n_bins)
        envelope.check_holds(position[rows], abs_z[rows])
        left, right = envelope.blocking(position[rows] // bin_bp, threshold.at(abs_z[rows]))
        low = np.maximum((left + 1) * bin_bp, position[rows] - MAX_SPAN_BP)
        high = np.minimum(right * bin_bp - 1, position[rows] + MAX_SPAN_BP)
        splits += [(row, lead, label, *at) for row, lead, at in
                   _assign(rows, position, low, high, lead_of, envelope)]
    row, lead, chrom, at, value = (list(c) for c in zip(*splits, strict=True)) if splits else (
        [], [], [], [], [])
    return Prominence(lead_of, Splits(np.array(row, dtype="int64"), np.array(lead, dtype="int64"),
                                      np.array(chrom, dtype=str), np.array(at, dtype="int64"),
                                      np.array(value, dtype="float64")), bin_bp)


def _assign(rows: np.ndarray, position: np.ndarray, low: np.ndarray, high: np.ndarray,
            lead_of: np.ndarray, envelope: Envelope) -> list[tuple[int, int, tuple[int, float]]]:
    """One chromosome's candidates, in rank order: each joins a lead or becomes one.

    Returns each split as (candidate, its nearest lead, the valley between them).
    """
    accepted, splits = _Accepted(), []
    bin_bp = envelope.bin_bp
    for row, lo, hi in zip(rows.tolist(), low.tolist(), high.tolist(), strict=True):
        lead = accepted.strongest_in(lo, hi)
        if lead is None:
            near = accepted.nearest(int(position[row]), MAX_SPAN_BP)
            if near is not None:
                splits.append((row, near, envelope.valley(int(position[near]) // bin_bp,
                                                          int(position[row]) // bin_bp)))
            accepted.add(row, int(position[row]))
            lead = row
        lead_of[row] = lead
    return splits


@dataclass
class _Accepted:
    """One chromosome's accepted leads in position order, each with its acceptance rank."""

    position: list[int] = field(default_factory=list)
    entries: list[tuple[int, int]] = field(default_factory=list)  # (rank, row)

    def add(self, row: int, position: int) -> None:
        at = bisect.bisect_right(self.position, position)
        self.position.insert(at, position)
        self.entries.insert(at, (len(self.entries), row))

    def strongest_in(self, low: int, high: int) -> int | None:
        """The first-accepted lead with position in [low, high], if any."""
        lo = bisect.bisect_left(self.position, low)
        hi = bisect.bisect_right(self.position, high)
        return min(self.entries[lo:hi])[1] if lo < hi else None

    def nearest(self, position: int, span: int) -> int | None:
        """The lead nearest `position`, at most `span` away (ties: the first accepted)."""
        at = bisect.bisect_left(self.position, position)
        gaps = [abs(self.position[k] - position) for k in (at - 1, at)
                if 0 <= k < len(self.position)]
        if not gaps or min(gaps) > span:
            return None
        lo = bisect.bisect_left(self.position, position - min(gaps))
        hi = bisect.bisect_right(self.position, position + min(gaps))
        return min(entry for entry, at_position in zip(self.entries[lo:hi],
                                                        self.position[lo:hi], strict=True)
                   if abs(at_position - position) == min(gaps))[1]
