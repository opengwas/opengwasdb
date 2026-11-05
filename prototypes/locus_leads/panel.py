"""The genotype panel, and which Store rows it holds.

The HGDP+1kGP panels are one pgen per autosome (`chr<N>.pgen/.psam/.pvar.zst`).
A pvar row is `CHROM POS ID REF ALT` in VCF orientation, and its ID column
truncates alleles past 100 bases, so only REF and ALT are read for identity.
A Store ALID spells its two alleles sorted (`canonical_alid`), so a Store row
matches a panel row on chromosome, position and allele set.

Two panel rows can share an allele set: an insertion and a deletion at one
position (C>CAATA and CAATA>C), 2,097 such pairs on chr22 of the EUR panel. A
Store ALID cannot say which it is, so a row that reaches either is ambiguous;
so is one whose panel ID is not unique, which plink2 could not address.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STATUSES = ("matched", "off_panel_chromosome", "absent_at_position", "allele_mismatch",
            "ambiguous")


@dataclass(frozen=True)
class PanelVariants:
    """One chromosome's panel rows, in pvar order."""

    position: np.ndarray
    variant_id: np.ndarray
    ref: np.ndarray
    alt: np.ndarray

    def by_allele_set(self) -> pd.DataFrame:
        """One row per (position, allele set), flagged where it names no single variant."""
        low, high = _allele_set(self.ref, self.alt)
        frame = pd.DataFrame({"position": self.position, "low": low, "high": high,
                              "variant_id": self.variant_id})
        frame["ambiguous"] = (frame.duplicated(["position", "low", "high"], keep=False)
                              | frame["variant_id"].duplicated(keep=False))
        return frame.drop_duplicates(["position", "low", "high"])


@dataclass(frozen=True)
class PanelMatch:
    """Per Store row: the panel ID it matched ("" when none) and why."""

    panel_id: np.ndarray
    status: np.ndarray

    def counts(self) -> dict[str, int]:
        return {status: int(np.sum(self.status == status)) for status in STATUSES}


class Panel:
    """A directory of per-autosome pgen files."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        labels = [path.name.removesuffix(".pgen").removeprefix("chr")
                  for path in directory.glob("chr*.pgen")]
        if not labels:
            raise FileNotFoundError(f"{directory}: no chr<N>.pgen files")
        self.chromosomes = sorted(labels, key=_chromosome_order)

    def pfile(self, chromosome: str) -> Path:
        return self.directory / f"chr{chromosome}"

    def variants(self, chromosome: str) -> PanelVariants:
        return read_pvar(self.directory / f"chr{chromosome}.pvar.zst", chromosome)


def match_to_panel(
    chromosome: np.ndarray, position: np.ndarray, allele_1: np.ndarray, allele_2: np.ndarray,
    panel: Mapping[str, PanelVariants],
) -> PanelMatch:
    """Match each Store row to the panel; a chromosome `panel` lacks is off-panel."""
    status = np.full(len(position), "off_panel_chromosome", dtype=object)
    panel_id = np.full(len(position), "", dtype=object)
    for label, variants in panel.items():
        on = np.flatnonzero(np.asarray(chromosome) == label)
        if on.size:
            status[on], panel_id[on] = _match_chromosome(
                np.asarray(position)[on], np.asarray(allele_1)[on], np.asarray(allele_2)[on],
                variants,
            )
    return PanelMatch(panel_id, status)


def read_pvar(path: Path, chromosome: str) -> PanelVariants:
    """A pvar (plain or .zst, decompressed by plink2), checked to hold `chromosome` only."""
    text = path.read_bytes()
    if path.suffix == ".zst":
        text = subprocess.run([plink2(), "--zst-decompress", str(path)], check=True,
                              capture_output=True).stdout
    frame = pd.read_csv(io.BytesIO(text[text.index(b"#CHROM"):]), sep="\t", dtype=str)
    if set(frame["#CHROM"]) != {chromosome}:
        raise ValueError(f"{path}: holds chromosomes {sorted(set(frame['#CHROM']))}, "
                         f"not {chromosome} alone")
    return PanelVariants(frame["POS"].astype("int64").to_numpy(), frame["ID"].to_numpy(),
                         frame["REF"].to_numpy(), frame["ALT"].to_numpy())


def plink2() -> str:
    found = shutil.which("plink2")
    if found is None:
        raise FileNotFoundError("plink2 is not on PATH; run under `pixi run -e dev`")
    return found


def _match_chromosome(position: np.ndarray, allele_1: np.ndarray, allele_2: np.ndarray,
                      variants: PanelVariants) -> tuple[np.ndarray, np.ndarray]:
    low, high = _allele_set(allele_1, allele_2)
    rows = pd.DataFrame({"position": position, "low": low, "high": high})
    merged = rows.merge(variants.by_allele_set(), on=["position", "low", "high"], how="left")
    found = merged["variant_id"].notna().to_numpy()
    ambiguous = merged["ambiguous"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    at_position = np.isin(position, variants.position)
    status = np.select(
        [found & ~ambiguous, found & ambiguous, at_position],
        ["matched", "ambiguous", "allele_mismatch"], "absent_at_position",
    ).astype(object)
    panel_id = np.where(status == "matched", merged["variant_id"].to_numpy(), "").astype(object)
    return status, panel_id


def _allele_set(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two alleles, upper-cased as `identity_allele` spells them, low then high."""
    a = pd.Series(first, dtype=str).str.upper()
    b = pd.Series(second, dtype=str).str.upper()
    return a.where(a <= b, b).to_numpy(), b.where(a <= b, a).to_numpy()


def _chromosome_order(label: str) -> tuple[int, str]:
    return (int(label), "") if label.isdigit() else (1_000, label)
