"""Every record of a converted chromosome, with the counts the variant filters read.

The filters are `docs/spec/build-inputs.md`'s, applied in its order; a dropped record is
counted under the first one it fails:

1. `filter`: FILTER is neither PASS nor `.` (when PASS is required);
2. `mac`: minor allele count on the kept samples below `mac_min`;
3. `zero_variance`: every sample has the same genotype, so r is undefined.

Genotypes must be complete. A missing call is not a filter: it stops the run, as it stops a build.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from test_panel.params import Params
from test_panel.tools import conversion_args, converted, plink2

REASONS = ("filter", "mac", "zero_variance")


@dataclass(frozen=True)
class Sites:
    """One converted chromosome; arrays are in pvar record order (the `row` of `prefix`)."""

    chrom: str
    prefix: Path
    pos: np.ndarray
    ref: np.ndarray
    alt: np.ndarray
    filt: np.ndarray
    hom_ref: np.ndarray
    het: np.ndarray
    hom_alt: np.ndarray
    missing: np.ndarray

    @property
    def called(self) -> np.ndarray:
        return self.hom_ref + self.het + self.hom_alt

    @property
    def mac(self) -> np.ndarray:
        alt = self.het + 2 * self.hom_alt
        return np.minimum(alt, 2 * self.called - alt)

    @property
    def maf(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.mac / (2 * self.called)

    @property
    def zero_variance(self) -> np.ndarray:
        c = self.called
        return (self.hom_ref == c) | (self.het == c) | (self.hom_alt == c)


def convert(params: Params, chrom: str) -> Path:
    """Convert one chromosome with the documented command, unless a matching run is on disk."""
    prefix = params.work / f"EUR.chr{chrom}"
    if not converted(prefix):
        source = params.source.format(chrom=chrom)
        plink2(params, conversion_args(source, params.keep, prefix), f"convert.chr{chrom}")
    return prefix


def load(params: Params, chrom: str, prefix: Path) -> Sites:
    # A separate --out, so this run's log never replaces the conversion's, which convert() reads.
    counts_prefix = Path(f"{prefix}.counts")
    if not Path(f"{counts_prefix}.gcount").exists():
        plink2(params, ["--pfile", str(prefix), "--geno-counts", "--out", str(counts_prefix)],
               f"gcount.chr{chrom}")
    cols = _columns(Path(f"{prefix}.pvar"), ("POS", "REF", "ALT", "FILTER"))  # FILTER required
    counts = _columns(Path(f"{counts_prefix}.gcount"), ("HOM_REF_CT", "HET_REF_ALT_CTS",
                                                 "TWO_ALT_GENO_CTS", "MISSING_CT"))
    ints = [np.array(c, dtype=np.int64) for c in counts]
    if ints[3].any():
        raise SystemExit(f"chr{chrom}: {int((ints[3] > 0).sum())} records have missing calls "
                         f"({int(ints[3].sum())} in all); genotypes must be complete")
    return Sites(chrom, prefix, np.array(cols[0], dtype=np.int64), np.array(cols[1]),
                 np.array(cols[2]), np.array(cols[3]), *ints)


def _columns(path: Path, names: tuple[str, ...]) -> list[list[str]]:
    with path.open() as f:
        header = f.readline().lstrip("#").rstrip("\n").split("\t")
        absent = [n for n in names if n not in header]
        if absent:
            raise SystemExit(f"{path} has no {', '.join(absent)} column")
        idx = [header.index(n) for n in names]
        out: list[list[str]] = [[] for _ in names]
        for line in f:
            fields = line.rstrip("\n").split("\t")
            for o, i in zip(out, idx, strict=True):
                o.append(fields[i])
    return out


def drop_reason(s: Sites, params: Params) -> np.ndarray:
    """Per record, the first filter it fails, or "" when it passes them all."""
    fails = [
        params.require_pass & ~np.isin(s.filt, ["PASS", "."]),
        s.mac < params.mac_min,
        s.zero_variance,
    ]
    reason = np.full(len(s.pos), "", dtype=object)
    for name, f in reversed(list(zip(REASONS, fails, strict=True))):
        reason[f] = name
    return reason
