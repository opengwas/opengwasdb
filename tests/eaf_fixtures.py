"""Fixture sources for the EAF orientation tests: one study written as GWAS-SSF
files whose frequency column is, or is not, reported against the other allele."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

# Comfortably above the 500-variant minimum overlap, so the fixtures exercise
# the check rather than its "not enough to say" gate.
N_VARIANTS = 900

SSF_COLUMNS = tuple(
    "chromosome base_pair_location effect_allele other_allele beta standard_error "
    "effect_allele_frequency rsid".split()
)


def frequencies(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    freqs = rng.beta(0.6, 0.6, size=N_VARIANTS)
    return {
        f"1:{100_000 + i * 41}:A:G": float(np.clip(f, 0.001, 0.999))
        for i, f in enumerate(freqs)
    }


def write_ssf(path: Path, freqs: dict[str, float], *, flipped: bool, seed: int = 0) -> Path:
    """A harmonised GWAS-SSF file. `flipped` reports the frequency of the
    *other* allele while leaving every other column alone -- exactly the
    GCST003566 defect."""
    rng = np.random.default_rng(seed)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(SSF_COLUMNS) + "\n")
        for i, (alid, f) in enumerate(freqs.items()):
            chrom, pos, a1, a2 = alid.split(":")
            # Sampling noise, so the correlation is a realistic |r| < 1 rather
            # than an exact copy of the reference correlating at exactly +-1.
            own = float(np.clip(f + rng.normal(0.0, 0.01), 0.001, 0.999))
            eaf = 1.0 - own if flipped else own
            beta = float(rng.normal(0.0, 0.05))
            # `effect_allele` is A2 here, so the reader negates z and reports
            # 1 - eaf; the stored frequency is the A1-oriented one either way.
            handle.write(
                f"{chrom}\t{pos}\t{a2}\t{a1}\t{beta:.6f}\t0.05\t{1.0 - eaf:.6f}\trs{i}\n"
            )
    return path


def dense_manifest(path: Path, sources: dict[str, Path]) -> Path:
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\tsource_assembly\tsource_reader_capability"
    ]
    for trait_id, source in sources.items():
        lines.append(
            f"{trait_id}\t{source}\t{trait_id}\t10000\tsd\tdeclared_standardised"
            "\thg38\topengwasdb.gwas-ssf"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def reference_table(path: Path, freqs: dict[str, float]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("alid\teaf\n")
        for alid, f in freqs.items():
            handle.write(f"{alid}\t{f:.6f}\n")
    return path


def panel_alids(path: Path, freqs: dict[str, float]) -> Path:
    path.write_text("\n".join(freqs) + "\n", encoding="utf-8")
    return path
