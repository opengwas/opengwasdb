"""Tiny genotype panels for the Locus Lead prototype's tests (#402).

Laid out as the HGDP+1kGP panels are, one `chr<N>.pgen/.psam/.pvar.zst` per
chromosome, and built by plink2 from a VCF so they are read exactly as the
real ones are. The contig line declares the assembly, as the real pvars do.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from prototype_modules import load_prototype

panel = load_prototype("locus_leads", "panel")


def write_panel(directory: Path, chromosome: str, variants: list[tuple[int, str, str]],
                genotypes: list[list[str]], *, assembly: str = "GRCh38") -> None:
    """One chromosome's pgen: `variants` are (position, REF, ALT), each with its own calls."""
    samples = [f"S{i}" for i in range(len(genotypes[0]))]
    lines = ["##fileformat=VCFv4.2",
             f"##contig=<ID={chromosome},length=10000000,assembly={assembly}>",
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
             "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO",
                        "FORMAT", *samples])]
    for (position, ref, alt), calls in zip(variants, genotypes, strict=True):
        lines.append("\t".join([chromosome, str(position), f"{chromosome}:{position}:{ref}:{alt}",
                                ref, alt, ".", "PASS", ".", "GT", *calls]))
    vcf = directory / f"chr{chromosome}.vcf"
    vcf.write_text("\n".join(lines) + "\n")
    subprocess.run([panel.plink2(), "--vcf", str(vcf), "--make-pgen", "vzs",
                    "--out", str(directory / f"chr{chromosome}")], check=True, capture_output=True)
