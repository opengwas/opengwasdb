"""Every setting the Test Panel is made with, recorded whole in its manifest.

The genotype source, the PC File and every filter are parameters, so a changed decision is one
run with a new value (ADR 0002). Paths default to the documented roots of `test_panel.roots`
(`$OGLD_PHASED_V2`, `$OGLD_PCS`, `$OGLD_REFERENCE`) and tools to `$OGLD_PLINK2`,
`$OGLD_BCFTOOLS` and `$OGLD_BGZIP`, or `PATH`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from opengwasdb.ld.format import FAR_BP, T_FAR_Q
from opengwasdb.ld.format import t_far as format_t_far
from test_panel.roots import root, tool

V2_NAME = "hgdp1kgp_chr{chrom}.filtered.SNV_INDEL.phased.shapeit5.bcf"


@dataclass(frozen=True)
class Params:
    """The Test Panel's inputs and settings. Positions are GRCh38, 1-based."""

    out: Path
    work: Path
    plink2: Path
    bcftools: Path
    bgzip: Path
    # The phased HGDP+1kGP v2 callset (Koenig et al. 2024), downloaded and CRC32C-verified.
    source: str
    keep: Path
    # The PC File, `#IID PC1 ... PC20`: make_pcs.py's for EUR.
    pcs: Path
    pcs_source: str = "ogld scripts/make_pcs.py on HGDP+1kGP v2 EUR (ADR 0002)"
    chromosomes: tuple[str, ...] = ("21", "22")
    # Variant filters (docs/spec/build-inputs.md). Genotypes must be complete: a missing call is
    # refused, never imputed.
    require_pass: bool = True
    mac_min: int = 20
    ks: tuple[int, ...] = (0, 10)
    far_bp: int = FAR_BP
    # Recorded in the manifest; t_far itself is opengwasdb.ld.format's.
    t_far_q: float = T_FAR_Q
    # The r scale the far-pair selection judges Stored r at: × 32767, the one encoding there was
    # when the committed panel was made. The Test Store is × 511 (ogld#16, ADR 0004); the panel's
    # Far Pairs reach t_far at both, and a regenerated panel may move this.
    r_scale: int = 32767
    # Selection. Each window was picked on the converted chromosomes (README, "Selection").
    q_arm_start: dict[str, int] = field(
        default_factory=lambda: {"21": 14_000_000, "22": 16_000_000}
    )
    proxy_window: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"21": (30_686_000, 30_740_000), "22": (38_850_000, 38_950_000)}
    )
    proxy_max: int = 150
    gwas_window: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"21": (42_000_000, 43_000_000), "22": (20_000_000, 21_500_000)}
    )
    gwas_max: dict[str, int] = field(default_factory=lambda: {"21": 200, "22": 300})
    backbone_bp: int = 300_000
    common_maf: float = 0.05
    far_search_thin_bp: int = 20_000
    far_pairs_per_chrom: int = 3
    edge_chrom: str = "22"
    exact_distances: tuple[int, ...] = (3_000_000, 3_000_001, 10_000_000, 10_000_001)
    fail_mac_values: tuple[int, ...] = (0, 1, 5, 10, 19)
    fail_filters_max: int = 4
    # The simulated GWAS: causal variants nearest these positions, each explaining h2_each.
    causal_anchor: dict[str, tuple[int, ...]] = field(
        default_factory=lambda: {
            "21": (42_150_000, 42_500_000, 42_850_000),
            "22": (20_150_000, 20_450_000, 20_750_000, 21_050_000, 21_350_000),
        }
    )
    causal_min_maf: float = 0.2
    h2_each: float = 0.06
    seed: int = 20261107
    gwas_pcs: int = 10
    clump_p1: float = 5e-8
    clump_kb: int = 3000
    clump_r2: tuple[float, ...] = (0.01, 0.001)
    size_budget_bytes: int = 5_000_000
    threads: int = 8
    memory_mb: int = 32000

    def t_far(self, n: int, k: int) -> float:
        """t_far = max(0.001, q / (n - k - 1)) (ADR 0001), as opengwasdb.ld.format has it."""
        return format_t_far(n, k)


def parse(argv: list[str] | None = None) -> Params:
    """Command-line settings over the defaults above."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("tests/ld/data/test-panel"))
    p.add_argument(
        "--work",
        type=Path,
        required=True,
        help="scratch directory for the converted chromosomes (several GB)",
    )
    p.add_argument("--plink2", type=Path, help="default: $OGLD_PLINK2, else plink2 on PATH")
    p.add_argument("--bcftools", type=Path, help="default: $OGLD_BCFTOOLS, else PATH")
    p.add_argument("--bgzip", type=Path, help="default: $OGLD_BGZIP, else PATH")
    p.add_argument(
        "--source",
        help=f"per-chromosome BCF or VCF with {{chrom}}; default $OGLD_PHASED_V2/{V2_NAME}",
    )
    p.add_argument(
        "--keep", type=Path, help="default: $OGLD_REFERENCE/hgdp1kgp-hg38/members/EUR.keep"
    )
    p.add_argument("--pcs", type=Path, help="the PC File; default $OGLD_PCS/EUR.eigenvec")
    p.add_argument("--pcs-source", default=Params.pcs_source)
    p.add_argument("--mac-min", type=int, default=Params.mac_min)
    p.add_argument("--no-require-pass", action="store_true")
    p.add_argument("--threads", type=int, default=Params.threads)
    a = p.parse_args(argv)
    return Params(
        out=a.out,
        work=a.work,
        plink2=a.plink2 or tool("plink2"),
        bcftools=a.bcftools or tool("bcftools"),
        bgzip=a.bgzip or tool("bgzip"),
        source=a.source or str(root("OGLD_PHASED_V2") / V2_NAME),
        keep=a.keep or root("OGLD_REFERENCE") / "hgdp1kgp-hg38/members/EUR.keep",
        pcs=a.pcs or root("OGLD_PCS") / "EUR.eigenvec",
        pcs_source=a.pcs_source,
        mac_min=a.mac_min,
        require_pass=not a.no_require_pass,
        threads=a.threads,
    )
