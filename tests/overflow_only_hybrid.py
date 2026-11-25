"""A small Hybrid release with an Overflow-only Analysis, for its Rho Matrix (#496).

Panel variants sit every `SPACING` bp on chromosomes 1 and 2, and an off-panel variant
`SPACING // 2` bp after every `OFF_PANEL_EVERY`-th of them, so the Ragged Overflow's rows
interleave with the Dense Component's on both chromosomes: a mapping that confused the
two would misplace rows on each.

- `trait_a` and `trait_b` are observed at every variant, their null Z correlated by
  `CORRELATION`.
- `trait_c` (`OVERFLOW_ONLY`) is observed off the panel only: its Dense Component column
  holds no observed association, so it is an Overflow-only Analysis.
- `trait_d` (`UNTHINNED_ONLY`) is observed off the panel and at one panel variant that
  distance thinning at `RHO_OPTIONS["window_bp"]` drops. It is a Dense Analysis with
  no null support at the thinned rows: only a scan of the whole grid tells it apart from
  an Overflow-only one.

Each Analysis has a few strong signals, so every Top-Hit tier holds rows. The Z come
from a fixed seed. `RHO_OPTIONS` is the Rho build a release this small can estimate:
two panel variants a window, of which thinning keeps the first, and a support floor
the shared nulls of `trait_a` and `trait_b` clear.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import test_indel_pairs as pair
from test_hybrid_completion import build_over_panel

ANALYSES = ("trait_a", "trait_b", "trait_c", "trait_d")
DENSE_ANALYSES = ("trait_a", "trait_b", "trait_d")
OVERFLOW_ONLY = "trait_c"
UNTHINNED_ONLY = "trait_d"
CHROMOSOMES = ("1", "2")
N_PANEL = 300  # a chromosome
SPACING = 1000
OFF_PANEL_EVERY = 5
CORRELATION = 0.6
SE = 0.01
#: Every `SIGNAL_EVERY`-th variant of an Analysis is a signal of z `SIGNAL_Z`.
SIGNAL_EVERY = 50
SIGNAL_Z = 8.0
SEED = 496
RHO_OPTIONS = {"window_bp": 2 * SPACING, "z_thresh": 2.0, "min_nulls": 50}
#: The one panel variant `UNTHINNED_ONLY` is observed at, `(chromosome, position)`: the
#: second of its window, which thinning drops.
UNTHINNED_VARIANT = ("1", 3 * SPACING)


def _variants() -> list[tuple[str, int, bool]]:
    """`(chromosome, position, on_panel)` in genome order."""
    out: list[tuple[str, int, bool]] = []
    for chrom in CHROMOSOMES:
        for i in range(1, N_PANEL + 1):
            out.append((chrom, i * SPACING, True))
            if i % OFF_PANEL_EVERY == 0:
                out.append((chrom, i * SPACING + SPACING // 2, False))
    return out


def _z(n: int) -> dict[str, np.ndarray]:
    """Each Analysis's Z at every one of the `n` variants."""
    rng = np.random.default_rng(SEED)
    shared, own_b, own_c, own_d = rng.standard_normal((4, n))
    z = {
        "trait_a": shared,
        "trait_b": CORRELATION * shared + np.sqrt(1 - CORRELATION**2) * own_b,
        "trait_c": own_c,
        "trait_d": own_d,
    }
    for values in z.values():
        values[::SIGNAL_EVERY] = SIGNAL_Z
    return z


def _observed(analysis: str, chrom: str, position: int, on_panel: bool) -> bool:
    if analysis == OVERFLOW_ONLY:
        return not on_panel
    if analysis == UNTHINNED_ONLY:
        return not on_panel or (chrom, position) == UNTHINNED_VARIANT
    return True


def _rows(analysis: str, z: np.ndarray) -> list[tuple[str, ...]]:
    """`(chrom, pos, ref, alt, beta, se, af)` for the variants `analysis` is observed at."""
    rows: list[tuple[str, ...]] = []
    for i, (chrom, position, on_panel) in enumerate(_variants()):
        if not _observed(analysis, chrom, position, on_panel):
            continue
        ref, alt = ("A", "G") if on_panel else ("C", "T")
        freq = 0.05 + 0.9 * (i % 97) / 97
        rows.append(
            (chrom, str(position), ref, alt, f"{z[i] * SE:.6f}", f"{SE}", f"{freq:.4f}")
        )
    return rows


def overflow_only_hybrid(root: Path) -> Path:
    """Build the release under `root` (which must exist) and return its path."""
    variants = _variants()
    z = _z(len(variants))
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\tsource_reader_capability\tsource_assembly\n"
    ]
    for analysis in ANALYSES:
        source = pair._gwas_vcf(root / f"{analysis}.vcf", _rows(analysis, z[analysis]))
        lines.append(
            f"{analysis}\t{source}\t{analysis}\t500000\tsd\tdeclared_standardised\t"
            "\topengwasdb.gwas-vcf\tGRCh38\n"
        )
    manifest = root / "manifest.tsv"
    manifest.write_text("".join(lines), encoding="utf-8")
    panel = [f"{chrom}:{position}:A:G" for chrom, position, on_panel in variants if on_panel]
    return build_over_panel(root, manifest, panel)
