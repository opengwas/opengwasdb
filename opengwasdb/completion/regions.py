"""Which LD blocks an Analysis is completed against: its retained regions (ADR 0014, ADR 0039).

One definition for `complete-ragged`, which completes against these blocks, and for the
Observed-View Check (issue #296), which rebuilds them to test that every imputed
association lies inside one. An Analysis with a Trait position is completed against the
blocks touching its cis window; one without, against the blocks holding enough of its own
observations, which `completion.ld_panel.blocks_over_variants` resolves for many Analyses
in one pass.
"""

from __future__ import annotations

from pathlib import Path

from opengwasdb.completion.ld_panel import LDBlock, find_blocks
from opengwasdb.model.analyses import Analysis

__all__ = ["blocks_for_analysis", "cis_window_blocks", "has_trait_position"]


def has_trait_position(a: Analysis) -> bool:
    """Whether the Analysis names a Trait genomic position to centre a cis window on."""
    return bool(a.trait_chr) and bool(a.trait_bp)


def cis_window_blocks(
    a: Analysis, ld_dir: str | Path, ancestry: str, cis_window_bp: int
) -> list[LDBlock]:
    """The LD blocks touching one Analysis's cis window around its Trait position."""
    start = max(1, int(a.trait_bp) - cis_window_bp)
    end = int(a.trait_bp) + cis_window_bp
    return find_blocks(ld_dir, ancestry, a.trait_chr, start, end)


def blocks_for_analysis(
    a: Analysis,
    own_variant_blocks: list[LDBlock],
    ld_dir: str | Path,
    ancestry: str,
    cis_window_bp: int,
) -> list[LDBlock]:
    """An Analysis's completable blocks: its cis window where it has a Trait position, the
    blocks it already holds enough observations in (`own_variant_blocks`) otherwise."""
    if has_trait_position(a):
        return cis_window_blocks(a, ld_dir, ancestry, cis_window_bp)
    return own_variant_blocks
