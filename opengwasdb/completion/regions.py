"""Which LD blocks an Analysis is completed against: its retained regions (ADR 0014, ADR 0039).

One definition for `complete-ragged`, which completes against these blocks, and for the
Observed-View Check (issue #296), which rebuilds them to test that every imputed
association lies inside one. An Analysis with a Trait position is completed against the
blocks touching its cis window; one without, against the blocks holding enough of its own
observations, which `completion.ld_panel.blocks_over_variants` resolves for many Analyses
in one pass.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from opengwasdb.completion.ld_panel import LDBlock, blocks_over_variants, find_blocks
from opengwasdb.model.analyses import Analysis

__all__ = ["blocks_for_analysis", "cis_window_blocks", "has_trait_position", "trait_less_blocks"]


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


def trait_less_blocks(
    held: Mapping[int, Iterable[tuple[str, str]]], ld_dir: str | Path, ancestry: str
) -> dict[int, list[LDBlock]]:
    """The blocks each Analysis with no Trait position already holds enough observations in.

    `held` maps an Analysis index to the `(ALID, chromosome)` of every association it holds.
    They are resolved in one ALID-keyed pass, so the panel is read once rather than once per
    Analysis (issue #102). An Analysis holding too few at any block is absent from the result.
    """
    analyses_by_alid: dict[str, list[int]] = {}
    chromosomes: set[str] = set()
    for index, associations in held.items():
        for alid, chromosome in associations:
            analyses_by_alid.setdefault(alid, []).append(index)
            chromosomes.add(chromosome)
    return blocks_over_variants(ld_dir, ancestry, analyses_by_alid, sorted(chromosomes))
