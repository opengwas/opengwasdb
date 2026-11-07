"""Building an LD Store from a Source Panel's genotypes (layer 2; ogld#5).

The builder runs where the genotypes are and writes only the LD Store (ADR 0001,
decision 5). Its inputs are specified in `docs/spec/build-inputs.md` (ADR 0002): a PLINK
2 pgen, a keep file and a PC File; ogld runs no PCA. It imports `ogld.store`,
`ogld.format` and `ogld.identity`, never `ogld.query`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ogld.format import Provenance


@dataclass(frozen=True)
class BuildInputs:
    """Everything a build reads and every setting it applies (`docs/spec/build-inputs.md`).

    Each field is recorded in the store's provenance (spec §7).

    - `pgen`: one or more PLINK 2 `.pgen` files, one per chromosome or one for the
      genome, each with its `.pvar` (or `.pvar.zst`) and `.psam` beside it; GRCh38.
    - `keep`: the samples to use, one `#IID` per line; n is their count.
    - `pcs`: the PC File (plink2's `.eigenvec` layout); PC1…PCk are regressed out. None
      only with k = 0.
    - `pc_source`: the PC File's declared source, recorded beside its SHA-256; None only
      with k = 0.
    - `panel_id`, `ancestry`: the Source Panel, as provenance names it.
    - `genotype_source`: what the genotypes are, such as the gnomAD release and files.
    - `k`: PCs regressed out, and t_far's k (ADR 0001).
    - `mac_min`: the minor allele count floor, counted on the kept samples.
    - `info_min`: the INFO floor; a pgen that carries no INFO with it set is refused,
      so a sequenced panel passes None explicitly rather than skip a filter quietly.
    - `call_rate_min`: the call-rate floor; missing calls are handled by the rule in
      `docs/spec/build-inputs.md`, and None means no floor.
    """

    pgen: tuple[Path, ...]
    keep: Path
    pcs: Path | None
    pc_source: str | None
    panel_id: str
    ancestry: str
    genotype_source: str
    k: int = 10
    mac_min: int = 20
    info_min: float | None = 0.8
    call_rate_min: float | None = 0.95

    def __post_init__(self) -> None:
        if not self.pgen:
            raise ValueError("BuildInputs needs at least one pgen")
        if self.k < 0:
            raise ValueError(f"k counts PCs, so it cannot be negative: {self.k}")
        if self.k > 0 and (self.pcs is None or self.pc_source is None):
            raise ValueError(
                f"k = {self.k} needs a PC File and its declared source; only k = 0 runs without"
            )


def build(inputs: BuildInputs, out: str | os.PathLike[str], *, threads: int = 1) -> Provenance:
    """Build the LD Store of `inputs` at `out` and return its provenance (spec §7; ogld#5).

    Filters the Variant Set, regresses PC1…PCk out of the standardised dosages, writes
    the Near Band exactly and the Proxy Index at t_far, and finishes through
    `StoreWriter.finish`, which validates the store. The PC Check runs first: a PC that
    tracks a long-range LD region fails the build, naming the PC and the region.
    """
    raise NotImplementedError("ogld#5")


__all__ = ["BuildInputs", "build"]
