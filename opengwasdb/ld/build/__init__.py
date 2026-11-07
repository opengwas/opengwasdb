"""Building an LD Store from a Source Panel's genotypes (layer 2; ogld#5).

The builder runs where the genotypes are and writes only the LD Store (ADR 0001,
decision 5). Its inputs are specified in `docs/spec/build-inputs.md` (ADR 0002): a PLINK
2 pgen with complete genotypes, a keep file and a PC File. ogld runs no PCA: the PCs are
computed outside the library by a recipe script, and the builder only chooses which of
them to adjust for, by the PC Check. It imports `ogld.store`, `ogld.format` and
`ogld.identity`, never `ogld.query`.

pgenlib, the pgen reader, is imported inside the build's read path only, never at module
import: `import ogld` and every reader must work on an install without it
(tests/test_surface.py holds this).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ogld.format import InputError, Provenance


@dataclass(frozen=True)
class BuildInputs:
    """Everything a build reads and every setting it applies (`docs/spec/build-inputs.md`).

    Each field is recorded in the store's provenance (spec §7).

    - `pgen`: one or more PLINK 2 `.pgen` files, one per chromosome or one for the
      genome, each with its `.pvar` (or `.pvar.zst`) and `.psam` beside it; GRCh38. The
      genotypes must be complete: the build refuses any missing call, naming the variant
      and the panel, rather than impute it (owner, 7 November 2026; ADR 0002).
    - `keep`: the samples to use, one `#IID` per line; n is their count.
    - `pcs`: the PC File (plink2's `.eigenvec` layout). The PC Check scans every PC in
      it and drops the local ones; the first up to k PCs it keeps are regressed out.
      None only with k = 0.
    - `pc_source`: the PC File's declared source, recorded beside its SHA-256; None only
      with k = 0.
    - `panel_id`, `ancestry`: the Source Panel, as provenance names it.
    - `genotype_source`: what the genotypes are, such as the release and its files.
    - `k`: the target number of PCs to regress out, taken in order from those the PC
      Check keeps (owner, 7 November 2026).
    - `k_min`: the fewest PCs a build may use. When fewer than `k` survive the PC Check,
      the build uses those that survive as long as at least `k_min` do, and refuses
      otherwise, naming how many survived. The number used is t_far's k, and provenance
      records it beside `k` and `k_min`. With k = 0 no PC is read and `k_min` does not
      apply.
    - `mac_min`: the minor allele count floor, counted on the kept samples.
    - `info_min`: the INFO floor; a pgen that carries no INFO with it set is refused,
      so a sequenced panel passes None explicitly rather than skip a filter quietly.
    """

    pgen: tuple[Path, ...]
    keep: Path
    pcs: Path | None
    pc_source: str | None
    panel_id: str
    ancestry: str
    genotype_source: str
    k: int = 10
    k_min: int = 5
    mac_min: int = 20
    info_min: float | None = 0.8

    def __post_init__(self) -> None:
        if not self.pgen:
            raise InputError("BuildInputs needs at least one pgen")
        _check_pc_counts(self.k, self.k_min)
        if self.k > 0 and (self.pcs is None or self.pc_source is None):
            raise InputError(
                f"k = {self.k} needs a PC File and its declared source; only k = 0 runs without"
            )


def _check_pc_counts(k: int, k_min: int) -> None:
    """The target and the minimum count PCs, and the minimum cannot exceed the target."""
    if k < 0:
        raise InputError(f"k counts PCs, so it cannot be negative: {k}")
    if k_min < 0:
        raise InputError(f"k_min counts PCs, so it cannot be negative: {k_min}")
    if k > 0 and k_min > k:
        raise InputError(f"k_min = {k_min} cannot exceed the target k = {k}")


def build(inputs: BuildInputs, out: str | os.PathLike[str], *, threads: int = 1) -> Provenance:
    """Build the LD Store of `inputs` at `out` and return its provenance (spec §7; ogld#5).

    Refuses an input with a missing call. Filters the Variant Set, then runs the PC
    Check (`docs/spec/build-inputs.md` §8): a PC is local, and dropped, when one 4 Mb
    window explains ≥ 10% of it, that share beats the window-side null, and rest share /
    window share ≤ 0.05. The covariates are the first up to k PCs kept; with fewer than k,
    those that survive, and a refusal naming how many survived, and which were dropped,
    when fewer than k_min do. It regresses them out of the standardised
    dosages, writes the
    Near Band exactly and the Proxy Index at t_far, and finishes through
    `StoreWriter.finish`, which validates the store. Provenance records the scan, the
    dropped PCs and the null the scan was judged against.
    """
    raise NotImplementedError("ogld#5")


__all__ = ["BuildInputs", "build"]
