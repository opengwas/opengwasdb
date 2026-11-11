"""The tests' writer of LD-panel blocks: a block's variant table, and its LD matrix, its
eigendecomposition, or both, in the panel's flat layout (`ld_dir/{ancestry}/{chr}/`).
"""

from __future__ import annotations

from gzip import GzipFile
from io import BytesIO
from pathlib import Path

import numpy as np


def write_ld_block(
    block_dir: Path,
    block_name: str,
    snps: list[tuple[str, float, int]],
    seed: int = 0,
    *,
    write_matrix: bool = True,
    write_npz: bool = False,
    npz_k: int | None = None,
    snp_id_style: str = "alid",
) -> None:
    """Write one flat-layout LD block.

    A block may carry an LD matrix, an eigendecomposition, or both — panels built
    under ADR 0031 ship only the decomposition, so tests need to construct each
    combination. ``npz_k`` truncates the stored eigenvectors to force the
    under-resolved case. ``snp_id_style`` selects the panel's identifier
    convention: canonical ALIDs, or the legacy ``chr:pos_ref_alt`` form used by the
    production UKB EUR panel.
    """
    block_dir.mkdir(parents=True, exist_ok=True)
    n = len(snps)

    tsv_lines = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
    for alid, eaf, bp in snps:
        chrom, pos, a1, a2 = alid.split(":")
        snp_id = alid if snp_id_style == "alid" else f"{chrom}:{pos}_{a1}_{a2}"
        tsv_lines.append(f"{chrom}\t{snp_id}\t{a2}\t{a1}\t{eaf}\t{bp}")
    (block_dir / f"{block_name}.tsv").write_text("\n".join(tsv_lines) + "\n")

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    ld = A @ A.T + np.eye(n) * n * 0.1

    if write_matrix:
        buf = BytesIO()
        with GzipFile(fileobj=buf, mode="wb") as gz:
            for row in ld:
                gz.write(("\t".join(f"{v:.6f}" for v in row) + "\n").encode())
        (block_dir / f"{block_name}.unphased.vcor1.gz").write_bytes(buf.getvalue())

    if write_npz:
        vals, vecs = np.linalg.eigh(ld)
        vals = vals[::-1]
        vecs = vecs[:, ::-1]
        k = vecs.shape[1] if npz_k is None else min(npz_k, vecs.shape[1])
        np.savez_compressed(
            block_dir / f"{block_name}.ldeig", values=vals, vectors=vecs[:, :k]
        )
