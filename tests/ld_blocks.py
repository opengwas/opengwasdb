"""The tests' writer of LD-panel blocks: a block's variant table, and its LD matrix, its
eigendecomposition, or both, in the panel's flat layout (`ld_dir/{ancestry}/{chr}/`).
"""

from __future__ import annotations

from gzip import GzipFile
from io import BytesIO
from pathlib import Path

import numpy as np


def write_gzipped_matrix(path: Path, ld: np.ndarray) -> None:
    """An LD matrix as a gzip of tab-separated rows, with no time in its header so
    that a fixture panel's digest is the same on every run."""
    buf = BytesIO()
    with GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        for row in ld:
            gz.write(("\t".join(f"{v:.6f}" for v in row) + "\n").encode())
    path.write_bytes(buf.getvalue())


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
        write_gzipped_matrix(block_dir / f"{block_name}.unphased.vcor1.gz", ld)

    if write_npz:
        vals, vecs = np.linalg.eigh(ld)
        vals = vals[::-1]
        vecs = vecs[:, ::-1]
        k = vecs.shape[1] if npz_k is None else min(npz_k, vecs.shape[1])
        np.savez_compressed(
            block_dir / f"{block_name}.ldeig", values=vals, vectors=vecs[:, :k]
        )


def regenerated_panel(source: Path, destination: Path) -> Path:
    """A copy of the panel at ``source`` in which one block's first variant has a
    different frequency: the panel a regeneration from other samples leaves,
    same layout and same blocks, other content. Asserts the copy differs."""
    import shutil

    shutil.copytree(source, destination)
    table = sorted(destination.rglob("*.tsv"))[0]
    lines = table.read_text().splitlines()
    columns = lines[0].split("\t")
    first = lines[1].split("\t")
    before = first[columns.index("EAF")]
    first[columns.index("EAF")] = "0.5" if before != "0.5" else "0.25"
    lines[1] = "\t".join(first)
    table.write_text("\n".join(lines) + "\n")
    assert table.read_text() != (source / table.relative_to(destination)).read_text()
    return destination


def change_a_block(panel: Path) -> None:
    """Rewrite one block's table in place, as a regeneration under a resumed run would."""
    table = sorted(panel.rglob("*.tsv"))[0]
    before = table.read_text()
    table.write_text(before.replace("\t0.", "\t0.1", 1))
    assert table.read_text() != before


def add_a_table_with_no_matrix(panel: Path) -> Path:
    """A copy of the first block's table beside it under another name, with no matrix:
    not a loadable block, but one the EAF reader opens. Returns it."""
    first = sorted(panel.rglob("*.tsv"))[0]
    orphan = first.with_name("orphan.tsv")
    orphan.write_bytes(first.read_bytes())
    assert not orphan.with_suffix(".ldeig.npz").exists()
    assert not orphan.with_suffix(".unphased.vcor1.gz").exists()
    return orphan


def assert_named_panel_record(record: dict, name: str) -> None:
    """A completion's `provenance.completion` names the panel as the caller did."""
    assert record["ld_panel_id"] == name
    assert record["ld_panel_id_source"] == "caller"
    assert len(record["ld_panel_digest"]) == 64
