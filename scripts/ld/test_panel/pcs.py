"""The PC File the Test Panel uses: `#IID PC1 ... PC20`, plink2's `.eigenvec` layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from test_panel.params import Params

N_PCS = 20


def load(params: Params, iids: list[str]) -> np.ndarray:
    """(samples, 20) PCs in the order of `iids`; every sample must have all 20, finite."""
    rows = _pc_file(params.pcs)
    missing = [i for i in iids if i not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} kept samples have no PCs, such as {missing[:3]}")
    pcs = np.array([rows[i] for i in iids], dtype=np.float64)
    if pcs.shape != (len(iids), N_PCS) or not np.isfinite(pcs).all():
        raise SystemExit(f"PCs must be {N_PCS} finite values per sample; got shape {pcs.shape}")
    return pcs


def _pc_file(path: Path) -> dict[str, list[float]]:
    lines = path.read_text().splitlines()
    if lines[0].split("\t")[0] != "#IID":
        raise SystemExit(f"{path}: a PC File's header starts #IID")
    return {f[0]: [float(v) for v in f[1 : N_PCS + 1]]
            for f in (line.split("\t") for line in lines[1:])}


def write(path: Path, iids: list[str], pcs: np.ndarray) -> None:
    """plink2's `.eigenvec` layout, `#IID PC1 ... PC20`, values written at full precision so
    the reference r and a builder read exactly the same numbers."""
    head = "#IID\t" + "\t".join(f"PC{k}" for k in range(1, N_PCS + 1)) + "\n"
    body = "".join(i + "\t" + "\t".join(repr(float(v)) for v in row) + "\n"
                   for i, row in zip(iids, pcs, strict=True))
    path.write_text(head + body)
