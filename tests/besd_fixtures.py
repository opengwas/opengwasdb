"""Synthetic BESD source writers shared by the BESD build and resolver tests.

A BESD source is three files beside one prefix: `.esi` (one row per variant,
with an optional seventh frequency column), `.epi` (one row per probe) and the
`.besd` association matrix. These write the minimal shapes the readers accept.
"""

from __future__ import annotations

import struct
from pathlib import Path

#: Passed as an `.esi` row's `freq` to write a six-column row with no
#: frequency column at all, which `read_esi` must read as "unknown".
NO_FREQUENCY_COLUMN = object()


def write_esi(path: Path, snps: list[dict]) -> None:
    """One `.esi` row per dict; `freq` defaults to the literal `NA`."""
    with open(path, "w") as fh:
        for s in snps:
            freq = s.get("freq", "NA")
            cells = [s["chr"], s["snp_id"], "0", s["bp"], s["a1"], s["a2"]]
            if freq is not NO_FREQUENCY_COLUMN:
                cells.append(freq)
            fh.write("\t".join(str(cell) for cell in cells) + "\n")


def write_epi(path: Path, probes: list[dict]) -> None:
    with open(path, "w") as fh:
        for p in probes:
            gene = p.get("gene", "NA")
            fh.write(f"{p['chr']}\t{p['probe_id']}\t0\t{p['bp']}\t{gene}\t+\n")


def write_besd_sparse_3f(
    path: Path,
    n_probes: int,
    probe_assocs: list[list[tuple[int, float, float]]],
    *,
    file_type: str = "3F",
) -> None:
    """Write a minimal SPARSE_FILE_TYPE_3F (or `file_type="3"`) BESD file.

    Type 3 differs only in its header: magic `3` and 15 reserved int32s
    before `val_num`. eQTLGen's `.besd` is type 3.

    BESD layout: for each probe p, val contains [betas | SEs] and rowid contains
    [snp_indices | zeros] at the same offsets. cols[2p]=beta_start, cols[2p+1]=se_start.
    """
    rowid: list[int] = []
    val: list[float] = []
    cols: list[int] = []
    offset = 0

    for assocs in probe_assocs:
        n = len(assocs)
        cols.append(offset)          # beta_start for this probe
        cols.append(offset + n)      # se_start for this probe
        # Beta positions: meaningful snp_idx and beta values
        for snp_idx, beta, _ in assocs:
            rowid.append(snp_idx)
            val.append(beta)
        # SE positions: snp_idx is don't-care; SE values follow immediately
        for _, _, se in assocs:
            rowid.append(0)
            val.append(se)
        offset += 2 * n

    cols.append(offset)   # final sentinel
    val_num = len(val)
    col_num = (n_probes << 1) + 1

    with open(path, "wb") as fh:
        if file_type == "3":
            fh.write(struct.pack("<16i", 3, *([0] * 15)))  # magic 3 + reserved
        else:
            fh.write(struct.pack("<I", 0x40400000))     # magic 3F
        fh.write(struct.pack("<Q", val_num))             # val_num
        fh.write(struct.pack(f"<{col_num}Q", *cols))    # column offsets
        fh.write(struct.pack(f"<{val_num}I", *rowid))   # row indices
        fh.write(struct.pack(f"<{val_num}f", *val))     # beta/SE values
