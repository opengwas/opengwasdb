"""ogld#5's realdata contract: the builder against plink2 on real HGDP+1kGP EUR chr22.

The Test Panel proves the logic; this proves it survives real genotypes (CLAUDE.md). It
cuts a 3 Mb slice of the shipped EUR chr22 panel with plink2, builds it at k = 0, and holds
300 of its variants' Near Band to `plink2 --r-unphased` within the quantisation step: epic
#1's acceptance that the k = 0 builder matches plink2 within int16 rounding.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import ogld
from ogld.identity import orient

PANELS = Path("/data/opengwasdb/reference/hgdp1kgp-hg38")
EUR_CHR22 = PANELS / "pgen" / "EUR" / "chr22"
EUR_KEEP = PANELS / "members" / "EUR.keep"
#: plink2 prints r to six significant digits.
PRINTED = 5e-7


def plink2(*args: str | Path) -> None:
    exe = os.environ.get("OGLD_PLINK2", "")
    assert exe, "set OGLD_PLINK2, or run this tier through `pixi run test-realdata`"
    subprocess.run(
        [exe, *map(str, args), "--threads", "4", "--memory", "8000"],
        check=True,
        capture_output=True,
    )


def pvar_rows(path: Path) -> list[list[str]]:
    lines = path.read_text().splitlines()
    return [line.split("\t") for line in lines if not line.startswith("#")]


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="ogld#5")
def test_the_builder_matches_plink2_on_a_real_chr22_slice(tmp_path: Path) -> None:
    window = ["--chr", "22", "--from-bp", "20000000", "--to-bp", "23000000"]
    plink2(
        "--pfile",
        EUR_CHR22,
        "vzs",
        "--keep",
        EUR_KEEP,
        *window,
        "--make-pgen",
        "--out",
        tmp_path / "slice",
    )
    inputs = ogld.BuildInputs(
        pgen=(tmp_path / "slice.pgen",),
        keep=EUR_KEEP,
        pcs=None,
        pc_source=None,
        panel_id="hgdp1kgp-hg38-EUR-chr22-20-23Mb",
        ancestry="EUR",
        genotype_source=f"{EUR_CHR22}.pgen, chr22:20,000,000-23,000,000",
        k=0,
        info_min=None,
    )
    ogld.build(inputs, tmp_path / "store")
    store = ogld.open(tmp_path / "store")

    rows = pvar_rows(tmp_path / "slice.pvar")
    found = [orient(r[0], int(r[1]), r[3], r[4]) for r in rows]
    m = store.matrix([o.alid for o in found])
    present = [i for i, o in enumerate(found) if o.alid not in set(m.absent)]
    sample = present[:: max(1, len(present) // 300)][:300]
    assert len(sample) == 300, "the slice must hold 300 variants that pass the filters"
    (tmp_path / "sample.txt").write_text("\n".join(rows[i][2] for i in sample) + "\n")
    plink2(
        "--pfile",
        tmp_path / "slice",
        "--extract",
        tmp_path / "sample.txt",
        "--r-unphased",
        "square",
        "--out",
        tmp_path / "r",
    )

    theirs = np.loadtxt(tmp_path / "r.unphased.vcor1")
    order = (tmp_path / "r.unphased.vcor1.vars").read_text().split()
    by_id = {rows[i][2]: found[i] for i in sample}
    sign = np.array([1.0 if by_id[v].a1_is_alt else -1.0 for v in order])
    ours = store.matrix([by_id[v].alid for v in order]).r
    # r of A1 dosages is r of ALT dosages times both variants' signs (spec §3).
    worst = np.abs(ours - theirs * sign[:, None] * sign[None, :]).max()
    assert worst <= 1 / 65534 + PRINTED, f"the builder is {worst} from plink2"
