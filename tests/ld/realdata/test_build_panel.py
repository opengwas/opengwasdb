"""ogld#5's realdata contract: the builder on real HGDP+1kGP EUR chr22.

The Test Panel proves the logic; this proves it survives real genotypes (CLAUDE.md).

- The builds read the phased HGDP+1kGP v2 callset, which has no missing calls (owner,
  7 November 2026; ADR 0002). On a 3 Mb slice of its EUR chr22, the k = 0 Near Band of 300
  variants matches `plink2 --r-unphased` within the quantisation step: epic #1's acceptance.
  `OGLD_V2_EUR_CHR22` names that chromosome's pgen fileset, converted as
  `docs/spec/build-inputs.md` says.
- Real LD survives PC Adjustment: on the same slice, pairs within 100 kb change little from
  k = 0 to k = 10, and the provenance records the change's quantiles. `OGLD_EUR_PC_FILE`
  names EUR's PC File, made by the recipe script.
- The gnomAD v3.1.2 pgens shipped under /data/opengwasdb/reference/ have missing calls, at
  11,725 of the 15,272 variants of the same slice. The builder refuses them.
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
SHIPPED_EUR_CHR22 = PANELS / "pgen" / "EUR" / "chr22"
EUR_KEEP = PANELS / "members" / "EUR.keep"
WINDOW = ["--chr", "22", "--from-bp", "20000000", "--to-bp", "23000000"]
#: plink2 prints r to six significant digits.
PRINTED = 5e-7
OGLD_5 = pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="ogld#5")


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


def inputs(pgen: Path, source: str, pcs: Path | None = None) -> ogld.BuildInputs:
    return ogld.BuildInputs(
        pgen=(pgen,),
        keep=EUR_KEEP,
        pcs=pcs,
        pc_source="the recipe script's EUR PC File" if pcs else None,
        panel_id="hgdp1kgp-hg38-EUR-chr22-20-23Mb",
        ancestry="EUR",
        genotype_source=source,
        k=10 if pcs else 0,
        info_min=None,
    )


@OGLD_5
def test_the_builder_refuses_the_shipped_pgen_for_its_missing_calls(tmp_path: Path) -> None:
    slice_ = tmp_path / "shipped"
    plink2(
        "--pfile",
        SHIPPED_EUR_CHR22,
        "vzs",
        "--keep",
        EUR_KEEP,
        *WINDOW,
        "--make-pgen",
        "--out",
        slice_,
    )
    plink2("--pfile", slice_, "--missing", "variant-only", "--out", tmp_path / "missing")
    counts = np.loadtxt(tmp_path / "missing.vmiss", usecols=2, skiprows=1, dtype=np.int64)
    assert (counts > 0).any(), "the shipped slice must have missing calls for this to mean anything"
    with pytest.raises(ValueError, match="missing"):
        ogld.build(
            inputs(tmp_path / "shipped.pgen", "gnomAD v3.1.2, shipped pgen"), tmp_path / "store"
        )


@OGLD_5
def test_the_builder_matches_plink2_on_a_real_chr22_slice(tmp_path: Path) -> None:
    v2 = os.environ.get("OGLD_V2_EUR_CHR22", "")
    assert v2, "set OGLD_V2_EUR_CHR22 to the phased v2 EUR chr22 pgen fileset (no suffix)"
    plink2("--pfile", v2, "--keep", EUR_KEEP, *WINDOW, "--make-pgen", "--out", tmp_path / "slice")
    ogld.build(inputs(tmp_path / "slice.pgen", f"{v2}, chr22:20-23 Mb"), tmp_path / "store")
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


@OGLD_5
def test_real_ld_survives_pc_adjustment(tmp_path: Path) -> None:
    # Epic #1 and ogld#5: pairs within 100 kb change little outside the Long-Range LD
    # Regions (none is in chr22:20–23 Mb), and the distribution is recorded.
    v2 = os.environ.get("OGLD_V2_EUR_CHR22", "")
    pcs = os.environ.get("OGLD_EUR_PC_FILE", "")
    assert v2, "set OGLD_V2_EUR_CHR22 to the phased v2 EUR chr22 pgen fileset (no suffix)"
    assert pcs, "set OGLD_EUR_PC_FILE to EUR's PC File from the recipe script"
    plink2("--pfile", v2, "--keep", EUR_KEEP, *WINDOW, "--make-pgen", "--out", tmp_path / "slice")
    pgen = tmp_path / "slice.pgen"
    ogld.build(inputs(pgen, f"{v2}, chr22:20-23 Mb"), tmp_path / "k0")
    adjusted = ogld.build(inputs(pgen, f"{v2}, chr22:20-23 Mb", Path(pcs)), tmp_path / "k10")

    rows = pvar_rows(tmp_path / "slice.pvar")
    found = [orient(r[0], int(r[1]), r[3], r[4]) for r in rows]
    unadjusted_store, adjusted_store = ogld.open(tmp_path / "k0"), ogld.open(tmp_path / "k10")
    present = unadjusted_store.matrix([o.alid for o in found])
    names = list(present.variants.alid)
    before, after = present.r, adjusted_store.matrix(names).r
    pos = present.variants.position.astype(np.int64)
    near = (np.abs(pos[:, None] - pos[None, :]) <= 100_000) & ~np.eye(len(names), dtype=bool)
    assert near.sum() > 10_000, "the slice must hold many pairs within 100 kb"
    change = np.abs(after[near] - before[near])
    assert np.median(change) < 0.01, f"real LD moved: median |change| {np.median(change)}"
    diagnostic = adjusted["pc_diagnostic"]
    assert diagnostic is not None
    recorded = diagnostic["near_100kb_change"]
    assert set(recorded) == {"0", "0.5", "0.9", "0.99", "1"}
    assert abs(recorded["0.5"] - float(np.quantile(after[near] - before[near], 0.5))) <= 2 / 32767
