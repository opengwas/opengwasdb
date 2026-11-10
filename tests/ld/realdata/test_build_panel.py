"""ogld#5's realdata contract: the builder on real HGDP+1kGP EUR chr22.

The Test Panel proves the logic; this proves it survives real genotypes (CLAUDE.md).

- The builds read the phased HGDP+1kGP v2 callset, which has no missing calls (owner,
  7 November 2026; ADR 0002). `OGLD_V2_EUR_CHR22` names its chr22 BCF
  (`docs/spec/build-inputs.md` §10), and the suite converts chr22:20–32 Mb for the EUR samples
  once, with §3's flags. On 20–23 Mb of it, the k = 0 Near Band of 300 variants matches
  `plink2 --r-unphased` within the quantisation step: epic #1's acceptance.
- Real LD survives PC Adjustment: on the 12 Mb slice, pairs within 100 kb change little from
  k = 0 to k = 10, and the provenance records the change's quantiles. 12 Mb, not 3, because
  the PC Check scans the build's own input and needs windows apart from the best one.
  `OGLD_EUR_PC_FILE` names EUR's PC File from `scripts/make_pcs.py` (§10).
- The gnomAD v3.1.2 pgens shipped under /data/opengwasdb/reference/ have missing calls, at
  11,725 of the 15,272 variants of chr22:20–23 Mb. The builder refuses them.

An unset or missing `OGLD_PLINK2`, `OGLD_V2_EUR_CHR22` or `OGLD_EUR_PC_FILE` stops a test as an
expected failure naming it (`realdata_inputs.required`), before ogld or plink2 runs.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from realdata_inputs import required

import ogld
from ogld.identity import orient

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
PANELS = Path("/data/opengwasdb/reference/hgdp1kgp-hg38")
SHIPPED_EUR_CHR22 = PANELS / "pgen" / "EUR" / "chr22"
EUR_KEEP = PANELS / "members" / "EUR.keep"
NARROW = ["--chr", "22", "--from-bp", "20000000", "--to-bp", "23000000"]
WIDE = ["--chr", "22", "--from-bp", "20000000", "--to-bp", "32000000"]
#: plink2 prints r to six significant digits.
PRINTED = 5e-7
_SLICES: dict[str, Path] = {}


def plink2(*args: str | Path) -> None:
    exe = required("OGLD_PLINK2", "plink2, the independent reference; run.sh finds one")
    subprocess.run(
        [str(exe), *map(str, args), "--threads", "4", "--memory", "8000"],
        check=True,
        capture_output=True,
    )


def section_3_conversion(source: Path, out: Path) -> list[str]:
    """build-inputs.md §3's conversion of `source` for the EUR samples, as lane 2d's Test Panel
    script spells it, so the flags live in one place."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from test_panel.tools import conversion_args

    args: list[str] = conversion_args(source, EUR_KEEP, out)
    return args


def v2_slice(name: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The v2 EUR chr22 slice `name` ("wide" 20–32 Mb, "narrow" 20–23 Mb), once per session."""
    if name not in _SLICES:
        bcf = required("OGLD_V2_EUR_CHR22", "the phased v2 chr22 BCF, build-inputs.md §10")
        out = tmp_path_factory.mktemp(f"v2-{name}") / "slice"
        if name == "wide":
            plink2(*section_3_conversion(bcf, out), *WIDE)
        else:
            wide = v2_slice("wide", tmp_path_factory)
            plink2("--pfile", wide, *NARROW, "--make-pgen", "pvar-cols=filter", "--out", out)
        _SLICES[name] = out
    return _SLICES[name]


def pvar_rows(path: Path) -> list[list[str]]:
    lines = path.read_text().splitlines()
    return [line.split("\t") for line in lines if not line.startswith("#")]


def inputs(pgen: Path, source: str, pcs: Path | None = None) -> ogld.BuildInputs:
    return ogld.BuildInputs(
        pgen=(pgen,),
        keep=EUR_KEEP,
        pcs=pcs,
        pc_source="ogld scripts/make_pcs.py on HGDP+1kGP v2, EUR" if pcs else None,
        panel_id="hgdp1kgp-v2-EUR-chr22-slice",
        ancestry="EUR",
        genotype_source=source,
        k=10 if pcs else 0,
        info_min=None,
    )


def test_the_builder_refuses_the_shipped_pgen_for_its_missing_calls(tmp_path: Path) -> None:
    slice_ = tmp_path / "shipped"
    plink2(
        "--pfile",
        SHIPPED_EUR_CHR22,
        "vzs",
        "--keep",
        EUR_KEEP,
        *NARROW,
        "--make-pgen",
        "--out",
        slice_,
    )
    plink2("--pfile", slice_, "--missing", "variant-only", "--out", tmp_path / "missing")
    counts = np.loadtxt(tmp_path / "missing.vmiss", usecols=2, skiprows=1, dtype=np.int64)
    assert (counts > 0).any(), "the shipped slice must have missing calls for this to mean anything"
    with pytest.raises(ogld.InputError, match="missing"):
        ogld.build(
            inputs(tmp_path / "shipped.pgen", "gnomAD v3.1.2, shipped pgen"), tmp_path / "store"
        )


def test_the_builder_matches_plink2_on_a_real_chr22_slice(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    slice_ = v2_slice("narrow", tmp_path_factory)
    ogld.build(inputs(slice_.with_suffix(".pgen"), "v2 EUR chr22:20-23 Mb"), tmp_path / "store")
    store = ogld.open(tmp_path / "store")

    rows = pvar_rows(slice_.with_suffix(".pvar"))
    found = [orient(r[0], int(r[1]), r[3], r[4]) for r in rows]
    # 3,000 candidates spread over the slice: a matrix of all of them stays small.
    spread = list(range(0, len(found), max(1, len(found) // 3000)))
    m = store.matrix([found[i].alid for i in spread])
    # A reversed-indel pair is two records with one ALID, and the store holds one of them
    # (build-inputs §6): compare only records whose ALID no other record shares.
    shared = Counter(o.alid for o in found)
    present = [
        i for i in spread if found[i].alid not in set(m.absent) and shared[found[i].alid] == 1
    ]
    sample = present[:: max(1, len(present) // 300)][:300]
    assert len(sample) == 300, "the slice must hold 300 variants that pass the filters"
    (tmp_path / "sample.txt").write_text("\n".join(rows[i][2] for i in sample) + "\n")
    plink2(
        "--pfile",
        slice_,
        "--extract",
        tmp_path / "sample.txt",
        "--r-unphased",
        "square",
        "ref-based",  # plink2 otherwise signs r to each variant's major allele
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


def test_real_ld_survives_pc_adjustment(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # Epic #1 and ogld#5: pairs within 100 kb change little outside the Long-Range LD
    # Regions (none is in chr22:20–32 Mb), and the distribution is recorded.
    pcs = required("OGLD_EUR_PC_FILE", "EUR's PC File from scripts/make_pcs.py, §10")
    slice_ = v2_slice("wide", tmp_path_factory)
    pgen = slice_.with_suffix(".pgen")
    ogld.build(inputs(pgen, "v2 EUR chr22:20-32 Mb"), tmp_path / "k0")
    adjusted = ogld.build(inputs(pgen, "v2 EUR chr22:20-32 Mb", pcs), tmp_path / "k10")

    unadjusted_store, adjusted_store = ogld.open(tmp_path / "k0"), ogld.open(tmp_path / "k10")
    # Twelve regions 100 kb wide, one per Mb: every pair inside one is within 100 kb, and
    # each matrix stays a few thousand variants square.
    changes = []
    for centre in range(20_500_000, 32_000_000, 1_000_000):
        before = unadjusted_store.region("22", centre, 50_000)
        after = adjusted_store.region("22", centre, 50_000)
        assert list(before.variants.alid) == list(after.variants.alid)
        upper = np.triu_indices(len(before.variants), k=1)
        changes.append(after.r[upper] - before.r[upper])
    change = np.concatenate(changes)
    assert len(change) > 10_000, "the slice must hold many pairs within 100 kb"
    assert np.median(np.abs(change)) < 0.01, (
        f"real LD moved: median |change| {np.median(np.abs(change))}"
    )
    diagnostic = adjusted["pc_diagnostic"]
    assert diagnostic is not None
    recorded = diagnostic["near_100kb_change"]
    assert set(recorded) == {"0", "0.5", "0.9", "0.99", "1"}
    # The provenance's quantiles are over every pair within 100 kb; these regions sample them.
    assert abs(recorded["0.5"] - float(np.quantile(change, 0.5))) <= 0.01
