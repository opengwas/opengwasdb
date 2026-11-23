"""ogld#5's realdata contract: the builder on real HGDP+1kGP EUR chr22.

The Test Panel proves the logic; this proves it survives real genotypes (CLAUDE.md).

- The builds read the phased HGDP+1kGP v2 callset, which has no missing calls (owner,
  7 November 2026; ADR 0002). `OGLD_V2_EUR_CHR22` names its chr22 BCF
  (`docs/spec/build-inputs.md` §10), and the suite converts chr22:20–32 Mb once, with §3's flags,
  for EUR's 665 unrelated samples: `OGLD_EUR_KEEP` names the keep file `scripts/make_keeps.py`
  writes (ADR 0002, decision 11). On 20–23 Mb of it, the k = 0 Near Band matches
  `plink2 --r-unphased` within the quantisation step: epic #1's acceptance. The variants compared
  include every choice build-inputs §6's reversed-indel rule makes there (both records at MAC >=
  20, judged here from plink2's own allele counts): each must keep the record whose REF is A1.
- Real LD survives PC Adjustment: on the 12 Mb slice, pairs within 100 kb change little from
  k = 0 to k = 10, and each of the provenance's five quantiles of that change is its own
  computation over every such pair, from the two stores' Near Bands. 12 Mb, not 3, because the
  PC Check scans the build's own input and needs windows apart from the best one.
  `OGLD_EUR_PC_FILE` names EUR's PC File from `scripts/make_pcs.py` (§10).
- The gnomAD v3.1.2 pgens shipped under /data/opengwasdb/reference/ have missing calls, at
  11,725 of the 15,272 variants of chr22:20–23 Mb. The builder refuses them.

An unset or missing `OGLD_PLINK2`, `OGLD_V2_EUR_CHR22`, `OGLD_EUR_KEEP` or `OGLD_EUR_PC_FILE`
stops a test as an expected failure naming it (`realdata_inputs.required`), before ogld or plink2
runs.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import zarr
from realdata_inputs import required

import opengwasdb.ld
from opengwasdb.ld.build.diagnostic import outside_regions
from opengwasdb.ld.identity import Orientation, orient

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "ld"
PANELS = Path("/data/opengwasdb/reference/hgdp1kgp-hg38")
SHIPPED_EUR_CHR22 = PANELS / "pgen" / "EUR" / "chr22"
NARROW = ["--chr", "22", "--from-bp", "20000000", "--to-bp", "23000000"]
WIDE = ["--chr", "22", "--from-bp", "20000000", "--to-bp", "32000000"]
#: plink2 prints r to six significant digits.
PRINTED = 5e-7
#: HGDP+1kGP's encoding, which ogld#7 builds every panel at (ADR 0004): every build here uses it.
HGDP_SCALE = 511
#: Half the bins the PC Diagnostic counts the change in r into, 2⁻²⁰ wide (spec §7).
HALF_BIN = 2.0**-21
_SLICES: dict[str, Path] = {}


def plink2(*args: str | Path) -> None:
    exe = required("OGLD_PLINK2", "plink2, the independent reference; run.sh finds one")
    subprocess.run(
        [str(exe), *map(str, args), "--threads", "4", "--memory", "8000"],
        check=True,
        capture_output=True,
    )


def eur_keep() -> Path:
    return required("OGLD_EUR_KEEP", "EUR's unrelated keep file, scripts/make_keeps.py, §10")


def section_3_conversion(source: Path, out: Path) -> list[str]:
    """build-inputs.md §3's conversion of `source` for the EUR samples, as lane 2d's Test Panel
    script spells it, so the flags live in one place."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from test_panel.tools import conversion_args

    args: list[str] = conversion_args(source, eur_keep(), out)
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


def inputs(
    pgen: Path, source: str, r_scale: int, pcs: Path | None = None
) -> opengwasdb.ld.BuildInputs:
    return opengwasdb.ld.BuildInputs(
        pgen=(pgen,),
        keep=eur_keep(),
        pcs=pcs,
        pc_source="ogld scripts/make_pcs.py on HGDP+1kGP v2, EUR" if pcs else None,
        panel_id="hgdp1kgp-v2-EUR-chr22-slice",
        ancestry="EUR",
        genotype_source=source,
        r_scale=r_scale,
        t_partner=0.01,
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
        eur_keep(),
        *NARROW,
        "--make-pgen",
        "--out",
        slice_,
    )
    plink2("--pfile", slice_, "--missing", "variant-only", "--out", tmp_path / "missing")
    counts = np.loadtxt(tmp_path / "missing.vmiss", usecols=2, skiprows=1, dtype=np.int64)
    assert (counts > 0).any(), "the shipped slice must have missing calls for this to mean anything"
    with pytest.raises(opengwasdb.ld.InputError, match="missing"):
        opengwasdb.ld.build(
            inputs(tmp_path / "shipped.pgen", "gnomAD v3.1.2, shipped pgen", HGDP_SCALE),
            tmp_path / "store",
        )


@dataclass(frozen=True)
class Kept:
    """The slice's records the store holds, judged here, not by the builder (build-inputs §6):
    MAC >= 20 from plink2's own allele counts (v2's FILTER is all `.`, and none of its variants
    lacks variance); of a reversed-indel pair that both pass, the record whose REF is A1. Those
    pairs are `choices`, the only ones the rule decides; of a pair one passes, that one."""

    records: list[int]
    choices: list[int]
    ref_frequency: npt.NDArray[np.float64]


def kept_records(slice_: Path, found: list[Orientation], tmp_path: Path) -> Kept:
    plink2("--pfile", slice_, "--freq", "counts", "--out", tmp_path / "freq")
    table = [line.split("\t") for line in (tmp_path / "freq.acount").read_text().splitlines()]
    head = [name.lstrip("#") for name in table[0]]
    alt = np.array([int(row[head.index("ALT_CTS")]) for row in table[1:]])
    obs = np.array([int(row[head.index("OBS_CT")]) for row in table[1:]])
    by_alid: dict[str, list[int]] = {}
    for i in np.flatnonzero(np.minimum(alt, obs - alt) >= 20):
        by_alid.setdefault(found[int(i)].alid, []).append(int(i))
    choices = sorted(
        next(i for i in g if not found[i].a1_is_alt) for g in by_alid.values() if len(g) > 1
    )
    records = sorted({g[0] for g in by_alid.values() if len(g) == 1} | set(choices))
    return Kept(records, choices, 1 - alt / obs)


def test_the_builder_matches_plink2_on_a_real_chr22_slice(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    slice_ = v2_slice("narrow", tmp_path_factory)
    narrow = inputs(slice_.with_suffix(".pgen"), "v2 EUR chr22:20-23 Mb", HGDP_SCALE)
    opengwasdb.ld.build(narrow, tmp_path / "store")
    store = opengwasdb.ld.open(tmp_path / "store")

    rows = pvar_rows(slice_.with_suffix(".pvar"))
    found = [orient(r[0], int(r[1]), r[3], r[4]) for r in rows]
    kept = kept_records(slice_, found, tmp_path)
    # Every reversed-indel pair whose two records both pass is a choice §6 makes: the store must
    # hold the record whose REF is A1, which its A1 frequency, that record's REF frequency, shows.
    assert len(kept.choices) >= 20, "the slice must hold 20 choices for this to mean anything"
    held = store.matrix([found[i].alid for i in kept.choices])
    assert held.absent == ()
    frequency = held.variants.a1_frequency
    assert np.allclose(frequency, kept.ref_frequency[kept.choices], atol=1e-6), "REF = A1 kept"
    # And r: 300 records spread over the slice and every choice, against plink2 (§7: the k = 0
    # criterion holds for every pair).
    sample = sorted({*kept.records[:: max(1, len(kept.records) // 300)][:300], *kept.choices})
    assert store.matrix([found[i].alid for i in sample]).absent == (), "the store holds each"
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
    assert worst <= 1 / (2 * HGDP_SCALE) + PRINTED, f"the builder is {worst} from plink2"


def band_within(store: Path, distance: int) -> tuple[npt.NDArray[np.float64], int]:
    """Every chr22 Near Band pair of `store` within `distance`, as Stored r, row after row, and
    the scale the store records, which its codes are read at."""
    root = zarr.open_group(str(store), mode="r", zarr_format=3)
    position = np.asarray(root["variants/position"][:]).astype(np.int64)
    indptr = np.asarray(root["near/22/indptr"][:])
    codes = np.asarray(root["near/22/r"][:])
    reach = np.searchsorted(position, position + distance, side="right")
    lengths = reach - np.arange(len(position)) - 1
    offset = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    scale: int = root.attrs["ld_store"]["encoding"]["r_scale"]
    assert scale == HGDP_SCALE, "the store records the scale the test built it at"
    stored: npt.NDArray[np.float64] = codes[np.repeat(indptr[:-1], lengths) + offset] / scale
    return stored, scale


def test_real_ld_survives_pc_adjustment(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # Epic #1 and ogld#5: pairs within 100 kb change little outside the Long-Range LD
    # Regions (none is in chr22:20–32 Mb), and the distribution is recorded.
    pcs = required("OGLD_EUR_PC_FILE", "EUR's PC File from scripts/make_pcs.py, §10")
    slice_ = v2_slice("wide", tmp_path_factory)
    pgen = slice_.with_suffix(".pgen")
    opengwasdb.ld.build(inputs(pgen, "v2 EUR chr22:20-32 Mb", HGDP_SCALE), tmp_path / "k0")
    adjusted = opengwasdb.ld.build(
        inputs(pgen, "v2 EUR chr22:20-32 Mb", HGDP_SCALE, pcs), tmp_path / "k10"
    )
    held = [zarr.open_group(str(tmp_path / s), mode="r")["variants"] for s in ("k0", "k10")]
    assert list(held[0]["alid"][:]) == list(held[1]["alid"][:]), "one Variant Set, both builds"
    assert outside_regions("22", np.asarray(held[0]["position"][:]).astype(np.int64)).all()

    (after, scale), (before, _) = (band_within(tmp_path / s, 100_000) for s in ("k10", "k0"))
    change = after - before
    assert len(change) > 500_000, "the slice must hold many pairs within 100 kb"
    assert np.median(np.abs(change)) < 0.01, (
        f"real LD moved: median |change| {np.median(np.abs(change))}"
    )
    diagnostic = adjusted["pc_diagnostic"]
    assert diagnostic is not None
    recorded = diagnostic["near_100kb_change"]
    assert set(recorded) == {"0", "0.5", "0.9", "0.99", "1"}
    # The provenance's quantiles are of the computed change; these are of the Stored r's. Each
    # Stored r is within half the store's step of its r, so a change in Stored r is within one
    # step of the computed change, and an order statistic moves no further; the recorded quantile
    # is the computed one to within half a bin. At × 511 that is 1.96×10⁻³ (#16).
    tolerance = 1 / scale + HALF_BIN
    for q in (0.0, 0.5, 0.9, 0.99, 1.0):
        ours = float(np.quantile(change, q, method="lower"))
        assert abs(recorded[f"{q:g}"] - ours) <= tolerance, (q, recorded[f"{q:g}"], ours)
