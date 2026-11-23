"""The Test Panel loader: the one way every test reads `tests/data/test-panel/` (ADR 0088).

The Test Panel is a real slice of HGDP+1kGP EUR (`CONTEXT.md`); its README says what is in it,
where each edge case is, and how it was made. A test reads it through these functions, never by
parsing the files, so a change of layout is one change here.

**Rows.** Every per-variant array is indexed by **Test Panel row**: the 0-based record order of
`panel.pvar`, header excluded. A row is not an ALID: ALIDs are ogld#3's to form, so each row
carries chrom, pos, ref and alt for a caller to form one.

**Orientation.** Every r here is the correlation of **ALT-allele dosages** (pvar ALT). Signing r
to the engine A1 is ogld#3's rule and the builder's job, not this loader's.

Loads are cached, and everything returned is read-only (arrays, tuples, mapping proxies), so one
test cannot change what the next reads. Only numpy and the standard library are used.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

TEST_PANEL = Path(__file__).resolve().parent / "data" / "test-panel"
KS = (0, 10)
CLUMP_R2 = (0.01, 0.001)


@dataclass(frozen=True)
class TestPanel:
    """The Test Panel's genotype files and sizes. `pcs` is its PC File (`#IID PC1 ... PC20`)."""

    __test__ = False  # a dataclass, not a pytest class

    pgen: Path
    pvar: Path
    psam: Path
    keep: Path
    pcs: Path
    n_samples: int
    n_variants: int


@dataclass(frozen=True)
class Variants:
    """One entry per Test Panel row, in pvar order.

    `chrom` is the engine's label (`"21"`, `"22"`); `pos` is GRCh38, 1-based; `id` is the pvar
    ID, `chrom:pos:REF:ALT`. `passes_filters` is whether the build's variant filters keep the row
    (MAC >= 20 on the 771 samples, FILTER PASS or `.`, nonzero variance), and `drop_reason` the
    first filter a dropped row fails (`""` when it passes).
    """

    chrom: npt.NDArray[np.str_]
    pos: npt.NDArray[np.int64]
    ref: npt.NDArray[np.str_]
    alt: npt.NDArray[np.str_]
    id: npt.NDArray[np.str_]
    passes_filters: npt.NDArray[np.bool_]
    drop_reason: npt.NDArray[np.str_]

    def __len__(self) -> int:
        return len(self.pos)


@dataclass(frozen=True)
class ReferencePairs:
    """r for every pair of passing rows on one chromosome at most 10 Mb apart, row_i < row_j.

    `r` is the correlation of the two rows' ALT dosages after regressing out an intercept and
    the first k PCs the PC Check keeps (`k` = 0: none), in float64. For the Test Panel's PC
    File the PC Check keeps PC1-PC10, so k = 10 is PC1-PC10 (manifest `pc_check`). A perfectly
    correlated pair can read a few ulps beyond +-1; quantising clamps it (ld-store.md section 3).
    """

    k: int
    row_i: npt.NDArray[np.int64]
    row_j: npt.NDArray[np.int64]
    r: npt.NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.r)


@dataclass(frozen=True)
class Gwas:
    """`z` per Test Panel row: the simulated GWAS z, NaN where the row fails the filters."""

    z: npt.NDArray[np.float64]


@dataclass(frozen=True)
class PcFile:
    """The Test Panel's PC File: `values[s, p]` is sample `iids[s]`'s value of PC `names[p]`."""

    iids: tuple[str, ...]
    names: tuple[str, ...]
    values: npt.NDArray[np.float64]


@dataclass(frozen=True)
class Dosages:
    """`plink2 --export A` for a few Test Panel rows (README, "Files"), an independent copy of
    their genotypes: `alt_counts[s, v]` is sample `iids[s]`'s ALT count at row `rows[v]`."""

    rows: npt.NDArray[np.int64]
    iids: tuple[str, ...]
    alt_counts: npt.NDArray[np.int8]


@dataclass(frozen=True)
class ClumpParams:
    """The `plink2 --clump` settings the clump goldens were made with (README, "The simulated
    GWAS"): leads have p <= `p1`, members p <= `p2`, within `kb` kilobases; p = erfc(|z|/sqrt 2).
    """

    p1: float
    p2: float
    kb: int
    r2: tuple[float, ...]


def _frozen(a: npt.ArrayLike, dtype: npt.DTypeLike) -> Any:
    out = np.array(a, dtype=dtype)
    out.flags.writeable = False
    return out


def _read_only(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _read_only(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_read_only(v) for v in value)
    return value


@cache
def manifest() -> Mapping[str, Any]:
    """`manifest.json`: every parameter, input, tool version, count and file SHA-256, read-only
    all the way down (objects are mapping proxies, arrays are tuples)."""
    data: Mapping[str, Any] = _read_only(json.loads((TEST_PANEL / "manifest.json").read_text()))
    return data


@cache
def panel() -> TestPanel:
    counts = manifest()["counts"]
    return TestPanel(
        pgen=TEST_PANEL / "panel.pgen",
        pvar=TEST_PANEL / "panel.pvar",
        psam=TEST_PANEL / "panel.psam",
        keep=TEST_PANEL / "panel.keep",
        pcs=TEST_PANEL / "pcs.eigenvec",
        n_samples=int(counts["n_samples"]),
        n_variants=int(counts["n_variants"]),
    )


@cache
def variants() -> Variants:
    """Per row, from `panel.pvar` and `variants.tsv`, which must name the same records."""
    pvar = [line.split("\t") for line in panel().pvar.read_text().splitlines()[1:]]
    tsv = [line.split("\t") for line in (TEST_PANEL / "variants.tsv").read_text().splitlines()]
    col = {name: i for i, name in enumerate(tsv[0])}
    rows = tsv[1:]
    if [(p[0], p[1], p[3], p[4]) for p in pvar] != [
        (r[col["chrom"]], r[col["pos"]], r[col["ref"]], r[col["alt"]]) for r in rows
    ]:
        raise ValueError("panel.pvar and variants.tsv name different records")
    reason = [r[col["drop_reason"]] for r in rows]
    return Variants(
        chrom=_frozen([p[0] for p in pvar], np.str_),
        pos=_frozen([p[1] for p in pvar], np.int64),
        ref=_frozen([p[3] for p in pvar], np.str_),
        alt=_frozen([p[4] for p in pvar], np.str_),
        id=_frozen([p[2] for p in pvar], np.str_),
        passes_filters=_frozen([r[col["passes_filters"]] == "1" for r in rows], np.bool_),
        drop_reason=_frozen(["" if x == "." else x for x in reason], np.str_),
    )


@cache
def _pairs() -> dict[str, Any]:
    with np.load(TEST_PANEL / "reference" / "r.npz") as f:
        return {name: f[name] for name in f.files}


@cache
def reference_r(k: int) -> ReferencePairs:
    """The independent reference r at k = 0 or k = 10 (README, "Files")."""
    if k not in KS:
        raise ValueError(f"the Test Panel holds reference r at k in {KS}, not k = {k}")
    p = _pairs()
    return ReferencePairs(
        k,
        _frozen(p["row_i"], np.int64),
        _frozen(p["row_j"], np.int64),
        _frozen(p[f"r_k{k}"], np.float64),
    )


@cache
def plink2_r() -> ReferencePairs:
    """`plink2 --r-unphased` at k = 0 for exactly the pairs of `reference_r(0)`, in its order."""
    p = _pairs()
    with np.load(TEST_PANEL / "plink2" / "r_unphased_k0.npz") as f:
        r = f["r"]
    return ReferencePairs(
        0, _frozen(p["row_i"], np.int64), _frozen(p["row_j"], np.int64), _frozen(r, np.float64)
    )


@cache
def gwas() -> Gwas:
    rows = [line.split("\t") for line in (TEST_PANEL / "gwas" / "z.tsv").read_text().splitlines()]
    if [int(r[0]) for r in rows[1:]] != list(range(len(rows) - 1)):
        raise ValueError("gwas/z.tsv is not one line per Test Panel row, in order")
    return Gwas(_frozen([float(r[1]) for r in rows[1:]], np.float64))


@cache
def clump_params() -> ClumpParams:
    c = manifest()["clump"]
    return ClumpParams(
        float(c["p1"]), float(c["p2"]), int(c["kb"]), tuple(float(x) for x in c["r2"])
    )


@cache
def pcs() -> PcFile:
    lines = [line.split("\t") for line in panel().pcs.read_text().splitlines()]
    return PcFile(
        tuple(f[0] for f in lines[1:]),
        tuple(lines[0][1:]),
        _frozen([[float(x) for x in f[1:]] for f in lines[1:]], np.float64),
    )


@cache
def dosages() -> Dosages:
    """`dosages.raw`: plink2's sample-major `--export A` table; each variant column is named
    `r<row>_<counted allele>`, and the counted allele is the row's ALT."""
    lines = [line.split() for line in (TEST_PANEL / "dosages.raw").read_text().splitlines()]
    head = lines[0]
    first = head.index("PHENOTYPE") + 1
    names = [h.split("_", 1) for h in head[first:]]
    rows = [int(name.removeprefix("r")) for name, _ in names]
    alt = variants().alt
    if any(allele != alt[row] for row, (_, allele) in zip(rows, names, strict=True)):
        raise ValueError("dosages.raw does not count each row's ALT allele")
    return Dosages(
        _frozen(rows, np.int64),
        tuple(f[head.index("IID")] for f in lines[1:]),
        _frozen([[int(x) for x in f[first:]] for f in lines[1:]], np.int8),
    )


@cache
def clump_leads(r2: float) -> tuple[int, ...]:
    """plink2's `--clump` lead rows at k = 0 and `clump_params()`, for r2 0.01 or 0.001, in
    plink2's output order: most significant first."""
    if r2 not in CLUMP_R2:
        raise ValueError(f"the Test Panel holds clumps at r2 in {CLUMP_R2}, not {r2}")
    lines = (TEST_PANEL / "plink2" / f"clump_r2_{r2}.clumps").read_text().splitlines()
    col = lines[0].lstrip("#").split("\t").index("ID")
    return tuple(int(line.split("\t")[col].removeprefix("r")) for line in lines[1:])
