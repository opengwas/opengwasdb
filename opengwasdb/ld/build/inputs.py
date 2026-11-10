"""Reading a build's inputs: the keep file, each fileset's psam and pvar, and the PC File.

`docs/spec/build-inputs.md` §2, §4 and §5 say what each must hold. Every refusal is an
`InputError` that names the panel and, where there is one, the file, sample or record: a build
stops rather than guess (CONTRIBUTING.md, "Fail loudly over degrading quietly").

The samples have one order through the whole build, the **sample order**: the order of the
kept samples in the first fileset's psam. pgenlib returns a fileset's kept samples in its own
psam order, so each fileset carries the permutation to the sample order, and the PC File is
reindexed to it. A covariate aligned to the wrong sample is a silent error, which is why there
is one order and one place that builds it.
"""

from __future__ import annotations

import hashlib
import importlib
import itertools
import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from ogld.format import InputError

#: Records a build reads at a time: bounds the genotype block a pass holds.
CHUNK_RECORDS: Final = 65_536
#: The columns a pvar must have, by name (build-inputs §2); INFO is read only with an INFO floor.
REQUIRED_COLUMNS: Final = ("#CHROM", "POS", "ID", "REF", "ALT")


@dataclass(frozen=True)
class Samples:
    """The kept samples: their IIDs in the sample order, and the keep file they came from."""

    iids: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.iids)

    def sha256(self) -> str:
        """`panel.samples_sha256`: SHA-256 of the sorted IIDs, one per line (spec §7)."""
        return hashlib.sha256(("\n".join(sorted(self.iids)) + "\n").encode()).hexdigest()


@dataclass(frozen=True)
class Fileset:
    """One pgen, its pvar and psam, and how its kept samples map to the sample order.

    `subset` is the psam index of each kept sample, ascending, as pgenlib takes it; pgenlib's
    columns are in that order, and `order` reorders them to the sample order (None when they
    already are).
    """

    pgen: Path
    pvar: Path
    psam: Path
    subset: npt.NDArray[np.uint32]
    order: npt.NDArray[np.int64] | None


@dataclass(frozen=True)
class Records:
    """A block of consecutive pvar records of one fileset, from pgen row `first`."""

    first: int
    chrom: list[str]
    pos: npt.NDArray[np.int64]
    ids: list[str]
    ref: list[str]
    alt: list[str]
    filter: list[str] | None
    info: list[str] | None

    def __len__(self) -> int:
        return len(self.ids)


def read_keep(keep: Path, panel: str) -> list[str]:
    """The keep file's IIDs, in file order (build-inputs §4): a `#IID` header, one per line."""
    lines = [line.strip() for line in _text(keep, panel).splitlines()]
    if not lines or lines[0].split() != ["#IID"]:
        raise InputError(f"panel {panel}: the keep file {keep.name} must start with '#IID'")
    listed = [line for line in lines[1:] if line]
    if not listed:
        raise InputError(f"panel {panel}: the keep file {keep.name} lists no sample")
    _refuse_repeats(listed, f"the keep file {keep.name}", panel)
    return listed


def _refuse_repeats(listed: list[str], where: str, panel: str) -> None:
    repeated = sorted(iid for iid, count in Counter(listed).items() if count > 1)
    if repeated:
        raise InputError(f"panel {panel}: {where} has a duplicate IID: {', '.join(repeated[:5])}")


def _text(path: Path, panel: str) -> str:
    if not path.is_file():
        raise InputError(f"panel {panel}: {path} does not exist")
    return path.read_text()


def _iid_column(header: list[str], where: str, panel: str) -> int:
    """The IID column of a `#IID …` or `#FID IID …` header."""
    names = [header[0].lstrip("#"), *header[1:]]
    if "IID" not in names:
        raise InputError(f"panel {panel}: {where} has no IID column in its header")
    return names.index("IID")


def psam_iids(psam: Path, panel: str) -> list[str]:
    """A psam's IIDs, in its order."""
    lines = _text(psam, panel).splitlines()
    column = _iid_column(lines[0].split(), psam.name, panel)
    return [line.split()[column] for line in lines[1:] if line.strip()]


def _beside(pgen: Path, panel: str) -> tuple[Path, Path]:
    """A pgen's pvar (plain or `.pvar.zst`) and psam, each of which must be there."""
    pvar = pgen.with_suffix(".pvar")
    if not pvar.is_file():
        pvar = pgen.with_suffix(".pvar.zst")
    psam = pgen.with_suffix(".psam")
    if not all(path.is_file() for path in (pgen, pvar, psam)):
        raise InputError(f"panel {panel}: {pgen} needs its .pgen, .pvar (or .pvar.zst) and .psam")
    return pvar, psam


def _subset(iids: list[str], kept: list[str], psam: Path, panel: str) -> npt.NDArray[np.uint32]:
    """The psam indexes of the kept samples, ascending: every one must be there."""
    where = {iid: i for i, iid in enumerate(iids)}
    absent = [iid for iid in kept if iid not in where]
    if absent:
        raise InputError(
            f"panel {panel}: {psam.name} lacks {len(absent)} keep-file sample(s), "
            f"such as {', '.join(absent[:3])} (build-inputs §2)"
        )
    return np.sort(np.array([where[iid] for iid in kept], dtype=np.uint32))


def fileset(pgen: Path, kept: list[str], sample_order: list[str] | None, panel: str) -> Fileset:
    """The fileset of `pgen`; `sample_order` is None for the first, which sets the order."""
    pvar, psam = _beside(pgen, panel)
    iids = psam_iids(psam, panel)
    subset = _subset(iids, kept, psam, panel)
    mine = [iids[i] for i in subset]
    if sample_order is None or mine == sample_order:
        return Fileset(pgen, pvar, psam, subset, None)
    column = {iid: i for i, iid in enumerate(mine)}
    order = np.array([column[iid] for iid in sample_order], dtype=np.int64)
    return Fileset(pgen, pvar, psam, subset, order)


def _pvar_lines(pvar: Path) -> Iterator[str]:
    """A pvar's lines, `##` meta lines dropped; a `.pvar.zst` is decompressed whole."""
    if pvar.suffix == ".zst":
        # numcodecs, a dependency already, decodes plink2's .pvar.zst; it ships no type stubs.
        zstd = importlib.import_module("numcodecs").Zstd()
        lines: Iterator[str] = iter(bytes(zstd.decode(pvar.read_bytes())).decode().splitlines())
    else:
        lines = (line.rstrip("\n") for line in pvar.open())
    return (line for line in lines if not line.startswith("##"))


def records(pvar: Path, panel: str, info: bool) -> Iterator[Records]:
    """The pvar's records in blocks of `CHUNK_RECORDS`, in pgen row order (build-inputs §2).

    The FILTER column is optional here, and INFO is read only when `info` is set: a build
    refuses a pvar without FILTER when it applies the filters, after it has judged the block's
    genotypes, so that a missing call is the refusal an input with both gets (build-inputs §7).
    """
    lines = _pvar_lines(pvar)
    at = _columns(next(lines, "").split("\t"), pvar, panel)
    first = 0
    for batch in itertools.batched(lines, CHUNK_RECORDS):
        yield _block([line.split("\t") for line in batch], first, at, info)
        first += len(batch)


def _columns(header: list[str], pvar: Path, panel: str) -> dict[str, int]:
    """Each column's index; the columns every pvar needs must be there (build-inputs §2)."""
    missing = [name for name in REQUIRED_COLUMNS if name not in header]
    if missing:
        raise InputError(
            f"panel {panel}: {pvar.name} needs the columns {' '.join(REQUIRED_COLUMNS)} FILTER; "
            f"it has no {', '.join(missing)} (build-inputs §2)"
        )
    return {name: header.index(name) for name in header}


def _block(rows: list[list[str]], first: int, at: dict[str, int], info: bool) -> Records:
    column = list(zip(*rows, strict=True))
    filters = list(column[at["FILTER"]]) if "FILTER" in at else None
    infos = list(column[at["INFO"]]) if info and "INFO" in at else None
    return Records(
        first=first,
        chrom=list(column[at["#CHROM"]]),
        pos=np.array(column[at["POS"]], dtype=np.int64),
        ids=list(column[at["ID"]]),
        ref=list(column[at["REF"]]),
        alt=list(column[at["ALT"]]),
        filter=filters,
        info=infos,
    )


@dataclass(frozen=True)
class PcFile:
    """The PC File's PCs for the kept samples, in the sample order: `values[s, j]` is PC
    `names[j]` of sample s."""

    names: tuple[str, ...]
    values: npt.NDArray[np.float64]
    sha256: str


def read_pc_file(path: Path, samples: Samples, panel: str) -> PcFile:
    """The PC File (build-inputs §5): plink2's `.eigenvec`, `#IID PC1 … PCm` or `#FID IID PC1 …`,
    tab-separated, one row per sample. Every kept sample needs a row; extra rows are ignored."""
    lines = [line.split("\t") for line in _text(path, panel).splitlines() if line.strip()]
    column, names = _pc_header(lines[0], path, panel)
    rows = _pc_rows(lines[1:], column, path, panel)
    absent = [iid for iid in samples.iids if iid not in rows]
    if absent:
        raise InputError(
            f"panel {panel}: the PC File {path.name} has no row for {len(absent)} kept "
            f"sample(s), such as {', '.join(absent[:3])} (build-inputs §5)"
        )
    values = np.array([_finite(rows[iid], iid, names, panel) for iid in samples.iids])
    return PcFile(names, values, hashlib.sha256(path.read_bytes()).hexdigest())


def _pc_header(header: list[str], path: Path, panel: str) -> tuple[int, tuple[str, ...]]:
    """The IID column, and the PCs' names, which must be PC1 … PCm in order (build-inputs §5)."""
    column = _iid_column(header, path.name, panel)
    names = tuple(header[column + 1 :])
    if not names or names != tuple(f"PC{i}" for i in range(1, len(names) + 1)):
        raise InputError(f"panel {panel}: {path.name}'s PCs must be named PC1 … PCm, in order")
    return column, names


def _pc_rows(lines: list[list[str]], column: int, path: Path, panel: str) -> dict[str, list[str]]:
    """Each sample's PC values, as text, by IID; an IID twice is refused."""
    _refuse_repeats([line[column] for line in lines], f"the PC File {path.name}", panel)
    return {line[column]: line[column + 1 :] for line in lines}


def _finite(fields: list[str], iid: str, names: tuple[str, ...], panel: str) -> list[float]:
    """One sample's PCs as floats; a NaN, an infinity or a short row is refused, naming it."""
    if len(fields) != len(names):
        raise InputError(f"panel {panel}: sample {iid}'s PC File row has {len(fields)} values")
    try:
        values = [float(field) for field in fields]
    except ValueError as bad:
        raise InputError(
            f"panel {panel}: sample {iid}'s PC File row is not numbers: {bad}"
        ) from bad
    for name, value in zip(names, values, strict=True):
        if not math.isfinite(value):
            raise InputError(f"panel {panel}: sample {iid}'s {name} is {value} in the PC File")
    return values
