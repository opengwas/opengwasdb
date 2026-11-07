"""The Test Panel's rows: what each is, why it is there, and whether the filters keep it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from test_panel.params import Params
from test_panel.select import Components
from test_panel.sites import Sites
from test_panel.tools import export_rows


@dataclass(frozen=True)
class Chosen:
    """One chromosome's chosen records: their source rows, filter verdicts and components."""

    sites: Sites
    reason: np.ndarray
    components: Components
    rowid_pvar: Path

    @property
    def rows(self) -> list[int]:
        return sorted({r for rows in self.components.values() for r in rows})


@dataclass(frozen=True)
class Panel:
    """Per Test Panel row (0-based record order of `panel.pvar`). `reason` is "" for a row that
    passes the filters, else the first filter it fails."""

    chrom: list[str]
    pos: np.ndarray
    ref: list[str]
    alt: list[str]
    vid: list[str]
    filt: list[str]
    mac: np.ndarray
    reason: np.ndarray
    components: list[list[str]]

    @property
    def passing(self) -> np.ndarray:
        return np.flatnonzero(self.reason == "")


def table(chosen: list[Chosen], pvar: Path) -> Panel:
    """Join the selection to the converted panel's pvar, refusing any difference in records."""
    cols = _columns(chosen)
    lines = [x.split("\t") for x in pvar.read_text().splitlines()[1:]]
    got = [(f[0], int(f[1]), f[3], f[4]) for f in lines]
    if got != list(zip(cols["chrom"], cols["pos"], cols["ref"], cols["alt"], strict=True)):
        raise SystemExit("the converted Test Panel's records differ from the selection")
    return Panel(
        chrom=cols["chrom"], pos=np.array(cols["pos"]), ref=cols["ref"], alt=cols["alt"],
        vid=[f[2] for f in lines], filt=cols["filt"], mac=np.array(cols["mac"]),
        reason=np.array(cols["reason"], dtype=object), components=cols["components"])


def _columns(chosen: list[Chosen]) -> dict[str, list[Any]]:
    cols: dict[str, list[Any]] = {k: [] for k in ("chrom", "pos", "ref", "alt", "filt", "mac",
                                                  "reason", "components")}
    for c in chosen:
        for r in c.rows:
            _append(cols, c, r)
    return cols


def _append(cols: dict[str, list[Any]], c: Chosen, r: int) -> None:
    s = c.sites
    cols["chrom"].append(s.chrom)
    cols["pos"].append(int(s.pos[r]))
    cols["ref"].append(str(s.ref[r]))
    cols["alt"].append(str(s.alt[r]))
    cols["filt"].append(str(s.filt[r]))
    cols["mac"].append(int(s.mac[r]))
    cols["reason"].append(str(c.reason[r]))
    cols["components"].append(sorted(n for n, v in c.components.items() if r in v))


def same_genotypes(params: Params, chosen: list[Chosen], panel_g: np.ndarray,
                   panel_fam: list[str]) -> None:
    """The cut-and-convert path must reproduce the whole-chromosome conversion exactly."""
    parts = []
    for c in chosen:
        _, fam, g = export_rows(params, c.sites.prefix, c.rowid_pvar, c.rows,
                                params.work / f"check.chr{c.sites.chrom}")
        if fam != panel_fam:
            raise SystemExit("sample order differs between the two conversions")
        parts.append(g)
    if not np.array_equal(np.concatenate(parts), panel_g):
        raise SystemExit("the Test Panel's genotypes differ from the chromosome conversion's")
