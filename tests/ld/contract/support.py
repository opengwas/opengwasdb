"""What the contract tests share: the Test Panel's paths, the Test Store, and the Test Panel
read as ALIDs and as reference matrices.

The contract tests are epic #1's tickets written as tests before the tickets started, one module
a ticket. Every ticket has landed, so each test now holds its ticket's behaviour as an ordinary
test, with no mark, and none may change a public signature (tests/test_surface.py).

Every test calls the ogld function it is about before it reads the Test Panel through lane
2d's loader, `tests/testdata.py`. A build is given the Test Panel's paths, `PANEL`, which
need no loader; `check_panel_paths` holds them to the loader's once the call has returned.
So a failing call surfaces as itself, never as a missing loader or file.
The Test Store is written from `reference_r(10)` by the walking skeleton's script into
`tests/data/test-panel/store/`; no test writes there.

The expected-value helpers are vectorised and cached, so the default tier stays within its
2 s per test (CONTRIBUTING.md, "Tests").
"""

from __future__ import annotations

import http.server
import importlib
import io
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import BinaryIO

import numpy as np
import numpy.typing as npt

# The Test Store's encoding as the tests compute it, re-exported: the contract tests read it here.
from store_copies import FLOOR as FLOOR
from store_copies import FLOOR_K0 as FLOOR_K0
from store_copies import FLOOR_K7 as FLOOR_K7
from store_copies import HALF_STEP as HALF_STEP
from store_copies import R_SCALE as R_SCALE
from store_copies import codes_at as codes_at
from store_copies import stored_at as stored_at

from ogld.format import t_far
from ogld.identity import orient

TESTS = Path(__file__).resolve().parents[1]
TEST_PANEL = TESTS / "data" / "test-panel"
TEST_STORE = TEST_PANEL / "store"

#: The Test Panel: HGDP+1kGP EUR's 771 samples; the Test Store is built at k = 10.
N_SAMPLES = 771
K = 10
T_FAR = t_far(N_SAMPLES, K)
#: Two independent float64 computations of one r may differ by a few ulps before rounding.
SLACK = 1e-9


@dataclass(frozen=True)
class PanelPaths:
    pgen: Path
    pvar: Path
    psam: Path
    keep: Path
    pcs: Path


#: The Test Panel's files, named without the loader so a build can be called before it.
PANEL = PanelPaths(
    pgen=TEST_PANEL / "panel.pgen",
    pvar=TEST_PANEL / "panel.pvar",
    psam=TEST_PANEL / "panel.psam",
    keep=TEST_PANEL / "panel.keep",
    pcs=TEST_PANEL / "pcs.eigenvec",
)


def panel_data() -> ModuleType:
    """Lane 2d's loader, imported on first use so a test meets its ogld call first."""
    return importlib.import_module("testdata")


def check_panel_paths() -> None:
    """`PANEL` names the files the loader names. Call it after the ogld call under test."""
    loaded = panel_data().panel()
    for field in ("pgen", "pvar", "psam", "keep", "pcs"):
        assert Path(getattr(loaded, field)) == getattr(PANEL, field), field


@cache
def oriented() -> tuple[npt.NDArray[np.str_], npt.NDArray[np.float64]]:
    """Each Test Panel row's ALID, and +1 where A1 is its ALT allele, −1 where it is REF.

    The Test Panel's reference r is the correlation of ALT dosages, so a stored r, signed to
    A1, is the reference r times both rows' signs (spec §3).
    """
    v = panel_data().variants()
    rows = zip(v.chrom, v.pos, v.ref, v.alt, strict=True)
    found = [orient(str(c), int(p), str(ref), str(alt)) for c, p, ref, alt in rows]
    names = np.array([o.alid for o in found])
    return names, np.array([1.0 if o.a1_is_alt else -1.0 for o in found])


def alids(rows: npt.ArrayLike) -> list[str]:
    names, _ = oriented()
    return [str(name) for name in names[np.atleast_1d(np.asarray(rows, dtype=np.int64))]]


@cache
def passing_rows() -> npt.NDArray[np.int64]:
    """The Test Panel rows that pass the build's filters: the Test Store's Variant Set."""
    return np.flatnonzero(panel_data().variants().passes_filters).astype(np.int64)


@dataclass(frozen=True)
class Pairs:
    """The reference pairs at one k, with distances and r signed to each row's A1."""

    row_i: npt.NDArray[np.int64]
    row_j: npt.NDArray[np.int64]
    distance: npt.NDArray[np.int64]
    r: npt.NDArray[np.float64]

    @property
    def code(self) -> npt.NDArray[np.int64]:
        return codes_at(self.r)

    def where(self, mask: npt.NDArray[np.bool_], what: str) -> Pairs:
        """The pairs `mask` keeps. An empty fixture proves nothing, so there must be one."""
        assert mask.any(), f"the Test Panel must hold {what} for this test to mean anything"
        return Pairs(self.row_i[mask], self.row_j[mask], self.distance[mask], self.r[mask])


@cache
def reference(k: int = K) -> Pairs:
    ref = panel_data().reference_r(k)
    _, sign = oriented()
    pos = panel_data().variants().pos.astype(np.int64)
    i, j = ref.row_i.astype(np.int64), ref.row_j.astype(np.int64)
    return Pairs(i, j, pos[j] - pos[i], ref.r * sign[i] * sign[j])


@cache
def reference_matrix(k: int, rows: tuple[int, ...]) -> npt.NDArray[np.float64]:
    """The reference r among Test Panel `rows`, in that order: 1 on the diagonal, NaN for a
    pair the reference does not hold (more than 10 Mb apart, or across chromosomes)."""
    ref = reference(k)
    where = np.full(len(panel_data().variants().pos), -1, dtype=np.int64)
    where[list(rows)] = np.arange(len(rows))
    a, b = where[ref.row_i], where[ref.row_j]
    keep = (a >= 0) & (b >= 0)
    out = np.full((len(rows), len(rows)), np.nan)
    out[a[keep], b[keep]] = ref.r[keep]
    out[b[keep], a[keep]] = ref.r[keep]
    np.fill_diagonal(out, 1.0)
    return out


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """A static file server that honours one byte range, as Zarr over HTTP needs."""

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def send_head(self) -> BinaryIO | io.BytesIO | None:
        header = self.headers.get("Range")
        if not header or not header.startswith("bytes="):
            return super().send_head()
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404)
            return None
        data = path.read_bytes()
        start_text, _, stop_text = header[len("bytes=") :].partition("-")
        if start_text:
            start = int(start_text)
            stop = int(stop_text) if stop_text else len(data) - 1
        else:
            start, stop = len(data) - int(stop_text), len(data) - 1
        body = data[start : stop + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{start + len(body) - 1}/{len(data)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return None
