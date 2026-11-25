"""What ogld#5's own builder tests share: inputs made from the Test Panel, and the store read back.

A test that needs a malformed input (a missing call, a FILTER that fails, a record on X) makes
it from the Test Panel's own files in its `tmp_path`: the Test Panel holds none of them
(its README, "Not in v2, so built by hand in unit tests"), and no test writes into
`tests/ld/data/`. Genotypes are written with pgenlib, the builder's own reader's library.
"""

from __future__ import annotations

import functools
import shutil
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest
import store_copies
import testdata
from contract import support
from ogquery.ld.store import StoreReader

import opengwasdb.ld

#: The Test Panel is built at the Test Store's scale, so a build compares with it code for code.
R_SCALE: Final = store_copies.R_SCALE
HALF_STEP: Final = store_copies.HALF_STEP
#: Two float64 computations of one r may differ by a few ulps before rounding.
SLACK: Final = 1e-9
PANEL_ID: Final = "test-panel-hgdp1kgp-EUR"
#: The Partner Threshold a build of the Test Panel is given: the HGDP+1kGP panels' 0.01
#: (partner-rows §3), which is below its t_far 0.0274.
T_PARTNER: Final = 0.01


def inputs(fileset: Path, k: int = 0, **overrides: Any) -> opengwasdb.ld.BuildInputs:
    """BuildInputs for `fileset` (a .pgen), and the Test Panel's keep file and PC File."""
    panel = testdata.panel()
    fields: dict[str, Any] = {
        "k": k,
        "info_min": None,
        "panel_id": PANEL_ID,
        "ancestry": "EUR",
        "genotype_source": "the Test Panel: HGDP+1kGP phased v2 EUR, chr21 and chr22",
        "r_scale": R_SCALE,
        "t_partner": T_PARTNER,
        "pgen": (fileset,),
        "keep": panel.keep,
        "pcs": panel.pcs if k else None,
        "pc_source": "tests/data/test-panel/pcs.eigenvec" if k else None,
    }
    return opengwasdb.ld.BuildInputs(**{**fields, **overrides})


_BUILDS: dict[int, tuple[opengwasdb.ld.Provenance, Path]] = {}


def test_panel_build(
    k: int, factory: pytest.TempPathFactory
) -> tuple[opengwasdb.ld.Provenance, Path]:
    """The Test Panel built at k, once per test process and shared by every module that reads
    one: tests/ld/contract/test_build.py, test_builder.py, test_build_diagnostic.py and
    test_walking_path.py. Each would otherwise build its own."""
    if k not in _BUILDS:
        out = factory.mktemp(f"test-panel-k{k}") / "store"
        _BUILDS[k] = (opengwasdb.ld.build(inputs(testdata.panel().pgen, k), out), out)
    return _BUILDS[k]


@functools.cache
def hard_calls() -> npt.NDArray[np.int8]:
    """Every Test Panel genotype, rows by psam samples: 0, 1 or 2 ALT alleles."""
    import pgenlib

    reader = pgenlib.PgenReader(str(testdata.panel().pgen).encode())
    out = np.empty((reader.get_variant_ct(), reader.get_raw_sample_ct()), dtype=np.int8)
    reader.read_range(0, reader.get_variant_ct(), out)
    reader.close()
    out.flags.writeable = False
    return out


def pvar_lines() -> list[str]:
    """The Test Panel's pvar, header first."""
    return testdata.panel().pvar.read_text().splitlines()


def write_fileset(
    prefix: Path,
    rows: npt.ArrayLike | None = None,
    *,
    genotypes: npt.NDArray[np.int8] | None = None,
    dosages: npt.NDArray[np.float32] | None = None,
    lines: list[str] | None = None,
    samples: npt.ArrayLike | None = None,
) -> Path:
    """A fileset of Test Panel `rows` (all by default) at `prefix`; its .pgen's path.

    `genotypes` or `dosages` replace the rows' genotypes, and `lines` their pvar records
    (header included). `samples` orders the psam's samples, and the genotypes' columns with it.
    """
    picked = np.arange(len(testdata.variants())) if rows is None else np.asarray(rows)
    calls = hard_calls()[picked] if genotypes is None else genotypes
    order = np.arange(calls.shape[1]) if samples is None else np.asarray(samples)
    pgen = prefix.with_name(prefix.name + ".pgen")
    _write_genotypes(pgen, calls[:, order], None if dosages is None else dosages[:, order])
    text = pvar_lines()
    records = lines or [text[0], *(text[1 + int(r)] for r in picked)]
    prefix.with_name(prefix.name + ".pvar").write_text("\n".join(records) + "\n")
    psam = testdata.panel().psam.read_text().splitlines()
    psam_out = [psam[0], *(psam[1 + int(i)] for i in order)]
    prefix.with_name(prefix.name + ".psam").write_text("\n".join(psam_out) + "\n")
    return pgen


def _write_genotypes(
    pgen: Path, calls: npt.NDArray[np.int8], dosages: npt.NDArray[np.float32] | None
) -> None:
    import pgenlib

    count, samples = calls.shape
    if dosages is None:
        writer = pgenlib.PgenWriter(str(pgen).encode(), samples, count, False)
        for row in calls:
            writer.append_biallelic(np.ascontiguousarray(row))
    else:
        writer = pgenlib.PgenWriter(str(pgen).encode(), samples, count, False, dosage_present=True)
        for row in dosages:
            writer.append_dosages(np.ascontiguousarray(row, dtype=np.float32))
    writer.close()


def copy_panel(prefix: Path) -> Path:
    """The Test Panel's fileset copied to `prefix`; its .pgen's path."""
    panel = testdata.panel()
    for source in (panel.pgen, panel.pvar, panel.psam):
        shutil.copyfile(source, prefix.with_name(prefix.name + source.suffix))
    return prefix.with_name(prefix.name + ".pgen")


def a1_signs() -> npt.NDArray[np.float64]:
    """+1 where a Test Panel row's A1 is its ALT, −1 where it is REF (ld-store §3): the test
    oracle's, `tests/ld/contract/support.py` (`docs/architecture/ownership.md`, "Test oracles")."""
    return support.oriented()[1]


def alids(rows: npt.ArrayLike) -> list[str]:
    """The ALIDs of Test Panel `rows`, from the test oracle (`support.alids`)."""
    return support.alids(rows)


def stored_r(store: Path, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """The Stored r among Test Panel `rows`, through the store layer's reader: NaN unstored."""
    reader = StoreReader.open(store)
    found = reader.rows_of(alids(rows))
    assert (found >= 0).all(), "every row asked for is in the Variant Set"
    return reader.pair_r(found)


def reference_pairs(
    k: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """The Test Panel's independent reference r at k, signed to A1: (row_i, row_j, r), from the
    test oracle (`support.reference`)."""
    ref = support.reference(k)
    return ref.row_i, ref.row_j, ref.r


def store_arrays(store: Path) -> dict[str, npt.NDArray[Any]]:
    """Every array of a store, by path: two builds are the same store when these are equal."""
    import zarr

    root = zarr.open_group(str(store), mode="r", zarr_format=3)
    out: dict[str, npt.NDArray[Any]] = {}
    for path, array in root.arrays():
        out[path] = np.asarray(array[:])
    for path, group in root.groups():
        for name, array in group.arrays():
            out[f"{path}/{name}"] = np.asarray(array[:])
        for sub, inner in group.groups():
            for name, array in inner.arrays():
                out[f"{path}/{sub}/{name}"] = np.asarray(array[:])
    return out


def same_store(one: Path, other: Path) -> None:
    a, b = store_arrays(one), store_arrays(other)
    assert a.keys() == b.keys()
    assert len(a) >= 11, "a store holds the variant table, a Near Band and a Proxy Index"
    for path in a:
        assert np.array_equal(a[path], b[path]), path
