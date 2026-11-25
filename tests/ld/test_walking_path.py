"""The whole path, once: Test Panel pgen → k = 0 r → `StoreWriter` → `opengwasdb.ld.open` → one
pair.

Nothing in it comes from `reference_r` except the expected value. The r the store is made from was
computed from the pgen's genotypes, read by `opengwasdb.ld.genotypes.Genotypes`, by this file's own
plain
k = 0 product (`k0_r`), not by the builder's tiles. The store makes its Near Band and Proxy
Index from it (ogld#17); no index is built here.
"""

from __future__ import annotations

from pathlib import Path

import build_fixtures as fx
import numpy as np
import numpy.typing as npt
import pytest
import testdata
from ld_script_modules import load_script
from ogquery.ld.store import StoreReader, Variants
from store_copies import HALF_STEP, R_SCALE

import opengwasdb.ld
from opengwasdb.ld.format import FAR_BP, NEAR_BP, REncoding
from opengwasdb.ld.store import StoreWriter


def k0_r(alt: npt.NDArray[np.int8], a1_is_alt: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """The k = 0 r of every pair, signed to A1, written out plainly from standardised A1 dosages:
    this test's own computation of what `opengwasdb.ld.build` tiles."""
    z = alt - alt.mean(axis=1, keepdims=True)
    z /= np.sqrt((z * z).sum(axis=1, keepdims=True))
    z[~a1_is_alt] *= -1
    r: npt.NDArray[np.float64] = z @ z.T
    return r


def test_a_pair_from_the_pgen_is_answered_by_the_store_it_wrote(tmp_path: Path) -> None:
    script = load_script("make_test_store")
    vs = script.variant_set()
    on22 = vs.variants.chrom == "22"
    assert on22.any(), "the Test Store must hold chr22"
    assert (~on22).any(), "and another chromosome"
    assert (vs.sign[on22] < 0).any(), "chr22 must hold a variant whose A1 is REF"
    assert (vs.sign[on22] > 0).any(), "and one whose A1 is ALT"
    rows = vs.panel_rows[on22]
    v = vs.variants
    table = Variants(
        chrom=v.chrom[on22],
        position=v.position[on22],
        alid=v.alid[on22],
        a1=v.a1[on22],
        a2=v.a2[on22],
        a1_frequency=v.a1_frequency[on22],
    )
    out = tmp_path / "walk"
    writer = StoreWriter.create(
        out,
        variants=table,
        encoding=REncoding(R_SCALE),
        near_bp=NEAR_BP,
        far_bp=FAR_BP,
        n=testdata.panel().n_samples,
        k=0,
        t_partner=0.01,
    )
    writer.write(writer.cut(0, k0_r(script.alt_counts(rows), vs.sign[on22] > 0)))
    sub = script.VariantSet(rows, table, vs.sign[on22], 0)
    # finish validates (ld-store §12), and a genotypes store's pc_diagnostic is never null: this one
    # is the Test Panel's own, measured at k = 0 on its chr21 × chr22 pairs.
    diagnostic = script.pc_diagnostic(vs, 0)
    written = writer.finish(script.sourced(0, sub, diagnostic, script.build_record(0.0)))
    assert written["counts"]["far_pairs"] > 0, "the store found Far Pairs in the r it was given"

    store = opengwasdb.ld.open(out)
    ref = testdata.reference_r(0)
    sign = np.full(len(testdata.variants()), 0.0)
    sign[vs.panel_rows] = vs.sign
    pos = testdata.variants().pos
    wanted = np.flatnonzero(np.isin(ref.row_i, rows) & np.isin(ref.row_j, rows))
    wanted = wanted[(pos[ref.row_j[wanted]] - pos[ref.row_i[wanted]]) <= NEAR_BP]
    pick = wanted[len(wanted) // 2]
    i, j = int(ref.row_i[pick]), int(ref.row_j[pick])
    mine = StoreReader.open(out).rows_of(
        [
            str(table.alid[np.flatnonzero(rows == i)[0]]),
            str(table.alid[np.flatnonzero(rows == j)[0]]),
        ]
    )
    assert (mine >= 0).all()
    got = StoreReader.open(out).pair_r(mine)[0, 1]
    assert abs(got - ref.r[pick] * sign[i] * sign[j]) <= HALF_STEP + 1e-9
    assert store.near_bp == NEAR_BP


def test_the_test_store_script_refuses_a_missing_call_rather_than_average_it(
    tmp_path: Path,
) -> None:
    script = load_script("make_test_store")
    rows = np.arange(5, dtype=np.int64)
    calls = fx.hard_calls()[rows].copy()
    calls[2, 7] = -9
    pgen = fx.write_fileset(tmp_path / "gappy", rows, genotypes=calls)
    with pytest.raises(opengwasdb.ld.InputError, match="missing call"):
        script.alt_counts(rows, pgen)
    np.testing.assert_array_equal(script.alt_counts(rows[:2], pgen), calls[:2][:, _kept()])


def _kept() -> npt.NDArray[np.int64]:
    """The psam columns of the Test Panel's kept samples (every sample, in psam order)."""
    iids = [line.split("\t")[0] for line in testdata.panel().psam.read_text().splitlines()[1:]]
    kept = set(testdata.panel().keep.read_text().splitlines()[1:])
    return np.array([i for i, iid in enumerate(iids) if iid in kept], dtype=np.int64)
