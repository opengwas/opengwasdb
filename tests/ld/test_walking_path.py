"""The whole path, once: Test Panel pgen → k = 0 Near Band → `StoreWriter` → `ogld.open` → one pair.

Nothing in it comes from `reference_r` except the expected value: the codes the store answers with
were computed from the pgen's genotypes by the build's own k = 0 path (ogld#12).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import testdata
from script_modules import load_script

import ogld
from ogld.format import FAR_BP, NEAR_BP, far_code_floor, near_row_lengths, quantise, t_far
from ogld.format.geometry import indptr_of, stop_within
from ogld.store import StoreReader, StoreWriter, Variants

HALF_STEP = 1 / 65534


@dataclass(frozen=True)
class NearBand:
    """One chromosome's Near Band as `StoreWriter.write_near` takes it: row l's codes are
    `codes[indptr[l]:indptr[l + 1]]`, its partners l + 1, l + 2, …"""

    indptr: npt.NDArray[np.int64]
    codes: npt.NDArray[np.int16]

    @property
    def lengths(self) -> npt.NDArray[np.int64]:
        return np.diff(self.indptr)


def k0_near_band(
    alt: npt.NDArray[np.float64], a1_is_alt: npt.NDArray[np.bool_], positions: npt.NDArray[np.int64]
) -> NearBand:
    """The k = 0 Near Band, written out plainly from standardised A1 dosages: this test's own
    oracle for the path, where `ogld.build` tiles the same product."""
    z = alt - alt.mean(axis=1, keepdims=True)
    z /= np.sqrt((z * z).sum(axis=1, keepdims=True))
    z[~a1_is_alt] *= -1
    stop = stop_within(positions, NEAR_BP)
    indptr = indptr_of(near_row_lengths(positions, NEAR_BP))
    codes = np.concatenate([quantise(z[i] @ z[i + 1 : stop[i]].T) for i in range(len(z))])
    return NearBand(indptr, codes.astype(np.int16))


def band_index(
    band: NearBand, start: int, size: int, floor: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int16]]:
    """The Proxy Index of a chromosome whose pairs are the Near Band's: both directions."""
    rows = np.repeat(np.arange(size), np.diff(band.indptr))
    cols = np.concatenate([np.arange(r + 1, r + 1 + n) for r, n in enumerate(np.diff(band.indptr))])
    keep = np.abs(band.codes.astype(np.int32)) >= floor
    a, b, c = (
        np.concatenate([rows[keep], cols[keep]]),
        np.concatenate([cols[keep], rows[keep]]),
        (np.concatenate([band.codes[keep], band.codes[keep]])),
    )
    order = np.lexsort((b, a))
    indptr = np.concatenate([[0], np.cumsum(np.bincount(a, minlength=size))]).astype(np.int64)
    return indptr, b[order].astype(np.int64) + start, c[order]


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
    band = k0_near_band(script.alt_counts(rows), vs.sign[on22] > 0, table.position)
    out = tmp_path / "walk"
    writer = StoreWriter.create(out, variants=table, near_bp=NEAR_BP, far_bp=FAR_BP)
    writer.write_near(0, band.lengths, band.codes)
    floor = far_code_floor(t_far(771, 0))
    indptr, partner, codes = band_index(band, 0, len(table), floor)
    writer.write_index("22", indptr, partner, codes)
    counts = (len(band.codes), 0, 0, len(partner))
    build = script.build_record(0.0)
    sub = script.VariantSet(rows, table, vs.sign[on22], 0)
    # finish validates (spec §12), and a genotypes store's pc_diagnostic is never null: this one
    # is the Test Panel's own, measured at k = 0 on its chr21 × chr22 pairs.
    diagnostic = script.pc_diagnostic(vs, 0)
    writer.finish(script.provenance(0, sub, counts, diagnostic, build))

    store = ogld.open(out)
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
