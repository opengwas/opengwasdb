"""The whole path, once: Test Panel pgen → k = 0 Near Band → `StoreWriter` → `ogld.open` → one pair.

Nothing in it comes from `reference_r` except the expected value: the codes the store answers with
were computed from the pgen's genotypes by the build's own k = 0 path (ogld#12).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import testdata
from script_modules import load_script
from store_copies import FLOOR_K0, HALF_STEP, R_SCALE

import ogld
from ogld.build.near_band import NearBand, read_alt_counts, unadjusted_near_band
from ogld.format import FAR_BP, NEAR_BP, REncoding
from ogld.store import StoreReader, StoreWriter, Variants


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
    panel = testdata.panel()
    alt = read_alt_counts(panel.pgen, panel.psam, panel.keep, rows)
    encoding = REncoding(R_SCALE)
    band = unadjusted_near_band(alt, vs.sign[on22] > 0, table.position, NEAR_BP, encoding=encoding)
    out = tmp_path / "walk"
    writer = StoreWriter.create(
        out, variants=table, near_bp=NEAR_BP, far_bp=FAR_BP, encoding=encoding
    )
    writer.write_near(0, band.lengths, band.codes)
    indptr, partner, codes = band_index(band, 0, len(table), FLOOR_K0)
    writer.write_index("22", indptr, partner, codes)
    counts = (len(band.codes), 0, 0, len(partner))
    build = script.build_record(0.0)
    sub = script.VariantSet(rows, table, vs.sign[on22], 0)
    # finish validates (spec §12), and a genotypes store's pc_diagnostic is never null: this one
    # is the Test Panel's own, measured at k = 0 on its chr21 × chr22 pairs.
    diagnostic = script.pc_diagnostic(vs, 0)
    writer.finish(script.provenance(0, sub, counts, diagnostic, build, encoding))

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
