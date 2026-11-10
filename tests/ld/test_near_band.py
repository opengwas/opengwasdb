"""Seam 1 of the walking skeleton: the k = 0 Near Band of a Test Panel chromosome.

pgen → standardised A1 dosages → r → int16 codes, against `reference_r(0)`, the Test Panel's
independent r (ALT dosages, so signed to A1 here by `ogld.identity.orient`; spec §3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import testdata

from ogld.build.near_band import NearBand, read_alt_counts, unadjusted_near_band
from ogld.format import NEAR_BP, InputError, dequantise
from ogld.identity import orient

HALF_STEP = 1 / 65534
SLACK = 1e-9


def chromosome_rows(chrom: str) -> npt.NDArray[np.int64]:
    v = testdata.variants()
    return np.flatnonzero(v.passes_filters & (v.chrom == chrom)).astype(np.int64)


def a1_is_alt(rows: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
    v = testdata.variants()
    return np.array(
        [
            orient(str(v.chrom[r]), int(v.pos[r]), str(v.ref[r]), str(v.alt[r])).a1_is_alt
            for r in rows
        ]
    )


def near_band(chrom: str) -> tuple[npt.NDArray[np.int64], NearBand]:
    panel, v = testdata.panel(), testdata.variants()
    rows = chromosome_rows(chrom)
    alt = read_alt_counts(panel.pgen, panel.psam, panel.keep, rows)
    return rows, unadjusted_near_band(alt, a1_is_alt(rows), v.pos[rows], NEAR_BP)


def test_the_fixture_has_variants_whose_a1_is_ref_and_pairs_across_the_band() -> None:
    rows = chromosome_rows("22")
    flipped = ~a1_is_alt(rows)
    assert flipped.any(), "a Test Panel chromosome must hold a variant whose A1 is REF"
    assert (~flipped).any(), "and one whose A1 is ALT"
    ref = testdata.reference_r(0)
    d = testdata.variants().pos[ref.row_j] - testdata.variants().pos[ref.row_i]
    assert ((d > 0) & (d <= NEAR_BP)).sum() > 1000, "the reference must hold many Near Band pairs"


@pytest.mark.parametrize("chrom", ["21", "22"])
def test_the_k0_near_band_matches_reference_r_signed_to_a1(chrom: str) -> None:
    rows, band = near_band(chrom)
    v = testdata.variants()
    local = {int(r): i for i, r in enumerate(rows)}
    sign = np.where(a1_is_alt(rows), 1.0, -1.0)
    stored = dequantise(band.codes)
    ref = testdata.reference_r(0)
    seen = 0
    for i, j, r in zip(ref.row_i, ref.row_j, ref.r, strict=True):
        if int(i) not in local or int(j) not in local or v.chrom[i] != chrom:
            continue
        a, b = local[int(i)], local[int(j)]
        if v.pos[j] - v.pos[i] > NEAR_BP:
            continue
        got = stored[band.indptr[a] + (b - a - 1)]
        want = r * sign[a] * sign[b]
        assert abs(got - want) <= HALF_STEP + SLACK, f"rows {i} and {j}"
        seen += 1
    # Every Near Band pair is a reference pair: the band holds exactly the ones counted.
    assert seen == len(band.codes) > 500


def test_the_band_rows_hold_the_pairs_within_near_bp_after_each_variant() -> None:
    rows, band = near_band("22")
    pos = testdata.variants().pos[rows]
    expected = [
        int(np.searchsorted(pos, p + NEAR_BP, side="right")) - i - 1 for i, p in enumerate(pos)
    ]
    np.testing.assert_array_equal(np.diff(band.indptr), expected)


def test_a_missing_call_is_refused_naming_the_variant_and_the_panel(tmp_path: Path) -> None:
    import pgenlib

    pgen = tmp_path / "gappy.pgen"
    writer = pgenlib.PgenWriter(str(pgen).encode(), 4, 2)
    writer.append_biallelic(np.array([0, 1, 2, 1], dtype=np.int8))
    writer.append_biallelic(np.array([1, 1, 0, -9], dtype=np.int8))
    writer.close()
    (tmp_path / "gappy.pvar").write_text(
        "#CHROM\tPOS\tID\tREF\tALT\n1\t5\tv0\tA\tG\n1\t9\tv1\tC\tT\n"
    )
    psam = tmp_path / "gappy.psam"
    psam.write_text("#IID\ns1\ns2\ns3\ns4\n")
    keep = tmp_path / "gappy.keep"
    keep.write_text("#IID\ns1\ns2\ns3\ns4\n")
    rows = np.array([0, 1], dtype=np.int64)
    with pytest.raises(InputError, match=r"missing call.*v1.*gappy\.pgen"):
        read_alt_counts(pgen, psam, keep, rows)


def test_a_keep_file_naming_a_sample_the_panel_lacks_is_refused(tmp_path: Path) -> None:
    panel = testdata.panel()
    keep = tmp_path / "stranger.keep"
    keep.write_text("#IID\nnobody-in-the-panel\n")
    with pytest.raises(InputError, match="keep file"):
        read_alt_counts(panel.pgen, panel.psam, keep, np.array([0], dtype=np.int64))


def test_a_variant_without_variance_is_refused_naming_it_and_the_panel() -> None:
    alt = np.array([[0, 1, 2, 1], [1, 1, 1, 1], [2, 0, 1, 1]], dtype=np.int8)
    with pytest.raises(InputError, match=r"variant rs-flat of panel the-test-panel"):
        unadjusted_near_band(
            alt,
            np.array([True, True, True]),
            np.array([100, 200, 300]),
            NEAR_BP,
            names=["rs-ok", "rs-flat", "rs-also-ok"],
            panel="the-test-panel",
        )


def test_a_duplicate_iid_in_the_keep_file_is_refused_naming_it(tmp_path: Path) -> None:
    panel = testdata.panel()
    iids = panel.keep.read_text().splitlines()
    keep = tmp_path / "twice.keep"
    keep.write_text("\n".join([*iids, iids[3]]) + "\n")
    with pytest.raises(InputError, match=rf"duplicate.*{iids[3]}"):
        read_alt_counts(panel.pgen, panel.psam, keep, np.array([0], dtype=np.int64))
