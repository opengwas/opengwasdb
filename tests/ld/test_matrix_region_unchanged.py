"""`matrix` and `region` answer on the Test Store exactly as they did before opengwas/ogld#72.

The Proxy Index's removal touched the reader's chromosome-loading path and the arrays a store
holds, and neither may change a matrix or a region. `FROZEN` holds each answer as a digest of
every column it carries, recorded by running `fingerprint` against the **integration tip's code**
(`88b12e7`, before opengwas/ogld#72) on this same committed Test Store, never against the code this
file ships with. So a difference here is the removal's, and the digests say nothing about what the
new code would answer.

The cases are the reads opengwas/ogld#8's `bench-latency` and the contract make: every variant's
matrix, a sample, a thin slice of one chromosome; whole-chromosome regions and windows inside them;
a scattered `pair_r`; `r_of_pairs` on a ladder of one row against many and with its arguments
genuinely reversed; and the far side of both queries: the pairs the store holds as Far Pairs, a
`region_r` two rows apart by more than `near_bp`, and a public `region` wide enough to reach
beyond it. `test_the_fixture_reaches_the_far_side` asserts those properties of the cases before
any digest is compared, so a case that stopped reaching the far side would fail there.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from fingerprint import answer_fingerprint
from store_copies import TEST_STORE

import opengwasdb.ld
from opengwasdb.ld.store import StoreReader

#: The Test Store's rows, and the samples the cases read.
ALL = np.arange(1056, dtype=np.int64)
SAMPLE = ALL[::7]
SMALL = ALL[:40]
SLICE = ALL[600:660]


def matrix(store: opengwasdb.ld.LDStore, rows: np.ndarray, *, fill: float | None = None) -> str:
    names = list(store._reader.alids_of(rows))
    got = store.matrix(names, fill=fill)
    if fill is not None:
        return answer_fingerprint({"r": got.r})
    return answer_fingerprint(
        {
            "chrom": got.variants.chrom,
            "position": got.variants.position,
            "alid": got.variants.alid,
            "r": got.r,
            "t_far": got.t_far,
        }
    )


def on_one_chromosome(reader: StoreReader, earlier: np.ndarray, later: np.ndarray) -> bool:
    """Whether every pair is two rows of one chromosome, the earlier first."""
    for chrom in ("21", "22"):
        start, stop = reader.chromosome_range(chrom)
        mine = (earlier >= start) & (later < stop)
        if (mine & (later <= earlier)).any():
            return False
    return True


def far_pairs(reader: StoreReader) -> tuple[np.ndarray, np.ndarray]:
    """Every pair the Test Store holds as a Far Pair, as (earlier row, later row), read from its
    own `far/<chrom>/` arrays: the pairs a matrix or a region answers beyond `near_bp`."""
    root = zarr.open_group(str(TEST_STORE), mode="r")
    earlier, later = [], []
    for chrom in ("21", "22"):
        start, _ = reader.chromosome_range(chrom)
        group = root[f"far/{chrom}"]
        indptr = np.asarray(group["indptr"][:], dtype=np.int64)
        rows = np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))
        earlier.append(rows + start)
        later.append(np.asarray(group["partner"][:], dtype=np.int64) + start)
    return np.concatenate(earlier), np.concatenate(later)


def narrowest_far_pair(reader: StoreReader) -> tuple[np.ndarray, np.ndarray, int, int]:
    """The stored Far Pair with the fewest rows between its ends, and its chromosome,
    position gap and the centre both ends sit nearest: what a wide region is drawn around."""
    earlier, later = far_pairs(reader)
    if not len(earlier):
        raise AssertionError("the Test Store holds no Far Pair: no case would reach the far side")
    at = int(np.argmin(later - earlier))
    lo, hi = int(earlier[at]), int(later[at])
    position = reader.variants(np.array([lo, hi])).position
    return np.array([lo]), np.array([hi]), int(position[0]), int(position[1])


def ladder() -> tuple[np.ndarray, np.ndarray]:
    """One row against many: `first` a row, `second` the forty rows after it, so the pairs are
    strictly ascending and every index is a row of the Variant Set."""
    first = np.repeat(ALL[:1000:20], 40)
    return first, first + np.tile(np.arange(1, 41, dtype=np.int64), len(first) // 40)


def ages(store: opengwasdb.ld.LDStore, reader: StoreReader, rows: np.ndarray) -> dict[str, str]:
    """Every case's answer, keyed as `FROZEN` is."""
    found: dict[str, str] = {}
    for name, subset in {"all": ALL, "sample": SAMPLE, "small": SMALL, "chr22": SLICE}.items():
        found[f"matrix-{name}"] = matrix(store, subset)
        found[f"matrix-fill-{name}"] = matrix(store, subset, fill=0.0)
    for chrom in ("21", "22"):
        start, stop = reader.chromosome_range(chrom)
        for label, a, b in [
            ("whole", start, stop),
            ("head", start, start + 200),
            ("tail", stop - 200, stop),
        ]:
            found[f"region_r-{chrom}-{label}"] = answer_fingerprint({"r": reader.region_r(a, b)})
            got = store.region(chrom, int(reader._variant_table._columns["position"][a]), 500_000)
            found[f"region-{chrom}-{label}-500k"] = region_fingerprint(got)
    found["pair_r-sample"] = answer_fingerprint({"r": reader.pair_r(ALL[::5])})
    first, second = ladder()
    found["r_of_pairs-ladder"] = answer_fingerprint({"r": reader.r_of_pairs(first, second)})
    # The same pairs asked for in the other order: r is symmetric, so these two digests must
    # agree, and the reversed call is a different path through `_step_r` (`min`/`max` swap).
    found["r_of_pairs-reversed"] = answer_fingerprint({"r": reader.r_of_pairs(second, first)})

    # The far side: the pairs the store holds as Far Pairs, and the two ways a region reaches
    # its pairs beyond near_bp — a row span, and a public window wide enough to hold them.
    far_lo, far_hi = far_pairs(reader)
    found["r_of_pairs-far"] = answer_fingerprint({"r": reader.r_of_pairs(far_lo, far_hi)})
    found["r_of_pairs-far-reversed"] = answer_fingerprint({"r": reader.r_of_pairs(far_hi, far_lo)})
    lo, hi, at, partner = narrowest_far_pair(reader)
    found["region_r-far"] = answer_fingerprint({"r": reader.region_r(int(lo[0]), int(hi[0]) + 1)})
    radius = (partner - at) // 2 + 1
    wide = store.region("22" if int(lo[0]) >= 468 else "21", (at + partner) // 2, radius)
    found["region-far"] = region_fingerprint(wide)
    return found


def region_fingerprint(got: opengwasdb.ld.LDMatrix) -> str:
    return answer_fingerprint(
        {
            "r": got.r,
            "chrom": got.variants.chrom,
            "position": got.variants.position,
            "alid": got.variants.alid,
        }
    )


def test_the_fixture_reaches_the_far_side(found: dict[str, str]) -> None:
    """The far cases are meaningful before any digest is compared (CONTRIBUTING.md)."""
    reader = StoreReader.open(TEST_STORE)
    near_bp = reader.provenance["near_bp"]
    far_bp = reader.provenance["far_bp"]
    far_lo, far_hi = far_pairs(reader)
    assert len(far_lo) > 0, "the Test Store must hold Far Pairs"
    assert on_one_chromosome(reader, far_lo, far_hi), "each is two rows of one chromosome"
    position = reader.variants(np.concatenate([far_lo, far_hi]))
    earlier = position.position[: len(far_lo)].astype(np.int64)
    later = position.position[len(far_lo) :].astype(np.int64)
    gap = later - earlier
    assert (gap > near_bp).all(), "a Far Pair is beyond near_bp"
    assert (gap <= far_bp).all(), "and at most far_bp"
    lo, hi, at, partner = narrowest_far_pair(reader)
    assert partner - at > near_bp, "the region the far cases draw is wider than near_bp"
    assert (partner - at) // 2 + 1 <= far_bp // 2, "and its radius is one region answers"
    span = reader.variants(np.array([int(lo[0]), int(hi[0])])).position
    centre = (at + partner) // 2
    assert abs(int(span[0]) - centre) <= (partner - at) // 2 + 1, "the earlier end is inside"
    assert abs(int(span[1]) - centre) <= (partner - at) // 2 + 1, "and the later end too"
    first, second = ladder()
    assert (second > first).all(), "the reversed ladder really is reversed"
    assert (second < 1056).all(), "and every row is a row of the Variant Set"


#: Each case's answer on the Test Store, from the integration tip's code at `88b12e7`.
FROZEN: dict[str, str] = {
    "matrix-all": "664b3403af1f531fbd6b70a9d7dc3f2bd28ed1a779d6cc8b74b40e52fb61e45d",
    "matrix-chr22": "0ce0357853e4c31289a142b271152c41f224fb4cf0d93dd08d83550418551cd0",
    "matrix-fill-all": "c28b99f4eee1fa24c5c1bfc5a99c322f73ad25d62efa60907c71d3b361ec2bb2",
    "matrix-fill-chr22": "e66c6f45f7ada53d9c9ff3f49dfcc78c7925ce29e8e5330d93e6970e1be099a0",
    "matrix-fill-sample": "a2253e4c2df9f236348eebdaffe854243b2538ee94706d24a90016094c4240a6",
    "matrix-fill-small": "874dd3cacb3333bb535374b77f7124a10ec0a55fca80f41f7f7c9356bb2304a4",
    "matrix-sample": "77ecb33d0629b2a6d8962c9b2510ce9ebbb1f97ddafa5ee84882ccd0a478904d",
    "matrix-small": "cfbcb03db055059bac1734000b90345ba062300a0b811faadbdadaf67d06ac58",
    "pair_r-sample": "48055caf5f6390967ee794b8e115d8438e0ad909e3d469a2de9e17f36f9edc0c",
    "r_of_pairs-far": "8eb47c87d6945738faf5745422b6cb9b4bb57454654df7d3c11e1909cbb158c4",
    "r_of_pairs-far-reversed": "8eb47c87d6945738faf5745422b6cb9b4bb57454654df7d3c11e1909cbb158c4",
    "r_of_pairs-ladder": "0775294fefa463deab71658d64e17fc48c9639e3595b77d6ee38fe88f11f184d",
    "r_of_pairs-reversed": "0775294fefa463deab71658d64e17fc48c9639e3595b77d6ee38fe88f11f184d",
    "region-21-head-500k": "08fd4535b50471db5e04a459602591febc5c32d5b0f7bd382f85696cd11359ca",
    "region-21-tail-500k": "afba087e347c60479ecddb63a17a6c0355514b82785b29d0154c1810ce73af88",
    "region-21-whole-500k": "08fd4535b50471db5e04a459602591febc5c32d5b0f7bd382f85696cd11359ca",
    "region-22-head-500k": "041c32bf57e885054cb5e13cf5fd1bf481c3557561ee5a8fbf856166e10bef15",
    "region-22-tail-500k": "f6c01c31f8d660679824a87eba6f844ed6e3bcb3036b39e25564c266629fa06a",
    "region-22-whole-500k": "041c32bf57e885054cb5e13cf5fd1bf481c3557561ee5a8fbf856166e10bef15",
    "region-far": "d251318f017e7abe35a39505e6a9b0ffcbe8a530a9aadc6b995e28bf58c49fa9",
    "region_r-21-head": "956d65b437ca1788502c179b7f547f4ab349460eac0d4bcc3116b4794774650f",
    "region_r-21-tail": "ae97ed713f980738bc51be1f93eace8f6d9da7eaf8d9e00501308419c295aaff",
    "region_r-21-whole": "f2b1930c26b23075639c5a56d7823e0b0b1caa29c53114c6cbe18d1af89a2d87",
    "region_r-22-head": "5fe396ae4a862312817a93629261a3ea8274f577316d4060f718b702ad666016",
    "region_r-22-tail": "8e8a4f489b43ca17f350bf9b6cb1e58b2ff28132b8b21acca4b4ee5b28bf4039",
    "region_r-22-whole": "be9b3081ccceadf7529f9094adef3b8ba295528b3204892e2722e23675588cb7",
    "region_r-far": "9052a9df89613e94a95fcf082f85aedfe751c66218ac2c65ba5f920a9c4a4df9",
}


@pytest.fixture(scope="module")
def found() -> dict[str, str]:
    """Every case's answer on the committed Test Store, read once for the module."""
    return ages(opengwasdb.ld.open(TEST_STORE), StoreReader.open(TEST_STORE), ALL)


def test_the_frozen_answers_cover_every_case(found: dict[str, str]) -> None:
    assert set(found) == set(FROZEN), "a case was never frozen"


@pytest.mark.parametrize("case", sorted(FROZEN))
def test_a_matrix_or_a_region_answers_as_it_did_before_the_index_was_removed(
    case: str, found: dict[str, str]
) -> None:
    assert found[case] == FROZEN[case], f"{case} changed with the Proxy Index's removal"
