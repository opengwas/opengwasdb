"""`matrix` and `region` answer on the Test Store exactly as they did before #72.

The Proxy Index's removal touched the reader's chromosome-loading path and the arrays a store
holds, and neither may change a matrix or a region. `FROZEN` holds each answer as a digest of
every column it carries, recorded by running `fingerprint` against the **integration tip's code**
(`88b12e7`, before #72) on this same committed Test Store, never against the code this file
ships with. So a difference here is the removal's, and the digests say nothing about what the new
code would answer.

The cases are the reads #8's `bench-latency` and the contract make: every variant's matrix, a
sample, a thin slice of one chromosome; whole-chromosome regions and windows inside them; a
scattered `pair_r`; and `r_of_pairs` on a ladder of one row against many, both argument orders.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest
from store_copies import TEST_STORE

import ogld
from ogld.store import StoreReader

#: The Test Store's rows, and the samples the cases read.
ALL = np.arange(1056, dtype=np.int64)
SAMPLE = ALL[::7]
SMALL = ALL[:40]
SLICE = ALL[600:660]


def fingerprint(columns: dict[str, Any]) -> str:
    """Every column of an answer, read as bytes: an answer's fingerprint, never a subset of it."""
    hasher = hashlib.sha256()
    for name, value in columns.items():
        if isinstance(value, tuple):
            hasher.update(name.encode())
            for item in value:
                hasher.update(str(item).encode())
                hasher.update(b"\0")
            continue
        array = np.ascontiguousarray(value)
        hasher.update(name.encode())
        hasher.update(str(array.dtype).encode())
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def matrix(store: ogld.LDStore, rows: np.ndarray, *, fill: float | None = None) -> str:
    names = list(store._reader.alids_of(rows))
    got = store.matrix(names, fill=fill)
    if fill is not None:
        return fingerprint({"r": got.r})
    return fingerprint(
        {
            "chrom": got.variants.chrom,
            "position": got.variants.position,
            "alid": got.variants.alid,
            "r": got.r,
            "t_far": got.t_far,
        }
    )


def ages(store: ogld.LDStore, reader: StoreReader, rows: np.ndarray) -> dict[str, str]:
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
            found[f"region_r-{chrom}-{label}"] = fingerprint({"r": reader.region_r(a, b)})
            got = store.region(chrom, int(reader._columns["position"][a]), 500_000)
            found[f"region-{chrom}-{label}-500k"] = fingerprint(
                {
                    "r": got.r,
                    "chrom": got.variants.chrom,
                    "position": got.variants.position,
                    "alid": got.variants.alid,
                }
            )
    found["pair_r-sample"] = fingerprint({"r": reader.pair_r(ALL[::5])})
    first = np.repeat(ALL[::37], 40)
    second = np.minimum(first + np.tile(np.arange(40, dtype=np.int64), len(first) // 40), 1055)
    found["r_of_pairs-ladder"] = fingerprint({"r": reader.r_of_pairs(first, second)})
    found["r_of_pairs-reversed"] = fingerprint(
        {"r": reader.r_of_pairs(np.minimum(first, second), np.maximum(first, second))}
    )
    return found


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
    "r_of_pairs-ladder": "9b32a41ce1197c97367f1faf125c6d0f33069b79a25dc4ecfc2cb6c8ff216dca",
    "r_of_pairs-reversed": "9b32a41ce1197c97367f1faf125c6d0f33069b79a25dc4ecfc2cb6c8ff216dca",
    "region-21-head-500k": "08fd4535b50471db5e04a459602591febc5c32d5b0f7bd382f85696cd11359ca",
    "region-21-tail-500k": "afba087e347c60479ecddb63a17a6c0355514b82785b29d0154c1810ce73af88",
    "region-21-whole-500k": "08fd4535b50471db5e04a459602591febc5c32d5b0f7bd382f85696cd11359ca",
    "region-22-head-500k": "041c32bf57e885054cb5e13cf5fd1bf481c3557561ee5a8fbf856166e10bef15",
    "region-22-tail-500k": "f6c01c31f8d660679824a87eba6f844ed6e3bcb3036b39e25564c266629fa06a",
    "region-22-whole-500k": "041c32bf57e885054cb5e13cf5fd1bf481c3557561ee5a8fbf856166e10bef15",
    "region_r-21-head": "956d65b437ca1788502c179b7f547f4ab349460eac0d4bcc3116b4794774650f",
    "region_r-21-tail": "ae97ed713f980738bc51be1f93eace8f6d9da7eaf8d9e00501308419c295aaff",
    "region_r-21-whole": "f2b1930c26b23075639c5a56d7823e0b0b1caa29c53114c6cbe18d1af89a2d87",
    "region_r-22-head": "5fe396ae4a862312817a93629261a3ea8274f577316d4060f718b702ad666016",
    "region_r-22-tail": "8e8a4f489b43ca17f350bf9b6cb1e58b2ff28132b8b21acca4b4ee5b28bf4039",
    "region_r-22-whole": "be9b3081ccceadf7529f9094adef3b8ba295528b3204892e2722e23675588cb7",
}


@pytest.fixture(scope="module")
def found() -> dict[str, str]:
    """Every case's answer on the committed Test Store, read once for the module."""
    return ages(ogld.open(TEST_STORE), StoreReader.open(TEST_STORE), ALL)


def test_the_frozen_answers_cover_every_case(found: dict[str, str]) -> None:
    assert set(found) == set(FROZEN), "a case was never frozen"


@pytest.mark.parametrize("case", sorted(FROZEN))
def test_a_matrix_or_a_region_answers_as_it_did_before_the_index_was_removed(
    case: str, found: dict[str, str]
) -> None:
    assert found[case] == FROZEN[case], f"{case} changed with the Proxy Index's removal"
