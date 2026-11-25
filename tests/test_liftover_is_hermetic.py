"""The test suite lifts hg19 -> hg38 without the network (#474).

pyliftover's built-in lift downloads UCSC's chain into `~/.pyliftover` the first time it
runs; on a runner where that download failed, 347 tests failed or errored. The suite reads
the committed chain instead (`tests/data/liftover/`, through the session fixture in
`conftest.py`). Here the network is blocked, and a built-in lift must still work, from
that file and no other: this host has a `~/.pyliftover`, so a lift that works is not
enough to show the committed chain was read.
"""

from __future__ import annotations

import gzip
import hashlib
import urllib.request
from pathlib import Path

import pyliftover.chainfile
import pyliftover.liftover
import pytest
from pyliftover import LiftOver

from opengwasdb.build import liftover

CHAIN = Path(__file__).resolve().parent / "data" / "liftover" / "hg19ToHg38.over.chain.gz"

#: UCSC's hg19ToHg38.over.chain.gz (`tests/data/liftover/README.md`).
CHAIN_SHA256 = "5c0598e500ceb5a78c73086929e8ef993aec309bcafb595139b53d440b125a1d"


def _offline(url: str, *args: object, **kwargs: object) -> None:
    raise OSError(f"a test reached the network: {url}")


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No download: `urllib.request.urlretrieve` raises, and so does the copy pyliftover
    bound when it was imported. The process-wide `LiftOver` cache starts empty."""
    monkeypatch.setattr(urllib.request, "urlretrieve", _offline)
    monkeypatch.setattr(pyliftover.chainfile, "urlretrieve", _offline)
    monkeypatch.setattr(liftover, "_LIFTOVER_CACHE", {})


def test_the_committed_chain_is_ucscs_hg19_to_hg38():
    assert hashlib.sha256(CHAIN.read_bytes()).hexdigest() == CHAIN_SHA256
    with gzip.open(CHAIN, "rt") as chain:
        assert chain.readline().startswith("chain ")
    assert (CHAIN.parent / "README.txt").read_text().startswith(
        'The data files ("UCSC Chain Files") in this directory are property of The\n'
        "Regents of the University of California"
    ), "UCSC's licence must travel with the chain"


@pytest.mark.usefixtures("offline")
def test_the_built_in_hg19_to_hg38_opens_the_committed_chain_with_the_network_blocked():
    opened = pyliftover.liftover.open_liftover_chain_file(from_db="hg19", to_db="hg38")
    assert opened is not None
    with opened:
        assert Path(opened.name) == CHAIN  # not ~/.pyliftover's copy, nor a download
    with pytest.raises(OSError, match="reached the network"):
        urllib.request.urlretrieve("http://hgdownload.cse.ucsc.edu/")


@pytest.mark.usefixtures("offline")
def test_a_built_in_hg19_to_hg38_lift_works_with_the_network_blocked():
    # Positions `test_dense_vcf_build.py` documents: hg19 1:1,000,000 is hg38 1:1,064,620.
    assert LiftOver("hg19", "hg38").convert_coordinate("chr1", 999_999)[0][:2] == (
        "chr1", 1_064_619
    )
    rows, attempts, failures = liftover.lifted_rows(
        [("1", 1_000_000, "C", "T"), ("1", 1_500_000, "G", "A")]
    )
    assert (attempts, failures) == (2, 0)
    assert [r.alid for r in rows.values()] == ["1:1064620:C:T", "1:1564620:A:G"]


@pytest.mark.usefixtures("offline")
def test_a_build_pair_with_no_committed_chain_fails_rather_than_downloading():
    with pytest.raises(AttributeError, match="readline"):
        LiftOver("hg38", "hg19")
