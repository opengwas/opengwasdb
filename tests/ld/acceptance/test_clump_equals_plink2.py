"""Epic #1's acceptance for `clump` (ogld#6): plink2's leads, on the whole panel.

On an unadjusted (k = 0) HGDP+1kGP EUR LD Store, `clump` gives the same lead variants as
`plink2 --clump` for opengwas/opengwasdb#403's 37 EUR Analyses, the 5e-8 tier of each, at r²
0.01 and 0.001 with a 3 Mb window. It reads 8.6 M variants and takes about 25 minutes, so it is an
acceptance run, not part of the realdata tier: `pixi run -e dev acceptance-clump`
(CONTRIBUTING.md, "Acceptance runs"). Its result is recorded on the ticket.

Two inputs, named by environment variables that `tests/acceptance/run.sh` passes on:

- `OGLD_EUR_K0_STORE`: the k = 0 EUR LD Store.
- `OGLD_403_ANALYSES`: one directory per Analysis, each holding `hits.tsv` (a header, then
  ALID and z per row: the 5e-8 tier) and `leads_r2_0.01.txt` and `leads_r2_0.001.txt` (plink2's
  index variants, one ALID per line, from the same genotypes at k = 0).
  `scripts/realdata/make_403_inputs.py` makes them.

Every lead that differs is listed in `explained_differences.json` with the pairs that explain it
(spec §3): a pair whose stored r̂² and plink2's true r² lie on opposite sides of the threshold. The
file names the store it was made on, by its encoding and a hash of its provenance, and the test
asserts that first. Then it asserts the differences equal the file's, and that every listed pair
touches a differing lead, lies within the window, takes its r̂² from the store, has plink2's r² on
the other side of the threshold, and is within the store's own encoding band.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import ogld
from ogld.format import REncoding

EXPLAINED_FILE = Path(__file__).with_name("explained_differences.json")
KB = 3000


def provenance_sha256(store: ogld.LDStore) -> str:
    """A hash of the store's provenance as recorded: it names the build, not just the panel."""
    text = json.dumps(dict(store.provenance), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def named(variable: str, what: str) -> Path:
    """The path `$variable` names. An acceptance run with an input missing fails: it was asked
    for, and a pass would claim it ran."""
    value = os.environ.get(variable, "")
    assert value, f"${variable} must name {what}; it is unset"
    assert Path(value).exists(), f"${variable} must name {what}; {value} is not there"
    return Path(value)


def positions(alid: str, other: str) -> int:
    return abs(int(alid.split(":")[1]) - int(other.split(":")[1]))


def check_pair(store: ogld.LDStore, case: dict, pair: dict, differing: set[str]) -> None:
    """A listed pair is tied to its difference and to the store, not taken on trust."""
    a, b, r2 = pair["a"], pair["b"], case["r2"]
    where = f"{case['analysis']} at {r2}: {pair}"
    assert {a, b} & differing, f"{where} touches no differing lead"
    assert positions(a, b) < KB * 1000, f"{where} is not within the window"
    held = store.matrix([a, b])
    assert held.absent == (), f"{where} is not in the store"
    r = float(held.r[0, 1])
    assert r * r == pytest.approx(pair["store_r2"], abs=1e-7), f"{where}: the store holds {r * r}"
    assert (pair["store_r2"] > r2) != (pair["plink2_r2"] > r2), f"{where} is not on the threshold"
    e = REncoding.of(store.provenance["encoding"]).max_error
    assert abs(r * r - r2) <= 2 * abs(r) * e + e * e, f"{where} is outside the store's band"


@pytest.mark.parametrize("r2", [0.01, 0.001])
def test_leads_equal_plink2_s_on_403_s_37_analyses(r2: float) -> None:
    store = ogld.open(named("OGLD_EUR_K0_STORE", "the k = 0 EUR LD Store"))
    root = named("OGLD_403_ANALYSES", "#403's 37 Analyses with plink2's leads")
    assert store.provenance["adjustment"]["k"] == 0, "the comparison is with unadjusted LD"
    snapshot = json.loads(EXPLAINED_FILE.read_text())
    assert snapshot["store"]["encoding"] == store.provenance["encoding"], "another encoding"
    assert snapshot["store"]["provenance_sha256"] == provenance_sha256(store), (
        f"{EXPLAINED_FILE.name} was made on another store than {store.provenance['build']}"
    )
    analyses = sorted(path for path in root.iterdir() if path.is_dir())
    assert len(analyses) == 37, f"#403 has 37 EUR Analyses; {root} holds {len(analyses)}"
    differences = {}
    for analysis in analyses:
        hits = np.loadtxt(analysis / "hits.tsv", dtype=str, skiprows=1, ndmin=2)
        alid, z = list(hits[:, 0]), np.abs(hits[:, 1].astype(np.float64))
        ours = set(map(str, store.clump(alid, z, r2=r2, kb=KB).leads()))
        theirs = set((analysis / f"leads_r2_{r2}.txt").read_text().split())
        if ours != theirs:
            differences[analysis.name] = (sorted(ours - theirs), sorted(theirs - ours))
    cases = {c["analysis"]: c for c in snapshot["cases"] if c["r2"] == r2}
    listed = {name: (c["ogld_only"], c["plink2_only"]) for name, c in cases.items()}
    assert listed == differences, "leads differ from plink2's other than as the file lists them"
    for case in cases.values():
        differing = set(case["ogld_only"]) | set(case["plink2_only"])
        for pair in case["threshold_pairs"]:
            check_pair(store, case, pair, differing)
