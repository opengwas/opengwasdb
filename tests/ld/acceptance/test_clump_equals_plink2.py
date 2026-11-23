"""Epic #1's acceptance for `clump` (ogld#6): plink2's leads, on the whole panel.

On an unadjusted (k = 0) HGDP+1kGP EUR LD Store, `clump` gives the same lead variants as
`plink2 --clump` for opengwas/opengwasdb#403's 37 EUR Analyses, the 5e-8 tier of each, at r²
0.01 with a 3 Mb window. It reads 8.6 M variants and takes under an hour, so it is an acceptance
run, not part of the realdata tier: `pixi run -e dev acceptance-clump` (CONTRIBUTING.md,
"Acceptance runs"). Its result is recorded on the ticket.

**r² 0.001 is not run** (owner, #64; partner-rows §8.1, D2): from format 0.2.0 a request within
3 Mb whose r2 is below the store's Partner Threshold is refused with
`threshold_below_t_partner`, and the HGDP+1kGP stores record t_partner = 0.01, so an r² 0.001
clump on this store is a Refusal, not an answer to compare. The comparison is made at the
threshold the store supports; a panel whose t_partner were 0.001 — UK Biobank EUR's — could
compare at 0.001, and none of the HGDP+1kGP stores can.

Two inputs, named by environment variables that `tests/acceptance/run.sh` passes on:

- `OGLD_EUR_K0_STORE`: the k = 0 EUR LD Store.
- `OGLD_403_ANALYSES`: one directory per Analysis, each holding `hits.tsv` (a header, then
  ALID and z per row: the 5e-8 tier) and `leads_r2_0.01.txt` (plink2's index variants, one ALID
  per line, from the same genotypes at k = 0). `scripts/realdata/make_403_inputs.py` makes them,
  and `prepare` then `clump` write the pins that name the store they were made from.

Every lead that differs is listed in `explained_differences.json` with the pairs that explain it
(spec §3): a pair whose stored r̂² and plink2's true r² lie on opposite sides of the threshold. The
file names the store it was made on, by its encoding and a hash of its provenance, and the test
asserts that first. Then it asserts the differences equal the file's, and that every listed pair
touches a differing lead, lies within the window, takes its r̂² from the store, has plink2's r² on
the other side of the threshold, and is within the store's own encoding band.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import clump_rule
import make_403_inputs as maker
import numpy as np
import pytest

import opengwasdb.ld
from opengwasdb.ld.format import REncoding
from opengwasdb.ld.identity import alid_locus

EXPLAINED_FILE = Path(__file__).with_name("explained_differences.json")
KB = 3000
#: The one threshold this store is compared at: its own t_partner, which is the lowest r2 a
#: 0.2.0 request within 3 Mb may ask it for (partner-rows §8.1; the module docstring says why
#: 0.001, which plink2 was also run at, is no longer a comparison here).
R2 = 0.01


def named(variable: str, what: str) -> Path:
    """The path `$variable` names. An acceptance run with an input missing fails: it was asked
    for, and a pass would claim it ran."""
    value = os.environ.get(variable, "")
    assert value, f"${variable} must name {what}; it is unset"
    assert Path(value).exists(), f"${variable} must name {what}; {value} is not there"
    return Path(value)


def distance(alid: str, other: str) -> float:
    """The bp between two ALIDs on one chromosome; infinite across chromosomes (no pair there)."""
    here, there = alid_locus(alid), alid_locus(other)
    assert here is not None, f"{alid} must be an ALID"
    assert there is not None, f"{other} must be an ALID"
    return abs(here[1] - there[1]) if here[0] == there[0] else math.inf


def check_pair(store: opengwasdb.ld.LDStore, case: dict, pair: dict, differing: set[str]) -> None:
    """A listed pair is tied to its difference and to the store, not taken on trust."""
    a, b, r2 = pair["a"], pair["b"], case["r2"]
    where = f"{case['analysis']} at {r2}: {pair}"
    assert {a, b} & differing, f"{where} touches no differing lead"
    assert distance(a, b) < KB * 1000, f"{where} is not within the window"
    held = store.matrix([a, b])
    assert held.absent == (), f"{where} is not in the store"
    r = float(held.r[0, 1])
    assert r * r == pytest.approx(pair["store_r2"], abs=1e-7), f"{where}: the store holds {r * r}"
    assert (pair["store_r2"] > r2) != (pair["plink2_r2"] > r2), f"{where} is not on the threshold"
    e = REncoding.of(store.provenance["encoding"]).max_error
    assert clump_rule.within_band(r, r2, e), f"{where} is outside the store's band"


def test_the_inputs_say_which_store_and_keep_file_they_were_made_from() -> None:
    """The pins `make_403_inputs.py` writes are read here: inputs without them, or made from
    another store, keep file, or plink2, are refused, not compared."""
    store = opengwasdb.ld.open(named("OGLD_EUR_K0_STORE", "the k = 0 EUR LD Store"))
    root = named("OGLD_403_ANALYSES", "#403's 37 Analyses with plink2's leads")
    manifest = json.loads((root / "manifest.json").read_text())
    for key in ("opengwasdb_sha", "plink2_version", "provenance_sha256", "keep_sha256"):
        assert manifest.get(key), f"manifest.json has no {key}: remake the inputs with the script"
    assert manifest["provenance_sha256"] == maker.provenance_sha256(store.provenance), (
        "the inputs were made from another store than this one"
    )
    assert manifest["keep_sha256"] == store.provenance["panel"]["samples_sha256"]
    prepared = json.loads((root.parent / "EUR-k0.plink2" / "pins.json").read_text())
    assert prepared["provenance_sha256"] == manifest["provenance_sha256"]
    assert prepared["keep_sha256"] == manifest["keep_sha256"]
    assert prepared["plink2_version"] == manifest["plink2_version"]


def test_leads_equal_plink2_s_on_403_s_37_analyses() -> None:
    store = opengwasdb.ld.open(named("OGLD_EUR_K0_STORE", "the k = 0 EUR LD Store"))
    root = named("OGLD_403_ANALYSES", "#403's 37 Analyses with plink2's leads")
    assert store.provenance["adjustment"]["k"] == 0, "the comparison is with unadjusted LD"
    snapshot = json.loads(EXPLAINED_FILE.read_text())
    assert snapshot["store"]["encoding"] == store.provenance["encoding"], "another encoding"
    assert snapshot["store"]["provenance_sha256"] == maker.provenance_sha256(store.provenance), (
        f"{EXPLAINED_FILE.name} was made on another store than {store.provenance['build']}"
    )
    analyses = sorted(path for path in root.iterdir() if path.is_dir())
    assert len(analyses) == 37, f"#403 has 37 EUR Analyses; {root} holds {len(analyses)}"
    differences = {}
    for analysis in analyses:
        hits = np.loadtxt(analysis / "hits.tsv", dtype=str, skiprows=1, ndmin=2)
        alid, z = list(hits[:, 0]), np.abs(hits[:, 1].astype(np.float64))
        ours = set(map(str, store.clump(alid, z, r2=R2, kb=KB).leads()))
        theirs = set((analysis / f"leads_r2_{R2}.txt").read_text().split())
        if ours != theirs:
            differences[analysis.name] = (sorted(ours - theirs), sorted(theirs - ours))
    cases = {c["analysis"]: c for c in snapshot["cases"] if c["r2"] == R2}
    listed = {name: (c["ogld_only"], c["plink2_only"]) for name, c in cases.items()}
    assert listed == differences, "leads differ from plink2's other than as the file lists them"
    for case in cases.values():
        differing = set(case["ogld_only"]) | set(case["plink2_only"])
        for pair in case["threshold_pairs"]:
            check_pair(store, case, pair, differing)
