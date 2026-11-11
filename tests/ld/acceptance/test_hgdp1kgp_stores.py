"""Epic #1's acceptance for the five HGDP+1kGP LD Stores (ogld#7): every validation rule passes
on each, and each records what a reader of its provenance needs.

The stores are the ones `scripts/build_hgdp1kgp.py build` writes under `$OGLD_STORES`
(`/data/opengwasdb/work/ogld/stores`, `hgdp1kgp-v2-<ANC>-k10.store`); `$OGLD_STORE_<ANC>`
names one elsewhere. `$OGLD_KEEP` holds the keep files (default as `scripts/make_keeps.py`'s), so
n is checked against the unrelated samples, not against what the store says of itself.
Validating all five takes about an hour of the node (`pixi run -e dev acceptance-stores`),
which is why this is an acceptance run and not the realdata tier. An input missing fails the
run; it never skips (CONTRIBUTING.md, "Acceptance runs").
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

import ogld
from ogld.format import VALIDATION_RULES
from ogld.store.validation import validate_at

#: ADR 0002, decision 11: each panel's unrelated samples (`scripts/make_keeps.py`).
PANELS = {"EUR": 665, "AFR": 608, "EAS": 724, "SAS": 672, "AMR": 341}
STORES = "/data/opengwasdb/work/ogld/stores"
KEEP = "/data/opengwasdb/work/ogld/keep"
R_SCALE = 511  # ADR 0004
K_TARGET = 10
K_MIN = 5


def store_path(panel: str) -> Path:
    default = Path(os.environ.get("OGLD_STORES", STORES)) / f"hgdp1kgp-v2-{panel}-k10.store"
    path = Path(os.environ.get(f"OGLD_STORE_{panel}", default))
    assert path.exists(), f"the {panel} store is not at {path}: build it or set $OGLD_STORE_{panel}"
    return path


def keep_count(panel: str) -> int:
    keep = Path(os.environ.get("OGLD_KEEP", KEEP)) / f"{panel}.keep"
    assert keep.exists(), f"{keep} is missing: scripts/make_keeps.py writes it"
    return len(keep.read_text().splitlines()) - 1  # the #IID header


@pytest.mark.parametrize("panel", PANELS)
def test_every_rule_passes(panel: str) -> None:
    threads = int(os.environ.get("OGLD_VALIDATE_THREADS", "16"))
    found = validate_at(store_path(panel), threads=threads)
    assert list(found) == list(VALIDATION_RULES)
    failing = {rule: findings[:3] for rule, findings in found.items() if findings}
    assert not failing, f"{panel} breaks {len(failing)} rule(s): {failing}"


@pytest.mark.parametrize("panel", PANELS)
def test_provenance_records_what_a_reader_needs(panel: str) -> None:
    p = ogld.open(store_path(panel)).provenance
    assert p["panel"]["id"] == f"hgdp1kgp-v2-{panel}"
    assert p["panel"]["ancestry"] == panel
    assert p["panel"]["n"] == PANELS[panel] == keep_count(panel)
    assert p["encoding"]["r_scale"] == R_SCALE
    assert p["n_variants"] > 0
    assert p["counts"]["near_pairs"] > 0, "the Near Band's pair count"
    assert p["counts"]["index_entries"] > 0, "the Proxy Index's size"
    t_far = p["t_far"]
    assert isinstance(t_far, float), f"t_far is {t_far!r}"
    assert 0 < t_far < 1, f"t_far is {t_far!r}"
    build = p["build"]
    assert build["wall_seconds"] > 0, "the build time"
    assert build["peak_memory_bytes"] > 0
    assert p["software"]["ogld"], "the builder's version"


@pytest.mark.parametrize("panel", PANELS)
def test_the_pc_check_and_its_diagnostic_are_recorded(panel: str) -> None:
    adjustment = ogld.open(store_path(panel)).provenance["adjustment"]
    assert adjustment["k_target"] == K_TARGET
    assert adjustment["k_min"] == K_MIN
    dropped = adjustment["pc_check"]["dropped"]
    assert isinstance(dropped, list), "the PC Check's dropped PCs are recorded, even when none"
    assert K_MIN <= adjustment["k"] <= K_TARGET, f"{panel} kept {adjustment['k']} PCs"
    assert len(adjustment["covariates"]) == adjustment["k"]
    diagnostic = ogld.open(store_path(panel)).provenance["pc_diagnostic"]
    assert diagnostic["pairs"]["n_pairs"] > 0
    for rate in (
        "rate_r2_ge_0_01_unadjusted",
        "rate_r2_ge_0_01_adjusted",
        "rate_r2_ge_0_01_sampling",
    ):
        assert math.isfinite(diagnostic[rate]), f"{rate} is {diagnostic[rate]!r}"
    change = diagnostic["near_100kb_change"]
    assert set(change) == {"0", "0.5", "0.9", "0.99", "1"}
    assert all(math.isfinite(v) for v in change.values())
