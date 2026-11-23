"""On format 0.3.0 `range_phewas` is a binary search of the position sidecar in every layout (#468).

A 0.2 release answers a range by a tabix fetch of its BGZF Variant Table, which loads
pysam. From 0.3.0 the position sidecar answers it (ADR 0105 §6). Each layout is built
once per session (`range_releases`, from `one_source_releases.ROWS`): three
chromosomes, a position three variants share, and a chromosome-2 variant at a position
below chromosome 1's last, so a range that leaks over a boundary shows. The Hybrid
answers from its Dense Component and its Ragged Overflow both. Every range the
rewrite's proof asks (`range_questions`) is asked in a fresh interpreter: it answers
without importing pysam, and exactly as the same release turned back into its 0.2.3
form (`format_0_2_3.as_0_2_3`) answers by tabix.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from format_0_2_3 import as_0_2_3
from one_source_releases import PANEL, QUESTIONS, ROWS

from opengwasdb.query import query_store
from opengwasdb.store.query_digests import RANGE_QUESTIONS, range_questions

EMPTY = {"empty/between", "empty/chromosome"}

#: Ranges beyond the proof's, where tabix's own conventions decide the answer: an end
#: before the start, a start at or below 0, and two spellings of a chromosome.
EDGES = {
    "edge/inverted": ("1", 9000, 1000),
    "edge/touching": ("1", 5001, 5000),
    "edge/from_below_zero": ("1", -5, 1000),
    "edge/chr_prefix": ("chr1", 1, 5000),
    "edge/numeric_x": ("23", 1, 10**6),
}

PROBE = r"""
import json, sys
from opengwasdb.query import query_store
from opengwasdb.store.query_digests import result_digests

path, questions = sys.argv[1], json.loads(sys.argv[2])
out = {}
with query_store(path) as q:
    for name, (chrom, start, end) in sorted(questions.items()):
        for observed_only in (False, True):
            result = q.range_phewas(chrom, start, end, observed_only=observed_only)
            out[f"{name}/{observed_only}"] = {
                "rows": sorted(set(result["variant_index"].tolist())),
                "digests": result_digests(dict(result)),
            }
    out["pysam"] = "pysam" in sys.modules
print(json.dumps(out))
"""


@pytest.fixture(params=["dense", "ragged", "hybrid"])
def release(request: pytest.FixtureRequest, range_releases: dict[str, Path]) -> Path:
    """One layout's 0.3.0 release, from the session's `range_releases`. Never written
    to: copy one to change it."""
    return range_releases[request.param]


def _questions(store: Path) -> dict[str, tuple[str, int, int]]:
    with query_store(store) as q:
        return range_questions(q.variants_table())


def _probe(store: Path, questions: dict[str, tuple[str, int, int]]) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", PROBE, str(store), json.dumps(questions)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def test_the_fixture_poses_every_range_question(release: Path) -> None:
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == "0.3.0"
    with query_store(release) as q:
        questions = range_questions(q.variants_table())
        assert set(questions) == QUESTIONS == set(RANGE_QUESTIONS)
        found = {
            name: sorted(set(q.range_phewas(*asked)["variant_index"].tolist()))
            for name, asked in questions.items()
        }
    for name in QUESTIONS:
        assert (found[name] == []) == (name in EMPTY), (name, questions[name], found[name])
    assert len(found["shared"]) == 3, "three variants share the position"
    # A range from chromosome 1's last variant to its end must not reach chromosome 2's
    # first, whose position is lower: the case a boundary leak would answer wrongly.
    assert len(found["boundary/before"]) == 1
    panel_file = release / "dense" / "dense_to_shared.npy"
    if panel_file.exists():
        on_panel = set(np.load(panel_file).tolist())
        assert 0 < len(on_panel) < len(ROWS), "a Hybrid answers from both of its parts"
        shared = set(found["shared"])
        assert shared & on_panel and shared - on_panel, "the shared position spans both"


def test_the_hybrid_panel_splits_the_ranges_between_its_two_parts() -> None:
    """The first variant, and one of the shared position's three, are on the Dense
    Component's panel; the last, and the other two, are Ragged Overflow."""
    alids = [f"{c}:{p}:{r}:{a}" for c, p, r, a, *_ in ROWS]
    assert alids[0] in PANEL and alids[-1] not in PANEL
    shared = [alid for alid in alids if alid.startswith("1:5000:")]
    assert 0 < len(set(shared) & set(PANEL)) < len(shared)


def test_range_phewas_answers_as_0_2_3_without_pysam(release: Path, tmp_path: Path) -> None:
    questions = {**_questions(release), **EDGES}
    found = _probe(release, questions)
    assert found["edge/inverted/False"]["rows"] == found["edge/touching/False"]["rows"] == []
    assert len(found["edge/chr_prefix/False"]["rows"]) == 4
    assert len(found["edge/numeric_x/False"]["rows"]) == 1
    assert found.pop("pysam") is False, "a 0.3.0 range loaded pysam"

    legacy = Path(shutil.copytree(release, tmp_path / "as-0.2.3"))
    as_0_2_3(legacy)
    by_tabix = _probe(legacy, questions)
    assert by_tabix.pop("pysam") is True, "the 0.2.3 form must answer by tabix"
    assert found == by_tabix
