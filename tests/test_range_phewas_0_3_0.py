"""On format 0.3.0 `range_phewas` is a binary search of the position sidecar in every layout (#468).

A 0.2 release answers a range by a tabix fetch of its BGZF Variant Table, which loads
pysam. From 0.3.0 the position sidecar answers it (ADR 0105 §6). Each layout below is
built from one source spanning three chromosomes, with a position three variants share
and a chromosome-2 variant at a position below chromosome 1's last, so a range that
leaks over a boundary shows. The Hybrid answers from its Dense Component and its
Ragged Overflow both. Every range the rewrite's proof asks (`range_questions`) is asked
in a fresh interpreter: it answers without importing pysam, and exactly as the same
release turned back into its 0.2.3 form (`format_0_2_3.as_0_2_3`) answers by tabix.
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
from one_source_releases import dense_release, hybrid_release, ragged_release

from opengwasdb.query import query_store
from opengwasdb.store.query_digests import range_questions

#: ``(chrom, pos, ref, alt, beta, se, af_alt)``, as test_indel_pairs. Every z is 6 or
#: more, so a signals-only Ragged build keeps every row.
ROWS = [
    ("1", "1000", "A", "G", "0.06", "0.01", "0.2"),
    ("1", "5000", "C", "A", "0.07", "0.01", "0.3"),
    ("1", "5000", "C", "G", "-0.08", "0.01", "0.25"),
    ("1", "5000", "C", "T", "0.09", "0.01", "0.35"),
    ("1", "9000", "G", "A", "-0.06", "0.01", "0.4"),
    ("2", "300", "T", "C", "0.07", "0.01", "0.2"),
    ("2", "7000", "A", "C", "-0.07", "0.01", "0.3"),
    ("X", "2500", "G", "T", "0.08", "0.01", "0.45"),
]
#: The Hybrid's Dense Component panel: the rest are Ragged Overflow.
PANEL = ["1:1000:A:G", "1:5000:C:A", "1:5000:C:T", "2:300:T:C"]

#: Every question `range_questions` can pose; the fixture must pose them all.
QUESTIONS = {
    "first", "to_first", "last", "from_last", "boundary/before", "boundary/after",
    "empty/between", "empty/chromosome", "shared", "shared/from_before", "shared/to_after",
}
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


LAYOUTS = {
    "dense": lambda root: dense_release(root, ROWS),
    "ragged": lambda root: ragged_release(root, ROWS),
    "hybrid": lambda root: hybrid_release(root, ROWS, PANEL),
}


@pytest.fixture(scope="module", params=sorted(LAYOUTS))
def release(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One layout's 0.3.0 release. Never written to: copy one to change it."""
    return LAYOUTS[request.param](tmp_path_factory.mktemp(request.param))


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
    questions = _questions(release)
    assert set(questions) == QUESTIONS
    found = _probe(release, questions)
    for name in QUESTIONS:
        rows = found[f"{name}/False"]["rows"]
        assert (rows == []) == (name in EMPTY), (name, questions[name], rows)
    assert len(found["shared/False"]["rows"]) == 3, "three variants share the position"
    # A range from chromosome 1's last variant to its end must not reach chromosome 2's
    # first, whose position is lower: the case a boundary leak would answer wrongly.
    assert len(found["boundary/before/False"]["rows"]) == 1
    panel_file = release / "dense" / "dense_to_shared.npy"
    if panel_file.exists():
        on_panel = set(np.load(panel_file).tolist())
        assert 0 < len(on_panel) < len(ROWS), "a Hybrid answers from both of its parts"
        shared = set(found["shared/False"]["rows"])
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
