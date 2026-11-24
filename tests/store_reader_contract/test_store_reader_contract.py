"""The Store Reader contract: the committed bundle answers as `expected/` says (#474).

Reads only the committed releases, through the Store Reader alone (`contract_reader`),
and builds nothing, so it runs wherever the Store Reader does: ogquery runs it on Linux,
macOS and Windows. It imports nothing from opengwasdb's test helpers.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from contract_reader import (
    BUNDLE,
    RELEASES,
    expected,
    mismatches,
    open_query,
    release_names,
    version,
)

NAMES = release_names()

#: What importing and asking the Store Reader every question must not load.
HEAVY = ("pandas", "scipy", "pysam", "sqlite3", "threadpoolctl")


def test_the_bundle_holds_every_layout_in_both_completion_states():
    layouts = {
        (m["primary_layout"], m["completion_state"])
        for m in (_manifest(RELEASES / name) for name in NAMES)
    }
    assert layouts == {
        (layout, state)
        for layout in ("dense", "ragged", "hybrid")
        for state in ("observed_only", "reference_completed")
    }
    assert sorted(p.stem for p in (BUNDLE / "expected").glob("*.json")) == NAMES


def test_version_names_the_format_every_release_and_answer_is():
    stamped = version()
    assert set(stamped) == {"bundle", "format_version"}
    for name in NAMES:
        for manifest in sorted((RELEASES / name).rglob("manifest.json")):
            assert _stamp(manifest) == stamped["format_version"], manifest
        want = expected(name)
        assert want["format_version"] == stamped["format_version"], name
        assert want["bundle_version"] == stamped["bundle"], name
        assert want["release"] == name


def test_the_readme_names_every_release_and_no_other():
    readme = (BUNDLE / "README.md").read_text(encoding="utf-8")
    named = re.findall(r"^\| `([a-z_]+)` \| (?:Dense|Ragged|Hybrid),", readme, flags=re.M)
    assert sorted(named) == NAMES
    assert len(set(named)) == len(named), "a release is named twice"


@pytest.mark.parametrize("name", NAMES)
def test_every_release_answers_as_expected(name: str):
    assert mismatches(RELEASES / name, expected(name)) == []


def test_the_bundle_holds_every_state_it_claims():
    """Asserted of the expected answers, so the contract cannot pass by asking nothing."""
    every = {name: expected(name) for name in NAMES}
    readable = {name: e["readable"] for name, e in every.items()}
    long = {
        len(row["alid"].encode())
        for r in readable.values()
        for row in r["variants_table"]["probed"]
    }
    assert {23, 24, 25, 32, 33} <= long and max(long) > 100, sorted(long)
    site = readable["dense_ranges"]["variant_reads"]["variants_in_range/shared"]["head"]
    assert [row["rsid"] for row in site] == ["rs5000", "rs5000"], "a multi-allelic rsID"
    refused = {
        a["error"]: a["fields"]
        for r in readable.values()
        for section in ("refusals", "shapes")
        for a in r[section].values()
        if "error" in a
    }
    assert set(refused) == {
        "UnknownAnalysisError", "UnstoredThresholdError", "UnsupportedChromosome",
    }
    # A port compares a refusal's type and its named fields, never its message (README).
    assert refused["UnknownAnalysisError"] == {"analysis_id": "absent"}
    assert refused["UnsupportedChromosome"] == {"chromosome": "27"}
    assert set(refused["UnstoredThresholdError"]) == {"threshold", "stored"}
    for name in NAMES:
        unknown = {k: a for k, a in readable[name]["refusals"].items() if "unknown" in k}
        assert {a["error"] for a in unknown.values()} == {"UnknownAnalysisError"}, name
        assert len(unknown) == 3, (name, sorted(unknown))
    infinite = [
        row
        for shape in readable["dense_ranges"]["shapes"].values()
        for row in shape.get("head", [])
        if row["se"] == "Infinity"
    ]
    assert infinite and all(r["pval"] == r["beta"] == "NaN" for r in infinite)
    assert all(every[n]["digests"]["indexed_subsets"] for n in NAMES if n.endswith(
        ("observed", "completed")) and not n.startswith("ragged"))
    offsetless = sorted(
        tier.relative_to(RELEASES).as_posix()
        for tier in RELEASES.glob("**/top_hits/p_*")
        if not (tier / "analysis_offsets").is_dir()
    )
    assert len(offsetless) == 4, offsetless  # a Hybrid's two components
    for name in NAMES:
        answered = [s for s, a in readable[name]["shapes"].items() if a.get("rows")]
        assert len(answered) > 10, (name, len(answered))


def _statuses(readable: dict) -> set[str]:
    return {
        status
        for section in ("shapes", "alid_probes", "misses", "range_probes", "indexed_subsets")
        for answer in readable[section].values()
        for status in answer.get("by_status", {})
    }


def test_the_readable_rows_reach_every_status_a_release_holds():
    """A port checks imputed and missing rows from `by_status`, without the digests: each
    Reference-Completed release spells out an imputed row, and the Ragged one, whose CSR
    stores a row where imputation failed, a missing row (spec §19)."""
    for name in NAMES:
        statuses = _statuses(expected(name)["readable"])
        want = {"observed"} | ({"imputed"} if name.endswith("_completed") else set())
        want |= {"missing"} if name == "ragged_completed" else set()
        assert statuses == want, (name, statuses)


@pytest.mark.parametrize("name", [n for n in NAMES if not n.endswith("_ranges")])
def test_each_digest_release_holds_its_planted_rows(name: str):
    """Read through the Store Reader's Variant Axis: `source_alid` in states 0, 1 and 2
    (ADR 0105 §8), rows with no rsID, and a cell naming two rsIDs."""
    with open_query(RELEASES / name) as q:
        rows = list(q.variants_table().values())
    states = {
        0 if row["source_alid"] is None else 1 if row["source_alid"] == row["alid"] else 2
        for row in rows
    }
    assert states == {0, 1, 2}
    rsids = [row["rsid"] for row in rows]
    assert rsids.count(None) == 2 and rsids.count("rs77;rs78") == 1, rsids[:4]


#: Sidecar files a byte of which the contract must notice changed: an ALID's text, a
#: position, a lifted `source_alid`'s text and an rsID.
CORRUPTED = (
    "variant_alid_utf8.npy",
    "variant_position.npy",
    "variant_source_alid_utf8.npy",
    "variant_rsid_bytes.npy",
)


@pytest.mark.parametrize("sidecar", CORRUPTED)
def test_one_corrupted_sidecar_byte_fails_the_contract(tmp_path: Path, sidecar: str):
    release = Path(shutil.copytree(RELEASES / "dense_observed", tmp_path / "dense_observed"))
    path = release / sidecar
    values = np.load(path)
    assert values.size, f"{sidecar} holds nothing to corrupt"
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01  # the last byte is the last value's, never the header's
    path.write_bytes(bytes(data))
    assert not np.array_equal(np.load(path), values), "the flip changed no value"
    try:
        found = mismatches(release, expected("dense_observed"))
    except Exception as refused:  # a reader that refuses the release notices too
        found = [f"{type(refused).__name__}: {refused}"]
    assert found, f"one byte of {sidecar} changed and the contract still passed"


def test_asking_every_question_loads_no_heavy_dependency():
    """Every adapter asked every question in a fresh interpreter: none of `HEAVY` loads."""
    probe = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(BUNDLE)!r})\n"
        "from pathlib import Path\n"
        "from contract_reader import RELEASES, answers\n"
        "for name in ('dense_observed', 'ragged_completed', 'hybrid_completed'):\n"
        "    answers(RELEASES / name)\n"
        f"print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], cwd=BUNDLE, capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


def _manifest(release: Path) -> dict:
    return json.loads((release / "manifest.json").read_text(encoding="utf-8"))


def _stamp(manifest: Path) -> str:
    return str(json.loads(manifest.read_text(encoding="utf-8"))["format_version"])
