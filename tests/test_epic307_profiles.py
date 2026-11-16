"""The language decision distinguishes native leaves from Python caller frames."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.summarise_epic307_profiles import native_summary, python_summary


def test_native_shares_are_weighted_by_samples_not_by_recordings(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("fit (solver.py:10);dot (libopenblas.so) 9\n")
    second.write_text(
        "fit (solver.py:10);check (solver.py:20) 1\n"
        "fit (solver.py:10);solve (_cd_fast.so) 2\n"
        "fit (solver.py:10);sum (numpy/_multiarray_umath.so) 3\n"
        "fit (solver.py:10);malloc (libc.so) 5\n"
    )

    report = native_summary([first, second])

    assert report["total_samples"] == 20
    assert report["samples"]["OpenBLAS"] == 9
    assert report["shares_percent"]["OpenBLAS"] == 45.0
    assert report["shares_percent"]["interpreted Python frame"] == 5.0
    assert report["samples"]["numpy C"] == 3
    assert report["samples"]["scikit-learn Cython"] == 2
    assert report["samples"]["other native"] == 5
    assert len(report["recordings"]) == 2
    assert all(len(row["sha256"]) == 64 for row in report["recordings"])


def test_python_summary_separates_parent_and_worker_without_calling_numpy_python(tmp_path):
    path = tmp_path / "snapshot.txt"
    path.write_text(
        'process 10:"driver";thread 10;write (build.py:1) 3\n'
        'process 10:"driver";process 11:"worker";dot (numpy/core.py:2) 7\n'
        'process 10:"driver";process 11:"worker";dot (numpy/core.py:3) 2\n'
    )

    report = python_summary(path)

    assert report["roles"]["parent"]["samples"] == 3
    worker = report["roles"]["worker"]
    assert worker["samples"] == 9
    assert worker["leaf_frames"] == {"dot (numpy/core.py)": 9}
    assert "interpreted" not in json.dumps(report)


def test_pass2_reader_percentage_in_docs_matches_the_weighted_recordings():
    repo = Path(__file__).resolve().parents[1]
    artifact_path = repo / "docs/benchmark-output/opengwasdb_epic307_profiles.json"
    artifact = json.loads(artifact_path.read_text())
    names = {"stream_associations", "swap_one", "_rows_of", "_validated_cells",
             "batched", "_row", "_listed", "_oriented"}
    reader = samples = snapshots = 0
    for row in artifact["python_snapshots"]:
        if row["step"] == "build" and "Pass 2:" in row["last_phase_line"]:
            worker = row["roles"]["worker"]
            snapshots += 1
            samples += worker["samples"]
            reader += sum(count for leaf, count in worker["leaf_frames"].items()
                          if leaf.split(" (")[0] in names)
    assert snapshots == 2 and reader > 0 and samples > reader
    assert (reader, samples) == (90160, 114751)
    percent = f"{100 * reader / samples:.2f}% pooled"
    adr = repo / "docs/adr/0081-no-second-implementation-language-after-epic-307.md"
    discussion = repo / "docs/benchmark-output/opengwasdb_epic307_after.md"
    assert percent in adr.read_text()
    assert percent in discussion.read_text()


def test_an_empty_profile_is_not_a_zero_cost_measurement(tmp_path):
    path = tmp_path / "failed.txt"
    path.write_text("Permission denied\n")
    with pytest.raises(ValueError, match="no samples"):
        native_summary([path])
    with pytest.raises(ValueError, match="no samples"):
        python_summary(path)
