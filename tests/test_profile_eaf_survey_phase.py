"""The EAF survey phase profile records what it measured, and only that (#310)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from eaf_fixtures import dense_manifest, frequencies, write_ssf

from benchmarks.profile_eaf_survey_phase import _phases, _split_one_column, _subset_manifest, main
from opengwasdb.build.eaf_orientation import site_hashes


def _manifest(tmp_path: Path, n: int) -> Path:
    freqs = frequencies(seed=3)
    sources = {
        f"A{i}": write_ssf(tmp_path / f"a{i}.tsv.gz", freqs, flipped=False, seed=i + 1)
        for i in range(n)
    }
    return dense_manifest(tmp_path / "manifest.tsv", sources)


def _run(tmp_path: Path, manifest: Path, stride: int, work: str = "work") -> dict:
    output = tmp_path / "artifact.json"
    main(
        [
            str(manifest),
            "--stride", str(stride),
            "--work", str(tmp_path / work),
            "--n-workers", "2",
            "--source-reader-capability", "opengwasdb.gwas-ssf",
            "--output", str(output),
        ]
    )
    return json.loads(output.read_text())


def test_the_profile_stops_after_the_phase_and_records_each_part(tmp_path) -> None:
    artifact = _run(tmp_path, _manifest(tmp_path, 4), stride=2)

    assert artifact["n_analyses_in_manifest"] == 4
    assert artifact["n_analyses_profiled"] == 2
    assert artifact["scale_to_manifest"] == 2.0
    assert set(artifact["phases"]) >= {
        "EAF site hashes",
        "EAF survey",
        "EAF orientation verification",
        "EAF survey and orientation verification",
    }
    assert isinstance(artifact["worktree_clean"], bool)
    assert artifact["nproc"] >= 1
    # Nothing after the survey ran: the build never reached the encoding plan.
    assert not (tmp_path / "work" / "store.opengwasdb").exists()
    assert len(artifact["column_split"]) == 2


def test_the_profile_refuses_an_existing_work_directory(tmp_path) -> None:
    manifest = _manifest(tmp_path, 2)
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "keep.txt").write_text("mine")

    with pytest.raises(SystemExit, match="already exists"):
        _run(tmp_path, manifest, stride=1)

    assert (tmp_path / "work" / "keep.txt").read_text() == "mine"


def test_subset_manifest_takes_every_nth_row_and_renumbers(tmp_path) -> None:
    source = tmp_path / "in.tsv"
    source.write_text(
        "analysis_index\tanalysis_id\n" + "".join(f"{i}\tX{i}\n" for i in range(7)),
        encoding="utf-8",
    )
    destination = tmp_path / "out.tsv"

    assert _subset_manifest(source, destination, 3) == 3
    assert destination.read_text() == "analysis_index\tanalysis_id\n0\tX0\n1\tX3\n2\tX6\n"


def test_phases_reads_start_and_elapsed_lines() -> None:
    lines = [
        {"t": 1.0, "message": "EAF survey: start"},
        {"t": 2.0, "message": "EAF survey: done in 12s"},
        {"t": 2.0, "message": "EAF survey accumulation: done in 3s across 252 Analyses"},
        {"t": 2.5, "message": "EAF survey: 25/252 — elapsed 4s, ETA 41s"},
    ]
    assert _phases(lines) == {
        "EAF survey": {"start": 1.0, "done_in": "12s"},
        "EAF survey accumulation": {"done_in": "3s"},
    }


def test_the_column_split_checks_itself_against_the_sampler(tmp_path) -> None:
    rows = np.arange(100, dtype=np.int64)
    eaf = np.linspace(0.1, 0.9, 100)
    np.savez(tmp_path / "0.npz", rows=rows, z=np.zeros(100), se=np.ones(100), eaf=eaf)
    hashes = site_hashes([f"1:{i}:A:G" for i in range(100)])

    split = _split_one_column(tmp_path, 0, hashes)

    assert split["n_rows"] == 100
    assert split["n_eaf_cells"] == 100
