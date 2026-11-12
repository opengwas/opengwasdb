"""Ragged Reference Completion's write, spread over worker processes (#321).

After the LD-block phase, Ragged completion assembles each Analysis's
completed rows -- its observed rows read from the source, its fills, its
missing block positions -- and the writer lays them end to end in Analysis
order. The Analyses are assembled in a fork pool, as Dense writes its bands
there, and the worker count must change nothing in the Store Release.

The fixture is a residual-SE source of four Analyses over one LD block, each
leaving different block positions unobserved, and a panel that adds positions
the source never held, so every Analysis gains fills. Each test checks that
before relying on it. "Identical" is `completion_releases.assert_same_release`'s.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from completion_releases import CompletionInputs, assert_same_release
from fork_order import CompletionOrder
from test_ragged_residual_completion import (
    _FREQS,
    _TRAIT_BP,
    N_SOURCE,
    _expected_se,
    _source_bp,
    _source_z,
    _write_filtered_ssf,
    write_eaf_carrying_panel,
)

import opengwasdb.layouts.ragged.complete as complete_module
from opengwasdb.completion.checkpoint import checkpoint_dir_for
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.layouts.ragged.complete import complete_ragged_store, resume_ragged_completion
from opengwasdb.layouts.ragged.zarr_csr import RAGGED_ZARR_PATH
from opengwasdb.model.manifest import StoreManifest

N_WORKERS = 3

#: Each Analysis's SE model offset, and the block positions it leaves unobserved.
ANALYSES = {
    "a": (0.0, frozenset()),
    "b": (0.2, frozenset({36, 37, 38, 39})),
    "c": (0.1, frozenset({5, 17, 30})),
    "d": (0.3, frozenset({1, 2, 3, 25})),
}
#: The source row of each Analysis whose SE is a full log-unit off its model,
#: which only an exact exception holds.
EXACT = 12


def _build_source(root: Path) -> Path:
    filtered = root / "filtered"
    filtered.mkdir(parents=True)
    lines = [
        "analysis_index\tanalysis_id\ttrait_id\ttrait_chr\ttrait_bp\tn\tfiltered_file\t"
        "source_genome_build"
    ]
    for index, (analysis_id, (offset, unobserved)) in enumerate(ANALYSES.items()):
        se = _expected_se(_FREQS, offset)
        se[EXACT] *= float(np.exp(1.0))
        rows = [
            (_source_bp(k), _source_z(k) * float(se[k]), float(se[k]), float(_FREQS[k]))
            for k in range(N_SOURCE)
            if k not in unobserved
        ]
        _write_filtered_ssf(filtered / f"{analysis_id}.tsv.gz", rows)
        lines.append(
            f"{index}\t{analysis_id}\tT{index}\t1\t{_TRAIT_BP}\t50000\t{analysis_id}.tsv.gz\t"
            "GRCh38"
        )
    (root / "manifest.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    store = root / "obs.opengwasdb"
    build_ragged_from_ssf(
        root / "manifest.tsv", filtered, store, store_id="parallel-rag", release_id="obs",
        allow_unverified_eaf=True, association_coverage="cis",
    )
    return store


def _complete(source: Path, panel: Path, dst: Path, n_workers: int) -> Path:
    complete_ragged_store(
        source, dst, panel, ancestry="EUR", cis_window_bp=1_000_000, min_cor=0.0,
        release_id="comp", n_workers=n_workers,
    )
    return dst


def _assert_fixture_is_meaningful(inputs: CompletionInputs) -> None:
    """Fills in at least three Analyses, a residual SE plane with exceptions."""
    assert StoreManifest.load(inputs.source).encoding.se.is_residual
    group = zarr.open_group(str(inputs.serial / RAGGED_ZARR_PATH), mode="r")
    offsets = np.asarray(group["offsets"][:])
    imputed = np.asarray(group["imputed"][:])
    filled = [
        ai for ai in range(len(offsets) - 1) if imputed[offsets[ai] : offsets[ai + 1]].any()
    ]
    assert len(filled) >= 3, filled
    assert int(group["se_exception_index"].shape[0]) >= 1


@pytest.fixture(scope="module")
def completion(tmp_path_factory: pytest.TempPathFactory) -> CompletionInputs:
    root = tmp_path_factory.mktemp("ragged-parallel-write")
    source = _build_source(root / "source")
    panel = write_eaf_carrying_panel(root / "ld_panel")
    serial = _complete(source, panel, root / "serial.opengwasdb", 1)
    inputs = CompletionInputs(source, panel, serial, _complete)
    _assert_fixture_is_meaningful(inputs)
    return inputs


# ── one worker against several, an Analysis finishing out of order ────────

#: Analyses 0 to 2: the first is held until the other two are assembled.
_ORDER = CompletionOrder(3)
#: The assembly's worker this module wraps; ``None`` where there is none.
_REAL_COMPLETE = getattr(complete_module, "_complete_analysis", None)


def _first_analysis_finishing_last(assembly: Any, item: tuple[int, Any]) -> Any:
    ai = item[0]
    if ai == 0:
        _ORDER.wait_for_the_rest()
    assert _REAL_COMPLETE is not None
    completed = _REAL_COMPLETE(assembly, item)
    if ai < 3:
        _ORDER.claim(ai)
    return completed


def test_a_release_completed_on_three_workers_is_the_one_completed_on_one(
    completion: CompletionInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first Analysis is held until two others are assembled, so a writer
    fed in the order the pool finished would put one Analysis's rows under
    another's offsets."""
    _ORDER.reset()
    monkeypatch.setattr(
        complete_module, "_complete_analysis", _first_analysis_finishing_last, raising=False
    )

    parallel = completion.complete(tmp_path / "parallel.opengwasdb", n_workers=N_WORKERS)

    finished = _ORDER.order()
    assert sorted(finished) == [0, 1, 2], "the Analyses were not assembled on a pool"
    assert finished[-1] == 0
    assert_same_release(completion.serial, parallel)


# ── an interrupted write ───────────────────────────────────────────────────

_FORK = multiprocessing.get_context("fork")
_ASSEMBLED = _FORK.Value("i", 0)
_CRASHED_IN_PID = _FORK.Value("i", 0)
_CRASHING_ANALYSIS = 2


def _analysis_crashing_after_two_others(assembly: Any, item: tuple[int, Any]) -> Any:
    if item[0] == _CRASHING_ANALYSIS:
        deadline = time.monotonic() + CompletionOrder.WAIT_SECONDS
        while _ASSEMBLED.value < 2:
            if time.monotonic() > deadline:
                raise AssertionError("no two Analyses were assembled beside the failing one")
            os.sched_yield()
        _CRASHED_IN_PID.value = os.getpid()
        raise RuntimeError("simulated crash in the assembly")
    assert _REAL_COMPLETE is not None
    completed = _REAL_COMPLETE(assembly, item)
    with _ASSEMBLED.get_lock():
        _ASSEMBLED.value += 1
    return completed


def test_a_write_interrupted_after_some_analyses_resumes_to_the_same_release(
    completion: CompletionInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker fails once other Analyses are assembled: the staged release
    goes, every LD block's checkpoint stays, and the resumed completion, which
    assembles and writes every Analysis again, is the release one worker writes."""
    dst = tmp_path / "resumed.opengwasdb"
    _ASSEMBLED.value = 0
    _CRASHED_IN_PID.value = 0
    with monkeypatch.context() as crash:
        crash.setattr(
            complete_module, "_complete_analysis", _analysis_crashing_after_two_others,
            raising=False,
        )
        with pytest.raises(RuntimeError, match="simulated crash in the assembly"):
            completion.complete(dst, n_workers=N_WORKERS)

    assert _ASSEMBLED.value >= 2
    assert _CRASHED_IN_PID.value not in (0, os.getpid()), "the Analysis ran in the parent"
    checkpoint_dir = checkpoint_dir_for(dst)
    assert list((checkpoint_dir / "blocks").glob("*.npz"))
    assert not dst.exists()
    assert list(dst.parent.glob(f".{dst.name}.tmp*")) == []

    resume_ragged_completion(checkpoint_dir, n_workers=N_WORKERS)

    assert not checkpoint_dir.exists()
    assert_same_release(completion.serial, dst)
