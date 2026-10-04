"""The seam's zarr-python 3 read configuration holds, and fork pools survive it (#244).

`opengwasdb.store.arrays` owns process-wide settings that zarr-python 3 would
otherwise leave against us:

* Blosc's internal threads.  ``import zarr`` sets
  ``numcodecs.blosc.use_threads = False`` for the whole process, which makes
  every chunk decode single-threaded (~4.5 ms against ~0.6 ms with Blosc's 8
  threads, for a ``[1000, 1000]`` int16 chunk on OGS-00009;
  ``benchmarks/zarr3_blosc_decode.py``).  The seam turns them back on.
* ``FusedCodecPipeline`` for every array the package opens, with one worker.
  With more than one, the pipeline keeps a module-level thread pool that zarr
  3.4's after-fork reset does not clear (zarr-developers/zarr-python#4478).  A
  forked build worker inherits it without its threads: a read there of more
  than one chunk, but of no more chunks than the idle permits the parent's pool
  left, queues work that nothing runs and never returns.  A single-chunk read
  never uses the pool, so the fork test reads several chunks, in the parent
  and in each worker.

A timeout in the fork test means ADR 0056's constraint was broken, not that the
test is slow; do not raise ``_TIMEOUT_S`` to make it pass.

Each case runs in a fresh interpreter.  The settings are process-global and
depend on import order, so asserting them inside the pytest process would pass
on whatever an earlier test happened to import.  A hung fork pool fails on the
subprocess timeout instead of stalling the suite.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

#: Long enough for a cold interpreter to import zarr and fork twice; a fork pool
#: that hangs never finishes, so the margin does not hide one.
_TIMEOUT_S = 120

_PRELUDE = """
import json
import sys
from pathlib import Path

import numpy as np

from opengwasdb.store import arrays
from opengwasdb.store.arrays import ArrayRole

TMP = Path(sys.argv[1])
DATA = np.arange(40 * 40, dtype=np.int16).reshape(40, 40)


def build() -> str:
    path = TMP / "data.zarr"
    root = arrays.open_group_for_write(path, "w")
    arrays.create_array(
        root, "z", ArrayRole.DENSE_STATISTIC_PLANE, data=DATA, hint=(10, 10)
    )
    return str(path)


def read_block(args: tuple[str, int]) -> int:
    path, start = args
    # 20 rows x 40 columns crosses eight 10 x 10 chunks: a multi-chunk read is
    # the one a fused pipeline with several workers hands to its thread pool.
    block = np.asarray(arrays.open_group(path)["z"][start : start + 20, :])
    assert np.array_equal(block, DATA[start : start + 20, :])
    return int(block.astype(np.int64).sum())
"""


def _run(tmp_path: Path, body: str) -> dict[str, object]:
    script = tmp_path / "probe.py"
    script.write_text(_PRELUDE + textwrap.dedent(body), encoding="utf-8")
    # A session of its own, so a timeout can kill the probe's forked workers
    # too; a hung worker would otherwise outlive the test.
    probe = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = probe.communicate(timeout=_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(probe.pid, signal.SIGKILL)
        probe.communicate()
        pytest.fail(f"probe did not finish within {_TIMEOUT_S}s: a fork-pool worker hung")
    assert probe.returncode == 0, stderr
    result: dict[str, object] = json.loads(stdout.strip().splitlines()[-1])
    return result


def test_blosc_threads_stay_on_after_zarr_reads_and_writes(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        """
        import importlib
        from concurrent.futures import ThreadPoolExecutor

        import numcodecs.blosc

        # The module that turns them off; importing it again must not undo the seam.
        importlib.import_module("zarr.codecs.blosc")
        path = build()
        assert np.array_equal(np.asarray(arrays.open_group(path)["z"][:]), DATA)
        # zarr 3 decodes on its own worker threads, never the main thread, so
        # numcodecs' adaptive default would also say no there.
        with ThreadPoolExecutor(1) as pool:
            on_worker = pool.submit(numcodecs.blosc._get_use_threads).result()
        print(json.dumps({"use_threads": numcodecs.blosc.use_threads, "on_worker": on_worker}))
        """,
    )
    assert result == {"use_threads": True, "on_worker": True}


def test_store_arrays_read_through_the_fused_pipeline(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        """
        path = build()
        array = arrays.open_group(path)["z"]
        print(json.dumps({"pipeline": type(array._async_array.codec_pipeline).__name__}))
        """,
    )
    assert result == {"pipeline": "FusedCodecPipeline"}


#: What the probe's `read_block` returns for rows 0-19 and 20-39 of its grid.
_BLOCK_SUMS = [sum(range(0, 20 * 40)), sum(range(20 * 40, 40 * 40))]


def test_fork_pools_finish_after_the_parent_read_through_the_pipeline(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        """
        from opengwasdb.build.ordered_pool import ordered_map
        from opengwasdb.completion.parallel import run_block_tasks

        path = build()
        # The parent reads every chunk first, so whatever the read path keeps
        # alive is live when the build forks -- as in a real build, which reads
        # its staged planes before each parallel phase.
        array = arrays.open_group(path)["z"]
        assert np.array_equal(np.asarray(array[:]), DATA)
        items = [(path, 0), (path, 20)] * 4
        sums = list(ordered_map(read_block, items, n_workers=2))
        run_block_tasks(items, 2, read_block)
        print(json.dumps({
            "pipeline": type(array._async_array.codec_pipeline).__name__,
            "chunks": array.nchunks,
            "sums": sums,
        }))
        """,
    )
    # Meaningful only if the parent read through the fused pipeline across
    # several chunks: that is the read that would leave a thread pool behind.
    assert result["pipeline"] == "FusedCodecPipeline"
    assert result["chunks"] == 16
    assert result["sums"] == _BLOCK_SUMS * 4
