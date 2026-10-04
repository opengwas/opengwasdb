"""The seam's zarr-python 3 read configuration holds (#244).

`opengwasdb.store.arrays` owns process-wide settings that zarr-python 3 would
otherwise leave against us.  The one checked here is Blosc's internal threads:
``import zarr`` sets ``numcodecs.blosc.use_threads = False`` for the whole
process, which makes every chunk decode single-threaded (~4.5 ms against
~1.1 ms for a ``[1000, 1000]`` int16 chunk on OGS-00009).  The seam turns them
back on.

Each case runs in a fresh interpreter.  The settings are process-global and
depend on import order, so asserting them inside the pytest process would pass
on whatever an earlier test happened to import.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

#: Long enough for a cold interpreter to import zarr and build a store.
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
"""


def _run(tmp_path: Path, body: str) -> dict[str, object]:
    script = tmp_path / "probe.py"
    script.write_text(_PRELUDE + textwrap.dedent(body), encoding="utf-8")
    try:
        done = subprocess.run(
            [sys.executable, str(script), str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"probe did not finish within {_TIMEOUT_S}s")
    assert done.returncode == 0, done.stderr
    result: dict[str, object] = json.loads(done.stdout.strip().splitlines()[-1])
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
