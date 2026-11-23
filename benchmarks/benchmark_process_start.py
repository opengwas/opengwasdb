"""What a fresh process pays before its first answer, on a 0.2.x Store Release (#465).

The store-comparison harness opens a release once and times queries on it, so it
cannot see what #465 changed: the handles and imports a *process* pays once. Each
measurement here is a fresh interpreter, repeated, and reported as median and range:

- ``imports``: ``import pysam, sqlite3``;
- ``tabix_open``: ``pysam.TabixFile`` on the release's Store Variant Table;
- ``index_open``: ``sqlite3.connect`` on its ``index.sqlite``;
- ``open_and_lookup``: ``query_store`` then one ``lookup`` (``--alids`` by every
  Analysis the selection names), wall time of the whole process, for the source tree
  named by ``--tree`` (a ``git archive`` of the commit to compare, or the worktree).
- ``phewas``: ``query_store`` then one ``phewas`` by ``--phewas-alid``; the time of
  the ``phewas`` call itself, in the process.

    benchmark_process_start.py --store S --selection sel.json --tree T --runs 7 --output out.json

``--selection`` is a store-comparison artifact (its ``selection`` block) or a bare
``{random_alids, random_analyses}`` file.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

IMPORTS = (
    "import time; t=time.perf_counter(); import pysam, sqlite3; "
    "print((time.perf_counter()-t)*1e3)"
)
TABIX = (
    "import sys, time, pysam; t=time.perf_counter(); h=pysam.TabixFile(sys.argv[1]); "
    "print((time.perf_counter()-t)*1e3)"
)
INDEX = (
    "import sys, time, sqlite3; t=time.perf_counter(); c=sqlite3.connect(sys.argv[1]); "
    "print((time.perf_counter()-t)*1e3)"
)
LOOKUP = """
import json, sys
from opengwasdb.query import query_store
path, alids, analyses = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
with query_store(path) as q:
    n = len(q.lookup(alids, analyses)["variant_index"])
assert n > 0
"""
PHEWAS = """
import sys, time
from opengwasdb.query import query_store
with query_store(sys.argv[1]) as q:
    t = time.perf_counter()
    n = len(q.phewas(sys.argv[2])["variant_index"])
    print((time.perf_counter() - t) * 1e3)
assert n > 0
"""


def run(code: str, *args: str, tree: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if tree is not None:
        env["PYTHONPATH"] = str(tree)
    # A neutral cwd: from a worktree, `opengwasdb` would resolve to that worktree.
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd="/", env=env, capture_output=True, text=True, check=True,
    )


def summary(values: list[float]) -> dict[str, object]:
    return {
        "median_ms": round(statistics.median(values), 1),
        "min_ms": round(min(values), 1),
        "max_ms": round(max(values), 1),
        "runs_ms": [round(v, 1) for v in values],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--phewas-alid")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--n-alids", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    selection = selection.get("selection", selection)
    alids = selection["random_alids"][: args.n_alids]
    analyses = selection["random_analyses"]
    probe = run("import opengwasdb; print(opengwasdb.__file__)", tree=args.tree).stdout.strip()
    assert probe.startswith(str(args.tree.resolve())), probe

    table, index = str(args.store / "variants.tsv.gz"), str(args.store / "index.sqlite")
    out: dict[str, object] = {"tree": probe, "store": str(args.store), "runs": args.runs}
    out["imports"] = summary([float(run(IMPORTS).stdout) for _ in range(args.runs)])
    out["tabix_open"] = summary([float(run(TABIX, table).stdout) for _ in range(args.runs)])
    out["index_open"] = summary([float(run(INDEX, index).stdout) for _ in range(args.runs)])
    walls = []
    for _ in range(args.runs):
        start = time.perf_counter()
        run(LOOKUP, str(args.store), json.dumps(alids), json.dumps(analyses), tree=args.tree)
        walls.append((time.perf_counter() - start) * 1e3)
    out["open_and_lookup"] = summary(walls)
    if args.phewas_alid:
        out["phewas"] = summary(
            [
                float(run(PHEWAS, str(args.store), args.phewas_alid, tree=args.tree).stdout)
                for _ in range(args.runs)
            ]
        )
    out["load_average_1m"] = os.getloadavg()[0]
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v["median_ms"] if isinstance(v, dict) else v for k, v in out.items()}))


if __name__ == "__main__":
    main()
