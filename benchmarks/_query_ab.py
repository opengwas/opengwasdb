"""Shared mechanics of the interleaved A/B runners (#253, #246).

`eaf_read_once_ab.py` alternates two `opengwasdb` code revisions;
`top_hit_shard_ab.py` alternates two Store Releases that differ in one physical
shape. Both need the same three things, and this module holds one copy of each
so the two artifacts are comparable about what they measured:

* `digest` -- a per-array sha256 of a query result, so an A/B can assert the
  two sides answered identically before it reports a ratio;
* `time_shapes` -- the child's whole job: open one store, build the shared #242
  query shapes, warm each once, time `reps` samples and digest the result;
* `interleave` -- the parent's round loop: run the sides `A, B` then `B, A`,
  keep every sample, and record the child records round by round, so machine
  drift cancels instead of deciding the answer.

A child imports `time_shapes` only after its own `sys.path` is set, which is why
this module imports `opengwasdb` lazily.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def digest(result: dict[str, np.ndarray]) -> dict[str, str]:
    """sha256 per returned array (dtype, shape, values; NaN positions kept).

    NaN is canonicalised as a position: two stores may carry a different NaN
    payload for the same missing cell and still mean the same thing.
    """
    out: dict[str, str] = {}
    for key, values in sorted(result.items()):
        values = np.asarray(values)
        hashed = hashlib.sha256()
        hashed.update(str(values.dtype).encode())
        hashed.update(str(values.shape).encode())
        if values.dtype.kind == "f":
            missing = np.isnan(values)
            hashed.update(b"nan\x1f")
            hashed.update(np.packbits(missing).tobytes())
            hashed.update(values[~missing].tobytes())
        elif values.dtype.kind in "OUSV":
            hashed.update(repr(values.tolist()).encode())
        else:
            hashed.update(np.ascontiguousarray(values).tobytes())
        out[key] = hashed.hexdigest()
    return out


def time_shapes(
    store: str | Path, selection: dict[str, Any], shapes: list[str], reps: int
) -> dict[str, Any]:
    """Time each of `shapes` on `store` in this interpreter, with a digest each."""
    from benchmarks import _query_shapes
    from opengwasdb.query import query_store

    region = (
        selection["region"]["chrom"],
        int(selection["region"]["start"]),
        int(selection["region"]["end"]),
    )
    measured: dict[str, Any] = {}
    with query_store(store) as query:
        patterns = _query_shapes.common_query_patterns(
            query,
            exposure=selection["exposure_analysis_id"],
            phewas_alid=selection["phewas_alid"],
            region=region,
            random_alids=selection["random_alids"],
            random_analyses=selection["random_analyses"],
        )
        for name in shapes:
            fn = patterns[name]
            warm = fn()
            shape_digest = digest(warm)
            n_rows = len(warm["z"])
            samples: list[float] = []
            for _ in range(reps):
                t0 = time.perf_counter()
                result = fn()
                samples.append(round((time.perf_counter() - t0) * 1000, 4))
                if len(result["z"]) != n_rows:
                    raise SystemExit(
                        f"{name}: returned {len(result['z'])} rows after {n_rows}; "
                        "a shape whose size changes cannot be timed"
                    )
            measured[name] = {
                "samples_ms": samples,
                "digest": shape_digest,
                "n_rows": n_rows,
            }
    return measured


def run_child(script: Path, argv: list[str], marker: str) -> dict[str, Any]:
    """Run one A/B child and return the single `<marker> <json>` line it printed."""
    proc = subprocess.run([sys.executable, str(script), *argv], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"A/B child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    lines = [line for line in proc.stdout.splitlines() if line.startswith(f"{marker} ")]
    if len(lines) != 1:
        raise SystemExit(f"A/B child printed {len(lines)} result lines:\n{proc.stdout[-4000:]}")
    return json.loads(lines[0][len(marker) + 1 :])


#: The shapes the interleave returns: samples in ms, digests per round, records.
Samples = dict[str, dict[str, list[float]]]
Digests = dict[str, dict[str, dict[str, str]]]
Rounds = list[dict[str, Any]]


def interleave(
    sides: list[str],
    run_side: Callable[[str, int], dict[str, Any]],
    rounds: int,
) -> tuple[Samples, Digests, Rounds]:
    """Run the sides round by round, alternating the order, keeping every sample.

    Returns the samples and per-round digests by side, and the child records for
    the artifact. The first side is the reference: round 0 runs it first, round
    1 runs it last, so neither side is systematically measured on the warmer or
    quieter half of the window.
    """
    samples: dict[str, dict[str, list[float]]] = {side: {} for side in sides}
    digests: dict[str, dict[str, dict[str, str]]] = {side: {} for side in sides}
    records: list[dict[str, Any]] = []
    for index in range(rounds):
        order = list(sides) if index % 2 == 0 else list(reversed(sides))
        record: dict[str, Any] = {"round": index, "order": order, "results": {}}
        for side in order:
            child = run_side(side, index)
            record["results"][side] = child
            for name, rec in child["shapes"].items():
                samples[side].setdefault(name, []).extend(rec["samples_ms"])
                digests[side].setdefault(name, {})[str(index)] = rec["digest"]
            print(
                f"round {index} {side}: "
                f"{json.dumps({n: rec['n_rows'] for n, rec in child['shapes'].items()})}",
                flush=True,
            )
        records.append(record)
    return samples, digests, records


def differing_shapes(
    sides: list[str], digests: dict[str, dict[str, dict[str, str]]]
) -> list[str]:
    """Shapes whose digests are not equal between the first two sides."""
    reference, other = sides[0], sides[1]
    return sorted(
        name for name in digests[reference] if digests[reference][name] != digests[other][name]
    )


def medians(samples: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
    return {
        side: {name: round(float(np.median(values)), 3) for name, values in shapes.items()}
        for side, shapes in samples.items()
    }


def child_argv(
    store: str | Path,
    selection: str | Path,
    shapes: list[str],
    reps: int,
    *,
    module_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> list[str]:
    """The argv one A/B child is re-invoked with, minus the interpreter and script.

    `module_path`/`repo_root` are the code-comparison A/B's (#253) extra flags;
    the store-comparison A/B (#246) passes neither.
    """
    argv = ["--child"]
    if module_path is not None:
        argv += ["--module-path", str(module_path), "--repo-root", str(repo_root)]
    return argv + [
        "--store",
        str(store),
        "--selection",
        str(selection),
        "--shapes",
        *shapes,
        "--reps",
        str(reps),
    ]


def identity_block(differing: list[str]) -> dict[str, Any]:
    """The artifact's identity record, one shape of block for both A/B runners."""
    return {
        "identical": not differing,
        "differing_shapes": differing,
        "note": "sha256 per returned array, compared between the two sides every round",
    }


def environment_block() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "hostname": __import__("socket").gethostname(),
    }


def round_block(
    samples: dict[str, dict[str, list[float]]], rounds: list[dict[str, Any]]
) -> dict[str, Any]:
    """The artifact's round record: the order, every child record, every sample."""
    return {
        "round_order": [record["order"] for record in rounds],
        "rounds": rounds,
        "samples_ms": {
            side: {name: values for name, values in shapes.items()}
            for side, shapes in samples.items()
        },
    }


def measurement_block(
    selection: dict[str, Any],
    shapes: list[str],
    reps: int,
    rounds_requested: int,
    samples: Samples,
    rounds: Rounds,
) -> dict[str, Any]:
    """The artifact's request and round record, shared by both A/B runners."""
    return {
        "selection": selection,
        "shapes": list(shapes),
        "reps": reps,
        "rounds_requested": rounds_requested,
        **round_block(samples, rounds),
    }


def identity_verdict(differing: list[str]) -> int:
    """Print and return the process exit for an A/B that compared its sides."""
    if differing:
        print(f"IDENTITY FAILED for {differing}", flush=True)
        return 1
    return 0
