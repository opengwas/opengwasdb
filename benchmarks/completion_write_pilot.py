"""Time a completion's write on a pilot, base against head, from one set of LD blocks (#321).

The LD-block phase is the same code at both revisions and most of a pilot's
time, so it runs once: `blocks` stops a completion when its blocks are
checkpointed. `resume` copies that checkpoint directory beside a new
destination, points its `build_params.json` there, and resumes it with
whichever `opengwasdb` PYTHONPATH resolves (its path is recorded), timing each
phase after the blocks and sampling the destination's checkpoint and staged
directories every second for the peak scratch disk.

  completion_write_pilot.py blocks dense|ragged SRC DEST LD N_WORKERS [SUBSET]
  completion_write_pilot.py resume TEMPLATE_CHECKPOINT DEST N_WORKERS OUT_JSON

`SUBSET` completes a Ragged Store's first N Analyses and carries the rest
Observed-Only, as #424's benchmark does. The OGS-00003 and OGS-00006 records in
`docs/benchmark-output/reference_completion_321/` were made with this script.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import sys
import threading
import time
from pathlib import Path

import opengwasdb


class _Stop(Exception):
    pass


def _store_id(src: Path) -> str:
    return str(json.loads((src / "manifest.json").read_text())["store_id"])


def _layout(params: dict) -> str:
    return "ragged" if "cis_window_bp" in params else "dense"


def blocks(layout: str, src: Path, dst: Path, ld: Path, n_workers: int, subset: int) -> None:
    if layout == "dense":
        from opengwasdb.layouts.dense import complete as module

        def stop(*_a, **_k):
            raise _Stop

        module._merge_checkpoint_fills = stop
        try:
            module.complete_dense_store(
                src, dst, ld, ancestry="EUR", release_id=f"{_store_id(src)}-completed",
                n_workers=n_workers, overwrite=True,
            )
        except _Stop:
            print("stopped after the LD-block phase", flush=True)
        return
    from opengwasdb.completion import fill_shards
    from opengwasdb.layouts.ragged import complete as module
    from opengwasdb.model.analyses import read_analysis_records

    def stop(*_a, **_k):
        raise _Stop

    fill_shards.shard_checkpoint_fills = stop
    analyses = sorted(
        read_analysis_records(src / "analyses.tsv"), key=lambda a: int(a.analysis_index)
    )
    chosen = {a.analysis_id for a in analyses[:subset]} if subset > 0 else None
    try:
        module.complete_ragged_store(
            src, dst, ld, ancestry="EUR", n_workers=n_workers, overwrite=True,
            impute_analysis_ids=chosen,
        )
    except _Stop:
        print("stopped after the LD-block phase", flush=True)


def _tree_bytes(path: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except FileNotFoundError:
                pass
    return total


class ScratchPeak:
    """Apparent bytes of DEST's checkpoint and staged directories, sampled every
    `period` seconds on a thread: the peak scratch a completion holds."""

    def __init__(self, dst: Path, period: float = 1.0) -> None:
        self.dst, self.period = dst, period
        self.peak = {"total": 0, "checkpoint": 0, "staged": 0, "fill_shards": 0}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        checkpoint = _tree_bytes(self.dst.parent / f".{self.dst.name}.checkpoint")
        staged_dirs = [p for p in self.dst.parent.glob(f".{self.dst.name}.tmp*")]
        staged = sum(_tree_bytes(p) for p in staged_dirs)
        fills = sum(_tree_bytes(p / "fill_shards") for p in staged_dirs)
        for key, value in (("checkpoint", checkpoint), ("staged", staged),
                           ("fill_shards", fills), ("total", checkpoint + staged)):
            self.peak[key] = max(self.peak[key], value)

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            self._sample()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


#: The phases whose pool workers' private memory is sampled, every 0.2 s.
_POOLED = (
    "_merge_checkpoint_fills", "_write_completed_bands", "shard_checkpoint_fills",
    "_assemble_completed_csr", "flush_regions",
)


def _timed(owner, name: str, timings: dict, memory: dict | None = None) -> None:
    """Charge `owner.name`'s wall time to `timings`, and for a pooled phase its
    workers' peak private memory to `memory`."""
    from benchmarks.profile_reference_completion import WorkerMemory

    original = getattr(owner, name)

    def wrapped(*args, **kwargs):
        started = time.monotonic()
        sampler = WorkerMemory(0.2) if memory is not None and name in _POOLED else None
        try:
            if sampler is None:
                return original(*args, **kwargs)
            with sampler:
                return original(*args, **kwargs)
        finally:
            timings[name] = timings.get(name, 0.0) + time.monotonic() - started
            if sampler is not None:
                memory[name] = sampler.summary()

    setattr(owner, name, wrapped)


def resume(template: Path, dst: Path, n_workers: int, out: Path) -> None:
    from opengwasdb.completion import fill_shards
    from opengwasdb.completion.checkpoint import checkpoint_dir_for

    checkpoint_dir = checkpoint_dir_for(dst)
    if checkpoint_dir.exists() or dst.exists():
        raise SystemExit(f"{dst} or its checkpoint exists")
    shutil.copytree(template, checkpoint_dir)
    params_path = checkpoint_dir / "build_params.json"
    params = json.loads(params_path.read_text())
    params["dest_path"] = str(dst.resolve())
    params_path.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")
    timings: dict[str, float] = {}
    memory: dict[str, dict] = {}
    layout = _layout(params)
    if layout == "dense":
        from opengwasdb.layouts.dense import complete as module

        for name in ("_build_union_axis", "_complete_pending_blocks", "_merge_checkpoint_fills",
                     "_write_completed_bands", "_write_completed_manifest",
                     "_completed_analysis_rows", "_finalise_release"):
            _timed(module, name, timings, memory)
        resume_fn = module.resume_dense_completion
    else:
        from opengwasdb.layouts.ragged import complete as module

        for name in ("_read_source_state", "_plan_blocks", "_merged_axis", "_encode_plan",
                     "_run_pending_blocks", "_analysis_batches", "_assemble_completed_csr",
                     "_write_completed_zarr", "_write_top_hits_and_analyses"):
            _timed(module, name, timings, memory)
        _timed(fill_shards, "shard_checkpoint_fills", timings, memory)
        _timed(module, "build_variant_index", timings)
        from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRWriter

        for name in ("write_eaf_plane", "_write_n_plane", "flush_se", "flush_regions"):
            if hasattr(RaggedCSRWriter, name):
                _timed(RaggedCSRWriter, name, timings, memory)
        resume_fn = module.resume_ragged_completion
    started = time.monotonic()
    with ScratchPeak(dst) as scratch:
        result = resume_fn(checkpoint_dir, n_workers=n_workers)
    wall = time.monotonic() - started
    record = {
        "package": opengwasdb.__file__,
        "layout": layout,
        "template": str(template),
        "dest": str(dst),
        "n_workers": n_workers,
        "wall_seconds": wall,
        "phase_seconds": timings,
        "pool_worker_memory": memory,
        "peak_scratch_bytes": scratch.peak,
        "release_bytes": _tree_bytes(dst),
        "uptime": os.getloadavg(),
        "ru_maxrss_kib_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "ru_maxrss_kib_children": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "result": {k: (int(v) if isinstance(v, (int, float)) else str(v))
                   for k, v in vars(result).items() if not k.startswith("_")},
    }
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


def main() -> None:
    print("opengwasdb from", opengwasdb.__file__, flush=True)
    command = sys.argv[1]
    if command == "blocks":
        layout, src, dst, ld = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
        subset = int(sys.argv[7]) if len(sys.argv) > 7 else 0
        blocks(layout, src, dst, ld, int(sys.argv[6]), subset)
    elif command == "resume":
        resume(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5]))
    else:
        raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main()
