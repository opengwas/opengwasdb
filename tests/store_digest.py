"""What a completed Store Release holds, as a digest a golden file can pin.

`store_digest` describes every file a release writes, by the rules the store
comparison benchmarks already apply: each Zarr array by its metadata
(`compare_store_releases.discover_arrays`) and a hash of its decoded values, so
the pin does not depend on the compressor's bytes; each group's attributes; and
every other file as `routing_fixture_trees.comparable` reads it -- the manifest
without its `created_at`, a gzip file decompressed, `index.sqlite` as its dump.
Two releases with equal digests hold the same data; a difference names the
file and array it is in.

Two values are read coarser than they are stored, because they are not the
release's data: a manifest names each source by its file name only (the
directory is the run's), and `completion_quality.pearson_r` is read as
`float32`. That correlation is the imputation kernel's raw `float64`, which
moves in its last bits with the host's BLAS kernels (`OPENBLAS_CORETYPE`)
while every stored array stays the same.

`matches_pin` compares a digest with `tests/golden/completion_stores.json`.
That file was written by the code at `1ab2b09` (issue #424), before
completion's Phase 3 streamed: running the identity tests with
`OPENGWASDB_WRITE_PINS` set to a directory writes each fixture's digest there
instead of comparing it, and the pins are those files, merged. A test compares
the new code with the old one, never the new code with itself. The two manifest
entries #381 changed are the one exception: each pin carries them as that code
writes them (`test_completion_identity`).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from benchmarks.compare_store_releases import discover_arrays
from benchmarks.routing_fixture_trees import comparable

GOLDEN = Path(__file__).parent / "golden" / "completion_stores.json"

#: One `completion_quality` row as `sqlite3.iterdump` writes it.
_QUALITY_ROW = re.compile(
    r'^(INSERT INTO "completion_quality" VALUES\(\d+,\'[^\']*\',)([^,]+)(,.*)$'
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _arrays(store: Path) -> dict[str, Any]:
    """Every array's metadata, less its bytes on disk, and its decoded values' hash."""
    found = discover_arrays(store)
    for rel, record in found.items():
        record.pop("nbytes_on_disk")
        values = np.ascontiguousarray(zarr.open_array(store / rel, mode="r")[...])
        record["values"] = _sha256(values.tobytes())
    return found


def _group_attrs(store: Path) -> dict[str, Any]:
    """Every Zarr group's attributes, by its path relative to `store`."""
    out: dict[str, Any] = {}
    for meta in sorted(store.rglob("zarr.json")):
        raw = json.loads(meta.read_text(encoding="utf-8"))
        if raw.get("node_type") == "group":
            out[meta.parent.relative_to(store).as_posix()] = raw.get("attributes", {})
    return out


def _pearson_at_float32(statement: str) -> str:
    """A dumped `completion_quality` row with its `pearson_r` read as `float32`."""
    row = _QUALITY_ROW.match(statement)
    if row is None or row.group(2) == "NULL":
        return statement
    return f"{row.group(1)}{float(np.float32(row.group(2)))!r}{row.group(3)}"


def _source_names(value: Any) -> Any:
    """A manifest with each absolute path reduced to its file name."""
    if isinstance(value, dict):
        return {k: _source_names(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_source_names(v) for v in value]
    if isinstance(value, str) and os.path.isabs(value):
        return Path(value).name
    return value


def _file(path: Path) -> Any:
    content = comparable(path, dump_sqlite=True)
    if path.name == "manifest.json":
        return _source_names(content)
    if path.suffix == ".sqlite":
        return _sha256("\n".join(map(_pearson_at_float32, content)).encode())
    return _sha256(content)


def store_digest(store: Path) -> dict[str, Any]:
    """The digest of every file under `store` (see the module docstring)."""
    files = sorted(
        p.relative_to(store).as_posix()
        for p in store.rglob("*")
        if p.is_file() and not any(part.endswith(".zarr") for part in p.relative_to(store).parts)
    )
    digest: dict[str, Any] = {"files": files, "zarr": _arrays(store), "attrs": _group_attrs(store)}
    digest.update({name: _file(store / name) for name in files})
    return digest


def matches_pin(name: str, store: Path, totals: dict[str, int]) -> None:
    """Assert `store`, and the totals its completion returned, match pin `name`."""
    actual = {"totals": totals, "store": store_digest(store)}
    out = os.environ.get("OPENGWASDB_WRITE_PINS")
    if out:
        import opengwasdb

        actual["written_by"] = str(Path(opengwasdb.__file__).parent)
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / f"{name}.json").write_text(json.dumps(actual, indent=1, sort_keys=True))
        return
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))[name]
    assert actual["totals"] == expected["totals"], name
    assert actual["store"].keys() == expected["store"].keys(), name
    for key, value in expected["store"].items():
        if isinstance(value, dict):
            for inner, pinned in value.items():
                assert actual["store"][key].get(inner) == pinned, (name, key, inner)
            assert actual["store"][key].keys() == value.keys(), (name, key)
        else:
            assert actual["store"][key] == value, (name, key)
