#!/usr/bin/env python3
"""Inventory one Store Release, read-only: the apparent size of every component (ADR 0102, #463).

    inventory.py RELEASE OUT_JSON

Apparent size is the sum of `st_size` over regular files, with directories counted as 0, so the
figure depends neither on filesystem blocks nor on coreutils' `du` version. Every file under
`data.zarr` is attributed to its innermost Zarr array; `index.sqlite` is opened read-only and
immutable, its tables counted and sized with `dbstat`. A Hybrid's `dense/` is inventoried as a
nested release. Imports nothing from opengwasdb.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def walk_files(root: Path) -> Iterator[tuple[Path, int]]:
    stack = [root]
    while stack:
        with os.scandir(stack.pop()) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    stack.append(Path(e.path))
                elif e.is_file(follow_symlinks=False):
                    yield Path(e.path), e.stat(follow_symlinks=False).st_size


def zarr_inventory(zroot: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """Every array under a Zarr v3 root with its bytes and files, and the groups' metadata bytes."""
    arrays: dict[str, dict[str, Any]] = {}
    groups_meta = 0
    for d, dirs, files in os.walk(zroot):
        if ".zarray" in files or ".zgroup" in files:
            raise SystemExit(f"zarr v2 node at {d}: not handled")
        if "zarr.json" not in files:
            continue
        meta = json.loads((Path(d) / "zarr.json").read_text())
        if meta.get("node_type") != "array":
            groups_meta += (Path(d) / "zarr.json").stat().st_size
            continue
        dirs[:] = []  # everything below an array is that array's
        sizes = [size for _, size in walk_files(Path(d))]
        arrays[os.path.relpath(d, zroot)] = {
            "bytes": sum(sizes),
            "files": len(sizes),
            "shape": meta.get("shape"),
            "dtype": meta.get("data_type"),
        }
    return arrays, groups_meta


def sqlite_inventory(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"bytes": path.stat().st_size, "tables": {}, "metadata": {}}
    if out["bytes"] == 0:
        out["empty_file"] = True
        return out
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        names = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
        dbstat: dict[str, int] = {}
        try:
            for name, size in con.execute("select name, sum(pgsize) from dbstat group by name"):
                dbstat[name] = size
        except sqlite3.Error as exc:
            out["dbstat_error"] = str(exc)
        for n in names:
            rows = con.execute(f'select count(*) from "{n}"').fetchone()[0]
            out["tables"][n] = {"rows": rows, "bytes": dbstat.get(n)}
        out["indexes"] = {
            r[0]: dbstat.get(r[0])
            for r in con.execute("select name from sqlite_master where type='index'")
        }
        if "metadata" in names:
            for key, value in con.execute("select key, value from metadata"):
                out["metadata"][key] = len(value) if isinstance(value, str | bytes) else value
    finally:
        con.close()
    return out


def release_inventory(rel: Path) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    manifest = json.loads((rel / "manifest.json").read_text())
    for e in sorted(os.scandir(rel), key=lambda e: e.name):
        p = Path(e.path)
        if e.is_dir(follow_symlinks=False):
            if (p / "zarr.json").exists():
                arrays, gmeta = zarr_inventory(p)
                entries[e.name] = {
                    "kind": "zarr",
                    "bytes": sum(a["bytes"] for a in arrays.values()) + gmeta,
                    "group_meta_bytes": gmeta,
                    "arrays": arrays,
                }
            elif e.name == "dense":
                nested = release_inventory(p)
                nested["bytes"] = sum(v["bytes"] for v in nested["entries"].values())
                entries[e.name] = {"kind": "nested_release", **nested}
            else:
                sizes = [size for _, size in walk_files(p)]
                entries[e.name] = {"kind": "dir", "bytes": sum(sizes), "files": len(sizes)}
        else:
            entry: dict[str, Any] = {"kind": "file", "bytes": e.stat(follow_symlinks=False).st_size}
            if e.name == "index.sqlite":
                entry["sqlite"] = sqlite_inventory(p)
            entries[e.name] = entry
    return {
        "format_version": manifest.get("format_version"),
        "primary_layout": manifest.get("primary_layout"),
        "completion": manifest.get("completion_state"),
        "manifest_keys": sorted(manifest),
        "entries": entries,
    }


def main() -> None:
    rel, out = Path(sys.argv[1]), Path(sys.argv[2])
    t0 = time.monotonic()
    inv = release_inventory(rel)
    inv["path"] = str(rel)
    inv["total_bytes"] = sum(v["bytes"] for v in inv["entries"].values())
    inv["walk_seconds"] = round(time.monotonic() - t0, 1)
    out.write_text(json.dumps(inv, indent=1, sort_keys=True))
    print(rel, inv["total_bytes"], f"{inv['walk_seconds']} s")


if __name__ == "__main__":
    main()
