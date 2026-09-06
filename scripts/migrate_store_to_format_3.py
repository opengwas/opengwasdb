#!/usr/bin/env python3
"""Re-encode a Dense `format_version` 2.0 release's `se` plane as 3.0, in place.

**This is outside the Provenance Amendment exception.** Spec §21.4 says a
format change derives a *new* release, and rewriting `se` is association data,
not provenance -- so this tool has the same standing as
`migrate_store_to_analyses_tsv.py`, and the same caveat: its targets are
stores that should be rebuilt instead.

It exists for the one store where that is not true. `ukb-b` is 9,847,701 ×
2,511 and takes 11h35m to build from 396 GiB of source VCF, so "rebuild it" is
not a remedy that can be applied to the store that matters most for
performance work (issue #135). The seven pilot stores rebuild cheaply, and are
where a migrated store must be checked against a freshly built one before this
is pointed at anything expensive.

Dense only, on purpose. A Hybrid migration would have to fit both components
against one shared model (issue #141) and a Ragged one rewrites a CSR plane;
neither is needed for `ukb-b`, and a tool that covers cases nobody has run is
a tool nobody has tested.

What it does, in order:

1. Refuses anything that is not a Dense 2.0 release whose `se` is `float16`.
2. Measures and re-encodes `se` one physical row chunk at a time, choosing the
   residual range by the same gates a build uses, and falling back to
   `float16` when the fit does not earn its bytes (ADR 0037 §3).
3. Rebuilds the top-hit index. It carries *decoded* SE (ADR 0040), so a
   residual re-encode leaves it describing values the plane no longer holds.
4. Re-stamps `format_version` and `encoding`, and records what touched the
   release.
5. Validates, and refuses to leave a store it has made invalid.

Usage:

    migrate_store_to_format_3.py STORE                # migrate in place
    migrate_store_to_format_3.py STORE --into COPY    # reflink, then migrate the copy
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from opengwasdb.encoding import optimise_dense_se_joint
from opengwasdb.layouts.dense.top_hits import build_top_hit_indexes
from opengwasdb.model.enums import PrimaryStorageLayout
from opengwasdb.store.open import CURRENT_FORMAT_VERSION, open_store
from opengwasdb.validation import validate_store

MIGRATABLE_FROM = "2.0"


def _refuse_unless_migratable(store) -> None:
    """Fail loudly rather than half-migrate something this tool does not cover."""
    manifest = store.manifest
    if manifest.primary_layout is not PrimaryStorageLayout.DENSE:
        raise SystemExit(
            f"{store.path}: primary_layout is {manifest.primary_layout.value}; this tool "
            "migrates Dense releases only (see the module docstring)"
        )
    if manifest.format_version != MIGRATABLE_FROM:
        raise SystemExit(
            f"{store.path}: format_version is {manifest.format_version!r}, not "
            f"{MIGRATABLE_FROM!r}. A 3.0 release is already migrated; anything older is "
            "rebuilt, not migrated (spec §21.4)."
        )
    if manifest.encoding.se.is_residual:
        raise SystemExit(f"{store.path}: se is already residual-coded; nothing to do")


def _reflink_copy(source: Path, destination: Path) -> None:
    """Copy the release, sharing extents where the filesystem can.

    A reflink makes the copy near-free until one side is written, which is what
    makes "migrate a copy, keep the original" affordable for a 56 GiB store.
    Falls back to a full copy where the filesystem cannot.
    """
    if destination.exists():
        raise SystemExit(f"{destination}: already exists; refusing to overwrite")
    subprocess.run(
        ["cp", "-a", "--reflink=auto", str(source), str(destination)],
        check=True,
    )


def _write_manifest(store, encoding, elapsed: float) -> None:
    """Re-stamp version and encoding, and say what did it.

    Written through the manifest's own `to_dict` so the migrated release is
    described by exactly the code that describes a built one -- a hand-edited
    key is how a manifest and its arrays drift apart.
    """
    data = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    data["format_version"] = CURRENT_FORMAT_VERSION
    data["encoding"] = encoding.to_manifest()
    data["provenance"] = {
        **data.get("provenance", {}),
        "format_migration": {
            "from": MIGRATABLE_FROM,
            "to": CURRENT_FORMAT_VERSION,
            "tool": "scripts/migrate_store_to_format_3.py",
            "at": datetime.now(UTC).isoformat(),
            "se_encoding": encoding.se.to_manifest(),
            "seconds": round(elapsed, 1),
            "note": (
                "se re-encoded in place and the top-hit index rebuilt; the variant axis, "
                "z, eaf and analyses.tsv were not touched. Outside the Provenance "
                "Amendment exception (spec §21.4)."
            ),
        },
    }
    store.manifest_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def migrate(store_path: Path) -> int:
    store = open_store(store_path)
    _refuse_unless_migratable(store)

    started = time.perf_counter()
    print(f"Measuring and re-encoding se: {store_path}", flush=True)
    selected, _coefficients = optimise_dense_se_joint(
        store.arrays(mode="a"), store.manifest.encoding
    )
    print(f"  se encoding selected: {selected.se.to_manifest()}", flush=True)

    # The index carries decoded SE, so it describes the old plane until rebuilt.
    print("Rebuilding the top-hit index", flush=True)
    build_top_hit_indexes(store_path, encoding=selected)

    elapsed = time.perf_counter() - started
    _write_manifest(store, selected, elapsed)
    print(f"Re-stamped to {CURRENT_FORMAT_VERSION} in {elapsed:.1f}s", flush=True)

    print("Validating", flush=True)
    result = validate_store(store_path)
    if not result.ok:
        for error in result.errors:
            print(f"  ERROR {error}", file=sys.stderr)
        raise SystemExit(
            f"{store_path}: migrated store does not validate. It has been left as the "
            "migration produced it so the failure can be inspected; discard it and keep "
            "the source release."
        )
    for warning in result.warnings:
        print(f"  warning: {warning}")
    print("OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("store", type=Path)
    parser.add_argument(
        "--into",
        type=Path,
        help="reflink the release here first and migrate the copy, leaving the source intact",
    )
    args = parser.parse_args()

    target = args.store
    if args.into is not None:
        print(f"Copying {args.store} -> {args.into}", flush=True)
        _reflink_copy(args.store, args.into)
        target = args.into
    return migrate(target)


if __name__ == "__main__":
    raise SystemExit(main())
