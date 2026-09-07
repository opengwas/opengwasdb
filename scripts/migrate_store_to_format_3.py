#!/usr/bin/env python3
"""Derive a Dense `format_version` 3.0 release from a 2.0 one, re-encoding its
`se` plane as the format-3 conditional residual coding.

**This is outside the Provenance Amendment exception, and the source release
is never written.** Spec §21.4 says a format change derives a *new* release,
and rewriting `se` is association data, not provenance. A migration of
`ukb-b` would have the same standing as `migrate_store_to_analyses_tsv.py`'s
in-place rewrite if it mutated the release it was given; it does not. The
destination is built in a `.name.tmp` staging directory beside the target and
published by rename only when the migrated copy validates, so a failure at any
point leaves the source untouched and nothing where the destination was meant
to appear (issue #156). The source release is opened read-only for the
before-migration validation that distinguishes inherited faults from
introduced ones.

It exists for the one store where a rebuild is not a remedy. `ukb-b` is
9,847,701 × 2,511 and takes 11h35m to build from 396 GiB of source VCF, so
"rebuild it" cannot be applied to the store that matters most for performance
work (issue #135). The seven pilot stores rebuild cheaply, and are where a
migrated store must be checked against a freshly built one before this is
pointed at anything expensive.

Dense only, on purpose. A Hybrid migration would have to fit both components
against one shared model (issue #141) and a Ragged one rewrites a CSR plane;
neither is needed for `ukb-b`, and a tool that covers cases nobody has run is
a tool nobody has tested.

What it does, in order:

1. Refuses anything that is not a Dense 2.0 release whose `se` is `float16` --
   against the *source*, before any bytes are copied.
2. Measures and re-encodes `se` one physical row chunk at a time, choosing the
   residual range by the same gates a build uses, and falling back to
   `float16` when the fit does not earn its bytes (ADR 0037 §3).
3. Rebuilds the top-hit index. It carries *decoded* SE (ADR 0040), so a
   residual re-encode leaves it describing values the plane no longer holds.
4. Re-stamps `format_version` and `encoding`, and records what touched the
   release.
5. Validates the staged copy, and refuses to publish one it has made invalid.

Usage:

    migrate_store_to_format_3.py STORE --into DEST

`--into` is required: the source release is immutable and is never written.
The new release appears at `DEST` only when the migrated copy validates; it
must not already exist.
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
from opengwasdb.encoding.timing import PhaseTimer
from opengwasdb.layouts.dense.top_hits import build_top_hit_indexes
from opengwasdb.model.enums import PrimaryStorageLayout
from opengwasdb.store.open import CURRENT_FORMAT_VERSION, OpenGWASDBStore, open_store
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
    """Copy the release's contents into ``destination``, sharing extents where
    the filesystem can.

    ``destination`` is an existing directory (the staging directory the new
    release is built in), so the copy fills it rather than nesting a
    subdirectory. A reflink makes the copy near-free until one side is
    written, which is what makes "migrate a copy, keep the original"
    affordable for a 56 GiB store. Falls back to a full copy where the
    filesystem cannot.
    """
    subprocess.run(
        ["cp", "-a", "--reflink=auto", f"{source}/.", f"{destination}/"],
        check=True,
    )


def _write_manifest(store, encoding, elapsed: float, timer: PhaseTimer) -> None:
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
            # Per-phase, not just the total: the same migration takes 2,913 s on
            # a release carrying issue #135's unchunked `eaf_baseline` and 244 s
            # on a repaired one, and only the breakdown says which you have.
            "phase_seconds": {name: round(seconds, 1) for name, seconds, _ in timer.report()},
            "note": (
                "se re-encoded and the top-hit index rebuilt in a new release; the source "
                "release was not modified. The variant axis, z, eaf and analyses.tsv were "
                "not touched. Outside the Provenance Amendment exception (spec §21.4)."
            ),
        },
    }
    store.manifest_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _inherited_errors(store_path: Path) -> set[str]:
    """What was already wrong before this tool touched anything.

    Every 2.0 pilot carries some: the FinnGen rebuild fails on #127's truncated
    ALID index and #135's unchunked `eaf_baseline`, neither of which a `se`
    re-encode can fix or is answerable for. Judging the result against "no
    errors at all" would blame the migration for its input, and would reject
    every store it exists to serve. Measured against the *source*, which the
    staged copy is a byte-for-byte copy of.
    """
    print("Validating before, to tell inherited faults from introduced ones", flush=True)
    inherited = set(validate_store(store_path).errors)
    if inherited:
        print(f"  {len(inherited)} pre-existing error(s), carried through:", flush=True)
        for error in sorted(inherited):
            print(f"    - {error}", flush=True)
    return inherited


def _report_outcome(staged_path: Path, destination: Path, inherited: set[str]) -> None:
    """Fail on what the migration introduced; report what it merely carried.

    Runs inside the staging context: an introduced error raises `SystemExit`,
    which the staging context manager does not treat as its cleanup trigger
    (it only cleans up on `Exception`), so the failed release stays at
    `staged_path` for inspection while the source and the destination are both
    untouched.
    """
    print("Validating after", flush=True)
    result = validate_store(staged_path)
    introduced = [error for error in result.errors if error not in inherited]
    if introduced:
        for error in introduced:
            print(f"  ERROR {error}", file=sys.stderr)
        raise SystemExit(
            f"the migration introduced {len(introduced)} error(s) the source release did "
            f"not have. The failed release is at {staged_path} so it can be inspected; the "
            f"source release was not modified and nothing was published to {destination}. "
            "Discard the staging directory and keep the source release."
        )
    for warning in result.warnings:
        print(f"  warning: {warning}")
    if inherited:
        print(
            f"OK — introduced no new errors. {len(inherited)} pre-existing error(s) remain, "
            "and are not this tool's to fix: the release needs rebuilding for those."
        )
    else:
        print("OK")


def migrate(source: Path, destination: Path) -> int:
    """Derive a format-3.0 release at ``destination`` from ``source``.

    The source release is opened read-only and never written (spec §21.4,
    issue #156). The destination is built in a staging directory and published
    by rename only when the migrated copy validates; the destination must not
    already exist.
    """
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination:
        raise SystemExit(
            f"source and --into are the same path ({source}); a migration derives a new "
            "release and cannot write into the one it was given (spec §21.4)"
        )
    if destination.exists():
        raise SystemExit(
            f"{destination}: already exists; refusing to overwrite. A migration never "
            "replaces an existing release."
        )
    store = open_store(source)
    _refuse_unless_migratable(store)
    inherited = _inherited_errors(source)

    timer = PhaseTimer()
    with OpenGWASDBStore.staging(destination) as staged:
        print(f"Copying {source} -> {staged.path}", flush=True)
        _reflink_copy(source, staged.path)
        staged_store = open_store(staged.path)

        # The reflink copy is near-free, but on a filesystem that cannot reflink
        # it is a full copy of a multi-GiB store -- not a migration pass, and not
        # a number the phase accounting should carry. The clock starts when the
        # re-encode does.
        started = time.perf_counter()
        print(f"Measuring and re-encoding se: {staged.path}", flush=True)
        selected, _coefficients = optimise_dense_se_joint(
            staged_store.arrays(mode="a"), staged_store.manifest.encoding, timer=timer
        )
        print(f"  se encoding selected: {selected.se.to_manifest()}", flush=True)

        # The index carries decoded SE, so it describes the old plane until rebuilt.
        print("Rebuilding the top-hit index", flush=True)
        build_top_hit_indexes(staged.path, encoding=selected, timer=timer)

        elapsed = time.perf_counter() - started
        print("Phase accounting (issue #144):", flush=True)
        print(timer.format_report(), flush=True)
        _write_manifest(staged_store, selected, elapsed, timer)
        print(f"Re-stamped to {CURRENT_FORMAT_VERSION} in {elapsed:.1f}s", flush=True)

        _report_outcome(staged.path, destination, inherited)
    print(f"Published {destination}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "store",
        type=Path,
        metavar="STORE",
        help="the Dense 2.0 release to migrate; never modified",
    )
    parser.add_argument(
        "--into",
        type=Path,
        required=True,
        metavar="DEST",
        help="where the new release is published (must not already exist); the source "
        "release is immutable and is never written",
    )
    args = parser.parse_args(argv)
    return migrate(args.store, args.into)


if __name__ == "__main__":
    raise SystemExit(main())
