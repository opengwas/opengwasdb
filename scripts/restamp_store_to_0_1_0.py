#!/usr/bin/env python3
"""Derive a `format_version` 0.1.0 release from a 3.0 one, changing nothing but
the stamp.

The format reset (issue #143, ADR 0041) renumbered the store format and deleted
the readers for every pre-release version. That leaves the seven pilots and
`ukb-b` in formats this build refuses. The pilots are rebuilt -- they are ten to
twenty Analyses and cheap, and a rebuild is the only remedy that proves the
builder still produces what the format says. `ukb-b` is 9,847,701 x 2,511 and
takes 13h30m from 425 GB of source VCF (issue #148), which is why this tool
exists.

**It restamps a 3.0 release and nothing else.** That is the whole safety
argument: the reset changed the version string and the decoders this build
carries, not the bytes a build writes, so a 3.0 release's arrays, indexes and
`analyses.tsv` are exactly what a 0.1.0 build produces today. `0.1`, `1.0` and
`2.0` are genuinely different encodings -- `float16` planes, a `float32` `eaf`
plane -- and no stamp makes their bytes mean what 0.1.0 says. Those releases are
rebuilt, and this tool refuses them by version rather than trying.

It reads `manifest.json` as JSON rather than through `StoreManifest`, because
the model refuses a pre-reset release by design (that refusal is the feature).
Nothing else here looks at the store's contents: this tool has no decoder, and
one that could read a 3.0 release would be the compatibility surface the reset
deleted.

**Outside the Provenance Amendment exception, and the source is never written.**
Spec §21.4 says a format change derives a *new* release. The destination is
built in a `.name.tmp` staging directory and published by rename only when the
staged copy validates with **no** errors, so a failure at any point leaves the
source untouched and nothing where the destination was meant to appear (issues
#156, #164). The published release is genuinely new -- a fresh UUID4
`release_id` and a current-UTC `created_at`, never the source's -- and its
`overview.html` is regenerated so the page humans browse names the new identity
(ADR 0032).

The validation at the end is what makes this sound rather than asserted: the
staged copy is validated *as a 0.1.0 release*, by the build that reads only
0.1.0. A release that needed more than a stamp fails there and is discarded.

Every manifest in the release is restamped, which is what covers a Hybrid
release: its Dense Component is a nested Store Release with its own manifest
(CONTEXT.md), and a half-restamped Hybrid store is one this build could open at
the top and refuse one directory down.

Usage:

    restamp_store_to_0_1_0.py STORE --into DEST
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from opengwasdb.encoding import ENCODING_VERSION
from opengwasdb.layouts.dense.overview import write_overview_html
from opengwasdb.model.analyses import read_analyses
from opengwasdb.store.open import CURRENT_FORMAT_VERSION, OpenGWASDBStore
from opengwasdb.validation import validate_store

#: The only version whose bytes a 0.1.0 stamp describes truthfully.
RESTAMPABLE_FROM = "3.0"


class RestampError(Exception):
    """A refusal raised inside the staging block.

    An `Exception` subclass on purpose: `OpenGWASDBStore.staging` discards the
    staging directory on any `Exception` (issue #164). A `SystemExit` would
    escape that cleanup -- it is a `BaseException` -- and leave a failed copy
    behind.
    """


def _manifest_paths(store_path: Path) -> list[Path]:
    """Every manifest in the release, outermost first.

    A Hybrid release nests a Dense Component with a manifest of its own, and
    both carry `format_version`.
    """
    return sorted(store_path.rglob("manifest.json"), key=lambda p: len(p.parts))


def _refuse_unless_restampable(store_path: Path) -> None:
    """Fail against the *source*, before any bytes are copied."""
    manifests = _manifest_paths(store_path)
    if not manifests:
        raise SystemExit(f"{store_path}: no manifest.json; this is not a Store Release")
    for path in manifests:
        data = json.loads(path.read_text(encoding="utf-8"))
        version = str(data.get("format_version"))
        if version != RESTAMPABLE_FROM:
            raise SystemExit(
                f"{path}: format_version is {version!r}, not {RESTAMPABLE_FROM!r}. Only a "
                f"{RESTAMPABLE_FROM} release holds the bytes {CURRENT_FORMAT_VERSION} "
                "describes; every earlier format is a different encoding and is rebuilt, "
                "not restamped (ADR 0041, spec §21.4)."
            )
        declared = data.get("encoding")
        if declared is None or int(declared.get("version", 0)) != ENCODING_VERSION:
            raise SystemExit(
                f"{path}: declares encoding block version "
                f"{None if declared is None else declared.get('version')}, not "
                f"{ENCODING_VERSION}. The restamp asserts this release's planes are "
                "already what this build writes, and that is the claim it checks."
            )


def _reflink_copy(source: Path, destination: Path) -> None:
    """Copy the release's contents into ``destination``, sharing extents where
    the filesystem can.

    ``destination`` is an existing directory (the staging directory), so the
    copy fills it rather than nesting a subdirectory. A reflink makes the copy
    near-free until one side is written, which is what makes "restamp a copy,
    keep the original" affordable for a 42 GB store. Falls back to a full copy
    where the filesystem cannot.
    """
    subprocess.run(
        ["cp", "-a", "--reflink=auto", f"{source}/.", f"{destination}/"],
        check=True,
    )


def _restamp_manifests(staged_path: Path) -> str:
    """Re-stamp every manifest, minting one new release identity for them all.

    A derived release is a genuinely new one (issue #164): the source's
    `release_id` and `created_at` describe the release the copy came from. One
    UUID4 covers the whole release, including a Hybrid store's Dense Component,
    which a build gives the same `release_id` as the release that nests it.
    """
    now = datetime.now(UTC).isoformat()
    release_id = str(uuid.uuid4())
    for path in _manifest_paths(staged_path):
        data = json.loads(path.read_text(encoding="utf-8"))
        source_release_id = data["release_id"]
        data["release_id"] = release_id
        data["created_at"] = now
        data["format_version"] = CURRENT_FORMAT_VERSION
        data["provenance"] = {
            **data.get("provenance", {}),
            "format_restamp": {
                "from": RESTAMPABLE_FROM,
                "to": CURRENT_FORMAT_VERSION,
                "source_release_id": source_release_id,
                "tool": "scripts/restamp_store_to_0_1_0.py",
                "at": now,
                "note": (
                    "format_version restamped and a fresh release identity minted in a "
                    "new release, whose overview.html was regenerated to carry it. No "
                    "array, index or table was read or written: the format reset "
                    "renumbered the format and deleted the pre-release decoders, and "
                    "did not change the bytes a build writes, so a 3.0 release already "
                    "holds what 0.1.0 describes. The source release was not modified. "
                    "Outside the Provenance Amendment exception (spec §21.4)."
                ),
            },
        }
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"  restamped {path.relative_to(staged_path)}", flush=True)
    return release_id


def _refresh_overview(store_path: Path) -> None:
    """Regenerate ``overview.html`` so the release's own page names itself.

    The copy brought the source's page along, and its header embeds
    ``store_id · release <release_id> · …`` read from ``manifest.json``
    (ADR 0032, issue #164).
    """
    print("Regenerating overview.html for the new release identity", flush=True)
    write_overview_html(store_path, read_analyses(store_path / "analyses.tsv"))


def _require_valid_staged_release(staged_path: Path, destination: Path) -> None:
    """Publish only a staged copy that validates with no errors (issue #164).

    This is where the restamp is proved rather than asserted: the staged copy
    is validated by a build that reads only 0.1.0, so a release the stamp does
    not describe fails here and is discarded. No error is subtracted for having
    been inherited -- an error string identical to the source's is exactly how a
    defect this tool introduced would hide.
    """
    print("Validating the staged release", flush=True)
    result = validate_store(staged_path)
    for error in result.errors:
        print(f"  ERROR {error}", file=sys.stderr)
    for warning in result.warnings:
        print(f"  warning: {warning}")
    if not result.ok:
        raise RestampError(
            f"the restamped release has {len(result.errors)} validation error(s); a "
            "release that does not validate is never published. The staging directory "
            "was removed, the source release is unchanged, and nothing was published "
            f"to {destination}."
        )
    print("OK")


def restamp(source: Path, destination: Path) -> int:
    """Derive a 0.1.0 release at ``destination`` from a 3.0 one at ``source``."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination:
        raise SystemExit(
            f"source and --into are the same path ({source}); a restamp derives a new "
            "release and cannot write into the one it was given (spec §21.4)"
        )
    if destination.exists():
        raise SystemExit(
            f"{destination}: already exists; refusing to overwrite. A restamp never "
            "replaces an existing release."
        )
    _refuse_unless_restampable(source)

    with OpenGWASDBStore.staging(destination) as staged:
        print(f"Copying {source} -> {staged.path}", flush=True)
        _reflink_copy(source, staged.path)
        release_id = _restamp_manifests(staged.path)
        _refresh_overview(staged.path)
        _require_valid_staged_release(staged.path, destination)

    print(f"Published {destination} as {CURRENT_FORMAT_VERSION}, release_id {release_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "store",
        type=Path,
        help=f"the {RESTAMPABLE_FROM} release to restamp; never modified",
    )
    parser.add_argument(
        "--into",
        type=Path,
        required=True,
        help="where the new release is published (must not already exist); the source "
        "release is immutable and is never written",
    )
    args = parser.parse_args(argv)
    return restamp(args.store, args.into)


if __name__ == "__main__":
    raise SystemExit(main())
