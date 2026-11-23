"""A 0.2.3 Store Release made from a 0.3.0 build, for the tests that hold the 0.2 path.

From #467 every build and completion writes format 0.3.0. The 0.2 series is still
read, and `ogdb rewrite-release` turns a 0.2.3 release into a 0.3.0 one, so the tests
need 0.2.3 releases. A 0.3.0 release differs from the 0.2.3 one the same build would
have written in one thing (ADR 0105): it carries the variable-length ALID sidecar
where 0.2.3 carried the fixed-width pair. `as_0_2_3` undoes exactly that, in every
component of a release (a Hybrid's nested Dense Component included):

- the sidecar's four files are deleted;
- `variant_alid_bytes.npy`/`variant_alid_rows.npy` are written by
  `write_fixed_width_alid_index`, a copy of what `write_variant_axis` wrote at
  `e93fba8f` -- the only other writer of the pair, and only ever in tests;
- each manifest's `format_version` is re-spelt `0.2.3`, byte for byte otherwise.

That this gives what the builders at `e93fba8f` wrote was checked once, by building
the six `store_reader_digests` releases at `e93fba8f` and at #467 and comparing every
file (#467's report).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from opengwasdb.store_reader.alid import ALID_DTYPE, is_indexable_alid
from opengwasdb.store_reader.alid_sidecar import ALID_SIDECAR_FILENAMES, AlidSidecar
from opengwasdb.store_reader.format import UnsupportedFormatVersion, carries_alid_sidecar
from opengwasdb.store_reader.variant_axis import variant_alid_bytes_path, variant_alid_rows_path


def write_fixed_width_alid_index(directory: Path, alids: list[str]) -> None:
    """The 0.2 series' ALID index for `alids` (one per row), as `e93fba8f` wrote it."""
    indexable = [(alid, i) for i, alid in enumerate(alids) if is_indexable_alid(alid)]
    alid_bytes = np.array([alid for alid, _ in indexable], dtype=ALID_DTYPE)
    row_indices = np.array([i for _, i in indexable], dtype="int32")
    sort_order = np.argsort(alid_bytes, kind="stable")
    np.save(variant_alid_bytes_path(directory), alid_bytes[sort_order])
    np.save(variant_alid_rows_path(directory), row_indices[sort_order])


def components(release: Path) -> list[Path]:
    """The release, and its nested Dense Component if it is a Hybrid."""
    nested = release / "dense"
    return [release, nested] if (nested / "manifest.json").exists() else [release]


def with_fixed_width_alid_index(directory: Path) -> None:
    """Replace `directory`'s ALID sidecar by the fixed-width pair a 0.2 release carries."""
    sidecar = AlidSidecar.open(directory)
    assert sidecar is not None, f"{directory} has no ALID sidecar: not a 0.3.0 build"
    alids = sidecar.alids(np.arange(sidecar.n_rows)).tolist()
    del sidecar
    for name in ALID_SIDECAR_FILENAMES:
        (directory / name).unlink()
    write_fixed_width_alid_index(directory, alids)


def follow_stamp(directory: Path) -> None:
    """Give `directory` the ALID index its manifest's `format_version` carries.

    For a test that restamps a release this build wrote (0.3.0) as an earlier version:
    a 0.2 or 0.1 release carries the fixed-width pair, so a restamped one that kept the
    sidecar would be a release no build ever wrote, and fail `envelope.required`.
    """
    version = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))[
        "format_version"
    ]
    if not _sidecar_version(version) and AlidSidecar.open(directory) is not None:
        with_fixed_width_alid_index(directory)


def _sidecar_version(version: str) -> bool:
    try:
        return carries_alid_sidecar(version)
    except UnsupportedFormatVersion:
        return False


def as_0_2_3(release: Path) -> Path:
    """Turn the 0.3.0 release at `release` into the 0.2.3 one its build would have written."""
    for directory in components(release):
        component_as_0_2_3(directory)
    return release


def component_as_0_2_3(directory: Path) -> Path:
    """One Store Release directory -- a release, or a Hybrid's nested Dense Component --
    turned from 0.3.0 into 0.2.3: the fixed-width pair, and the stamp that carries it."""
    with_fixed_width_alid_index(directory)
    manifest = directory / "manifest.json"
    text = manifest.read_text(encoding="utf-8")
    stamp = '"format_version": "0.3.0"'
    assert text.count(stamp) == 1, f"{manifest} is not stamped 0.3.0 once"
    manifest.write_text(text.replace(stamp, '"format_version": "0.2.3"'), encoding="utf-8")
    return directory
