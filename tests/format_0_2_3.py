"""A 0.2.3 Store Release made from a 0.3.0 build, for the tests that hold the 0.2 path.

From #467 every build and completion writes format 0.3.0. The 0.2 series is still
read, and `ogdb rewrite-release` turns a 0.2.3 release into a 0.3.0 one, so the tests
need 0.2.3 releases. A 0.3.0 release differs from the 0.2.3 one the same build would
have written in what 0.3.0 changed: it carries the variable-length ALID sidecar where
0.2.3 carried the fixed-width pair (ADR 0105), and it drops what nothing but the
validator read (ADR 0102, #471). `as_0_2_3` undoes exactly that, in every component of
a release (a Hybrid's nested Dense Component included):

- the sidecar's four files are deleted;
- `variant_alid_bytes.npy`/`variant_alid_rows.npy` are written by
  `write_fixed_width_alid_index`, a copy of what `write_variant_axis` wrote at
  `e93fba8f` -- the only other writer of the pair, and only ever in tests;
- `index.sqlite` and the tiers' `abs_z` and `p_value` come back
  (`with_0_2_derived_components`, as the builders before #471 wrote them);
- each manifest's `format_version` is re-spelt `0.2.3`, byte for byte otherwise.

That this gives what the builders at `e93fba8f` wrote was checked once, by building
the six `store_reader_digests` releases at `e93fba8f` and at #467 and comparing every
file (#467's report).
"""

from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from opengwasdb import top_hits
from opengwasdb.index.sqlite import initialise_schema, set_metadata
from opengwasdb.store import arrays as store_arrays
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


def with_0_2_derived_components(directory: Path) -> None:
    """Give a 0.3.0 component back what the 0.2 builders also wrote and 0.3.0 drops
    (ADR 0102, #471), as the builders before #471 wrote it:

    - `index.sqlite`. A Ragged release's own is empty (`build_ssf` created it), or
      holds only `completion_quality`. Every other one has the `metadata` table --
      `schema_version`, `n_variants`, `n_analyses` and the `dense` blob, the layout
      its manifest records -- and an empty `variant_aliases`, as every real release
      measured held (ADR 0102 §1). An existing index (a completion's, holding
      `completion_quality`) keeps that table.
    - each tier's `abs_z` (|z|, `float32`) and `p_value` (`erfc(|z| / sqrt 2)`,
      `float64`), as `opengwasdb.top_hits._write_tier` wrote them, through the
      package's one 0.2 writer of the pair (`top_hits.write_stored_p`).

    Idempotent: what a component already carries is left as it is, so a fixture that
    gave a component these before re-laying it out (`legacy_fixtures`) keeps them.
    """
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    _with_0_2_index(directory, manifest)
    _with_stored_p(directory / "data.zarr")


def _with_0_2_index(directory: Path, manifest: dict) -> None:
    index = directory / "index.sqlite"
    if manifest["primary_layout"] == "ragged":
        index.touch()
        return
    if index.exists():
        with closing(sqlite3.connect(index)) as connection:
            listed = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            if "metadata" in {row[0] for row in listed}:
                return
    provenance = manifest.get("provenance", {})
    recorded = provenance.get("hybrid") if (directory / "dense").is_dir() else None
    recorded = recorded or provenance.get("dense") or {}
    keys = ("chunk_shape", "shard_shape", "compressor", "zarr_format")
    with (directory / "analyses.tsv").open(encoding="utf-8") as handle:
        n_analyses = sum(1 for _ in csv.DictReader(handle, delimiter="\t"))
    n_variants = len(np.load(directory / "variant_offsets.npy", mmap_mode="r"))
    with sqlite3.connect(index) as connection:
        initialise_schema(connection)
        set_metadata(connection, "schema_version", 2)
        set_metadata(connection, "n_variants", n_variants)
        set_metadata(connection, "n_analyses", n_analyses)
        set_metadata(connection, "dense", {key: recorded[key] for key in keys if key in recorded})
        connection.commit()


def _with_stored_p(data: Path) -> None:
    if not (data / "zarr.json").is_file():
        # A Zarr v2 tree, re-laid out by `legacy_fixtures` after this gave it its arrays:
        # opening it for writing here would add v3 metadata beside its `.zgroup`.
        return
    root = store_arrays.open_group_for_write(data, "a")
    tiers = root["top_hits"] if "top_hits" in root else {}
    if tiers and not any("abs_z" in tiers[key] for key in tiers):
        # The package's 0.2 writer of the pair; `test_top_hit_tiers_0_3_0` holds what it
        # writes to `_write_tier`'s formula before #471, `erfc(|z| / sqrt 2)`.
        top_hits.write_stored_p(root)


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
        with_0_2_derived_components(directory)


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
    turned from 0.3.0 into 0.2.3: the fixed-width pair, what 0.3.0 dropped, and the stamp
    that carries them."""
    with_fixed_width_alid_index(directory)
    with_0_2_derived_components(directory)
    manifest = directory / "manifest.json"
    text = manifest.read_text(encoding="utf-8")
    stamp = '"format_version": "0.3.0"'
    assert text.count(stamp) == 1, f"{manifest} is not stamped 0.3.0 once"
    manifest.write_text(text.replace(stamp, '"format_version": "0.2.3"'), encoding="utf-8")
    return directory
