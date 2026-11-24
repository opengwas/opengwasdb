"""A 0.2.3 Store Release made from a 0.3.0 build, for the tests that hold the 0.2 path.

From #467 every build and completion writes format 0.3.0. The 0.2 series is still
read, and `ogdb rewrite-release` turns a 0.2.3 release into a 0.3.0 one, so the tests
need 0.2.3 releases. A 0.3.0 release differs from the 0.2.3 one the same build would
have written in five things: it carries the variable-length ALID sidecar where 0.2.3
carried the fixed-width pair (#467, ADR 0105), the position sidecar 0.2.3 did not carry
(#468), no `variant_offsets.npy`, which 0.2.3 carried (#469), it drops what nothing
but the validator read (#471, ADR 0102), and a Hybrid's Dense Component carries no copy
of the shared variants or Analyses (#472). `as_0_2_3` undoes exactly that, in every
component of a release (a Hybrid's nested Dense Component included):

- a Hybrid's Dense Component first gets its copies back (`with_dense_copies`): its
  Variant Table and sidecars, written by `write_variant_axis` from the shared rows
  `dense_to_shared.npy` names, as the build wrote them (a completed component's
  without `source_alid`, which its completion never wrote, #449); and its
  `analyses.tsv` and `overview.html`, the root's rows with the component's own Top-Hit
  Counts. Its `n_missing_se_out_of_range` cannot be recovered from 0.3.0, which keeps
  only the root's count of both components, so it is restored only where the root
  counts none and is refused otherwise;

- the ALID sidecar's four files and the position sidecar's three are deleted;
- `variant_alid_bytes.npy`/`variant_alid_rows.npy` are written by
  `write_fixed_width_alid_index`, a copy of what `write_variant_axis` wrote at
  `e93fba8f` -- the only other writer of the pair, and only ever in tests;
- `variant_offsets.npy` is written by `write_variant_offsets`, each row's BGZF virtual
  offset as a read of the table finds it. The writer at `e93fba8f` took them while
  writing; the two agree on a table of one BGZF block, as every fixture's is, and on a
  longer one may name a block boundary by either side of it, which seeks the same row;
- a Dense build's `provenance.dense.variant_axis.row_offsets` is not restored, nor is
  the `row_offsets` the old Dense builder put in the `dense` blob (the restored blob
  holds the layout keys only): no reader or rule reads either;
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
from opengwasdb.store_reader.alid_sidecar import (
    ALID_SIDECAR_FILENAMES,
    AlidSidecar,
    write_alid_sidecar,
)
from opengwasdb.store_reader.format import UnsupportedFormatVersion, carries_alid_sidecar
from opengwasdb.store_reader.position_sidecar import POSITION_SIDECAR_FILENAMES
from opengwasdb.store_reader.variant_axis import variant_alid_bytes_path, variant_alid_rows_path


def write_fixed_width_alid_index(directory: Path, alids: list[str]) -> None:
    """The 0.2 series' ALID index for `alids` (one per row), as `e93fba8f` wrote it."""
    indexable = [(alid, i) for i, alid in enumerate(alids) if is_indexable_alid(alid)]
    alid_bytes = np.array([alid for alid, _ in indexable], dtype=ALID_DTYPE)
    row_indices = np.array([i for _, i in indexable], dtype="int32")
    sort_order = np.argsort(alid_bytes, kind="stable")
    np.save(variant_alid_bytes_path(directory), alid_bytes[sort_order])
    np.save(variant_alid_rows_path(directory), row_indices[sort_order])


def write_variant_offsets(directory: Path) -> None:
    """`variant_offsets.npy`: each row's BGZF virtual offset in `directory`'s Variant
    Table, header lines skipped, as the 0.2 series carries it."""
    import pysam

    offsets: list[int] = []
    with pysam.BGZFile(str(directory / "variants.tsv.gz"), "r", None) as handle:
        while True:
            at = int(handle.tell())
            line = handle.readline()
            if not line:
                break
            if not line.startswith(b"#"):
                offsets.append(at)
    np.save(directory / "variant_offsets.npy", np.asarray(offsets, dtype=np.uint64))


def with_0_2_derived_components(directory: Path) -> None:
    """Give a 0.3.0 component back what the 0.2 builders also wrote and 0.3.0 drops
    (ADR 0102, #471), as the builders before #471 wrote it:

    - `index.sqlite`. A Ragged release's own is empty (`build_ssf` created it), or
      holds only `completion_quality`. Every other one has the `metadata` table --
      `schema_version`, `n_variants`, `n_analyses` and the `dense` blob, the layout
      its manifest records -- and an empty `variant_aliases`, as every real release
      measured held (ADR 0102 §1). An existing index (a completion's, holding
      `completion_quality`) keeps that table.
    - `variant_offsets.npy` (#469), where it is not there yet (`write_variant_offsets`);
    - each tier's `abs_z` (|z|, `float32`) and `p_value` (`erfc(|z| / sqrt 2)`,
      `float64`), as `opengwasdb.top_hits._write_tier` wrote them, through the
      package's one 0.2 writer of the pair (`top_hits.write_stored_p`).

    Idempotent: what a component already carries is left as it is, so a fixture that
    gave a component these before re-laying it out (`legacy_fixtures`) keeps them.
    """
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not (directory / "variant_offsets.npy").exists():  # 0.3.0 drops it too (#469)
        write_variant_offsets(directory)
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
    """Replace `directory`'s sidecars by the files a 0.2 release carries: the ALID
    sidecar's files give way to the fixed-width pair, the position sidecar's are deleted,
    and `variant_offsets.npy` is written (`write_variant_offsets`)."""
    sidecar = AlidSidecar.open(directory)
    assert sidecar is not None, f"{directory} has no ALID sidecar: not a 0.3.0 build"
    alids = sidecar.alids(np.arange(sidecar.n_rows)).tolist()
    del sidecar
    for name in ALID_SIDECAR_FILENAMES:
        (directory / name).unlink()
    for name in POSITION_SIDECAR_FILENAMES:
        (directory / name).unlink()
    write_fixed_width_alid_index(directory, alids)
    write_variant_offsets(directory)


def follow_stamp(directory: Path) -> None:
    """Give `directory` the ALID index its manifest's `format_version` carries.

    For a test that restamps a release this build wrote (0.3.0) as an earlier version:
    a 0.2 or 0.1 release carries the fixed-width pair, so a restamped one that kept the
    sidecar would be a release no build ever wrote, and fail `envelope.required`.
    """
    version = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))[
        "format_version"
    ]
    if not _sidecar_version(version):
        with_own_copies(directory)
    if not _sidecar_version(version) and AlidSidecar.open(directory) is not None:
        with_fixed_width_alid_index(directory)
        with_0_2_derived_components(directory)


def with_own_copies(directory: Path) -> None:
    """Give `directory`, if it is a 0.3.0 Hybrid's Dense Component, the copies of the
    shared variants and Analyses a release before 0.3.0 carries (`with_dense_copies`)."""
    if directory.name == "dense" and not (directory / "variants.tsv.gz").exists():
        if (directory / "dense_to_shared.npy").exists():
            with_dense_copies(directory.parent)


def _sidecar_version(version: str) -> bool:
    try:
        return carries_alid_sidecar(version)
    except UnsupportedFormatVersion:
        return False


def count_one_row_fewer(directory: Path) -> int:
    """Make what counts `directory`'s variants claim one row fewer than its table holds,
    and return the count it now claims: `variant_offsets.npy` on a 0.2 release, the ALID
    sidecar on 0.3.0, which counts them there (#469)."""
    offsets_path = directory / "variant_offsets.npy"
    version = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))[
        "format_version"
    ]
    if not _sidecar_version(version):
        offsets = np.load(offsets_path)
        assert len(offsets) >= 2, "the release must have a row to drop"
        np.save(offsets_path, offsets[:-1])
        return len(offsets) - 1
    sidecar = AlidSidecar.open(directory)
    assert sidecar is not None and sidecar.n_rows >= 2, "the release must have a row to drop"
    alids = sidecar.alids(np.arange(sidecar.n_rows - 1)).tolist()
    del sidecar
    write_alid_sidecar(directory, alids)
    return len(alids)


def as_0_2_3(release: Path) -> Path:
    """Turn the 0.3.0 release at `release` into the 0.2.3 one its build would have written."""
    for directory in components(release):
        component_as_0_2_3(directory)
    return release


def with_dense_copies(release: Path) -> None:
    """Give the 0.3.0 Hybrid at `release` the copies its Dense Component carried before
    0.3.0 (#472): its Variant Table and sidecars, `analyses.tsv` and `overview.html`."""
    from opengwasdb.layouts.dense.build import write_analyses_tsv
    from opengwasdb.model.analyses import read_analysis_records, reset_top_hit_counts
    from opengwasdb.top_hits import add_hit_counts
    from opengwasdb.variants import CanonicalVariant, write_variant_axis
    from opengwasdb.variants.legacy_table import iter_variant_records

    dense = release / "dense"
    mapped = np.load(dense / "dense_to_shared.npy").tolist()
    wanted = set(mapped)
    shared = {
        record.variant_index: record
        for record in iter_variant_records(release / "variants.tsv.gz")
        if record.variant_index in wanted
    }
    records = [shared[row] for row in mapped]
    manifest = json.loads((dense / "manifest.json").read_text(encoding="utf-8"))
    completed = manifest["completion_state"] == "reference_completed"
    write_variant_axis(
        dense,
        [CanonicalVariant(r.chromosome, r.position, r.effect_allele, r.other_allele)
         for r in records],
        {r.alid: r.rsid for r in records if r.rsid},
        None if completed else [r.source_alid for r in records],
    )
    analyses = read_analysis_records(release / "analyses.tsv")
    counted = [a for a in analyses if a.n_missing_se_out_of_range not in ("", "0")]
    assert not counted, (
        f"{release}: the root counts associations left out for an SE above 65,504 for "
        f"{counted[0].analysis_id!r}; which of them were its Dense Component's is not "
        "recorded at 0.3.0, so its 0.2.3 analyses.tsv cannot be restored"
    )
    write_analyses_tsv(dense, add_hit_counts(dense, reset_top_hit_counts(analyses)))


def component_as_0_2_3(directory: Path) -> Path:
    """One Store Release directory -- a release, or a Hybrid's nested Dense Component --
    turned from 0.3.0 into 0.2.3: the fixed-width pair, what 0.3.0 dropped, and the stamp
    that carries them."""
    with_own_copies(directory)
    with_fixed_width_alid_index(directory)
    with_0_2_derived_components(directory)
    manifest = directory / "manifest.json"
    text = manifest.read_text(encoding="utf-8")
    stamp = '"format_version": "0.3.0"'
    assert text.count(stamp) == 1, f"{manifest} is not stamped 0.3.0 once"
    manifest.write_text(text.replace(stamp, '"format_version": "0.2.3"'), encoding="utf-8")
    return directory
