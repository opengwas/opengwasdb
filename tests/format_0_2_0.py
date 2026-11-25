"""A 0.2.0-shaped Dense Store Release, for the tests of `--convert-legacy` (#492).

The hosted Stores the conversion is for (OGS-00009, OGS-00012 to 00017) are 0.2.0
Dense releases built by `build-dense-vcf`. They differ from the 0.2.3 release a build
of today would have written (`format_0_2_3.as_0_2_3`) in what `as_0_2_0` gives one:

- `format_version` 0.2.0, in the manifest and in each Indexed Variant Subset's record
  of the release it was built on;
- no `n_missing_se_out_of_range` column in `analyses.tsv` (ADR 0068): it came with
  0.2.3;
- a blank `source_alid` on every row: they were built before #449, and every row of
  every one of them reads `.` (measured on OGS-00015, 11,650,615 rows);
- `provenance.chain_file`, which every Dense VCF build records, lifted or not;
- optionally an indel spelt sorted, effect and other allele too (`sorted_indel`), as
  before #381: the table and its ALID index agree with each other and not with the
  reference.

`plant_se` stores an SE above 65,504 at a cell as a writer before 0.2.3 did -- an exact
exception in a residual plane -- and in every Top-Hit tier that holds the cell.
`round_se_exceptions` makes every present SE an exception holding its `float16`
rounding, as the hosted Stores' builds did before #355 (99.07% to 100% of their
exceptions are `float16` values).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from format_0_2_3 import write_fixed_width_alid_index, write_variant_offsets

from opengwasdb.encoding.codec import write_exact_table
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store_reader.encoding.codec import SeExceptionTable
from opengwasdb.store_reader.encoding.plan import SE_EXCEPTION
from opengwasdb.store_reader.layout import ArrayRole
from opengwasdb.variants.axis import rewrite_source_alid_column
from opengwasdb.variants.legacy_table import iter_variant_records

#: What every hosted 0.2.0 Dense VCF build records as its chain.
CHAIN_LABEL = "pyliftover_builtin_hg19_hg38"


def as_0_2_0(release: Path) -> Path:
    """Turn the 0.2.3 Dense release at `release` into the 0.2.0 shape (module docstring)."""
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_version"] == "0.2.3" and manifest["primary_layout"] == "dense"
    manifest["format_version"] = "0.2.0"
    manifest["provenance"]["chain_file"] = CHAIN_LABEL
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    for subset in sorted((release / "data.zarr" / "indexed_subsets").glob("*/zarr.json")):
        text = subset.read_text(encoding="utf-8")
        subset.write_text(text.replace('"0.2.3"', '"0.2.0"'), encoding="utf-8")
    _without_count_column(release / "analyses.tsv")
    rewrite_source_alid_column(release, lambda _row, _alid, _held: None)
    write_variant_offsets(release)
    return release


def _without_count_column(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        names = [n for n in reader.fieldnames or [] if n != "n_missing_se_out_of_range"]
        rows = list(reader)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, names, delimiter="\t", extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sorted_indel(release: Path, reference_first: str) -> str:
    """Spell the indel `reference_first` sorted throughout -- ALID, effect and other
    allele -- in a 0.2 release's table and fixed-width ALID index; return the new ALID."""
    chrom, pos, a1, a2 = reference_first.split(":")
    spelt = f"{chrom}:{pos}:{min(a1, a2)}:{max(a1, a2)}"
    assert spelt != reference_first, "the indel must sort the other way"
    records = list(iter_variant_records(release / "variants.tsv.gz"))
    assert reference_first in {r.alid for r in records}, "the fixture must hold the indel"
    from test_indel_alid import _rewrite_table

    def edit(line: str) -> str:
        fields = line.rstrip("\n").split("\t")
        if fields[5] == reference_first:
            fields[3], fields[4], fields[5] = min(a1, a2), max(a1, a2), spelt
        return "\t".join(fields) + "\n"

    _rewrite_table(release, edit)
    write_fixed_width_alid_index(
        release, [r.alid for r in iter_variant_records(release / "variants.tsv.gz")]
    )
    return spelt


def plant_se(release: Path, row: int, analysis: int, value: float) -> None:
    """Store an SE of `value` at one cell of a residual SE plane, as an exact exception,
    and in the `se` of every tier holding the cell."""
    root = store_arrays.open_group_for_write(release / "data.zarr", "a")
    n_analyses = int(root["se"].shape[1])
    position = row * n_analyses + analysis
    store_arrays.write_shard_cells(root["se"], [row], [analysis], np.array([SE_EXCEPTION], np.int8))
    table = SeExceptionTable.read(root)
    keep = table.index != position
    index = np.append(table.index[keep], position)
    values = np.append(table.value[keep], np.float32(value))
    order = np.argsort(index)
    write_exact_table(
        SeExceptionTable(index=index[order], value=values[order].astype(np.float32)), root,
        compressor=store_arrays.compressor(),
    )
    tiers = root["top_hits"] if "top_hits" in root else {}
    for key in list(tiers):
        tier = tiers[key]
        held = (tier["variant_index"][:] == row) & (tier["analysis_index"][:] == analysis)
        if held.any():
            se = np.asarray(tier["se"][:], dtype=np.float32)
            se[held] = value
            store_arrays.create_array(
                tier, "se", ArrayRole.TOP_HIT_INDEX, data=se, dtype="float32",
                compressor=store_arrays.compressor(), overwrite=True,
            )


def a_top_hit(release: Path, *, imputed: bool = False) -> tuple[int, int]:
    """A cell of the loosest tier, observed (or, with `imputed`, imputed)."""
    import zarr

    tiers = zarr.open_group(str(release / "data.zarr" / "top_hits"), mode="r")
    tier = tiers[sorted(tiers.group_keys())[-1]]
    flags = tier["imputed"][:] if "imputed" in tier else np.zeros(tier["z"].shape, np.uint8)
    (slots,) = np.nonzero(flags == (1 if imputed else 0))
    assert slots.size, "the fixture's tier must hold such a cell"
    slot = int(slots[0])
    return int(tier["variant_index"][slot]), int(tier["analysis_index"][slot])


def round_se_exceptions(release: Path) -> int:
    """Hold every present SE of the residual plane as an exact exception rounded to
    `float16`, as a build before #355 wrote it; return how many exceptions there are."""
    from opengwasdb.store.open import open_store
    from opengwasdb.store_reader.associations import Cells, DenseGridReader

    store = open_store(release)
    completed = store.manifest.completion_state.value == "reference_completed"
    grid = DenseGridReader.open(store.arrays(), store.manifest.encoding, requires_mask=completed)
    cells = grid.inspect(Cells.variants(np.arange(grid.n_variants)))
    present = np.isfinite(cells.se)
    positions = (
        cells.variant_index[present].astype(np.int64) * grid.n_analyses
        + cells.analysis_index[present]
    )
    order = np.argsort(positions)
    values = cells.se[present].astype(np.float16).astype(np.float32)[order]
    root = store_arrays.open_group_for_write(release / "data.zarr", "a")
    store_arrays.write_shard_cells(
        root["se"], positions // grid.n_analyses, positions % grid.n_analyses,
        np.full(positions.size, SE_EXCEPTION, np.int8),
    )
    write_exact_table(
        SeExceptionTable(index=positions[order], value=values), root,
        compressor=store_arrays.compressor(),
    )
    tiers = root["top_hits"] if "top_hits" in root else {}
    for key in list(tiers):  # harvested from the rounded plane, as the build's were
        tier = tiers[key]
        se = np.asarray(tier["se"][:], dtype=np.float32).astype(np.float16).astype(np.float32)
        store_arrays.create_array(
            tier, "se", ArrayRole.TOP_HIT_INDEX, data=se, dtype="float32",
            compressor=store_arrays.compressor(), overwrite=True,
        )
    return int(positions.size)
