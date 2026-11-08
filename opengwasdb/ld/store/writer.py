"""Writing an LD Store: the seam the builder writes through (spec §1–§7, §12; ogld#4).

`StoreWriter` takes Stored r as int16 codes the builder quantised with
`ogld.format.quantise`, chromosome by chromosome and the Near Band in row blocks, because
one chromosome's band can exceed 70 GB raw (spec §4). A store is not one until `finish`
writes its provenance; a reader refuses a directory without provenance, so a half-written
store cannot be read as a short one.

This is the walking skeleton's version (ogld#12): it holds one chromosome's Near Band in
memory until the chromosome is complete, and `finish` does not yet run the spec's
validation rules. ogld#4 replaces both, without changing a signature.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import zarr

from ogld.format import ENCODING, RESERVED_CODE, Provenance
from ogld.store import layout
from ogld.store.tables import Variants


class StoreWriter:
    """Writes one LD Store (spec §1)."""

    def __init__(
        self,
        root: zarr.Group,
        variants: Variants,
        ranges: dict[str, tuple[int, int]],
        near_bp: int,
        far_bp: int,
    ) -> None:
        self._root = root
        self._position = variants.position
        self._ranges = ranges
        self._near_bp = near_bp
        self._far_bp = far_bp
        self._pending: dict[str, list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int16]]]] = {}
        self._next_row = {chrom: start for chrom, (start, _) in ranges.items()}
        self._near_done: set[str] = set()
        self._index_done: set[str] = set()

    @classmethod
    def create(
        cls, location: str | os.PathLike[str], *, variants: Variants, near_bp: int, far_bp: int
    ) -> StoreWriter:
        """Start a store at a new `location`: write the Variant Set, in engine order, and
        its chromosome ranges (spec §2). Refuses a location that exists (ogld#4)."""
        path = Path(location)
        if path.exists():
            raise FileExistsError(f"{path} exists: a store is never written over another")
        ranges = chromosome_ranges(variants.chrom)
        root = zarr.open_group(str(path), mode="w-", zarr_format=3)
        group = root.create_group("variants")
        layout.put_numbers(group, "position", variants.position.astype(np.int32), 0)
        layout.put_strings(group, "alid", variants.alid)
        layout.put_strings(group, "a1", variants.a1)
        layout.put_strings(group, "a2", variants.a2)
        layout.put_numbers(group, "a1_frequency", variants.a1_frequency.astype(np.float32), np.nan)
        group.attrs["chromosomes"] = [
            {"name": name, "row_start": start, "row_stop": stop}
            for name, (start, stop) in ranges.items()
        ]
        return cls(root, variants, ranges, near_bp, far_bp)

    def _chromosome_of(self, row: int) -> str:
        for chrom, (start, stop) in self._ranges.items():
            if start <= row < stop:
                return chrom
        raise ValueError(f"row {row} is not in the Variant Set of {len(self._position)} rows")

    def write_near(
        self, first_row: int, lengths: npt.NDArray[np.int64], codes: npt.NDArray[np.int16]
    ) -> None:
        """Append the Near Band rows `first_row`, `first_row + 1`, … (global rows, one
        chromosome, in order): `lengths[i]` codes for each, back to back (spec §4)."""
        if len(lengths) == 0:
            return
        chrom = self._chromosome_of(first_row)
        _, stop = self._ranges[chrom]
        if first_row != self._next_row[chrom]:
            raise ValueError(
                f"Near Band rows of chromosome {chrom} must be written in order: "
                f"expected row {self._next_row[chrom]}, got {first_row}"
            )
        self._check_near(first_row, stop, lengths, codes)
        self._pending.setdefault(chrom, []).append((lengths.astype(np.int64), codes))
        self._next_row[chrom] = first_row + len(lengths)
        if self._next_row[chrom] == stop:
            self._flush_near(chrom)

    def _check_near(
        self,
        first_row: int,
        stop: int,
        lengths: npt.NDArray[np.int64],
        codes: npt.NDArray[np.int16],
    ) -> None:
        """Refuse Near Band rows that run past their chromosome, disagree with their codes,
        hold the reserved code or are not the rows spec §4 gives."""
        chrom = self._chromosome_of(first_row)
        if first_row + len(lengths) > stop:
            raise ValueError(f"Near Band rows run past the end of chromosome {chrom}")
        if int(lengths.sum()) != len(codes):
            raise ValueError("the Near Band's lengths and codes disagree on how many codes")
        if (codes == RESERVED_CODE).any():
            raise ValueError(f"code {RESERVED_CODE} is reserved and never a correlation (spec §3)")
        expected = self._near_lengths(first_row, first_row + len(lengths))
        if not np.array_equal(lengths, expected):
            raise ValueError(
                f"a Near Band row holds every later variant within near_bp = {self._near_bp}: "
                f"rows {first_row}–{first_row + len(lengths) - 1} of chromosome {chrom} do not"
            )

    def _near_lengths(self, first: int, last: int) -> npt.NDArray[np.int64]:
        """The Near Band row lengths spec §4 gives rows `first` … `last − 1` of one chromosome."""
        start, stop = self._ranges[self._chromosome_of(first)]
        position = self._position[start:stop].astype(np.int64)
        local = np.arange(first - start, last - start)
        reach = np.searchsorted(position, position[local] + self._near_bp, side="right")
        return cast(npt.NDArray[np.int64], (reach - local - 1).astype(np.int64))

    def _flush_near(self, chrom: str) -> None:
        parts = self._pending.pop(chrom)
        lengths = np.concatenate([p[0] for p in parts])
        codes = np.concatenate([p[1] for p in parts])
        indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        group = self._root.require_group(f"near/{chrom}")
        layout.put_numbers(group, "indptr", indptr, layout.INT64_FILL)
        array = layout.new_r_array(group, "r", len(codes), layout.NEAR_R)
        if len(codes):
            array[:] = codes
        self._near_done.add(chrom)

    def write_index(
        self,
        chrom: str,
        indptr: npt.NDArray[np.int64],
        partner: npt.NDArray[np.int64],
        codes: npt.NDArray[np.int16],
    ) -> None:
        """Write one chromosome's Proxy Index in CSR form over its rows, partners as
        global rows ascending within each row, both directions of every pair (spec §5)."""
        start = self._check_index(chrom, indptr, partner, codes)
        group = self._root.require_group(f"index/{chrom}")
        layout.put_numbers(group, "indptr", indptr.astype(np.int64), layout.INT64_FILL)
        local = (partner - start).astype(np.int32)
        layout.put_numbers(group, "partner", local, layout.PARTNER_FILL)
        layout.put_numbers(group, "r", codes.astype(layout.R_DTYPE), RESERVED_CODE)
        self._index_done.add(chrom)

    def _check_index(
        self,
        chrom: str,
        indptr: npt.NDArray[np.int64],
        partner: npt.NDArray[np.int64],
        codes: npt.NDArray[np.int16],
    ) -> int:
        """Refuse a Proxy Index that does not fit its chromosome; return its first row."""
        if chrom not in self._ranges:
            raise ValueError(f"chromosome {chrom} is not in the Variant Set")
        start, stop = self._ranges[chrom]
        if len(indptr) != stop - start + 1 or indptr[0] != 0 or indptr[-1] != len(partner):
            raise ValueError(
                f"the Proxy Index's indptr must span the {stop - start} rows of {chrom}"
            )
        _check_entries(chrom, (start, stop), partner, codes)
        return start

    def finish(self, provenance: Provenance) -> Provenance:
        """Write the provenance and return it as written (spec §7).

        Refuses, naming the chromosome, a store whose Near Band or Proxy Index is incomplete,
        and a provenance that disagrees with what was written. It does not yet run the
        spec's validation rules (§12): `ogld.validate` is ogld#4's.
        """
        chromosomes = set(self._ranges)
        for what, done in (("Near Band", self._near_done), ("Proxy Index", self._index_done)):
            if done != chromosomes:
                missing = ", ".join(sorted(chromosomes - done))
                raise ValueError(f"the {what} of chromosome {missing} was not written")
        if provenance["n_variants"] != len(self._position):
            raise ValueError("provenance n_variants is not the Variant Set's size")
        if (provenance["near_bp"], provenance["far_bp"]) != (self._near_bp, self._far_bp):
            raise ValueError("provenance near_bp and far_bp are not the distances written with")
        if provenance["encoding"] != ENCODING:
            raise ValueError("provenance records an encoding this writer does not write (spec §3)")
        attributes: dict[str, Any] = {"ld_store": dict(provenance)}
        self._root.attrs.update(attributes)
        return provenance


def _check_entries(
    chrom: str,
    rows: tuple[int, int],
    partner: npt.NDArray[np.int64],
    codes: npt.NDArray[np.int16],
) -> None:
    """Refuse entries that disagree in number, leave the chromosome or hold the reserved code."""
    if len(partner) != len(codes):
        raise ValueError("the Proxy Index's partners and codes disagree on how many entries")
    if len(partner) and (partner.min() < rows[0] or partner.max() >= rows[1]):
        raise ValueError(f"a Proxy Index partner lies outside chromosome {chrom}")
    if (codes == RESERVED_CODE).any():
        raise ValueError(f"code {RESERVED_CODE} is reserved and never a correlation (spec §3)")


def chromosome_ranges(chrom: npt.NDArray[np.str_]) -> dict[str, tuple[int, int]]:
    """Each chromosome's [row_start, row_stop), in order of appearance; a chromosome that
    reappears after another breaks the Variant Set's order (spec §2)."""
    ranges: dict[str, tuple[int, int]] = {}
    if len(chrom) == 0:
        return ranges
    edges = np.flatnonzero(chrom[1:] != chrom[:-1]) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [len(chrom)]])
    for start, stop in zip(starts, stops, strict=True):
        name = str(chrom[start])
        if name in ranges:
            raise ValueError(f"chromosome {name} is not contiguous in the Variant Set (spec §2)")
        ranges[name] = (int(start), int(stop))
    return ranges
