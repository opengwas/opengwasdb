"""Writing an LD Store: the seam the builder writes through (spec §1–§7, §12; ogld#4).

`StoreWriter` takes Stored r as int16 codes the builder quantised with
`ogld.format.quantise`, chromosome by chromosome and the Near Band in row blocks, because
one chromosome's band can exceed 70 GB raw (spec §4). A store is not one until `finish`
writes its provenance; a reader refuses a directory without provenance, so a half-written
store cannot be read as a short one.

`finish` runs every validation rule of spec §12 on what was written, judged with the
provenance it is given, and writes that provenance only when all 18 pass. So a store that
breaks a rule is refused, naming the rule and where, and stays unreadable.

It holds one chromosome's Near Band in memory until the chromosome is complete (ogld#12).
Bounding that is the builder's concern (ogld#5), and needs no change of signature.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import zarr

from ogld.format import ENCODING, RESERVED_CODE, Provenance
from ogld.format.geometry import chromosome_of, chromosome_ranges, listed_ranges, near_row_lengths
from ogld.store import layout
from ogld.store.tables import Variants
from ogld.store.validation import ValidationFailed, validate_group


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
        self._lengths: dict[str, npt.NDArray[np.int64]] = {}
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
        layout.put_numbers(group, "position", _positions(variants.position), layout.POSITION_FILL)
        layout.put_strings(group, "alid", variants.alid)
        layout.put_strings(group, "a1", variants.a1)
        layout.put_strings(group, "a2", variants.a2)
        layout.put_numbers(
            group,
            "a1_frequency",
            variants.a1_frequency.astype(np.float32),
            layout.FREQUENCY_FILL,
        )
        group.attrs["chromosomes"] = listed_ranges(ranges)
        return cls(root, variants, ranges, near_bp, far_bp)

    def write_near(
        self, first_row: int, lengths: npt.NDArray[np.int64], codes: npt.NDArray[np.int16]
    ) -> None:
        """Append the Near Band rows `first_row`, `first_row + 1`, … (global rows, one
        chromosome, in order): `lengths[i]` codes for each, back to back (spec §4)."""
        if len(lengths) == 0:
            return
        chrom = chromosome_of(self._ranges, first_row)
        _, stop = self._ranges[chrom]
        if first_row != self._next_row[chrom]:
            raise ValueError(
                f"Near Band rows of chromosome {chrom} must be written in order: "
                f"expected row {self._next_row[chrom]}, got {first_row}"
            )
        _check_dtype(codes)
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
        chrom = chromosome_of(self._ranges, first_row)
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
        chrom = chromosome_of(self._ranges, first)
        if chrom not in self._lengths:
            start, stop = self._ranges[chrom]
            self._lengths[chrom] = near_row_lengths(self._position[start:stop], self._near_bp)
        start, _ = self._ranges[chrom]
        return self._lengths[chrom][first - start : last - start]

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
        _check_dtype(codes)
        start = self._check_index(chrom, indptr, partner, codes)
        group = self._root.require_group(f"index/{chrom}")
        layout.put_numbers(group, "indptr", indptr.astype(np.int64), layout.INT64_FILL)
        local = (partner - start).astype(np.int32)
        layout.put_numbers(group, "partner", local, layout.PARTNER_FILL)
        layout.put_numbers(group, "r", codes, RESERVED_CODE)
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

    def finish(self, provenance: Provenance, *, threads: int = 1) -> Provenance:
        """Validate the store and write its provenance; return it as written (spec §7, §12).

        Refuses, naming the chromosome, a store whose Near Band or Proxy Index is incomplete,
        and a provenance that disagrees with what was written. Then it runs every rule of
        spec §12, reading and checking `threads` blocks at once, and raises `ValidationFailed`
        if any fails, before the provenance is written: a reader refuses a store without one,
        so a broken store is never read.
        """
        chromosomes = set(self._ranges)
        for what, done in (("Near Band", self._near_done), ("Proxy Index", self._index_done)):
            if done != chromosomes:
                missing = ", ".join(sorted(chromosomes - done))
                raise ValueError(f"the {what} of chromosome {missing} was not written")
        self._check_provenance(provenance)
        recorded = _as_json(provenance)
        findings = validate_group(self._root, recorded, threads=threads)
        if any(findings.values()):
            raise ValidationFailed(findings)
        self._root.attrs.update({"ld_store": recorded})
        return cast(Provenance, recorded)

    def _check_provenance(self, provenance: Provenance) -> None:
        """Refuse a provenance that disagrees with what this writer wrote."""
        if provenance["n_variants"] != len(self._position):
            raise ValueError("provenance n_variants is not the Variant Set's size")
        if (provenance["near_bp"], provenance["far_bp"]) != (self._near_bp, self._far_bp):
            raise ValueError("provenance near_bp and far_bp are not the distances written with")
        if provenance["encoding"] != ENCODING:
            raise ValueError("provenance records an encoding this writer does not write (spec §3)")


def _as_json(provenance: Provenance) -> dict[str, Any]:
    """The provenance as the JSON attribute will hold it, so what is validated is what is
    stored; a value JSON cannot hold, such as a numpy integer, is refused, not converted."""
    try:
        return cast(dict[str, Any], json.loads(json.dumps(provenance)))
    except TypeError as unwritable:
        raise ValueError(f"provenance must be plain JSON values: {unwritable}") from unwritable


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


def _check_dtype(codes: npt.NDArray[np.int16]) -> None:
    """Codes are stored as they come: any dtype but the encoding's would wrap, so it is refused."""
    if codes.dtype != layout.R_DTYPE:
        raise ValueError(
            f"codes must be {layout.R_DTYPE} as `ogld.format.quantise` makes them, not "
            f"{codes.dtype}: a cast would wrap an out-of-range code into a wrong r (spec §3)"
        )


def _positions(position: npt.NDArray[np.int64]) -> npt.NDArray[np.int32]:
    """Positions as the int32 array stores them; one that does not fit is refused, not wrapped."""
    if len(position) and (position.min() < 0 or position.max() > np.iinfo(np.int32).max):
        raise ValueError("a position does not fit the int32 `position` array (spec §2)")
    return position.astype(np.int32)
