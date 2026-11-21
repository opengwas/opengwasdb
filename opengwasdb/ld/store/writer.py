"""Writing an LD Store: the seam every producer writes through (spec §1–§7, §12; ogld#4, ogld#17).

A producer supplies r and what only its source knows; the store assembly behind this seam
(`ogld/store/assembly.py`) does the rest (owner, 10 November 2026). That covers quantising at the
store's encoding, the row lengths, the thresholds at the far code floor and the Partner Floor, the
Far Pairs, the counts, the Partner Rows' frames (`ogld/store/frames.py`), the provenance a store
derives from itself, and validation.

How each producer calls it:

- **The builder** creates the writer with its encoding, `NEAR_BP`, `FAR_BP`, `t_partner`, the
  kept samples' n and the covariates' k. Each of its workers computes a tile's product,
  `ogld.genotypes.r(rows, span)`, with `span` running from the tile's first row to the far stop
  of row `last − 1`. It cuts the tile, `writer.cut(start + first, r)`, and measures its own PC
  Diagnostic on the same `r`. The build's thread writes the cuts in row order,
  `writer.write(cut)`, then calls `finish`.
- **The Test Store's script** cuts each chromosome whole from the reference r as a dense matrix.
  The matrix is NaN where the reference holds no pair, so a pair it lacks within far_bp is
  refused by `cut`.
- **A Pan-UKB import (later)** creates the writer with the far_bp its lift to GRCh38 leaves whole,
  its N and k = 15, cuts blocks of rows from the published block matrix, and finishes with
  `source.kind = "ld_matrix"`.

A store is not one until `finish` writes its provenance. A reader refuses a directory without
provenance, so a half-written store cannot be read as a short one. `finish` runs every rule of
spec §12 on what was written and writes the provenance only when all 27 pass.

What a store holds is `format_version` 0.2.0's: the Near Band, the Far Pairs and the Partner Rows
(partner-rows §5; ADR 0011). It holds no Proxy Index: #72 removed the component, and no array of it
is written here or read by a reader.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import zarr

from ogld.format import Provenance, REncoding, Sourced
from ogld.format.geometry import chromosome_ranges
from ogld.store import layout
from ogld.store.assembly import (
    Chromosome,
    Cut,
    Cutter,
    Tally,
    derive_alid_index,
    derive_alid_mmap,
    provenance,
)
from ogld.store.tables import Variants
from ogld.store.validation import ValidationFailed, validate_group


class StoreWriter:
    """Writes one LD Store (spec §1) from the r its producer supplies, block by block."""

    def __init__(
        self,
        root: zarr.Group,
        cutter: Cutter,
        ranges: dict[str, tuple[int, int]],
        *,
        compress_workers: int = 1,
    ) -> None:
        self._root = root
        self._cutter = cutter
        self._ranges = ranges
        self._compress_workers = compress_workers
        self._rows = sum(stop - start for start, stop in ranges.values())
        self._next = 0
        self._open: Chromosome | None = None
        self._tally = Tally()
        self._failure: tuple[str, BaseException] | None = None

    @classmethod
    def create(
        cls,
        location: str | os.PathLike[str],
        *,
        variants: Variants,
        encoding: REncoding,
        near_bp: int,
        far_bp: int,
        n: int,
        k: int,
        t_partner: float,
        compress_workers: int = 1,
    ) -> StoreWriter:
        """Start a store at a new `location`: write the Variant Set, in engine order, and its
        chromosome ranges (spec §2), and fix what every block is cut by. That is the encoding
        (spec §3), the distances (§4, §5), t_far for `n` samples and `k` covariates with its far
        code floor (§3), and `t_partner` with its Partner Floor (partner-rows §3).

        `n` and `k` are given here because blocks are thresholded as they arrive; `finish`
        refuses a `Sourced` whose `panel.n` or `adjustment.k` differs. `t_partner` is the
        **Partner Threshold**, a build input with no default, and never above t_far: one that is
        raises, naming both (ADR 0011, D3; partner-rows §3). `compress_workers` is how many
        worker threads compress the Partner Rows' frames; their bytes do not depend on it.

        Refuses, before writing anything, a location that exists, an n and k with no t_far, a
        t_partner above t_far, and a position the int32 `position` array cannot hold.
        """
        path = Path(location)
        if path.exists():
            raise FileExistsError(f"{path} exists: a store is never written over another")
        ranges = chromosome_ranges(variants.chrom)
        cutter = Cutter(
            variants.position,
            ranges,
            (near_bp, far_bp),
            encoding,
            n=n,
            k=k,
            t_partner=t_partner,
        )
        root = write_variant_table(path, variants, ranges)
        return cls(root, cutter, ranges, compress_workers=compress_workers)

    @classmethod
    def join(
        cls, location: Path, chromosomes: Sequence[Path], sourced: Sourced, *, threads: int = 1
    ) -> Provenance:
        """Join closed single-chromosome stores without recomputing a code (ADR 0015).

        This additive import path lets a producer discover diagonal losses one chromosome
        at a time, whereas `create` deliberately fixes a genotype build's Variant Set up
        front. Inputs remain immutable and restartable. Component files are hard-linked
        on the same filesystem; the final store owns independent variant/index metadata.
        As for `create`, a pre-existing output is refused and `finish` validates every rule
        before claiming a store. No genotype build calls this path.
        """
        from ogld.store.joining import joined

        root, cutter, ranges, tally = joined(location, chromosomes, sourced)
        writer = cls(root, cutter, ranges)
        writer._tally = tally
        writer._next = writer._rows
        return writer.finish(sourced, threads=threads)

    @property
    def encoding(self) -> REncoding:
        """The store's encoding, as `create` was given it: read-only."""
        return self._cutter.encoding

    @property
    def far_bp(self) -> int:
        """The distance up to which every block's r must reach, as `create` was given it."""
        return self._cutter.far_bp

    def cut(self, first: int, r: npt.NDArray[np.floating[Any]]) -> Cut:
        """Cut the block of rows `first`, `first + 1`, … `first + len(r) − 1` (global rows, one
        chromosome) from their r, ready to `write` (spec §3–§5).

        `r[t, c]` is r(first + t, first + c), signed to A1 after PC Adjustment (spec §3). It runs
        for every column c up to the last variant within far_bp of the block's last row, and no
        further: `r.shape[1]` is that variant's row + 1 − first. That is what
        `ogld.genotypes.r(rows, span)` gives for the block against those rows, and the
        chromosome's whole matrix for a block of all its rows. Only each row's later partners
        within far_bp are read. On and below the diagonal, and past a row's own far stop, `r` may
        hold anything, NaN included. Any floating dtype is accepted.

        It quantises every pair within near_bp at the store's encoding, for the Near Band, and
        keeps every pair within far_bp whose code reaches the far code floor, for the Far Pairs
        and the Partner Rows. It refuses a block that leaves its chromosome and an `r` of another
        width. It
        also refuses an r within a row's far_bp that is not a correlation (NaN, infinite or past
        ±1), naming the pair. So a pair the producer did not measure is refused, never stored
        and never read as a Bound.

        It is pure and thread-safe: it reads only what `create` fixed, so a producer cuts blocks
        on its own worker threads.
        """
        return self._cutter.cut(first, r)

    def write(self, cut: Cut) -> None:
        """Write a cut's rows. They continue where the last cut stopped, in global row order, so
        each chromosome is finished before the next begins.

        The Near Band's codes are written in whole shards of `near/<chrom>/r`, the rest carried;
        its `indptr` comes from the positions when the chromosome's first cut arrives (spec §4).
        The Far Pairs are written row by row, each in its earlier row (spec §5); the Partner Rows
        are framed as their rows complete. The counts are added up. When the chromosome's last
        row arrives, what is carried is written and the chromosome is complete.

        A write that fails — a frame that will not compress, a stream that will not close —
        poisons the writer: the rows are counted as written only after the chromosome closed, and
        every later `write` or `finish` raises naming this first failure. So a half-written
        chromosome is never finished over, and no provenance is ever written for it (issue #67).
        """
        self._live()
        if cut.by is not self._cutter:
            raise ValueError(
                "a cut is written only by the writer that cut it, not by another writer"
            )
        if cut.first != self._next:
            raise ValueError(
                f"cuts are written in row order: expected row {self._next}, got {cut.first}"
            )
        name, _, stop = self._cutter.chromosome(cut.first)
        try:
            self._write(cut, name, stop)
        except BaseException as failed:
            self._failure = (name, failed)
            raise

    def _write(self, cut: Cut, name: str, stop: int) -> None:
        """Write one cut's rows, and close the chromosome when it is its last."""
        if self._open is None:
            self._open = Chromosome(
                self._root, self._cutter, name, compress_workers=self._compress_workers
            )
        self._open.add(cut)
        self._tally.add(cut)
        ends = self._next + cut.rows
        if ends == stop:
            self._tally.partner_entries += self._open.close()
            self._open = None
        self._next = ends

    def _live(self) -> None:
        """Refuse a writer that already failed, naming the chromosome and the first failure."""
        if self._failure is not None:
            name, failed = self._failure
            raise ValueError(
                f"chromosome {name} could not be written, so this writer is never finished: "
                f"{failed!r}"
            ) from failed

    def finish(self, sourced: Sourced, *, threads: int = 1) -> Provenance:
        """Assemble the provenance, validate the store and write it; return it as written
        (spec §7, §12).

        `sourced` is what only the producer knows (`ogld.format.Sourced`). The writer adds what
        the store derives from itself: `format_version`, `encoding`, `n_variants`, `near_bp`,
        `far_bp`, `t_far`, `t_far_q`, `far_code_floor` and `counts`. It refuses, naming the
        chromosome, a store with rows not written. It also refuses a `sourced` with a key of the
        store's or an n or k that is not what `create` thresholded with. Then it runs every rule
        of spec §12 on `threads` workers and raises `ValidationFailed` if any fails, before the
        provenance is written: a store without one is never read.
        """
        self._live()
        if self._next != self._rows:
            unwritten = [name for name, (_, stop) in self._ranges.items() if stop > self._next]
            raise ValueError(
                f"the rows of chromosome {', '.join(unwritten)} were not written, from row "
                f"{self._next}"
            )
        # The ALID index is the store's, from the table it wrote (spec §2; ADR 0010): no producer
        # supplies or orders one, and a store is unreadable until `finish` writes its provenance.
        variants = self._root["variants"]
        assert isinstance(variants, zarr.Group)
        left_out = derive_alid_index(variants, variants.create_group("alid_index"))
        made = provenance(sourced, self._cutter, self._rows, self._tally, alids_left_out=left_out)
        recorded = _as_json(made)
        findings = validate_group(self._root, recorded, threads=threads)
        if any(findings.values()):
            raise ValidationFailed(findings)
        self._root.attrs.update({"ld_store": recorded})
        path = layout.local_path(self._root)
        assert path is not None
        derive_alid_mmap(path)
        return cast(Provenance, recorded)


def write_variant_table(
    location: Path, variants: Variants, ranges: dict[str, tuple[int, int]]
) -> zarr.Group:
    """Write the shared variant half, without LD components or a provenance claim (#91).

    The LD writer and Genotype Store builder use exactly the same columns and layout. `ranges`
    is the chromosome record its caller computed (`chromosome_ranges`), so the write path has
    one source for it, not two. Index derivation follows once the producer has completed its own
    arrays.
    """
    position = _positions(variants.position)
    root = zarr.open_group(str(location), mode="w-", zarr_format=3)
    group = root.create_group("variants")
    layout.put_numbers(group, "position", position, layout.POSITION_FILL)
    layout.put_strings(group, "alid", variants.alid)
    layout.put_strings(group, "a1", variants.a1)
    layout.put_strings(group, "a2", variants.a2)
    layout.put_numbers(
        group, "a1_frequency", variants.a1_frequency.astype(np.float32), layout.FREQUENCY_FILL
    )
    group.attrs["chromosomes"] = layout.chromosomes_record(ranges)
    return root


def _as_json(provenance: Provenance) -> dict[str, Any]:
    """The provenance as the JSON attribute will hold it, so what is validated is what is
    stored; a value JSON cannot hold, such as a numpy integer, is refused, not converted."""
    try:
        return cast(dict[str, Any], json.loads(json.dumps(provenance)))
    except TypeError as unwritable:
        raise ValueError(f"provenance must be plain JSON values: {unwritable}") from unwritable


def _positions(position: npt.NDArray[np.int64]) -> npt.NDArray[np.int32]:
    """Positions as the int32 array stores them; one that does not fit is refused, not wrapped."""
    if len(position) and (position.min() < 0 or position.max() > np.iinfo(np.int32).max):
        raise ValueError("a position does not fit the int32 `position` array (spec §2)")
    return position.astype(np.int32)
