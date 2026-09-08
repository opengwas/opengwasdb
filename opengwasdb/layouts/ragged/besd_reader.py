"""BESD format reader — port of besdq.besd_reader for OpenGWASDB."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np


class SnpRecord(NamedTuple):
    row_idx: int
    chromosome: str
    snp_id: str
    bp: int
    a1: str | None
    a2: str | None
    freq: float | None


class ProbeRecord(NamedTuple):
    row_idx: int
    chromosome: str
    probe_id: str
    probe_bp: int
    gene: str | None
    orientation: str | None


class BESDMetadataError(ValueError):
    """An .esi/.epi metadata row could not be parsed (issue #130).

    The message names the source file and the 1-based physical line so the
    offending row can be located without re-scanning. These rows used to be
    skipped silently, and because row_idx is assigned by enumeration, every
    later row silently shifted against the `.besd` association file —
    orphaned associations with no error raised anywhere.
    """


_ESI_COLUMNS = "chr, snp_id, genetic distance, bp, a1, a2"
_EPI_COLUMNS = "chr, probe_id, genetic distance, probe_bp, gene, orientation"


def _iter_index_rows(
    path: str | Path,
    *,
    kind: str,
    expected_columns: str,
    min_fields: int,
    max_fields: int,
) -> Iterator[tuple[int, list[str]]]:
    """Shared ESI/EPI data-row seam (issue #130).

    Yields ``(line_no, fields)`` for every non-comment, non-blank line of a
    metadata index file. Comment and blank lines are still skippable; any
    other line that does not carry the field count the format defines raises
    :class:`BESDMetadataError` with file and line context instead of being
    dropped, which is what used to desynchronise row indices from the
    `.besd` association stream.
    """
    file_path = Path(path)
    with file_path.open() as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if not min_fields <= len(fields) <= max_fields:
                raise BESDMetadataError(
                    f"{file_path}:{line_no}: {kind} data row has "
                    f"{len(fields)} columns; expected {expected_columns}"
                )
            yield line_no, fields


def read_esi(esi_path: str | Path) -> list[SnpRecord]:
    """Read an .esi SNP index file into SnpRecord rows.

    A data row must carry 6 columns (chr, snp_id, genetic distance, bp, a1,
    a2) or 7 with a trailing frequency. Comment and blank lines are
    ignored. A missing frequency column and a literal ``NA`` frequency both
    mean the frequency is unknown and parse to ``None`` (issue #130:
    absence is not a number). Any other malformed data row raises
    :class:`BESDMetadataError` naming the file and line — a silently
    skipped row would shift every later row index against the `.besd`
    association file.
    """
    snps: list[SnpRecord] = []
    for line_no, fields in _iter_index_rows(
        esi_path,
        kind="ESI",
        expected_columns=f"6 or 7 ({_ESI_COLUMNS}[, frequency])",
        min_fields=6,
        max_fields=7,
    ):
        try:
            bp = int(fields[3])
        except ValueError:
            raise BESDMetadataError(
                f"{esi_path}:{line_no}: ESI bp column {fields[3]!r} is not an "
                f"integer; expected {_ESI_COLUMNS}[, frequency]"
            ) from None
        freq: float | None = None
        if len(fields) > 6 and fields[6] != "NA":
            try:
                freq = float(fields[6])
            except ValueError:
                raise BESDMetadataError(
                    f"{esi_path}:{line_no}: ESI frequency column {fields[6]!r} "
                    f"is neither a number nor 'NA'"
                ) from None
            if not math.isfinite(freq):
                raise BESDMetadataError(
                    f"{esi_path}:{line_no}: ESI frequency column {fields[6]!r} "
                    f"is not a finite number"
                )
        snps.append(
            SnpRecord(
                row_idx=len(snps),
                chromosome=fields[0],
                snp_id=fields[1],
                bp=bp,
                a1=fields[4],
                a2=fields[5],
                freq=freq,
            )
        )
    return snps


def read_epi(epi_path: str | Path) -> list[ProbeRecord]:
    """Read an .epi probe index file into ProbeRecord rows.

    A data row must carry 6 columns: chr, probe_id, genetic distance,
    probe_bp, gene, orientation. Comment and blank lines are ignored. Any
    other malformed data row raises :class:`BESDMetadataError` naming the
    file and line rather than silently dropping the probe and shifting every
    later row index against the `.besd` association file (issue #130).
    """
    probes: list[ProbeRecord] = []
    for line_no, fields in _iter_index_rows(
        epi_path,
        kind="EPI",
        expected_columns=f"exactly 6 ({_EPI_COLUMNS})",
        min_fields=6,
        max_fields=6,
    ):
        try:
            probe_bp = int(fields[3])
        except ValueError:
            raise BESDMetadataError(
                f"{epi_path}:{line_no}: EPI probe_bp column {fields[3]!r} is "
                f"not an integer; expected {_EPI_COLUMNS}"
            ) from None
        probes.append(
            ProbeRecord(
                row_idx=len(probes),
                chromosome=fields[0],
                probe_id=fields[1],
                probe_bp=probe_bp,
                gene=fields[4],
                orientation=fields[5],
            )
        )
    return probes


_MAGIC_SPARSE_3F = 0x40400000
_MAGIC_SPARSE_3 = 3
_RESERVED_UNITS = 16


class BESDReader:
    """Read SPARSE_FILE_TYPE_3 and SPARSE_FILE_TYPE_3F BESD format files.

    Stores the entire file in numpy arrays for fast per-probe slicing.
    For very large datasets (>100M associations) consider streaming — see issue 036.
    """

    def __init__(self, besd_path: str | Path, n_probes: int):
        self._besd_path = Path(besd_path)
        self._n_probes = n_probes
        self.format_type: str = ""
        self._cols: np.ndarray | None = None
        self._rowid: np.ndarray | None = None
        self._val: np.ndarray | None = None
        self._val_num: int = 0
        self._load()

    def _load(self) -> None:
        with open(self._besd_path, "rb") as fh:
            magic = struct.unpack("<I", fh.read(4))[0]
            if magic == _MAGIC_SPARSE_3F:
                self.format_type = "3F"
                self._parse(fh, skip_reserved=False)
            elif magic == _MAGIC_SPARSE_3:
                self.format_type = "3"
                self._parse(fh, skip_reserved=True)
            else:
                raise ValueError(f"Unsupported BESD magic: 0x{magic:08x}")

    def _parse(self, fh, skip_reserved: bool) -> None:
        if skip_reserved:
            fh.read((_RESERVED_UNITS - 1) * 4)
        self._val_num = struct.unpack("<Q", fh.read(8))[0]
        col_num = (self._n_probes << 1) + 1
        self._cols = np.frombuffer(fh.read(col_num * 8), dtype=np.int64).copy()
        self._rowid = np.frombuffer(fh.read(self._val_num * 4), dtype=np.uint32).copy()
        self._val = np.frombuffer(fh.read(self._val_num * 4), dtype=np.float32).copy()

    def get_probe_associations(
        self, probe_idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (snp_indices, betas, ses) as numpy arrays for one probe.

        snp_indices: uint32 ESI row indices
        betas: float32
        ses: float32
        """
        if (
            probe_idx >= self._n_probes
            or self._cols is None
            or self._rowid is None
            or self._val is None
        ):
            empty = np.empty(0, dtype=np.uint32)
            return empty, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

        beta_start = int(self._cols[probe_idx << 1])
        se_start = int(self._cols[(probe_idx << 1) + 1])
        n = se_start - beta_start
        if n <= 0:
            empty = np.empty(0, dtype=np.uint32)
            return empty, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

        snp_idx = self._rowid[beta_start:se_start]
        betas = self._val[beta_start:se_start]
        ses = self._val[se_start: se_start + n]
        return snp_idx, betas, ses
