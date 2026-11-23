"""How an LD Store's Zarr arrays are laid out (ld-store §6): the one place that creates them.

Every array is Zarr v3 with `sharding_indexed`, except the Partner Rows' two: those are one chunk
each with the `bytes` codec alone, because their bytes are the frames themselves and a reader
addresses one frame's byte range (partner-rows §5.1; ADR 0097). The chunk and shard shapes, the
codec chain and the fill value of each kind of array are the spec's table, kept here as data so the
writer cannot create an array the table does not describe, and the reader's tests can name the fill
values it promises.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from os import stat_result
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import zarr
from zarr.codecs import ZstdCodec
from zarr.storage import LocalStore, StorePath, WrapperStore

from opengwasdb.ld.format import InputError, REncoding

#: (inner chunk, shard) in elements, for every array but the Near Band's `r`.
SIDE: Final = (16_384, 1_048_576)
#: The Near Band's `r`: layout C (ld-store §6; ADR 0092).
NEAR_R: Final = (262_144, 67_108_864)


def _create_array(group: zarr.Group, name: str, **spec: Any) -> zarr.Array[Any]:
    """`group.create_array`, the one call every array here is made by, with how it is written
    fixed: a chunk holding only the fill value is left out and reads as the fill value, as zarr's
    default had it for every Store ogld built. `import opengwasdb` sets zarr's process-wide
    `array.write_empty_chunks` to True for the GWAS Store (opengwasdb/store/arrays.py); an LD
    Store or Genotype Store array does not inherit it (#455)."""
    return group.create_array(name, config={"write_empty_chunks": False}, **spec)


#: The ALID search index (ld-store §2, §6; ADR 0096): the variant table's ALIDs in one array a
#: search can run over. A key is an ALID's UTF-8 bytes, NUL right-padded to `ALID_INDEX_WIDTH`; an
#: ALID wider than that is left out of the index and found by an exact scan at its position, never
#: truncated into it: two indels at one position that agree over the slot would then share a key,
#: and one would answer to the other's name (opengwasdb issue #127, its ALID width).
ALID_INDEX_WIDTH: Final = 32
#: The dtype string of one key's bytes (numpy's own fixed-width bytes cast pads with NUL).
_BYTES: Final = f"|S{ALID_INDEX_WIDTH}"
ALID_KEY_DTYPE: Final = np.dtype(_BYTES)
#: The index's group and its two arrays, by their path.
ALID_INDEX: Final = "variants/alid_index"
ALID_INDEX_ALID: Final = f"{ALID_INDEX}/alid"
ALID_INDEX_ROW: Final = f"{ALID_INDEX}/row"
#: Their (inner chunk, shard) in elements, as `SIDE` is. A key is 32 bytes, so an inner chunk of
#: 262,144 bytes holds 8,192 keys and a 16 MiB shard 524,288: the whole index is 275 MB on EUR, and
#: a search reads it one shard at a time. `docs/benchmarks/alid-index.md` measured this against
#: `SIDE`'s 16 KiB chunks, whose 16,693 of them made a whole read 4.6 s against 0.2 s.
ALID_KEYS: Final = (262_144, 16_777_216)
ALID_ROWS: Final = (262_144, 1_048_576)
#: The keys' fill value: no ALID is empty, so a whole key of NUL bytes is one no store holds. It is
#: 0 because every padding byte is NUL too, so a key is judged whole rather than byte by byte
#: (`refused_if_lost`), and a lost chunk reads as keys of no bytes.
ALID_KEY_FILL: Final = 0
#: The rows': a row is never negative.
ALID_ROW_FILL: Final = -1

#: Each array's fill value: one a valid LD Store never holds (ld-store §6). The `r` arrays' is
#: their encoding's reserved code and the strings' is ""; both are set where those arrays are
#: made.
INT64_FILL: Final = -1
PARTNER_FILL: Final = -1
POSITION_FILL: Final = 0
FREQUENCY_FILL: Final = float("nan")

#: The Partner Rows (partner-rows §5.1): one group per chromosome, holding two arrays of one chunk
#: each. `frames` is the chromosome's zstd frames back to back, and its codec chain is `bytes`
#: alone with no compressor, so its chunk file is those frames: a frame is one positional read.
#: `frames_idx` is the uint64 offsets that address them, little-endian and read whole.
PARTNERS: Final = "partners"
FRAMES: Final = "frames"
FRAMES_INDEX: Final = "frames_idx"
#: The file the frames are appended to until the group is closed, when it becomes the `frames`
#: chunk: the array's shape is the whole file's, so it is not known until the last frame (§5.1).
FRAMES_BUILDING: Final = "frames.bin"
#: The frames' fill value: 0, because a frame is a zstd frame and 0 begins none (ld-store §3). A
#: chunk a store lost is refused by the frame's own checksum and length, not by a fill value.
FRAMES_FILL: Final = 0
#: The offsets': the largest uint64, which no offset of a whole number of bytes is.
FRAMES_INDEX_FILL: Final = int(np.iinfo(np.uint64).max)
#: The dtype the offsets are written and hashed in: little-endian whatever this machine is,
#: because their chunk file is the offsets' bytes (partner-rows §5.1, §5.3). The writer, the
#: reader and the rule that hashes the chunk all take it from here.
FRAMES_INDEX_DTYPE: Final = np.dtype("<u8")


def never_data(path: str, values: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Where `values` of the array at `path`, such as `far/22/partner`, are ones no valid LD
    Store holds there (ld-store §2–§6). The array's kind is its name, the path's last component.

    Each array's fill value must be such a value, because a chunk the store lost reads as its
    fill value: then every read can tell a lost chunk from data and refuse it. The reader
    refuses an array whose fill value is not, and `envelope.fill` fails it (ld-store §12).
    """
    v = np.asarray(values)
    name = path.rsplit("/", 1)[-1]
    if path == ALID_INDEX_ALID:
        # A key is judged 32 bytes at a time, never byte by byte: a padding byte is 0, which its
        # fill value is too, so no single byte can be the one no store holds. `refused_if_lost`
        # judges a whole key; the fill must be 0, or a lost chunk would read as a run of keys.
        return np.asarray(v == ALID_KEY_FILL)
    if name == "r":
        # Every admitted encoding shares the container and its reserved code (ADR 0090), so a
        # lost chunk is told from data before the store's own encoding is known.
        return np.asarray(v == REncoding.reserved_code)
    if name in {"indptr", "partner", "row"}:
        return np.asarray(v < 0)
    if name == "position":
        return np.asarray(v <= 0)
    if name in {"alid", "a1", "a2"}:
        return np.asarray(v == "")
    if name == "a1_frequency":
        with np.errstate(invalid="ignore"):
            return np.asarray(~((v > 0) & (v < 1)))
    raise KeyError(f"{name} is not an array of an LD Store (ld-store §1)")


def shown_fill(fill: object) -> str:
    """A fill value as a message names it: a string quoted, so an empty one shows."""
    return f'"{fill}"' if isinstance(fill, str) else str(fill)


def refused_if_lost(array: zarr.Array[Any], values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """`values` read from `array`, unless one is a value no LD Store holds there, as its fill
    value is: that is a chunk the store lost, and decoding it would be inventing data."""
    if array.path == ALID_INDEX_ALID:
        return _refused_if_lost_keys(array, values)
    lost = never_data(array.path, values)
    if lost.any():
        raise ValueError(
            f"{array.path} read {int(lost.sum())} element(s) no LD Store holds there, as its "
            f"fill value {shown_fill(array.fill_value)} is: a chunk the store lost, never data "
            "(ld-store §3, §6)"
        )
    return values


def _refused_if_lost_keys(array: zarr.Array[Any], values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """The keys `values` of `variants/alid_index/alid`, unless one is all NUL bytes: no ALID is
    empty, so that is a chunk the store lost rather than a key (ld-store §6)."""
    keys = np.asarray(values, dtype=np.uint8).reshape(-1, ALID_INDEX_WIDTH)
    lost = ~keys.any(axis=1)
    if lost.any():
        raise ValueError(
            f"{array.path} read {int(lost.sum())} key(s) of no bytes at all, as its fill value "
            f"{shown_fill(array.fill_value)} is: a chunk the store lost, never data (ld-store §6)"
        )
    return values


def alid_fits(alids: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Whether each ALID fits the index's slot, which is the one rule for what the index holds
    (ld-store §2; ADR 0096): the engine's own, an ALID whose UTF-8 form is at most
    `ALID_INDEX_WIDTH` bytes (`is_indexable_alid`). A wider ALID is left out of the index, never
    truncated into it, and is found by the exact scan at its position instead."""
    text = np.asarray(alids)
    encoded = np.char.encode(text, "utf-8") if len(text) else np.empty(0, dtype="S1")
    lengths = np.strings.str_len(encoded) if len(text) else np.empty(0, dtype=np.int64)
    return np.asarray(lengths <= ALID_INDEX_WIDTH)


def alid_keys(
    alids: npt.ArrayLike, first: int = 0
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """The key of each ALID of `alids` that fits the slot, and its row, in the order given (ld-store
    §2; ADR 0096). An ALID wider than `ALID_INDEX_WIDTH` bytes has no key, and its row is not
    left out of the count here: `derive_alid_index` counts them from `n_variants` and the keys.

    `first` is the Variant Set row of `alids[0]`, so a block of a store's table gives the rows
    the whole table's index has. The padding is numpy's own `|S32` cast, which is the engine's
    (ADR 0096), and nothing is truncated: every ALID that reaches it fits.
    """
    text = np.asarray(alids)
    # UTF-8 first, so a key is the ALID's bytes however wide the characters are; the cast to the
    # slot then only pads, because the wider ones are already gone.
    encoded = np.char.encode(text, "utf-8") if len(text) else np.empty(0, dtype="S1")
    fits = alid_fits(text)
    keys = np.array(encoded[fits], dtype=_BYTES)
    return (
        keys.view(np.uint8).reshape(-1, ALID_INDEX_WIDTH),
        np.flatnonzero(fits).astype(np.int64) + first,
    )


def ordered_index(
    keys: npt.NDArray[np.uint8], rows: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """`keys` and `rows` in the engine's order (ADR 0096): by key bytewise, ties by row, which
    is what its stable sort of the padded keys gives."""
    order = np.argsort(keys.view(f"|S{ALID_INDEX_WIDTH}").ravel(), kind="stable")
    return keys[order], rows[order]


def alid_index(
    alids: npt.ArrayLike,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """The ALID index of one Variant Set's `alids`: its keys, and the row of each, in the
    engine's order (ld-store §2, §6; ADR 0096). The one place the width, the padding and the sort
    are written; a store writes what this gives, and validation computes it too."""
    return ordered_index(*alid_keys(alids))


def put_alid_index(
    index: zarr.Group, keys: npt.NDArray[np.uint8], rows: npt.NDArray[np.int64]
) -> None:
    """Write the index's two arrays into the group `index`: the keys' bytes, and each entry's
    Variant Set row (ld-store §6)."""
    put_numbers(index, "alid", keys.reshape(-1), ALID_KEY_FILL, ALID_KEYS)
    put_numbers(index, "row", rows.astype(np.int32), ALID_ROW_FILL, ALID_ROWS)


#: The keys of one record of `variants.attrs["chromosomes"]` (ld-store §2).
CHROMOSOME_KEYS: Final = ("name", "row_start", "row_stop")


def chromosomes_record(ranges: dict[str, tuple[int, int]]) -> list[dict[str, Any]]:
    """The `variants` group's `chromosomes` attribute for these ranges, in order (ld-store §2)."""
    return [
        dict(zip(CHROMOSOME_KEYS, (name, start, stop), strict=True))
        for name, (start, stop) in ranges.items()
    ]


def _is_row(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _chromosome_entry(entry: object) -> tuple[str, int, int]:
    """One `chromosomes` entry as (name, row_start, row_stop), or ValueError."""
    if not isinstance(entry, dict) or set(entry) != set(CHROMOSOME_KEYS):
        raise ValueError(f"a chromosomes entry does not have exactly the keys {CHROMOSOME_KEYS}")
    name, start, stop = (entry[key] for key in CHROMOSOME_KEYS)
    if not (isinstance(name, str) and _is_row(start) and _is_row(stop) and 0 <= start <= stop):
        raise ValueError(f"a chromosomes entry is not a name and rows 0 <= start <= stop: {entry}")
    return name, start, stop


def chromosomes_listed(record: object) -> list[tuple[str, int, int]]:
    """The (name, row_start, row_stop) of each entry of a `chromosomes` attribute, in order and
    with repeats kept, so a rule can see them. A record that is not a list of entries with
    exactly these keys, a string name and integer bounds 0 ≤ start ≤ stop raises ValueError."""
    if not isinstance(record, list):
        raise ValueError("the chromosomes attribute is not a list")
    return [_chromosome_entry(entry) for entry in record]


def clipped(length: int, inner: int, shard: int) -> tuple[int, int]:
    """The inner chunk and shard of an array of `length`: an inner chunk longer than the array
    is clipped to it, and a shard to the smallest whole number of inner chunks covering it."""
    inner = min(inner, max(1, length))
    shard = min(shard, -(-max(1, length) // inner) * inner)
    return inner, shard


def partners_path(chrom: str) -> str:
    """The Partner Rows group of one chromosome (partner-rows §5.1)."""
    return f"{PARTNERS}/{chrom}"


def new_frames_array(group: zarr.Group, length: int) -> zarr.Array[Any]:
    """The `frames` array: `uint8`, one chunk of its whole length, the `bytes` codec alone and no
    compressor, because its bytes are already zstd frames (partner-rows §5.1)."""
    if length < 1:
        raise ValueError("a Partner Rows' frames are at least one frame's bytes, never none")
    return _create_array(
        group,
        FRAMES,
        shape=(length,),
        dtype=np.uint8,
        chunks=(length,),
        compressors=[],
        fill_value=FRAMES_FILL,
    )


def new_frames_index(group: zarr.Group, length: int) -> zarr.Array[Any]:
    """The `frames_idx` array: the F + 1 uint64 byte offsets of the frames, one chunk of its
    whole length, little-endian in a `bytes` codec alone so the file is the offsets (§5.1)."""
    return _create_array(
        group,
        FRAMES_INDEX,
        shape=(length,),
        dtype=FRAMES_INDEX_DTYPE,
        chunks=(length,),
        compressors=[],
        fill_value=FRAMES_INDEX_FILL,
    )


def local_root(store_path: StorePath) -> Path:
    """The local directory a store at `store_path` is rooted at, refused for a store that is not
    one: a one-chunk array's chunk file is a file, which a remote store has none of (partner-rows
    §5.1).

    A reader's store is wrapped by the cache (`location._KeptIndexes`), so the wrapper is seen
    through: the file is the same one, and a local frame is read from it with `os.pread` (§7).
    """
    store = store_path.store
    while isinstance(store, WrapperStore):
        store = store._store
    if not isinstance(store, LocalStore):
        raise ValueError(
            f"the Partner Rows are written to a local directory, and {type(store).__name__} is "
            "not one (partner-rows §5.1)"
        )
    return Path(store.root)


def local_directory(store_path: StorePath) -> Path:
    """The local directory a store path names, refused for a store that is not local: the Partner
    Rows' frames are appended to a file, which a remote store has none of (partner-rows §5.1)."""
    return local_root(store_path) / store_path.path


def chunk_key(array: zarr.Array[Any]) -> str:
    """The key an array's whole chunk is stored under, resolved from the array's own metadata and
    chunk key encoding, never assumed to be `c/0` (partner-rows §5.1): what a reader fetches a
    frame's byte range by, and what a writer puts the file at."""
    return f"{array.store_path.path}/{array.metadata.encode_chunk_key((0,))}"


def chunk_file(array: zarr.Array[Any]) -> Path:
    """The local file `array`'s whole chunk is written to, resolved from the array's own metadata
    and its chunk key encoding, never assumed to be `c/0` (partner-rows §5.1)."""
    return local_root(array.store_path) / chunk_key(array)


def put_numbers(
    group: zarr.Group,
    name: str,
    data: npt.NDArray[np.generic],
    fill: float | int,
    shape: tuple[int, int] = SIDE,
) -> zarr.Array[Any]:
    """Create the numeric array `name` (bytes, then zstd level 3) and write all of `data`."""
    array = new_numbers(group, name, len(data), data.dtype, fill, shape)
    if len(data):
        array[:] = data
    return array


def new_numbers(
    group: zarr.Group,
    name: str,
    length: int,
    dtype: npt.DTypeLike,
    fill: float | int,
    shape: tuple[int, int] = SIDE,
) -> zarr.Array[Any]:
    """A numeric array `name` of `length` with nothing written: every chunk reads `fill`."""
    inner, shard = clipped(length, *shape)
    return _create_array(
        group,
        name,
        shape=(length,),
        dtype=dtype,
        chunks=(inner,),
        shards=(shard,),
        compressors=[ZstdCodec(level=3)],
        fill_value=fill,
    )


def put_strings(group: zarr.Group, name: str, data: npt.NDArray[np.str_]) -> zarr.Array[Any]:
    """Create the string array `name` (vlen-utf8, then zstd level 0) and write all of `data`."""
    inner, shard = clipped(len(data), *SIDE)
    array = _create_array(
        group,
        name,
        shape=data.shape,
        dtype=str,
        chunks=(inner,),
        shards=(shard,),
        fill_value="",
    )
    if len(data):
        array[:] = data
    return array


def new_r_array(
    group: zarr.Group, name: str, length: int, shape: tuple[int, int], encoding: REncoding
) -> zarr.Array[Any]:
    """An `r` array of `length` in `encoding`'s dtype with nothing written: every chunk reads
    the encoding's reserved code (ld-store §3, §6)."""
    inner, shard = clipped(length, *shape)
    return _create_array(
        group,
        name,
        shape=(length,),
        dtype=encoding.dtype,
        chunks=(inner,),
        shards=(shard,),
        compressors=[ZstdCodec(level=3)],
        fill_value=encoding.reserved_code,
    )


# ADR 0098: an optional local representation of the canonical index, never another ordering.
ALID_MMAP: Final = "variants/alid_mmap"
ALID_MMAP_ATTR: Final = "alid_mmap"
ALID_MMAP_SCHEMA: Final = 2
ALID_MMAP_KEYS: Final = "keys.bin"
ALID_MMAP_ROWS: Final = "rows.i32le"
ALID_MMAP_MANIFEST: Final = "manifest.json"
ALID_MMAP_ROW_DTYPE: Final = "<i4"


@dataclass(frozen=True)
class AlidPayload:
    """One companion payload's canonical source and physical representation (ADR 0098)."""

    name: str
    source: str
    dtype: np.dtype[Any]
    width: int


ALID_MMAP_PAYLOADS: Final = (
    AlidPayload(ALID_MMAP_KEYS, ALID_INDEX_ALID, np.dtype("u1"), ALID_INDEX_WIDTH),
    AlidPayload(ALID_MMAP_ROWS, ALID_INDEX_ROW, np.dtype(ALID_MMAP_ROW_DTYPE), 1),
)


def local_path(root: zarr.Group) -> Path | None:
    """The local hierarchy behind wrappers, or None for HTTP (ADR 0098)."""
    store = root.store
    while isinstance(store, WrapperStore):
        store = store._store
    return Path(store.root) / root.path if isinstance(store, LocalStore) else None


def json_bytes(document: object) -> bytes:
    """Deterministic companion JSON; source metadata is normalized the same way (ADR 0098)."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def alid_mmap_sources(path: Path) -> dict[str, str]:
    """Pin canonical content, not just metadata: a foreign companion must refuse (ADR 0098)."""
    root = json.loads((path / "zarr.json").read_bytes())
    root["attributes"].pop(ALID_MMAP_ATTR, None)
    sources = {"zarr.json": hashlib.sha256(json_bytes(root)).hexdigest()}
    for payload in ALID_MMAP_PAYLOADS:
        # Metadata is mandatory even when a whole array directory was lost. Include every
        # other file too, so deleting, replacing or adding a shard changes the pinned set.
        metadata = path / payload.source / "zarr.json"
        if not metadata.is_file():
            raise ValueError(f"{metadata}: canonical metadata is missing or not a regular file")
        sources[str(metadata.relative_to(path))] = file_sha256(metadata)
        for source in _alid_index_files(path / payload.source):
            if source != metadata:
                sources[str(source.relative_to(path))] = file_sha256(source)
    return sources


def _alid_index_files(directory: Path) -> Iterator[Path]:
    """Follow local shard links without silently missing files or looping (ADR 0098)."""
    seen = set()
    for parent, directories, files in directory.walk(
        on_error=_source_walk_error, follow_symlinks=True
    ):
        identity = file_identity(parent.stat())[:2]
        if identity in seen:
            raise ValueError(f"{parent}: repeated canonical directory (cycle or alias)")
        seen.add(identity)
        directories.sort()
        for name in sorted(files):
            source = parent / name
            if not source.is_file():
                raise ValueError(f"{source}: canonical source is not a regular file")
            yield source


def _source_walk_error(error: OSError) -> None:
    raise error


def alid_mmap_source_change(before: dict[str, str], after: dict[str, str]) -> str | None:
    """The concrete source file whose content or membership changed (ADR 0098)."""
    return next(
        (
            name
            for name in sorted(before.keys() | after.keys())
            if before.get(name) != after.get(name)
        ),
        None,
    )


def alid_mmap_pointer(raw: bytes) -> dict[str, str]:
    """The one generation digest and publication document (ADR 0098)."""
    return {"generation": hashlib.sha256(raw).hexdigest()}


def alid_mmap_generation(pointer: Any) -> str:
    """Refuse an ambiguous pointer before using it as a directory name (ADR 0098)."""
    if not isinstance(pointer, dict) or set(pointer) != {"generation"}:
        raise ValueError("publication pointer must name one generation")
    generation = pointer["generation"]
    if not isinstance(generation, str) or re.fullmatch("[0-9a-f]{64}", generation) is None:
        raise ValueError("generation must be a SHA-256 digest")
    return generation


def alid_mmap_error(path: Path, reason: object) -> InputError:
    """A generation's full file path identifies the Store as well as the failed input."""
    return InputError(f"{path}: invalid ALID companion: {reason}")


@dataclass(frozen=True)
class AlidCompanion:
    """Integrity/content-bound descriptor; the reader owns mapping and release (ADR 0098)."""

    path: Path
    count: int
    identity: tuple[tuple[str, tuple[int, ...]], ...]


def alid_companion(root: zarr.Group) -> AlidCompanion | None:
    """Check source content pins and payload hashes before trusting any mapped byte.

    Absence and HTTP retain ADR 0096. A present pointer is a commitment: never silently fall
    back from integrity/pin mismatches. Exact correspondence is checked at trusted publication
    and by validation, not here; a consistently re-hashed forgery is outside reader detection.
    """
    path = local_path(root)
    pointer = root.attrs.get(ALID_MMAP_ATTR)
    if path is None or ALID_MMAP_ATTR not in root.attrs:
        return None
    try:
        generation = alid_mmap_generation(pointer)
    except ValueError as error:
        raise alid_mmap_error(path / "zarr.json", f"generation {pointer!r}: {error}") from error
    manifest_path = path / ALID_MMAP / generation / ALID_MMAP_MANIFEST
    try:
        return _alid_companion(path, pointer)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise alid_mmap_error(manifest_path, error) from error


def _companion_document(raw: bytes, pointer: Any) -> dict[str, Any]:
    if alid_mmap_pointer(raw) != pointer:
        raise ValueError("manifest digest differs from published generation")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    schema = manifest["schema"]
    if type(schema) is int and schema == 1:
        raise ValueError("schema 1 is unsupported; re-derive with scripts/derive_alid_mmap.py")
    count = manifest["count"]
    if type(count) is not int or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if not isinstance(manifest["sources"], dict):
        raise ValueError("sources must pin canonical files by name")
    return manifest


def _companion_manifest(path: Path, pointer: Any) -> tuple[Path, dict[str, Any], int]:
    directory = path / ALID_MMAP / alid_mmap_generation(pointer)
    manifest = _companion_document((directory / ALID_MMAP_MANIFEST).read_bytes(), pointer)
    count = manifest["count"]
    changed = alid_mmap_source_change(manifest["sources"], alid_mmap_sources(path))
    if changed is not None:
        raise ValueError(f"canonical source {path / changed}: content pin differs")
    expected = alid_mmap_manifest(count, manifest["sources"], manifest["payloads"])
    if type(manifest["schema"]) is not int or manifest != expected:
        raise ValueError("schema or representation differs")
    return directory, manifest, count


def _alid_companion(path: Path, pointer: Any) -> AlidCompanion:
    directory, manifest, count = _companion_manifest(path, pointer)
    identities = []
    for descriptor in ALID_MMAP_PAYLOADS:
        payload = directory / descriptor.name
        stat = payload.stat()
        size = count * descriptor.width * descriptor.dtype.itemsize
        if stat.st_size != size or file_sha256(payload) != manifest["payloads"][descriptor.name]:
            raise ValueError(f"{payload}: length or content hash differs")
        identities.append((descriptor.name, file_identity(stat)))
    return AlidCompanion(directory, count, tuple(identities))


def file_identity(stat: stat_result) -> tuple[int, ...]:
    """The local companion file identity, including opened-descriptor checks (ADR 0098)."""
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def alid_mmap_manifest(
    count: int, sources: dict[str, str], payloads: dict[str, str]
) -> dict[str, Any]:
    """The one schema for a canonical index's deterministic companion (ADR 0098)."""
    if set(payloads) != {ALID_MMAP_KEYS, ALID_MMAP_ROWS}:
        raise ValueError("manifest must hash both payloads")
    return {
        "schema": ALID_MMAP_SCHEMA,
        "count": count,
        "key_dtype": _BYTES,
        "row_dtype": ALID_MMAP_ROW_DTYPE,
        "sources": sources,
        "payloads": payloads,
    }
