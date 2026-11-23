"""Copies of the Test Store for tests that must change one: re-sharded, or with a piece lost;
and `snv_variants`, the Variant Set of a small store a test writes itself.

No test writes into `tests/ld/data/`; each works on a copy under its own `tmp_path`.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import zarr
from zarr.codecs import ZstdCodec
from zarr.storage import LocalStore

from opengwasdb.ld.format import REncoding, Sourced
from opengwasdb.ld.store import StoreReader, StoreWriter, Variants
from opengwasdb.variants.sorted_spelling import sorted_identity

TEST_STORE = Path(__file__).resolve().parent / "data" / "test-panel" / "store"
#: The Test Store's encoding, r̂ = code / R_SCALE, as a literal: every test that needs the Test
#: Store's step, or builds the Test Panel to compare with it, takes it from here, never from
#: ogld (ld-store §3; ADR 0090). A test holds the store's record to it.
R_SCALE = 511
#: Half a step: |r̂ − r| never exceeds it.
HALF_STEP = 1 / (2 * R_SCALE)
#: The far code floor at R_SCALE for the Test Panel's n = 771, all of EUR's v2 members by the
#: owner's ruling, where a panel's build keeps 665 (ld-store §3's table): FLOOR at the Test Store's
#: k = 10, FLOOR_K0 at k = 0, and FLOOR_K7 at k = 7, where a build that drops a local PC lands.
#: At × 511 the three t_far (0.0274211, 0.0270649, 0.0273132) share a floor; at × 32767 they
#: were 5426, 5391 and 5416.
FLOOR = 85
FLOOR_K0 = 85
FLOOR_K7 = 85

#: The Test Store's Partner Threshold, and its Partner Floor at R_SCALE: the smallest code c ≥ 0
#: with (c / 511)² ≥ 0.01, computed here rather than read from the store. 0.01 is the owner's for
#: the HGDP+1kGP panels, below every panel's t_far (partner-rows §3, §11).
T_PARTNER = 0.01
PARTNER_FLOOR = 52


def snv_variants(chrom: list[str], position: npt.NDArray[np.int64]) -> Variants:
    """An A/C SNV at each of `chrom` and `position`, identified as an LD build identifies a
    record (`sorted_identity`), each A1 at frequency 0.3."""
    oriented = [sorted_identity(c, int(p), "A", "C") for c, p in zip(chrom, position, strict=True)]
    return Variants(
        np.array(chrom),
        position,
        np.array([o.alid for o in oriented]),
        np.array([o.a1 for o in oriented]),
        np.array([o.a2 for o in oriented]),
        np.full(len(oriented), 0.3),
    )


def codes_at(r: npt.ArrayLike, scale: int = R_SCALE) -> npt.NDArray[np.int64]:
    """r as a store at × `scale` codes it, computed here rather than by ogld: ld-store §3's
    clamp(rint(r × scale), −scale, +scale), half to even."""
    return np.clip(np.rint(np.asarray(r, dtype=np.float64) * scale), -scale, scale).astype(np.int64)


def stored_at(r: npt.ArrayLike, scale: int = R_SCALE) -> npt.NDArray[np.float64]:
    """r as a store at × `scale` reads it back: its code / scale."""
    return codes_at(r, scale) / scale


@contextlib.contextmanager
def store_reads() -> Iterator[list[str]]:
    """Every key a reader fetches from a local store while the block runs (opengwas/ogld#71).

    That is each array's chunks and shard indexes, and every `zarr.json` metadata document, by the
    store's own key: `/c/` is followed by coordinates, so a predicate on the key's ending matches
    nothing. A test holds that `clump` and `proxies` read none of `near/` or `index/` with it.
    """
    keys: list[str] = []
    get, get_sync = LocalStore.get, LocalStore.get_sync

    async def recording_get(
        self: Any, key: str, prototype: Any = None, byte_range: Any = None
    ) -> Any:
        keys.append(key)
        return await get(self, key, prototype=prototype, byte_range=byte_range)

    def recording_get_sync(
        self: Any, key: str, prototype: Any = None, byte_range: Any = None
    ) -> Any:
        keys.append(key)
        return get_sync(self, key, prototype=prototype, byte_range=byte_range)

    with (
        patch.object(LocalStore, "get", recording_get),
        patch.object(LocalStore, "get_sync", recording_get_sync),
    ):
        yield keys


def copy_store(to: Path) -> Path:
    """A plain copy of the committed Test Store at `to`."""
    shutil.copytree(TEST_STORE, to)
    return to


def arrays(group: zarr.Group, prefix: str = "") -> Iterator[tuple[str, zarr.Array[Any]]]:
    """Every array under `group`, by its path."""
    for name, member in group.members():
        if isinstance(member, zarr.Array):
            yield prefix + name, member
        else:
            yield from arrays(member, f"{prefix}{name}/")


def resharded(to: Path, inner: int, per_shard: int) -> Path:
    """The Test Store rewritten with inner chunks of `inner` elements and `per_shard` of them
    to a shard: the same data and metadata on another layout, as ogld#9 may serve it (ld-store §6).
    """
    copy_store(to)
    source = zarr.open_group(TEST_STORE, mode="r")
    target = zarr.open_group(to, mode="r+")
    for path, array in arrays(source):
        if path.startswith("partners/"):
            # A Partner Rows group's `frames` is one chunk of its whole file and `frames_idx`
            # one chunk of its offsets (partner-rows §5.1): how a frame is one byte range. The
            # layout this rewrites is the Near Band's, so these are copied as they are.
            continue
        data = np.asarray(array[:])
        group, name = path.rsplit("/", 1)
        shutil.rmtree(to / path)
        chunk = min(inner, max(1, len(data)))
        shard = min(chunk * per_shard, -(-max(1, len(data)) // chunk) * chunk)
        made = zarr.open_group(to / group, mode="r+").create_array(
            name,
            shape=data.shape,
            dtype=str if data.dtype.kind in "OT" else data.dtype,
            chunks=(chunk,),
            shards=(shard,),
            compressors=[ZstdCodec(level=3)],
            fill_value=array.fill_value,
        )
        made[:] = data
    assert dict(target.attrs) == dict(source.attrs)
    return to


def unpartnered(to: Path) -> Path:
    """A copy of the committed Test Store as the published 0.1.0 stores are: `partners/` absent,
    the four keys that describe it dropped, and the Proxy Index a 0.1.0 store holds with its
    `counts.index_entries` (partner-rows §6.2, §13; ld-store §13).

    It is what `opengwasdb ld derive-partners` derives into, so a test holds the derived component
    against the committed one — the build's — byte for byte, and `--drop-index` has something to
    remove. The index's arrays are empty placeholders: nothing reads a Proxy Index since
    opengwas/ogld#72, and a test that reads one is testing the component this fixture stands for,
    not its data.
    """
    copy_store(to)
    shutil.rmtree(to / "partners")
    root = zarr.open_group(to, mode="r+")
    recorded = dict(root.attrs["ld_store"])
    counts = dict(recorded["counts"])
    software = dict(recorded["software"])
    counts.pop("partner_entries", None)
    counts["index_entries"] = 36_476
    software.pop("zstd", None)
    recorded.update({"format_version": "0.1.0", "counts": counts, "software": software})
    for key in ("t_partner", "partner_code_floor"):
        recorded.pop(key, None)
    root.attrs.update({"ld_store": recorded})
    for entry in root["variants"].attrs["chromosomes"]:
        rows = entry["row_stop"] - entry["row_start"]
        group = root.create_group(f"index/{entry['name']}")
        group.create_array(
            "indptr",
            shape=(rows + 1,),
            dtype=np.int64,
            chunks=(max(1, rows + 1),),
            shards=(max(1, rows + 1),),
            compressors=[ZstdCodec(level=3)],
            fill_value=-1,
        )[:] = np.zeros(rows + 1, dtype=np.int64)
        for name, dtype, fill in (("partner", np.int32, -1), ("r", np.int16, -32768)):
            group.create_array(
                name,
                shape=(0,),
                dtype=dtype,
                chunks=(1,),
                shards=(1,),
                compressors=[ZstdCodec(level=3)],
                fill_value=fill,
            )
    return to


#: The store-derived keys `finish` adds (ld-store §7), every other key being the producer's.
DERIVED = frozenset(
    {
        "format_version",
        "encoding",
        "n_variants",
        "near_bp",
        "far_bp",
        "t_far",
        "t_far_q",
        "far_code_floor",
        "t_partner",
        "partner_code_floor",
        "counts",
    }
)


def sourced(**changes: Any) -> Sourced:
    """The Test Store's provenance without the keys the store derives, with `changes`."""
    p: dict[str, Any] = json.loads(json.dumps(StoreReader.open(TEST_STORE).provenance))
    for key in DERIVED:
        del p[key]
    p.update(changes)
    return cast(Sourced, p)


def private_sidecar_copy(
    genotype_store: Path, tmp_path: Path
) -> tuple[Path, zarr.Group, dict[str, Any]]:
    """A private copy of a Genotype Store whose pins point at the original inputs (relative paths
    survive the copy into another directory), its ALID companion dropped (a store without one is
    not a fixture for this), with its root group and its genotype record, whose edits the caller
    writes back."""
    copy = tmp_path / "sidecar"
    shutil.copytree(genotype_store, copy)
    root = zarr.open_group(copy, mode="a")
    record = dict(root.attrs["genotype_store"])
    for fileset in record["filesets"]:
        for name in ("pgen", "psam", "pvar"):
            pin = fileset[name]
            pin["path"] = os.path.relpath((genotype_store / pin["path"]).resolve(), copy)
    for pin in record["inputs"]:
        pin["path"] = os.path.relpath((genotype_store / pin["path"]).resolve(), copy)
    del root.attrs["alid_mmap"]
    return copy, root, record


def writer_for_test_store(path: Path, variants: Variants, **more: Any) -> StoreWriter:
    """A writer with the Test Store's geometry and encoding: near_bp 3 Mb, far_bp 10 Mb, n = 771,
    k = 10, the Test Store's scale and `T_PARTNER`; `more` adds or replaces arguments."""
    arguments: dict[str, Any] = {
        "encoding": REncoding(R_SCALE),
        "near_bp": 3_000_000,
        "far_bp": 10_000_000,
        "n": 771,
        "k": 10,
        "t_partner": T_PARTNER,
    }
    return StoreWriter.create(path, variants=variants, **{**arguments, **more})
