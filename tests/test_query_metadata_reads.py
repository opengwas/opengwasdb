"""A repeated query reads no array metadata from the store (#244).

zarr-python 3 reads an array's metadata from the store on every
``group[name]`` and every ``name in group`` -- about 1 ms each on a local
store, against ~0.1 ms of chunk work for a whole top-hit answer.  The
top-hit readers used to reopen the tier group and every field array on every
call, so a top-hit query on OGS-00009 spent most of its time on metadata
(21 ms as the reader did it, 5.8 ms with the arrays opened once); and a
Dense release with no `eaf` plane reopened its `z` plane for the grid width on
every regional query.

The cost a user pays is store reads, so that is what is counted: every
metadata-key read through zarr's ``LocalStore``.  Each layout's facade is
warmed with one call of every query shape, then the same calls are repeated
and must read none.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_hybrid_build import HG19_POS_1, HG19_POS_2, HG19_POS_3, _make_manifest, _make_vcf, _panel
from test_ragged_build_ssf import _make_fixture
from zarr.storage import LocalStore

from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.layouts.ragged.top_hits import build_ragged_top_hit_indexes
from opengwasdb.query import query_store

#: Store keys that hold Zarr metadata rather than chunk bytes, v2 and v3.
_METADATA_KEYS = (".zarray", ".zgroup", ".zattrs", ".zmetadata", "zarr.json")

#: Every fixture below has hits at this threshold, in every layout's index.
_THRESHOLD = 5e-4


@contextmanager
def _metadata_reads(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Record every metadata key read or probed through a LocalStore."""
    seen: list[str] = []

    def counting(name: str) -> None:
        original = getattr(LocalStore, name)

        async def wrapper(self: LocalStore, key: str, *args: Any, **kwargs: Any) -> Any:
            if key.endswith(_METADATA_KEYS):
                seen.append(f"{name}:{key}")
            return await original(self, key, *args, **kwargs)

        monkeypatch.setattr(LocalStore, name, wrapper)

    for name in ("get", "exists"):
        counting(name)
    yield seen
    monkeypatch.undo()


def _dense(tmp_path: Path, dense_store_path: Path) -> Path:
    return dense_store_path


def _ragged(tmp_path: Path, dense_store_path: Path) -> Path:
    manifest, filtered = _make_fixture(tmp_path)
    store = tmp_path / "ragged.opengwasdb"
    build_ragged_from_ssf(manifest, filtered, store, store_id="test", release_id="v1")
    build_ragged_top_hit_indexes(store)
    return store


def _hybrid(tmp_path: Path, dense_store_path: Path) -> Path:
    vcf1 = _make_vcf(
        tmp_path,
        "trait_a",
        [
            f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t2.0:0.5:0.2\n",
            f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE:AF\t1.5:0.3:0.3\n",  # off panel
            f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE:AF\t0.6:0.2:0.4\n",
        ],
    )
    vcf2 = _make_vcf(
        tmp_path,
        "trait_b",
        [
            f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t6.0:0.5:0.25\n",
            f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE:AF\t1.2:0.3:0.45\n",
        ],
    )
    manifest = _make_manifest(
        tmp_path, [("trait_a", vcf1, "Trait A"), ("trait_b", vcf2, "Trait B")]
    )
    store = tmp_path / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest,
        store,
        reference_panel=_panel(tmp_path),
        store_id="hybrid-test",
        release_id="v1",
        n_workers=1,
    )
    return store


_BUILDERS: dict[str, Callable[[Path, Path], Path]] = {
    "dense": _dense,
    "ragged": _ragged,
    "hybrid": _hybrid,
}


def _calls(query: Any) -> dict[str, Callable[[], dict[str, np.ndarray]]]:
    """Every query shape the facades share, on the first Analysis and variant."""
    analyses = query.analyses_table()
    analysis = next(iter(analyses.values()))["analysis_id"]
    # The per-Analysis top-hit slice needs an Analysis that has hits.
    first_hit = int(query.top_hits(threshold=_THRESHOLD)["analysis_index"][0])
    hit_analysis = analyses[first_hit]["analysis_id"]
    variant = next(iter(query.variants_table().values()))
    chromosome = str(variant["chromosome"])
    return {
        "analysis": lambda: query.analysis(analysis),
        "phewas": lambda: query.phewas(variant["alid"]),
        "range_phewas": lambda: query.range_phewas(chromosome, 0, 10_000_000),
        "lookup": lambda: query.lookup([variant["alid"]], [analysis]),
        "top_hits": lambda: query.top_hits(threshold=_THRESHOLD),
        "top_hits_one": lambda: query.top_hits(analysis_id=hit_analysis, threshold=_THRESHOLD),
    }


@pytest.mark.parametrize("layout", sorted(_BUILDERS))
def test_a_repeated_query_reads_no_metadata(
    layout: str, tmp_path: Path, dense_store_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _BUILDERS[layout](tmp_path, dense_store_path)
    with query_store(store) as query:
        calls = _calls(query)
        warm = {name: call() for name, call in calls.items()}
        empty = [name for name, result in warm.items() if len(result["z"]) == 0]
        assert empty == [], f"every shape must return rows for this to mean anything: {empty}"
        assert len(warm["top_hits"]["z"]) >= 2, "fixture must have several top hits"
        if layout == "hybrid":
            overflow = query._overflow_top_hits(_THRESHOLD)
            assert len(overflow["z"]) > 0, "hybrid fixture must have overflow hits too"
        with _metadata_reads(monkeypatch) as reads:
            again = {name: call() for name, call in calls.items()}
        assert reads == []
        for name, result in warm.items():
            assert result.keys() == again[name].keys()
            for key, values in result.items():
                # An absent frequency is NaN, and NaN == NaN for this purpose.
                nan_equal = np.asarray(values).dtype.kind == "f"
                assert np.array_equal(values, again[name][key], equal_nan=nan_equal), (name, key)
