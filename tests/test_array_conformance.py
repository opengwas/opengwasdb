"""Runtime conformance for the array-creation seam (#243).

``test_array_creation_seam`` scans source statically, and that scan is
best-effort: Python can reach an array through routes no source scan can follow.
This module is the behavioural backstop.  It builds one fixture store of every
builder path and inspects the arrays the builders actually wrote, so a direct
``zarr.open_group(...).create_dataset(...)``, a mapping assignment
(``group["z"] = values``, which uses Zarr's default lz4 codec), or any other
route that slipped past the scan still fails here.

Every array in every ``data.zarr`` must:

* map to a known :class:`ArrayRole` -- an array the test cannot name a role for
  fails, so a new array cannot ship without one;
* carry the seam's chunk layout for that role and shape (the role policy, with
  the component plane's chunk supplied for ``PER_VARIANT``);
* carry the seam's compressor -- the exact/overflow tables the seam deliberately
  writes uncompressed may carry none, but must carry no other codec;
* be filterless, and dtype-zero-filled for every role but the statistic grid.

The stores are small synthetic fixtures; building all of them costs a few
seconds, so this needs no slow marker.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from opengwasdb.store.arrays import (
    ArrayRole,
    chunk_layout,
    component_variant_chunk,
    compressor,
    open_group,
)

#: Array names that make up a Zarr group's exact/overflow side tables.  These
#: are the arrays the seam writes uncompressed in several paths.
_EXCEPTION_TABLE_NAMES = frozenset(
    {
        "z_overflow_index",
        "z_overflow_value",
        "eaf_exception_index",
        "eaf_exception_value",
        "se_exception_index",
        "se_exception_value",
    }
)

#: The dense statistic grid planes, the one role whose fill value is a per-plane
#: missing marker (spec §15) rather than the dtype default.
_DENSE_GRID_NAMES = frozenset({"z", "se", "eaf"})


def _build_stores(root: Path) -> list[Path]:
    """Build one fixture store of every builder path under `root`.

    The fixture writers are imported from the suites that already own them, so
    the shapes and inputs stay the ones those suites trust.
    """
    import test_dense_completion as tdc
    import test_dense_rho as tdr
    import test_dense_vcf_build as tdv
    import test_hybrid_build as thb
    import test_ragged_build_besd as trb
    import test_ragged_build_ssf as trs
    import test_ragged_completion as trc

    from opengwasdb.build.observed import build_dense_observed_from_sources
    from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
    from opengwasdb.layouts.dense.complete import complete_dense_store
    from opengwasdb.layouts.dense.rho import build_dense_rho
    from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
    from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd
    from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
    from opengwasdb.layouts.ragged.complete import complete_ragged_store
    from opengwasdb.layouts.ragged.top_hits import build_ragged_top_hit_indexes

    stores: list[Path] = []

    # Dense in-memory.
    d = root / "dense-in-memory"
    d.mkdir()
    source = d / "associations.tsv"
    source.write_text(
        tdc.SOURCE_HEADER + "\n" + "\n".join(tdc.SOURCE_ROWS) + "\n", encoding="utf-8"
    )
    dense = d / "store.opengwasdb"
    build_dense_observed_from_sources(
        [source],
        dense,
        store_id="fixture-store",
        release_id="observed-v1",
        reference_assembly="GRCh37",
    )
    stores.append(dense)

    # Dense VCF, serial and parallel.
    for n_workers in (1, 2):
        d = root / f"dense-vcf-n{n_workers}"
        d.mkdir()
        vcf1 = tdv._make_vcf(
            d,
            "trait_a",
            [
                f"1\t{tdv.HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
                f"1\t{tdv.HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",
                f"1\t{tdv.HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n",
            ],
        )
        vcf2 = tdv._make_vcf(
            d,
            "trait_b",
            [
                f"1\t{tdv.HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t6.0:0.5\n",
                f"1\t{tdv.HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE\t1.2:0.3\n",
            ],
        )
        manifest = tdv._make_manifest(
            d,
            [("trait_a", vcf1, "Trait A"), ("trait_b", vcf2, "Trait B")],
            scales={"trait_b": "log_or"},
        )
        store = d / "store.opengwasdb"
        build_dense_from_vcf_manifest(
            manifest,
            store,
            store_id="test-store",
            release_id="v1",
            n_workers=n_workers,
        )
        stores.append(store)

    # Dense Reference Completion.
    d = root / "dense-completion"
    d.mkdir()
    source = d / "associations.tsv"
    source.write_text(
        tdc.SOURCE_HEADER + "\n" + "\n".join(tdc.SOURCE_ROWS) + "\n", encoding="utf-8"
    )
    observed = d / "obs.opengwasdb"
    build_dense_observed_from_sources(
        [source], observed, store_id="test", release_id="obs-v1", reference_assembly="GRCh38"
    )
    completed = d / "comp.opengwasdb"
    complete_dense_store(
        observed,
        completed,
        tdc._make_ld_panel(d),
        ancestry="EUR",
        min_cor=0.0,
        release_id="comp-v1",
    )
    stores.append(completed)

    # Ragged SSF, plus its top-hit index.
    d = root / "ragged-ssf"
    d.mkdir()
    manifest, filtered = trs._make_fixture(d)
    ssf = d / "store.opengwasdb"
    build_ragged_from_ssf(manifest, filtered, ssf, store_id="test", release_id="v1")
    build_ragged_top_hit_indexes(ssf)
    stores.append(ssf)

    # Ragged BESD.
    d = root / "ragged-besd"
    d.mkdir()
    besd = d / "store.opengwasdb"
    build_ragged_from_besd(
        trb._make_besd_fixture(d), besd, store_id="test", release_id="v1", tissue="Whole_Blood"
    )
    stores.append(besd)

    # Ragged Reference Completion.
    d = root / "ragged-completion"
    d.mkdir()
    observed = d / "obs.opengwasdb"
    build_ragged_from_besd(
        trc._make_besd_fixture(d), observed, store_id="test", release_id="obs-v1", tissue="Blood"
    )
    completed = d / "comp.opengwasdb"
    complete_ragged_store(
        observed,
        completed,
        trc._make_ld_panel(d, "1", 900_000, 1_300_000),
        ancestry="EUR",
        cis_window_bp=500_000,
        min_cor=0.0,
        release_id="comp-v1",
    )
    stores.append(completed)

    # Hybrid, serial and parallel.
    for n_workers in (1, 2):
        d = root / f"hybrid-n{n_workers}"
        d.mkdir()
        vcf1 = thb._make_vcf(
            d,
            "trait_a",
            [
                f"1\t{thb.HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t2.0:0.5:0.2\n",
                f"1\t{thb.HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE:AF\t1.5:0.3:0.3\n",
                f"1\t{thb.HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE:AF\t0.6:0.2:0.4\n",
            ],
        )
        vcf2 = thb._make_vcf(
            d,
            "trait_b",
            [
                f"1\t{thb.HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t6.0:0.5:0.25\n",
                f"1\t{thb.HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE:AF\t1.2:0.3:0.45\n",
            ],
        )
        manifest = thb._make_manifest(
            d, [("trait_a", vcf1, "Trait A"), ("trait_b", vcf2, "Trait B")]
        )
        store = d / "store.opengwasdb"
        build_hybrid_from_vcf_manifest(
            manifest,
            store,
            reference_panel=thb._panel(d),
            store_id="hybrid-test",
            release_id="v1",
            n_workers=n_workers,
        )
        stores.append(store)

    # Rho Matrix.
    d = root / "dense-rho"
    d.mkdir()
    store = d / "store.opengwasdb"
    build_dense_observed_from_sources(
        [tdr._make_rho_source(d)],
        store,
        store_id="rho-fixture",
        release_id="observed-v1",
        reference_assembly="GRCh37",
    )
    build_dense_rho(store, window_bp=50, z_thresh=1.0, min_nulls=5, n_workers=1)
    stores.append(store)

    return stores


@pytest.fixture(scope="session")
def conformance_stores(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """One fixture store of every builder path, built once for the session."""
    root = tmp_path_factory.mktemp("array-conformance")
    return _build_stores(root)


def _iter_arrays(group: Any, prefix: str = "") -> Iterator[tuple[str, Any, Any]]:
    """Every array in a Zarr group tree, with its dotted path and parent group."""
    for name in sorted(group.array_keys()):
        yield f"{prefix}/{name}", group[name], group
    for name in sorted(group.group_keys()):
        yield from _iter_arrays(group[name], f"{prefix}/{name}")


def _role_for(path: str, group: Any) -> ArrayRole:
    """The role an array plays, from its path and its parent group.

    Deliberately explicit: a name this does not recognise raises, so a new array
    without a role fails the test rather than being skipped.
    """
    parts = path.strip("/").split("/")
    name = parts[-1]
    if "top_hits" in parts:
        if name == "analysis_offsets":
            return ArrayRole.TOP_HIT_ANALYSIS_OFFSETS
        return ArrayRole.TOP_HIT_INDEX
    if "rho" in parts:
        return ArrayRole.RHO_ARRAY
    if "offsets" in group:  # a Ragged CSR component
        if name == "offsets":
            return ArrayRole.ASSOCIATION_OFFSETS
        if name in {"variant_index", "z", "se", "eaf", "imputed"}:
            return ArrayRole.ASSOCIATION_SEQUENCE
    if name in _EXCEPTION_TABLE_NAMES:
        return ArrayRole.EXCEPTION_TABLE
    if name in {"eaf_baseline", "eaf_reference"}:
        return ArrayRole.PER_VARIANT
    if name == "se_coefficients":
        return ArrayRole.SE_COEFFICIENTS
    if name in _DENSE_GRID_NAMES:
        return ArrayRole.DENSE_STATISTIC_PLANE
    if name == "imputed":
        return ArrayRole.DENSE_IMPUTED_MASK
    if name == "on_panel":
        return ArrayRole.DENSE_ON_PANEL
    raise AssertionError(f"no ArrayRole is known for array {path!r}")


def test_every_built_array_matches_the_seam_policy(conformance_stores: list[Path]) -> None:
    """Every array a builder wrote obeys the seam: role, layout, compressor, fill."""
    seam = compressor().get_config()
    checked = 0
    for store in conformance_stores:
        for data_zarr in sorted(store.rglob("data.zarr")):
            root = open_group(data_zarr)
            for path, array, group in _iter_arrays(root):
                where = f"{store.name}/{data_zarr.relative_to(store)}/{path}"
                role = _role_for(path, group)
                expected = chunk_layout(
                    role,
                    tuple(int(size) for size in array.shape),
                    component_chunk=component_variant_chunk(group),
                )
                assert tuple(int(size) for size in array.chunks) == expected, (
                    f"{where}: {role} chunks {tuple(array.chunks)} != seam layout {expected}"
                )
                actual_compressor = array.compressor.get_config() if array.compressor else None
                allowed = [seam, None] if role is ArrayRole.EXCEPTION_TABLE else [seam]
                assert actual_compressor in allowed, (
                    f"{where}: {role} compressor {actual_compressor} is not the seam's {seam}"
                )
                assert array.filters in (None, []), f"{where}: filters {array.filters}"
                if role is not ArrayRole.DENSE_STATISTIC_PLANE:
                    assert array.fill_value == 0, (
                        f"{where}: {role} fill {array.fill_value!r} is not the dtype default"
                    )
                checked += 1
    # Assert the fixture is meaningful before trusting a clean walk: these ten
    # stores have far more than a handful of arrays between them.
    assert checked > 100, f"only {checked} arrays checked; the fixture set is wrong"
    assert np is not None
