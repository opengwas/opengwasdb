"""Indexed Variant Subsets: build, atomic publication, and standalone validation.

Issue #264 implements the storage/lifecycle half of ADR 0061: one full-statistic,
all-Analysis Observed-Only Dense index, built atomically and validated against
the primary planes. Query integration is #265 and Reference-Completed support is
#266, so neither is exercised here.

The fixture is deliberately rich enough to reach every class the writer and the
validator must handle: a present cell and a missing one, a Z overflow cell, an
EAF exception and an SE exception. The suite asserts each of those exists in the
*primary* store before it asserts anything the index does with it -- a fixture
that never reaches the broken path proves nothing (CONTRIBUTING.md, "A test that
cannot fail is worse than no test").

Corruption tests are the one place a test reaches into the index's Zarr group:
the thing under test is precisely that a hand-damaged group fails validation, so
there is no API that could produce the damage.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest
import zarr
from legacy_fixtures import relayout_as_0_1_0
from typer.testing import CliRunner, Result

from opengwasdb.build.source import NormalisedAssociation
from opengwasdb.cli.main import app
from opengwasdb.encoding import DenseEafPlane, DenseSePlane, DenseZPlane
from opengwasdb.layouts.dense.build import build_dense_observed_store
from opengwasdb.layouts.dense.indexed_subsets import (
    DEFAULT_BAND_CELLS,
    INDEXED_SUBSETS_GROUP,
    IndexedSubsetError,
    IndexedSubsetExistsError,
    IndexedSubsetNameError,
    IndexedSubsetStaleError,
    VariantListError,
    _ensure_indexed_subset_namespace,
    build_indexed_subset,
    list_indexed_subsets,
    open_indexed_subset,
    parse_indexed_subset_name,
    read_variant_list,
    remove_indexed_subset,
    resolve_variant_list,
)
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.query import query_store
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store.arrays import ConsolidatedMetadataError
from opengwasdb.store.open import destination_lock, directory_lock, zarr_format_for_version
from opengwasdb.validation import validate_store
from opengwasdb.variants import CanonicalVariant

N_VARIANTS = 320
N_ANALYSES = 4
#: Variants whose cells reach the special paths below. Small indices keep the
#: fixture cheap; the point is that the classes exist, not how many of them.
OVERFLOW_VARIANT = 5
MISSING_VARIANT = 11
EAF_EXCEPTION_VARIANT = 20
SE_EXCEPTION_VARIANT = 30
#: A canonical ALID that is not in the store at all.
ABSENT_ALID = "1:999999:A:G"


def alid(variant_index: int) -> str:
    return f"1:{variant_index + 1}:A:G"


def _fixture_records() -> list[NormalisedAssociation]:
    freqs = np.linspace(0.05, 0.95, N_VARIANTS, dtype=np.float32)
    records: list[NormalisedAssociation] = []
    for col in range(N_ANALYSES):
        analysis_id = f"a{col}"
        # log(se) tracks the MAF-predicted value, which is what makes the plan
        # residual-code `se`; the wobble keeps it off the exact model.
        se = np.exp(
            (-3.0 + col * 0.2)
            - 0.5 * np.log(2 * freqs * (1 - freqs))
            + 0.10 * np.sin(np.arange(N_VARIANTS) * (0.07 + col * 0.01))
        ).astype(np.float32)
        eaf = freqs.copy()
        z = np.full(N_VARIANTS, 1.0 + 0.1 * col, dtype=np.float64)
        z[OVERFLOW_VARIANT] = 100.0 + col  # beyond int16 fixed-point range
        if col == 0:
            z[MISSING_VARIANT] = np.nan
            se[MISSING_VARIANT] = np.nan
            eaf[MISSING_VARIANT] = np.nan
        if col == 1:
            eaf[EAF_EXCEPTION_VARIANT] = 0.999  # far from every baseline
        if col == 2:
            se[SE_EXCEPTION_VARIANT] = 0.0  # no logit, so an exact exception
        for row in range(N_VARIANTS):
            records.append(
                NormalisedAssociation(
                    analysis_id=analysis_id,
                    variant=CanonicalVariant("1", row + 1, "A", "G"),
                    z=float(z[row]),
                    se=None if np.isnan(se[row]) else float(se[row]),
                    eaf=None if np.isnan(eaf[row]) else float(eaf[row]),
                )
            )
    return records


def build_rich_store(store: Path) -> Path:
    """Build the rich Observed-Only Dense release the writer and query suites share.

    Extracted from the `rich_store` fixture so #265's query suite can build its
    own indexed copy without copying the fixture body (and so the two cannot
    drift into subtly different releases).
    """
    build_dense_observed_store(
        _fixture_records(),
        store,
        store_id="idx-fixture",
        release_id="observed-v1",
        reference_assembly="GRCh38",
        chunk_shape=(100, N_ANALYSES),
    )
    return store


@pytest.fixture(scope="module")
def rich_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_rich_store(tmp_path_factory.mktemp("indexed") / "rich.opengwasdb")


@pytest.fixture
def variant_list(tmp_path: Path) -> Path:
    """An unsorted list of present ALIDs, one absent one, and a blank line."""
    alids = [
        alid(SE_EXCEPTION_VARIANT),
        alid(OVERFLOW_VARIANT),
        alid(200),
        alid(MISSING_VARIANT),
        alid(EAF_EXCEPTION_VARIANT),
        alid(42),
        ABSENT_ALID,
    ]
    path = tmp_path / "subset.alid.txt"
    path.write_text("\n".join(alids) + "\n\n", encoding="utf-8")
    return path


def _root(store: Path) -> zarr.Group:
    return zarr.open_group(str(store / "data.zarr"), mode="r")


def _index_group(store: Path, name: str) -> zarr.Group:
    return zarr.open_group(
        str(store / "data.zarr" / INDEXED_SUBSETS_GROUP / name), mode="r+"
    )


def _build(
    store: Path,
    name: str,
    variant_list: Path,
    *,
    reference_assembly: str = "GRCh38",
    overwrite: bool = False,
    band_cells: int = DEFAULT_BAND_CELLS,
) -> Path:
    build_indexed_subset(
        store,
        name,
        variant_list,
        reference_assembly=reference_assembly,
        overwrite=overwrite,
        band_cells=band_cells,
    )
    return store


def _concurrent_builds(
    store: Path, names: tuple[str, ...], variant_list: Path
) -> list[BaseException]:
    """Build `names` at once and return whatever each worker raised."""
    errors: list[BaseException] = []
    barrier = threading.Barrier(len(names))

    def worker(name: str) -> None:
        barrier.wait()
        try:
            _build(store, name, variant_list)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(name,)) for name in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


# ── Fixture is meaningful (asserted before anything is asserted about it) ────


def test_fixture_reaches_every_class_the_index_must_carry(rich_store: Path) -> None:
    encoding = StoreManifest.load(rich_store).encoding
    assert encoding.z.is_fixed_point, "the fixture must reach the Z overflow path"
    assert encoding.eaf.is_residual, "the fixture must reach the EAF exception path"
    assert encoding.se.is_residual, "the fixture must reach the SE exception path"
    root = _root(rich_store)
    assert len(root["z_overflow_index"][:]) > 0
    assert len(root["eaf_exception_index"][:]) > 0
    assert len(root["se_exception_index"][:]) > 0
    assert validate_store(rich_store).ok


# ── Input contract ──────────────────────────────────────────────────────────


def test_subset_name_grammar_rejects_staging_and_internal_names() -> None:
    assert parse_indexed_subset_name("hm3-eur.v1") == "hm3-eur.v1"
    for bad in (".tmp", ".name.tmp.1", "a/b", "", "..", " leading", "trailing "):
        with pytest.raises(IndexedSubsetNameError):
            parse_indexed_subset_name(bad)


def test_read_variant_list_rejects_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("1:100:A:G\nnot-an-alid\n", encoding="utf-8")
    with pytest.raises(VariantListError, match="canonical ALID"):
        read_variant_list(path)


def test_read_variant_list_rejects_non_canonical_alid(tmp_path: Path) -> None:
    path = tmp_path / "noncanonical.txt"
    path.write_text("chr1:100:A:G\n", encoding="utf-8")
    with pytest.raises(VariantListError, match="canonical"):
        read_variant_list(path)


def test_read_variant_list_rejects_duplicate_alids(tmp_path: Path) -> None:
    path = tmp_path / "dupes.txt"
    path.write_text("1:100:A:G\n1:100:A:G\n", encoding="utf-8")
    with pytest.raises(VariantListError, match="duplicate"):
        read_variant_list(path)


def test_read_variant_list_counts_only_nonblank_lines_and_hashes_bytes(tmp_path: Path) -> None:
    path = tmp_path / "list.txt"
    raw = "1:100:A:G\n\n1:200:A:G\n"
    path.write_text(raw, encoding="utf-8")
    parsed = read_variant_list(path)
    assert parsed.alids == ("1:100:A:G", "1:200:A:G")
    assert parsed.sha256 == hashlib.sha256(raw.encode("utf-8")).hexdigest()


def test_resolve_variant_list_keeps_store_order_and_counts_absent(rich_store: Path) -> None:
    resolved = resolve_variant_list(
        rich_store, (alid(200), alid(OVERFLOW_VARIANT), alid(42), ABSENT_ALID)
    )
    assert resolved.variant_index.tolist() == [OVERFLOW_VARIANT, 42, 200]
    assert resolved.requested == 4
    assert resolved.resolved == 3
    assert resolved.absent == 1


def test_build_rejects_assembly_mismatch(rich_store: Path, variant_list: Path) -> None:
    with pytest.raises(ValueError, match="assembly"):
        build_indexed_subset(
            rich_store, "mismatch", variant_list, reference_assembly="GRCh37"
        )


def test_build_rejects_zero_match(tmp_path: Path, rich_store: Path) -> None:
    only_absent = tmp_path / "absent.txt"
    only_absent.write_text(ABSENT_ALID + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no Store variants"):
        build_indexed_subset(
            rich_store, "empty", only_absent, reference_assembly="GRCh38"
        )


# ── Build ───────────────────────────────────────────────────────────────────


def test_build_stores_variant_indices_in_store_order(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)

    subset = open_indexed_subset(store, "hm3")
    assert subset.variant_index.tolist() == [OVERFLOW_VARIANT, MISSING_VARIANT,
                                             EAF_EXCEPTION_VARIANT, SE_EXCEPTION_VARIANT,
                                             42, 200]
    assert subset.requested_count == 7
    assert subset.resolved_count == 6
    assert subset.absent_count == 1
    assert subset.n_analyses == N_ANALYSES
    assert subset.n_subset_variants == 6


def test_built_index_carries_overflow_and_both_exception_tables(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    subset = open_indexed_subset(store, "hm3")
    assert subset.n_z_overflow > 0, "the fixture must force Z overflow into the index"
    assert subset.n_eaf_exceptions > 0, "the fixture must force an EAF exception"
    assert subset.n_se_exceptions > 0, "the fixture must force an SE exception"


def test_decoded_index_values_equal_the_primary_planes(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    subset = open_indexed_subset(store, "hm3")
    encoding = StoreManifest.load(store).encoding
    root = _root(store)
    z_plane = DenseZPlane.open(root, encoding)
    eaf_plane = DenseEafPlane.open(root, encoding)
    se_plane = DenseSePlane.open(root, encoding)

    for analysis_index in range(N_ANALYSES):
        decoded = subset.decode_analysis(analysis_index)
        primary_se_values = eaf_plane.read_column(analysis_index).values
        primary_eaf = primary_se_values[subset.variant_index]
        primary_z = z_plane.column(analysis_index)[subset.variant_index]
        primary_se = se_plane.column(analysis_index, eaf=primary_se_values)[
            subset.variant_index
        ]

        np.testing.assert_array_equal(decoded.z, primary_z, err_msg=f"z col {analysis_index}")
        np.testing.assert_array_equal(decoded.se, primary_se, err_msg=f"se col {analysis_index}")
        np.testing.assert_array_equal(decoded.eaf, primary_eaf, err_msg=f"eaf col {analysis_index}")
    assert validate_store(store).ok


def test_band_size_does_not_change_the_index(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    wide = shutil.copytree(rich_store, tmp_path / "wide.opengwasdb")
    narrow = shutil.copytree(rich_store, tmp_path / "narrow.opengwasdb")
    _build(wide, "hm3", variant_list, band_cells=10_000_000)
    _build(narrow, "hm3", variant_list, band_cells=1)

    wide_group = _index_group(wide, "hm3")
    narrow_group = _index_group(narrow, "hm3")
    assert np.array_equal(wide_group["z"][:], narrow_group["z"][:])
    assert np.array_equal(wide_group["se"][:], narrow_group["se"][:])
    assert np.array_equal(wide_group["eaf"][:], narrow_group["eaf"][:])


# ── Atomic lifecycle ────────────────────────────────────────────────────────


def _digest(path: Path) -> str:
    """A SHA-256 over a tree's relative paths and file bytes."""
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode("utf-8"))
            digest.update(item.read_bytes())
    return digest.hexdigest()


def _authoritative_digest(store: Path) -> str:
    """Everything in the release except the optional indexed-subset namespace."""
    digest = hashlib.sha256()
    for item in sorted(store.rglob("*")):
        if item.is_file() and INDEXED_SUBSETS_GROUP not in item.relative_to(store).parts:
            digest.update(str(item.relative_to(store)).encode("utf-8"))
            digest.update(item.read_bytes())
    return digest.hexdigest()


def test_interrupted_build_leaves_no_published_group_or_temp(
    tmp_path: Path, rich_store: Path, variant_list: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    before = _authoritative_digest(store)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        "opengwasdb.layouts.dense.indexed_subsets.write_indexed_subset_group", boom
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _build(store, "hm3", variant_list)

    assert "hm3" not in list_indexed_subsets(store)
    namespace = store / "data.zarr" / INDEXED_SUBSETS_GROUP
    if namespace.exists():
        assert not [p for p in namespace.iterdir() if p.name.startswith(".")]
    assert _authoritative_digest(store) == before


def test_failed_overwrite_preserves_the_published_group_byte_for_byte(
    tmp_path: Path, rich_store: Path, variant_list: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    published = store / "data.zarr" / INDEXED_SUBSETS_GROUP / "hm3"
    before = _digest(published)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        "opengwasdb.layouts.dense.indexed_subsets.write_indexed_subset_group", boom
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _build(store, "hm3", variant_list, overwrite=True)

    assert _digest(published) == before
    assert not [p for p in published.parent.iterdir() if p.name.startswith(".")]


def test_same_name_concurrent_builds_yield_one_winner(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    # Publish and remove once so the namespace group exists before the race:
    # creating the namespace is not what this test is about.
    _build(store, "warmup", variant_list)
    remove_indexed_subset(store, "warmup")

    outcomes: list[object] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        try:
            _build(store, "race", variant_list)
            outcomes.append("won")
        except IndexedSubsetExistsError as exc:
            outcomes.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count("won") == 1, outcomes
    assert sum(isinstance(item, IndexedSubsetExistsError) for item in outcomes) == 1
    assert validate_store(store).ok


def test_no_overwrite_build_raises_the_domain_exists_error(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    with pytest.raises(IndexedSubsetExistsError):
        _build(store, "hm3", variant_list)


def test_first_time_concurrent_different_name_builds_share_one_namespace(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    """No warm-up: both builds race to create `indexed_subsets` itself.

    The namespace does not exist when the threads start, so namespace creation
    happens concurrently with the first publications -- the path the lock in
    `_ensure_indexed_subset_namespace` exists for.
    """
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    assert not (store / "data.zarr" / INDEXED_SUBSETS_GROUP).exists()

    errors = _concurrent_builds(store, ("one", "two"), variant_list)

    assert not errors, errors
    assert sorted(list_indexed_subsets(store)) == ["one", "two"]
    assert validate_store(store).ok


def test_namespace_creation_is_serialised_across_threads(
    tmp_path: Path, rich_store: Path
) -> None:
    """Eight concurrent creators produce one valid namespace group, not a race."""
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    fmt = zarr_format_for_version(StoreManifest.load(store).format_version)
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        try:
            _ensure_indexed_subset_namespace(store, fmt)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    root = zarr.open_group(str(store / "data.zarr"), mode="r")
    assert INDEXED_SUBSETS_GROUP in root
    assert validate_store(store).ok


def test_namespace_creation_waits_for_the_shared_directory_lock(
    tmp_path: Path, rich_store: Path
) -> None:
    """Namespace creation takes the data.zarr lock, so it cannot race itself.

    Deterministic where a timing race is not: while another holder owns the
    shared directory lock, the namespace must not appear.
    """
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    fmt = zarr_format_for_version(StoreManifest.load(store).format_version)
    namespace = store / "data.zarr" / INDEXED_SUBSETS_GROUP
    created = threading.Event()

    def worker() -> None:
        _ensure_indexed_subset_namespace(store, fmt)
        created.set()

    with directory_lock(store / "data.zarr"):
        thread = threading.Thread(target=worker)
        thread.start()
        assert not created.wait(timeout=0.5), (
            "namespace creation ran without holding the data.zarr lock"
        )
        assert not namespace.exists(), "namespace appeared while the lock was held"
    thread.join(timeout=10)
    assert created.is_set()
    assert INDEXED_SUBSETS_GROUP in zarr.open_group(str(store / "data.zarr"), mode="r")


def test_different_name_concurrent_builds_do_not_touch_each_other(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "warmup", variant_list)
    remove_indexed_subset(store, "warmup")

    errors = _concurrent_builds(store, ("one", "two"), variant_list)

    assert not errors, errors
    assert sorted(list_indexed_subsets(store)) == ["one", "two"]
    assert validate_store(store).ok


def test_removal_takes_the_publication_lock(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    """A removal cannot run while a publication holds the same destination lock."""
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    dest = store / "data.zarr" / INDEXED_SUBSETS_GROUP / "hm3"
    removed = threading.Event()

    def worker() -> None:
        remove_indexed_subset(store, "hm3")
        removed.set()

    with destination_lock(dest):
        thread = threading.Thread(target=worker)
        thread.start()
        assert not removed.wait(timeout=0.5), (
            "remove_indexed_subset ran without holding the publication lock"
        )
    thread.join(timeout=10)
    assert removed.is_set()
    assert list_indexed_subsets(store) == ()
    assert validate_store(store).ok


def test_concurrent_removes_of_one_subset_have_one_winner(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    outcomes: list[bool] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        try:
            outcomes.append(remove_indexed_subset(store, "hm3"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert sorted(outcomes) == [False, True]
    namespace = store / "data.zarr" / INDEXED_SUBSETS_GROUP
    assert not [p for p in namespace.iterdir() if p.name.startswith(".")]
    assert validate_store(store).ok


def test_removing_the_derived_group_leaves_a_valid_store_with_unchanged_queries(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    with query_store(store) as query:
        before = query.analysis("a0")

    assert remove_indexed_subset(store, "hm3") is True
    assert list_indexed_subsets(store) == ()
    assert validate_store(store).ok
    with query_store(store) as query:
        after = query.analysis("a0")
    for field in before:
        np.testing.assert_array_equal(before[field], after[field], err_msg=field)


def test_no_index_store_validates_unchanged(rich_store: Path) -> None:
    assert list_indexed_subsets(rich_store) == ()
    assert validate_store(rich_store).ok


def test_build_on_a_0_1_0_store_keeps_the_v2_layout(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    legacy = relayout_as_0_1_0(rich_store, tmp_path / "legacy.opengwasdb")
    _build(legacy, "hm3", variant_list)

    data = legacy / "data.zarr"
    assert (data / ".zgroup").exists(), "a 0.1.0 release keeps its v2 root group"
    assert not (data / "zarr.json").exists(), (
        "building an index must not stamp a v3 root onto a 0.1.0 release"
    )
    subset = data / INDEXED_SUBSETS_GROUP / "hm3"
    assert (subset / ".zgroup").exists()
    assert not (subset / "zarr.json").exists()
    assert validate_store(legacy).ok
    assert open_indexed_subset(legacy, "hm3").n_subset_variants == 6


# ── Encoding branches ───────────────────────────────────────────────────────
#
# The rich fixture reaches one combination (fixed-point z, residual se and
# residual eaf).  These fixtures reach the other shapes the writer and validator
# branch on, and each asserts the fixture reaches its branch before asserting
# anything the index does with it.


def _absent_eaf_store(tmp_path: Path) -> Path:
    """A release with no eaf plane and a float16 se plane."""
    store = tmp_path / "absent-eaf.opengwasdb"
    records = [
        NormalisedAssociation("a1", CanonicalVariant("1", 100, "A", "G"), z=2.0, se=0.1),
        NormalisedAssociation("a2", CanonicalVariant("1", 100, "A", "G"), z=6.0, se=0.2),
        NormalisedAssociation("a1", CanonicalVariant("1", 200, "C", "T"), z=3.0, se=0.2),
        NormalisedAssociation("a2", CanonicalVariant("1", 300, "A", "G"), z=-6.0, se=0.5),
    ]
    build_dense_observed_store(
        records,
        store,
        store_id="absent-eaf",
        release_id="r",
        reference_assembly="GRCh37",
        chunk_shape=(2, 2),
    )
    return store


def test_index_omits_eaf_and_se_side_tables_when_the_plan_has_none(tmp_path: Path) -> None:
    store = _absent_eaf_store(tmp_path)
    encoding = StoreManifest.load(store).encoding
    assert encoding.eaf.is_absent, "the fixture must reach the absent-eaf branch"
    assert not encoding.se.is_residual, "the fixture must reach the float16-se branch"

    variant_list = tmp_path / "absent.alid.txt"
    variant_list.write_text("1:100:A:G\n1:200:C:T\n1:300:A:G\n", encoding="utf-8")
    _build(store, "absent", variant_list, reference_assembly="GRCh37")

    group = _index_group(store, "absent")
    assert "eaf" not in group
    assert "eaf_baseline" not in group and "eaf_exception_index" not in group
    assert "se_coefficients" not in group and "se_exception_index" not in group
    subset = open_indexed_subset(store, "absent")
    assert subset.has_eaf is False
    assert subset.decode_analysis(0).eaf is None
    assert validate_store(store).ok


def _float32_eaf_store(tmp_path: Path) -> Path:
    """One Analysis: the EAF baseline cannot amortise, so the plan is float32."""
    store = tmp_path / "float32-eaf.opengwasdb"
    records = [
        NormalisedAssociation(
            "a",
            CanonicalVariant("1", row + 1, "A", "G"),
            z=1.0,
            se=0.2,
            eaf=0.05 + 0.1 * row,
        )
        for row in range(8)
    ]
    build_dense_observed_store(
        records,
        store,
        store_id="float32-eaf",
        release_id="r",
        reference_assembly="GRCh38",
        chunk_shape=(4, 1),
    )
    return store


def test_index_handles_a_float32_eaf_plane(tmp_path: Path) -> None:
    store = _float32_eaf_store(tmp_path)
    encoding = StoreManifest.load(store).encoding
    assert encoding.eaf.kind == "float32", "the fixture must reach the float32-eaf branch"
    assert not encoding.eaf.is_residual

    variant_list = tmp_path / "float32.alid.txt"
    variant_list.write_text("\n".join(f"1:{row + 1}:A:G" for row in range(8)) + "\n")
    _build(store, "eaf32", variant_list)

    group = _index_group(store, "eaf32")
    assert str(group["eaf"].dtype) == "float32", "the index must store the declared dtype"
    assert "eaf_baseline" not in group and "eaf_exception_index" not in group
    subset = open_indexed_subset(store, "eaf32")
    primary = DenseEafPlane.open(_root(store), encoding)
    for analysis_index in range(subset.n_analyses):
        decoded = subset.decode_analysis(analysis_index)
        np.testing.assert_array_equal(
            decoded.eaf, primary.read_column(analysis_index).values[subset.variant_index]
        )
    assert validate_store(store).ok


# ── Chunk and shard boundaries ──────────────────────────────────────────────

#: More variants than one Indexed Variant Subset inner chunk (65,536), so the
#: Analysis-major planes span more than one chunk without being a real corpus.
BIG_VARIANTS = 65_600


@pytest.fixture(scope="module")
def big_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    store = tmp_path_factory.mktemp("big") / "big.opengwasdb"
    records = [
        NormalisedAssociation(
            "a",
            CanonicalVariant("1", row + 1, "A", "G"),
            z=float((row % 7) - 3),
            se=0.2,
        )
        for row in range(BIG_VARIANTS)
    ]
    build_dense_observed_store(
        records,
        store,
        store_id="big",
        release_id="r",
        reference_assembly="GRCh38",
        chunk_shape=(1000, 1),
    )
    return store


def _big_variant_list(tmp_path: Path) -> Path:
    path = tmp_path / "big.alid.txt"
    path.write_text("\n".join(f"1:{row + 1}:A:G" for row in range(BIG_VARIANTS)) + "\n")
    return path


def test_index_plane_spans_multiple_inner_chunks(tmp_path: Path, big_store: Path) -> None:
    store = shutil.copytree(big_store, tmp_path / "store.opengwasdb")
    _build(store, "big", _big_variant_list(tmp_path))

    group = _index_group(store, "big")
    assert tuple(group["z"].shape) == (1, BIG_VARIANTS)
    assert tuple(group["z"].chunks) == (1, store_arrays.INDEXED_SUBSET_CHUNK)
    assert group["z"].shape[1] > group["z"].chunks[1], "the fixture must span >1 chunk"

    subset = open_indexed_subset(store, "big")
    encoding = StoreManifest.load(store).encoding
    primary = DenseZPlane.open(_root(store), encoding).column(0)
    np.testing.assert_array_equal(subset.decode_analysis(0).z, primary[subset.variant_index])
    assert validate_store(store).ok


def test_index_plane_splits_into_multiple_shards(
    tmp_path: Path, big_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One inner chunk per shard, so the 65,600-variant row needs two shards.
    monkeypatch.setattr(
        store_arrays, "INDEXED_SUBSET_SHARD", store_arrays.INDEXED_SUBSET_CHUNK
    )
    store = shutil.copytree(big_store, tmp_path / "store.opengwasdb")
    _build(store, "big", _big_variant_list(tmp_path))

    group = _index_group(store, "big")
    assert tuple(group["z"].shards) == (1, store_arrays.INDEXED_SUBSET_CHUNK)
    assert group["z"].shape[1] > group["z"].shards[1], "the fixture must span >1 shard"

    subset = open_indexed_subset(store, "big")
    encoding = StoreManifest.load(store).encoding
    primary = DenseZPlane.open(_root(store), encoding).column(0)
    np.testing.assert_array_equal(subset.decode_analysis(0).z, primary[subset.variant_index])
    assert validate_store(store).ok


# ── Standalone validation ───────────────────────────────────────────────────


def _errors_for(store: Path) -> list[str]:
    return validate_store(store).errors


def test_validation_rejects_a_temporary_group(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    namespace = zarr.open_group(
        str(store / "data.zarr" / INDEXED_SUBSETS_GROUP), mode="r+"
    )
    namespace.create_group(".hm3.tmp.1234.abcd")

    errors = _errors_for(store)
    assert any("temporary" in error for error in errors), errors


def test_validation_rejects_an_unknown_entry_in_the_namespace(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    (store / "data.zarr" / INDEXED_SUBSETS_GROUP / "stray.txt").write_text(
        "not a group", encoding="utf-8"
    )
    errors = _errors_for(store)
    assert any("stray.txt" in error for error in errors), errors


def test_validation_rejects_an_unknown_array_in_a_subset(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    group.create_array(
        "surprise", shape=(1,), chunks=(1,), shards=(1,), dtype="int16"
    )
    errors = _errors_for(store)
    assert any("surprise" in error for error in errors), errors


def test_validation_rejects_a_stale_release_identity(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    group.attrs["source_release_id"] = "some-other-release"
    errors = _errors_for(store)
    assert any("release" in error for error in errors), errors


def test_validation_rejects_unsorted_and_duplicated_variant_indices(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    group["variant_index"][:] = np.array([5, 5, 11, 20, 30, 42], dtype="int32")
    errors = _errors_for(store)
    assert any("sorted" in error for error in errors), errors


def test_validation_rejects_out_of_bounds_variant_indices(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    values = np.asarray(group["variant_index"][:])
    values[-1] = 10_000_000
    group["variant_index"][:] = values
    errors = _errors_for(store)
    assert any("bounds" in error for error in errors), errors


def test_validation_rejects_a_wrong_plane_shape(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    del group["z"]
    group.create_array("z", shape=(N_ANALYSES, 3), chunks=(1, 3), shards=(1, 3), dtype="int16")
    errors = _errors_for(store)
    assert any("shape" in error for error in errors), errors


def test_validation_rejects_a_corrupted_value(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    values = np.asarray(group["z"][:])
    values[0, 0] = values[0, 0] + 1
    group["z"][:] = values
    errors = _errors_for(store)
    assert any("z values" in error for error in errors), errors


def test_validation_rejects_a_missing_overflow_table(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    del group["z_overflow_index"]
    errors = _errors_for(store)
    assert any("z_overflow_index" in error for error in errors), errors


def test_validation_rejects_a_missing_eaf_baseline(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    del group["eaf_baseline"]
    errors = _errors_for(store)
    assert any("eaf_baseline" in error for error in errors), errors


def test_validation_rejects_z_se_missingness_disagreement(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    # Mark one present z cell missing without touching se.
    values = np.asarray(group["z"][:])
    values[0, 1] = -32768
    group["z"][:] = values
    # ... and one se cell missing without touching z.
    se_values = np.asarray(group["se"][:])
    se_values[2, 2] = -128
    group["se"][:] = se_values
    errors = _errors_for(store)
    assert any("missingness" in error for error in errors), errors


def test_invalid_group_attributes_fail_validation(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    del group.attrs["input_sha256"]
    errors = _errors_for(store)
    assert any("input_sha256" in error for error in errors), errors


def test_validation_rejects_an_explicit_empty_group(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    """An empty directory-turned-group is not a published subset."""
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    namespace = zarr.open_group(
        str(store / "data.zarr" / INDEXED_SUBSETS_GROUP), mode="r+"
    )
    namespace.create_group("empty")

    errors = _errors_for(store)
    assert any("explicit empty group" in error for error in errors), errors


def test_validation_rejects_a_zero_variant_subset(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    """A zero-length Variant Index is rejected, not accepted as an empty index."""
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    del group["variant_index"]
    group.create_array(
        "variant_index", shape=(0,), chunks=(1,), shards=(1,), dtype="int32"
    )
    group.attrs["n_subset_variants"] = 0

    errors = _errors_for(store)
    assert any("contains no variants" in error for error in errors), errors


def test_remove_refuses_a_consolidated_namespace(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    """A raw rename would leave the consolidated record listing the deleted group."""
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    zarr.consolidate_metadata(str(store / "data.zarr"), zarr_format=3)
    published = store / "data.zarr" / INDEXED_SUBSETS_GROUP / "hm3"
    before = _digest(published)

    listed = zarr.open_group(str(store / "data.zarr"), mode="r")
    assert "hm3" in listed[INDEXED_SUBSETS_GROUP], (
        "the consolidated record must list the subset for this to mean anything"
    )

    with pytest.raises(ConsolidatedMetadataError):
        remove_indexed_subset(store, "hm3")

    assert list_indexed_subsets(store) == ("hm3",)
    assert _digest(published) == before, "a refused removal must change nothing"
    reread = zarr.open_group(str(store / "data.zarr"), mode="r")
    assert "hm3" in reread[INDEXED_SUBSETS_GROUP]


def test_open_refuses_a_subset_from_another_release(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    """The read seam, which #265 builds on, rejects a stale index itself."""
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    group.attrs["source_release_id"] = "another-release"

    with pytest.raises(IndexedSubsetStaleError, match="stale"):
        open_indexed_subset(store, "hm3")


def test_validation_rejects_a_corrupted_se_value(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    values = np.asarray(group["se"][:])
    values[0, 0] = values[0, 0] + 1
    group["se"][:] = values
    errors = _errors_for(store)
    assert any("se values" in error for error in errors), errors


def test_validation_rejects_a_corrupted_eaf_value(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    values = np.asarray(group["eaf"][:])
    values[0, 0] = values[0, 0] + 1
    group["eaf"][:] = values
    errors = _errors_for(store)
    assert any("eaf values" in error for error in errors), errors


def test_validation_rejects_a_corrupted_eaf_baseline(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    baseline = np.asarray(group["eaf_baseline"][:])
    baseline[0] = baseline[0] + 0.1
    group["eaf_baseline"][:] = baseline
    errors = _errors_for(store)
    assert errors, "a corrupted baseline must fail validation"


def test_validation_rejects_a_wrong_plane_dtype(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    n_analyses, n_subset = np.asarray(group["se"].shape)
    del group["se"]
    group.create_array(
        "se", shape=(n_analyses, n_subset), chunks=(1, n_subset), shards=(1, n_subset),
        dtype="float16",
    )
    errors = _errors_for(store)
    assert any("dtype" in error for error in errors), errors


def _recreate_array(
    group: zarr.Group,
    name: str,
    *,
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None = None,
    dtype: str | None = None,
) -> None:
    """Replace an index array with the same data in an explicitly wrong layout.

    The corruption tests are the one place the suite reaches into the Zarr group
    by hand, so this keeps the damaged group readable and lets the rule under
    test be the one that fails rather than a decode error.
    """
    data = np.asarray(group[name][:])
    if dtype is not None:
        data = data.astype(dtype)
    del group[name]
    group.create_array(name, data=data, chunks=chunks, shards=shards)


def test_validation_rejects_a_plane_chunk_that_spans_analyses(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    """A chunk wider than one Analysis defeats the index's narrow-read contract."""
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    n_analyses, n_subset = (int(size) for size in group["z"].shape)
    _recreate_array(
        group, "z", chunks=(n_analyses, n_subset), shards=(n_analyses, n_subset)
    )

    errors = _errors_for(store)
    assert any(
        "z" in error and "chunk" in error and "one Analysis row" in error
        for error in errors
    ), errors
    with pytest.raises(IndexedSubsetError, match="one Analysis row"):
        open_indexed_subset(store, "hm3")


def test_validation_rejects_a_plane_shard_that_spans_analyses(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    """A shard spanning Analyses makes a one-Analysis read decompress another's cells."""
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    n_analyses, n_subset = (int(size) for size in group["z"].shape)
    _recreate_array(
        group, "z", chunks=(1, n_subset), shards=(n_analyses, n_subset)
    )

    errors = _errors_for(store)
    assert any("z" in error and "shard" in error for error in errors), errors
    with pytest.raises(IndexedSubsetError, match="shard"):
        open_indexed_subset(store, "hm3")


def test_validation_rejects_a_mis_chunked_per_variant_side_array(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    """A per-variant side array's chunk is part of the format, not a free choice."""
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    n_subset = int(group.attrs["n_subset_variants"])
    _recreate_array(group, "eaf_baseline", chunks=(1,), shards=(n_subset,))

    errors = _errors_for(store)
    assert any("eaf_baseline" in error for error in errors), errors
    assert not any("Analysis row" in error for error in errors), (
        "eaf_baseline is a 1-D per-variant side array, not an Analysis-major plane, "
        "so its chunk mismatch must not claim a one-Analysis-row rule"
    )
    with pytest.raises(IndexedSubsetError, match="eaf_baseline") as excinfo:
        open_indexed_subset(store, "hm3")
    assert "Analysis row" not in str(excinfo.value)


@pytest.mark.parametrize("dtype", ["float32", "int64"])
def test_validation_requires_an_int32_variant_index(
    tmp_path: Path, rich_store: Path, variant_list: Path, dtype: str
) -> None:
    """The Variant Index is stored int32; a widened or float axis is refused."""
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    _build(store, "hm3", variant_list)
    group = _index_group(store, "hm3")
    n_subset = int(group.attrs["n_subset_variants"])
    _recreate_array(
        group, "variant_index", chunks=(n_subset,), shards=(n_subset,), dtype=dtype
    )

    errors = _errors_for(store)
    assert any("variant_index" in error and "int32" in error for error in errors), errors
    with pytest.raises(IndexedSubsetError, match="int32"):
        open_indexed_subset(store, "hm3")


def test_build_rejects_an_unsafe_name(rich_store: Path, variant_list: Path) -> None:
    with pytest.raises(IndexedSubsetNameError):
        build_indexed_subset(
            rich_store, "../escape", variant_list, reference_assembly="GRCh38"
        )


# ── CLI ─────────────────────────────────────────────────────────────────────


def _invoke_build_cli(
    store: Path, name: str, variant_list: Path, assembly: str
) -> Result:
    return CliRunner().invoke(
        app,
        [
            "build-indexed-subset",
            str(store),
            name,
            "--variant-list",
            str(variant_list),
            "--reference-assembly",
            assembly,
        ],
    )


def test_cli_builds_an_indexed_subset(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    result = _invoke_build_cli(store, "hm3", variant_list, "GRCh38")
    assert result.exit_code == 0, result.output
    assert list_indexed_subsets(store) == ("hm3",)


def test_cli_refuses_an_assembly_mismatch(
    tmp_path: Path, rich_store: Path, variant_list: Path
) -> None:
    store = shutil.copytree(rich_store, tmp_path / "store.opengwasdb")
    result = _invoke_build_cli(store, "hm3", variant_list, "GRCh37")
    assert result.exit_code == 1
    assert "assembly" in result.output
    assert list_indexed_subsets(store) == ()
