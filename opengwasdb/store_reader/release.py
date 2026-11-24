"""A Store Release, opened read-only: its paths, its manifest and its arrays.

`open_release` reads a release's manifest and refuses a `format_version` this
build cannot interpret before anything else is read (ADR 0038 §2).
`StoreRelease` is the opened release every reader holds: where its artifacts
live, its Zarr data group opened read-only, its Hybrid Dense Component, its
Variant Axis, and the query facade over it.

From format 0.3.0 a Hybrid's Dense Component carries no Variant Axis and no
`analyses.tsv` of its own (#472): opened as a release, its `variant_axis` is the
shared one seen through `dense_to_shared.npy` (`panel_axis.PanelVariantAxis`), and
its `analyses_path` is the root's. Every reader of the component -- the query
facade, completion, the validator, the Indexed Variant Subset builder -- gets
them from here, so none reads a second copy.

Writing to a release -- staging, publishing, amending its provenance, opening
its arrays for writing or its `index.sqlite` -- is `opengwasdb.store.open`'s,
whose `OpenGWASDBStore` is this class with those added (#464). So is the Store
Variant Table's 0.2.x access (`opengwasdb.variants.legacy_table`): a release
opened with no `legacy` opener has no Variant Axis, and `variant_axis` says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from opengwasdb.store_reader.arrays import open_group
from opengwasdb.store_reader.enums import PrimaryStorageLayout
from opengwasdb.store_reader.format import check_format_version, dense_component_shares_axis
from opengwasdb.store_reader.hybrid import DENSE_TO_SHARED_FILENAME, hybrid_root_of
from opengwasdb.store_reader.manifest import StoreManifest
from opengwasdb.store_reader.panel_axis import PanelVariantAxis
from opengwasdb.store_reader.variant_axis import LegacyTableOpener, VariantAxis

if TYPE_CHECKING:
    from opengwasdb.store_reader.facade import HybridStoreQuery, RaggedStoreQuery, StoreQuery


def release_paths(path: Path) -> dict[str, Path]:
    return {
        "manifest": path / "manifest.json",
        "data": path / "data.zarr",
        "index": path / "index.sqlite",
        "analyses": path / "analyses.tsv",
        "variant_table": path / "variants.tsv.gz",
        "variant_tabix": path / "variants.tsv.gz.tbi",
        "variant_offsets": path / "variant_offsets.npy",
    }


def release_arrays(release: str | Path) -> Any:
    """The `data.zarr` group of the release at `release`, read-only.

    The one read-only open of a release's arrays: `StoreRelease.arrays` is this, for
    an opened release, and a reader holding only a path -- a staged release not yet
    published, a Hybrid's Dense Component, a builder adding to a release -- calls it
    directly (#473).
    """
    return open_group(release_paths(Path(release))["data"])


class ReleasePaths:
    """Artifact-path properties shared by an opened release and a staged one.

    Both ``StoreRelease`` (and so ``OpenGWASDBStore``) and ``StagedRelease`` set ``_paths`` (via
    ``release_paths()``) in ``__post_init__`` and inherit these instead of
    each re-deriving the same suffixes off ``self.path``.
    """

    _paths: dict[str, Path]

    @property
    def manifest_path(self) -> Path:
        return self._paths["manifest"]

    @property
    def data_path(self) -> Path:
        return self._paths["data"]

    @property
    def index_path(self) -> Path:
        return self._paths["index"]

    @property
    def analyses_path(self) -> Path:
        return self._paths["analyses"]

    @property
    def variant_table_path(self) -> Path:
        return self._paths["variant_table"]

    @property
    def variant_tabix_path(self) -> Path:
        return self._paths["variant_tabix"]

    @property
    def variant_offsets_path(self) -> Path:
        return self._paths["variant_offsets"]


@dataclass(frozen=True)
class StoreRelease(ReleasePaths):
    """A local Store Release opened from an explicit path, read-only.

    Frozen: the manifest a caller holds never changes under them. `legacy` opens
    the release's 0.2.x Store Variant Table (`LegacyTableOpener`); it is carried
    to the Hybrid Dense Component, and it is what `variant_axis` reads through.
    """

    path: Path
    manifest: StoreManifest
    legacy: LegacyTableOpener | None = field(default=None, repr=False, compare=False)
    _paths: dict[str, Path] = field(init=False, repr=False)
    _shared_root: Path | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_paths", release_paths(self.path))
        root = shares_root_of(self.path, self.manifest)
        object.__setattr__(self, "_shared_root", root)
        if root is not None:
            self._paths["analyses"] = release_paths(root)["analyses"]

    @property
    def shared_root(self) -> Path | None:
        """The Hybrid release whose Variant Axis and `analyses.tsv` this one reads: set
        for a Dense Component from format 0.3.0 (#472), None for every other release."""
        return self._shared_root

    def dense_component(self) -> StoreRelease:
        """Open a Hybrid release's nested Dense Component release.

        The Dense Component (ADR 0026) is a Dense Store Release at
        ``<store>/dense``, built and completed by the unchanged dense machinery --
        it is opened the same way any release is. From 0.3.0 its Variant Axis and
        `analyses_path` are its Hybrid's (`shared_root`, #472).
        """
        self._require_hybrid()
        return open_release(self.path / "dense", legacy=self.legacy)

    def _require_hybrid(self) -> None:
        if self.manifest.primary_layout is not PrimaryStorageLayout.HYBRID:
            raise ValueError(
                "dense_component() is only valid for a Hybrid release "
                f"(primary_layout={self.manifest.primary_layout})"
            )

    def arrays(self) -> Any:
        """Open this release's ``data.zarr`` group read-only (`release_arrays`)."""
        return release_arrays(self.path)

    def variant_axis(self, *, aliases: bool) -> VariantAxis:
        """This release's Variant Axis; `aliases` also consults `variant_aliases`."""
        if self.legacy is None:
            raise ValueError(
                f"{self.path} was opened without a reader for its Store Variant Table "
                "(a 0.2.x BGZF/tabix file); open it with opengwasdb.store.open.open_store, "
                "or pass open_release(..., legacy=...)"
            )
        if self._shared_root is not None:
            # The Dense Component's rows, named by the shared axis: one ALID, position
            # and rsID index, the root's (#472). A 0.3.0 release has no `variant_aliases`.
            root = open_release(self._shared_root, legacy=self.legacy)
            return PanelVariantAxis(
                root.variant_axis(aliases=False),
                np.load(self.path / DENSE_TO_SHARED_FILENAME, mmap_mode="r"),
            )
        return VariantAxis(
            self.path,
            self.legacy(self.path, aliases=aliases),
            format_version=self.manifest.format_version,
        )

    def query(self) -> StoreQuery | RaggedStoreQuery | HybridStoreQuery:
        """The query facade over this release, by its layout (ADR 0033)."""
        from opengwasdb.store_reader.facade import query_release

        return query_release(self)


def shares_root_of(path: Path, manifest: StoreManifest) -> Path | None:
    """The Hybrid root a release at `path` reads its variants and Analyses from: its
    nesting release, when it is a Dense Component of a version that carries no copy of
    its own (`dense_component_shares_axis`); otherwise None."""
    if manifest.primary_layout is not PrimaryStorageLayout.DENSE:
        return None
    if not dense_component_shares_axis(manifest.format_version):
        return None
    return hybrid_root_of(path)


def open_release(path: str | Path, *, legacy: LegacyTableOpener | None = None) -> StoreRelease:
    """Open a local Store Release directory, read-only.

    The function intentionally opens exactly the path supplied by the
    caller. Release discovery and default selection belong to a higher-level
    catalogue. Raises ``UnsupportedFormatVersion`` if the release declares a
    format_version this build cannot interpret (ADR 0038 §2).
    """
    store_path = Path(path)
    manifest = StoreManifest.load(store_path)
    check_format_version(manifest.format_version, source=f"release at {store_path}")
    return StoreRelease(path=store_path, manifest=manifest, legacy=legacy)
