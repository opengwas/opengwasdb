"""A Store Release, opened read-only: its paths, its manifest and its arrays.

`open_release` reads a release's manifest and refuses a `format_version` this
build cannot interpret before anything else is read (ADR 0038 §2).
`StoreRelease` is the opened release every reader holds: where its artifacts
live, its Zarr data group opened read-only, its Hybrid Dense Component, its
its Variant Axis, and the query facade over it.

Writing to a release -- staging, publishing, amending its provenance, opening
its arrays for writing or its `index.sqlite` -- is `opengwasdb.store.open`'s,
whose `OpenGWASDBStore` is this class with those added (#464). So is the Store
Variant Table's 0.2.x access: a release opened with no `legacy` opener has no
Variant Axis, and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opengwasdb.store_reader.arrays import open_group
from opengwasdb.store_reader.enums import PrimaryStorageLayout
from opengwasdb.store_reader.format import check_format_version
from opengwasdb.store_reader.manifest import StoreManifest
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "_paths", release_paths(self.path))

    def dense_component(self) -> StoreRelease:
        """Open a Hybrid release's nested Dense Component release.

        The Dense Component (ADR 0026) is a self-contained Dense Store
        Release at ``<store>/dense``, built and completed by the unchanged
        dense machinery -- it is opened the same way any release is.
        """
        self._require_hybrid()
        return open_release(self.path / "dense", legacy=self.legacy)

    def _require_hybrid(self) -> None:
        if self.manifest.primary_layout is not PrimaryStorageLayout.HYBRID:
            raise ValueError(
                "dense_component() is only valid for a Hybrid release "
                f"(primary_layout={self.manifest.primary_layout})"
            )

    def arrays(self, mode: str = "r") -> Any:
        """Open this release's ``data.zarr`` group read-only."""
        return open_group(self.data_path, mode)

    def variant_axis(self, *, aliases: bool) -> VariantAxis:
        """This release's Variant Axis; `aliases` also consults `variant_aliases`."""
        if self.legacy is None:
            raise ValueError(
                f"{self.path} was opened without a reader for its Store Variant Table "
                "(a 0.2.x BGZF/tabix file); open it with opengwasdb.store.open.open_store, "
                "or pass open_release(..., legacy=...)"
            )
        return VariantAxis(self.path, self.legacy(self.path, aliases=aliases))

    def query(self) -> StoreQuery | RaggedStoreQuery | HybridStoreQuery:
        """The query facade over this release, by its layout (ADR 0033)."""
        from opengwasdb.store_reader.facade import query_release

        return query_release(self)


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
