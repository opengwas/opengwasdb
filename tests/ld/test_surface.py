"""The public surface and the seams between tickets, pinned signature by signature.

Epic #1's tickets build against these signatures in parallel: ogld#4 implements the store
seams that ogld#5 writes through and ogld#6 reads through. A ticket's job is to make its
contract tests (tests/contract/) pass without changing any of these. cleat's public-api
gate records only a declaration's first line at column 0, so it cannot see a method or a
signature that ruff wraps; this test sees both. Changing an entry is an API change, made on
purpose, with a changelog fragment that says so.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pytest

import ogld
from ogld import cli, identity, store

SURFACE: dict[str, tuple[Callable[..., Any], str]] = {
    "ogld.open": (ogld.open, "(location: 'str | os.PathLike[str]') -> 'LDStore'"),
    "ogld.build": (
        ogld.build,
        "(inputs: 'BuildInputs', out: 'str | os.PathLike[str]', *, threads: 'int' = 1)"
        " -> 'Provenance'",
    ),
    "ogld.validate": (
        ogld.validate,
        "(location: 'str | os.PathLike[str]') -> 'dict[str, list[Finding]]'",
    ),
    "ogld.BuildInputs": (
        ogld.BuildInputs,
        "(pgen: 'tuple[Path, ...]', keep: 'Path', pcs: 'Path | None', pc_source: 'str | None', "
        "panel_id: 'str', ancestry: 'str', genotype_source: 'str', k: 'int' = 10, "
        "mac_min: 'int' = 20, info_min: 'float | None' = 0.8) -> None",
    ),
    "ogld.LDStore.matrix": (
        ogld.LDStore.matrix,
        "(self, alids: 'Sequence[str]', *, fill: 'float | None' = None) -> 'LDMatrix'",
    ),
    "ogld.LDStore.region": (
        ogld.LDStore.region,
        "(self, chrom: 'str', centre: 'int', radius: 'int', *, fill: 'float | None' = None)"
        " -> 'LDMatrix'",
    ),
    "ogld.LDStore.proxies": (
        ogld.LDStore.proxies,
        "(self, alids: 'Sequence[str]', *, r2: 'float' = 0.8, kb: 'float' = 10000) -> 'Proxies'",
    ),
    "ogld.LDStore.clump": (
        ogld.LDStore.clump,
        "(self, alids: 'Sequence[str]', score: 'npt.ArrayLike', *, r2: 'float', kb: 'float')"
        " -> 'Clumps'",
    ),
    "ogld.LDMatrix": (
        ogld.LDMatrix,
        "(r: 'npt.NDArray[np.float64]', variants: 'Variants', absent: 'tuple[str, ...]', "
        "t_far: 'float', near_bp: 'int', far_bp: 'int', fill: 'float | None') -> None",
    ),
    "ogld.LDMatrix.kinds": (ogld.LDMatrix.kinds, "(self) -> 'npt.NDArray[np.str_]'"),
    "ogld.Proxies": (
        ogld.Proxies,
        "(query: 'npt.NDArray[Any]', proxy: 'npt.NDArray[Any]', r: 'npt.NDArray[np.float64]', "
        "r2: 'npt.NDArray[np.float64]', distance: 'npt.NDArray[np.int64]', "
        "absent: 'tuple[str, ...]', r2_min: 'float', window_bp: 'float', t_far: 'float') -> None",
    ),
    "ogld.Clumps": (
        ogld.Clumps,
        "(alid: 'npt.NDArray[Any]', chrom: 'npt.NDArray[np.str_]', "
        "position: 'npt.NDArray[np.int64]', score: 'npt.NDArray[np.float64]', "
        "lead: 'npt.NDArray[Any]', r: 'npt.NDArray[np.float64]', absent: 'tuple[str, ...]', "
        "r2: 'float', window_bp: 'float') -> None",
    ),
    "ogld.Clumps.leads": (ogld.Clumps.leads, "(self) -> 'npt.NDArray[Any]'"),
    "ogld.Variants": (
        ogld.Variants,
        "(chrom: 'npt.NDArray[np.str_]', position: 'npt.NDArray[np.int64]', "
        "alid: 'npt.NDArray[Any]', a1: 'npt.NDArray[Any]', a2: 'npt.NDArray[Any]', "
        "a1_frequency: 'npt.NDArray[np.float64]') -> None",
    ),
    "ogld.Finding": (ogld.Finding, "(rule: 'str', where: 'str', message: 'str') -> None"),
    "ogld.Refusal": (ogld.Refusal, "(reason: 'RefusalReason', detail: 'str') -> 'None'"),
    "ogld.identity.orient": (
        identity.orient,
        "(chrom: 'str', position: 'int', ref: 'str', alt: 'str') -> 'Orientation'",
    ),
    "ogld.identity.normalise_chromosome": (
        identity.normalise_chromosome,
        "(label: 'str') -> 'str'",
    ),
    "ogld.identity.alid_locus": (identity.alid_locus, "(alid: 'str') -> 'tuple[str, int] | None'"),
    "ogld.identity.Orientation": (
        identity.Orientation,
        "(chrom: 'str', position: 'int', alid: 'str', a1: 'str', a2: 'str', a1_is_alt: 'bool')"
        " -> None",
    ),
    "ogld.store.StoreReader.open": (
        store.StoreReader.open,
        "(location: 'str | os.PathLike[str]') -> 'StoreReader'",
    ),
    "ogld.store.StoreReader.rows_of": (
        store.StoreReader.rows_of,
        "(self, alids: 'Sequence[str]') -> 'npt.NDArray[np.int64]'",
    ),
    "ogld.store.StoreReader.rows_in": (
        store.StoreReader.rows_in,
        "(self, chrom: 'str', start: 'int', stop: 'int') -> 'npt.NDArray[np.int64]'",
    ),
    "ogld.store.StoreReader.variants": (
        store.StoreReader.variants,
        "(self, rows: 'npt.NDArray[np.int64]') -> 'Variants'",
    ),
    "ogld.store.StoreReader.pair_r": (
        store.StoreReader.pair_r,
        "(self, rows: 'npt.NDArray[np.int64]') -> 'npt.NDArray[np.float64]'",
    ),
    "ogld.store.StoreReader.index": (
        store.StoreReader.index,
        "(self, rows: 'npt.NDArray[np.int64]') -> 'Partners'",
    ),
    "ogld.store.StoreReader.band": (
        store.StoreReader.band,
        "(self, rows: 'npt.NDArray[np.int64]') -> 'Partners'",
    ),
    "ogld.store.StoreWriter.create": (
        store.StoreWriter.create,
        "(location: 'str | os.PathLike[str]', *, variants: 'Variants', near_bp: 'int', "
        "far_bp: 'int') -> 'StoreWriter'",
    ),
    "ogld.store.StoreWriter.write_near": (
        store.StoreWriter.write_near,
        "(self, first_row: 'int', lengths: 'npt.NDArray[np.int64]', "
        "codes: 'npt.NDArray[np.int16]') -> 'None'",
    ),
    "ogld.store.StoreWriter.write_index": (
        store.StoreWriter.write_index,
        "(self, chrom: 'str', indptr: 'npt.NDArray[np.int64]', partner: 'npt.NDArray[np.int64]', "
        "codes: 'npt.NDArray[np.int16]') -> 'None'",
    ),
    "ogld.store.StoreWriter.finish": (
        store.StoreWriter.finish,
        "(self, provenance: 'Provenance') -> 'Provenance'",
    ),
    "ogld.store.Partners": (
        store.Partners,
        "(indptr: 'npt.NDArray[np.int64]', partner: 'npt.NDArray[np.int64]', "
        "r: 'npt.NDArray[np.float64]') -> None",
    ),
    "ogld.cli.main": (cli.main, "(argv: 'Sequence[str] | None' = None) -> 'int'"),
}


@pytest.mark.parametrize("name", sorted(SURFACE))
def test_the_signature_is_the_recorded_one(name: str) -> None:
    function, signature = SURFACE[name]
    assert str(inspect.signature(function)) == signature


def test_the_top_level_names_are_the_public_surface() -> None:
    expected = (
        "AnswerKind BuildInputs Clumps Finding LDMatrix LDStore Provenance Proxies Refusal "
        "RefusalReason Variants __version__ build open validate"
    )
    assert " ".join(sorted(ogld.__all__)) == expected
    for name in ogld.__all__:
        assert hasattr(ogld, name), name


def test_every_unbuilt_body_names_the_ticket_that_builds_it() -> None:
    # A stub that raised a bare NotImplementedError would let a contract test xfail for the
    # wrong ticket; each names its own (ogld#3, #4, #5 or #6).
    unbuilt = {
        identity.orient: ("1", 1, "A", "G"),
        identity.normalise_chromosome: ("1",),
        identity.alid_locus: ("1:1:A:G",),
        store.StoreReader.open: ("x",),
        ogld.validate: ("x",),
        ogld.open: ("x",),
    }
    for function, args in unbuilt.items():
        with pytest.raises(NotImplementedError, match=r"^ogld#[3-6]$"):
            function(*args)
