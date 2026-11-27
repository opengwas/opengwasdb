"""Importing and querying through `opengwasdb.ld`'s read names loads none of the LD build code.

opengwas/ogquery#2: "Importing the LD read path loads none of the LD build code", on
opengwasdb's side. `opengwasdb.ld`'s initialisers are lazy (`opengwasdb.lazy_exports`), its read
names are ogquery's, and ogquery's `open` imports the Genotype Store reader only for a Genotype
Store (opengwas/ogquery#11). So a process that opens an LD Store or a Genotype Store and queries it
never imports the builder, the writer, the derivations, validation, the sidecar's writer or the CLI,
and one that opens an LD Store loads no genotype reader, pgenlib or threadpoolctl.

Each case runs in a fresh interpreter, because this process has imported everything. The same
run records zarr's configuration after the query: `opengwasdb.ld` reads under zarr's defaults,
as it did when its reader was in-tree, and `opengwasdb.clumps`, which also reads GWAS Stores,
under the GWAS read path's configuration (ogquery ADR 0002; opengwasdb ADR 0056).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from store_copies import TEST_STORE

#: Every module of the LD build, write, derive and validate side, and the CLI that runs them.
BUILD_SIDE = (
    "opengwasdb.ld.build",
    "opengwasdb.ld.cli",
    "opengwasdb.ld.query.validation",
    "opengwasdb.ld.store.assembly",
    "opengwasdb.ld.store.joining",
    "opengwasdb.ld.store.validation",
    "opengwasdb.ld.store.writer",
    "opengwasdb.ld.genotypes",
)

PROBE = """
import json, sys
{code}
import zarr
print(json.dumps({{
    "loaded": sorted(m for m in sys.modules if m.startswith(("opengwasdb.ld", "ogquery.ld"))),
    "native": sorted(m for m in ("pgenlib", "threadpoolctl") if m in sys.modules),
    "pipeline": zarr.config.get("codec_pipeline.path"),
}}))
"""

READ = """
from opengwasdb.ld import (
    AnswerKind, Clumps, InputError, LDMatrix, LDStore, Proxies, Provenance, Refusal,
    RefusalReason, Variants, open,
)
store = open({store!r})
alids = list(store.variants_in("22", 1, 2**31 - 1).alid[:20])
assert len(alids) == 20, "the probe must query something"
store.matrix(alids)
store.proxies(alids, r2=0.2, kb=1000)
store.clump(alids, [1.0] * len(alids), r2=0.2, kb=1000)
"""


def _probe(code: str) -> dict[str, object]:
    done = subprocess.run(
        [sys.executable, "-c", PROBE.format(code=code)],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert done.returncode == 0, done.stderr[-2000:]
    probed: dict[str, object] = json.loads(done.stdout.strip().splitlines()[-1])
    return probed


def _build_side(loaded: object) -> list[str]:
    assert isinstance(loaded, list)
    return [m for m in loaded if any(m == b or m.startswith(b + ".") for b in BUILD_SIDE)]


def test_an_ld_store_query_loads_none_of_the_build_and_leaves_zarr_s_defaults() -> None:
    probed = _probe(READ.format(store=str(TEST_STORE)))
    assert "ogquery.ld.query.ldstore" in probed["loaded"], "the probe read through opengwasdb.ld"
    assert _build_side(probed["loaded"]) == []
    assert "ogquery.ld.genotypes.reader" not in probed["loaded"], "an LD Store needs no pgen"
    assert probed["native"] == [], "an LD Store read imports neither pgenlib nor threadpoolctl"
    assert probed["pipeline"] == "zarr.core.codec_pipeline.BatchedCodecPipeline"


def test_a_genotype_store_query_loads_none_of_the_build(committed_genotype_store: Path) -> None:
    code = READ.format(store=str(committed_genotype_store))
    probed = _probe(code)
    assert "ogquery.ld.genotypes.reader" in probed["loaded"]
    assert _build_side(probed["loaded"]) == []
    assert probed["pipeline"] == "zarr.core.codec_pipeline.BatchedCodecPipeline"


@pytest.mark.parametrize(
    ("code", "pipeline"),
    [
        ("import opengwasdb.ld", "BatchedCodecPipeline"),
        ("import opengwasdb.ld.cli", "BatchedCodecPipeline"),
        ("import opengwasdb.ld; opengwasdb.ld.validate({store!r})", "BatchedCodecPipeline"),
        ("import opengwasdb.clumps", "FusedCodecPipeline"),
        ("import opengwasdb.cli.main", "FusedCodecPipeline"),
    ],
)
def test_each_path_runs_under_the_configuration_it_ran_under(code: str, pipeline: str) -> None:
    """The table opengwasdb's LD paths had while their reader was in-tree (c8ebe4ca)."""
    probed = _probe(code.format(store=str(TEST_STORE)))
    assert probed["pipeline"] == f"zarr.core.codec_pipeline.{pipeline}"


def test_the_check_sees_the_build_side_when_it_is_loaded() -> None:
    probed = _probe("import opengwasdb.ld; opengwasdb.ld.build")
    assert "opengwasdb.ld.build" in _build_side(probed["loaded"])


SUBMODULE_FIRST = """
import sys, types
{unguard}
import opengwasdb.ld.build
from opengwasdb.ld import build
print(type(build).__name__)
"""


@pytest.mark.parametrize(
    ("unguard", "kind"),
    [
        ("", "function"),
        # The control: the package as a plain module, as it is without `_Package`. Importing the
        # submodule first then binds it over the name, and `build` is a module, not callable.
        (
            "import opengwasdb.ld; sys.modules['opengwasdb.ld'].__class__ = types.ModuleType",
            "module",
        ),
    ],
)
def test_build_is_the_function_when_its_submodule_is_imported_first(
    unguard: str, kind: str
) -> None:
    """`opengwasdb.ld`'s names are lazy, and importing `opengwasdb.ld.build` binds that submodule
    on the package under the function's name. `_Package` keeps the function, as the eager
    initialiser did by rebinding it after the import; the control shows the test fails without."""
    done = subprocess.run(
        [sys.executable, "-c", SUBMODULE_FIRST.format(unguard=unguard)],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == kind
