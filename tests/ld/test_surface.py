"""The public surface, pinned signature by signature.

What a caller of ogld sees is pinned here: `opengwasdb.ld.open`, the four queries and their results,
`opengwasdb.ld.build` and its inputs, `opengwasdb.ld.validate`, the identity functions an LD
Store's ALIDs are spelt by (`opengwasdb.variants.sorted_spelling`) and the CLI's entry.
Changing an entry is an API change, made on purpose, with a changelog fragment that says so.
cleat's public-api gate records only a declaration's first line at column 0, so it cannot see
a method or a signature that ruff wraps; this test sees both.

The seams inside the package are not pinned (owner, 10 November 2026): `StoreReader`, which
the queries read through, and `StoreWriter`, which a build writes through. They were pinned
while epic opengwas/ogld#1's first tickets were built in parallel against them. They now move with
what the reads and the builds are measured to need, held by the contract tests (tests/ld/contract/)
and by `docs/architecture/ownership.md`, which says what each module owns.
"""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import testdata

import opengwasdb.ld
from opengwasdb.ld import cli
from opengwasdb.store_reader import identity
from opengwasdb.variants import sorted_spelling

#: The committed Test Store (tests/ld/contract/support.py), written by the walking skeleton.
TEST_STORE = Path(__file__).resolve().parent / "data" / "test-panel" / "store"

SURFACE: dict[str, tuple[Callable[..., Any], str]] = {
    "opengwasdb.ld.open": (
        opengwasdb.ld.open,
        "(location: 'str | os.PathLike[str]', *, cache_bytes: 'int | None' = None) -> 'LDStore'",
    ),
    "opengwasdb.ld.build": (
        opengwasdb.ld.build,
        "(inputs: 'BuildInputs', out: 'str | os.PathLike[str]', *, threads: 'int' = 1)"
        " -> 'Provenance'",
    ),
    "opengwasdb.ld.validate": (
        opengwasdb.ld.validate,
        "(location: 'str | os.PathLike[str]') -> 'dict[str, list[Finding]]'",
    ),
    "opengwasdb.ld.BuildInputs": (
        opengwasdb.ld.BuildInputs,
        "(pgen: 'tuple[Path, ...]', keep: 'Path', pcs: 'Path | None', pc_source: 'str | None', "
        "panel_id: 'str', ancestry: 'str', genotype_source: 'str', r_scale: 'int', "
        "t_partner: 'float', k: 'int' = 10, k_min: 'int' = 5, mac_min: 'int' = 20, "
        "info_min: 'float | None' = 0.8) -> None",
    ),
    "opengwasdb.ld.LDStore.matrix": (
        opengwasdb.ld.LDStore.matrix,
        "(self, alids: 'Sequence[str]', *, fill: 'float | None' = None) -> 'LDMatrix'",
    ),
    "opengwasdb.ld.LDStore.region": (
        opengwasdb.ld.LDStore.region,
        "(self, chrom: 'str', centre: 'int', radius: 'int', *, fill: 'float | None' = None)"
        " -> 'LDMatrix'",
    ),
    "opengwasdb.ld.LDStore.proxies": (
        opengwasdb.ld.LDStore.proxies,
        "(self, alids: 'Sequence[str]', *, r2: 'float' = 0.8, kb: 'float' = 10000) -> 'Proxies'",
    ),
    "opengwasdb.ld.LDStore.clump": (
        opengwasdb.ld.LDStore.clump,
        "(self, alids: 'Sequence[str]', score: 'npt.ArrayLike', *, r2: 'float', kb: 'float')"
        " -> 'Clumps'",
    ),
    "opengwasdb.ld.LDMatrix": (
        opengwasdb.ld.LDMatrix,
        "(r: 'npt.NDArray[np.float64]', variants: 'Variants', absent: 'tuple[str, ...]', "
        "t_far: 'float', near_bp: 'int', far_bp: 'int', fill: 'float | None') -> None",
    ),
    "opengwasdb.ld.LDMatrix.kinds": (
        opengwasdb.ld.LDMatrix.kinds,
        "(self) -> 'npt.NDArray[np.str_]'",
    ),
    "opengwasdb.ld.Proxies": (
        opengwasdb.ld.Proxies,
        "(query: 'npt.NDArray[np.str_]', proxy: 'npt.NDArray[np.str_]', "
        "r: 'npt.NDArray[np.float64]', r2: 'npt.NDArray[np.float64]', "
        "distance: 'npt.NDArray[np.int64]', "
        "absent: 'tuple[str, ...]', r2_min: 'float', window_bp: 'float', t_far: 'float') -> None",
    ),
    "opengwasdb.ld.Clumps": (
        opengwasdb.ld.Clumps,
        "(alid: 'npt.NDArray[np.str_]', chrom: 'npt.NDArray[np.str_]', "
        "position: 'npt.NDArray[np.int64]', score: 'npt.NDArray[np.float64]', "
        "lead: 'npt.NDArray[np.str_]', r: 'npt.NDArray[np.float64]', absent: 'tuple[str, ...]', "
        "r2: 'float', window_bp: 'float') -> None",
    ),
    "opengwasdb.ld.Clumps.leads": (opengwasdb.ld.Clumps.leads, "(self) -> 'npt.NDArray[np.str_]'"),
    "opengwasdb.ld.Variants": (
        opengwasdb.ld.Variants,
        "(chrom: 'npt.NDArray[np.str_]', position: 'npt.NDArray[np.int64]', "
        "alid: 'npt.NDArray[np.str_]', a1: 'npt.NDArray[np.str_]', a2: 'npt.NDArray[np.str_]', "
        "a1_frequency: 'npt.NDArray[np.float64]') -> None",
    ),
    "opengwasdb.ld.Finding": (
        opengwasdb.ld.Finding,
        "(rule: 'str', where: 'str', message: 'str') -> None",
    ),
    "opengwasdb.ld.Refusal": (
        opengwasdb.ld.Refusal,
        "(reason: 'RefusalReason', detail: 'str') -> 'None'",
    ),
    "opengwasdb.variants.sorted_spelling.sorted_identity": (
        sorted_spelling.sorted_identity,
        "(chrom: 'str', position: 'int', ref: 'str', alt: 'str') -> 'SortedIdentity'",
    ),
    "opengwasdb.store_reader.identity.normalise_chromosome": (
        identity.normalise_chromosome,
        "(chromosome: 'str') -> 'str'",
    ),
    "opengwasdb.variants.sorted_spelling.sorted_alid_locus": (
        sorted_spelling.sorted_alid_locus,
        "(alid: 'str') -> 'tuple[str, int] | None'",
    ),
    "opengwasdb.variants.sorted_spelling.SortedIdentity": (
        sorted_spelling.SortedIdentity,
        "(chrom: 'str', position: 'int', alid: 'str', a1: 'str', a2: 'str', a1_is_alt: 'bool')"
        " -> None",
    ),
    "opengwasdb.ld.cli.main": (cli.main, "(argv: 'Sequence[str] | None' = None) -> 'int'"),
}


@pytest.mark.parametrize("name", sorted(SURFACE))
def test_the_signature_is_the_recorded_one(name: str) -> None:
    function, signature = SURFACE[name]
    assert str(inspect.signature(function)) == signature


def test_the_top_level_names_are_the_public_surface() -> None:
    expected = (
        "AnswerKind BuildInputs Clumps Finding InputError LDMatrix LDStore Provenance Proxies "
        "Refusal RefusalReason Variants __version__ build open validate"
    )
    assert " ".join(sorted(opengwasdb.ld.__all__)) == expected
    for name in opengwasdb.ld.__all__:
        assert hasattr(opengwasdb.ld, name), name


def test_no_body_in_ogld_is_a_stub() -> None:
    # Epic opengwas/ogld#1 built every public body (opengwas/ogld#3–opengwas/ogld#6,
    # opengwas/ogld#12–opengwas/ogld#17); a stub raising NotImplementedError(…) would be a call that
    # looks wired and is not. An abstract method's bare `raise NotImplementedError` is not one.
    package = Path(opengwasdb.ld.__file__).parent
    stubs = [
        f"{path.relative_to(package)}:{n}"
        for path in sorted(package.rglob("*.py"))
        for n, line in enumerate(path.read_text().splitlines(), start=1)
        if "raise NotImplementedError(" in line
    ]
    assert stubs == [], f"stubs in ogld/: {stubs}"


def test_ogld_imports_and_reads_the_test_store_without_pgenlib() -> None:
    # pgenlib is the builder's pgen reader. A reader-only install may lack it, so nothing
    # outside the builder's read path may import it: the committed Test Store is opened and
    # read with pgenlib unimportable, and any error fails the test.
    code = (
        "import sys\n"
        "sys.modules['pgenlib'] = None\n"
        "import opengwasdb.ld\n"
        f"store = opengwasdb.ld.open({str(TEST_STORE)!r})\n"
        "store.region('22', 30_000_000, 100_000)\n"
        "print('read', store.provenance['n_variants'])\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert re.fullmatch(r"read \d+", done.stdout.strip()), done.stdout


def test_a_pgen_read_without_pgenlib_refuses_naming_the_extra() -> None:
    panel = testdata.panel()
    code = (
        "import sys\n"
        "sys.modules['pgenlib'] = None\n"
        "from pathlib import Path\n"
        "import opengwasdb.ld\n"
        "from opengwasdb.ld.genotypes import Genotypes\n"
        "from opengwasdb.ld.build.inputs import fileset, read_keep\n"
        f"pgen, keep = Path({str(panel.pgen)!r}), Path({str(panel.keep)!r})\n"
        "opened = fileset(pgen, read_keep(keep, 'p'), None, 'p')\n"
        "try:\n"
        "    Genotypes(opened)\n"
        "except opengwasdb.ld.InputError as refusal:\n"
        "    print(refusal)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert "opengwasdb[ld-build]" in done.stdout
