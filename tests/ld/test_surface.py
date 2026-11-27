"""The public surface, pinned signature by signature.

What a caller of `opengwasdb.ld` sees of opengwasdb's own is pinned here: `opengwasdb.ld.open`
(the dispatcher that also opens a Genotype Store), `opengwasdb.ld.build` and its inputs,
`opengwasdb.ld.validate` and its `Finding`, the sorted identity a build spells an LD Store's ALIDs
by (`opengwasdb.variants.sorted_spelling`) and the CLI's entry. The four queries, their results,
`Refusal` and the read side of the identity (`normalise_chromosome`, `sorted_alid_locus`) are
ogquery's objects since opengwas/ogquery#2, re-exported here, and ogquery's own `tests/ld`
pins them; `tests/ld/test_read_side_is_ogquery.py` holds that they are the very objects.
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
from opengwasdb.variants import sorted_spelling

#: The committed Test Store (tests/ld/contract/support.py), written by the walking skeleton.
TEST_STORE = Path(__file__).resolve().parent / "data" / "test-panel" / "store"

SURFACE: dict[str, tuple[Callable[..., Any], str]] = {
    # ogquery's object since opengwas/ogquery#11, but downstream code calls it by this name: its
    # signature is pinned here whoever owns it (`REEXPORTED`).
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
    "opengwasdb.ld.Finding": (
        opengwasdb.ld.Finding,
        "(rule: 'str', where: 'str', message: 'str') -> None",
    ),
    "opengwasdb.variants.sorted_spelling.sorted_identity": (
        sorted_spelling.sorted_identity,
        "(chrom: 'str', position: 'int', ref: 'str', alt: 'str') -> 'SortedIdentity'",
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


#: Pinned here though ogquery defines them: the public name is opengwasdb's, and callers rely on it.
REEXPORTED = {"opengwasdb.ld.open": "ogquery.ld.query.ldstore"}


def test_the_pins_are_opengwasdb_s_own_or_a_named_reexport() -> None:
    """A pin here is for an object opengwasdb defines, or one it re-exports by name (`REEXPORTED`);
    ogquery pins the rest of its own."""
    for name, (function, _) in SURFACE.items():
        owner = REEXPORTED.get(name, "opengwasdb.")
        assert function.__module__.startswith(owner), (name, function.__module__)


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
        "from ogquery.opening import GenotypeExtraMissing\n"
        "try:\n"
        "    Genotypes(opened)\n"
        "except GenotypeExtraMissing as refusal:\n"
        "    print(refusal)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert "pip install 'ogquery[genotype]'" in done.stdout
