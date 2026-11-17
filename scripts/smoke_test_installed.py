#!/usr/bin/env python3
"""Use an installed opengwasdb the way a user would, outside any checkout (#438).

The test suite runs against Pixi's editable install of the working tree, so it
cannot see what a `pip install opengwasdb` user gets: a wheel that left out a
packaged resource, a dependency only the developer environment supplies, or a
version that is not the one the archive was built as. This runs with the
interpreter of a clean environment the wheel was installed into, and checks:

* `opengwasdb` is imported from that environment's site-packages, never from a
  checkout;
* it reports the version the wheel was built as;
* every built-in reader Column Map loads from the installed package;
* the installed `opengwasdb` command answers `--help`;
* the tiny Dense example builds, validates `valid`, and `query-phewas rs111`
  returns exactly the two rows docs/getting-started.md promises.

Run it isolated (`-I`), so neither the current directory nor the script's own
directory can put a checkout on the import path, and from a scratch directory:

    venv/bin/python -I scripts/smoke_test_installed.py \
        --fixture docs/examples/tiny --version 0.11.0

The fixture is copied, so the Store is built from files outside the checkout.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from importlib.metadata import version as distribution_version
from pathlib import Path

# The rows docs/getting-started.md shows for `query-phewas rs111 --variant-info`
# on the tiny example. A change here is a change to that page.
EXPECTED_PHEWAS = (
    "analysis_id\tanalysis_label\trsid\tchromosome\tposition\talid\teffect_allele"
    "\tother_allele\tz\tse\tp\teaf\tassociation_status\n"
    "height_eur\tHeight EUR pilot\trs111\t1\t100\t1:100:A:G\tA\tG\t2\t0.0999756"
    "\t0.0455\t.\tobserved\n"
    "ldl_eur\tLDL EUR pilot\trs111\t1\t100\t1:100:A:G\tA\tG\t-6\t0.199951"
    "\t1.97e-09\t.\tobserved\n"
)
# Where the tiny manifest's `source_file` paths expect the fixture, relative to
# the directory the build runs in.
FIXTURE_PLACE = Path("docs/examples/tiny")


class SmokeTestFailure(Exception):
    """The installed package does not behave as a user is promised."""


def check_import_location() -> Path:
    """The package directory, which must be in this environment's site-packages."""
    import opengwasdb

    package = Path(opengwasdb.__file__).resolve().parent
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    if not package.is_relative_to(site_packages):
        raise SmokeTestFailure(
            f"opengwasdb was imported from {package}, not from this environment's "
            f"site-packages {site_packages}: a checkout is on the import path"
        )
    return package


def check_version(required: str) -> None:
    import opengwasdb

    installed = distribution_version("opengwasdb")
    if (opengwasdb.__version__, installed) != (required, required):
        raise SmokeTestFailure(
            f"opengwasdb.__version__ is {opengwasdb.__version__} and the installed "
            f"metadata says {installed}; the wheel was built as {required}"
        )


def check_column_maps() -> list[str]:
    """Load every built-in Column Map from the installed package."""
    from opengwasdb.readers.column_map import BUILTIN_COLUMN_MAPS, load_column_map

    if not BUILTIN_COLUMN_MAPS:
        raise SmokeTestFailure("the installed package declares no built-in Column Maps")
    for name in BUILTIN_COLUMN_MAPS:
        try:
            load_column_map(name)
        except (OSError, ValueError) as error:  # ColumnMapError is a ValueError
            raise SmokeTestFailure(
                f"built-in Column Map {name!r} does not load from the installed package: {error}"
            ) from error
    return list(BUILTIN_COLUMN_MAPS)


def _run(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SmokeTestFailure(
            f"{' '.join(command)} exited {result.returncode}\n{result.stderr.strip()}"
        )
    return result.stdout


def installed_command() -> str:
    """The `opengwasdb` entry point this environment installed, beside its Python."""
    command = Path(sys.executable).parent / "opengwasdb"
    if not command.is_file():
        raise SmokeTestFailure(f"the environment has no opengwasdb command at {command}")
    return str(command)


def check_tiny_store(command: str, fixture: Path, scratch: Path) -> None:
    """Build, validate and query the tiny Dense example from a copy of it."""
    shutil.copytree(fixture, scratch / FIXTURE_PLACE)
    store = scratch / "tiny.opengwasdb"
    _run(
        [
            command,
            "build-dense",
            str(store),
            "--manifest",
            str(FIXTURE_PLACE / "manifest.tsv"),
            "--store-id",
            "tiny",
            "--release-id",
            "observed-v1",
            "--admission-record",
            str(scratch / "tiny.admission"),
        ],
        cwd=scratch,
    )
    validation = _run([command, "validate", str(store)], cwd=scratch).strip()
    if validation != "valid":
        raise SmokeTestFailure(f"validate said {validation!r}, not 'valid'")
    rows = _run([command, "query-phewas", str(store), "rs111", "--variant-info"], cwd=scratch)
    if rows != EXPECTED_PHEWAS:
        raise SmokeTestFailure(
            f"query-phewas rs111 returned\n{rows}\nnot the documented\n{EXPECTED_PHEWAS}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--fixture", type=Path, required=True, help="docs/examples/tiny")
    parser.add_argument("--version", required=True, help="the version the wheel was built as")
    args = parser.parse_args(argv)
    try:
        package = check_import_location()
        check_version(args.version)
        maps = check_column_maps()
        command = installed_command()
        _run([command, "--help"], cwd=Path.cwd())
        with tempfile.TemporaryDirectory() as scratch:
            check_tiny_store(command, args.fixture.resolve(), Path(scratch))
    except SmokeTestFailure as failure:
        print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(
        f"ok: opengwasdb {args.version} from {package}; column maps {', '.join(maps)}; "
        "tiny Store built, valid, and rs111 returned the documented rows"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
