"""Build one fixture Store Release of every builder path, with a given checkout's code.

#243 and #244 compared the bytes builds write before and after a change: build
the same fixture stores with two checkouts (each under its own environment),
then compare the trees with `benchmarks/zarr3_compare_trees.py`. The stores are
the array-conformance suite's (`tests/test_array_conformance.py::_build_stores`,
from the checkout under test), so their inputs stay the ones the suite trusts.

    BLOSC_NTHREADS=1 python benchmarks/zarr3_fixture_trees.py --repo . --out /tmp/trees-head

Set `BLOSC_NTHREADS=1` for a byte-for-byte comparison: a chunk of two or more
Blosc blocks compressed with threads writes its blocks in completion order, so
its bytes are not reproducible run to run (ADR 0056).

#243 and #244 built their trees with #243's own `build_stores.py`, which built
the same paths except Hybrid Reference Completion; the suite's builder adds it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def use_checkout(repo: Path) -> None:
    """Put `repo` and its tests first on `sys.path`, before anything imports opengwasdb."""
    if "opengwasdb" in sys.modules:
        raise SystemExit("opengwasdb was imported before the checkout was chosen")
    sys.path.insert(0, str(repo.resolve() / "tests"))
    sys.path.insert(0, str(repo.resolve()))
    import opengwasdb

    if not Path(opengwasdb.__file__).resolve().is_relative_to(repo.resolve()):
        raise SystemExit(f"opengwasdb resolved to {opengwasdb.__file__}, not {repo}")


def build_all(out: Path) -> list[str]:
    """Every conformance-suite fixture store, under `out`; returns their labels."""
    import test_array_conformance

    out.mkdir(parents=True, exist_ok=True)
    stores = test_array_conformance._build_stores(out)
    if not list(out.rglob("data.zarr")):
        raise SystemExit(f"no data.zarr written under {out}")
    return [store.label for store in stores]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", type=Path, required=True, help="the checkout whose code builds")
    ap.add_argument("--out", type=Path, required=True, help="an empty or new directory")
    args = ap.parse_args()
    use_checkout(args.repo)
    print("built", build_all(args.out))


if __name__ == "__main__":
    main()
