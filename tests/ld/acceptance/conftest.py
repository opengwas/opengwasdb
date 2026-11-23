"""Every test under tests/acceptance/ is in the `acceptance` tier (CONTRIBUTING.md, "Acceptance
runs"): the default tier and the realdata tier deselect it. Each run has its own task:
`pixi run -e dev acceptance-clump` runs `test_clump_equals_plink2.py` (ogld#6),
`pixi run -e dev acceptance-stores` runs `test_hgdp1kgp_stores.py` (ogld#7), and
`pixi run -e dev acceptance-genotypes` runs
`test_genotype_store_equals_ld_store.py` (ogld#91). Marking by directory means a test placed
here cannot run in either tier by forgetting its marker."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
# The run imports the scripts it exercises (`clump_rule`, `make_403_inputs`) by name.
sys.path.insert(0, str(HERE.parents[2] / "scripts" / "ld" / "realdata"))
# The stores' acceptance reads the recipe's constants and paths (`build_hgdp1kgp`), not copies.
sys.path.insert(0, str(HERE.parents[2] / "scripts" / "ld"))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if HERE in Path(item.path).resolve().parents:
            item.add_marker(pytest.mark.acceptance)
