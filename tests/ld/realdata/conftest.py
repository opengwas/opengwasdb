"""Every test under tests/ld/realdata/ is in the `realdata` tier.

The tier reads the HGDP+1kGP panels under /data/opengwasdb/reference/, so it runs on the IEU
compute nodes only, through `pixi run -e dev test-realdata` (CONTRIBUTING.md, "Tests"). Marking by
directory means a test placed here cannot run in the default tier by forgetting its marker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from realdata_inputs import installed_mismatch

HERE = Path(__file__).resolve().parent
# The tier's tests import the scripts they exercise (`clump_rule`, `make_403_inputs`) by name.
sys.path.insert(0, str(HERE.parents[2] / "scripts" / "ld" / "realdata"))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if HERE in Path(item.path).resolve().parents:
            item.add_marker(pytest.mark.realdata)


@pytest.fixture(scope="session", autouse=True)
def the_installed_ogld_names_this_checkout() -> None:
    """Every realdata test fails, naming both, when the installed opengwasdb's version does not name
    the checkout's HEAD: the stores it builds record software.ogld (ogld#5's review)."""
    mismatch = installed_mismatch()
    if mismatch:
        pytest.fail(mismatch, pytrace=False)
