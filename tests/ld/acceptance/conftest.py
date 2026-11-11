"""Every test under tests/acceptance/ is in the `acceptance` tier (CONTRIBUTING.md, "Acceptance
runs"): the default tier and the realdata tier deselect it, and `pixi run -e dev
acceptance-clump` selects it. Marking by directory means a test placed here cannot run in either
by forgetting its marker."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
# The run imports the scripts it exercises (`clump_rule`, `make_403_inputs`) by name.
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "realdata"))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if HERE in Path(item.path).resolve().parents:
            item.add_marker(pytest.mark.acceptance)
