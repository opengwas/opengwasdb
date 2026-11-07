"""Every test under tests/realdata/ is in the `realdata` tier.

The tier reads the HGDP+1kGP panels under /data/opengwasdb/reference/, so it runs on the IEU
compute nodes only, through `pixi run test-realdata` (CONTRIBUTING.md, "Tests"). Marking by
directory means a test placed here cannot run in the default tier by forgetting its marker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if HERE in Path(item.path).resolve().parents:
            item.add_marker(pytest.mark.realdata)
