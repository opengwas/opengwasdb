"""Suite-wide settings: the Hypothesis profiles.

`ci` is what CI loads (`HYPOTHESIS_PROFILE=ci`): about 50 examples a property,
derandomised so a red run reproduces, with a deadline so one slow example fails
rather than eating the 30 s budget of the default tier (CONTRIBUTING.md,
"Tests"). `dev` is the local default: the same deadline, more examples, and
random, so local runs keep exploring.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from hypothesis import settings
from tier_markers import refuse_misplaced

settings.register_profile(
    "ci", max_examples=50, deadline=timedelta(milliseconds=200), derandomize=True
)
settings.register_profile("dev", max_examples=100, deadline=timedelta(milliseconds=200))
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[Any]) -> None:
    """Before `-m` deselects anything, so a misplaced test is seen however the run selects: a mark
    on a test outside its tier's directory runs nowhere, so it is a collection error."""
    refuse_misplaced(items, Path(__file__).resolve().parent)
