"""The realdata tier's inputs, each named by an environment variable that `run.sh` passes on.

A missing input is not a defect of ogld. A test that needs one asks `required` first, before it
calls ogld or runs plink2. Unset, or naming nothing on disk, the test stops there as an
expected failure whose reason names the variable and what it is for; `run.sh` prints those
reasons (`-rfEx`). So it never reaches `ogld.open("")`, where a refusal would read as a
defect, and it never counts as a pass. It is an expected failure, not a skip, because the
escapes gate counts every `pytest.skip` site in the tree (quality.json, `escapes`).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def required(name: str, what: str) -> Path:
    """The path `$name` holds; unset or naming nothing, the test stops here, naming both."""
    value = os.environ.get(name, "")
    if not value:
        pytest.xfail(f"input missing: ${name} is unset ({what})")
    if not Path(value).exists():
        pytest.xfail(f"input missing: ${name} names nothing at {value} ({what})")
    return Path(value)
